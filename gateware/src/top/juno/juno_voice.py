# Copyright (c) 2024 S. Holzapfel <me@sebholzapfel.com>
#
# SPDX-License-Identifier: CERN-OHL-S-2.0

"""
Juno voice engine. Copied from ``tiliqua.dsp.voice_block`` (upstream, unmodified) so it can be
changed for the Juno port without touching the shared library. Top-level class renamed to
``JunoVoice``.

Original description:

``VoiceBlock`` - a polyphonic voice block - N oscillators, ADSRs, filters, etc and downmix.

This takes a different approach to the other streaming cores, whereby each core exchanges
``Block``s of samples, kind of a packet with one entry per voice delineated by a ``first``
flag. This keeps the interconnect logic required between blocks much smaller than would
otherwise be the case if every element was simply duplicated N times. This trades off
maximum throughput, but for polyphony like this, time multiplexing is necessary anyway.
For state variables, local BRAMs save and restore voice-specific state for each sample.

TODO: Come up with a principled approach of combining these cores with the existing
single-in, single-out ones? Perhaps we could instantiate the same type of operation as e.g.
a ``MultiSVF`` or just an ``SVF`` based on how the core is instantiated or connected.
Such automation quickly becomes something like an HLS problem, which I really don't want to
solve...

``VoiceBlock`` is constructed from the following components (all in this file):

    .. code-block:: text

        MultiJunoOsc ───────┐
                            ├── BlockMerge ──> MultiSVF (and DC block) ──> VoiceMixer
        MultiADSR ──────────┘
"""

from amaranth import *
from amaranth.lib import data, enum, stream, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

import math

from amaranth.utils import exact_log2
from amaranth_future import fixed

from tiliqua.dsp import ASQ, mac
from tiliqua.dsp.block import Block, BlockMerge


class PipelinedMAC(mac.MAC):

    """
    Juno port: a dedicated multiplier with registered operands and result.

    ``mac.MuxMAC`` (the library default) is purely combinational, so a block with
    many multiply steps ends up with a long path: FSM-driven operand mux ->
    multiplier -> accumulate -> register. This version registers the operands and
    the product, so ``valid`` comes 2 cycles after the operands are presented.
    ``MAC.Multiply`` already supports latency (operands are held until ``valid``).

    Back-to-back ``Multiply`` calls in consecutive FSM states keep ``strobe`` high;
    the stage counter wraps on ``valid`` so the next operation starts fresh.
    """

    LATENCY = 2

    def elaborate(self, platform):
        m = Module()
        a = Signal.like(self.operands.a)
        b = Signal.like(self.operands.b)
        stage = Signal(range(self.LATENCY + 1))
        m.d.sync += [
            a.eq(self.operands.a),
            b.eq(self.operands.b),
            self.result.z.eq(a * b),
        ]
        with m.If(self.strobe):
            with m.If(stage == self.LATENCY):
                m.d.sync += stage.eq(0)
            with m.Else():
                m.d.sync += stage.eq(stage + 1)
        with m.Else():
            m.d.sync += stage.eq(0)
        m.d.comb += self.valid.eq(self.strobe & (stage == self.LATENCY))
        return m


class PitchToInc(wiring.Component):

    """
    Log-domain pitch -> 32-bit phase increment (Juno port).

    ``pitch`` is in semitones on the MIDI note scale (69.0 = A4 = 440 Hz), with
    1/1024-semitone resolution. ``inc`` is the per-sample phase increment as a
    fraction of a cycle in UQ(0, 32), i.e. f / FS * 2**32.

    Two lookup tables:

    - note table (128 entries): increment of each integer note, normalized to a
      32-bit mantissa in [2**31, 2**32) plus a right-shift, so even the lowest notes
      keep full precision;
    - fine table (1024 entries): 2**(k / 1024 / 12) - 1 for the fractional semitone.

    ``inc = (mant + mant * fine) >> shift``, with one 17x17 multiply.
    Pitch is clamped to notes 0..127 (+ fraction).

    Fixed latency: hold ``pitch`` stable; ``inc`` is valid ``LATENCY`` cycles later.
    """

    PitchSQ = fixed.SQ(9, 10)
    FINE_BITS = 10
    LATENCY = 3
    FS = 48000
    N_NOTES = 128
    # fine table scale: value * 2**FINE_SCALE (max 0.0595 * 2**21 < 2**17)
    FINE_SCALE = 21
    MANT_SHIFT = 15   # mantissa bits dropped before the 17x17 multiply

    @classmethod
    def exact_inc(cls, pitch):
        """Ideal (float) increment for a pitch in semitones."""
        return 440.0 * 2 ** ((pitch - 69) / 12) / cls.FS * 2 ** 32

    @classmethod
    def note_table(cls):
        table = []
        for n in range(cls.N_NOTES):
            inc = cls.exact_inc(n)
            shift = 0
            while inc * 2 ** (shift + 1) < 2 ** 32:
                shift += 1
            mant = min(round(inc * 2 ** shift), 2 ** 32 - 1)
            table.append((mant, shift))
        return table

    @classmethod
    def fine_table(cls):
        n = 1 << cls.FINE_BITS
        return [round((2 ** (k / n / 12) - 1) * 2 ** cls.FINE_SCALE) for k in range(n)]

    @classmethod
    def model(cls, pitch_raw):
        """Bit-exact Python model. ``pitch_raw`` = pitch * 1024 (integer)."""
        semi, frac = pitch_raw >> cls.FINE_BITS, pitch_raw & ((1 << cls.FINE_BITS) - 1)
        if semi < 0:
            semi, frac = 0, 0
        elif semi >= cls.N_NOTES:
            semi, frac = cls.N_NOTES - 1, (1 << cls.FINE_BITS) - 1
        mant, shift = cls.note_table()[semi]
        prod = (mant >> cls.MANT_SHIFT) * cls.fine_table()[frac]
        full = mant + (prod >> (cls.FINE_SCALE - cls.MANT_SHIFT))
        return (full >> shift) & 0xFFFFFFFF

    def __init__(self):
        super().__init__({
            "pitch": In(self.PitchSQ),
            "inc":   Out(unsigned(32)),
        })

    def elaborate(self, platform):
        m = Module()

        notes = self.note_table()
        shift_bits = max(s for _, s in notes).bit_length()
        note_layout = data.StructLayout({"mant": unsigned(32), "shift": unsigned(shift_bits)})
        m.submodules.note_mem = note_mem = Memory(
            shape=note_layout, depth=self.N_NOTES,
            init=[{"mant": mt, "shift": sh} for mt, sh in notes])
        m.submodules.fine_mem = fine_mem = Memory(
            shape=unsigned(17), depth=1 << self.FINE_BITS, init=self.fine_table())
        note_rd = note_mem.read_port()
        fine_rd = fine_mem.read_port()
        m.d.comb += [note_rd.en.eq(1), fine_rd.en.eq(1)]

        # Cycle 0: clamp and address the tables
        raw = self.pitch.as_value()
        semi = raw >> self.FINE_BITS
        frac = raw[:self.FINE_BITS]
        with m.If(semi < 0):
            m.d.comb += [note_rd.addr.eq(0), fine_rd.addr.eq(0)]
        with m.Elif(semi >= self.N_NOTES):
            m.d.comb += [note_rd.addr.eq(self.N_NOTES - 1),
                         fine_rd.addr.eq((1 << self.FINE_BITS) - 1)]
        with m.Else():
            m.d.comb += [note_rd.addr.eq(semi), fine_rd.addr.eq(frac)]

        # Cycle 1: 17x17 multiply on the mantissa top bits
        prod = Signal(34)
        mant = Signal(32)
        shift = Signal(shift_bits)
        m.d.sync += [
            prod.eq((note_rd.data.mant >> self.MANT_SHIFT) * fine_rd.data),
            mant.eq(note_rd.data.mant),
            shift.eq(note_rd.data.shift),
        ]

        # Cycle 2: add and shift down by the octave
        full = Signal(33)
        m.d.comb += full.eq(mant + (prod >> (self.FINE_SCALE - self.MANT_SHIFT)))
        m.d.sync += self.inc.eq(full >> shift)

        return m


class MultiJunoOsc(wiring.Component):

    """
    N Juno DCOs (Juno port): curved saw, pulse, sub-oscillator and shared noise
    with the Juno mixer, band-limited with 2-point PolyBLEP.

    Bit-exact with ``top.juno.osc_model.OscModel`` (see there for the algorithm).
    Pitch per voice in semitones (``PitchToInc``); mixer settings are global:
    ``saw_en``, ``pulse_en``, ``pw_point`` (UQ(0,16) phase of the pulse rising edge),
    ``sub_gain`` and ``noise_gain`` (UQ(0,16)).

    One ``PipelinedMAC`` for 6 multiplies per voice, plus a 16-cycle restoring
    divider used twice per voice (~70 cycles per voice).

    Free-running: sample rate is determined by backpressure on ``self.o``.
    """

    DIV_BITS = 16
    LFSR_TAPS = 0x80200003
    LFSR_SEED = 0xACE1ACE1
    CURVE = 4915        # 0.15 * 2**15
    ONE = 1 << 15

    def __init__(self, n, sq=ASQ):
        self.n = n
        self.sq = sq
        super().__init__({
            "voice_pitch": In(data.ArrayLayout(PitchToInc.PitchSQ, n)),
            "saw_en":      In(1),
            "pulse_en":    In(1),
            "pw_point":    In(unsigned(16)),
            "sub_gain":    In(unsigned(16)),
            "noise_gain":  In(unsigned(16)),
            "o": Out(stream.Signature(Block(sq))),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        ONE = self.ONE
        W = 20  # internal signed width for 15-bit fixed point audio values

        state_layout = data.StructLayout({
            "phase":  unsigned(32),
            "sub":    unsigned(1),
            "sub_lp": signed(W),
        })
        m.submodules.state_mem = state_mem = Memory(shape=state_layout, depth=N, init=[])
        st_rport = state_mem.read_port()
        st_wport = state_mem.write_port()
        m.d.comb += st_rport.en.eq(1)

        m.submodules.pitch2inc = p2i = PitchToInc()
        m.submodules.macp = mp = PipelinedMAC()

        voice_ix   = Signal(range(N))
        cur_pitch  = Signal(PitchToInc.PitchSQ)
        pitch_wait = Signal(range(PitchToInc.LATENCY + 1))
        m.d.comb += p2i.pitch.eq(cur_pitch)

        phase   = Signal(32)
        sub     = Signal(1)
        sub_lp  = Signal(signed(W))
        inc     = Signal(32)
        t       = Signal(32)

        # shared noise
        lfsr     = Signal(32, init=self.LFSR_SEED)
        noise_lp = Signal(signed(W))

        # PolyBLEP bookkeeping
        w_point   = Signal(32)
        wrap_near = Signal()
        wrap_after = Signal()
        d_wrap    = Signal(32)
        pw_near   = Signal()
        pw_after  = Signal()
        d_pw      = Signal(32)
        c_w       = Signal(signed(W))
        c_p       = Signal(signed(W))

        # restoring divider: q = floor(num * 2**16 / den), num < den
        div_rem = Signal(34)
        div_q   = Signal(self.DIV_BITS)
        div_cnt = Signal(range(self.DIV_BITS + 1))

        # MAC operands (held until valid) and helpers
        mac_a = Signal(mac.SQNative)
        mac_b = Signal(mac.SQNative)
        mac_z = Signal(signed(40))
        m.d.comb += mac_z.eq(mp.result.z.as_value() >> 15)

        p_lin  = Signal(signed(W))
        q_curv = Signal(signed(W))
        ramp   = Signal(signed(W))
        saw    = Signal(signed(W))
        pulse  = Signal(signed(W))
        acc    = Signal(signed(W + 2))
        out    = Signal(signed(16))

        def corr(near, after, c, after_sign):
            """+-c (after_sign for 'after the edge', opposite before), 0 if not near."""
            v = Signal(signed(W))
            with m.If(near):
                with m.If(after):
                    m.d.comb += v.eq(c if after_sign > 0 else -c)
                with m.Else():
                    m.d.comb += v.eq(-c if after_sign > 0 else c)
            return v

        corr_wrap = corr(wrap_near, wrap_after, c_w, +1)   # saw reset / pulse fall
        corr_pw   = corr(pw_near, pw_after, c_p, -1)       # pulse rise

        with m.FSM():

            with m.State('LOAD'):
                m.d.comb += st_rport.addr.eq(voice_ix)
                m.next = 'LATCH'

            with m.State('LATCH'):
                m.d.sync += [
                    phase.eq(st_rport.data.phase),
                    sub.eq(st_rport.data.sub),
                    sub_lp.eq(st_rport.data.sub_lp),
                    pitch_wait.eq(0),
                    w_point.eq(self.pw_point << 16),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += cur_pitch.eq(self.voice_pitch[n])
                with m.If(voice_ix == 0):
                    # shared noise: advance once per frame, one-pole tilt
                    nxt = Signal(32)
                    with m.If(lfsr[0]):
                        m.d.comb += nxt.eq((lfsr >> 1) ^ self.LFSR_TAPS)
                    with m.Else():
                        m.d.comb += nxt.eq(lfsr >> 1)
                    n16 = nxt[16:32].as_signed()
                    m.d.sync += [
                        lfsr.eq(nxt),
                        noise_lp.eq(noise_lp + ((n16 - noise_lp) >> 1)),
                    ]
                m.next = 'PITCH-WAIT'

            with m.State('PITCH-WAIT'):
                m.d.sync += pitch_wait.eq(pitch_wait + 1)
                with m.If(pitch_wait == PitchToInc.LATENCY - 1):
                    m.next = 'ADVANCE'

            with m.State('ADVANCE'):
                t_full = Signal(33)
                t_new = Signal(32)
                m.d.comb += [
                    t_full.eq(phase + p2i.inc),
                    t_new.eq(t_full[:32]),
                ]
                inv_t = Signal(33)
                m.d.comb += inv_t.eq((1 << 32) - t_new)
                m.d.sync += [
                    inc.eq(p2i.inc),
                    t.eq(t_new),
                    phase.eq(t_new),
                    sub.eq(sub ^ t_full[32]),
                    # wrap edge
                    wrap_after.eq(t_new < p2i.inc),
                    wrap_near.eq((t_new < p2i.inc) | (inv_t < p2i.inc)),
                    d_wrap.eq(Mux(t_new < p2i.inc, t_new, inv_t)),
                    # pulse rising edge
                    pw_after.eq(t_new >= w_point),
                    pw_near.eq(Mux(t_new >= w_point,
                                   (t_new - w_point) < p2i.inc,
                                   (w_point - t_new) < p2i.inc)),
                    d_pw.eq(Mux(t_new >= w_point, t_new - w_point, w_point - t_new)),
                ]
                m.next = 'DIV-W-INIT'

            # --- PolyBLEP for the wrap edge ---------------------------------
            with m.State('DIV-W-INIT'):
                m.d.sync += [div_rem.eq(d_wrap), div_q.eq(0), div_cnt.eq(0)]
                m.next = 'DIV-W'

            with m.State('DIV-W'):
                self._div_step(m, div_rem, div_q, div_cnt, inc, 'SQ-W')

            with m.State('SQ-W'):
                om = ((1 << self.DIV_BITS) - div_q) >> 1
                m.d.comb += [mac_a.as_value().eq(om), mac_b.as_value().eq(om)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += c_w.eq(Mux(wrap_near, mac_z, 0))
                    m.next = 'DIV-P-INIT'

            # --- PolyBLEP for the pulse rising edge --------------------------
            with m.State('DIV-P-INIT'):
                m.d.sync += [div_rem.eq(d_pw), div_q.eq(0), div_cnt.eq(0)]
                m.next = 'DIV-P'

            with m.State('DIV-P'):
                self._div_step(m, div_rem, div_q, div_cnt, inc, 'SQ-P')

            with m.State('SQ-P'):
                om = ((1 << self.DIV_BITS) - div_q) >> 1
                m.d.comb += [mac_a.as_value().eq(om), mac_b.as_value().eq(om)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += [
                        c_p.eq(Mux(pw_near, mac_z, 0)),
                        p_lin.eq(t >> 17),
                    ]
                    m.next = 'CURVE-1'

            # --- curved saw: ramp = p + 0.15 p (1 - p) -----------------------
            with m.State('CURVE-1'):
                m.d.comb += [mac_a.as_value().eq(p_lin), mac_b.as_value().eq(ONE - p_lin)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += q_curv.eq(mac_z)
                    m.next = 'CURVE-2'

            with m.State('CURVE-2'):
                m.d.comb += [mac_a.as_value().eq(q_curv), mac_b.as_value().eq(self.CURVE)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += ramp.eq(p_lin + mac_z)
                    m.next = 'SHAPE'

            with m.State('SHAPE'):
                sub_x = Signal(signed(W))
                m.d.comb += sub_x.eq(Mux(sub, ONE - c_w, -ONE + c_w))
                m.d.sync += [
                    saw.eq(2 * ramp - ONE + corr_wrap),
                    pulse.eq(Mux(t >= w_point, ONE, -ONE) + corr_wrap + corr_pw),
                    sub_lp.eq(sub_lp + ((sub_x - sub_lp) >> 1)),
                ]
                m.next = 'MIX-SUB'

            # --- mixer --------------------------------------------------------
            with m.State('MIX-SUB'):
                m.d.comb += [mac_a.as_value().eq(sub_lp),
                             mac_b.as_value().eq(self.sub_gain >> 1)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += acc.eq(
                        Mux(self.saw_en, saw >> 1, 0) +
                        Mux(self.pulse_en, pulse >> 1, 0) +
                        mac_z)
                    m.next = 'MIX-NOISE'

            with m.State('MIX-NOISE'):
                m.d.comb += [mac_a.as_value().eq(noise_lp),
                             mac_b.as_value().eq(self.noise_gain >> 1)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    total = acc + mac_z
                    with m.If(total > ONE - 1):
                        m.d.sync += out.eq(ONE - 1)
                    with m.Elif(total < -ONE):
                        m.d.sync += out.eq(-ONE)
                    with m.Else():
                        m.d.sync += out.eq(total)
                    m.next = 'STORE'

            with m.State('STORE'):
                m.d.comb += [
                    st_wport.addr.eq(voice_ix),
                    st_wport.data.phase.eq(phase),
                    st_wport.data.sub.eq(sub),
                    st_wport.data.sub_lp.eq(sub_lp),
                    st_wport.en.eq(1),
                ]
                m.next = 'EMIT'

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.sample.as_value().eq(out),
                    self.o.payload.first.eq(voice_ix == 0),
                ]
                with m.If(self.o.ready):
                    with m.If(voice_ix == N - 1):
                        m.d.sync += voice_ix.eq(0)
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)
                    m.next = 'LOAD'

        return m

    def _div_step(self, m, rem, q, cnt, den, next_state):
        """One iteration of the restoring divider per cycle."""
        shifted = Signal(34, name=f"div_shift_{next_state.lower().replace('-', '_')}")
        m.d.comb += shifted.eq(rem << 1)
        with m.If(shifted >= den):
            m.d.sync += [rem.eq(shifted - den), q.eq((q << 1) | 1)]
        with m.Else():
            m.d.sync += [rem.eq(shifted), q.eq(q << 1)]
        m.d.sync += cnt.eq(cnt + 1)
        with m.If(cnt == self.DIV_BITS - 1):
            m.next = next_state


class MultiADSR(wiring.Component):
    """
    N ADSR envelope generators with:
    - Independent gate/velocity per voice (output is velocity * adsr product)
    - Global attack/decay/sustain/release control
    - Outgoing bitfield of which voices are still traversing through ADSR states
      (e.g. gate released, but voice is still releasing - used for culling)

    Free-running: sample rate is determined by backpressure on ``self.o``.
    """

    class Phase(enum.Enum, shape=unsigned(3)):
        IDLE    = 0
        ATTACK  = 1
        DECAY   = 2
        SUSTAIN = 3
        RELEASE = 4

    EnvUQ = fixed.UQ(0, 16)

    def __init__(self, n):
        self.n = n
        super().__init__({
            # Gate and velocity inputs (independent per-voice)
            "voice_gates":    In(data.ArrayLayout(unsigned(1), n)),
            "voice_velocity": In(data.ArrayLayout(self.EnvUQ, n)),
            # Global ADSR rates
            "attack_rate":    In(self.EnvUQ),
            "decay_rate":     In(self.EnvUQ),
            "sustain_level":  In(self.EnvUQ),
            "release_rate":   In(self.EnvUQ),
            # Samples out and bitfield of active voices
            "o": Out(stream.Signature(Block(self.EnvUQ))),
            "voice_active": Out(data.ArrayLayout(unsigned(1), n)),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        EnvUQ = self.EnvUQ

        # Per-voice ADSR state memory

        adsr_state_layout = data.StructLayout({
            "l_gate":     unsigned(1),
            "adsr_phase": self.Phase,
            "adsr_level": EnvUQ,
        })
        m.submodules.state_mem = state_mem = Memory(
            shape=adsr_state_layout, depth=N, init=[])
        st_rport = state_mem.read_port()
        st_wport = state_mem.write_port()
        m.d.comb += st_rport.en.eq(1)

        # Registers fetched / stored from above state mem

        l_gate      = Signal(1)
        adsr_phase  = Signal(self.Phase)
        adsr_level  = Signal(EnvUQ)
        cur_gate    = Signal(1)
        cur_vel_mod = Signal(EnvUQ)
        voice_ix    = Signal(range(N))

        with m.FSM():

            with m.State('LOAD'):
                m.d.comb += st_rport.addr.eq(voice_ix)
                m.next = 'LATCH-STATE'

            with m.State('LATCH-STATE'):
                m.d.sync += [
                    l_gate.eq(st_rport.data.l_gate),
                    adsr_phase.eq(st_rport.data.adsr_phase),
                    adsr_level.eq(st_rport.data.adsr_level),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            # Latch gate, vel for current voice
                            m.d.sync += [
                                cur_gate.eq(self.voice_gates[n]),
                                cur_vel_mod.eq(self.voice_velocity[n]),
                            ]
                m.next = 'ADSR-UPDATE'

            with m.State('ADSR-UPDATE'):

                gate_rising  = Signal()
                gate_falling = Signal()
                m.d.comb += [
                    gate_rising.eq(cur_gate & ~l_gate),
                    gate_falling.eq(~cur_gate & l_gate),
                ]

                active_phase = Signal(self.Phase)
                m.d.comb += active_phase.eq(adsr_phase)
                with m.If(gate_rising):
                    m.d.comb += active_phase.eq(self.Phase.ATTACK)
                with m.Elif(gate_falling):
                    m.d.comb += active_phase.eq(self.Phase.RELEASE)

                next_level = Signal(EnvUQ)
                out_phase = Signal(self.Phase)
                m.d.comb += [
                    next_level.eq(adsr_level),
                    out_phase.eq(active_phase),
                ]

                with m.Switch(active_phase):
                    with m.Case(self.Phase.IDLE):
                        m.d.comb += next_level.eq(0)
                    with m.Case(self.Phase.ATTACK):
                        attack_sum = adsr_level + self.attack_rate
                        with m.If(attack_sum >= EnvUQ.max()):
                            m.d.comb += [
                                next_level.eq(EnvUQ.max()),
                                out_phase.eq(self.Phase.DECAY),
                            ]
                        with m.Else():
                            m.d.comb += next_level.eq(attack_sum)
                    with m.Case(self.Phase.DECAY):
                        with m.If(adsr_level <= self.sustain_level + self.decay_rate):
                            m.d.comb += [
                                next_level.eq(self.sustain_level),
                                out_phase.eq(self.Phase.SUSTAIN),
                            ]
                        with m.Else():
                            m.d.comb += next_level.eq(adsr_level - self.decay_rate)
                    with m.Case(self.Phase.SUSTAIN):
                        m.d.comb += next_level.eq(self.sustain_level)
                    with m.Case(self.Phase.RELEASE):
                        with m.If(adsr_level <= self.release_rate):
                            m.d.comb += [
                                next_level.eq(0),
                                out_phase.eq(self.Phase.IDLE),
                            ]
                        with m.Else():
                            m.d.comb += next_level.eq(adsr_level - self.release_rate)

                m.d.sync += [
                    adsr_level.eq(next_level),
                    adsr_phase.eq(out_phase),
                    l_gate.eq(cur_gate),
                ]
                m.next = 'STORE'

            with m.State('STORE'):
                m.d.comb += [
                    st_wport.addr.eq(voice_ix),
                    st_wport.data.l_gate.eq(l_gate),
                    st_wport.data.adsr_phase.eq(adsr_phase),
                    st_wport.data.adsr_level.eq(adsr_level),
                    st_wport.en.eq(1),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += self.voice_active[n].eq(
                                adsr_phase != self.Phase.IDLE)
                m.next = 'EMIT'

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.first.eq(voice_ix == 0),
                    self.o.payload.sample.eq(adsr_level * cur_vel_mod),
                ]
                with m.If(self.o.ready):
                    with m.If(voice_ix == N - 1):
                        m.d.sync += voice_ix.eq(0)
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)
                    m.next = 'LOAD'

        return m


class MultiSVF(wiring.Component):

    """
    N oversampled Chamberlin state variable filters with:
    - Per-voice samples in and envelope
    - Global resonance (all filters)
    - LPF output has a DC-blocking stage so 0 cutoff doesn't emit DC
    - Per-voice VCA after the DC block: output = filtered * env

    Juno port additions (all on the same multiplier, no extra DSP tile):

    - Exponential cutoff. Per voice, a cutoff control ``c`` in 'normalized octaves'
      (0..1 spans ``CUTOFF_OCTAVES`` octaves above ``CUTOFF_F_MIN``) is computed as::

          c = voice_cutoff_base[n] + env_amt * env + 0.8192 * cutoff_cv

      ``voice_cutoff_base`` comes from firmware (knob + keyboard tracking + mod wheel),
      ``env_amt`` is bipolar, and the CV scaling makes 1 V = 1 octave (ASQ 1.0 = 8.192 V).
      ``c`` is clamped to [0, 1) and converted to the SVF coefficient by a lookup table.
    - VCA after the DC block: output = filtered * env.

    Otherwise this is N copies of the one-in-one-out ``SVF`` and ``DCBlock`` from ``dsp.filters``.
    """

    N_OVERSAMPLE = 2

    # Exponential cutoff mapping: c in [0, 1) -> CUTOFF_F_MIN * 2**(c * CUTOFF_OCTAVES) Hz.
    CUTOFF_F_MIN   = 12.0
    CUTOFF_OCTAVES = 10
    CUTOFF_LUT_BITS = 10
    FS = 48000

    @classmethod
    def cutoff_hz(cls, c):
        return cls.CUTOFF_F_MIN * 2 ** (c * cls.CUTOFF_OCTAVES)

    @classmethod
    def cutoff_lut(cls):
        """SVF coefficient f = 2 sin(pi fc / fs_oversampled), raw SQNative values."""
        size = 1 << cls.CUTOFF_LUT_BITS
        fs_os = cls.FS * cls.N_OVERSAMPLE
        lut = []
        for i in range(size):
            fc = cls.cutoff_hz(i / size)
            lut.append(fixed.Const(2 * math.sin(math.pi * fc / fs_os),
                                   shape=mac.SQNative)._value)
        return lut

    def __init__(self, n, sq=ASQ, env_uq=MultiADSR.EnvUQ, macp=None):
        self.n = n
        self.sq = sq
        self.macp = macp or mac.MAC.default()
        super().__init__({
            "i":    In(stream.Signature(Block(data.StructLayout({"x": sq, "env": env_uq})))),
            "reso": In(unsigned(16)),
            "voice_cutoff_base": In(data.ArrayLayout(ASQ, n)),
            "env_amt":    In(ASQ),
            "cutoff_cv":  In(ASQ),
            "o":    Out(stream.Signature(Block(sq))),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        m.submodules.macp = mp = self.macp

        # Per-voice SVF state memory

        svf_sh = fixed.SQ(mac.SQNative.i_bits, mac.SQNative.f_bits + 2)
        state_layout = data.StructLayout({
            "abp":  svf_sh,
            "alp":  svf_sh,
            "ahp":  svf_sh,
            "dc_x": svf_sh,
            "dc_y": svf_sh,
        })
        m.submodules.state_mem = state_mem = Memory(
            shape=state_layout, depth=N, init=[])
        st_rport = state_mem.read_port()
        st_wport = state_mem.write_port()
        m.d.comb += st_rport.en.eq(1)

        # Working registers
        svf_abp = Signal(svf_sh)
        svf_alp = Signal(svf_sh)
        svf_ahp = Signal(svf_sh)

        svf_x     = Signal(mac.SQNative)
        svf_kK    = Signal(mac.SQNative)
        svf_kQinv = Signal(mac.SQNative)
        oversample = Signal(range(self.N_OVERSAMPLE))

        # DC block: multiply-free first-order highpass
        # y[n] = x[n] - x[n-1] + y[n-1] - (y[n-1] >> DC_SHIFT)
        DC_SHIFT = 9  # 1/512 ≈ 0.998, ~1.5Hz corner at 48kHz
        dc_x = Signal(svf_sh)
        dc_y = Signal(svf_sh)

        # VCA: the per-voice envelope scales the output.
        vca_env = Signal(mac.SQNative)
        vca_out = Signal(svf_sh)

        # Exponential cutoff: c = base + env_amt * env + cv (normalized octaves)
        cut_base = Signal(mac.SQNative)
        cut_amt  = Signal(mac.SQNative)
        cut_cv   = Signal(mac.SQNative)
        cut_c    = Signal(mac.SQNative)
        m.d.comb += [
            cut_amt.eq(self.env_amt),
            # 0.8192 ~= 1 - 1/8 - 1/16 + 1/128 (0.13 % error): 1 V/oct on the CV input
            cut_cv.eq(self.cutoff_cv - (self.cutoff_cv >> 3) - (self.cutoff_cv >> 4)
                      + (self.cutoff_cv >> 7)),
        ]
        lut = self.cutoff_lut()
        m.submodules.cutoff_lut = cutoff_mem = Memory(
            shape=signed(mac.SQNative.as_shape().width), depth=len(lut), init=lut)
        lut_rport = cutoff_mem.read_port()
        m.d.comb += lut_rport.en.eq(1)
        idx_shift = mac.SQNative.f_bits - self.CUTOFF_LUT_BITS
        idx_max = len(lut) - 1

        in_first = Signal(1)
        voice_ix = Signal(range(N))

        with m.FSM():

            with m.State('WAIT-VALID'):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += [
                        svf_x.eq(self.i.payload.sample.x >> 1),
                        vca_env.eq(self.i.payload.sample.env),
                        svf_kQinv.as_value().eq(self.reso),
                        in_first.eq(self.i.payload.first),
                        oversample.eq(0),
                    ]
                    m.d.comb += st_rport.addr.eq(voice_ix)
                    m.next = 'LOAD'

            with m.State('LOAD'):
                m.d.sync += [
                    svf_abp.eq(st_rport.data.abp),
                    svf_alp.eq(st_rport.data.alp),
                    svf_ahp.eq(st_rport.data.ahp),
                    dc_x.eq(st_rport.data.dc_x),
                    dc_y.eq(st_rport.data.dc_y),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += cut_base.eq(self.voice_cutoff_base[n])
                m.next = 'CUTOFF-MUL'

            with m.State('CUTOFF-MUL'):
                # c = base + env_amt * env + cv
                with mp.Multiply(m, a=vca_env, b=cut_amt):
                    m.d.sync += cut_c.eq(cut_base + mp.result.z + cut_cv)
                    m.next = 'CUTOFF-ADDR'

            with m.State('CUTOFF-ADDR'):
                # clamp c to [0, 1) and look up the SVF coefficient
                c_raw = cut_c.as_value()
                with m.If(c_raw < 0):
                    m.d.comb += lut_rport.addr.eq(0)
                with m.Elif((c_raw >> idx_shift) > idx_max):
                    m.d.comb += lut_rport.addr.eq(idx_max)
                with m.Else():
                    m.d.comb += lut_rport.addr.eq(c_raw >> idx_shift)
                m.next = 'CUTOFF-READ'

            with m.State('CUTOFF-READ'):
                m.d.sync += svf_kK.as_value().eq(lut_rport.data)
                m.next = 'SVF-MAC0'

            with m.State('SVF-MAC0'):
                # alp = abp*kK + alp
                with mp.Multiply(m, a=svf_abp, b=svf_kK):
                    m.d.sync += svf_alp.eq(mp.result.z + svf_alp)
                    m.next = 'SVF-MAC1'

            with m.State('SVF-MAC1'):
                # ahp = abp*(-kQinv) + (x - alp)
                with mp.Multiply(m, a=svf_abp, b=-svf_kQinv):
                    m.d.sync += svf_ahp.eq(mp.result.z + (svf_x - svf_alp))
                    m.next = 'SVF-MAC2'

            with m.State('SVF-MAC2'):
                # abp = ahp*kK + abp
                with mp.Multiply(m, a=svf_ahp, b=svf_kK):
                    m.d.sync += svf_abp.eq(mp.result.z + svf_abp)
                    m.next = 'SVF-OVER'

            with m.State('SVF-OVER'):
                with m.If(oversample == self.N_OVERSAMPLE - 1):
                    m.next = 'DC-BLOCK'
                with m.Else():
                    m.d.sync += oversample.eq(oversample + 1)
                    m.next = 'SVF-MAC0'

            with m.State('DC-BLOCK'):
                # Multiply-free DC blocking highpass on SVF lowpass output.
                svf_out = svf_alp >> 1
                m.d.sync += [
                    dc_y.eq(svf_out - dc_x + dc_y - (dc_y >> DC_SHIFT)),
                    dc_x.eq(svf_out),
                ]
                m.next = 'VCA'

            with m.State('VCA'):
                with mp.Multiply(m, a=dc_y, b=vca_env):
                    m.d.sync += vca_out.eq(mp.result.z)
                    m.next = 'STORE'

            with m.State('STORE'):
                m.d.comb += [
                    st_wport.addr.eq(voice_ix),
                    st_wport.data.abp.eq(svf_abp),
                    st_wport.data.alp.eq(svf_alp),
                    st_wport.data.ahp.eq(svf_ahp),
                    st_wport.data.dc_x.eq(dc_x),
                    st_wport.data.dc_y.eq(dc_y),
                    st_wport.en.eq(1),
                ]
                m.next = 'EMIT'

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.first.eq(in_first),
                    self.o.payload.sample.eq(vca_out),
                ]
                with m.If(self.o.ready):
                    with m.If(voice_ix == N - 1):
                        m.d.sync += voice_ix.eq(0)
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)
                    m.next = 'WAIT-VALID'

        return m


class VoiceMixer(wiring.Component):

    """
    Downmix N voices into a stereo output stream.
    - Even-indexed voices sum to left, odd-indexed to right
    - Gain-scaled by 2/N to prevent clipping
    """

    def __init__(self, n, sq=ASQ):
        self.n = n
        self.sq = sq
        super().__init__({
            "i": In(stream.Signature(Block(sq))),
            "o": Out(stream.Signature(data.ArrayLayout(sq, 2))),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n

        sq = self.sq
        mix_gain = fixed.Const(0.75 * 2.0 / N, shape=sq)

        acc_l = Signal(sq)
        acc_r = Signal(sq)
        voice_ix = Signal(range(N))

        with m.FSM():

            with m.State('CLEAR'):
                m.d.sync += [
                    acc_l.eq(0),
                    acc_r.eq(0),
                    voice_ix.eq(0),
                ]
                m.next = 'WAIT-VALID'

            with m.State('WAIT-VALID'):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    scaled = self.i.payload.sample * mix_gain

                    # Even voices -> left, odd voices -> right
                    with m.If(voice_ix[0] == 0):
                        m.d.sync += acc_l.eq(acc_l + scaled)
                    with m.Else():
                        m.d.sync += acc_r.eq(acc_r + scaled)

                    with m.If(voice_ix == N - 1):
                        m.next = 'EMIT-OUTPUT'
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)

            with m.State('EMIT-OUTPUT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload[0].eq(acc_l),
                    self.o.payload[1].eq(acc_r),
                ]
                with m.If(self.o.ready):
                    m.next = 'CLEAR'

        return m


class JunoVoice(wiring.Component):

    def __init__(self, n_voices, sq=ASQ):
        self.n_voices = n_voices
        self.sq = sq
        super().__init__({
            # Stereo audio output stream
            "o": Out(stream.Signature(data.ArrayLayout(sq, 2))),

            # Voice control (from MidiVoiceTracker)
            "voice_gates":     In(data.ArrayLayout(unsigned(1), self.n_voices)),
            # Per-voice pitch in semitones (MIDI note scale, see PitchToInc)
            "voice_pitch":     In(data.ArrayLayout(PitchToInc.PitchSQ, self.n_voices)),
            "voice_velocity":  In(data.ArrayLayout(MultiADSR.EnvUQ, self.n_voices)),

            # Global synth parameters (from CSR)
            "reso":           In(unsigned(16)),
            # Exponential cutoff (see MultiSVF): per-voice base from firmware,
            # bipolar env amount, cutoff CV (1 V/oct)
            "voice_cutoff_base": In(data.ArrayLayout(ASQ, self.n_voices)),
            "env_amt":        In(ASQ),
            "cutoff_cv":      In(ASQ),
            "attack_rate":    In(MultiADSR.EnvUQ),
            "decay_rate":     In(MultiADSR.EnvUQ),
            "sustain_level":  In(MultiADSR.EnvUQ),
            "release_rate":   In(MultiADSR.EnvUQ),

            # Juno DCO mixer (global, see MultiJunoOsc)
            "saw_en":         In(1),
            "pulse_en":       In(1),
            "pw_point":       In(unsigned(16)),
            "sub_gain":       In(unsigned(16)),
            "noise_gain":     In(unsigned(16)),

            # Per-voice ADSR idle status
            "voice_active":   Out(data.ArrayLayout(unsigned(1), self.n_voices)),

            # Per-voice envelope level (top 8 bits of ADSR output)
            "voice_cutoffs":     Out(data.ArrayLayout(unsigned(8), self.n_voices)),
        })

    def elaborate(self, platform):
        m = Module()

        N = self.n_voices

        sq = self.sq
        m.submodules.osc = osc = MultiJunoOsc(n=N, sq=sq)
        m.submodules.adsr = adsr = MultiADSR(n=N)
        m.submodules.merge = merge = BlockMerge({"x": sq, "env": MultiADSR.EnvUQ})
        # Pipelined multiplier: keeps the 60 MHz domain clear of the SVF MAC path
        m.submodules.svf = svf = MultiSVF(n=N, sq=sq, macp=PipelinedMAC())
        m.submodules.mixer = mixer = VoiceMixer(n=N, sq=sq)

        # Wire external ports -> internal components
        for n in range(N):
            m.d.comb += [
                osc.voice_pitch[n].eq(self.voice_pitch[n]),
                adsr.voice_gates[n].eq(self.voice_gates[n]),
                adsr.voice_velocity[n].eq(self.voice_velocity[n]),
                self.voice_active[n].eq(adsr.voice_active[n]),
                svf.voice_cutoff_base[n].eq(self.voice_cutoff_base[n]),
            ]
        m.d.comb += [
            osc.saw_en.eq(self.saw_en),
            osc.pulse_en.eq(self.pulse_en),
            osc.pw_point.eq(self.pw_point),
            osc.sub_gain.eq(self.sub_gain),
            osc.noise_gain.eq(self.noise_gain),
            adsr.attack_rate.eq(self.attack_rate),
            adsr.decay_rate.eq(self.decay_rate),
            adsr.sustain_level.eq(self.sustain_level),
            adsr.release_rate.eq(self.release_rate),
            svf.reso.eq(self.reso),
            svf.env_amt.eq(self.env_amt),
            svf.cutoff_cv.eq(self.cutoff_cv),
        ]

        # Stream pipeline
        wiring.connect(m, osc.o, merge.x)
        wiring.connect(m, adsr.o, merge.env)
        wiring.connect(m, merge.o, svf.i)
        wiring.connect(m, svf.o, mixer.i)
        wiring.connect(m, mixer.o, wiring.flipped(self.o))

        # Tap ADSR output stream to capture per-voice envelope values.
        # Voices always arrive in order 0..N-1, so a simple counter tracks them.
        env_voice_ix = Signal(range(N))
        with m.If(adsr.o.valid & adsr.o.ready):
            with m.Switch(env_voice_ix):
                for n in range(N):
                    with m.Case(n):
                        m.d.sync += self.voice_cutoffs[n].eq(
                            adsr.o.payload.sample.as_value() >> 8)
            with m.If(env_voice_ix == N - 1):
                m.d.sync += env_voice_ix.eq(0)
            with m.Else():
                m.d.sync += env_voice_ix.eq(env_voice_ix + 1)

        return m
