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
                            ├── BlockMerge ──> MultiLadder (+ DC block, VCA) ──> VoiceMixer
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

# top.py runs as a script (sibling imports); tests / sim_render import the package.
try:
    import ladder_model as LM
    import adsr_model as AM
except ImportError:
    from top.juno import ladder_model as LM
    from top.juno import adsr_model as AM


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

    # Output is 18-bit (SQNative, +-4.0): the full Juno mix (saw + pulse + sub +
    # noise, up to +-2.67) reaches the ladder unclipped, like the analog mixer.
    OUT_SAT = (1 << 17) - 1

    def __init__(self, n, sq=mac.SQNative):
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
        out    = Signal(signed(18))

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
                    with m.If(total > self.OUT_SAT):
                        m.d.sync += out.eq(self.OUT_SAT)
                    with m.Elif(total < -self.OUT_SAT):
                        m.d.sync += out.eq(-self.OUT_SAT)
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
    N Juno-106-style envelope generators (Juno port, Phase 4.2), bit-exact with
    ``top.juno.adsr_model.AdsrModel`` (see there for the algorithm):

    - 32-bit level per voice; linear attack (32-bit increment);
    - exponential decay toward 0 that stops at the sustain level, exponential
      release, both ``L -= ((L >> 15) * eps) >> 4`` with ``eps`` (17 bit, 19
      fractional bits) from firmware; voice goes idle below ~-96 dB;
    - output ``velocity * level`` (UQ(0,16)), outgoing bitfield of active voices.

    One ``PipelinedMAC`` (decay/release step and velocity scaling).
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
            # Global envelope settings (see adsr_model)
            "attack_rate":    In(unsigned(32)),   # level increment per sample
            "decay_rate":     In(unsigned(17)),   # eps, 19 fractional bits
            "sustain_level":  In(self.EnvUQ),
            "release_rate":   In(unsigned(17)),   # eps, 19 fractional bits
            # Samples out and bitfield of active voices
            "o": Out(stream.Signature(Block(self.EnvUQ))),
            "voice_active": Out(data.ArrayLayout(unsigned(1), n)),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        L_MAX = AM.L_MAX
        m.submodules.macp = mp = PipelinedMAC()

        state_layout = data.StructLayout({
            "l_gate": unsigned(1),
            "phase":  self.Phase,
            "level":  unsigned(32),
        })
        m.submodules.state_mem = state_mem = Memory(shape=state_layout, depth=N, init=[])
        st_rport = state_mem.read_port()
        st_wport = state_mem.write_port()
        m.d.comb += st_rport.en.eq(1)

        voice_ix = Signal(range(N))
        l_gate   = Signal()
        phase    = Signal(self.Phase)
        level    = Signal(32)
        cur_gate = Signal()
        cur_vel  = Signal(16)
        out      = Signal(16)

        mac_a = Signal(mac.SQNative)
        mac_b = Signal(mac.SQNative)
        prod  = Signal(36)
        m.d.comb += prod.eq(mp.result.z.as_value())   # raw integer product

        sus32 = Signal(32)
        m.d.comb += sus32.eq(self.sustain_level.as_value() << 16)

        with m.FSM():

            with m.State('LOAD'):
                m.d.comb += st_rport.addr.eq(voice_ix)
                m.next = 'LATCH'

            with m.State('LATCH'):
                m.d.sync += [
                    l_gate.eq(st_rport.data.l_gate),
                    phase.eq(st_rport.data.phase),
                    level.eq(st_rport.data.level),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += [
                                cur_gate.eq(self.voice_gates[n]),
                                cur_vel.eq(self.voice_velocity[n].as_value()),
                            ]
                m.next = 'GATE'

            with m.State('GATE'):
                with m.If(cur_gate & ~l_gate):
                    m.d.sync += phase.eq(self.Phase.ATTACK)
                with m.Elif(~cur_gate & l_gate):
                    m.d.sync += phase.eq(self.Phase.RELEASE)
                m.d.sync += l_gate.eq(cur_gate)
                m.next = 'UPDATE'

            with m.State('UPDATE'):
                with m.Switch(phase):
                    with m.Case(self.Phase.IDLE):
                        m.d.sync += level.eq(0)
                        m.next = 'VEL'
                    with m.Case(self.Phase.ATTACK):
                        nxt = Signal(33)
                        m.d.comb += nxt.eq(level + self.attack_rate)
                        with m.If(nxt >= L_MAX):
                            m.d.sync += [level.eq(L_MAX), phase.eq(self.Phase.DECAY)]
                        with m.Else():
                            m.d.sync += level.eq(nxt)
                        m.next = 'VEL'
                    with m.Case(self.Phase.SUSTAIN):
                        m.d.sync += level.eq(sus32)
                        m.next = 'VEL'
                    with m.Case(self.Phase.DECAY, self.Phase.RELEASE):
                        m.next = 'EXP'

            with m.State('EXP'):
                # L -= ((L >> 15) * eps) >> (EPS_FRAC - 15)
                eps = Mux(phase == self.Phase.DECAY, self.decay_rate, self.release_rate)
                m.d.comb += [mac_a.as_value().eq(level >> 15), mac_b.as_value().eq(eps)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    nl = Signal(33)
                    m.d.comb += nl.eq(level - (prod >> (AM.EPS_FRAC - 15)))
                    with m.If(phase == self.Phase.DECAY):
                        with m.If(nl <= sus32):
                            m.d.sync += [level.eq(sus32), phase.eq(self.Phase.SUSTAIN)]
                        with m.Else():
                            m.d.sync += level.eq(nl)
                    with m.Else():
                        with m.If(nl < AM.L_SILENT):
                            m.d.sync += [level.eq(0), phase.eq(self.Phase.IDLE)]
                        with m.Else():
                            m.d.sync += level.eq(nl)
                    m.next = 'VEL'

            with m.State('VEL'):
                # out = ((L >> 16) * velocity) >> 16
                m.d.comb += [mac_a.as_value().eq(level >> 16), mac_b.as_value().eq(cur_vel)]
                with mp.Multiply(m, a=mac_a, b=mac_b):
                    m.d.sync += out.eq(prod >> 16)
                    m.next = 'STORE'

            with m.State('STORE'):
                m.d.comb += [
                    st_wport.addr.eq(voice_ix),
                    st_wport.data.l_gate.eq(l_gate),
                    st_wport.data.phase.eq(phase),
                    st_wport.data.level.eq(level),
                    st_wport.en.eq(1),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += self.voice_active[n].eq(phase != self.Phase.IDLE)
                m.next = 'EMIT'

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.first.eq(voice_ix == 0),
                    self.o.payload.sample.as_value().eq(out),
                ]
                with m.If(self.o.ready):
                    with m.If(voice_ix == N - 1):
                        m.d.sync += voice_ix.eq(0)
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)
                    m.next = 'LOAD'

        return m


def clamp_bits(m, x, bits):
    """Saturate ``x`` to a signed ``bits``-wide range [-2**(bits-1), 2**(bits-1) - 1].
    Checks that all bits above ``bits - 1`` equal the sign (a few LUTs), instead of
    two full-width comparisons (carry chains) per call site."""
    x = Value.cast(x)
    v = Signal(signed(bits))
    upper = x[bits - 1:]
    in_range = (upper == 0) | (upper == (1 << len(upper)) - 1)
    m.d.comb += v.eq(Mux(in_range, x[:bits].as_signed(),
                         Mux(x[-1], -(1 << (bits - 1)), (1 << (bits - 1)) - 1)))
    return v


class MultiLadder(wiring.Component):

    """
    N Juno-style 4-pole ladder low-pass filters (Juno port, layer 1), with the
    Phase 1b exponential cutoff path, a DC block, the VCA and VCA Gate mode.

    Bit-exact with ``top.juno.ladder_model.LadderModel`` (see there for the
    algorithm): zero-delay-feedback TPT ladder, tanh (LUT) on the stage input,
    resonance ``k = 4.5 res``, Q compensation ``1 + 0.5 res^2``, 2x oversampling
    (input held, outputs averaged). ``reso`` is the resonance knob 0..32768.

    One ``PipelinedMAC``: ~29 multiplies per voice, ~130 cycles per voice.
    """

    N_OVERSAMPLE = LM.N_OVERSAMPLE
    CUTOFF_F_MIN = LM.CUTOFF_F_MIN
    CUTOFF_OCTAVES = LM.CUTOFF_OCTAVES
    CUTOFF_LUT_BITS = LM.CUTOFF_LUT_BITS
    FS = LM.FS

    @classmethod
    def cutoff_hz(cls, c):
        return LM.cutoff_hz(c)

    @classmethod
    def cutoff_lut(cls):
        """TPT stage coefficient G = g / (1 + g) (15-bit)."""
        return LM.g_lut()

    def __init__(self, n, sq=ASQ, env_uq=MultiADSR.EnvUQ, macp=None, x_sq=mac.SQNative):
        self.n = n
        self.sq = sq
        self.macp = macp or PipelinedMAC()
        super().__init__({
            # x: oscillator mix, 18 bit (+-4.0, see MultiJunoOsc.OUT_SAT)
            "i":    In(stream.Signature(Block(data.StructLayout({"x": x_sq, "env": env_uq})))),
            "reso": In(unsigned(16)),
            "voice_cutoff_base": In(data.ArrayLayout(ASQ, n)),
            "env_amt":    In(ASQ),
            "cutoff_cv":  In(ASQ),
            # LFO -> cutoff (normalized octaves, added like cutoff_cv but unscaled)
            "cutoff_mod": In(ASQ),
            # VCA Gate mode: VCA follows gate x velocity (~0.7 ms ramp) instead of env
            "vca_gate_mode":  In(1),
            "voice_gates":    In(data.ArrayLayout(unsigned(1), n)),
            "voice_velocity": In(data.ArrayLayout(env_uq, n)),
            "o":    Out(stream.Signature(Block(sq))),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        ONE = LM.ONE
        SAT = LM.SAT
        W = 22
        m.submodules.macp = mp = self.macp

        def sat(x):
            return clamp_bits(m, x, 18)

        # per-voice state
        state_layout = data.StructLayout({
            "s0": signed(18), "s1": signed(18), "s2": signed(18), "s3": signed(18),
            "dc_x": signed(W), "dc_y": signed(W),
            "gate_env": unsigned(16),
        })
        m.submodules.state_mem = state_mem = Memory(shape=state_layout, depth=N, init=[])
        st_rport = state_mem.read_port()
        st_wport = state_mem.write_port()
        m.d.comb += st_rport.en.eq(1)

        # lookup tables
        g_lut = LM.g_lut()
        m.submodules.g_lut = g_mem = Memory(shape=unsigned(16), depth=len(g_lut), init=g_lut)
        g_rd = g_mem.read_port()
        tanh_layout = data.StructLayout({"val": unsigned(16), "delta": unsigned(16)})
        tl = LM.tanh_lut()
        m.submodules.tanh_lut = tanh_mem = Memory(
            shape=tanh_layout, depth=len(tl), init=[{"val": a, "delta": b} for a, b in tl])
        tanh_rd = tanh_mem.read_port()
        m.d.comb += [g_rd.en.eq(1), tanh_rd.en.eq(1)]

        # MAC: z = (a * b) >> 15
        mac_a = Signal(mac.SQNative)
        mac_b = Signal(mac.SQNative)
        z = Signal(signed(40))
        m.d.comb += z.eq(mp.result.z.as_value() >> 15)

        def multiply(a, b, assigns, next_state):
            """``assigns(z)`` -> list of sync statements once a * b is ready."""
            m.d.comb += [mac_a.as_value().eq(a), mac_b.as_value().eq(b)]
            with mp.Multiply(m, a=mac_a, b=mac_b):
                m.d.sync += assigns(z)
                m.next = next_state

        voice_ix = Signal(range(N))
        in_first = Signal()
        x       = Signal(signed(W))
        env15   = Signal(signed(W))
        s       = [Signal(signed(18), name=f"s{i}") for i in range(4)]
        dc_x    = Signal(signed(W))
        dc_y    = Signal(signed(W))
        gate_env = Signal(16)
        cut_base = Signal(signed(W))
        c       = Signal(signed(W))
        G       = Signal(signed(W))
        beta    = Signal(signed(W))
        k_half  = Signal(signed(W))
        comp    = Signal(signed(W))
        xc      = Signal(signed(W))
        G2      = Signal(signed(W))
        G4      = Signal(signed(W))
        e       = Signal(signed(W))
        inv     = Signal(signed(W))
        t       = Signal(signed(W))
        S       = Signal(signed(W))
        kS      = Signal(signed(W))
        u       = Signal(signed(W))
        u_abs   = Signal(18)
        u_neg   = Signal()
        t_val   = Signal(signed(W))
        t_delta = Signal(signed(W))
        xi      = Signal(signed(W))
        acc     = Signal(signed(W))
        os_ix   = Signal(range(self.N_OVERSAMPLE))
        vca     = Signal(signed(W))
        out     = Signal(signed(16))

        cv_raw = self.cutoff_cv.as_value()
        cv_scaled = Signal(signed(W))
        m.d.comb += cv_scaled.eq(cv_raw - (cv_raw >> 3) - (cv_raw >> 4) + (cv_raw >> 7))

        with m.FSM():

            with m.State('WAIT-VALID'):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += [
                        x.eq(self.i.payload.sample.x.as_value()),
                        env15.eq(self.i.payload.sample.env.as_value() >> 1),
                        in_first.eq(self.i.payload.first),
                        os_ix.eq(0),
                        acc.eq(0),
                    ]
                    m.d.comb += st_rport.addr.eq(voice_ix)
                    m.next = 'LOAD'

            with m.State('LOAD'):
                m.d.sync += [
                    s[0].eq(st_rport.data.s0), s[1].eq(st_rport.data.s1),
                    s[2].eq(st_rport.data.s2), s[3].eq(st_rport.data.s3),
                    dc_x.eq(st_rport.data.dc_x),
                    dc_y.eq(st_rport.data.dc_y),
                    gate_env.eq(st_rport.data.gate_env),
                ]
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.sync += cut_base.eq(self.voice_cutoff_base[n].as_value())
                m.next = 'CUT-MUL'

            # --- exponential cutoff -> G -------------------------------------
            with m.State('CUT-MUL'):
                multiply(env15, self.env_amt.as_value(), lambda z: [
                    c.eq(cut_base + z + cv_scaled + self.cutoff_mod.as_value())], 'CUT-ADDR')

            with m.State('CUT-ADDR'):
                idx_shift = 15 - self.CUTOFF_LUT_BITS
                idx_max = (1 << self.CUTOFF_LUT_BITS) - 1
                with m.If(c < 0):
                    m.d.comb += g_rd.addr.eq(0)
                with m.Elif((c >> idx_shift) > idx_max):
                    m.d.comb += g_rd.addr.eq(idx_max)
                with m.Else():
                    m.d.comb += g_rd.addr.eq(c >> idx_shift)
                m.next = 'CUT-READ'

            with m.State('CUT-READ'):
                m.d.sync += [G.eq(g_rd.data), beta.eq(ONE - g_rd.data)]
                m.next = 'K-HALF'

            # --- per-sample coefficients --------------------------------------
            with m.State('K-HALF'):
                multiply(self.reso, LM.K_HALF_SCALE, lambda z: [k_half.eq(z)], 'COMP')
            with m.State('COMP'):
                multiply(self.reso, self.reso, lambda z: [comp.eq(ONE + (z >> 1))], 'XCOMP')
            with m.State('XCOMP'):
                # drive calibrated against the KR-106: input / 4, output summed
                multiply(x >> LM.IN_SHIFT, comp, lambda z: [xc.eq(z)], 'G2')
            with m.State('G2'):
                multiply(G, G, lambda z: [G2.eq(z)], 'G4')
            with m.State('G4'):
                multiply(G2, G2, lambda z: [G4.eq(z)], 'E')
            with m.State('E'):
                multiply(k_half, G4, lambda z: [e.eq(2 * z)], 'INV')
            with m.State('INV'):
                multiply(e, e, lambda z: [inv.eq(ONE - e + z)], 'S1')

            # --- one filter iteration (x2) ------------------------------------
            with m.State('S1'):
                multiply(G, s[0], lambda z: [t.eq(z)], 'S2')
            with m.State('S2'):
                multiply(G, t + s[1], lambda z: [t.eq(z)], 'S3')
            with m.State('S3'):
                multiply(G, t + s[2], lambda z: [t.eq(z)], 'S4')
            with m.State('S4'):
                multiply(beta, t + s[3], lambda z: [S.eq(z)], 'KS')
            with m.State('KS'):
                multiply(k_half, S, lambda z: [kS.eq(sat(2 * z))], 'U')
            with m.State('U'):
                multiply(sat(xc - kS), inv, lambda z: [u.eq(z)], 'TANH-ADDR')

            with m.State('TANH-ADDR'):
                a = Signal(18)
                m.d.comb += a.eq(Mux(u < 0, -u, u))
                a_cl = Signal(17)
                m.d.comb += a_cl.eq(Mux(a > (1 << 17) - 1, (1 << 17) - 1, a))
                m.d.comb += tanh_rd.addr.eq(a_cl >> LM.TANH_FRAC_BITS)
                m.d.sync += [u_abs.eq(a_cl), u_neg.eq(u < 0)]
                m.next = 'TANH-READ'

            with m.State('TANH-READ'):
                # latch the LUT output: the read address is only valid for one cycle,
                # but the multiply below needs stable operands for its whole latency
                m.d.sync += [
                    t_val.eq(tanh_rd.data.val),
                    t_delta.eq(tanh_rd.data.delta.as_signed()),
                ]
                m.next = 'TANH-INT'

            with m.State('TANH-INT'):
                frac = (u_abs & ((1 << LM.TANH_FRAC_BITS) - 1)) << (15 - LM.TANH_FRAC_BITS)
                tv = Signal(signed(W))
                m.d.comb += tv.eq(t_val + z)
                multiply(t_delta, frac, lambda z: [xi.eq(Mux(u_neg, -tv, tv))], 'STG0')

            for i in range(4):
                with m.State(f'STG{i}'):
                    nxt = f'STG{i + 1}' if i < 3 else 'ACC'
                    def stage(z, i=i):
                        y = sat(z + s[i])
                        return [s[i].eq(sat(y + z)), xi.eq(y)]
                    multiply(G, sat(xi - s[i]), stage, nxt)

            with m.State('ACC'):
                m.d.sync += acc.eq(acc + xi)
                with m.If(os_ix == self.N_OVERSAMPLE - 1):
                    m.next = 'DC-BLOCK'
                with m.Else():
                    m.d.sync += os_ix.eq(os_ix + 1)
                    m.next = 'S1'

            # --- DC block, gate level, VCA ------------------------------------
            with m.State('DC-BLOCK'):
                o = Signal(signed(W))
                m.d.comb += o.eq(acc >> LM.OUT_SHIFT)
                m.d.sync += [
                    dc_y.eq(o - dc_x + dc_y - (dc_y >> LM.DC_SHIFT)),
                    dc_x.eq(o),
                ]
                m.next = 'GATE-ENV'

            with m.State('GATE-ENV'):
                cur_gate = Signal()
                cur_vel = Signal(16)
                with m.Switch(voice_ix):
                    for n in range(N):
                        with m.Case(n):
                            m.d.comb += [
                                cur_gate.eq(self.voice_gates[n]),
                                cur_vel.eq(self.voice_velocity[n].as_value() >> 1),
                            ]
                target = Signal(16)
                m.d.comb += target.eq(Mux(cur_gate, cur_vel, 0))
                new_ge = Signal(16)
                with m.If(gate_env + LM.GATE_STEP <= target):
                    m.d.comb += new_ge.eq(gate_env + LM.GATE_STEP)
                with m.Elif(gate_env >= target + LM.GATE_STEP):
                    m.d.comb += new_ge.eq(gate_env - LM.GATE_STEP)
                with m.Else():
                    m.d.comb += new_ge.eq(target)
                m.d.sync += [
                    gate_env.eq(new_ge),
                    vca.eq(Mux(self.vca_gate_mode, new_ge, env15)),
                ]
                m.next = 'VCA'

            with m.State('VCA'):
                def vca_out(z):
                    return [out.eq(clamp_bits(m, z, 16))]
                multiply(sat(dc_y), vca, vca_out, 'STORE')

            with m.State('STORE'):
                m.d.comb += [
                    st_wport.addr.eq(voice_ix),
                    st_wport.data.s0.eq(s[0]), st_wport.data.s1.eq(s[1]),
                    st_wport.data.s2.eq(s[2]), st_wport.data.s3.eq(s[3]),
                    st_wport.data.dc_x.eq(dc_x),
                    st_wport.data.dc_y.eq(dc_y),
                    st_wport.data.gate_env.eq(gate_env),
                    st_wport.en.eq(1),
                ]
                m.next = 'EMIT'

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.first.eq(in_first),
                    self.o.payload.sample.as_value().eq(out),
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
    Sum N voices to mono (Juno: the stereo image comes from the chorus only).
    Each voice is scaled by 0.75 * 2 / N (polysyn's per-side gain, now summed),
    the sum saturates instead of wrapping.
    """

    def __init__(self, n, sq=ASQ):
        self.n = n
        self.sq = sq
        super().__init__({
            "i": In(stream.Signature(Block(sq))),
            "o": Out(stream.Signature(sq)),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n
        sq = self.sq
        mix_gain = fixed.Const(0.75 * 2.0 / N, shape=sq)
        acc_sq = fixed.SQ(sq.i_bits + 3, sq.f_bits)   # headroom for N voices
        acc = Signal(acc_sq)
        voice_ix = Signal(range(N))

        with m.FSM():
            with m.State('CLEAR'):
                m.d.sync += [acc.eq(0), voice_ix.eq(0)]
                m.next = 'WAIT-VALID'

            with m.State('WAIT-VALID'):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += acc.eq(acc + self.i.payload.sample * mix_gain)
                    with m.If(voice_ix == N - 1):
                        m.next = 'EMIT-OUTPUT'
                    with m.Else():
                        m.d.sync += voice_ix.eq(voice_ix + 1)

            with m.State('EMIT-OUTPUT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload.eq(acc.saturate(sq)),
                ]
                with m.If(self.o.ready):
                    m.next = 'CLEAR'

        return m


class JunoVoice(wiring.Component):

    def __init__(self, n_voices, sq=ASQ):
        self.n_voices = n_voices
        self.sq = sq
        super().__init__({
            # Mono voice sum (stereo comes from the chorus in JunoCore / JunoFx)
            "o": Out(stream.Signature(sq)),

            # Voice control (from MidiVoiceTracker)
            "voice_gates":     In(data.ArrayLayout(unsigned(1), self.n_voices)),
            # Per-voice pitch in semitones (MIDI note scale, see PitchToInc)
            "voice_pitch":     In(data.ArrayLayout(PitchToInc.PitchSQ, self.n_voices)),
            "voice_velocity":  In(data.ArrayLayout(MultiADSR.EnvUQ, self.n_voices)),

            # Global synth parameters (from CSR)
            "reso":           In(unsigned(16)),
            # Exponential cutoff (see MultiLadder): per-voice base from firmware,
            # bipolar env amount, cutoff CV (1 V/oct)
            "voice_cutoff_base": In(data.ArrayLayout(ASQ, self.n_voices)),
            "env_amt":        In(ASQ),
            "cutoff_cv":      In(ASQ),
            # LFO -> cutoff (normalized octaves) and VCA Gate mode
            "cutoff_mod":     In(ASQ),
            "vca_gate_mode":  In(1),
            "attack_rate":    In(unsigned(32)),
            "decay_rate":     In(unsigned(17)),
            "sustain_level":  In(MultiADSR.EnvUQ),
            "release_rate":   In(unsigned(17)),

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
        m.submodules.osc = osc = MultiJunoOsc(n=N)   # 18-bit output, see OUT_SAT
        m.submodules.adsr = adsr = MultiADSR(n=N)
        m.submodules.merge = merge = BlockMerge({"x": mac.SQNative, "env": MultiADSR.EnvUQ})
        # Juno 4-pole ladder (layer 1) with cutoff path, DC block, VCA, Gate mode
        m.submodules.svf = svf = MultiLadder(n=N, sq=sq, macp=PipelinedMAC())
        m.submodules.mixer = mixer = VoiceMixer(n=N, sq=sq)

        # Wire external ports -> internal components
        for n in range(N):
            m.d.comb += [
                osc.voice_pitch[n].eq(self.voice_pitch[n]),
                adsr.voice_gates[n].eq(self.voice_gates[n]),
                adsr.voice_velocity[n].eq(self.voice_velocity[n]),
                self.voice_active[n].eq(adsr.voice_active[n]),
                svf.voice_cutoff_base[n].eq(self.voice_cutoff_base[n]),
                svf.voice_gates[n].eq(self.voice_gates[n]),
                svf.voice_velocity[n].eq(self.voice_velocity[n]),
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
            svf.cutoff_mod.eq(self.cutoff_mod),
            svf.vca_gate_mode.eq(self.vca_gate_mode),
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
