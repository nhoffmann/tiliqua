# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Juno LFO (Juno port): one global triangle LFO with delay, routed to pitch
(vibrato), filter cutoff and pulse width.

Behaviour reference: KR-106 ``docs/DSP_ARCHITECTURE.md`` (algorithm only).

Per audio frame (``tick``):

1. ``phase += rate_inc`` (32 bit). Triangle ``tri = 1 - 4 |phase - 0.5|`` with a
   cubic soft clip at the peaks, ``s = tri (1.5 - 0.5 tri^2)``.
2. Delay: amplitude ramps linearly from 0 to 1 at ``delay_step`` per frame
   (30-bit accumulator). It restarts when a note is played after all keys were
   released (Juno "auto" mode). The KR-106 uses an RC curve; linear avoids a
   high-precision multiply.
3. ``lfo = s * amp``.
4. Routes:
   - ``vib = lfo * dco_depth`` (pitch offset, 1/1024 semitones),
   - ``cutoff_mod = lfo * vcf_depth`` (normalized octaves, see ``MultiSVF``),
   - PWM: ``k = knob`` (manual) or ``knob * (lfo + 1) / 2`` (LFO mode), then the
     256-entry ``pw_curve_lut`` maps the knob to the pulse rising-edge phase.

All audio-rate values use 15 fractional bits (32768 = 1.0). ``JunoLfoModel`` is a
bit-exact Python model; ``tests/test_juno_voice.py`` checks the gateware against it.
"""

import math

from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from tiliqua.dsp import ASQ, mac

ONE = 1 << 15
AMP_BITS = 30
PW_LUT_BITS = 8


def pw_knob_to_point(knob):
    """PW knob 0..1 -> pulse rising-edge phase, UQ(0,16).
    Juno: comparator at threshold 0.5 + 0.46 * knob on the curved ramp
    p (1 + 0.15 (1 - p)); solved for the linear phase p (duty ~54 % .. ~4.5 %)."""
    th = 0.5 + 0.46 * knob
    p = (1.15 - math.sqrt(1.15 ** 2 - 0.6 * th)) / 0.3
    return min(65535, round(p * 65536))


def pw_curve_lut():
    n = 1 << PW_LUT_BITS
    return [pw_knob_to_point(i / (n - 1)) for i in range(n)]


def pw_lut_index(k):
    """k: knob 0..32768 (15-bit) -> LUT index 0..255."""
    return min((1 << PW_LUT_BITS) - 1, (k * ((1 << PW_LUT_BITS) - 1) + (ONE >> 1)) >> 15)


class JunoLfoModel:
    """Bit-exact model of ``JunoLfo``. Call ``frame()`` once per tick."""

    def __init__(self):
        self.phase = 0
        self.amp = 0
        self.last_any_gate = 0
        self.lut = pw_curve_lut()

    def frame(self, rate_inc, delay_step, any_gate, dco_depth, vcf_depth,
              pw_knob, pwm_lfo):
        self.phase = (self.phase + rate_inc) & 0xFFFFFFFF
        if any_gate and not self.last_any_gate:
            self.amp = 0
        else:
            self.amp = min(1 << AMP_BITS, self.amp + delay_step)
        self.last_any_gate = any_gate

        d = abs((self.phase >> 16) - ONE)
        tri = ONE - 2 * d
        t2 = (tri * tri) >> 15
        s = (tri * (ONE + (ONE >> 1) - (t2 >> 1))) >> 15
        amp15 = self.amp >> (AMP_BITS - 15)
        lfo = max(-ONE, min(ONE - 1, (s * amp15) >> 15))

        vib = (lfo * dco_depth) >> 15
        cutoff_mod = (lfo * vcf_depth) >> 15
        k = (pw_knob * ((lfo + ONE) >> 1)) >> 15 if pwm_lfo else pw_knob
        pw_point = self.lut[pw_lut_index(k)]
        return dict(lfo=lfo, vib=vib, cutoff_mod=cutoff_mod, pw_point=pw_point)


class JunoLfo(wiring.Component):

    """Gateware LFO, bit-exact with ``JunoLfoModel``. Starts one update per
    ``tick`` strobe (once per audio frame); outputs hold between updates."""

    def __init__(self, macp):
        """``macp``: a ``juno_voice.PipelinedMAC`` (passed in to avoid an import
        cycle / the ``top.py`` script-name clash)."""
        self.macp = macp
        super().__init__({
            "tick":       In(1),
            "rate_inc":   In(unsigned(32)),
            "delay_step": In(unsigned(32)),
            "any_gate":   In(1),
            "dco_depth":  In(unsigned(16)),   # vibrato depth, 1/1024 semitones at lfo = 1
            "vcf_depth":  In(unsigned(16)),   # cutoff depth, normalized octaves (15-bit)
            "pw_knob":    In(unsigned(16)),   # 0..32768
            "pwm_lfo":    In(1),              # 1 = PWM source LFO, 0 = manual
            "lfo":        Out(signed(16)),
            "vib":        Out(signed(16)),
            "cutoff_mod": Out(signed(16)),
            "pw_point":   Out(unsigned(16)),
        })

    def elaborate(self, platform):
        m = Module()

        m.submodules.macp = mp = self.macp
        lut = pw_curve_lut()
        m.submodules.pw_lut = pw_mem = Memory(shape=unsigned(16), depth=len(lut), init=lut)
        pw_rd = pw_mem.read_port()
        m.d.comb += pw_rd.en.eq(1)

        phase = Signal(32)
        amp = Signal(AMP_BITS + 1)
        last_any_gate = Signal()
        tri = Signal(signed(18))
        t2 = Signal(signed(18))
        s = Signal(signed(18))
        lfo = Signal(signed(18))
        k = Signal(unsigned(16))

        mac_a = Signal(mac.SQNative)
        mac_b = Signal(mac.SQNative)
        mac_z = Signal(signed(40))
        m.d.comb += mac_z.eq(mp.result.z.as_value() >> 15)

        # lfo = clamp(s * amp) to the signed 16-bit range
        lfo_clamped = Signal(signed(18))
        with m.If(mac_z > ONE - 1):
            m.d.comb += lfo_clamped.eq(ONE - 1)
        with m.Elif(mac_z < -ONE):
            m.d.comb += lfo_clamped.eq(-ONE)
        with m.Else():
            m.d.comb += lfo_clamped.eq(mac_z)

        def multiply(a, b, dest, value, next_state):
            """dest <= value (an expression of mac_z) once a * b is ready."""
            m.d.comb += [mac_a.as_value().eq(a), mac_b.as_value().eq(b)]
            with mp.Multiply(m, a=mac_a, b=mac_b):
                m.d.sync += dest.eq(value)
                m.next = next_state

        with m.FSM():

            with m.State('IDLE'):
                with m.If(self.tick):
                    new_phase = Signal(32)
                    m.d.comb += new_phase.eq(phase + self.rate_inc)
                    d = Signal(17)
                    ph16 = new_phase[16:32]
                    m.d.comb += d.eq(Mux(ph16 >= ONE, ph16 - ONE, ONE - ph16))
                    m.d.sync += [
                        phase.eq(new_phase),
                        tri.eq(ONE - 2 * d),
                        last_any_gate.eq(self.any_gate),
                    ]
                    amp_next = Signal(AMP_BITS + 2)
                    m.d.comb += amp_next.eq(amp + self.delay_step)
                    with m.If(self.any_gate & ~last_any_gate):
                        m.d.sync += amp.eq(0)
                    with m.Elif(amp_next > (1 << AMP_BITS)):
                        m.d.sync += amp.eq(1 << AMP_BITS)
                    with m.Else():
                        m.d.sync += amp.eq(amp_next)
                    m.next = 'TRI-SQ'

            with m.State('TRI-SQ'):
                multiply(tri, tri, t2, mac_z, 'SOFTCLIP')

            with m.State('SOFTCLIP'):
                multiply(tri, ONE + (ONE >> 1) - (t2 >> 1), s, mac_z, 'AMP')

            with m.State('AMP'):
                multiply(s, amp >> (AMP_BITS - 15), lfo, lfo_clamped, 'VIB')

            with m.State('VIB'):
                multiply(lfo, self.dco_depth, self.vib, mac_z, 'VCF')

            with m.State('VCF'):
                multiply(lfo, self.vcf_depth, self.cutoff_mod, mac_z, 'PWM')

            with m.State('PWM'):
                with m.If(self.pwm_lfo):
                    multiply(self.pw_knob, (lfo + ONE) >> 1, k, mac_z, 'PW-ADDR')
                with m.Else():
                    m.d.sync += k.eq(self.pw_knob)
                    m.next = 'PW-ADDR'

            with m.State('PW-ADDR'):
                idx = Signal(17)
                m.d.comb += idx.eq((k * ((1 << PW_LUT_BITS) - 1) + (ONE >> 1)) >> 15)
                m.d.comb += pw_rd.addr.eq(Mux(idx > (1 << PW_LUT_BITS) - 1,
                                              (1 << PW_LUT_BITS) - 1, idx))
                m.next = 'PW-READ'

            with m.State('PW-READ'):
                m.d.sync += [
                    self.pw_point.eq(pw_rd.data),
                    self.lfo.eq(lfo),
                ]
                m.next = 'IDLE'

        return m

