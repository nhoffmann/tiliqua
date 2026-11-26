# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
``JunoCvRouter``: audio-rate CV routing for the Juno port (Phase 5.2).

All 4 input jacks are assignable in the menu. Pitch and cutoff are routed here in
gateware (every sample); all other targets are applied by the firmware at its tick
(``fw/src/cv.rs``). The firmware tells this block which jack feeds pitch / cutoff
(``*_src``: 0 = off, 1..4 = jack 0..3) and the amount (SQ(2,14), 1.0 = 16384 =
exactly 1 V/oct). An unpatched jack (``jack`` bit low) contributes 0.

- CV samples: ASQ raw, 1 V = 4000 counts (+-32768 = +-8.192 V).
- ``pitch``: semitone offset in PitchSQ raw (1/1024 semitone), added to the pitch
  bend after the glide (``JunoCore.pitch_bend``):
  ``cv * amt / 2^14 * 12 * 1024 / 4000`` = ``x * 3.072``, done with shift-adds
  ``3 + 1/16 + 1/128 + 1/512`` (= 3.0723, +0.01 % = 0.12 cents over 10 octaves).
- ``cutoff``: ASQ raw for ``JunoVoice.cutoff_cv`` (1 V/oct there), ``cv * amt / 2^14``
  saturated to +-1.
- Latency: 2 clocks (registered product), irrelevant at 48 kHz.
"""

from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from tiliqua.dsp import ASQ

try:
    from juno_voice import PitchToInc, clamp_bits
except ImportError:
    from top.juno.juno_voice import PitchToInc, clamp_bits

AMT_ONE = 1 << 14


def pitch_from_cv(cv, amt):
    """Bit-exact model: CV raw x amount raw -> PitchSQ raw."""
    x = (cv * amt) >> 14
    return 3 * x + (x >> 4) + (x >> 7) + (x >> 9)


def cutoff_from_cv(cv, amt):
    x = (cv * amt) >> 14
    return max(-32768, min(32767, x))


class JunoCvRouter(wiring.Component):

    def __init__(self):
        super().__init__({
            "cv":         In(data.ArrayLayout(ASQ, 4)),
            "jack":       In(4),
            "pitch_src":  In(3),
            "pitch_amt":  In(signed(16)),
            "cutoff_src": In(3),
            "cutoff_amt": In(signed(16)),
            "pitch":      Out(signed(PitchToInc.PitchSQ.as_shape().width)),
            "cutoff":     Out(ASQ),
        })

    def elaborate(self, platform):
        m = Module()

        def select(src):
            v = Signal(signed(16))
            with m.Switch(src):
                for j in range(4):
                    with m.Case(j + 1):
                        m.d.comb += v.eq(Mux(self.jack[j], self.cv[j].as_value(), 0))
            return v

        p_cv, c_cv = select(self.pitch_src), select(self.cutoff_src)
        p_prod = Signal(signed(32))
        c_prod = Signal(signed(32))
        m.d.sync += [p_prod.eq(p_cv * self.pitch_amt), c_prod.eq(c_cv * self.cutoff_amt)]

        x = Signal(signed(18))
        m.d.comb += x.eq(p_prod >> 14)
        m.d.sync += self.pitch.eq(3 * x + (x >> 4) + (x >> 7) + (x >> 9))

        y = Signal(signed(18))
        m.d.comb += y.eq(c_prod >> 14)
        m.d.sync += self.cutoff.as_value().eq(clamp_bits(m, y, 16))
        return m


# CV output sources (Phase 5.3), menu-assignable to out0 / out1.
OUT_OFF, OUT_LFO, OUT_ENV, OUT_GATE, OUT_PITCH = range(5)
GATE_HIGH = 20000          # 5 V
PITCH_REF_NOTE = 60        # C4 = 0 V, 1 V/oct


def cv_out_model(src, lfo, env, gate, note):
    """Bit-exact model of one ``JunoCvOut`` output (ASQ raw)."""
    if src == OUT_LFO:      # +-1 -> +-5 V (x 0.6113)
        return (lfo >> 1) + (lfo >> 3) - (lfo >> 6) + (lfo >> 9)
    if src == OUT_ENV:      # 0..1 -> 0..8.19 V
        return env >> 1
    if src == OUT_GATE:
        return GATE_HIGH if gate else 0
    if src == OUT_PITCH:    # (note - 60) * 333.33 counts = 1 V/oct
        return ((note - PITCH_REF_NOTE) * 21333) >> 6
    return 0


class JunoCvOut(wiring.Component):

    """
    Two menu-assignable CV outputs (out0 / out1, Phase 5.3). Sources: LFO (+-5 V,
    after the delay fade, before the depth knobs), envelope of the last-triggered
    voice (0..8 V, includes velocity), gate of the last-triggered voice (0 / 5 V),
    its note as 1 V/oct pitch (C4 = 0 V; note only, no glide / bend).
    Registered outputs; ``src*``: 0 off, 1 LFO, 2 env, 3 gate, 4 pitch.
    """

    def __init__(self):
        super().__init__({
            "src0": In(3),
            "src1": In(3),
            "lfo":  In(signed(16)),
            "env":  In(unsigned(16)),
            "gate": In(1),
            "note": In(unsigned(8)),
            "out0": Out(ASQ),
            "out1": Out(ASQ),
        })

    def elaborate(self, platform):
        m = Module()
        lfo = self.lfo
        v_lfo = Signal(signed(16))
        v_pitch = Signal(signed(16))
        n = Signal(signed(9))
        m.d.comb += [
            v_lfo.eq((lfo >> 1) + (lfo >> 3) - (lfo >> 6) + (lfo >> 9)),
            n.eq(self.note - PITCH_REF_NOTE),
            v_pitch.eq((n * 21333) >> 6),
        ]
        for src, out in ((self.src0, self.out0), (self.src1, self.out1)):
            with m.Switch(src):
                with m.Case(OUT_LFO):
                    m.d.sync += out.as_value().eq(v_lfo)
                with m.Case(OUT_ENV):
                    m.d.sync += out.as_value().eq(self.env >> 1)
                with m.Case(OUT_GATE):
                    m.d.sync += out.as_value().eq(Mux(self.gate, GATE_HIGH, 0))
                with m.Case(OUT_PITCH):
                    m.d.sync += out.as_value().eq(v_pitch)
                with m.Default():
                    m.d.sync += out.as_value().eq(0)
        return m
