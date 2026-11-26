# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
``JunoGlide``: per-voice portamento on the log pitch (Juno port, Phase 4.5).

Each voice's pitch follows its note target with a one-pole (exponential) glide,
so the glide time does not depend on the interval, as an RC on the pitch CV:

    d = target - g;  g += round(d * coef / 2**24)

``g`` keeps ``FRAC`` extra fractional bits below the 1/1024-semitone pitch LSB.
When the rounded step becomes 0 (|d| below ~0.5 cent at the slowest setting) ``g``
snaps to the target, so the glide always lands exactly.

- ``coef`` = (1 - exp(-1 / (tau * fs))) * 2**24, 18 bit (shortest tau ~64 samples).
  ``coef == 0`` = portamento off: the step is 0, so ``g`` snaps to the target.
- Target = note pitch + per-voice ``detune`` (1/1024 semitones, signed 10 bit;
  firmware: per-voice pitch variance + unison spread, Phase 4.6).
- Updated once per output frame (``tick``), one voice after the other on one
  registered multiplier and one adder (~3 cycles per voice). A new note therefore
  reaches the oscillator at the next frame (<= 1 sample late).
- Pitch bend, vibrato and the DCO range are added after the glide (``JunoCore``),
  so they stay immediate.

Behaviour reference: KR-106 ``docs/DSP_ARCHITECTURE.md`` ("Portamento & Unison":
per-voice one-pole glide, cubic knob -> time). Own implementation.
"""

import math

from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

try:
    from juno_voice import PitchToInc
except ImportError:
    from top.juno.juno_voice import PitchToInc

FRAC = 14
COEF_BITS = 18
COEF_FRAC = 24
PITCH_W = PitchToInc.PitchSQ.as_shape().width
DETUNE_BITS = 10
RESET_NOTE = 60          # g starts at C4 (first glide after power-up comes from there)
GLIDE_MAX_S = 3.0        # knob 1 -> 3 s time constant (KR-106: knob^3 * 3 s)


def glide_time(knob):
    """Portamento knob 0..1 -> time constant in seconds (0 = off)."""
    k = min(max(knob, 0.0), 1.0)
    return k ** 3 * GLIDE_MAX_S


def glide_coef(seconds, fs=48000):
    """Time constant -> ``coef`` register (0 = off)."""
    if seconds <= 0:
        return 0
    c = round(-math.expm1(-1.0 / (seconds * fs)) * (1 << COEF_FRAC))
    return max(1, min((1 << COEF_BITS) - 1, c))


class GlideModel:

    """Bit-exact model of ``JunoGlide`` (one ``tick`` = one output frame)."""

    def __init__(self, n):
        self.g = [RESET_NOTE << (PitchToInc.FINE_BITS + FRAC)] * n

    def tick(self, targets, coef, detune=None):
        """``targets``: raw PitchSQ per voice, ``detune``: raw per voice."""
        detune = detune or [0] * len(targets)
        for i, t in enumerate(targets):
            tt = (t + detune[i]) << FRAC
            step = ((tt - self.g[i]) * coef + (1 << (COEF_FRAC - 1))) >> COEF_FRAC
            self.g[i] = tt if step == 0 else self.g[i] + step

    def pitch(self, i):
        return self.g[i] >> FRAC


class JunoGlide(wiring.Component):

    def __init__(self, n_voices):
        self.n_voices = n_voices
        super().__init__({
            "tick":   In(1),
            "coef":   In(unsigned(COEF_BITS)),
            "target": In(data.ArrayLayout(PitchToInc.PitchSQ, n_voices)),
            "detune": In(data.ArrayLayout(signed(DETUNE_BITS), n_voices)),
            "pitch":  Out(data.ArrayLayout(PitchToInc.PitchSQ, n_voices)),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n_voices
        W = PITCH_W + FRAC

        g = [Signal(signed(W), name=f"g_{n}",
                    init=RESET_NOTE << (PitchToInc.FINE_BITS + FRAC)) for n in range(N)]
        tgt = [Signal(signed(W), name=f"tgt_{n}") for n in range(N)]
        for n in range(N):
            m.d.comb += [
                tgt[n].eq(self.target[n].as_value() << FRAC),
                self.pitch[n].as_value().eq(g[n][FRAC:]),
            ]

        idx = Signal(range(N))
        d = Signal(signed(W + 1))
        coef = Signal(unsigned(COEF_BITS))
        prod = Signal(signed(W + 1 + COEF_BITS + 1))
        step = Signal(signed(W + 1 + COEF_BITS + 1 - COEF_FRAC))
        cur_t = Signal(signed(W))
        cur_g = Signal(signed(W))
        m.d.comb += [
            cur_t.eq(Array(tgt)[idx] + (Array(self.detune)[idx] << FRAC)),
            cur_g.eq(Array(g)[idx]),
            step.eq((prod + (1 << (COEF_FRAC - 1))) >> COEF_FRAC),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += idx.eq(0)
                with m.If(self.tick):
                    m.d.sync += coef.eq(self.coef)
                    m.next = "DIFF"
            with m.State("DIFF"):
                m.d.sync += d.eq(cur_t - cur_g)
                m.next = "MUL"
            with m.State("MUL"):
                m.d.sync += prod.eq(d * coef)
                m.next = "ACC"
            with m.State("ACC"):
                # one shared adder; snap to the target once the step rounds to 0
                new_g = Signal(signed(W))
                m.d.comb += new_g.eq(Mux(step == 0, cur_t, cur_g + step))
                for n in range(N):
                    with m.If(idx == n):
                        m.d.sync += g[n].eq(new_g)
                m.d.sync += idx.eq(idx + 1)
                m.next = "IDLE" if N == 1 else "DIFF"
                with m.If(idx == N - 1):
                    m.next = "IDLE"

        return m
