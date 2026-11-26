# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Bit-exact Python model of ``JunoFx`` (``juno_fx.py``): the Juno global effects on
the mono voice sum: 4-position HPF, then the BBD-style stereo chorus.

Behaviour reference: KR-106 ``docs/DSP_ARCHITECTURE.md`` (algorithm only).

HPF (one TPT one-pole low-pass ``lp``, ``v = G (x - s); lp = v + s; s' = lp + v``):

- 0: low-shelf boost, ``y = x + K lp`` with ``lp`` at 150 Hz, ``K = 10^(10/20) - 1`` (+10 dB)
- 1: flat, ``y = x``
- 2: high-pass 240 Hz, ``y = x - lp``
- 3: high-pass 720 Hz, ``y = x - lp``

Chorus (mono in, stereo out):

- 15 kHz one-pole pre-filter, then a 512-sample delay line.
- Two triangle LFOs (I: 0.513 Hz, +-0.5 ms; II: 0.863 Hz, +-1.1 ms). The delay is
  ``3.5 ms + mI * dI * triI + mII * dII * triII`` (``mI``, ``mII`` from the mode:
  I, II or both), the right tap uses the inverted LFOs (180 degrees).
- Linear interpolation between delay samples, 15 kHz one-pole post-filter per tap.
- Mix per channel ``1.2 (0.4 dry + 0.6 wet) = 0.48 dry + 0.72 wet``; chorus off
  outputs the dry signal on both channels.

All values are integers with 15 fractional bits (32768 = 1.0), ``mul(a, b) =
(a * b) >> 15`` like the gateware MAC. Delays are in 1/256 samples.
Not modelled yet (see "Nice to have"): BBD charge saturation, delay-dependent
BBD bandwidth filter, Hermite interpolation.
"""

import math

ONE = 1 << 15
SAT = (1 << 17) - 1
FS = 48000
DELAY_BITS = 9                       # 512-sample delay line
DELAY_FRAC = 8                       # delays in 1/256 samples
CENTER = round(0.0035 * FS * (1 << DELAY_FRAC))        # 3.5 ms
DEPTH_I = round(0.0005 * FS * (1 << DELAY_FRAC))       # +-0.5 ms
DEPTH_II = round(0.0011 * FS * (1 << DELAY_FRAC))      # +-1.1 ms
RATE_I = round(0.513 / FS * 2 ** 32)
RATE_II = round(0.863 / FS * 2 ** 32)
SHELF_K = round((10 ** (10 / 20) - 1) * ONE)           # +10 dB shelf
DRY = round(0.48 * ONE)
WET = round(0.72 * ONE)

HPF_FLAT = 1
CHORUS_OFF, CHORUS_I, CHORUS_II, CHORUS_I_II = 0, 1, 2, 3


def mul(a, b):
    return (a * b) >> 15


def sat(x):
    """Clamp to the 18-bit signed MAC operand range (as the gateware's clamp_bits)."""
    return max(-SAT - 1, min(SAT, x))


def tpt_g(fc):
    g = math.tan(math.pi * fc / FS)
    return round(g / (1 + g) * ONE)


HPF_G = [tpt_g(150), 0, tpt_g(240), tpt_g(720)]
G_15K = tpt_g(15000)


def tri(phase):
    """Triangle in 15-bit fixed point: 1 - 4 |phase - 0.5| (-32768 .. 32768)."""
    return ONE - 2 * abs((phase >> 16) - ONE)


class FxModel:

    def __init__(self):
        self.hpf_s = 0
        self.pre_s = 0
        self.post_s = [0, 0]
        self.buf = [0] * (1 << DELAY_BITS)
        self.wptr = 0
        self.ph_i = 0
        self.ph_ii = 0

    @staticmethod
    def _onepole(G, x, s):
        v = mul(G, sat(x - s))
        lp = sat(v + s)
        return lp, sat(lp + v)

    def sample(self, x, hpf_mode=HPF_FLAT, chorus_mode=CHORUS_OFF):
        """x: mono voice sum (ASQ raw). Returns (left, right) ASQ raw."""
        # HPF
        lp, self.hpf_s = self._onepole(HPF_G[hpf_mode], x, self.hpf_s)
        if hpf_mode == 0:
            y = sat(x + mul(SHELF_K, lp))
        elif hpf_mode == 1:
            y = x
        else:
            y = sat(x - lp)

        # chorus: pre-filter + write
        pre, self.pre_s = self._onepole(G_15K, y, self.pre_s)
        mask = (1 << DELAY_BITS) - 1
        self.buf[self.wptr] = pre

        self.ph_i = (self.ph_i + RATE_I) & 0xFFFFFFFF
        self.ph_ii = (self.ph_ii + RATE_II) & 0xFFFFFFFF
        mod = 0
        if chorus_mode in (CHORUS_I, CHORUS_I_II):
            mod += mul(tri(self.ph_i), DEPTH_I)
        if chorus_mode in (CHORUS_II, CHORUS_I_II):
            mod += mul(tri(self.ph_ii), DEPTH_II)

        outs = []
        for ch, d in enumerate((CENTER + mod, CENTER - mod)):
            n, frac = d >> DELAY_FRAC, d & ((1 << DELAY_FRAC) - 1)
            a = self.buf[(self.wptr - n) & mask]          # delay n
            b = self.buf[(self.wptr - n - 1) & mask]      # delay n + 1
            tap = a + mul(sat(b - a), frac << (15 - DELAY_FRAC))
            wet, self.post_s[ch] = self._onepole(G_15K, tap, self.post_s[ch])
            if chorus_mode == CHORUS_OFF:
                out = y
            else:
                out = mul(DRY, y) + mul(WET, wet)
            outs.append(max(-ONE, min(ONE - 1, out)))
        self.wptr = (self.wptr + 1) & mask
        return tuple(outs)
