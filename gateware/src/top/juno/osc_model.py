# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Bit-exact Python model of ``MultiJunoOsc`` (``juno_voice.py``).

All values are integers. Audio-rate values use 15 fractional bits
(32768 = 1.0), phases and increments are UQ(0, 32) fractions of a cycle.
The gateware must produce exactly the same samples; ``tests/test_juno_voice.py``
checks this. Keep both in sync when changing the algorithm.

Per voice and sample:

1. ``t = phase + inc`` (wraps at 2**32). On wrap the sub flip-flop toggles.
2. PolyBLEP: for the wrap and for the pulse rising edge, if the edge is within
   one increment of ``t``, ``y = d / inc`` (16-bit, restoring division) and
   ``c = (1 - y)**2``; ``c`` is added or subtracted depending on edge direction
   and on which side of the edge ``t`` is.
3. Curved saw ``ramp = p + 0.15 p (1 - p)`` (Juno capacitor ramp), ``saw = 2 ramp - 1``.
4. Pulse: high when ``t >= pw_point`` (linear phase; firmware pre-compensates the
   curvature), low otherwise.
5. Sub: square one octave down, gentle one-pole low-pass (``y += (x - y) / 2``).
6. Noise: one 32-bit Galois LFSR shared by all voices, advanced once per frame,
   one-pole low-pass (spectral tilt).
7. Mixer: saw / 2 + pulse / 2 + sub * sub_gain + noise * noise_gain, saturated.
"""

ONE = 1 << 15                 # 1.0 in 15-bit fixed point
CURVE = 4915                  # 0.15 * 2**15
LFSR_TAPS = 0x80200003        # 32-bit maximal-length Galois LFSR
LFSR_SEED = 0xACE1ACE1
DIV_BITS = 16


def div_frac(d, den):
    """Restoring division: floor(d * 2**16 / den) for 0 <= d < den."""
    rem, q = d, 0
    for _ in range(DIV_BITS):
        rem <<= 1
        q <<= 1
        if rem >= den:
            rem -= den
            q |= 1
    return q


def blep_c(d, inc):
    """(1 - d/inc)**2 in 15-bit fixed point (d < inc)."""
    om = ((1 << DIV_BITS) - div_frac(d, inc)) >> 1
    return (om * om) >> 15


def lfsr_next(x):
    return (x >> 1) ^ LFSR_TAPS if x & 1 else x >> 1


def to_s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


class OscModel:
    """Model of the whole N-voice block, one ``frame()`` per sample frame."""

    def __init__(self, n):
        self.n = n
        self.phase = [0] * n
        self.sub = [0] * n
        self.sub_lp = [0] * n
        self.lfsr = LFSR_SEED
        self.noise_lp = 0

    def voice(self, v, inc, saw_en, pulse_en, pw_point, sub_gain, noise_gain):
        """One sample of voice ``v``. ``pw_point`` UQ(0,16), gains UQ(0,16)."""
        if v == 0:
            self.lfsr = lfsr_next(self.lfsr)
            n = to_s16(self.lfsr >> 16)
            self.noise_lp += (n - self.noise_lp) >> 1

        t_full = self.phase[v] + inc
        wrapped = t_full >> 32
        t = t_full & 0xFFFFFFFF
        self.phase[v] = t
        if wrapped:
            self.sub[v] ^= 1

        # wrap edge (saw reset, pulse fall, sub toggle)
        corr_wrap = 0
        c_w = 0
        if t < inc:                      # just after the wrap
            c_w = blep_c(t, inc)
            corr_wrap = c_w
        elif (1 << 32) - t < inc:        # just before the wrap
            c_w = blep_c((1 << 32) - t, inc)
            corr_wrap = -c_w

        # pulse rising edge at pw
        w = pw_point << 16
        corr_pw = 0
        if t >= w and t - w < inc:       # just after the rise
            corr_pw = -blep_c(t - w, inc)
        elif t < w and w - t < inc:      # just before the rise
            corr_pw = blep_c(w - t, inc)

        # curved saw
        p = t >> 17
        q = (p * (ONE - p)) >> 15
        ramp = p + ((q * CURVE) >> 15)
        saw = 2 * ramp - ONE + corr_wrap

        pulse = (ONE if t >= w else -ONE) + corr_wrap + corr_pw

        if self.sub[v]:
            sub = ONE - (c_w if c_w else 0)
        else:
            sub = -ONE + (c_w if c_w else 0)
        self.sub_lp[v] += (sub - self.sub_lp[v]) >> 1

        acc = 0
        if saw_en:
            acc += saw >> 1
        if pulse_en:
            acc += pulse >> 1
        acc += (self.sub_lp[v] * (sub_gain >> 1)) >> 15
        acc += (self.noise_lp * (noise_gain >> 1)) >> 15
        return max(-ONE, min(ONE - 1, acc))
