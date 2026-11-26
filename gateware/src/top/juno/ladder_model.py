# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Bit-exact Python model of ``MultiLadder`` (``juno_voice.py``): Juno-style 4-pole
ladder low-pass with exponential cutoff, DC block, VCA and Gate mode.

Layer 1 of the Juno VCF (see ``docs/JUNO_PORTING_PLAN.md``):

- Four TPT one-pole stages (``v = G (x - s); y = v + s; s' = y + v``) with the
  global feedback solved without a unit delay (zero-delay feedback):
  ``y4 = G^4 u + S``, ``u = (x - k S) / (1 + k G^4)``.
- Saturation: ``tanh`` (lookup table + linear interpolation) on ``u``, the signal
  entering the stages; this limits the resonance.
- Resonance ``k = 4.5 res`` (self-oscillates near the top), Q compensation
  ``x * (1 + 0.5 res^2)`` (Juno BA662 behaviour, per KR-106 docs).
- 2x oversampling: the input is held for two filter iterations, the two outputs
  are summed; input scaled by 1/4 (drive calibrated against the KR-106).
- ``G = g / (1 + g)``, ``g = tan(pi fc / 96 kHz)`` from a 1024-entry cutoff LUT;
  ``1 / (1 + k G^4) ~= 1 - e + e^2`` (``e = k G^4 <= 0.04``, error < 1e-4).

All values are integers with 15 fractional bits (32768 = 1.0); ``mul(a, b) =
(a * b) >> 15`` like the gateware's MAC. Keep in sync with the gateware.
"""

import math

ONE = 1 << 15
SAT = (1 << 17) - 1            # 18-bit signed MAC operand range
FS = 48000
N_OVERSAMPLE = 2
CUTOFF_F_MIN = 20.0           # 20 Hz .. 20.5 kHz over 10 octaves (Phase 4.1; was 12 Hz .. 12.2 kHz)
CUTOFF_OCTAVES = 10
CUTOFF_LUT_BITS = 10
TANH_LUT_BITS = 10             # over |u| in [0, 4)
TANH_FRAC_BITS = 17 - TANH_LUT_BITS   # |u| raw < 2**17
K_HALF_SCALE = 73728           # 2.25 * 2**15: k / 2 = res * 2.25
DC_SHIFT = 9
GATE_STEP = 1 << 10
# Drive / gain staging, calibrated against the KR-106 (tools/kr106_compare.py):
# a full-level oscillator mix overdrove the tanh (saw + sub balance -10.9 dB vs
# KR-106 -8.3 dB). The input is scaled down by 2**IN_SHIFT and the two
# oversampled outputs are summed (not averaged / halved), so the linear gain
# through the filter is unchanged but the drive into the saturation is 12 dB lower.
IN_SHIFT = 2
OUT_SHIFT = 0

# Resonance knob -> internal resonance (the `reso` register, k = 4.5 res), piecewise
# linear at knob = 0, 1/8, ..., 1. Calibrated so the resonance peak (saw C2, cutoff
# ~800 Hz) matches the KR-106 Juno-106 RES slider; the KR-106 reaches
# self-oscillation only at the very top. Applied by the firmware (fw/src/vcf.rs)
# and by sim_render.prepare(); the gateware takes the register value.
RES_CURVE = [0.0, 0.19, 0.375, 0.4625, 0.55, 0.62, 0.715, 0.825, 0.95]


# Cutoff knob (0..32768) -> cutoff c (normalized octaves, 15 bit), behaving like
# the Juno-106 VCF FREQ slider (knob = slider / 127). The KR-106 measurement
# (tools/kr106_compare.py) is linear in octaves: c = 0.011079 (slider - 16), i.e.
# c = 1.40703 knob - 0.17726. Not clamped here: env / kbd / mod are added first,
# the gateware clamps the sum. Same formula in firmware fw/src/vcf.rs.
CUT_KNOB_SCALE = 46106          # 1.40703 * 2**15
CUT_KNOB_OFFSET = 5809          # 0.17726 * 2**15


def cutoff_knob_to_c(knob_raw):
    """Cutoff knob 0..32768 -> c raw (15 bit, may be < 0 or > 1.0)."""
    return ((knob_raw * CUT_KNOB_SCALE) >> 15) - CUT_KNOB_OFFSET


# Keyboard tracking reference: C1 (MIDI 24, 32.703 Hz), as the KR-106 formula
# log(noteFreq / 32.703) * kbd.
KBD_REF_NOTE = 24


def res_knob_to_reg(knob):
    """Resonance knob 0..1 -> `reso` register value 0..32768."""
    k = min(max(knob, 0.0), 1.0) * (len(RES_CURVE) - 1)
    i = min(int(k), len(RES_CURVE) - 2)
    return round((RES_CURVE[i] + (k - i) * (RES_CURVE[i + 1] - RES_CURVE[i])) * ONE)


def mul(a, b):
    return (a * b) >> 15


def sat(x):
    """Clamp to the 18-bit signed MAC operand range (as the gateware's clamp_bits)."""
    return max(-SAT - 1, min(SAT, x))


def cutoff_hz(c):
    return CUTOFF_F_MIN * 2 ** (c * CUTOFF_OCTAVES)


def g_lut():
    """TPT stage coefficient G = g / (1 + g), g = tan(pi fc / fs_os)."""
    n = 1 << CUTOFF_LUT_BITS
    fs_os = FS * N_OVERSAMPLE
    out = []
    for i in range(n):
        g = math.tan(math.pi * cutoff_hz(i / n) / fs_os)
        out.append(round(g / (1 + g) * ONE))
    return out


def tanh_lut():
    """(value, delta to next entry) for |u| = i * 4 / 1024."""
    n = 1 << TANH_LUT_BITS
    vals = [round(math.tanh(4.0 * i / n) * ONE) for i in range(n + 1)]
    return [(vals[i], vals[i + 1] - vals[i]) for i in range(n)]


G_LUT = g_lut()
TANH_LUT = tanh_lut()


def tanh_q(u):
    a = min(abs(u), (1 << 17) - 1)
    val, delta = TANH_LUT[a >> TANH_FRAC_BITS]
    frac = (a & ((1 << TANH_FRAC_BITS) - 1)) << (15 - TANH_FRAC_BITS)
    t = val + mul(delta, frac)
    return -t if u < 0 else t


def cutoff_index(c):
    if c < 0:
        return 0
    return min(c >> (15 - CUTOFF_LUT_BITS), (1 << CUTOFF_LUT_BITS) - 1)


class LadderModel:
    """N voices. ``voice()`` processes one sample of one voice."""

    def __init__(self, n):
        self.s = [[0, 0, 0, 0] for _ in range(n)]
        self.dc_x = [0] * n
        self.dc_y = [0] * n
        self.gate_env = [0] * n

    def voice(self, v, x, env, *, base, env_amt, cutoff_cv, cutoff_mod, res,
              gate=1, velocity=0, gate_mode=0):
        """x: oscillator sample (ASQ raw), env: velocity x ADSR (UQ(0,16) raw),
        base / env_amt / cutoff_cv / cutoff_mod: ASQ raw, res: 0..32768,
        velocity: UQ(0,16) raw. Returns the output sample (ASQ raw)."""
        env15 = env >> 1

        # exponential cutoff (as Phase 1b)
        cv = cutoff_cv - (cutoff_cv >> 3) - (cutoff_cv >> 4) + (cutoff_cv >> 7)
        c = base + mul(env15, env_amt) + cv + cutoff_mod
        G = G_LUT[cutoff_index(c)]
        beta = ONE - G

        # resonance and Q compensation (global, recomputed per voice)
        k_half = mul(res, K_HALF_SCALE)
        comp = ONE + (mul(res, res) >> 1)
        xc = mul(x >> IN_SHIFT, comp)
        G2 = mul(G, G)
        G4 = mul(G2, G2)
        e = 2 * mul(k_half, G4)
        inv = ONE - e + mul(e, e)

        s = self.s[v]
        acc = 0
        for _ in range(N_OVERSAMPLE):
            S = mul(beta, mul(G, mul(G, mul(G, s[0]) + s[1]) + s[2]) + s[3])
            kS = sat(2 * mul(k_half, S))
            u = mul(sat(xc - kS), inv)
            xi = tanh_q(u)
            for i in range(4):
                vv = mul(G, sat(xi - s[i]))
                y = sat(vv + s[i])
                s[i] = sat(y + vv)
                xi = y
            acc += xi
        out = acc >> OUT_SHIFT              # sum of the 2 oversampled outputs

        # DC block
        self.dc_y[v] = out - self.dc_x[v] + self.dc_y[v] - (self.dc_y[v] >> DC_SHIFT)
        self.dc_x[v] = out

        # Gate-mode level
        target = (velocity >> 1) if gate else 0
        ge = self.gate_env[v]
        if ge + GATE_STEP <= target:
            ge += GATE_STEP
        elif ge >= target + GATE_STEP:
            ge -= GATE_STEP
        else:
            ge = target
        self.gate_env[v] = ge

        vca = ge if gate_mode else env15
        return max(-ONE, min(ONE - 1, mul(sat(self.dc_y[v]), vca)))
