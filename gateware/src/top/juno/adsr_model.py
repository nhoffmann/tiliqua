# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Bit-exact Python model of ``JunoADSR`` (``juno_voice.py``): Juno-106-style envelope.

Behaviour reference: KR-106 ``docs/DSP_ARCHITECTURE.md`` ("Juno-106 mode").

- Level ``L``: 32 bit per voice (UQ(0,32), ``L_MAX`` = full scale).
- Attack: linear, ``L += attack_inc`` (32 bit) until full scale.
- Decay: exponential toward 0, ``L -= ((L >> 15) * decay_eps) >> 4``, stops at the
  sustain level (``sustain << 16``).
- Sustain: holds the sustain level.
- Release: exponential, ``L -= ((L >> 15) * release_eps) >> 4``; below ``L_SILENT``
  (about -96 dB) the voice goes idle.
- ``eps`` is the per-sample decay fraction with 19 fractional bits: for a time ``t``
  to fall by 60 dB, ``eps = 3 ln(10) / (t * fs) * 2**19`` (1 ms .. > 10 s fit in 17 bit).
- Output: ``((L >> 16) * velocity) >> 16`` (UQ(0,16)), as the upstream ADSR.

Not modelled (KR-106 has them): attack overshoot, sustain ramping on knob changes,
per-voice timing variance (Phase 4.6).
"""

# Knob (= Juno-106 slider / 127) -> envelope times in seconds, measured on the
# KR-106 (tools/kr106_compare.py --envelope, C7, 2 ms RMS windows): attack = time
# to 95 %, decay / release = time to -60 dB. Points at slider 0, 16, ..., 112, 127,
# interpolated in log time. Attack 0 (too fast to measure) set to 1.5 ms; decay /
# release at 127 measured ~91 / ~96 s (extrapolated) clamped to the 75 s longest
# time the 17-bit eps can represent. Same tables in firmware fw/src/env.rs.
ENV_SLIDERS = [0, 16, 32, 48, 64, 80, 96, 112, 127]
ATTACK_S  = [0.0016, 0.1347, 0.2632, 0.3979, 0.5347, 0.8168, 1.3263, 2.1537, 3.2063]   # time to FULL level (measured time to 95 % / 0.95)
DECAY_S   = [0.0087, 0.3406, 0.7812, 3.5125, 5.7882, 8.1155, 11.7507, 22.6675, 75.0]
RELEASE_S = [0.0120, 0.3553, 0.7442, 3.0000, 4.7017, 6.6780, 11.1710, 24.0906, 75.0]


def knob_to_time(knob, table):
    """Knob 0..1 -> seconds, log-interpolated between the slider points."""
    import math
    sl = min(max(knob, 0.0), 1.0) * 127
    for i in range(len(ENV_SLIDERS) - 1):
        if sl <= ENV_SLIDERS[i + 1] or i == len(ENV_SLIDERS) - 2:
            f = (sl - ENV_SLIDERS[i]) / (ENV_SLIDERS[i + 1] - ENV_SLIDERS[i])
            return math.exp(math.log(table[i]) + f * (math.log(table[i + 1]) - math.log(table[i])))


L_MAX = (1 << 32) - 1
L_SILENT = 1 << 16            # ~ -96 dB
EPS_FRAC = 19

IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = range(5)


def eps_for_time(seconds, fs=48000):
    """Per-sample decay fraction (19 fractional bits) for -60 dB after ``seconds``."""
    import math
    return max(1, min((1 << 17) - 1, round(3 * math.log(10) / (seconds * fs) * (1 << EPS_FRAC))))


def attack_inc_for_time(seconds, fs=48000):
    return max(1, min(L_MAX, round(L_MAX / (seconds * fs))))


def decrement(L, eps):
    return ((L >> 15) * eps) >> (EPS_FRAC - 15)


class AdsrModel:

    def __init__(self, n):
        self.phase = [IDLE] * n
        self.level = [0] * n
        self.last_gate = [0] * n

    def voice(self, v, gate, velocity, attack_inc, decay_eps, sustain, release_eps):
        """One sample of voice ``v``. ``velocity`` / ``sustain`` UQ(0,16) raw.
        Returns (output UQ(0,16) raw, active)."""
        phase, L = self.phase[v], self.level[v]
        if gate and not self.last_gate[v]:
            phase = ATTACK
        elif not gate and self.last_gate[v]:
            phase = RELEASE
        self.last_gate[v] = gate
        sus = sustain << 16

        if phase == IDLE:
            L = 0
        elif phase == ATTACK:
            L += attack_inc
            if L >= L_MAX:
                L, phase = L_MAX, DECAY
        elif phase == DECAY:
            L -= decrement(L, decay_eps)
            if L <= sus:
                L, phase = sus, SUSTAIN
        elif phase == SUSTAIN:
            L = sus
        elif phase == RELEASE:
            L -= decrement(L, release_eps)
            if L < L_SILENT:
                L, phase = 0, IDLE

        self.phase[v], self.level[v] = phase, L
        out = ((L >> 16) * velocity) >> 16
        return out, int(phase != IDLE)
