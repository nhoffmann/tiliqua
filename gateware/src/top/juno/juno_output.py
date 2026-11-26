# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Juno output stage (Phase 4.7): level, optional drive, soft limiter.

Per channel (``top.py``, both on the shared ring MAC)::

    core (ASQ, +-1) --x4--> pre VCA (x pre_gain, clip +-1) --> knee LUT --> post VCA (x post_gain)

- ``knee(x)``: clean up to ``KNEE`` of full scale (~5 V with 1.0 = 8.192 V), then a
  tanh shoulder that is flat (0.9996) at the clipped input, so the pre-VCA clip is
  inaudible. Slope 2 in the linear part (the LUT input is half the virtual range).
- ``pre_gain`` = level gain x drive gain (+ drive CV in gateware), SQ(3,15), <= 4.
- ``post_gain`` = 1 / sqrt(drive gain): with drive the output gets denser, not much
  louder. Drive off = 1.0 (clean Juno).

Small-signal gain core -> out = 4 * pre_gain * 2 * post_gain.

Firmware (``fw/src/out.rs``, same formulas):

- level knob k: ``level_gain = 4 k^2`` (default k = 0.45 -> 0.81: a default
  6-voice chord peaks at the knee, a single note at ~1.6 V).
- drive knob d: ``drive_gain = 1 + 7 d^2`` (0 = off .. x8 = +18 dB into the knee;
  ``pre_gain`` is clamped to < 4, so at high level less drive is available).
"""

import math

KNEE = 0.6
LEVEL_DEFAULT = 0.45
DRIVE_MAX = 8.0
PRE_MAX = 4.0 - 2 ** -15


def knee(x):
    """LUT function on the ASQ domain (-1..1) -> -1..1."""
    u = 2.0 * abs(x)
    if u <= KNEE:
        y = u
    else:
        y = KNEE + (1 - KNEE) * math.tanh((u - KNEE) / (1 - KNEE))
    return math.copysign(min(y, 1.0 - 2 ** -15), x)


def level_gain(k):
    k = min(max(k, 0.0), 1.0)
    return 4.0 * k * k


def drive_gain(d):
    d = min(max(d, 0.0), 1.0)
    return 1.0 + (DRIVE_MAX - 1.0) * d * d


def registers(level_knob, drive_knob):
    """Knobs 0..1 -> (drive register raw, out_gain register raw).

    ``drive`` register: pre_gain * 8192 (gateware: ``drive << 2`` = SQ(3,15) raw).
    ``out_gain`` register: post_gain as UQ(0,16) raw (65535 = 1.0)."""
    g = drive_gain(drive_knob)
    pre = min(level_gain(level_knob) * g, PRE_MAX)
    return min(32767, round(pre * 8192)), min(65535, round(65536 / math.sqrt(g)))


def output(core, level_knob=LEVEL_DEFAULT, drive_knob=0.0):
    """Float model of the stage for level tables / listening (numpy array in, out)."""
    import numpy as np
    pre_raw, post_raw = registers(level_knob, drive_knob)
    x = np.clip(4 * core * (pre_raw / 8192), -1, 1)
    return np.vectorize(knee)(x) * (post_raw / 65536)
