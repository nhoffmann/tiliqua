# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Tests for the Juno voice engine (``top/juno/juno_voice.py``), driven through
the offline renderer in ``top/juno/sim_render.py``.

Simulation is slow (~3 min per second of audio), so each test renders at most
a few thousand samples. Longer renders for listening:
``pdm run python -m top.juno.sim_render``.
"""

import unittest

import numpy as np

from top.juno.sim_render import FS, Note, Params, render


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def estimate_freq(x):
    """Fundamental via autocorrelation peak with parabolic interpolation."""
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    # skip the zero-lag lobe: start search after the first negative value
    start = int(np.argmax(ac < 0))
    lag = start + int(np.argmax(ac[start:len(x) // 2]))
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    lag_f = lag + 0.5 * (a - c) / (a - 2 * b + c)
    return FS / lag_f


class JunoVoiceTests(unittest.TestCase):

    def test_idle_is_silent(self):
        out = render([], seconds=200 / FS)
        self.assertEqual(np.abs(out).max(), 0.0)

    def test_voice_panning(self):
        """Even voices go to the left channel, odd voices to the right."""
        for voice, active, silent in [(0, 0, 1), (1, 1, 0)]:
            with self.subTest(voice=voice):
                out = render([Note(voice, 60, 0.0, 1.0)], seconds=0.01)
                self.assertGreater(rms(out[:, active]), 0.01)
                self.assertEqual(np.abs(out[:, silent]).max(), 0.0)

    def test_pitch(self):
        """A held A5 plays at 880 Hz (within 1 %)."""
        out = render([Note(0, 81, 0.0, 1.0)], seconds=0.04)
        # skip the attack / filter settling at the start
        f = estimate_freq(out[len(out) // 4:, 0])
        self.assertAlmostEqual(f, 880.0, delta=8.8)


if __name__ == "__main__":
    unittest.main()
