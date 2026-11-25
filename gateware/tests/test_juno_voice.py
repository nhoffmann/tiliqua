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

    # Envelope rates are per-sample increments of a 16-bit level: 2048 ≈ 16 samples
    # for a full-scale ramp. Fast settings keep these renders short.

    def test_vca_release_goes_silent(self):
        """After note-off with a fast release, the VCA brings the output to ~0."""
        p = Params(attack=8192, decay=8192, sustain=65535, release=2048)
        out = render([Note(0, 60, 0.0, 0.015)], seconds=0.025, params=p)
        held = rms(out[int(0.008 * FS):int(0.015 * FS), 0])
        tail = rms(out[int(0.020 * FS):, 0])
        self.assertGreater(held, 0.01)
        self.assertLess(tail, 0.01 * held)

    def test_vca_zero_sustain_decays_while_held(self):
        """Sustain 0: the note dies away although the gate is still held (pluck)."""
        p = Params(attack=8192, decay=2048, sustain=0, release=2048)
        out = render([Note(0, 60, 0.0, 1.0)], seconds=0.015, params=p)
        early = rms(out[:int(0.002 * FS), 0])
        late = rms(out[int(0.010 * FS):, 0])
        self.assertGreater(early, 0.005)
        self.assertLess(late, 0.01 * early)


if __name__ == "__main__":
    unittest.main()
