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

from top.juno.juno_voice import MultiSVF
from top.juno.sim_render import FS, Note, Params, render


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def centroid(x):
    """Power-weighted spectral centroid in Hz (Hann window). Power weighting keeps
    the fixed-point noise floor (~-80 dB) from dominating when the signal is quiet."""
    x = (x - np.mean(x)) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return float(np.sum(freqs * power) / np.sum(power))


def estimate_freq(x):
    """Fundamental via autocorrelation peak with parabolic interpolation."""
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    # skip the zero-lag lobe: start search after the first negative value
    start = int(np.argmax(ac < 0))
    window = ac[start:len(x) // 2]
    # first local maximum that is close to the strongest one (avoids picking
    # 2x the period when a near-sinusoid has almost equal peaks at T and 2T)
    threshold = 0.9 * window.max()
    peaks = [i for i in range(1, len(window) - 1)
             if window[i] >= window[i - 1] and window[i] >= window[i + 1]
             and window[i] >= threshold]
    lag = start + peaks[0]
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

    # --- Exponential cutoff (knob / env amount / keyboard tracking / CV) ---------
    # Brightness = spectral centroid of a held saw. Attack/decay are fast and
    # sustain is full, so the envelope sits at a constant level while measuring.

    FAST_FULL = dict(attack=8192, decay=8192, sustain=65535, release=2048)

    def _centroid(self, note=60, seconds=0.03, **kw):
        p = Params(**{**self.FAST_FULL, **kw})
        out = render([Note(0, note, 0.0, 1.0)], seconds=seconds, params=p)
        return centroid(out[len(out) // 3:, 0])

    def test_cutoff_lut(self):
        """LUT is monotonic and spans 12 Hz .. ~12 kHz over 10 octaves."""
        lut = MultiSVF.cutoff_lut()
        self.assertTrue(all(b >= a for a, b in zip(lut, lut[1:])))
        self.assertAlmostEqual(MultiSVF.cutoff_hz(0.0), 12.0)
        self.assertAlmostEqual(MultiSVF.cutoff_hz(0.5), 12.0 * 32)
        self.assertGreater(MultiSVF.cutoff_hz(1023 / 1024), 12000)

    def test_cutoff_knob_brightness(self):
        """Higher cutoff knob -> brighter sound."""
        dark   = self._centroid(cutoff=0.3, env_amt=0.0, kbd=0.0)
        bright = self._centroid(cutoff=0.7, env_amt=0.0, kbd=0.0)
        self.assertGreater(bright, 2 * dark)

    def test_env_amount_opens_filter(self):
        """Positive env amount opens the filter, negative closes it."""
        neutral = self._centroid(cutoff=0.4, env_amt=0.0, kbd=0.0)
        pos     = self._centroid(cutoff=0.4, env_amt=0.3, kbd=0.0)
        neg     = self._centroid(cutoff=0.4, env_amt=-0.15, kbd=0.0)
        self.assertGreater(pos, 1.5 * neutral)
        self.assertLess(neg, neutral / 1.2)

    def test_keyboard_tracking(self):
        """kbd=1: cutoff follows pitch, so brightness scales with the note
        (two octaves up -> ~4x centroid). kbd=0: much less than 4x."""
        ratios = {}
        for kbd in (0.0, 1.0):
            lo = self._centroid(note=48, cutoff=0.45, env_amt=0.0, kbd=kbd)
            hi = self._centroid(note=72, cutoff=0.45, env_amt=0.0, kbd=kbd)
            ratios[kbd] = hi / lo
        self.assertAlmostEqual(ratios[1.0], 4.0, delta=0.6)
        self.assertLess(ratios[0.0], 3.0)

    def test_cutoff_cv_one_volt_per_octave(self):
        """+1 V on the cutoff CV equals raising the knob by one octave (0.1)."""
        via_cv   = self._centroid(cutoff=0.4, env_amt=0.0, kbd=0.0, cutoff_cv_volts=1.0)
        via_knob = self._centroid(cutoff=0.5, env_amt=0.0, kbd=0.0)
        self.assertAlmostEqual(via_cv / via_knob, 1.0, delta=0.03)

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
