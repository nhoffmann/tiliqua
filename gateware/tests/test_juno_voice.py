# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Tests for the Juno voice engine (``top/juno/juno_voice.py``), driven through
the offline renderer in ``top/juno/sim_render.py``.

Simulation is slow (~3 min per second of audio), so each test renders at most
a few thousand samples. Longer renders for listening:
``pdm run python -m top.juno.sim_render``.
"""

import unittest

import random

import numpy as np
from amaranth import Module
from amaranth.sim import Simulator

from amaranth_future import fixed

from top.juno.juno_voice import MultiJunoOsc, MultiSVF, PitchToInc
from top.juno.osc_model import OscModel
from tiliqua.test import stream
from top.juno.sim_render import FS, Note, Params, pw_knob_to_point, render


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

    def test_pitch_to_inc_matches_model(self):
        """PitchToInc gateware is bit-exact with its Python model (incl. clamping),
        and the model is within 0.01 cents of the ideal increment."""
        rng = random.Random(1)
        pitches = [0, 1023, 69 * 1024, 127 * 1024 + 1023, -5000, 140 * 1024]
        pitches += [rng.randrange(-2048, 130 * 1024) for _ in range(40)]
        m = Module()
        m.submodules.dut = dut = PitchToInc()
        results = []

        async def testbench(ctx):
            for raw in pitches:
                ctx.set(dut.pitch.as_value(), raw)
                for _ in range(PitchToInc.LATENCY):
                    await ctx.tick()
                results.append(ctx.get(dut.inc))

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()
        for raw, got in zip(pitches, results):
            self.assertEqual(got, PitchToInc.model(raw), f"pitch raw {raw}")
            if 0 <= raw < 128 * 1024:
                cents = 1200 * np.log2(got / PitchToInc.exact_inc(raw / 1024))
                self.assertLess(abs(cents), 0.01, f"pitch raw {raw}")

    def test_juno_osc_matches_model(self):
        """MultiJunoOsc gateware is bit-exact with osc_model.OscModel: 3 voices at
        low / mid / very high pitch (PolyBLEP regions overlap a lot at the top),
        all mixer sources on, pulse width away from square."""
        n_voices = 3
        pitches = [36.0, 61.37, 115.0]
        cfg = dict(saw_en=1, pulse_en=1, pw_point=int(0.7 * 65536),
                   sub_gain=int(0.67 * 65535), noise_gain=int(0.3 * 65535))
        n_frames = 400
        m = Module()
        m.submodules.dut = dut = MultiJunoOsc(n=n_voices)
        got = []

        async def testbench(ctx):
            for k, v in cfg.items():
                ctx.set(getattr(dut, k), v)
            for v, ptch in enumerate(pitches):
                ctx.set(dut.voice_pitch[v], fixed.Const(ptch, shape=PitchToInc.PitchSQ))
            for _ in range(n_frames * n_voices):
                got.append(round((await stream.get(ctx, dut.o)).sample.as_float() * 32768))

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()

        model = OscModel(n_voices)
        incs = [PitchToInc.model(round(ptch * 1024)) for ptch in pitches]
        expected = [model.voice(v, incs[v], **cfg)
                    for _ in range(n_frames) for v in range(n_voices)]
        mismatches = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        self.assertEqual(mismatches[:5], [], f"first mismatch at sample {mismatches[:1]}: "
                         f"got {[got[i] for i in mismatches[:3]]}, "
                         f"expected {[expected[i] for i in mismatches[:3]]}")
        self.assertGreater(np.std(got), 1000)   # not silent

    def test_pitch_bend(self):
        """Pitch bend is exact in the log domain: +2 semitones on A4 -> B4."""
        p = Params(attack=8192, decay=8192, sustain=65535, release=2048,
                   cutoff=0.85, env_amt=0.0, kbd=0.0, bend_semitones=2.0)
        out = render([Note(0, 69, 0.0, 1.0)], seconds=0.03, params=p)
        f = estimate_freq(out[len(out) // 4:, 0])
        expected = 440.0 * 2 ** (2 / 12)
        self.assertLess(abs(1200 * np.log2(f / expected)), 2.0)

    def test_pitch_full_range(self):
        """Pitch within 2 cents from C1 to C7 (catches coarse bass tuning: the
        16-bit freq_inc of the upstream tracker is -25 cents at C1, +13.5 at C2)."""
        p = Params(attack=8192, decay=8192, sustain=65535, release=2048,
                   cutoff=0.85, env_amt=0.0, kbd=0.0)
        for note, seconds in [(24, 0.16), (36, 0.09), (48, 0.05), (84, 0.03), (96, 0.03)]:
            with self.subTest(note=note):
                expected = 440.0 * 2 ** ((note - 69) / 12)
                out = render([Note(0, note, 0.0, 1.0)], seconds=seconds, params=p)
                f = estimate_freq(out[len(out) // 4:, 0])
                cents = 1200 * np.log2(f / expected)
                self.assertLess(abs(cents), 2.0, f"{f:.3f} Hz vs {expected:.3f} Hz")

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

    # --- Juno DCO (MultiJunoOsc; the model is bit-exact with the gateware) ---

    @staticmethod
    def _osc(note, n=16384, **cfg):
        base = dict(saw_en=0, pulse_en=0, pw_point=0, sub_gain=0, noise_gain=0)
        model = OscModel(1)
        inc = PitchToInc.model(note * 1024)
        return np.array([model.voice(0, inc, **{**base, **cfg}) for _ in range(n)]) / 32768

    def test_pw_knob_mapping(self):
        """PW knob 0 -> ~54 % duty (near square), knob 1 -> ~4.5 % (as in Phase 1b)."""
        for knob, duty in [(0.0, 0.537), (1.0, 0.045)]:
            x = self._osc(60, pulse_en=1, pw_point=pw_knob_to_point(knob))
            self.assertAlmostEqual(np.mean(x > 0), duty, delta=0.01)

    def test_polyblep_reduces_aliasing(self):
        """C7 saw: PolyBLEP cuts non-harmonic (aliased) energy by > 10 dB vs naive."""
        note = 96
        f0 = 440.0 * 2 ** ((note - 69) / 12)
        def alias_db(x):
            x = (x - x.mean()) * np.blackman(len(x))
            P = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / FS)
            h = np.zeros_like(P, bool)
            for k in range(1, int(FS / 2 / f0) + 1):
                h |= np.abs(f - k * f0) < f0 * 0.03
            return 10 * np.log10(P[~h].sum() / P[h].sum())
        blep = self._osc(note, saw_en=1)
        inc, ph, naive = PitchToInc.model(note * 1024), 0, []
        for _ in range(len(blep)):
            ph = (ph + inc) & 0xFFFFFFFF
            naive.append(ph / 2 ** 31 - 1.0)
        self.assertLess(alias_db(blep), alias_db(np.array(naive)) - 10)

    def test_sub_is_one_octave_down(self):
        x = self._osc(48, n=32768, sub_gain=65535)
        f = estimate_freq(x)
        expected = 440.0 * 2 ** ((48 - 69) / 12) / 2
        self.assertLess(abs(1200 * np.log2(f / expected)), 2.0)

    def test_noise_is_aperiodic(self):
        """Noise: no autocorrelation peak (unlike a pitched source)."""
        x = self._osc(60, n=8192, noise_gain=65535)
        x = x - x.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        self.assertLess(np.max(np.abs(ac[10:2000])) / ac[0], 0.1)

    def test_dco_square_has_weak_even_harmonics(self):
        """Rendered through the voice: the near-square pulse has a much weaker
        2nd harmonic than the saw."""
        def h2_over_h1(**dco):
            p = Params(**self.FAST_FULL, cutoff=0.95, env_amt=0.0, kbd=0.0, **dco)
            out = render([Note(0, 69, 0.0, 1.0)], seconds=0.04, params=p)
            x = out[len(out) // 3:, 0]
            x = (x - x.mean()) * np.hanning(len(x))
            mag = np.abs(np.fft.rfft(x))
            f = np.fft.rfftfreq(len(x), 1 / FS)
            h = lambda k: mag[np.argmin(np.abs(f - 440.0 * k))]
            return h(2) / h(1)
        saw = h2_over_h1(saw=True, pulse=False)
        square = h2_over_h1(saw=False, pulse=True, pw=0.0)
        self.assertGreater(saw, 0.3)
        self.assertLess(square, saw / 3)

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
