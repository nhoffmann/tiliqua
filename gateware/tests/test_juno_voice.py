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

from tiliqua.dsp import ASQ, mac
LADDER_X_SQ = mac.SQNative
from top.juno.juno_voice import MultiADSR, MultiJunoOsc, MultiLadder, PipelinedMAC, PitchToInc
from top.juno.juno_lfo import JunoLfo, JunoLfoModel
from top.juno.osc_model import OscModel
from top.juno.ladder_model import LadderModel
from top.juno.fx_model import FxModel
from top.juno.adsr_model import AdsrModel, attack_inc_for_time, eps_for_time
from top.juno.juno_fx import JunoFx
from tiliqua.test import stream
from top.juno.sim_render import FS, Note, Params, pw_knob_to_point
from top.juno.sim_render import render as render_amaranth
from top.juno import vl_render

# Render-based tests use Verilator (bit-identical to amaranth.sim, see
# test_verilator_engine_matches_amaranth, ~100x faster); amaranth.sim is the
# fallback when verilator is not installed.
render = vl_render.render_vl if vl_render.available() else render_amaranth


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

    def test_mono_voice_sum(self):
        """Voices are summed to mono (Juno): with chorus off, every voice appears
        identically on both outputs."""
        for voice in (0, 1):
            with self.subTest(voice=voice):
                out = render([Note(voice, 60, 0.0, 1.0)], seconds=0.01)
                self.assertGreater(rms(out[:, 0]), 0.005)
                self.assertEqual(np.abs(out[:, 0] - out[:, 1]).max(), 0.0)

    def test_chorus_makes_stereo(self):
        """Chorus I widens the mono sum: left and right differ, similar level."""
        p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001, chorus=1)
        out = render([Note(0, 60, 0.0, 1.0), Note(1, 64, 0.0, 1.0)], seconds=0.2, params=p)
        L, R = out[FS // 20:, 0], out[FS // 20:, 1]
        self.assertGreater(np.mean(L != R), 0.9)
        self.assertLess(abs(20 * np.log10(rms(L) / rms(R))), 1.0)
        self.assertLess(np.corrcoef(L, R)[0, 1], 0.95)

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

    def test_ladder_matches_model(self):
        """MultiLadder gateware is bit-exact with LadderModel: 3 voices, saw +
        noise inputs, changing envelopes, high resonance, all cutoff inputs,
        Gate mode switched on halfway."""
        n_voices, n_frames = 3, 400
        rng = random.Random(7)
        cfg = dict(env_amt=13107, cutoff_cv=-1200, cutoff_mod=900, res=31130)
        bases = [8000, 14000, 20000]
        velocity = 50000

        def inputs(f, v):
            ph = (f * (0.011 + 0.017 * v)) % 1.0
            x = int((ph * 2 - 1) * 26000) + rng.randrange(-800, 800)
            env = min(65535, f * 400) if f < 200 else max(0, 65535 - (f - 200) * 300)
            gate = int(f < 300)
            gate_mode = int(f >= 200)
            return x, env, gate, gate_mode

        frames = [[inputs(f, v) for v in range(n_voices)] for f in range(n_frames)]
        m = Module()
        m.submodules.dut = dut = MultiLadder(n=n_voices)
        got = []

        async def testbench(ctx):
            ctx.set(dut.reso, cfg["res"])
            ctx.set(dut.env_amt.as_value(), cfg["env_amt"])
            ctx.set(dut.cutoff_cv.as_value(), cfg["cutoff_cv"])
            ctx.set(dut.cutoff_mod.as_value(), cfg["cutoff_mod"])
            for v in range(n_voices):
                ctx.set(dut.voice_cutoff_base[v].as_value(), bases[v])
                ctx.set(dut.voice_velocity[v].as_value(), velocity)
            for fr in frames:
                for v, (x, env, gate, gate_mode) in enumerate(fr):
                    ctx.set(dut.voice_gates[v], gate)
                    ctx.set(dut.vca_gate_mode, gate_mode)
                    await stream.put(ctx, dut.i, {"first": v == 0,
                                                  "sample": {"x": fixed.Const(x / 32768, shape=LADDER_X_SQ),
                                                             "env": fixed.Const(env / 65536, shape=MultiADSR.EnvUQ)}})
                    got.append(round((await stream.get(ctx, dut.o)).sample.as_float() * 32768))

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()

        model = LadderModel(n_voices)
        expected = []
        for fr in frames:
            for v, (x, env, gate, gate_mode) in enumerate(fr):
                expected.append(model.voice(v, x, env, base=bases[v], gate=gate,
                                            velocity=velocity, gate_mode=gate_mode, **cfg))
        mismatches = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        self.assertEqual(mismatches[:5], [], f"got {[got[i] for i in mismatches[:3]]}, "
                         f"expected {[expected[i] for i in mismatches[:3]]}")
        self.assertGreater(np.std(got), 300)

    # --- Ladder behaviour (model is bit-exact with the gateware) --------------

    @staticmethod
    def _ladder(x, c, res):
        model = LadderModel(1)
        return np.array([model.voice(0, int(v), 65535, base=int(c * 32768), env_amt=0,
                                     cutoff_cv=0, cutoff_mod=0, res=int(res * 32768))
                         for v in x]) / 32768

    def _gain_db(self, f, c, res):
        t = np.arange(6000) / FS
        y = self._ladder(0.2 * 32768 * np.sin(2 * np.pi * f * t), c, res)[3000:]
        return 20 * np.log10(np.sqrt(2) * np.std(y) / 0.2 + 1e-12)

    def test_ladder_24db_slope(self):
        """4-pole low-pass: >= 20 dB per octave well above cutoff."""
        fc = MultiLadder.cutoff_hz(0.5)
        self.assertGreater(self._gain_db(2 * fc, 0.5, 0.0) - self._gain_db(4 * fc, 0.5, 0.0), 20)

    def test_ladder_resonance_peak(self):
        """High resonance: the response at cutoff rises above the passband."""
        fc = MultiLadder.cutoff_hz(0.5)
        passband = self._gain_db(fc / 4, 0.5, 0.9)
        self.assertGreater(self._gain_db(fc, 0.5, 0.9), passband + 6)
        self.assertLess(self._gain_db(fc, 0.5, 0.0), self._gain_db(fc / 4, 0.5, 0.0))

    def test_ladder_self_oscillates(self):
        """Max resonance: a full-scale kick starts a sustained oscillation near the
        cutoff, and with a held note the resonance at the cutoff dominates. (Very
        small kicks die out: the input enters at 1/4 for KR-106 drive; noise
        seeding is ladder layer 2.)"""
        x = np.zeros(24000)
        x[0] = (1 << 17) - 1
        y = self._ladder(x, 0.5, 1.0)
        self.assertGreater(np.std(y[-4800:]), 0.02)
        f = estimate_freq(y[-8192:])
        self.assertLess(abs(np.log2(f / MultiLadder.cutoff_hz(0.5))), 0.5)
        osc = OscModel(1)
        inc = PitchToInc.model(48 * 1024)
        saw = [osc.voice(0, inc, saw_en=1, pulse_en=0, pw_point=0, sub_gain=0, noise_gain=0)
               for _ in range(24000)]
        y = self._ladder(saw, 0.5, 1.0)
        spec = np.abs(np.fft.rfft(y[-8192:] * np.hanning(8192)))
        fpk = np.fft.rfftfreq(8192, 1 / FS)[np.argmax(spec)]
        self.assertLess(abs(np.log2(fpk / MultiLadder.cutoff_hz(0.5))), 0.1)

    def test_juno_fx_matches_model(self):
        """JunoFx (HPF + chorus) gateware is bit-exact with FxModel: tones + noise,
        all HPF positions and chorus modes (incl. off) switched during the run."""
        rng = random.Random(3)
        n = 1600
        t = np.arange(n) / FS
        x = (0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 1700 * t)
             + 0.05 * np.array([rng.uniform(-1, 1) for _ in range(n)]))
        xs = [int(v * 32768) for v in x]
        modes = [((i // 200) % 4, (i // 400) % 4) for i in range(n)]   # (hpf, chorus)
        m = Module()
        m.submodules.dut = dut = JunoFx(PipelinedMAC())
        got = []

        async def testbench(ctx):
            for v, (hpf, cho) in zip(xs, modes):
                ctx.set(dut.hpf_mode, hpf)
                ctx.set(dut.chorus_mode, cho)
                await stream.put(ctx, dut.i, fixed.Const(v / 32768, shape=ASQ))
                o = await stream.get(ctx, dut.o)
                got.append((round(o[0].as_float() * 32768), round(o[1].as_float() * 32768)))

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()

        model = FxModel()
        expected = [model.sample(v, hpf, cho) for v, (hpf, cho) in zip(xs, modes)]
        mismatches = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        self.assertEqual(mismatches[:5], [], f"got {[got[i] for i in mismatches[:3]]}, "
                         f"expected {[expected[i] for i in mismatches[:3]]}")
        L, R = np.array(got).T
        self.assertGreater(np.std(L), 1000)
        self.assertGreater(np.mean(L[1200:] != R[1200:]), 0.9)   # chorus on: L != R

    def test_adsr_matches_model(self):
        """MultiADSR (Juno-106 style) gateware is bit-exact with AdsrModel: attack,
        decay to sustain, sustain, release to idle, release during attack,
        retrigger during release; 3 voices with different gates and velocities."""
        n_voices, n_frames = 3, 1500
        cfg = dict(attack_inc=attack_inc_for_time(0.002), decay_eps=eps_for_time(0.004),
                   sustain=26000, release_eps=eps_for_time(0.006))
        def gate(f, v):
            if v == 0:
                return int(f < 600)                       # full cycle
            if v == 1:
                return int(30 <= f < 60 or f >= 700)      # release during attack, later retrigger
            return int(f < 400 or 450 <= f < 900)          # retrigger during release
        vels = [65535, 40000, 20000]
        m = Module()
        m.submodules.dut = dut = MultiADSR(n=n_voices)
        got = []

        async def testbench(ctx):
            ctx.set(dut.attack_rate, cfg["attack_inc"])
            ctx.set(dut.decay_rate, cfg["decay_eps"])
            ctx.set(dut.sustain_level.as_value(), cfg["sustain"])
            ctx.set(dut.release_rate, cfg["release_eps"])
            for v in range(n_voices):
                ctx.set(dut.voice_velocity[v].as_value(), vels[v])
            for f in range(n_frames):
                for v in range(n_voices):
                    ctx.set(dut.voice_gates[v], gate(f, v))
                for v in range(n_voices):
                    got.append(round((await stream.get(ctx, dut.o)).sample.as_float() * 65536))

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()

        model = AdsrModel(n_voices)
        expected = [model.voice(v, gate(f, v), vels[v], cfg["attack_inc"], cfg["decay_eps"],
                                cfg["sustain"], cfg["release_eps"])[0]
                    for f in range(n_frames) for v in range(n_voices)]
        mismatches = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        self.assertEqual(mismatches[:5], [], f"got {[got[i] for i in mismatches[:3]]}, "
                         f"expected {[expected[i] for i in mismatches[:3]]}")
        self.assertGreater(max(got), 60000)

    def test_adsr_exponential_release(self):
        """Release is exponential (straight line in dB) and hits -60 dB at the set time."""
        t_rel = 0.25
        model = AdsrModel(1)
        args = (attack_inc_for_time(0.001), eps_for_time(0.001), 65535, eps_for_time(t_rel))
        y = [model.voice(0, int(i < 2000), 65535, *args)[0] for i in range(2000 + int(0.3 * FS))]
        rel = np.array(y[2000:], dtype=float)
        db = 20 * np.log10(np.maximum(rel, 1e-9) / rel[0])
        t60 = np.argmax(db < -60) / FS
        self.assertAlmostEqual(t60, t_rel, delta=0.01 * t_rel)
        mid = db[: int(0.8 * t_rel * FS)]
        fit = np.polyfit(np.arange(len(mid)), mid, 1)
        self.assertLess(np.max(np.abs(np.polyval(fit, np.arange(len(mid))) - mid)), 0.5)

    def test_juno_lfo_matches_model(self):
        """JunoLfo gateware is bit-exact with JunoLfoModel: fast rate (several
        cycles), delay ramp restarted by a new note after all keys released,
        PWM manual and LFO mode."""
        frames = []
        for f in range(240):
            frames.append(dict(
                rate_inc=(1 << 32) // 37 + 12345,
                delay_step=(1 << 30) // 60,          # 60-frame fade-in
                any_gate=int(not (100 <= f < 110)),  # keys released, then a new note
                dco_depth=2560, vcf_depth=9830,
                pw_knob=24576, pwm_lfo=int(f >= 120)))
        m = Module()
        m.submodules.dut = dut = JunoLfo(PipelinedMAC())
        outs = ["lfo", "vib", "cutoff_mod", "pw_point"]
        got = []

        async def testbench(ctx):
            for fr in frames:
                for k, v in fr.items():
                    ctx.set(getattr(dut, k), v)
                ctx.set(dut.tick, 1)
                await ctx.tick()
                ctx.set(dut.tick, 0)
                for _ in range(40):
                    await ctx.tick()
                got.append({o: ctx.get(getattr(dut, o)) for o in outs})

        sim = Simulator(m)
        sim.add_clock(1 / 60e6)
        sim.add_testbench(testbench)
        sim.run()

        model = JunoLfoModel()
        for i, fr in enumerate(frames):
            exp = model.frame(**fr)
            self.assertEqual(got[i], {o: exp[o] for o in outs}, f"frame {i}")
        lfo = np.array([g["lfo"] for g in got])
        self.assertGreater(lfo.max(), 30000)          # reaches full amplitude
        self.assertLess(lfo.min(), -30000)
        self.assertEqual(lfo[110], 0)                 # delay restarted by the new note (frame 110)
        self.assertNotEqual(lfo[109], 0)

    @unittest.skipUnless(__import__("shutil").which("verilator"), "verilator not installed")
    def test_verilator_engine_matches_amaranth(self):
        """vl_render (Verilator) is bit-identical with the amaranth.sim render:
        staggered notes + note-offs, PWM by LFO, vibrato, filter LFO, Gate mode,
        sub and noise, HPF bass boost, chorus I+II."""
        notes = [Note(0, 48, 0.0, 0.015), Note(1, 55, 0.003, 0.025), Note(2, 64, 0.008, 0.02)]
        p = Params(attack=0.0002, decay=0.0005, sustain=0.46, release=0.0005, pulse=True,
                   pw=0.6, pwm_lfo=True, dco_lfo_semi=1.0, vcf_lfo=0.2,
                   lfo_rate_hz=200.0, vca_gate=True, sub=0.5, noise=0.2,
                   hpf=0, chorus=3)
        a = render_amaranth(notes, 0.025, p)
        b = vl_render.render_vl(notes, 0.025, p)
        self.assertEqual(np.abs(a - b).max(), 0.0)
        self.assertGreater(np.abs(a).max(), 0.05)

    def test_pitch_bend(self):
        """Pitch bend is exact in the log domain: +2 semitones on A4 -> B4."""
        p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001,
                   cutoff=0.85, env_amt=0.0, kbd=0.0, res=0.0, bend_semitones=2.0)
        out = render([Note(0, 69, 0.0, 1.0)], seconds=0.03, params=p)
        f = estimate_freq(out[len(out) // 4:, 0])
        expected = 440.0 * 2 ** (2 / 12)
        self.assertLess(abs(1200 * np.log2(f / expected)), 2.0)

    def test_dco_range(self):
        """DCO range 16' / 4' shifts A4 by exactly one octave down / up."""
        for rng, expected in ((0, 220.0), (2, 880.0)):
            with self.subTest(range=rng):
                p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001,
                           cutoff=0.85, env_amt=0.0, kbd=0.0, res=0.0, dco_range=rng)
                out = render([Note(0, 69, 0.0, 1.0)], seconds=0.04, params=p)
                f = estimate_freq(out[len(out) // 4:, 0])
                self.assertLess(abs(1200 * np.log2(f / expected)), 2.0)

    def test_pitch_full_range(self):
        """Pitch within 2 cents from C1 to C7 (catches coarse bass tuning: the
        16-bit freq_inc of the upstream tracker is -25 cents at C1, +13.5 at C2)."""
        p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001,
                   cutoff=0.85, env_amt=0.0, kbd=0.0, res=0.0)
        for note, seconds in [(24, 0.16), (36, 0.09), (48, 0.05), (84, 0.03), (96, 0.03)]:
            with self.subTest(note=note):
                expected = 440.0 * 2 ** ((note - 69) / 12)
                out = render([Note(0, note, 0.0, 1.0)], seconds=seconds, params=p)
                f = estimate_freq(out[len(out) // 4:, 0])
                cents = 1200 * np.log2(f / expected)
                self.assertLess(abs(cents), 2.0, f"{f:.3f} Hz vs {expected:.3f} Hz")

    # Envelope times in seconds (attack to full, decay / release to -60 dB).
    # Fast settings keep these renders short.

    def test_vca_release_goes_silent(self):
        """After note-off with a fast release, the VCA brings the output to ~0."""
        p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001)
        out = render([Note(0, 60, 0.0, 0.015)], seconds=0.025, params=p)
        held = rms(out[int(0.008 * FS):int(0.015 * FS), 0])
        tail = rms(out[int(0.020 * FS):, 0])
        self.assertGreater(held, 0.01)
        self.assertLess(tail, 0.01 * held)

    # --- Exponential cutoff (knob / env amount / keyboard tracking / CV) ---------
    # Brightness = spectral centroid of a held saw. Attack/decay are fast and
    # sustain is full, so the envelope sits at a constant level while measuring.

    FAST_FULL = dict(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001)

    def _centroid(self, note=60, seconds=0.03, **kw):
        p = Params(**{**self.FAST_FULL, **kw})
        out = render([Note(0, note, 0.0, 1.0)], seconds=seconds, params=p)
        return centroid(out[len(out) // 3:, 0])

    def test_cutoff_lut(self):
        """LUT is monotonic and spans 20 Hz .. ~20 kHz over 10 octaves."""
        lut = MultiLadder.cutoff_lut()
        self.assertTrue(all(b >= a for a, b in zip(lut, lut[1:])))
        self.assertAlmostEqual(MultiLadder.cutoff_hz(0.0), 20.0)
        self.assertAlmostEqual(MultiLadder.cutoff_hz(0.5), 20.0 * 32)
        self.assertGreater(MultiLadder.cutoff_hz(1023 / 1024), 12000)

    def test_cutoff_knob_brightness(self):
        """Higher cutoff knob -> brighter sound."""
        dark   = self._centroid(cutoff=0.3, env_amt=0.0, kbd=0.0)
        bright = self._centroid(cutoff=0.7, env_amt=0.0, kbd=0.0)
        self.assertGreater(bright, 2 * dark)

    def test_env_amount_opens_filter(self):
        """Positive env amount opens the filter, negative closes it."""
        # cutoff 0.55 (~540 Hz): far lower and the closed filter output of a C4
        # rounds to zero in the ladder's 15-bit fixed point
        neutral = self._centroid(cutoff=0.55, env_amt=0.0, kbd=0.0)
        pos     = self._centroid(cutoff=0.55, env_amt=0.3, kbd=0.0)
        neg     = self._centroid(cutoff=0.55, env_amt=-0.15, kbd=0.0)
        self.assertGreater(pos, 1.5 * neutral)
        self.assertLess(neg, neutral / 1.2)

    def test_keyboard_tracking(self):
        """kbd=1: cutoff follows pitch, so brightness scales with the note
        (two octaves up -> ~4x centroid). kbd=0: much less than 4x."""
        ratios = {}
        for kbd in (0.0, 1.0):
            # cutoff 0.3 (~160 Hz; tracking is relative to C1, so kbd=1 adds +2 oct at
            # C3 and +4 oct at C5): both tracked cutoffs stay well below the top of
            # the range, and without tracking the ratio stays far from 4 (1.27)
            lo = self._centroid(note=48, cutoff=0.3, env_amt=0.0, kbd=kbd)
            hi = self._centroid(note=72, cutoff=0.3, env_amt=0.0, kbd=kbd)
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
        """PW knob 0 -> exact square, knob 1 -> 26.4 % duty (pulse high 73.6 % of
        the cycle), linear in between: measured on the KR-106."""
        for knob, duty in [(0.0, 0.5), (0.5, 0.618), (1.0, 0.736)]:
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

    def test_vca_gate_mode(self):
        """Gate mode: with sustain 0 the envelope dies away, but the VCA holds
        the note at full level while the key is held, and cuts it quickly on
        release (the filter still follows the envelope)."""
        p = Params(attack=0.0002, decay=0.001, sustain=0, release=0.001,
                   cutoff=0.8, env_amt=0.0, kbd=0.0, vca_gate=True)
        out = render([Note(0, 60, 0.0, 0.012)], seconds=0.018, params=p)
        early = rms(out[int(0.002 * FS):int(0.004 * FS), 0])
        late_held = rms(out[int(0.009 * FS):int(0.012 * FS), 0])
        after = rms(out[int(0.0145 * FS):, 0])
        self.assertGreater(late_held, 0.8 * early)    # env mode would be ~0 here
        self.assertLess(after, 0.01 * early)

    def test_lfo_vibrato(self):
        """LFO -> pitch: 1 semitone depth at 40 Hz swings the period by about
        +-1 semitone; without vibrato the swing is ~0."""
        def swing_semitones(depth):
            p = Params(attack=0.0002, decay=0.0002, sustain=1.0, release=0.001,
                       cutoff=0.9, env_amt=0.0, kbd=0.0, res=0.0, dco_lfo_semi=depth,
                       lfo_rate_hz=40.0)
            out = render([Note(0, 81, 0.0, 1.0)], seconds=0.06, params=p)
            x = out[len(out) // 6:, 0]
            x = x - x.mean()
            # rising zero crossings with linear interpolation
            idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
            zc = idx + x[idx] / (x[idx] - x[idx + 1])
            periods = np.diff(zc)
            return 12 * np.log2(periods.max() / periods.min()) / 2
        self.assertLess(swing_semitones(0.0), 0.15)
        self.assertAlmostEqual(swing_semitones(1.0), 1.0, delta=0.3)

    def test_vca_zero_sustain_decays_while_held(self):
        """Sustain 0: the note dies away although the gate is still held (pluck)."""
        p = Params(attack=0.0002, decay=0.004, sustain=0, release=0.001)
        out = render([Note(0, 60, 0.0, 1.0)], seconds=0.015, params=p)
        early = rms(out[:int(0.002 * FS), 0])
        late = rms(out[int(0.010 * FS):, 0])
        self.assertGreater(early, 0.005)
        self.assertLess(late, 0.01 * early)


if __name__ == "__main__":
    unittest.main()
