# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Offline renderer for the Juno voice engine (``JunoVoice``) using ``amaranth.sim``.

Drives the voice block with a note schedule and collects stereo output samples,
so changes can be listened to (WAV) and checked numerically without hardware.

Sample rate is nominal 48 kHz: the hardware pulls one stereo sample per audio
frame, here the testbench pulls them as fast as the simulated block produces
them. Pitch is sent in semitones (note + bend), like the juno top level.

Usage (writes WAVs to ``build/juno_sim/``)::

    pdm run python -m top.juno.sim_render [--seconds 0.5] [scene ...]
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from amaranth import Module
from amaranth.sim import Simulator

from amaranth_future import fixed
from tiliqua.dsp import ASQ
from tiliqua.test import stream

from top.juno.juno_voice import JunoVoice, MultiADSR, MultiSVF, PitchToInc

FS = 48000
N_VOICES = 6
EnvUQ = MultiADSR.EnvUQ


def adsr_ui_to_rate(ui_value):
    """Same mapping as the firmware (main.rs): UI 0..32768 -> 1 ms..2000 ms."""
    ms_x32k = 32768 + ui_value * 1999
    return min(45_000_000 // ms_x32k, 65535)


def cutoff_base(cutoff, kbd, note, mod_wheel=0.0):
    """Per-voice cutoff base, same formula as the firmware (normalized octaves):
    knob + keyboard tracking relative to C4 (note 60) + mod wheel offset."""
    octave = 1.0 / MultiSVF.CUTOFF_OCTAVES
    return cutoff + kbd * (note - 60) / 12 * octave + mod_wheel * MOD_WHEEL_RANGE


# Mod wheel fully up opens the cutoff by this much (normalized octaves).
MOD_WHEEL_RANGE = 0.5


def pw_knob_to_point(knob):
    """PW knob 0..1 -> pulse rising-edge phase, UQ(0,16). Same as the firmware.
    Juno: comparator at threshold 0.5 + 0.46 * knob on the curved ramp
    p (1 + 0.15 (1 - p)); solved for the linear phase p (duty ~54 % .. ~4.5 %)."""
    th = 0.5 + 0.46 * knob
    p = (1.15 - math.sqrt(1.15 ** 2 - 0.6 * th)) / 0.3
    return min(65535, round(p * 65536))


@dataclass
class Note:
    voice: int
    note: int
    on_s: float
    off_s: float
    velocity: float = 0.99


@dataclass
class Params:
    """Global synth params in hardware units. Defaults = firmware option defaults."""
    attack:  int = adsr_ui_to_rate(0)
    decay:   int = adsr_ui_to_rate(4096)
    sustain: int = 16384 * 65535 // 32768
    release: int = adsr_ui_to_rate(4096)
    reso:    int = 32768 - 16384 * 24576 // 32768
    # Exponential cutoff (normalized octaves, see MultiSVF); firmware defaults
    cutoff:  float = 0.35
    env_amt: float = 0.6
    kbd:     float = 0.5
    mod_wheel: float = 0.0
    bend_semitones: float = 0.0    # pitch bend applied to all notes (top.py: +-2 range)
    cutoff_cv_volts: float = 0.0   # CV in1, 1 V/oct
    # Juno DCO mixer (firmware defaults: saw on, rest off)
    saw:   bool = True
    pulse: bool = False
    pw:    float = 0.0     # PW knob 0..1
    sub:   float = 0.0     # sub level 0..1
    noise: float = 0.0     # noise level 0..1


def render(notes, seconds, params=None, n_voices=N_VOICES):
    """Render ``seconds`` of audio. Returns float array of shape (n_samples, 2)."""
    params = params or Params()
    n_samples = int(seconds * FS)
    out = np.zeros((n_samples, 2))

    m = Module()
    m.submodules.dut = dut = JunoVoice(n_voices=n_voices)

    # Gate schedule: per sample index, the state of every voice.
    events = []
    for n in notes:
        events.append((int(n.on_s * FS), n, 1))
        events.append((int(n.off_s * FS), n, 0))
    events.sort(key=lambda e: (e[0], e[2]))

    async def testbench(ctx):
        ctx.set(dut.saw_en, int(params.saw))
        ctx.set(dut.pulse_en, int(params.pulse))
        ctx.set(dut.pw_point, pw_knob_to_point(params.pw))
        ctx.set(dut.sub_gain, round(0.67 * params.sub * 65535))
        ctx.set(dut.noise_gain, round(params.noise * 65535))

        ctx.set(dut.attack_rate.as_value(), params.attack)
        ctx.set(dut.decay_rate.as_value(), params.decay)
        ctx.set(dut.sustain_level.as_value(), params.sustain)
        ctx.set(dut.release_rate.as_value(), params.release)
        ctx.set(dut.reso, params.reso)
        ctx.set(dut.env_amt, fixed.Const(params.env_amt, shape=ASQ))
        ctx.set(dut.cutoff_cv, fixed.Const(params.cutoff_cv_volts / 8.192, shape=ASQ))
        await ctx.tick()

        ev = 0
        for i in range(n_samples):
            while ev < len(events) and events[ev][0] <= i:
                _, note, gate = events[ev]
                if gate:
                    ctx.set(dut.voice_pitch[note.voice],
                            fixed.Const(note.note + params.bend_semitones,
                                        shape=PitchToInc.PitchSQ))
                    ctx.set(dut.voice_velocity[note.voice],
                            fixed.Const(note.velocity, shape=EnvUQ))
                    base = cutoff_base(params.cutoff, params.kbd, note.note, params.mod_wheel)
                    ctx.set(dut.voice_cutoff_base[note.voice],
                            fixed.Const(max(-1.0, min(base, 0.99997)), shape=ASQ))
                ctx.set(dut.voice_gates[note.voice], gate)
                ev += 1
            sample = await stream.get(ctx, dut.o)
            out[i, 0] = sample[0].as_float()
            out[i, 1] = sample[1].as_float()

    sim = Simulator(m)
    sim.add_clock(1 / 60e6)
    sim.add_testbench(testbench)
    sim.run()
    return out


def write_wav(path, audio):
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, FS, audio.astype(np.float32))


SCENES = {
    # name: (seconds -> notes, params)
    # single held note, then release (firmware default sound)
    "single_c3": (lambda s: [Note(0, 48, 0.0, s * 0.6)], Params()),
    # C minor chord on 3 voices (2 left, 1 right)
    "chord_cmin": (lambda s: [Note(0, 48, 0.0, s * 0.6), Note(1, 51, 0.0, s * 0.6),
                              Note(2, 55, 0.0, s * 0.6)], Params()),
    # short staccato notes, one per voice
    "staccato": (lambda s: [Note(v, 48 + 2 * v, v * s / 6, v * s / 6 + s / 12)
                            for v in range(N_VOICES)], Params()),
    # plucky bass: dark filter, strong env amount, zero sustain
    "pluck_bass": (lambda s: [Note(v % 2, n, i * s / 4, i * s / 4 + s / 5)
                              for i, (v, n) in enumerate(zip(range(4), [36, 36, 43, 39]))],
                   Params(cutoff=0.2, env_amt=0.7, kbd=0.3, sustain=0,
                          decay=adsr_ui_to_rate(2500), release=adsr_ui_to_rate(1500))),
    # slow attack brassy swell: filter opens with the envelope
    "brass_swell": (lambda s: [Note(0, 48, 0.0, s * 0.7), Note(1, 55, 0.0, s * 0.7),
                               Note(2, 60, 0.0, s * 0.7)],
                    Params(cutoff=0.3, env_amt=0.55, kbd=0.5,
                           attack=adsr_ui_to_rate(4000), decay=adsr_ui_to_rate(6000),
                           sustain=45000, release=adsr_ui_to_rate(3000))),
    # Juno DCO mixer: same phrase with saw, square, narrow pulse, saw+pulse
    "dco_saw":    (lambda s: [Note(0, 48, 0.0, s * 0.8)], Params(cutoff=0.6, env_amt=0.2)),
    "dco_square": (lambda s: [Note(0, 48, 0.0, s * 0.8)],
                   Params(cutoff=0.6, env_amt=0.2, saw=False, pulse=True, pw=0.0)),
    "dco_narrow": (lambda s: [Note(0, 48, 0.0, s * 0.8)],
                   Params(cutoff=0.6, env_amt=0.2, saw=False, pulse=True, pw=0.8)),
    "dco_saw_pulse": (lambda s: [Note(0, 48, 0.0, s * 0.8)],
                      Params(cutoff=0.6, env_amt=0.2, saw=True, pulse=True, pw=0.5)),
    # sub oscillator and noise
    "dco_saw_sub": (lambda s: [Note(0, 36, 0.0, s * 0.8)],
                    Params(cutoff=0.5, env_amt=0.3, saw=True, sub=1.0)),
    "dco_noise":   (lambda s: [Note(0, 60, 0.0, s * 0.8)],
                    Params(cutoff=0.8, env_amt=0.0, saw=False, noise=1.0)),
    # high note: PolyBLEP keeps aliasing down
    "dco_high_saw": (lambda s: [Note(0, 96, 0.0, s * 0.8)],
                     Params(cutoff=1.0, env_amt=0.0, kbd=0.0, saw=True)),
    # same chord, dark vs bright cutoff knob (no env), back to back
    "cutoff_dark": (lambda s: [Note(0, 48, 0.0, s * 0.8), Note(1, 55, 0.0, s * 0.8)],
                    Params(cutoff=0.4, env_amt=0.0, kbd=0.5)),
    "cutoff_bright": (lambda s: [Note(0, 48, 0.0, s * 0.8), Note(1, 55, 0.0, s * 0.8)],
                      Params(cutoff=0.75, env_amt=0.0, kbd=0.5)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--out", default="build/juno_sim")
    parser.add_argument("scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()
    for name in args.scenes:
        t0 = time.time()
        notes, params = SCENES[name]
        audio = render(notes(args.seconds), args.seconds, params)
        path = os.path.join(args.out, f"{name}.wav")
        write_wav(path, audio)
        print(f"{path}: {len(audio)} samples, peak {np.abs(audio).max():.3f}, "
              f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
