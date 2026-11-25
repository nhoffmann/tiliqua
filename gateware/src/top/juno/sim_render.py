# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Offline renderer for the Juno voice engine (``JunoVoice``) using ``amaranth.sim``.

Drives the voice block with a note schedule and collects stereo output samples,
so changes can be listened to (WAV) and checked numerically without hardware.

Sample rate is nominal 48 kHz: the hardware pulls one stereo sample per audio
frame, here the testbench pulls them as fast as the simulated block produces
them. Pitch is mapped exactly like ``MidiVoiceTracker`` (freq_inc = f / 48000).

Usage (writes WAVs to ``build/juno_sim/``)::

    pdm run python -m top.juno.sim_render [--seconds 0.5] [scene ...]
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np
from amaranth import Module
from amaranth.sim import Simulator

from amaranth_future import fixed
from tiliqua.dsp import ASQ
from tiliqua.test import stream

from top.juno.juno_voice import JunoVoice, MultiADSR

FS = 48000
N_VOICES = 6
WT_SIZE = JunoVoice.WAVETABLE_SIZE
EnvUQ = MultiADSR.EnvUQ


def note_to_freq_inc(note):
    return 440.0 * 2 ** ((note - 69) / 12.0) / FS


def adsr_ui_to_rate(ui_value):
    """Same mapping as the firmware (main.rs): UI 0..32768 -> 1 ms..2000 ms."""
    ms_x32k = 32768 + ui_value * 1999
    return min(45_000_000 // ms_x32k, 65535)


def saw_table(amplitude=0.9):
    return [int(amplitude * 32767 * (2.0 * i / WT_SIZE - 1.0)) for i in range(WT_SIZE)]


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
    wavetable: list = field(default_factory=saw_table)


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
        for addr, sample in enumerate(params.wavetable):
            ctx.set(dut.wt_write_addr, addr)
            ctx.set(dut.wt_write_data, sample)
            ctx.set(dut.wt_write_en, 1)
            await ctx.tick()
        ctx.set(dut.wt_write_en, 0)

        ctx.set(dut.attack_rate.as_value(), params.attack)
        ctx.set(dut.decay_rate.as_value(), params.decay)
        ctx.set(dut.sustain_level.as_value(), params.sustain)
        ctx.set(dut.release_rate.as_value(), params.release)
        ctx.set(dut.reso, params.reso)
        await ctx.tick()

        ev = 0
        for i in range(n_samples):
            while ev < len(events) and events[ev][0] <= i:
                _, note, gate = events[ev]
                if gate:
                    ctx.set(dut.voice_freq_incs[note.voice],
                            fixed.Const(note_to_freq_inc(note.note), shape=ASQ))
                    ctx.set(dut.voice_velocity[note.voice],
                            fixed.Const(note.velocity, shape=EnvUQ))
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
    # single held note, then release
    "single_c3": lambda s: [Note(0, 48, 0.0, s * 0.6)],
    # C minor chord on 3 voices (2 left, 1 right)
    "chord_cmin": lambda s: [Note(0, 48, 0.0, s * 0.6), Note(1, 51, 0.0, s * 0.6),
                             Note(2, 55, 0.0, s * 0.6)],
    # short staccato notes, one per voice
    "staccato": lambda s: [Note(v, 48 + 2 * v, v * s / 6, v * s / 6 + s / 12)
                           for v in range(N_VOICES)],
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
        audio = render(SCENES[name](args.seconds), args.seconds)
        path = os.path.join(args.out, f"{name}.wav")
        write_wav(path, audio)
        print(f"{path}: {len(audio)} samples, peak {np.abs(audio).max():.3f}, "
              f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
