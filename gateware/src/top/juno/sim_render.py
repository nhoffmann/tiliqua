# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Offline renderer for the Juno voice engine (``JunoVoice``) using ``amaranth.sim``.

Drives the voice block with a note schedule and collects stereo output samples,
so changes can be listened to (WAV) and checked numerically without hardware.

Sample rate is nominal 48 kHz: the hardware pulls one stereo sample per audio
frame, here the testbench pulls them as fast as the simulated block produces
them. Pitch is sent in semitones (note; bend is a global setting added after the
portamento), like the juno top level.

Usage (writes WAVs to ``build/juno_sim/``)::

    pdm run python -m top.juno.sim_render [--seconds 0.5] [--engine vl|amaranth] [scene ...]

``--engine vl`` (default) renders with Verilator (``vl_render.py``, seconds per
clip, bit-identical); ``amaranth`` uses amaranth.sim (~15 min per second).
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
from amaranth import Module
from amaranth.sim import Simulator

from amaranth_future import fixed
from tiliqua.dsp import ASQ
from tiliqua.test import stream

from top.juno.juno_core import JunoCore
from top.juno.juno_lfo import pw_knob_to_point
from top.juno.juno_voice import MultiADSR, MultiLadder, PitchToInc
from top.juno.ladder_model import KBD_REF_NOTE, res_knob_to_reg
from top.juno.adsr_model import attack_inc_for_time, eps_for_time
from top.juno.juno_glide import DETUNE_BITS, glide_coef

FS = 48000
N_VOICES = 6
EnvUQ = MultiADSR.EnvUQ


def cutoff_base(cutoff, kbd, note, mod_wheel=0.0):
    """Per-voice cutoff base, same formula as the firmware (normalized octaves):
    cutoff + keyboard tracking relative to C1 (note 24, KR-106) + mod wheel offset.
    ``cutoff`` is the internal c (0..1 = 20 Hz .. 20.5 kHz); the firmware gets it
    from the knob via ``ladder_model.cutoff_knob_to_c`` (Juno-106 slider curve)."""
    octave = 1.0 / MultiLadder.CUTOFF_OCTAVES
    return cutoff + kbd * (note - KBD_REF_NOTE) / 12 * octave + mod_wheel * MOD_WHEEL_RANGE


# Mod wheel fully up opens the cutoff by this much (normalized octaves).
MOD_WHEEL_RANGE = 0.5


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
    # Envelope (Juno-106 style, Phase 4.2): attack = seconds to full level,
    # decay / release = seconds to -60 dB, sustain 0..1
    attack:  float = 0.001
    decay:   float = 0.3
    sustain: float = 0.5
    release: float = 0.3
    res:     float = 0.5      # ladder resonance knob 0..1 (k = 4.5 res; firmware default 0.5)
    # Exponential cutoff (normalized octaves, see MultiLadder); firmware defaults
    cutoff:  float = 0.35
    env_amt: float = 0.6
    kbd:     float = 0.5
    mod_wheel: float = 0.0
    bend_semitones: float = 0.0    # pitch bend applied to all notes (top.py: +-2 range)
    portamento: float = 0.0        # glide time constant in seconds (0 = off, juno_glide)
    detune_cents: tuple = ()       # extra per-voice pitch offsets in cents
    variance: float = 0.0          # per-voice tolerances 0..1 (firmware default 1.0)
    unison_detune: float = 0.0     # unison spread knob 0..1 (+-25 cents; firmware: unison only)
    cutoff_cv_volts: float = 0.0   # CV in1, 1 V/oct
    # Juno DCO mixer (firmware defaults: saw on, rest off)
    saw:   bool = True
    pulse: bool = False
    pw:    float = 0.0     # PW knob 0..1
    sub:   float = 0.0     # sub level 0..1
    noise: float = 0.0     # noise level 0..1
    pwm_lfo: bool = False  # PWM source: LFO (True) or manual knob (False)
    # LFO (firmware defaults: no modulation)
    lfo_rate_hz:  float = 5.0
    lfo_delay_s:  float = 0.0
    dco_lfo_semi: float = 0.0   # vibrato depth, semitones at full LFO
    vcf_lfo:      float = 0.0   # LFO -> cutoff, normalized octaves (0.1 = 1 octave)
    vca_gate:     bool = False  # VCA Gate mode instead of envelope
    dco_range: int = 1          # 0 = 16', 1 = 8', 2 = 4'
    # Global effects (firmware defaults: HPF flat, chorus off)
    hpf:    int = 1             # 0 bass boost, 1 flat, 2 240 Hz, 3 720 Hz
    chorus: int = 0             # 0 off, 1 I, 2 II, 3 I+II


# Per-voice variance (component tolerances), same as firmware variance.rs.
VARIANCE_PITCH_CENTS = [1.8, -2.6, 0.7, -1.2, 2.9, -0.4]
VARIANCE_CUTOFF_PCT = [-3.1, 4.6, -0.8, 2.2, -4.7, 1.5]
UNISON_SPREAD = [-1.0, 1.0, -0.6, 0.6, -0.2, 0.2]
DETUNE_MAX_CENTS = 25.0


def voice_detune_cents(params, n_voices=6):
    extra = list(params.detune_cents) + [0.0] * n_voices
    return [VARIANCE_PITCH_CENTS[v] * params.variance
            + UNISON_SPREAD[v] * params.unison_detune * DETUNE_MAX_CENTS + extra[v]
            for v in range(n_voices)]


def variance_cutoff(params, voice):
    """Cutoff offset in normalized octaves (firmware: CUTOFF_RAW / 32768)."""
    import math
    return math.log2(1 + VARIANCE_CUTOFF_PCT[voice] / 100) / 10 * params.variance


def pack_detune(cents):
    """Per-voice cents -> packed ``voice_detune`` (10-bit signed, 1/1024 semitones)."""
    packed = 0
    for n, c in enumerate(cents):
        packed |= (round(c / 100 * 1024) & ((1 << DETUNE_BITS) - 1)) << (n * DETUNE_BITS)
    return packed


def _raw(value, shape):
    return fixed.Const(value, shape=shape)._value


def prepare(notes, seconds, params=None):
    """Notes + params -> raw register values and a sorted event list, shared by
    the Python simulation (``render``) and the Verilator harness (``vl_render``).

    Returns ``(n_samples, settings, events)``: ``settings`` maps ``JunoCore`` port
    names to raw integers; ``events`` are ``(sample, voice, gate, pitch_raw,
    velocity_raw, cutoff_base_raw)``, gate-off before gate-on at the same sample.
    Pitch / velocity / cutoff are only applied on gate-on.
    """
    params = params or Params()
    n_samples = int(seconds * FS)
    settings = {
        "saw_en":         int(params.saw),
        "pulse_en":       int(params.pulse),
        "pw_knob":        round(params.pw * 32768),
        "pwm_lfo":        int(params.pwm_lfo),
        "vca_gate_mode":  int(params.vca_gate),
        "lfo_rate_inc":   round(params.lfo_rate_hz / FS * 2 ** 32),
        "lfo_delay_step": (1 << 30) if params.lfo_delay_s <= 0
                          else max(1, round((1 << 30) / (params.lfo_delay_s * FS))),
        "lfo_dco_depth":  round(params.dco_lfo_semi * 1024),
        "lfo_vcf_depth":  round(params.vcf_lfo * 32768),
        "sub_gain":       round(0.67 * params.sub * 65535),
        "noise_gain":     round(params.noise * 65535),
        "attack_rate":    attack_inc_for_time(max(params.attack, 1 / FS)),
        "decay_rate":     eps_for_time(max(params.decay, 1 / FS)),
        "sustain_level":  min(65535, round(params.sustain * 65536)),
        "release_rate":   eps_for_time(max(params.release, 1 / FS)),
        "reso":           res_knob_to_reg(params.res),   # KR-106-calibrated curve
        "env_amt":        _raw(params.env_amt, ASQ),
        "cutoff_cv":      _raw(params.cutoff_cv_volts / 8.192, ASQ),
        "hpf_mode":       params.hpf,
        "dco_range":      params.dco_range,
        "pitch_bend":     _raw(params.bend_semitones, PitchToInc.PitchSQ),
        "glide_coef":     glide_coef(params.portamento),
        "voice_detune":   pack_detune(voice_detune_cents(params)),
        "chorus_mode":    params.chorus,
    }
    events = []
    for n in notes:
        base = (cutoff_base(params.cutoff, params.kbd, n.note, params.mod_wheel)
                + variance_cutoff(params, n.voice))
        on = (int(n.on_s * FS), n.voice, 1,
              _raw(n.note, PitchToInc.PitchSQ),
              _raw(n.velocity, EnvUQ),
              _raw(max(-1.0, min(base, 0.99997)), ASQ))
        events.append(on)
        events.append((int(n.off_s * FS), n.voice, 0, 0, 0, 0))
    events.sort(key=lambda e: (e[0], e[2]))
    return n_samples, settings, events


def render(notes, seconds, params=None, n_voices=N_VOICES):
    """Render ``seconds`` of audio with amaranth.sim (slow, ~15 min per second
    of audio). Returns float array of shape (n_samples, 2). ``vl_render.render_vl``
    is the fast, bit-identical alternative."""
    n_samples, settings, events = prepare(notes, seconds, params)
    out = np.zeros((n_samples, 2))

    m = Module()
    m.submodules.dut = dut = JunoCore(n_voices=n_voices)

    async def testbench(ctx):
        for name, raw in settings.items():
            port = getattr(dut, name)
            ctx.set(port.as_value() if hasattr(port, "as_value") else port, raw)
        await ctx.tick()

        ev = 0
        for i in range(n_samples):
            while ev < len(events) and events[ev][0] <= i:
                _, voice, gate, pitch, vel, cut = events[ev]
                if gate:
                    ctx.set(dut.voice_pitch_base[voice].as_value(), pitch)
                    ctx.set(dut.voice_velocity[voice].as_value(), vel)
                    ctx.set(dut.voice_cutoff_base[voice].as_value(), cut)
                ctx.set(dut.voice_gates[voice], gate)
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
                   Params(cutoff=0.2, env_amt=0.7, kbd=0.3, sustain=0.0,
                          decay=0.4, release=0.25)),
    # slow attack brassy swell: filter opens with the envelope
    "brass_swell": (lambda s: [Note(0, 48, 0.0, s * 0.7), Note(1, 55, 0.0, s * 0.7),
                               Note(2, 60, 0.0, s * 0.7)],
                    Params(cutoff=0.3, env_amt=0.55, kbd=0.5,
                           attack=0.25, decay=1.2, sustain=0.7, release=0.5)),
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
    # LFO: vibrato with delay, PWM, filter wobble, VCA gate mode
    "lfo_vibrato": (lambda s: [Note(0, 60, 0.0, s * 0.9)],
                    Params(cutoff=0.6, env_amt=0.1, dco_lfo_semi=0.5, lfo_rate_hz=6.0,
                           lfo_delay_s=0.3)),
    "lfo_pwm":     (lambda s: [Note(0, 48, 0.0, s * 0.9), Note(1, 55, 0.0, s * 0.9)],
                    Params(cutoff=0.65, env_amt=0.1, saw=False, pulse=True, pw=0.8,
                           pwm_lfo=True, lfo_rate_hz=1.5)),
    "lfo_vcf":     (lambda s: [Note(0, 43, 0.0, s * 0.9)],
                    Params(cutoff=0.4, env_amt=0.0, res=0.85, vcf_lfo=0.25,
                           lfo_rate_hz=3.0)),
    "vca_gate":    (lambda s: [Note(v, 48 + 3 * v, v * s / 6, v * s / 6 + s / 10)
                               for v in range(N_VOICES)],
                    Params(vca_gate=True, cutoff=0.3, env_amt=0.5, sustain=0.0,
                           decay=0.5)),
    # ladder: resonant envelope sweep, and self-oscillation ringing
    "ladder_sweep": (lambda s: [Note(0, 36, 0.0, s * 0.45), Note(1, 43, s * 0.5, s * 0.95)],
                     Params(cutoff=0.15, env_amt=0.75, res=0.85, sustain=0.12,
                            decay=0.8, release=0.5)),
    "ladder_selfosc": (lambda s: [Note(0, 60, 0.0, s * 0.8)],
                       Params(cutoff=0.6, env_amt=0.0, kbd=1.0, res=1.0, saw=True)),
    # Phase 3: chorus modes and HPF positions on a pad chord
    **{f"chorus_{name}": (lambda s: [Note(0, 48, 0.0, s * 0.85), Note(1, 55, 0.0, s * 0.85),
                                     Note(2, 60, 0.0, s * 0.85), Note(3, 64, 0.0, s * 0.85)],
                          Params(cutoff=0.65, env_amt=0.15, res=0.2, chorus=mode,
                                 attack=0.1, sustain=0.75, release=0.5))
       for name, mode in (("off", 0), ("I", 1), ("II", 2), ("I_II", 3))},
    **{f"hpf_{pos}": (lambda s: [Note(0, 36, 0.0, s * 0.85)],
                      Params(cutoff=0.8, env_amt=0.0, saw=True, sub=0.6, hpf=pos))
       for pos in range(4)},
    # same chord, dark vs bright cutoff knob (no env), back to back
    "cutoff_dark": (lambda s: [Note(0, 48, 0.0, s * 0.8), Note(1, 55, 0.0, s * 0.8)],
                    Params(cutoff=0.4, env_amt=0.0, kbd=0.5)),
    "cutoff_bright": (lambda s: [Note(0, 48, 0.0, s * 0.8), Note(1, 55, 0.0, s * 0.8)],
                      Params(cutoff=0.75, env_amt=0.0, kbd=0.5)),
    # unison (all 6 voices on each note, as the tracker assigns it; notes retriggered
    # here, the tracker plays overlapping notes legato), per-voice variance + spread
    # 0.3 (+-7.5 cents), short glide
    "unison": (lambda s: [Note(v, n, i * s / 4, (i + 1) * s / 4 - (0.02 if i == 3 else 0))
                          for i, n in enumerate([36, 43, 39, 36]) for v in range(6)],
               Params(variance=1.0, unison_detune=0.3, portamento=0.03, cutoff=0.5,
                      env_amt=0.5, sustain=0.7, release=0.4)),
    # portamento: one voice C3 -> C4 -> G3 -> C3 (time constant 80 ms), then a
    # two-voice chord moving up a fourth (each voice glides from its own last note)
    "portamento": (lambda s: [Note(0, n, i * s / 8, (i + 1) * s / 8)
                              for i, n in enumerate([48, 60, 55, 48])]
                   + [Note(0, 48, s / 2, s * 0.7), Note(1, 55, s / 2, s * 0.7),
                      Note(0, 53, s * 0.7, s * 0.95), Note(1, 60, s * 0.7, s * 0.95)],
                   Params(portamento=0.08, cutoff=0.6, sustain=0.8)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--out", default="build/juno_sim")
    parser.add_argument("--engine", choices=["vl", "amaranth"], default="vl")
    parser.add_argument("--raw", action="store_true",
                        help="write the JunoCore output (default: through the output "
                             "stage of top.py, juno_output.output, like the jacks)")
    parser.add_argument("--level", type=float, default=None, help="output level knob 0..1")
    parser.add_argument("--drive", type=float, default=0.0, help="output drive knob 0..1")
    parser.add_argument("scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()
    if args.engine == "vl":
        from top.juno.vl_render import build, render_vl as engine
        build(verbose=True)
    else:
        engine = render
    for name in args.scenes:
        t0 = time.time()
        notes, params = SCENES[name]
        audio = engine(notes(args.seconds), args.seconds, params)
        if not args.raw:
            from top.juno import juno_output
            level = juno_output.LEVEL_DEFAULT if args.level is None else args.level
            audio = juno_output.output(audio, level, args.drive)
        path = os.path.join(args.out, f"{name}.wav")
        write_wav(path, audio)
        print(f"{path}: {len(audio)} samples, peak {np.abs(audio).max():.3f}, "
              f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
