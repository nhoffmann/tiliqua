# Copyright (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
#
# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
6-voice MIDI polysynth (Juno port, work in progress).

    .. code-block:: text

              Audio / CV
           ┌────┐
           │in0 │◄─ -
           │in1 │◄─ filter cutoff (1V/oct)
           │in2 │◄─ drive
           │in3 │◄─ -
           └────┘
           ┌────┐
           │out0│─► -
           │out1│─► MIDI CLK
           │out2│─► audio out (L)
           │out3│─► audio out (R)
           └────┘

Voices are controlled using a MIDI keyboard (TRS MIDI or USB host is
supported, selected on the MISC page).

    - Voice mix is sent to output channels 2 and 3 (last 2 jacks).

    - Note velocity scales each voice's envelope (filter and VCA).

    - Filter cutoff is exponential (Juno-style): cutoff knob + keyboard tracking +
      mod wheel (computed per voice by firmware) + env amount x envelope + CV in1
      (1V/oct, computed per voice in gateware). The envelope also drives a per-voice VCA.

    - Oscillator: Juno DCO per voice (curved saw, pulse with width, sub-oscillator
      one octave down, shared noise), band-limited (PolyBLEP), mixed with the
      DCO page settings.

    - When a jack is patched into input 1 or 2, CV modulates all voices up to
      audio rate (filter cutoff 1V/oct, drive as a bipolar offset). Input 0 is
      unused for now (assignable CV inputs are planned).

All 6 voices are summed to mono (Juno-style); the stereo image comes from the
chorus. End of chain: 4-position HPF, BBD-style chorus (off / I / II / I+II),
then distortion (drive).

    .. code-block:: text

              USB MIDI   TRS MIDI
                 ─────┐ ┌─────  (`usb-host` setting)
                ┌─────▼─▼─────┐
                │Voice Tracker├──►┌─────────┐
                └─────────────┘   │ Voices  │ (x6, LFO: vibrato / PWM / VCF)
                                  │   DCO   │ (saw / pulse / sub / noise)
                                  │    ▼    │
                                  │ Ladder  │ (cutoff, reso, env, kbd)
                                  │    ▼    │
                                  │   VCA   │ (ADSR or gate)
                                  └────┬────┘
                                  ┌────▼────┐
                                  │Mix mono │
                                  └────┬────┘
                                  ┌────▼────┐
                                  │   HPF   │ (`hpf` setting)
                                  └────┬────┘
                                  ┌────▼────┐
                                  │ Chorus  │ (`chorus` setting, mono -> stereo)
                                  └────┬────┘
                                  ┌────▼────┐
                                  │  Drive  │ (`drive` setting)
                                  └────┬────┘
                                  ┌────┴───┐
                                  │Audio   │
                                  │OUT (4x)├──────► out2 (L)
                                  └────────┴──────► out3 (R)

If a MIDI clock is present on TRS or USB MIDI, it is divided by 24 (1PPQN)
and sent out on output jack 1. STOP/START/CONTINUE is respected. The LED
will pulse with the clock ONLY if that jack is connected.

The following options are tweakable in the menu. Note that the USB/TRS MIDI
can also be used to control most of these through CCs as follows:

    .. code-block:: text

        Page    Parameter     CC  Description
        ────    ─────────     ──  ───────────
        HELP    scroll         -  scroll help text up/down

        VOICE   assign        14  poly / unison (all voices, last note, legato)
        VOICE   porta          5  portamento (per-voice glide, 0 = off .. 3 s)
        VOICE   detune        94  unison spread (0 .. +-25 cents)
        VOICE   variance       9  per-voice tolerances (pitch +-3 ct, cutoff +-5 %)

        LFO     rate          24  LFO rate (0.05 .. 29.5 Hz)
        LFO     delay         25  LFO fade-in after a new note (0 .. 3.4 s)

        DCO     range         15  16' / 8' / 4' (octave)
        DCO     lfo           22  LFO -> pitch (vibrato depth)
        DCO     pwm-mode      16  pulse width source: manual / LFO
        DCO     saw           17  saw on/off (Juno curved ramp)
        DCO     pulse         18  pulse on/off
        DCO     pw            19  pulse width (0 = square .. narrow)
        DCO     sub           20  sub-oscillator level
        DCO     noise         21  noise level

        VCF     cutoff        26  filter cutoff (exponential)
        VCF     reso          23  ladder resonance (self-oscillates near max)
        VCF     env-amt       27  envelope -> cutoff amount
        VCF     env-pol       28  envelope polarity (+/-)
        VCF     kbd-track     29  keyboard tracking (0 = off, 1 = 100 %)
        VCF     lfo           34  LFO -> cutoff
        VCF     hpf           36  HPF: bass boost / flat / 240 Hz / 720 Hz

        ENV     vca-mode      35  VCA follows envelope or gate
        ENV     attack        30  envelope attack (VCF + VCA)
        ENV     decay         31  envelope decay
        ENV     sustain       32  envelope sustain
        ENV     release       33  envelope release

        EFFECT  drive         40  distortion amount
        EFFECT  chorus        41  chorus off / I / II / I+II (stereo from the mono voice sum)

        MISC    cc-highlight   -  highlight changed on CC input
        MISC    midi-ch        -  filter MIDI to specific channel (default: all)
        MISC    usb-host       -  enable USB host MIDI (disables TRS)
        MISC    serial-debug   -  dump MIDI data out serial port
        MISC    save-opts      -  save all options to flash
        MISC    wipe-opts      -  reset all options to defaults

    MIDI CC 1 (mod wheel) opens the filter cutoff. CC 64 (sustain
    pedal) holds voices. Pitch bend is also supported.

"""

import math
import os
import sys

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.fifo import SyncFIFOBuffered
from amaranth.lib.wiring import In, Out, connect, flipped
from amaranth_soc import csr

from amaranth_future import fixed

from guh.engines.midi import USBMIDIHost

from tiliqua import dsp, midi
from tiliqua.build import sim
from tiliqua.build.cli import top_level_cli
from tiliqua.build.types import BitstreamHelp
from tiliqua.dsp import ASQ
from tiliqua.tiliqua_soc import TiliquaSoc

# top.py is run as a script (pdm run juno), so import the sibling module directly.
from juno_voice import MultiADSR, PitchToInc
from juno_core import JunoCore
from juno_tracker import JunoVoiceState, JunoVoiceTracker

class PolySynth(wiring.Component):

    N_VOICES = 6

    i: In(stream.Signature(data.ArrayLayout(ASQ, 4)))
    o: Out(stream.Signature(data.ArrayLayout(ASQ, 4)))

    i_midi: In(stream.Signature(midi.MidiMessage))

    drive: In(unsigned(16))
    reso: In(unsigned(16))
    env_amt: In(ASQ)
    attack_rate: In(unsigned(32))
    decay_rate: In(unsigned(17))
    sustain_level: In(MultiADSR.EnvUQ)
    release_rate: In(unsigned(17))

    # Juno DCO mixer (global)
    saw_en: In(1)
    pulse_en: In(1)
    pw_knob: In(unsigned(16))
    pwm_lfo: In(1)
    sub_gain: In(unsigned(16))
    noise_gain: In(unsigned(16))

    # VCA mode, DCO range, HPF / chorus and LFO
    vca_gate_mode: In(1)
    dco_range: In(2)
    hpf_mode: In(2)
    chorus_mode: In(2)
    lfo_rate_inc: In(unsigned(32))
    lfo_delay_step: In(unsigned(32))
    lfo_dco_depth: In(unsigned(16))
    lfo_vcf_depth: In(unsigned(16))
    # portamento (juno_glide coef, 0 = off), unison, per-voice detune
    glide_coef: In(unsigned(18))
    unison: In(1)
    detune_we: In(1)
    detune_voice: In(3)
    detune_value: In(signed(10))


    # Jack detection (directly from pmod hardware)
    jack: In(unsigned(8))

    # Per-voice cutoff base (firmware: knob + keyboard tracking + mod wheel)
    voice_cutoff_base: In(ASQ).array(N_VOICES)

    voice_states: Out(JunoVoiceState).array(N_VOICES)
    voice_cutoffs: Out(unsigned(8)).array(N_VOICES)

    def __init__(self):
        super().__init__()

    def elaborate(self, platform):
        m = Module()

        n_voices = self.N_VOICES

        m.submodules.voice_tracker = voice_tracker = JunoVoiceTracker(
            max_voices=n_voices, zero_velocity_gate=False)

        # Pitch bend range: +-2 semitones (exact, log domain).
        # PitchSQ raw = semitones * 1024; ASQ raw bend = +-32768 -> +-2048 = +-2 semitones.
        BEND_SHIFT = 4

        # Connect MIDI stream -> voice tracker
        wiring.connect(m, wiring.flipped(self.i_midi), voice_tracker.i)

        # Juno voices + global LFO (same wiring as sim_render.py)
        m.submodules.voice_block = voice_block = JunoCore(n_voices=n_voices)

        # latch CV ins
        cv = Signal.like(self.i.payload)
        m.d.comb += self.i.ready.eq(1)
        with m.If(self.i.valid):
            m.d.sync += cv.eq(self.i.payload)

        # CV 0: unused for now (phase modulation removed with the wavetable oscillator)

        # CV 2: drive bipolar offset (when jack 2 patched)
        SQDrive = dsp.mac.SQNative  # SQ(3, 15)
        drive_base = Signal(SQDrive)
        m.d.comb += drive_base.as_value().eq(self.drive << 2)
        drive_sum = drive_base + cv[2]
        drive_val = Signal(SQDrive)
        m.d.comb += drive_val.eq(drive_sum.saturate(SQDrive))

        # per-voice detune (variance + unison spread), latched one voice per write
        for n in range(n_voices):
            detune = Signal(signed(10), name=f"detune_{n}")
            with m.If(self.detune_we & (self.detune_voice == n)):
                m.d.sync += detune.eq(self.detune_value)
            m.d.comb += voice_block.voice_detune[n].eq(detune)

        # per-voice params
        for n in range(n_voices):
            # Note-on velocity scales the envelope (VCF + VCA).
            vel_out = Signal(MultiADSR.EnvUQ, name=f"vel_out_{n}")
            m.d.comb += vel_out.eq(fixed.UQ(0, 8)(voice_tracker.o[n].velocity_on))
            # Log-domain note pitch = glide target (bend, vibrato, range added
            # after the glide in JunoCore)
            pitch = Signal(PitchToInc.PitchSQ, name=f"pitch_{n}")
            m.d.comb += pitch.as_value().eq(voice_tracker.o[n].note << PitchToInc.FINE_BITS)
            m.d.comb += [
                self.voice_states[n].eq(voice_tracker.o[n]),
                self.voice_cutoffs[n].eq(voice_block.voice_cutoffs[n]),
                voice_block.voice_gates[n].eq(voice_tracker.o[n].gate),
                voice_block.voice_pitch_base[n].eq(pitch),
                voice_block.voice_velocity[n].eq(vel_out),
                voice_block.voice_cutoff_base[n].eq(self.voice_cutoff_base[n]),
                voice_tracker.voice_active[n].eq(voice_block.voice_active[n]),
            ]

        # global params
        m.d.comb += [
            voice_block.reso.eq(self.reso),
            voice_block.env_amt.eq(self.env_amt),
            # CV 1: filter cutoff, 1V/oct (when jack 1 patched)
            voice_block.cutoff_cv.eq(Mux(self.jack[1], cv[1], 0)),
            voice_block.attack_rate.eq(self.attack_rate),
            voice_block.decay_rate.eq(self.decay_rate),
            voice_block.sustain_level.eq(self.sustain_level),
            voice_block.release_rate.eq(self.release_rate),
            voice_block.saw_en.eq(self.saw_en),
            voice_block.pulse_en.eq(self.pulse_en),
            voice_block.pw_knob.eq(self.pw_knob),
            voice_block.pwm_lfo.eq(self.pwm_lfo),
            voice_block.sub_gain.eq(self.sub_gain),
            voice_block.noise_gain.eq(self.noise_gain),
            voice_block.vca_gate_mode.eq(self.vca_gate_mode),
            voice_block.dco_range.eq(self.dco_range),
            voice_block.hpf_mode.eq(self.hpf_mode),
            voice_block.chorus_mode.eq(self.chorus_mode),
            voice_block.lfo_rate_inc.eq(self.lfo_rate_inc),
            voice_block.lfo_delay_step.eq(self.lfo_delay_step),
            voice_block.pitch_bend.as_value().eq(voice_tracker.pitch_bend.as_value() >> BEND_SHIFT),
            voice_block.glide_coef.eq(self.glide_coef),
            voice_tracker.unison.eq(self.unison),
            voice_block.lfo_dco_depth.eq(self.lfo_dco_depth),
            voice_block.lfo_vcf_depth.eq(self.lfo_vcf_depth),
        ]

        # JunoCore stereo output (HPF + chorus inside) -> drive
        m.submodules.vb_split2 = vb_split2 = dsp.Split(n_channels=2, source=voice_block.o)

        def scaled_tanh(x):
            return math.tanh(3.0*x)

        m.submodules.mac_server = mac_server = dsp.mac.RingMACServer()

        outs = []
        for lr in [0, 1]:
            vca = dsp.VCA(macp=mac_server.new_client())
            waveshaper = dsp.WaveShaper(lut_function=scaled_tanh,
                                        macp=mac_server.new_client())
            setattr(m.submodules, f"out_gainvca_{lr}", vca)
            setattr(m.submodules, f"out_waveshaper_{lr}", waveshaper)

            dsp.connect_remap(m, vb_split2.o[lr], vca.i, lambda o, i : [
                i.payload[0].eq(o.payload<<2),
                i.payload[1].eq(drive_val)
            ])

            wiring.connect(m, vca.o, waveshaper.i)
            outs.append(waveshaper.o)

        # Final outputs on channel 2, 3
        m.submodules.merge4 = merge4 = dsp.Merge(
                n_channels=4, sink=wiring.flipped(self.o))
        merge4.wire_valid(m, [0, 1])
        wiring.connect(m, outs[0], merge4.i[2])
        wiring.connect(m, outs[1], merge4.i[3])

        return m

class SynthPeripheral(wiring.Component):

    class Drive(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class Reso(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class AttackRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(32))

    class DecayRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(17))

    class SustainLevel(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class ReleaseRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(17))

    class Dco(csr.Register, access="w"):
        saw:      csr.Field(csr.action.W, unsigned(1))
        pulse:    csr.Field(csr.action.W, unsigned(1))
        pwm_lfo:  csr.Field(csr.action.W, unsigned(1))
        vca_gate: csr.Field(csr.action.W, unsigned(1))
        range:    csr.Field(csr.action.W, unsigned(2))   # 0 = 16', 1 = 8', 2 = 4'

    class PwKnob(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class LfoRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(32))

    class LfoDelay(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(32))

    class Glide(csr.Register, access="w"):
        coef: csr.Field(csr.action.W, unsigned(18))

    class VoiceCfg(csr.Register, access="w"):
        # per-voice detune (1/1024 semitones), written one voice at a time
        detune: csr.Field(csr.action.W, signed(10))
        voice:  csr.Field(csr.action.W, unsigned(3))
        unison: csr.Field(csr.action.W, unsigned(1))

    class LfoDepths(csr.Register, access="w"):
        dco: csr.Field(csr.action.W, unsigned(16))
        vcf: csr.Field(csr.action.W, unsigned(16))

    class MixLevels(csr.Register, access="w"):
        sub:   csr.Field(csr.action.W, unsigned(16))
        noise: csr.Field(csr.action.W, unsigned(16))

    class EnvAmt(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class CutoffBase(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class Voice(csr.Register, access="r"):
        note:   csr.Field(csr.action.R, unsigned(8))
        cutoff: csr.Field(csr.action.R, unsigned(8))

    class Fx(csr.Register, access="w"):
        """HPF position (0 bass boost, 1 flat, 2 240 Hz, 3 720 Hz), chorus (0 off, 1 I, 2 II, 3 I+II)."""
        hpf:    csr.Field(csr.action.W, unsigned(2))
        chorus: csr.Field(csr.action.W, unsigned(2))

    class MidiWrite(csr.Register, access="w"):
        msg: csr.Field(csr.action.W, unsigned(32))

    class MidiRead(csr.Register, access="r"):
        msg: csr.Field(csr.action.R, unsigned(32))

    class UsbMidiHost(csr.Register, access="w"):
        """USB MIDI host settings."""
        # 0 = off. 1 = enable VBUS and forward USB MIDI.
        host:     csr.Field(csr.action.W, unsigned(1))

    class UsbMidiCfg(csr.Register, access="w"):
        # Hardcoded MIDI streaming endpoint location (device specific)
        value: csr.Field(csr.action.W, unsigned(4))

    class MidiChannelFilter(csr.Register, access="w"):
        """Channel filter. 0 = no filter, 1..16 = single MIDI channel."""
        value: csr.Field(csr.action.W, unsigned(5))

    def __init__(self, synth=None):
        self.synth = synth
        # 8-bit address space (256 B): the 128 B window filled up with the LFO registers
        regs = csr.Builder(addr_width=8, data_width=8)
        voices_csr_end = 0x8+PolySynth.N_VOICES*4
        self._drive         = regs.add("drive",         self.Drive(),         offset=0x0)
        self._reso          = regs.add("reso",          self.Reso(),          offset=0x4)
        self._voices        = [regs.add(f"voices{i}",   self.Voice(),
                               offset=0x8+i*4) for i in range(PolySynth.N_VOICES)]
        # +0x0 / +0x4 were the Diffuser matrix registers (Diffuser replaced by JunoFx)
        self._fx            = regs.add("fx",            self.Fx(),            offset=voices_csr_end + 0x0)
        self._midi_write    = regs.add("midi_write",    self.MidiWrite(),     offset=voices_csr_end + 0x8)
        self._midi_read     = regs.add("midi_read",     self.MidiRead(),      offset=voices_csr_end + 0xC)
        self._midi_host     = regs.add("usb_midi_host", self.UsbMidiHost(),   offset=voices_csr_end + 0x10)
        self._midi_cfg      = regs.add("usb_midi_cfg",  self.UsbMidiCfg(),    offset=voices_csr_end + 0x14)
        self._midi_endp     = regs.add("usb_midi_endp", self.UsbMidiCfg(),    offset=voices_csr_end + 0x18)
        self._attack_rate   = regs.add("attack_rate",   self.AttackRate(),    offset=voices_csr_end + 0x1C)
        self._decay_rate    = regs.add("decay_rate",    self.DecayRate(),     offset=voices_csr_end + 0x20)
        self._sustain_level = regs.add("sustain_level", self.SustainLevel(),  offset=voices_csr_end + 0x24)
        self._release_rate  = regs.add("release_rate",  self.ReleaseRate(),   offset=voices_csr_end + 0x28)
        # 0x2C..0x34 were wavetable addr/data + LFO (removed with the wavetable oscillator)
        self._dco           = regs.add("dco",           self.Dco(),           offset=voices_csr_end + 0x2C)
        self._pw_knob       = regs.add("pw_knob",       self.PwKnob(),        offset=voices_csr_end + 0x30)
        self._mix_levels    = regs.add("mix_levels",    self.MixLevels(),     offset=voices_csr_end + 0x34)
        self._midi_ch_filt  = regs.add("midi_ch_filter",self.MidiChannelFilter(), offset=voices_csr_end + 0x38)
        self._env_amt       = regs.add("env_amt",       self.EnvAmt(),        offset=voices_csr_end + 0x3C)
        self._cutoff_base   = [regs.add(f"cutoff_base{i}", self.CutoffBase(),
                               offset=voices_csr_end + 0x40 + i*4) for i in range(PolySynth.N_VOICES)]
        self._lfo_rate      = regs.add("lfo_rate",      self.LfoRate(),       offset=voices_csr_end + 0x58)
        self._lfo_delay     = regs.add("lfo_delay",     self.LfoDelay(),      offset=voices_csr_end + 0x5C)
        self._lfo_depths    = regs.add("lfo_depths",    self.LfoDepths(),     offset=voices_csr_end + 0x60)
        self._glide         = regs.add("glide",         self.Glide(),         offset=voices_csr_end + 0x64)
        self._voice_cfg     = regs.add("voice_cfg",     self.VoiceCfg(),      offset=voices_csr_end + 0x68)
        self._bridge = csr.Bridge(regs.as_memory_map())
        super().__init__({
            "bus": In(csr.Signature(addr_width=regs.addr_width, data_width=regs.data_width)),
            "i_midi": In(stream.Signature(midi.MidiMessage)),
            "usb_midi_host": Out(1),
            "usb_midi_cfg_id": Out(4),
            "usb_midi_endpt_id": Out(4),
        })
        self.bus.memory_map = self._bridge.bus.memory_map

    def elaborate(self, platform):
        m = Module()
        m.submodules.bridge = self._bridge
        connect(m, flipped(self.bus), self._bridge.bus)

        # top-level tweakables
        with m.If(self._drive.f.value.w_stb):
            m.d.sync += self.synth.drive.eq(self._drive.f.value.w_data)
        with m.If(self._reso.f.value.w_stb):
            m.d.sync += self.synth.reso.eq(self._reso.f.value.w_data)
        with m.If(self._env_amt.f.value.w_stb):
            m.d.sync += self.synth.env_amt.as_value().eq(self._env_amt.f.value.w_data)
        for i, reg in enumerate(self._cutoff_base):
            with m.If(reg.f.value.w_stb):
                m.d.sync += self.synth.voice_cutoff_base[i].as_value().eq(reg.f.value.w_data)

        # ADSR parameters
        with m.If(self._attack_rate.f.value.w_stb):
            m.d.sync += self.synth.attack_rate.eq(self._attack_rate.f.value.w_data)
        with m.If(self._decay_rate.f.value.w_stb):
            m.d.sync += self.synth.decay_rate.eq(self._decay_rate.f.value.w_data)
        with m.If(self._sustain_level.f.value.w_stb):
            m.d.sync += self.synth.sustain_level.eq(self._sustain_level.f.value.w_data)
        with m.If(self._release_rate.f.value.w_stb):
            m.d.sync += self.synth.release_rate.eq(self._release_rate.f.value.w_data)

        # Juno DCO mixer
        with m.If(self._dco.element.w_stb):
            m.d.sync += [
                self.synth.saw_en.eq(self._dco.f.saw.w_data),
                self.synth.pulse_en.eq(self._dco.f.pulse.w_data),
                self.synth.pwm_lfo.eq(self._dco.f.pwm_lfo.w_data),
                self.synth.vca_gate_mode.eq(self._dco.f.vca_gate.w_data),
                self.synth.dco_range.eq(self._dco.f.range.w_data),
            ]
        with m.If(self._pw_knob.f.value.w_stb):
            m.d.sync += self.synth.pw_knob.eq(self._pw_knob.f.value.w_data)
        # LFO
        with m.If(self._lfo_rate.f.value.w_stb):
            m.d.sync += self.synth.lfo_rate_inc.eq(self._lfo_rate.f.value.w_data)
        with m.If(self._lfo_delay.f.value.w_stb):
            m.d.sync += self.synth.lfo_delay_step.eq(self._lfo_delay.f.value.w_data)
        m.d.sync += self.synth.detune_we.eq(self._voice_cfg.element.w_stb)
        with m.If(self._voice_cfg.element.w_stb):
            m.d.sync += [
                self.synth.unison.eq(self._voice_cfg.f.unison.w_data),
                self.synth.detune_voice.eq(self._voice_cfg.f.voice.w_data),
                self.synth.detune_value.eq(self._voice_cfg.f.detune.w_data),
            ]
        with m.If(self._glide.f.coef.w_stb):
            m.d.sync += self.synth.glide_coef.eq(self._glide.f.coef.w_data)
        with m.If(self._lfo_depths.element.w_stb):
            m.d.sync += [
                self.synth.lfo_dco_depth.eq(self._lfo_depths.f.dco.w_data),
                self.synth.lfo_vcf_depth.eq(self._lfo_depths.f.vcf.w_data),
            ]
        with m.If(self._mix_levels.element.w_stb):
            m.d.sync += [
                self.synth.sub_gain.eq(self._mix_levels.f.sub.w_data),
                self.synth.noise_gain.eq(self._mix_levels.f.noise.w_data),
            ]

        # USB MIDI flags
        with m.If(self._midi_host.f.host.w_stb):
            m.d.sync += self.usb_midi_host.eq(self._midi_host.f.host.w_data)
        with m.If(self._midi_cfg.f.value.w_stb):
            m.d.sync += self.usb_midi_cfg_id.eq(self._midi_cfg.f.value.w_data)
        with m.If(self._midi_endp.f.value.w_stb):
            m.d.sync += self.usb_midi_endpt_id.eq(self._midi_endp.f.value.w_data)

        # voice tracking
        for i, voice in enumerate(self._voices):
            m.d.comb += [
                voice.f.note.r_data  .eq(self.synth.voice_states[i].note),
                voice.f.cutoff.r_data.eq(self.synth.voice_cutoffs[i])
            ]

        # HPF / chorus
        with m.If(self._fx.element.w_stb):
            m.d.sync += [
                self.synth.hpf_mode.eq(self._fx.f.hpf.w_data),
                self.synth.chorus_mode.eq(self._fx.f.chorus.w_data),
            ]

        m.submodules.midi_ch_filt = midi_ch_filt = midi.MidiChannelFilter()
        wiring.connect(m, wiring.flipped(self.i_midi), midi_ch_filt.i)
        with m.If(self._midi_ch_filt.f.value.w_stb):
            m.d.sync += midi_ch_filt.channel.eq(self._midi_ch_filt.f.value.w_data)

        # MIDI injection and arbiter between SoC MIDI and HW MIDI -> synth MIDI.
        m.submodules.soc_midi_fifo = soc_midi_fifo = SyncFIFOBuffered(
            width=24, depth=8)
        m.d.comb += [
            soc_midi_fifo.w_data.eq(self._midi_write.f.msg.w_data),
            soc_midi_fifo.w_en.eq(self._midi_write.element.w_stb),
        ]
        wiring.connect(m, midi_ch_filt.o, self.synth.i_midi)
        with m.If(soc_midi_fifo.r_stream.valid):
            wiring.connect(m, soc_midi_fifo.r_stream, self.synth.i_midi)

        # Pipe filtered MIDI -> SoC read FIFO so SoC can inspect external
        # MIDI traffic.
        m.submodules.read_midi_fifo = read_midi_fifo = SyncFIFOBuffered(
            width=24, depth=8)
        m.d.comb += [
            read_midi_fifo.w_data.eq(midi_ch_filt.o.payload),
            read_midi_fifo.w_en.eq(midi_ch_filt.o.valid & midi_ch_filt.o.ready),
            read_midi_fifo.r_en.eq(self._midi_read.element.r_stb),
        ]

        with m.If(read_midi_fifo.r_level != 0):
            m.d.comb += self._midi_read.f.msg.r_data.eq(read_midi_fifo.r_data)
        with m.Else():
            m.d.comb += self._midi_read.f.msg.r_data.eq(0)


        return m

class PolySoc(TiliquaSoc):

    # Used by `tiliqua_soc.py` to create a MODULE_DOCSTRING rust constant used by the 'help' page.
    module_docstring = sys.modules[__name__].__doc__

    # Stored in manifest and used by bootloader for brief summary of each bitstream.
    bitstream_help = BitstreamHelp(
        brief="MIDI Polysynth (6-voice, Juno WIP)",
        io_left=['', 'filter cv', 'drive cv', '', '', 'clk', 'out L', 'out R'],
        io_right=['navigate menu', 'MIDI host', 'video out', '', '', 'TRS MIDI in']
    )

    def __init__(self, **kwargs):

        # don't finalize the CSR bridge in TiliquaSoc, we're adding more peripherals.
        super().__init__(finalize_csr_bridge=False, **kwargs)

        # WARN: TiliquaSoc ends at 0x00000900
        self.synth_periph_base   = 0x00001100

        # synth controls
        self.synth_periph = SynthPeripheral()
        self.csr_decoder.add(self.synth_periph.bus, addr=self.synth_periph_base, name="synth_periph")

        self.add_rust_constant(
            f"pub const N_VOICES: usize = {PolySynth.N_VOICES};\n")

        # now we can freeze the memory map
        self.finalize_csr_bridge()

    def elaborate(self, platform):

        m = Module()

        m.submodules.polysynth = polysynth = PolySynth()
        self.synth_periph.synth = polysynth

        m.submodules.synth_periph = self.synth_periph

        m.submodules += super().elaborate(platform)

        pmod0 = self.pmod0_periph.pmod

        m.submodules.clock_div = clock_div = midi.MidiClockDivider()

        if sim.is_hw(platform):
            # Polysynth hardware MIDI sources

            # TRS MIDI (serial)
            midi_pins = platform.request("midi")
            m.submodules.serialrx = serialrx = midi.SerialRx(
                    system_clk_hz=60e6, pins=midi_pins)
            m.submodules.midi_decode_trs = midi_decode_trs = midi.MidiDecodeSerial(
                forward_rt=True)
            wiring.connect(m, serialrx.o, midi_decode_trs.i)

            # USB MIDI host
            ulpi = platform.request(platform.default_usb_connection)
            m.submodules.usb = usb = USBMIDIHost(
                    bus=ulpi,
            )
            m.submodules.midi_decode_usb = midi_decode_usb = midi.MidiDecodeUSB(
                forward_rt=True)
            wiring.connect(m, usb.o_midi, midi_decode_usb.i)

            # Only enable VBUS if MIDI HOST is enabled.
            vbus_o = platform.request("usb_vbus_en").o
            with m.If(self.synth_periph.usb_midi_host):
                wiring.connect(m, midi_decode_usb.o, self.synth_periph.i_midi)
                m.d.comb += vbus_o.eq(1)
            with m.Else():
                wiring.connect(m, midi_decode_trs.o, self.synth_periph.i_midi)
                m.d.comb += vbus_o.eq(0)

            # Route RT messages from the active MIDI source to clock divider.
            # Drain the inactive source, connect the active one.
            with m.If(self.synth_periph.usb_midi_host):
                wiring.connect(m, midi_decode_usb.o_rt, clock_div.i)
                m.d.comb += midi_decode_trs.o_rt.ready.eq(1)
            with m.Else():
                wiring.connect(m, midi_decode_trs.o_rt, clock_div.i)
                m.d.comb += midi_decode_usb.o_rt.ready.eq(1)

        # polysynth audio + jack detection
        wiring.connect(m, pmod0.o_cal, polysynth.i)
        m.d.comb += polysynth.jack.eq(pmod0.jack)

        m.d.comb += [
            pmod0.i_cal.payload[0].eq(polysynth.o.payload[0]),
            # Channel 1 is MIDI clock square wave (0 / 0.5).
            pmod0.i_cal.payload[1].eq(
                Mux(clock_div.o, fixed.Const(0.5, shape=ASQ), 0)),
            # Channel 2/3 are stereo outputs
            pmod0.i_cal.payload[2].eq(polysynth.o.payload[2]),
            pmod0.i_cal.payload[3].eq(polysynth.o.payload[3]),
            pmod0.i_cal.valid.eq(polysynth.o.valid),
            polysynth.o.ready.eq(pmod0.i_cal.ready),
        ]

        return m


if __name__ == "__main__":
    this_path = os.path.dirname(os.path.realpath(__file__))
    top_level_cli(PolySoc, path=this_path, archiver_callback=lambda archiver: archiver.with_option_storage())
