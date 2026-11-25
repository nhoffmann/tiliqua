# Copyright (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
#
# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
6-voice MIDI polysynth (Juno port, work in progress).

    .. code-block:: text

              Audio / CV
           ┌────┐
           │in0 │◄─ phase modulation
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

    - When a jack is patched into input 0, 1 or 2, CV can be used to modulate
      all voices simultaneously up to audio rate (phase mod, filter cutoff and
      drive). Inputs 1 and 2 act as bipolar offsets on top of the existing
      envelope / knob value. Patch an LFO into phase CV for retro 'tape-wow'
      like detuning. Patch an oscillator into phase CV for FM effects, or into
      drive for AM effect.

Each voice is hard panned left or right in the stereo field, with 2 end-of-chain
effects: distortion and diffusion (delay), both of which can be mixed in with
the UI.

    .. code-block:: text

              USB MIDI   TRS MIDI
                 ─────┐ ┌─────  (`usb-host` setting)
                ┌─────▼─▼─────┐
                │Voice Tracker├──►┌─────────┐
                └─────────────┘   │ Voices  │
                                  │  (x6)   │
                                  │         │
                                  │ Wavetbl │
                                  │    ▼    │
                                  │  ADSR   │ (ADSR sets filter cutoff)
                                  │    ▼    │
                                  │ Filter  │ (resonance =
                                  └─────────┘  `reso` setting)
                                  ┌────▼────┐
                                  │Mix L/R  │
                                  │(stereo) │
                                  └────┬────┘
                                  ┌────▼────┐ (wet/dry mix =
                                  │Diffuser │  `diffuse` setting)
                                  └────┬────┘
                                  ┌────▼────┐
                                  │  Drive  │ (`drive` setting)
                                  └────┬────┘
                                  ┌────┴───┐
                                  │Audio   │
                                  │OUT (4x)├──────► out2 (L)
                                  └────────┴──────► out3 (R)

The VOICE page allows selection between many wavetables. The 'proc' option
allows applying effects (e.g. saturation, wavefolding, rectify) to
the wavetable **before** it hits the voice filter. When no CV is patched
into input 0, a built-in sine LFO modulates the phase of all voices
(rate and depth are adjustable on the VOICE page).

If a MIDI clock is present on TRS or USB MIDI, it is divided by 24 (1PPQN)
and sent out on output jack 1. STOP/START/CONTINUE is respected. The LED
will pulse with the clock ONLY if that jack is connected.

The following options are tweakable in the menu. Note that the USB/TRS MIDI
can also be used to control most of these through CCs as follows:

    .. code-block:: text

        Page    Parameter     CC  Description
        ────    ─────────     ──  ───────────
        HELP    scroll         -  scroll help text up/down

        VOICE   waveform      20  wavetable selection
        VOICE   proc          21  wavetable processing mode
        VOICE   proc-amt      22  wavetable processing amount

        VCF     cutoff        26  filter cutoff (exponential)
        VCF     reso          23  filter resonance
        VCF     env-amt       27  envelope -> cutoff amount
        VCF     env-pol       28  envelope polarity (+/-)
        VCF     kbd-track     29  keyboard tracking (0 = off, 1 = 100 %)
        VOICE   lfo-rate      24  phase modulation LFO rate
        VOICE   lfo-depth     25  phase modulation LFO depth

        ADSR    attack        30  envelope attack (VCF + VCA)
        ADSR    decay         31  envelope decay
        ADSR    sustain       32  envelope sustain
        ADSR    release       33  envelope release

        EFFECT  drive         40  distortion amount
        EFFECT  diffuse       41  diffusion wet/dry mix

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
from juno_voice import JunoVoice, MultiADSR

class Diffuser(wiring.Component):

    i: In(stream.Signature(data.ArrayLayout(ASQ, 4)))
    o: Out(stream.Signature(data.ArrayLayout(ASQ, 4)))

    def __init__(self):
        super().__init__()

        # 4 delay lines, backed by 4 independent SRAM banks.

        self.delay_lines = [
            dsp.DelayLine(max_delay=2048),
            dsp.DelayLine(max_delay=4096),
            dsp.DelayLine(max_delay=8192),
            dsp.DelayLine(max_delay=8192),
        ]

        self.diffuser = dsp.delay_effect.Diffuser(self.delay_lines)

        # Coefficients of this are tweaked by the SoC

        self.matrix   = self.diffuser.matrix_mix

    def elaborate(self, platform):
        m = Module()

        dsp.named_submodules(m.submodules, self.delay_lines)

        m.submodules.diffuser = self.diffuser

        wiring.connect(m, wiring.flipped(self.i), self.diffuser.i)
        wiring.connect(m, self.diffuser.o, wiring.flipped(self.o))

        return m

class PolySynth(wiring.Component):

    N_VOICES = 6

    i: In(stream.Signature(data.ArrayLayout(ASQ, 4)))
    o: Out(stream.Signature(data.ArrayLayout(ASQ, 4)))

    i_midi: In(stream.Signature(midi.MidiMessage))

    drive: In(unsigned(16))
    reso: In(unsigned(16))
    env_amt: In(ASQ)
    attack_rate: In(MultiADSR.EnvUQ)
    decay_rate: In(MultiADSR.EnvUQ)
    sustain_level: In(MultiADSR.EnvUQ)
    release_rate: In(MultiADSR.EnvUQ)

    # Wavetable write port (firmware fills via CSR)
    wt_write_addr: In(unsigned(9))
    wt_write_data: In(signed(16))
    wt_write_en: In(unsigned(1))

    lfo: In(ASQ)

    # Jack detection (directly from pmod hardware)
    jack: In(unsigned(8))

    # Per-voice cutoff base (firmware: knob + keyboard tracking + mod wheel)
    voice_cutoff_base: In(ASQ).array(N_VOICES)

    voice_states: Out(midi.MidiVoice).array(N_VOICES)
    voice_cutoffs: Out(unsigned(8)).array(N_VOICES)

    def __init__(self):
        super().__init__()

    def elaborate(self, platform):
        m = Module()

        n_voices = self.N_VOICES

        m.submodules.voice_tracker = voice_tracker = midi.MidiVoiceTracker(
            max_voices=n_voices, velocity_mod=False, zero_velocity_gate=False)

        # Connect MIDI stream -> voice tracker
        wiring.connect(m, wiring.flipped(self.i_midi), voice_tracker.i)

        m.submodules.voice_block = voice_block = JunoVoice(n_voices=n_voices)

        # latch CV ins
        cv = Signal.like(self.i.payload)
        m.d.comb += self.i.ready.eq(1)
        with m.If(self.i.valid):
            m.d.sync += cv.eq(self.i.payload)

        m.submodules.lfo_lpf = lfo_lpf = dsp.OnePole()
        m.d.comb += [
            lfo_lpf.i.payload.eq(self.lfo),
            lfo_lpf.i.valid.eq(self.i.valid),
            lfo_lpf.o.ready.eq(1),
            lfo_lpf.shift.eq(6),
        ]

        # CV 0: phase modulation (when jack 0 patched, otherwise use LFO)
        m.d.comb += voice_block.phase_mod.eq(
            Mux(self.jack[0], cv[0], lfo_lpf.o.payload))

        # CV 2: drive bipolar offset (when jack 2 patched)
        SQDrive = dsp.mac.SQNative  # SQ(3, 15)
        drive_base = Signal(SQDrive)
        m.d.comb += drive_base.as_value().eq(self.drive << 2)
        drive_sum = drive_base + cv[2]
        drive_val = Signal(SQDrive)
        m.d.comb += drive_val.eq(drive_sum.saturate(SQDrive))

        # per-voice params
        for n in range(n_voices):
            # Note velocity scales the envelope (VCF + VCA). The mod wheel no longer
            # caps it (velocity_mod=False), otherwise CC1=0 would mute the VCA.
            vel_out = Signal(MultiADSR.EnvUQ, name=f"vel_out_{n}")
            m.d.comb += vel_out.eq(fixed.UQ(0, 8)(voice_tracker.o[n].velocity_mod))
            m.d.comb += [
                self.voice_states[n].eq(voice_tracker.o[n]),
                self.voice_cutoffs[n].eq(voice_block.voice_cutoffs[n]),
                voice_block.voice_gates[n].eq(voice_tracker.o[n].gate),
                voice_block.voice_freq_incs[n].eq(voice_tracker.o[n].freq_inc),
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
            voice_block.wt_write_addr.eq(self.wt_write_addr),
            voice_block.wt_write_data.eq(self.wt_write_data),
            voice_block.wt_write_en.eq(self.wt_write_en),
        ]

        # Output diffuser

        m.submodules.diffuser = diffuser = Diffuser()
        self.diffuser = diffuser

        # Voice stereo output -> expand to 4-ch for diffuser

        m.submodules.vb_merge4 = vb_merge4 = dsp.Merge(n_channels=4, sink=diffuser.i)
        vb_merge4.wire_valid(m, [0, 1])

        m.submodules.vb_split2 = vb_split2 = dsp.Split(n_channels=2, source=voice_block.o)
        wiring.connect(m, vb_split2.o[0], vb_merge4.i[2])
        wiring.connect(m, vb_split2.o[1], vb_merge4.i[3])

        # Stereo distortion after diffuser

        m.submodules.diffuser_split4 = diffuser_split4 = dsp.Split(
                n_channels=4, source=diffuser.o)
        diffuser_split4.wire_ready(m, [0, 1])

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

            dsp.connect_remap(m, diffuser_split4.o[2+lr], vca.i, lambda o, i : [
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
        value: csr.Field(csr.action.W, unsigned(16))

    class DecayRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class SustainLevel(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class ReleaseRate(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class Lfo(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class WavetableAddr(csr.Register, access="w"):
        value: csr.Field(csr.action.W, unsigned(16))

    class WavetableData(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class EnvAmt(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class CutoffBase(csr.Register, access="w"):
        value: csr.Field(csr.action.W, signed(16))

    class Voice(csr.Register, access="r"):
        note:   csr.Field(csr.action.R, unsigned(8))
        cutoff: csr.Field(csr.action.R, unsigned(8))

    class Matrix(csr.Register, access="w"):
        """Mixing matrix coefficient: commit on write strobe, MatrixBusy set until done."""
        o_x:   csr.Field(csr.action.W, unsigned(4))
        i_y:   csr.Field(csr.action.W, unsigned(4))
        value: csr.Field(csr.action.W, signed(24))

    class MatrixBusy(csr.Register, access="r"):
        busy: csr.Field(csr.action.R, unsigned(1))

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
        regs = csr.Builder(addr_width=7, data_width=8)
        voices_csr_end = 0x8+PolySynth.N_VOICES*4
        self._drive         = regs.add("drive",         self.Drive(),         offset=0x0)
        self._reso          = regs.add("reso",          self.Reso(),          offset=0x4)
        self._voices        = [regs.add(f"voices{i}",   self.Voice(),
                               offset=0x8+i*4) for i in range(PolySynth.N_VOICES)]
        self._matrix        = regs.add("matrix",        self.Matrix(),        offset=voices_csr_end + 0x0)
        self._matrix_busy   = regs.add("matrix_busy",   self.MatrixBusy(),    offset=voices_csr_end + 0x4)
        self._midi_write    = regs.add("midi_write",    self.MidiWrite(),     offset=voices_csr_end + 0x8)
        self._midi_read     = regs.add("midi_read",     self.MidiRead(),      offset=voices_csr_end + 0xC)
        self._midi_host     = regs.add("usb_midi_host", self.UsbMidiHost(),   offset=voices_csr_end + 0x10)
        self._midi_cfg      = regs.add("usb_midi_cfg",  self.UsbMidiCfg(),    offset=voices_csr_end + 0x14)
        self._midi_endp     = regs.add("usb_midi_endp", self.UsbMidiCfg(),    offset=voices_csr_end + 0x18)
        self._attack_rate   = regs.add("attack_rate",   self.AttackRate(),    offset=voices_csr_end + 0x1C)
        self._decay_rate    = regs.add("decay_rate",    self.DecayRate(),     offset=voices_csr_end + 0x20)
        self._sustain_level = regs.add("sustain_level", self.SustainLevel(),  offset=voices_csr_end + 0x24)
        self._release_rate  = regs.add("release_rate",  self.ReleaseRate(),   offset=voices_csr_end + 0x28)
        self._wt_addr       = regs.add("wt_addr",       self.WavetableAddr(), offset=voices_csr_end + 0x2C)
        self._wt_data       = regs.add("wt_data",       self.WavetableData(), offset=voices_csr_end + 0x30)
        self._lfo           = regs.add("lfo",           self.Lfo(),           offset=voices_csr_end + 0x34)
        self._midi_ch_filt  = regs.add("midi_ch_filter",self.MidiChannelFilter(), offset=voices_csr_end + 0x38)
        self._env_amt       = regs.add("env_amt",       self.EnvAmt(),        offset=voices_csr_end + 0x3C)
        self._cutoff_base   = [regs.add(f"cutoff_base{i}", self.CutoffBase(),
                               offset=voices_csr_end + 0x40 + i*4) for i in range(PolySynth.N_VOICES)]
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

        with m.If(self._lfo.f.value.w_stb):
            m.d.sync += self.synth.lfo.as_value().eq(self._lfo.f.value.w_data)

        # Wavetable write: latch address on addr write, commit on data write
        wt_addr_reg = Signal(9)
        with m.If(self._wt_addr.f.value.w_stb):
            m.d.sync += wt_addr_reg.eq(self._wt_addr.f.value.w_data[:9])
        m.d.comb += [
            self.synth.wt_write_addr.eq(wt_addr_reg),
            self.synth.wt_write_data.eq(self._wt_data.f.value.w_data),
            self.synth.wt_write_en.eq(self._wt_data.f.value.w_stb),
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

        # matrix coefficient update logic
        matrix_busy = Signal()
        m.d.comb += self._matrix_busy.f.busy.r_data.eq(matrix_busy)
        with m.If(self._matrix.element.w_stb & ~matrix_busy):
            m.d.sync += [
                matrix_busy.eq(1),
                self.synth.diffuser.matrix.c.payload.o_x         .eq(self._matrix.f.o_x.w_data),
                self.synth.diffuser.matrix.c.payload.i_y         .eq(self._matrix.f.i_y.w_data),
                self.synth.diffuser.matrix.c.payload.v.as_value().eq(self._matrix.f.value.w_data),
                self.synth.diffuser.matrix.c.valid.eq(1),
            ]
        with m.If(matrix_busy & self.synth.diffuser.matrix.c.ready):
            # coefficient has been written
            m.d.sync += [
                matrix_busy.eq(0),
                self.synth.diffuser.matrix.c.valid.eq(0),
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
        io_left=['phase cv', 'filter cv', 'drive cv', '', '', 'clk', 'out L', 'out R'],
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
