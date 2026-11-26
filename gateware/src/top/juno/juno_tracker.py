# Copyright (c) 2024 S. Holzapfel <me@sebholzapfel.com>
#
# SPDX-License-Identifier: CERN-OHL-S-2.0
#

"""
MIDI voice tracking, allocation and culling for the Juno port.

Copied from ``tiliqua.midi.voice_tracker`` (upstream, unmodified) and reduced:

- No frequency output. The upstream tracker converts notes to a 16-bit linear
  ``freq_inc``, which is up to 25 cents out of tune in the bass. Here each voice only
  reports its note number, and a global ``pitch_bend`` (-1..1) is output. The Juno
  voice converts note + bend (+ later LFO / CV) to a phase increment in the log
  domain (``juno_voice.PitchToInc``).
- No mod wheel velocity capping (``velocity_mod`` option removed): with a VCA, CC1 = 0
  would mute the synth. ``velocity_on`` holds the note-on velocity (not changed by
  poly pressure).
- Unison (Phase 4.6, ``unison`` input): every NOTE_ON goes to all voices (last-note
  priority, no note stack). A new note while the voices are gated changes the pitch
  without retriggering the envelopes (legato; glides with portamento). NOTE_OFF of
  the sounding note releases all voices (the existing per-note cull).
"""

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from tiliqua.dsp import ASQ
from tiliqua.midi.types import *

class JunoVoiceState(data.Struct):
    note:         unsigned(8)
    velocity:     unsigned(8)
    gate:         unsigned(1)
    velocity_on:  unsigned(8)

class JunoVoiceTracker(wiring.Component):

    """
    Read a stream of MIDI messages. Decode it into :py:`max_voices` independent
    :py:`JunoVoiceState` registers, one per voice, with voice culling.

    After each :py:`NOTE_ON` event, a voice is selected, its :py:`JunoVoiceState.note` is set
    and the :py:`JunoVoiceState.gate` attribute is set to 1.

    Pitch bend is output globally on :py:`pitch_bend` (-1..1).

    After each :py:`NOTE_OFF` event, :py:`JunoVoiceState.gate` is set to 0. If :py:`zero_velocity_gate`
    is set, the velocity is also set to 0 (instead of the MIDI release velocity).

    Sustain pedal (CC64) is supported: while held, NOTE_OFF events mark voices as
    sustained instead of clearing their gates. When the pedal is released, all
    sustained voices have their gates cleared.
    """

    def __init__(self, max_voices=8, zero_velocity_gate=False):
        self.max_voices = max_voices
        self.zero_velocity_gate = zero_velocity_gate
        super().__init__({
            "i": In(stream.Signature(MidiMessage)),
            "voice_active": In(data.ArrayLayout(unsigned(1), max_voices)),
            "unison": In(1),
            "o": Out(JunoVoiceState).array(max_voices),
            "pitch_bend": Out(ASQ),
        });

    def elaborate(self, platform):
        m = Module()

        # State captured on each incoming MIDI message

        msg = Signal(MidiMessage)      # last MIDI message

        # write index for NOTE_ON select + commit
        voice_ix_write = Signal(range(self.max_voices), init=0)

        # voice mask (binary 1 is for an occupied voice slot)
        voice_mask = Signal(self.max_voices)

        # sustain pedal (CC64) state
        sustain_held = Signal()
        sustain_mask = Signal(self.max_voices)

        # FSM to process incoming MIDI messages one at a time and update
        # internal memories based on these messagse.

        with m.FSM() as fsm:

            with m.State('WAIT-VALID'):
                m.d.comb += self.i.ready.eq(1),
                with m.If(self.i.valid):
                    m.d.sync += msg.eq(self.i.payload)
                    with m.Switch(self.i.payload.status.kind):
                        with m.Case(Status.Kind.NOTE_ON):
                            with m.If(self.i.payload.midi_payload.note_on.velocity == 0):
                                # According to the MIDI standard, a device may transmit a
                                # NOTE_ON with velocity=0, and this should be treated exactly
                                # the same as a note OFF.
                                m.next = 'NOTE-OFF'
                            with m.Elif(self.unison):
                                m.next = 'NOTE-ON-COMMIT'
                            with m.Else():
                                m.d.sync += voice_ix_write.eq(0)
                                m.next = 'NOTE-ON-MATCH'
                        with m.Case(Status.Kind.NOTE_OFF):
                            m.next = 'NOTE-OFF'
                        with m.Case(Status.Kind.CONTROL_CHANGE):
                            m.next = 'CONTROL-CHANGE'
                        with m.Case(Status.Kind.PITCH_BEND):
                            m.next = 'PITCH-BEND'
                        with m.Case(Status.Kind.POLY_PRESSURE):
                            m.next = 'POLY-PRESSURE'
                        with m.Default():
                            m.next = 'WAIT-VALID'

            with m.State('NOTE-ON-MATCH'):
                # prefer reusing a voice slot that already holds the same note,
                # so the ADSR retrigger lands on the correct envelope state.
                match_found = Signal()
                with m.Switch(voice_ix_write):
                    for n in range(self.max_voices):
                        with m.Case(n):
                            m.d.comb += match_found.eq(
                                self.o[n].note == msg.midi_payload.note_on.note)
                with m.If(match_found):
                    m.next = 'NOTE-ON-COMMIT'
                with m.Elif(voice_ix_write == self.max_voices - 1):
                    m.d.sync += voice_ix_write.eq(0)
                    m.next = 'NOTE-ON-SELECT-IDLE'
                with m.Else():
                    m.d.sync += voice_ix_write.eq(voice_ix_write + 1)

            with m.State('NOTE-ON-SELECT-IDLE'):
                # prefer a free slot whose ADSR has fully decayed
                slot_active = Signal()
                with m.Switch(voice_ix_write):
                    for n in range(self.max_voices):
                        with m.Case(n):
                            m.d.comb += slot_active.eq(self.voice_active[n])
                with m.If(~voice_mask.bit_select(voice_ix_write, 1) & ~slot_active):
                    m.next = 'NOTE-ON-COMMIT'
                with m.Elif(voice_ix_write == self.max_voices - 1):
                    m.d.sync += voice_ix_write.eq(0)
                    m.next = 'NOTE-ON-SELECT'
                with m.Else():
                    m.d.sync += voice_ix_write.eq(voice_ix_write + 1)

            with m.State('NOTE-ON-SELECT'):
                # find an empty note slot to write to
                with m.If(~voice_mask.bit_select(voice_ix_write, 1)):
                    m.next = 'NOTE-ON-COMMIT'
                with m.Else():
                    m.d.sync += voice_ix_write.eq(voice_ix_write + 1)
                    with m.If(voice_ix_write == self.max_voices - 1):
                        # no free note slots
                        m.next = 'WAIT-VALID'

            with m.State('NOTE-ON-COMMIT'):
                # commit the new note to the found slot (unison: to all slots)
                for n in range(self.max_voices):
                    with m.If(self.unison | (voice_ix_write == n)):
                        m.d.sync += [
                            voice_mask.bit_select(n, 1).eq(1),
                            sustain_mask.bit_select(n, 1).eq(0),
                            self.o[n].note.eq(msg.midi_payload.note_on.note),
                            self.o[n].velocity.eq(msg.midi_payload.note_on.velocity),
                            self.o[n].gate.eq(1),
                            self.o[n].velocity_on.eq(msg.midi_payload.note_on.velocity),
                        ]
                m.next = 'WAIT-VALID'

            with m.State('NOTE-OFF'):
                # cull any voice that matches the MIDI payload note #
                for n in range(self.max_voices):
                    with m.If(self.o[n].note == msg.midi_payload.note_off.note):
                        with m.If(sustain_held):
                            # pedal held: keep gate, mark for deferred release
                            m.d.sync += sustain_mask.bit_select(n, 1).eq(1)
                        with m.Else():
                            m.d.sync += [
                                voice_mask.bit_select(n, 1).eq(0),
                                self.o[n].gate.eq(0),
                            ]
                            if self.zero_velocity_gate:
                                m.d.sync += self.o[n].velocity.eq(0)
                m.next = 'WAIT-VALID'

            with m.State('POLY-PRESSURE'):
                # update any voice that matches the MIDI payload note #
                # TODO: rather than piggybacking on velocity, this should probably be its own field?
                for n in range(self.max_voices):
                    with m.If((self.o[n].note == msg.midi_payload.poly_pressure.note) & self.o[n].gate):
                        m.d.sync += self.o[n].velocity.eq(msg.midi_payload.poly_pressure.pressure)
                m.next = 'WAIT-VALID'

            with m.State('CONTROL-CHANGE'):
                with m.If(msg.midi_payload.control_change.controller_number == 64):
                    # sustain pedal
                    with m.If(msg.midi_payload.control_change.data >= 64):
                        m.d.sync += sustain_held.eq(1)
                    with m.Else():
                        m.d.sync += sustain_held.eq(0)
                        # release all sustained voices
                        for n in range(self.max_voices):
                            with m.If(sustain_mask.bit_select(n, 1)):
                                m.d.sync += [
                                    voice_mask.bit_select(n, 1).eq(0),
                                    sustain_mask.bit_select(n, 1).eq(0),
                                    self.o[n].gate.eq(0),
                                ]
                                if self.zero_velocity_gate:
                                    m.d.sync += self.o[n].velocity.eq(0)
                with m.If(msg.midi_payload.control_change.controller_number == 123):
                    # all stop
                    for n in range(self.max_voices):
                        m.d.sync += self.o[n].gate.eq(0)
                        if self.zero_velocity_gate:
                            m.d.sync += self.o[n].velocity.eq(0)
                    m.d.sync += sustain_held.eq(0)
                    m.d.sync += sustain_mask.eq(0)
                m.next = 'WAIT-VALID'

            with m.State('PITCH-BEND'):
                # convert 14-bit pitch bend to 16-bit signed ASQ -1 .. 1
                pb = Signal(signed(16))
                m.d.comb += pb.eq(Cat(msg.midi_payload.pitch_bend.lsb,
                                      msg.midi_payload.pitch_bend.msb))
                m.d.sync += self.pitch_bend.as_value().eq(pb-(2*8192))
                m.next = 'WAIT-VALID'

        return m
