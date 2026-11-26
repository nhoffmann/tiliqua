# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
``JunoCore``: the Juno voice engine plus its global LFO, wired the same way for
the hardware top level (``top.py``) and the offline renderer (``sim_render.py``).

- ``JunoVoice``: 6 voices of DCO -> ADSR -> ladder + VCA -> mono sum.
- ``JunoFx``: 4-position HPF + BBD-style stereo chorus on the mono sum.
- ``JunoLfo``: one triangle LFO with delay, ticked once per output frame
  (each handshake on ``o``), routed to:
  - pitch: ``voice_pitch = glide(voice_pitch_base + detune) + bend + vib + range``
    (``JunoGlide`` portamento per voice; bend, vibrato, DCO range immediate),
  - filter: ``cutoff_mod`` (normalized octaves),
  - pulse width: ``pw_point`` (manual or LFO PWM).
"""

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from tiliqua.dsp import ASQ

# top.py runs as a script (sibling imports); tests / sim_render import the package.
try:
    from juno_voice import JunoVoice, MultiADSR, PipelinedMAC, PitchToInc
    from juno_lfo import JunoLfo
    from juno_fx import JunoFx
    from juno_glide import JunoGlide, COEF_BITS as GLIDE_COEF_BITS, DETUNE_BITS
except ImportError:
    from top.juno.juno_voice import JunoVoice, MultiADSR, PipelinedMAC, PitchToInc
    from top.juno.juno_lfo import JunoLfo
    from top.juno.juno_fx import JunoFx
    from top.juno.juno_glide import JunoGlide, COEF_BITS as GLIDE_COEF_BITS, DETUNE_BITS


class JunoCore(wiring.Component):

    def __init__(self, n_voices, sq=ASQ):
        self.n_voices = n_voices
        self.sq = sq
        n = n_voices
        super().__init__({
            "o": Out(stream.Signature(data.ArrayLayout(sq, 2))),

            # per voice (from the tracker): note pitch (glide target), gate, velocity
            "voice_pitch_base": In(data.ArrayLayout(PitchToInc.PitchSQ, n)),
            "voice_gates":      In(data.ArrayLayout(unsigned(1), n)),
            "voice_velocity":   In(data.ArrayLayout(MultiADSR.EnvUQ, n)),
            "voice_cutoff_base": In(data.ArrayLayout(ASQ, n)),

            # global pitch: bend (PitchSQ raw, added after the glide), portamento
            "pitch_bend": In(PitchToInc.PitchSQ),
            "glide_coef": In(unsigned(GLIDE_COEF_BITS)),   # 0 = off (juno_glide)
            # per-voice pitch offset, 1/1024 semitones (variance + unison spread)
            "voice_detune": In(data.ArrayLayout(signed(DETUNE_BITS), n)),

            # DCO
            "dco_range":  In(2),              # 0 = 16' (-1 oct), 1 = 8', 2 = 4' (+1 oct)
            "saw_en":     In(1),
            "pulse_en":   In(1),
            "pw_knob":    In(unsigned(16)),   # 0..32768
            "pwm_lfo":    In(1),
            "sub_gain":   In(unsigned(16)),
            "noise_gain": In(unsigned(16)),

            # VCF / VCA / ENV
            "reso":          In(unsigned(16)),
            "env_amt":       In(ASQ),
            "cutoff_cv":     In(ASQ),
            "vca_gate_mode": In(1),
            # Envelope (Juno-106 style, see adsr_model): attack increment (32 bit),
            # decay / release eps (17 bit, 19 fractional bits), sustain UQ(0,16)
            "attack_rate":   In(unsigned(32)),
            "decay_rate":    In(unsigned(17)),
            "sustain_level": In(MultiADSR.EnvUQ),
            "release_rate":  In(unsigned(17)),

            # HPF / chorus (JunoFx)
            "hpf_mode":      In(2),
            "chorus_mode":   In(2),

            # LFO
            "lfo_rate_inc":   In(unsigned(32)),
            "lfo_delay_step": In(unsigned(32)),
            "lfo_dco_depth":  In(unsigned(16)),
            "lfo_vcf_depth":  In(unsigned(16)),

            # status
            "voice_active":  Out(data.ArrayLayout(unsigned(1), n)),
            "voice_cutoffs": Out(data.ArrayLayout(unsigned(8), n)),
            "lfo":           Out(signed(16)),
        })

    def elaborate(self, platform):
        m = Module()
        N = self.n_voices

        m.submodules.voice = voice = JunoVoice(n_voices=N, sq=self.sq)
        m.submodules.lfo = lfo = JunoLfo(PipelinedMAC())
        m.submodules.fx = fx = JunoFx(PipelinedMAC())
        m.submodules.glide = glide = JunoGlide(N)
        m.d.comb += [glide.tick.eq(self.o.valid & self.o.ready), glide.coef.eq(self.glide_coef)]
        m.d.comb += [fx.hpf_mode.eq(self.hpf_mode), fx.chorus_mode.eq(self.chorus_mode)]

        # LFO: one update per output frame
        any_gate = Signal()
        m.d.comb += [
            any_gate.eq(Cat(*[self.voice_gates[n] for n in range(N)]).any()),
            lfo.tick.eq(self.o.valid & self.o.ready),
            lfo.rate_inc.eq(self.lfo_rate_inc),
            lfo.delay_step.eq(self.lfo_delay_step),
            lfo.any_gate.eq(any_gate),
            lfo.dco_depth.eq(self.lfo_dco_depth),
            lfo.vcf_depth.eq(self.lfo_vcf_depth),
            lfo.pw_knob.eq(self.pw_knob),
            lfo.pwm_lfo.eq(self.pwm_lfo),
            self.lfo.eq(lfo.lfo),
        ]

        # DCO range: -12 / 0 / +12 semitones (PitchSQ raw = semitones * 1024)
        range_off = Signal(signed(16))
        pitch_off = Signal(signed(PitchToInc.PitchSQ.as_shape().width + 1))
        m.d.comb += range_off.eq(Mux(self.dco_range == 0, -12 << PitchToInc.FINE_BITS,
                                 Mux(self.dco_range == 2, 12 << PitchToInc.FINE_BITS, 0)))
        m.d.comb += pitch_off.eq(self.pitch_bend.as_value() + lfo.vib + range_off)

        for n in range(N):
            pitch = Signal(PitchToInc.PitchSQ, name=f"pitch_{n}")
            m.d.comb += [
                glide.target[n].eq(self.voice_pitch_base[n]),
                glide.detune[n].eq(self.voice_detune[n]),
                pitch.as_value().eq(glide.pitch[n].as_value() + pitch_off),
                voice.voice_pitch[n].eq(pitch),
                voice.voice_gates[n].eq(self.voice_gates[n]),
                voice.voice_velocity[n].eq(self.voice_velocity[n]),
                voice.voice_cutoff_base[n].eq(self.voice_cutoff_base[n]),
                self.voice_active[n].eq(voice.voice_active[n]),
                self.voice_cutoffs[n].eq(voice.voice_cutoffs[n]),
            ]

        m.d.comb += [
            voice.saw_en.eq(self.saw_en),
            voice.pulse_en.eq(self.pulse_en),
            voice.pw_point.eq(lfo.pw_point),
            voice.sub_gain.eq(self.sub_gain),
            voice.noise_gain.eq(self.noise_gain),
            voice.reso.eq(self.reso),
            voice.env_amt.eq(self.env_amt),
            voice.cutoff_cv.eq(self.cutoff_cv),
            voice.cutoff_mod.as_value().eq(lfo.cutoff_mod),
            voice.vca_gate_mode.eq(self.vca_gate_mode),
            voice.attack_rate.eq(self.attack_rate),
            voice.decay_rate.eq(self.decay_rate),
            voice.sustain_level.eq(self.sustain_level),
            voice.release_rate.eq(self.release_rate),
        ]
        wiring.connect(m, voice.o, fx.i)
        wiring.connect(m, fx.o, wiring.flipped(self.o))

        return m
