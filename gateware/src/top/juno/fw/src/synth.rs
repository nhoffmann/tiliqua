use crate::pac;
use pac::constants::N_VOICES;

#[derive(Debug)]
pub struct Synth {
    registers: pac::SYNTH_PERIPH,
}

impl Synth {
    pub fn new(registers: pac::SYNTH_PERIPH) -> Self {
        Self { registers }
    }

    pub fn voice_notes(&self) -> [u8; N_VOICES] {
        [
            self.registers.voices0().read().note().bits(),
            self.registers.voices1().read().note().bits(),
            self.registers.voices2().read().note().bits(),
            self.registers.voices3().read().note().bits(),
            self.registers.voices4().read().note().bits(),
            self.registers.voices5().read().note().bits(),
        ]
    }

    pub fn voice_cutoffs(&self) -> [u8; N_VOICES] {
        [
            self.registers.voices0().read().cutoff().bits(),
            self.registers.voices1().read().cutoff().bits(),
            self.registers.voices2().read().cutoff().bits(),
            self.registers.voices3().read().cutoff().bits(),
            self.registers.voices4().read().cutoff().bits(),
            self.registers.voices5().read().cutoff().bits(),
        ]
    }

    /// HPF position (0 boost, 1 flat, 2 240 Hz, 3 720 Hz) and chorus (0 off, 1 I, 2 II, 3 I+II).
    pub fn set_fx(&mut self, hpf: u8, chorus: u8)  {
        self.registers.fx().write(|w| unsafe { w.hpf().bits(hpf).chorus().bits(chorus) } );
    }

    /// Output pre-gain (level x drive) * 8192 (gateware: `drive << 2`, SQ(3,15)).
    pub fn set_drive(&mut self, value: u16)  {
        self.registers.drive().write(|w| unsafe { w.value().bits(value) } );
    }

    /// Output post-gain, UQ(0,16).
    pub fn set_out_gain(&mut self, value: u16)  {
        self.registers.out_gain().write(|w| unsafe { w.value().bits(value) } );
    }

    pub fn set_reso(&mut self, value: u16)  {
        self.registers.reso().write(|w| unsafe { w.value().bits(value) } );
    }

    /// Envelope -> cutoff amount, bipolar (i16: -32768..32767 = -1.0..1.0).
    pub fn set_env_amt(&mut self, value: i16)  {
        self.registers.env_amt().write(|w| unsafe { w.value().bits(value as u16) } );
    }

    /// Per-voice cutoff base in normalized octaves (i16: 32768 = 1.0 = 10 octaves).
    pub fn set_cutoff_base(&mut self, voice: usize, value: i16)  {
        let v = value as u16;
        match voice {
            0 => { self.registers.cutoff_base0().write(|w| unsafe { w.value().bits(v) } ); }
            1 => { self.registers.cutoff_base1().write(|w| unsafe { w.value().bits(v) } ); }
            2 => { self.registers.cutoff_base2().write(|w| unsafe { w.value().bits(v) } ); }
            3 => { self.registers.cutoff_base3().write(|w| unsafe { w.value().bits(v) } ); }
            4 => { self.registers.cutoff_base4().write(|w| unsafe { w.value().bits(v) } ); }
            5 => { self.registers.cutoff_base5().write(|w| unsafe { w.value().bits(v) } ); }
            _ => {}
        }
    }

    pub fn set_attack_rate(&mut self, value: u32)  {
        self.registers.attack_rate().write(|w| unsafe { w.value().bits(value) } );
    }

    pub fn set_decay_rate(&mut self, value: u32)  {
        self.registers.decay_rate().write(|w| unsafe { w.value().bits(value) } );
    }

    pub fn set_sustain_level(&mut self, value: u16)  {
        self.registers.sustain_level().write(|w| unsafe { w.value().bits(value) } );
    }

    pub fn set_release_rate(&mut self, value: u32)  {
        self.registers.release_rate().write(|w| unsafe { w.value().bits(value) } );
    }

    /// Juno DCO mixer switches.
    /// DCO switches, VCA mode and DCO range (0 = 16', 1 = 8', 2 = 4').
    pub fn set_dco(&mut self, saw: bool, pulse: bool, pwm_lfo: bool, vca_gate: bool, range: u8)  {
        self.registers.dco().write(|w| unsafe { w.saw().bit(saw).pulse().bit(pulse)
                                         .pwm_lfo().bit(pwm_lfo).vca_gate().bit(vca_gate)
                                         .range().bits(range) } );
    }

    /// PW knob 0..32768 (gateware maps it through the curve LUT, incl. LFO PWM).
    pub fn set_pw_knob(&mut self, value: u16)  {
        self.registers.pw_knob().write(|w| unsafe { w.value().bits(value) } );
    }

    /// LFO phase increment per audio frame, UQ(0,32).
    pub fn set_lfo_rate(&mut self, inc: u32)  {
        self.registers.lfo_rate().write(|w| unsafe { w.value().bits(inc) } );
    }

    /// LFO delay: amplitude step per frame, 2**30 = immediate.
    pub fn set_lfo_delay(&mut self, step: u32)  {
        self.registers.lfo_delay().write(|w| unsafe { w.value().bits(step) } );
    }

    /// Unison mode + one voice's pitch offset (1/1024 semitones, 10 bit signed).
    pub fn set_voice_cfg(&mut self, voice: usize, detune: i16, unison: bool, bend_range: u8)  {
        let d = (detune as u16) & 0x3FF;
        self.registers.voice_cfg().write(|w| unsafe {
            w.detune().bits(d).voice().bits(voice as u8).unison().bit(unison)
             .bend_range().bits(bend_range.min(12)) } );
    }

    /// CV routing: pitch / cutoff source jack (0 off, 1..4), CV out sources.
    pub fn set_cv_route(&mut self, pitch_src: u8, cutoff_src: u8, out0: u8, out1: u8)  {
        self.registers.cv_route().write(|w| unsafe {
            w.pitch_src().bits(pitch_src).cutoff_src().bits(cutoff_src)
             .out0_src().bits(out0).out1_src().bits(out1) } );
    }

    /// CV amounts for the audio-rate routes, SQ(2,14) (16384 = 1V/oct).
    pub fn set_cv_amt(&mut self, pitch: i16, cutoff: i16)  {
        self.registers.cv_amt().write(|w| unsafe {
            w.pitch().bits(pitch as u16).cutoff().bits(cutoff as u16) } );
    }

    /// Current MIDI pitch bend, -32768..32767 (= -1..1).
    pub fn pitch_bend(&self) -> i16 {
        self.registers.bend().read().value().bits() as i16
    }

    /// Portamento coefficient (`juno_glide`), 0 = off.
    pub fn set_glide(&mut self, coef: u32)  {
        self.registers.glide().write(|w| unsafe { w.coef().bits(coef) } );
    }

    /// LFO depths: pitch (1/1024 semitones), cutoff (normalized octaves, 15 bit).
    pub fn set_lfo_depths(&mut self, dco: u16, vcf: u16)  {
        self.registers.lfo_depths().write(|w| unsafe { w.dco().bits(dco).vcf().bits(vcf) } );
    }

    /// Sub-oscillator and noise gains, UQ(0,16).
    pub fn set_mix_levels(&mut self, sub: u16, noise: u16)  {
        self.registers.mix_levels().write(|w| unsafe { w.sub().bits(sub).noise().bits(noise) } );
    }

    pub fn midi_write(&mut self, value: u32)  {
        self.registers.midi_write().write(|w| unsafe { w.msg().bits(value) } );
    }

    pub fn midi_read(&mut self) -> u32  {
        self.registers.midi_read().read().bits()
    }

    pub fn usb_host_vbus(&mut self, enable: bool)  {
        self.registers.usb_midi_host().write(|w| w.host().bit(enable) );
    }

    pub fn usb_host_midi(&mut self, cfg_id: u8, endpt_id: u8)  {
        self.registers.usb_midi_cfg().write(|w| unsafe { w.value().bits(cfg_id) } );
        self.registers.usb_midi_endp().write(|w| unsafe { w.value().bits(endpt_id) } );
    }

    // `None` listens on all channels.
    pub fn set_midi_channel_filter(&mut self, channel: Option<u8>)  {
        let value = channel.unwrap_or(0);
        self.registers.midi_ch_filter().write(|w| unsafe { w.value().bits(value) } );
    }
}
