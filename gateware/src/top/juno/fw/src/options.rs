use opts::*;
use strum_macros::{EnumIter, IntoStaticStr};
use serde_derive::{Serialize, Deserialize};

#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "SCREAMING-KEBAB-CASE")]
pub enum Page {
    #[default]
    Help,
    Voice,
    Lfo,
    Dco,
    Vcf,
    Env,
    Fx,
    Cv,
    Misc,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum Switch {
    Off,
    #[default]
    On,
}

// Juno DCO range (octave): 16' = one octave down, 8' = normal, 4' = one octave up.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum DcoRange {
    #[strum(serialize = "16'")]
    Ft16 = 0,
    #[default]
    #[strum(serialize = "8'")]
    Ft8 = 1,
    #[strum(serialize = "4'")]
    Ft4 = 2,
}

// Juno HPF positions (JunoFx): bass boost, flat, 240 Hz, 720 Hz.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum HpfMode {
    Boost = 0,
    #[default]
    Flat = 1,
    #[strum(serialize = "240hz")]
    Hz240 = 2,
    #[strum(serialize = "720hz")]
    Hz720 = 3,
}

// Juno chorus (JunoFx): off, I, II, I+II.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum ChorusMode {
    #[default]
    Off = 0,
    #[strum(serialize = "I")]
    I = 1,
    #[strum(serialize = "II")]
    II = 2,
    #[strum(serialize = "I+II")]
    Both = 3,
}

// CV input targets (fw cv.rs). Pitch / cutoff: audio rate in gateware (1V/oct at
// amount 1.0); the rest: firmware knob offsets (+-5 V x amount 1.0 = full range).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum CvTarget {
    #[default]
    Off,
    Pitch,
    Cutoff,
    Pw,
    Reso,
    EnvAmt,
    LfoRate,
    Vibrato,
    VcfLfo,
    Porta,
    Level,
    Drive,
}

// CV output sources (juno_cv.JunoCvOut; values = gateware source index).
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum CvOutSrc {
    Off = 0,
    #[default]
    Lfo = 1,
    Env = 2,
    Gate = 3,
    Pitch = 4,
}

// Key assign: poly (6 voices) or unison (all voices on the last note, legato).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum AssignMode {
    #[default]
    Poly,
    Unison,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum PwmMode {
    #[default]
    Man,
    Lfo,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum VcaMode {
    #[default]
    Env,
    Gate,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum EnvPolarity {
    #[default]
    Pos,
    Neg,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum UsbHost {
    #[default]
    Off,
    On,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum UsbMidiSerialDebug {
    #[default]
    Off,
    On,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum CcHighlight {
    Off,
    #[default]
    On,
}

#[repr(u8)]
#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum MidiChannel {
    #[default]
    All,
    Ch1,  Ch2,  Ch3,  Ch4,
    Ch5,  Ch6,  Ch7,  Ch8,
    Ch9,  Ch10, Ch11, Ch12,
    Ch13, Ch14, Ch15, Ch16,
}

impl MidiChannel {
    pub fn to_filter(self) -> Option<u8> {
        match self {
            Self::All => None,
            _ => Some(self as u8),
        }
    }
}

int_params!(PageNumParams<u16>    { step: 1, min: 0, max: 0 });
int_params!(BendParams<u16>       { step: 1, min: 0, max: 12 });
int_params!(CvAmtParams<i16>      { step: 512, min: -16384, max: 16384, format: IntFormat::Scaled { divisor: 16384, precision: 2, suffix: "" } });
int_params!(DriveParams<u16>    { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(LevelParams<u16>      { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(PwParams<u16>         { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(CutoffParams<u16>     { step: 512, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(EnvAmtParams<u16>     { step: 1024, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(KbdTrackParams<u16>   { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(ResoParams<u16>       { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
// ADSR times: knob = Juno-106 slider / 127 (fw env.rs maps it to KR-106-measured times)
int_params!(AdsrTimeParams<u16>  { step: 1024, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(AdsrLevelParams<u16> { step: 1024, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(ScrollParams<u8>      { step: 1, min: 0, max: 125 });

button_params!(OneShotButtonParams { mode: ButtonMode::OneShot });

#[derive(OptionPage, Clone)]
pub struct HelpOpts {
    #[option(0)]
    pub scroll: IntOption<ScrollParams>,
}

// Voice / keyboard controls (Juno-106 left-hand panel): key assign (poly /
// unison), portamento (per-voice exponential glide on the note pitch, juno_glide;
// knob^3 * 3 s, 0 = off), unison detune spread (0 .. +-25 cents) and per-voice
// variance (1.0 = KR-106 tolerances: pitch +-3 cents, cutoff +-5 %; fw variance.rs).
// Performance (Juno-106 bender section): bend range (semitones), bender -> VCF
// (0 .. +-2 octaves), mod wheel -> vibrato depth (the Juno-106 LFO lever; same
// curve as the DCO LFO knob, added to it).
#[derive(OptionPage, Clone)]
pub struct VoiceOpts {
    #[option]
    pub assign: EnumOption<AssignMode>,
    #[option(0)]
    pub porta: IntOption<LevelParams>,
    #[option(0)]
    pub detune: IntOption<LevelParams>,
    #[option(32768)]
    pub variance: IntOption<LevelParams>,
    #[option(2)]
    pub bend: IntOption<BendParams>,
    #[option(0)]
    pub bend_vcf: IntOption<LevelParams>,
    #[option(16384)]
    pub wheel: IntOption<LevelParams>,
}

// Juno LFO (JunoLfo): one global triangle LFO with delay (fade-in after a new
// note). Knobs = Juno-106 sliders, KR-106-measured curves (fw lfo.rs).
// Pages and option order follow the Juno-106 panel: LFO, DCO, VCF (incl. HPF),
// ENV (VCA mode + ADSR), FX (chorus, output level, drive).
#[derive(OptionPage, Clone)]
pub struct LfoOpts {
    #[option(21299)]
    pub rate: IntOption<LevelParams>,
    #[option(0)]
    pub delay: IntOption<LevelParams>,
}

// Juno DCO mixer (MultiJunoOsc): saw / pulse on-off, pulse width,
// sub-oscillator and noise levels. All live (no table rewrite).
#[derive(OptionPage, Clone)]
pub struct DcoOpts {
    #[option]
    pub range: EnumOption<DcoRange>,
    #[option(0)]
    pub lfo: IntOption<LevelParams>,
    #[option(0)]
    pub pw: IntOption<PwParams>,
    #[option]
    pub pwm_mode: EnumOption<PwmMode>,
    #[option(Switch::Off)]
    pub pulse: EnumOption<Switch>,
    #[option(Switch::On)]
    pub saw: EnumOption<Switch>,
    #[option(0)]
    pub sub: IntOption<LevelParams>,
    #[option(0)]
    pub noise: IntOption<LevelParams>,
}

// Juno-style VCF: exponential cutoff (20 Hz .. 20.5 kHz); the cutoff knob behaves like
// the Juno-106 VCF FREQ slider (fw vcf.rs). Keyboard tracking relative to C1.
#[derive(OptionPage, Clone)]
pub struct VcfOpts {
    #[option]
    pub hpf: EnumOption<HpfMode>,
    #[option(11469)]
    pub cutoff: IntOption<CutoffParams>,
    #[option(16384)]
    pub reso: IntOption<ResoParams>,
    #[option]
    pub env_pol: EnumOption<EnvPolarity>,
    #[option(19661)]
    pub env_amt: IntOption<EnvAmtParams>,
    #[option(0)]
    pub lfo: IntOption<LevelParams>,
    #[option(16384)]
    pub kbd_track: IntOption<KbdTrackParams>,
}

// Output stage (juno_output / fw out.rs): level (clean up to ~5 V, soft limiter
// above; default 0.45 = default 6-voice chord at ~5 V), drive (extra gain into the
// limiter with level compensation; 0 = off = clean Juno).
#[derive(OptionPage, Clone)]
pub struct EffectOpts {
    #[option]
    pub chorus: EnumOption<ChorusMode>,
    #[option(14746)]
    pub level: IntOption<LevelParams>,
    #[option(0)]
    pub drive: IntOption<DriveParams>,
}

#[derive(OptionPage, Clone)]
pub struct AdsrOpts {
    #[option]
    pub vca_mode: EnumOption<VcaMode>,
    #[option(0)]
    pub attack: IntOption<AdsrTimeParams>,
    #[option(8192)]
    pub decay: IntOption<AdsrTimeParams>,
    #[option(16384)]
    pub sustain: IntOption<AdsrLevelParams>,
    #[option(6554)]
    pub release: IntOption<AdsrTimeParams>,
}

// CV jacks: in0..in3 target + amount, out0 / out1 source.
#[derive(OptionPage, Clone)]
pub struct CvOpts {
    #[option(CvTarget::Pitch)]
    pub in0: EnumOption<CvTarget>,
    #[option(16384)]
    pub amt0: IntOption<CvAmtParams>,
    #[option(CvTarget::Cutoff)]
    pub in1: EnumOption<CvTarget>,
    #[option(16384)]
    pub amt1: IntOption<CvAmtParams>,
    #[option(CvTarget::Off)]
    pub in2: EnumOption<CvTarget>,
    #[option(16384)]
    pub amt2: IntOption<CvAmtParams>,
    #[option(CvTarget::Off)]
    pub in3: EnumOption<CvTarget>,
    #[option(16384)]
    pub amt3: IntOption<CvAmtParams>,
    #[option(CvOutSrc::Lfo)]
    pub out0: EnumOption<CvOutSrc>,
    #[option(CvOutSrc::Env)]
    pub out1: EnumOption<CvOutSrc>,
}

#[derive(OptionPage, Clone)]
pub struct MiscOpts {
    #[option]
    pub cc_highlight: EnumOption<CcHighlight>,
    #[option]
    pub midi_ch: EnumOption<MidiChannel>,
    #[option]
    pub usb_host: EnumOption<UsbHost>,
    #[option]
    pub serial_debug: EnumOption<UsbMidiSerialDebug>,
    #[option(false)]
    pub save_opts: ButtonOption<OneShotButtonParams>,
    #[option(false)]
    pub wipe_opts: ButtonOption<OneShotButtonParams>,
}

#[derive(Options, Clone)]
pub struct Opts {
    pub tracker: ScreenTracker<Page>,
    #[page(Page::Help)]
    pub help: HelpOpts,
    #[page(Page::Voice)]
    pub voice: VoiceOpts,
    #[page(Page::Lfo)]
    pub lfo: LfoOpts,
    #[page(Page::Dco)]
    pub dco: DcoOpts,
    #[page(Page::Vcf)]
    pub vcf: VcfOpts,
    #[page(Page::Env)]
    pub adsr: AdsrOpts,
    #[page(Page::Fx)]
    pub effect: EffectOpts,
    #[page(Page::Cv)]
    pub cv: CvOpts,
    #[page(Page::Misc)]
    pub misc: MiscOpts,
}
