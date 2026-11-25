use opts::*;
use strum_macros::{EnumIter, IntoStaticStr};
use serde_derive::{Serialize, Deserialize};

#[derive(Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "SCREAMING-KEBAB-CASE")]
pub enum Page {
    #[default]
    Help,
    Dco,
    Vcf,
    Adsr,
    Effect,
    Misc,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumIter, IntoStaticStr, Serialize, Deserialize)]
#[strum(serialize_all = "kebab-case")]
pub enum Switch {
    Off,
    #[default]
    On,
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
int_params!(DriveParams<u16>    { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(LevelParams<u16>      { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(PwParams<u16>         { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(CutoffParams<u16>     { step: 512, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(EnvAmtParams<u16>     { step: 1024, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(KbdTrackParams<u16>   { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(ResoParams<u16>       { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(DiffuseParams<u16>    { step: 2048, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(AdsrTimeParams<u16>  { step: 1024, min: 0, max: 16384, format: IntFormat::Scaled { divisor: 16384, precision: 2, suffix: "" } });
int_params!(AdsrLevelParams<u16> { step: 1024, min: 0, max: 32768, format: IntFormat::Scaled { divisor: 32768, precision: 2, suffix: "" } });
int_params!(ScrollParams<u8>      { step: 1, min: 0, max: 125 });

button_params!(OneShotButtonParams { mode: ButtonMode::OneShot });

#[derive(OptionPage, Clone)]
pub struct HelpOpts {
    #[option(0)]
    pub scroll: IntOption<ScrollParams>,
}

// Juno DCO mixer (MultiJunoOsc): saw / pulse on-off, pulse width,
// sub-oscillator and noise levels. All live (no table rewrite).
#[derive(OptionPage, Clone)]
pub struct DcoOpts {
    #[option(Switch::On)]
    pub saw: EnumOption<Switch>,
    #[option(Switch::Off)]
    pub pulse: EnumOption<Switch>,
    #[option(0)]
    pub pw: IntOption<PwParams>,
    #[option(0)]
    pub sub: IntOption<LevelParams>,
    #[option(0)]
    pub noise: IntOption<LevelParams>,
}

// Juno-style VCF: exponential cutoff (0..1 = 10 octaves, 12 Hz .. 12 kHz).
#[derive(OptionPage, Clone)]
pub struct VcfOpts {
    #[option(11469)]
    pub cutoff: IntOption<CutoffParams>,
    #[option(16384)]
    pub reso: IntOption<ResoParams>,
    #[option(19661)]
    pub env_amt: IntOption<EnvAmtParams>,
    #[option]
    pub env_pol: EnumOption<EnvPolarity>,
    #[option(16384)]
    pub kbd_track: IntOption<KbdTrackParams>,
}

#[derive(OptionPage, Clone)]
pub struct EffectOpts {
    #[option(8192)]
    pub drive: IntOption<DriveParams>,
    #[option(12288)]
    pub diffuse: IntOption<DiffuseParams>,
}

#[derive(OptionPage, Clone)]
pub struct AdsrOpts {
    #[option(0)]
    pub attack: IntOption<AdsrTimeParams>,
    #[option(4096)]
    pub decay: IntOption<AdsrTimeParams>,
    #[option(16384)]
    pub sustain: IntOption<AdsrLevelParams>,
    #[option(4096)]
    pub release: IntOption<AdsrTimeParams>,
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
    #[page(Page::Dco)]
    pub dco: DcoOpts,
    #[page(Page::Vcf)]
    pub vcf: VcfOpts,
    #[page(Page::Adsr)]
    pub adsr: AdsrOpts,
    #[page(Page::Effect)]
    pub effect: EffectOpts,
    #[page(Page::Misc)]
    pub misc: MiscOpts,
}
