#![no_std]
#![no_main]

use critical_section::Mutex;
use log::{info, error};
use riscv_rt::entry;
use irq::handler;
use core::cell::RefCell;
use embassy_sync::blocking_mutex::{Mutex as FlashMutex, raw::NoopRawMutex};

use micromath::F32Ext;
use midi_types::*;
use midi_convert::parse::MidiTryParseSlice;

use tiliqua_pac as pac;
use tiliqua_hal as hal;
use tiliqua_lib::*;
use tiliqua_lib::draw;
use tiliqua_lib::dsp::OnePoleSmoother;
use pac::constants::*;
use tiliqua_hal::persist::Persist;
use tiliqua_fw::*;
use tiliqua_fw::options::*;
use opts::{Options, OptionTrait};
use opts::cc_map::{MidiCcMapper, CcMapMode};

use tiliqua_hal::embedded_graphics::prelude::*;

use opts::persistence::*;
use hal::pca9635::Pca9635Driver;
use hal::tusb322::{TUSB322Driver, TUSB322Mode, AttachedState, AccessoryType};
use hal::i2c::{I2cBus, I2cDevice};
use static_cell::StaticCell;

use tiliqua_fw::dco;
use tiliqua_fw::lfo;
use tiliqua_fw::glide;
use tiliqua_fw::variance;
use tiliqua_fw::out;
use tiliqua_fw::cv;
use tiliqua_hal::pmod::EurorackPmod;
use tiliqua_fw::vcf;
use tiliqua_fw::env;
use tiliqua_fw::preset::{self, Action, PresetCtl, PresetStore};
use embassy_embedded_hal::flash::partition::BlockingPartition;

pub const TICK_PERIOD_MS: u32 = 5;

// Fixed menu appearance (BEAM page removed).
const HUE: u8 = 10;
const PERSIST: u8 = 15;
const PERSIST_HELP: u8 = 64;

// Mod wheel fully up opens the cutoff by 0.5 (5 octaves). Same as sim_render.py.
/// Bender -> VCF at full `bend-vcf`: 2 octaves (normalized octaves raw, 1 oct = 3276.8).
const BEND_VCF_RANGE: i32 = 6554;

/// Simulation only (`--features sim_pages`, used by `sim_ui.py`): the Verilator
/// SoC harness has no encoder, so step to the next menu page every
/// `SIM_PAGE_TICKS` and log it, one screenshot per page. Returns true on a change.
#[cfg(feature = "sim_pages")]
fn sim_step_pages(opts: &mut Opts) -> bool {
    use portable_atomic::{AtomicU32, Ordering};
    const SIM_PAGE_TICKS: u32 = 80; // 400 ms: previous page fades out; 10 pages fit the 5 s sim
    static TICKS: AtomicU32 = AtomicU32::new(0);
    let t = TICKS.fetch_add(1, Ordering::Relaxed) + 1;
    if t % SIM_PAGE_TICKS != 0 {
        return false;
    }
    let before = opts.page().value();
    opts.page_mut().tick_up();
    let after = opts.page().value();
    if after == before {
        return false;
    }
    info!("sim_pages: page {}", after.as_str());
    true
}

fn tick_handler(app: &Mutex<RefCell<App>>) {
    #[cfg(feature = "stack_watch")]
    tiliqua_fw::stack_watch::isr_enter();
    tick_body(app);
    #[cfg(feature = "stack_watch")]
    tiliqua_fw::stack_watch::isr_exit();
}

#[inline(never)]
fn tick_body(app: &Mutex<RefCell<App>>) {

    critical_section::with(|cs| {

        let mut app = app.borrow_ref_mut(cs);
        // Plain &mut App: lets the option reads below borrow `app.ui.opts` while
        // `app.synth` etc. are written (disjoint fields), instead of copying all
        // options (~0.9 KiB) onto the shared stack every tick.
        let app = &mut *app;

        //
        // Update UI and options
        //

        app.ui.update();
        app.uptime_ms = app.uptime_ms.wrapping_add(TICK_PERIOD_MS);
        #[cfg(feature = "sim_pages")]
        if sim_step_pages(&mut app.ui.opts) {
            app.ui.external_modify();
        }
        let serial_debug = app.ui.opts.misc.serial_debug.value == UsbMidiSerialDebug::On;
        let cc_highlight = app.ui.opts.misc.cc_highlight.value == CcHighlight::On;

        //
        // Check for TRS/USB MIDI traffic
        // (this is forwarded by the hardware to the synth
        //  for minimum possible latency, here we peek
        //  the FIFO contents for debugging purposes)
        //

        let mut last_cc_index = None;
        // Drain the whole read FIFO (depth 8) each tick, so fast CC streams
        // (e.g. mod wheel sweeps) aren't dropped.
        for _ in 0..8 {
            #[cfg(not(feature = "sim_stack"))]
            let midi_word = app.synth.midi_read();
            #[cfg(feature = "sim_stack")]
            let midi_word = match tiliqua_fw::sim_stack::midi_read(app.uptime_ms) {
                0 => app.synth.midi_read(),
                w => w,
            };
            if midi_word == 0 {
                break;
            }
            // Blink MIDI activity LED on TRS port
            app.ui.midi_activity();

            let bytes = [
                (midi_word & 0xFF) as u8,
                ((midi_word >> 8) & 0xFF) as u8,
                ((midi_word >> 16) & 0xFF) as u8,
            ];
            if let Ok(MidiMessage::ControlChange(_, cc, val)) = MidiMessage::try_parse_slice(&bytes) {
                // Mod wheel (CC1) opens the filter cutoff.
                let cc_num: u8 = cc.into();
                if cc_num == 1 {
                    app.mod_wheel = val.into();
                }
                if let Some(action) = app.cc_mapper.process(cc.into(), val.into()) {
                    last_cc_index = Some(action.global_index);
                    apply_cc_action(&mut app.ui.opts, &action);
                }
            }
            // Program change 0-31 -> preset A1..D8 (loaded by the main loop)
            if let Ok(MidiMessage::ProgramChange(_, prog)) = MidiMessage::try_parse_slice(&bytes) {
                let p: u8 = prog.into();
                if p < preset::N_SLOTS {
                    preset::set_location(&mut app.ui.opts, p);
                    app.program_change = true;
                }
            }

            // Optionally dump raw MIDI messages out the serial port: queued here,
            // printed by the main loop (no log formatting on the ISR stack).
            if serial_debug {
                app.midi_log.push_back(midi_word).ok();
            }
        }

        // Highlight the last CC-modified option (once per cycle)
        if cc_highlight {
            if let Some(idx) = last_cc_index {
                app.ui.opts.select_global(idx);
                app.ui.external_modify();
            }
        }

        //
        // Update synthesizer
        //

        let opts = &app.ui.opts;

        // CV inputs (fw cv.rs): pitch / cutoff routed in gateware at audio rate,
        // other targets as knob offsets here
        let cv_targets = [opts.cv.in0.value, opts.cv.in1.value, opts.cv.in2.value, opts.cv.in3.value];
        let cv_amts = [opts.cv.amt0.value, opts.cv.amt1.value, opts.cv.amt2.value, opts.cv.amt3.value];
        let cv_samples = app.ui.pmod.sample_i();
        let cv_jack = app.ui.pmod.jack();
        let cvk = |knob: u16, t: CvTarget| -> u16 {
            cv::apply(knob, cv::knob_offset(&cv_targets, &cv_amts, &cv_samples, cv_jack, t))
        };
        let (pitch_src, pitch_amt) = cv::fast_route(&cv_targets, &cv_amts, CvTarget::Pitch);
        let (cutoff_src, cutoff_amt) = cv::fast_route(&cv_targets, &cv_amts, CvTarget::Cutoff);
        app.synth.set_cv_route(pitch_src, cutoff_src,
                               opts.cv.out0.value as u8, opts.cv.out1.value as u8);
        app.synth.set_cv_amt(pitch_amt, cutoff_amt);

        // Output stage: level x drive pre-gain, drive-compensating post-gain
        let (pre, post) = out::registers(cvk(opts.effect.level.value, CvTarget::Level),
                                         cvk(opts.effect.drive.value, CvTarget::Drive));
        let pre = app.drive_smoother.proc_u16(pre);
        let post = app.out_gain_smoother.proc_u16(post);
        app.synth.set_drive(pre);
        app.synth.set_out_gain(post);

        // Map 0-1 UI range to 32768-8192 hardware range (inverted)
        // Ladder resonance: knob -> KR-106-calibrated curve (gateware: k = 4.5 res)
        let reso_smooth = app.reso_smoother.proc_u16(vcf::res_knob_to_reg(cvk(opts.vcf.reso.value, CvTarget::Reso)));
        app.synth.set_reso(reso_smooth);

        app.synth.set_fx(opts.vcf.hpf.value as u8, opts.effect.chorus.value as u8);

        // Exponential cutoff (normalized octaves, 32768 = 1.0 = 10 octaves).
        // Per voice: knob (Juno-106 slider curve) + keyboard tracking (relative to C1) + mod wheel.
        // Env amount (bipolar) and CV in1 are added per sample in gateware.
        {
            let cutoff = vcf::cutoff_knob_to_c(app.cutoff_smoother.proc_u16(opts.vcf.cutoff.value));
            let kbd = i32::from(opts.vcf.kbd_track.value);
            // bender -> VCF (Juno-106 bender VCF sensitivity)
            let bend = i32::from(app.synth.pitch_bend());
            let bend_off = ((bend * i32::from(opts.voice.bend_vcf.value)) >> 15) * BEND_VCF_RANGE >> 15;
            let notes = app.synth.voice_notes();
            for (voice, note) in notes.iter().enumerate() {
                // kbd * (note - C1) / 12 octaves, 1 octave = 1/10 = 3276.8 raw
                let track = kbd * (i32::from(*note) - vcf::KBD_REF_NOTE) / 120;
                let var = variance::cutoff_raw(voice, opts.voice.variance.value);
                let base = (cutoff + track + bend_off + var).clamp(-32768, 32767);
                app.synth.set_cutoff_base(voice, base as i16);
            }
            let amt = i32::from(cvk(opts.vcf.env_amt.value, CvTarget::EnvAmt)).min(32767);
            let amt = match opts.vcf.env_pol.value {
                EnvPolarity::Pos => amt,
                EnvPolarity::Neg => -amt,
            };
            app.synth.set_env_amt(amt as i16);
        }

        app.synth.set_midi_channel_filter(opts.misc.midi_ch.value.to_filter());

        // ADSR params
        // Juno-106-style envelope: knob = slider / 127 -> KR-106-measured times
        // (float math only when a knob moves)
        let env_knobs = (opts.adsr.attack.value, opts.adsr.decay.value, opts.adsr.release.value);
        if app.last_env_knobs != Some(env_knobs) {
            app.last_env_knobs = Some(env_knobs);
            app.synth.set_attack_rate(env::attack_inc(env::knob_to_time(env_knobs.0, &env::ATTACK_S)));
            app.synth.set_decay_rate(env::eps(env::knob_to_time(env_knobs.1, &env::DECAY_S)));
            app.synth.set_release_rate(env::eps(env::knob_to_time(env_knobs.2, &env::RELEASE_S)));
        }
        app.synth.set_sustain_level((u32::from(opts.adsr.sustain.value) * 65535 / 32768) as u16);

        // Juno DCO mixer + modes (live; cheap register writes every tick)
        app.synth.set_dco(opts.dco.saw.value == Switch::On,
                          opts.dco.pulse.value == Switch::On,
                          opts.dco.pwm_mode.value == PwmMode::Lfo,
                          opts.adsr.vca_mode.value == VcaMode::Gate,
                          opts.dco.range.value as u8);
        app.synth.set_pw_knob(cvk(opts.dco.pw.value, CvTarget::Pw));
        app.synth.set_mix_levels(dco::sub_gain(opts.dco.sub.value),
                                 dco::level_gain(opts.dco.noise.value));

        // Key assign + per-voice detune (variance, unison spread): only on change
        let unison = opts.voice.assign.value == AssignMode::Unison;
        let voice_knobs = (unison, opts.voice.detune.value, opts.voice.variance.value,
                           opts.voice.bend.value as u8);
        if app.last_voice_knobs != Some(voice_knobs) {
            app.last_voice_knobs = Some(voice_knobs);
            for voice in 0..N_VOICES {
                let d = variance::detune_raw(voice, voice_knobs.2, unison, voice_knobs.1);
                app.synth.set_voice_cfg(voice, d, unison, voice_knobs.3);
            }
        }

        // Portamento (float conversion only when the knob moves)
        let porta = cvk(opts.voice.porta.value, CvTarget::Porta);
        if app.last_porta != Some(porta) {
            app.last_porta = Some(porta);
            app.synth.set_glide(glide::coef(porta));
        }

        // LFO (float conversions only when a knob moves)
        let lfo_knobs = (cvk(opts.lfo.rate.value, CvTarget::LfoRate), opts.lfo.delay.value,
                         cvk(opts.dco.lfo.value, CvTarget::Vibrato),
                         cvk(opts.vcf.lfo.value, CvTarget::VcfLfo),
                         opts.voice.wheel.value, app.mod_wheel);
        if app.last_lfo_knobs != Some(lfo_knobs) {
            app.last_lfo_knobs = Some(lfo_knobs);
            app.synth.set_lfo_rate(lfo::rate_inc(lfo_knobs.0));
            app.synth.set_lfo_delay(lfo::delay_step(lfo_knobs.1));
            // mod wheel = Juno-106 LFO lever: adds vibrato on top of the DCO LFO knob
            let wheel = u32::from(lfo::dco_depth(opts.voice.wheel.value))
                        * u32::from(app.mod_wheel) / 127;
            let dco = (u32::from(lfo::dco_depth(lfo_knobs.2)) + wheel).min(65535) as u16;
            app.synth.set_lfo_depths(dco, lfo::vcf_depth(lfo_knobs.3));
        }
    });
}

fn global_index(opts: &Opts, opt: &dyn OptionTrait) -> usize {
    let key = opt.key().value();
    opts.all().enumerate()
        .find(|(_, o)| o.key().value() == key)
        .expect("cc_map: option key not found").0
}

fn apply_cc_action(opts: &mut Opts, action: &opts::cc_map::CcAction) {
    if let Some(opt) = opts.all_mut().nth(action.global_index) {
        match action.mode {
            CcMapMode::Absolute => { opt.set_from_cc(action.cc_value); }
        }
    }
}

fn build_cc_mapper(opts: &Opts) -> MidiCcMapper {
    let mut m = MidiCcMapper::new();
    // VOICE page (CC 5 = MIDI portamento time)
    m.add(5,  global_index(opts, &opts.voice.porta),     CcMapMode::Absolute);
    m.add(14, global_index(opts, &opts.voice.assign),    CcMapMode::Absolute);
    m.add(94, global_index(opts, &opts.voice.detune),    CcMapMode::Absolute);
    m.add(9,  global_index(opts, &opts.voice.variance),  CcMapMode::Absolute);
    m.add(42, global_index(opts, &opts.voice.bend),      CcMapMode::Absolute);
    m.add(43, global_index(opts, &opts.voice.bend_vcf),  CcMapMode::Absolute);
    m.add(44, global_index(opts, &opts.voice.wheel),     CcMapMode::Absolute);
    // LFO page (CC 24-25)
    m.add(24, global_index(opts, &opts.lfo.rate),        CcMapMode::Absolute);
    m.add(25, global_index(opts, &opts.lfo.delay),       CcMapMode::Absolute);
    // DCO page (CC 16-22)
    m.add(15, global_index(opts, &opts.dco.range),       CcMapMode::Absolute);
    m.add(16, global_index(opts, &opts.dco.pwm_mode),    CcMapMode::Absolute);
    m.add(22, global_index(opts, &opts.dco.lfo),         CcMapMode::Absolute);
    m.add(17, global_index(opts, &opts.dco.saw),         CcMapMode::Absolute);
    m.add(18, global_index(opts, &opts.dco.pulse),       CcMapMode::Absolute);
    m.add(19, global_index(opts, &opts.dco.pw),          CcMapMode::Absolute);
    m.add(20, global_index(opts, &opts.dco.sub),         CcMapMode::Absolute);
    m.add(21, global_index(opts, &opts.dco.noise),       CcMapMode::Absolute);
    m.add(23, global_index(opts, &opts.vcf.reso),        CcMapMode::Absolute);
    // VCF page (CC 23, 26-29)
    m.add(26, global_index(opts, &opts.vcf.cutoff),      CcMapMode::Absolute);
    m.add(27, global_index(opts, &opts.vcf.env_amt),     CcMapMode::Absolute);
    m.add(28, global_index(opts, &opts.vcf.env_pol),     CcMapMode::Absolute);
    m.add(29, global_index(opts, &opts.vcf.kbd_track),   CcMapMode::Absolute);
    m.add(34, global_index(opts, &opts.vcf.lfo),         CcMapMode::Absolute);
    m.add(36, global_index(opts, &opts.vcf.hpf),         CcMapMode::Absolute);
    // ADSR page (CC 30-33)
    m.add(30, global_index(opts, &opts.adsr.attack),  CcMapMode::Absolute);
    m.add(31, global_index(opts, &opts.adsr.decay),   CcMapMode::Absolute);
    m.add(32, global_index(opts, &opts.adsr.sustain), CcMapMode::Absolute);
    m.add(33, global_index(opts, &opts.adsr.release), CcMapMode::Absolute);
    m.add(35, global_index(opts, &opts.adsr.vca_mode), CcMapMode::Absolute);
    // Effect page (CC 40-41)
    m.add(40, global_index(opts, &opts.effect.drive),    CcMapMode::Absolute);
    m.add(7,  global_index(opts, &opts.effect.level),    CcMapMode::Absolute);
    m.add(41, global_index(opts, &opts.effect.chorus),   CcMapMode::Absolute);
    m
}

static I2C0_BUS: StaticCell<I2cBus<I2c0>> = StaticCell::new();

type Ui = ui::UI<Encoder0, EurorackPmod0, I2cDevice<'static, I2c0>, Opts>;

struct App {
    ui: Ui,
    synth: Synth,
    drive_smoother: OnePoleSmoother,
    out_gain_smoother: OnePoleSmoother,
    reso_smoother: OnePoleSmoother,
    cutoff_smoother: OnePoleSmoother,
    // last mod wheel (CC1) value, 0..127
    mod_wheel: u8,
    // last envelope knobs written (attack, decay, release)
    last_env_knobs: Option<(u16, u16, u16)>,
    // last LFO knobs written (rate, delay, dco depth, vcf depth)
    last_lfo_knobs: Option<(u16, u16, u16, u16, u16, u8)>,
    // last portamento knob written
    last_porta: Option<u16>,
    // last key assign / detune / variance written
    last_voice_knobs: Option<(bool, u16, u16, u8)>,
    // midi cc mapper
    cc_mapper: MidiCcMapper,
    // time base for the main loop (preset write timeout, edit buffer autosave)
    uptime_ms: u32,
    // MIDI program change received (bank / patch already set in the options)
    program_change: bool,
    // raw MIDI words for the `serial-debug` dump, printed by the main loop
    midi_log: heapless::Deque<u32, 16>,
}

impl App {
    pub fn new(ui: Ui, mut synth: Synth) -> Self {
        let drive_smoother = OnePoleSmoother::new(0.05f32);
        let out_gain_smoother = OnePoleSmoother::new(0.05f32);
        let reso_smoother = OnePoleSmoother::new(0.05f32);
        let cutoff_smoother = OnePoleSmoother::new(0.05f32);
        let cc_mapper = build_cc_mapper(&ui.opts);
        synth.usb_host_midi(1, 1);
        synth.usb_host_vbus(false); // default, but set anyway...
        Self {
            ui,
            synth,
            drive_smoother,
            out_gain_smoother,
            reso_smoother,
            cutoff_smoother,
            mod_wheel: 0,
            last_lfo_knobs: None,
            last_porta: None,
            last_voice_knobs: None,
            last_env_knobs: None,
            cc_mapper,
            uptime_ms: 0,
            program_change: false,
            midi_log: heapless::Deque::new(),
        }
    }
}

#[cfg(not(feature = "sim_stack"))]
type FlashT = SPIFlash0;
#[cfg(feature = "sim_stack")]
type FlashT = tiliqua_fw::sim_stack::PsramFlash;
type FlashPart = BlockingPartition<'static, NoopRawMutex, FlashT>;

// Long-lived objects live in statics instead of `main`'s stack frame. Block RAM
// (16 KiB) holds data and the one stack shared by the main loop and the tick ISR;
// as statics these objects have a fixed size at link time and are not duplicated
// by moves during startup (each move left a copy in `main`'s frame). Each is
// created by its own `#[inline(never)]` step below, so only one step's
// temporaries are on the stack at a time.

static FLASH: StaticCell<FlashMutex<NoopRawMutex, RefCell<FlashT>>> = StaticCell::new();
/// main loop's snapshot of the options (drawing, presets), see `copy_opts`
static VIEW: StaticCell<Opts> = StaticCell::new();
static FLASH_PERSIST: StaticCell<Option<FlashOptionsPersistence<FlashPart>>> = StaticCell::new();
static PRESET_STORE: StaticCell<Option<PresetStore<FlashPart>>> = StaticCell::new();
/// preset record buffer (load / write / edit buffer)
static REC: StaticCell<[u8; preset::BLOB_MAX + 8]> = StaticCell::new();
static APP: StaticCell<Mutex<RefCell<App>>> = StaticCell::new();

/// `*dst = src.clone()` in its own frame: the clone's temporary is freed on
/// return instead of being inlined into the main loop's frame.
#[inline(never)]
fn copy_opts(dst: &mut Opts, src: &Opts) {
    dst.clone_from(src);
}

/// Parsed bootinfo (manifest: name, tag, help texts, option window), ~1 KiB.
static BOOTINFO: StaticCell<bootinfo::BootInfo> = StaticCell::new();

#[inline(never)]
fn init_bootinfo() -> &'static bootinfo::BootInfo {
    BOOTINFO.init(unsafe { bootinfo::BootInfo::from_addr(BOOTINFO_BASE) }.expect("invalid bootinfo"))
}

#[inline(never)]
fn init_rec() -> &'static mut [u8; preset::BLOB_MAX + 8] {
    REC.init([0u8; preset::BLOB_MAX + 8])
}

/// Restore the edit buffer (sound + active location as at power-off) into
/// `opts`. Without one, keep the sound from the system save and mark it as
/// `init` (edited).
#[inline(never)]
fn restore_edit(opts: &mut Opts, store: &mut Option<PresetStore<FlashPart>>,
                rec: &mut [u8], default_patch_hash: u32) -> PresetCtl {
    let (active, snap) = store.as_mut()
        .and_then(|s| s.load(preset::KEY_EDIT, rec))
        .and_then(|n| preset::decode_edit(opts, &rec[..n]))
        .unwrap_or((None, default_patch_hash));
    if let Some(a) = active {
        preset::set_location(opts, a);
    }
    PresetCtl::new(preset::selected_slot(opts), active, snap, preset::patch_hash(opts))
}

#[inline(never)]
fn init_view() -> &'static mut Opts {
    VIEW.init(Opts::default())
}

#[inline(never)]
fn init_flash_persist(flash: &'static FlashMutex<NoopRawMutex, RefCell<FlashT>>,
                      window: Option<core::ops::Range<u32>>, opts: &mut Opts)
                      -> &'static mut Option<FlashOptionsPersistence<FlashPart>> {
    FLASH_PERSIST.init(FlashOptionsPersistence::open_partition(flash, window, opts))
}

#[inline(never)]
fn init_preset_store(flash: &'static FlashMutex<NoopRawMutex, RefCell<FlashT>>,
                     window: Option<core::ops::Range<u32>>)
                     -> &'static mut Option<PresetStore<FlashPart>> {
    PRESET_STORE.init(PresetStore::open_partition(flash, window))
}

#[inline(never)]
fn init_app(opts: &Opts, make_ui: impl FnOnce(Opts) -> Ui, synth: Synth)
            -> &'static Mutex<RefCell<App>> {
    APP.init(Mutex::new(RefCell::new(App::new(make_ui(opts.clone()), synth))))
}

#[entry]
fn main() -> ! {
    #[cfg(feature = "stack_watch")]
    tiliqua_fw::stack_watch::paint();
    let peripherals = pac::Peripherals::take().expect("unable to take peripherals");
    let serial = Serial0::new(peripherals.UART0);
    let mut persist = Persist0::new(peripherals.PERSIST_PERIPH);
    let spiflash = SPIFlash0::new(
        peripherals.SPIFLASH_CTRL,
        SPIFLASH_BASE,
        SPIFLASH_SZ_BYTES
    );
    crate::handlers::logger_init(serial);

    info!("Hello from Tiliqua JUNO!");

    let bootinfo = init_bootinfo();
    let modeline = bootinfo.modeline.clone().maybe_override_fixed(
        FIXED_MODELINE, CLOCK_DVI_HZ);
    let mut display = hal::new_framebuffer!(peripherals, modeline.clone());

    let mut i2c1 = I2c1::new(peripherals.I2C1);
    let mut pmod = EurorackPmod0::new(peripherals.PMOD0_PERIPH);
    calibration::CalibrationConstants::load_or_default(&mut i2c1, &mut pmod);

    //
    // Option storage (system options + presets behind them, preset.rs)
    //

    #[cfg(not(feature = "sim_stack"))]
    let (flash, window) = (FLASH.init(FlashMutex::new(RefCell::new(spiflash))),
                           bootinfo.manifest.get_option_storage_window());
    // Simulation: option storage in PSRAM (sim_stack.rs)
    #[cfg(feature = "sim_stack")]
    let (flash, window) = {
        let _ = spiflash;
        (FLASH.init(FlashMutex::new(RefCell::new(tiliqua_fw::sim_stack::PsramFlash::new()))),
         Some(tiliqua_fw::sim_stack::WINDOW))
    };

    //
    // Configure TUSB322 (CC controller) in DFP/Host mode
    // This is needed if Tiliqua is connected to a device with a true USB-C to USB-C cable.
    // It is also used for detecting (and preventing) applying VBUS if we have a host/host
    // connection.
    //
    // TODO: disable terminations when host mode is disabled?
    // TODO: draw error text if this fails?
    //

    let i2c0 = I2C0_BUS.init(hal::i2c::bus(I2c0::new(peripherals.I2C0)));
    let mut tusb322 = TUSB322Driver::new(I2cDevice::new(i2c0));
    tusb322.soft_reset().ok();
    tusb322.disable_term().ok();

    //
    // Create App instance
    //

    palette::ColorPalette::default().write_to_hardware(&mut display);

    let encoder = Encoder0::new(peripherals.ENCODER0);
    let pca9635 = Pca9635Driver::new(I2cDevice::new(i2c0));
    let synth = Synth::new(peripherals.SYNTH_PERIPH);
    // Options: defaults, then the system save, then the edit buffer.
    let opts = init_view();
    let default_patch_hash = preset::patch_hash(opts);
    #[cfg(feature = "sim_stack")]
    if tiliqua_fw::sim_stack::MIDI_LOG {
        opts.misc.serial_debug.value = UsbMidiSerialDebug::On;
    }
    // Option storage window: upstream system options first, presets behind (preset.rs)
    let (system_window, preset_window) = preset::split_window(window);
    let flash_persist_opt = init_flash_persist(flash, system_window, opts);
    let preset_store = init_preset_store(flash, preset_window);
    let rec = init_rec();
    let mut preset_ctl = restore_edit(opts, preset_store, rec, default_patch_hash);
    let app = init_app(opts,
        |opts| ui::UI::new(opts, TICK_PERIOD_MS, encoder, pca9635, pmod), synth);

    handler!(tick_isr = || tick_handler(app));

    irq::scope(|s| {

        s.register(tick::Interrupt::TICK, tick_isr);

        tick::enable(TICK_PERIOD_MS);

        let h_active = display.size().width;
        let v_active = display.size().height;

        let mut last_attached_state = AttachedState::NotAttached;
        let mut last_opt_host_enabled = false;

        loop {

            let (notes, cutoffs, draw_options, save_opts, wipe_opts,
                 preset_write, preset_init, program_change, now) = critical_section::with(|cs| {
                let mut app = app.borrow_ref_mut(cs);
                // Snapshot of the options for this iteration (drawing, presets)
                // into the persistent buffer: no Opts temporaries on the stack.
                copy_opts(opts, &app.ui.opts);
                let save_opts = app.ui.opts.misc.save_opts.poll();
                let wipe_opts = app.ui.opts.misc.wipe_opts.poll();
                let preset_write = app.ui.opts.preset.write.poll();
                let preset_init = app.ui.opts.preset.init.poll();
                let program_change = core::mem::take(&mut app.program_change);
                let now = app.uptime_ms;

                // USB Host: Disable DFP terminations if `MISC->usb-host` option is off
                //
                // This stops us supplying VCONN on the CC pins for e-tagged cables, fixing an
                // edge case where we may backpower VBUS due to the following bug on
                // early Tiliqua R5.1s: https://github.com/apfaudio/tiliqua-hardware/issues/1
                //

                let opt_host_enabled = app.ui.opts.misc.usb_host.value == UsbHost::On;
                if last_opt_host_enabled != opt_host_enabled {
                    if opt_host_enabled {
                        tusb322.soft_reset().ok();
                        tusb322.set_mode(TUSB322Mode::Dfp).ok();
                    } else {
                        tusb322.disable_term().ok();
                        app.synth.usb_host_vbus(false);
                    }
                }
                last_opt_host_enabled = opt_host_enabled;

                // Type-C hotplugging detection
                //
                // Only enable VBUS if the TUSB322 Type-C controller says we are attached as:
                // - Attached.SRC
                // - Attached.ACCESSORY && AccessoryType.DebugDfp
                //
                // Due to the same hardware bug as above, Tiliqua's host port in DFP mode may
                // have the TUSB322I see some rare adapter combinations as `DebugDfp` accessories,
                // rather than `Attached.SRC`. To work around this, we still power VBUS on `DebugDfp`.
                //

                if let Ok(ctrl) = tusb322.read_connection_status_control() {
                    if let Ok(conn) = tusb322.read_connection_status() {
                        // Only update on valid reads to reduce risk of unintended toggling mid-stream
                        if ctrl.attached_state != last_attached_state {
                            info!("USB CC hotplug: {ctrl:?} | {conn:?}");
                            last_attached_state = ctrl.attached_state;
                        }
                        // Check how we are attached
                        let attached_src = ctrl.attached_state == AttachedState::AttachedSrc;
                        let attached_debug_dfp =
                            (ctrl.attached_state == AttachedState::AttachedAccessory) &&
                            (conn.accessory == AccessoryType::DebugDfp);
                        app.synth.usb_host_vbus(opt_host_enabled && (attached_src || attached_debug_dfp));
                    }
                }

                //
                // Copy out all the bits of state we need for drawing
                //

                (app.synth.voice_notes(),
                 app.synth.voice_cutoffs(),
                 app.ui.draw(),
                 save_opts,
                 wipe_opts,
                 preset_write,
                 preset_init,
                 program_change,
                 now)
            });

            #[cfg(feature = "sim_stack")]
            let (preset_write, preset_init, save_opts, wipe_opts) = {
                use tiliqua_fw::sim_stack::Event;
                let mut v = (preset_write, preset_init, save_opts, wipe_opts);
                match tiliqua_fw::sim_stack::event(now) {
                    Some(Event::Write) => v.0 = true,
                    Some(Event::Init) => v.1 = true,
                    Some(Event::SaveOpts) => v.2 = true,
                    Some(Event::WipeOpts) => v.3 = true,
                    Some(Event::Slot(slot)) => {
                        critical_section::with(|cs| {
                            preset::set_location(&mut app.borrow_ref_mut(cs).ui.opts, slot);
                        });
                        preset::set_location(&mut *opts, slot);
                    }
                    None => {}
                }
                v
            };

            // MIDI dump (MISC serial-debug), queued by the tick ISR
            let midi_words = critical_section::with(|cs| {
                core::mem::take(&mut app.borrow_ref_mut(cs).midi_log)
            });
            for w in midi_words.iter() {
                info!("midi: 0x{:x} 0x{:x} 0x{:x}", w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF);
            }

            if save_opts {
                if let Some(ref mut flash_persist) = flash_persist_opt {
                    if let Err(e) = flash_persist.save_options(&*opts) {
                        error!("failed to save options: {e:?}");
                    }
                }
            }

            if wipe_opts {
                critical_section::with(|cs| {
                    let mut app = app.borrow_ref_mut(cs);
                    app.ui.opts = Opts::default();
                    if let Some(ref mut flash_persist) = flash_persist_opt {
                        if let Err(e) = flash_persist.erase_all() {
                            error!("failed to erase options: {e:?}");
                        }
                    }
                });
            }

            //
            // Presets (preset.rs): load / write / init, edited marker, edit buffer
            // autosave. Flash IO happens here, never in the tick ISR.
            //

            let sel = preset::selected_slot(&opts);
            let on_preset_page = opts.tracker.page.value == Page::Preset;
            #[cfg(feature = "sim_stack")]
            let on_preset_page = { let _ = on_preset_page; true };  // pages step on their own
            // Loads / init are applied in place (no `Opts` copies: data and both
            // stacks share 16 KiB block RAM), then mirrored into the local copy.
            let apply = |ctl: &mut PresetCtl, rec: Option<&[u8]>, active: Option<u8>, opts: &mut Opts| -> bool {
                let ok = critical_section::with(|cs| {
                    let live = &mut app.borrow_ref_mut(cs).ui.opts;
                    let ok = rec.map(|r| preset::decode(live, r)).unwrap_or(false);
                    if !ok {
                        preset::reset(live);
                    }
                    preset::copy_patch(opts, live);
                    ok
                });
                ctl.loaded(active, preset::patch_hash(opts), active.is_some() && !ok);
                ok
            };
            match preset_ctl.step(now, sel, on_preset_page, preset_write, preset_init, program_change) {
                Action::Load(slot) => {
                    let n = preset_store.as_mut().and_then(|s| s.load(slot as u32, rec));
                    let ok = apply(&mut preset_ctl, n.map(|n| &rec[..n]), Some(slot), &mut *opts);
                    info!("preset: load {}{}", preset::slot_name(slot), if ok { "" } else { " (empty)" });
                }
                Action::Init => {
                    apply(&mut preset_ctl, None, None, &mut *opts);
                }
                Action::Write(slot) => {
                    let ok = match (preset_store.as_mut(), preset::encode(&opts, rec)) {
                        (Some(s), Some(n)) => s.save(slot as u32, &rec[..n]),
                        _ => false,
                    };
                    if ok {
                        preset_ctl.loaded(Some(slot), preset::patch_hash(&opts), false);
                        info!("preset: wrote {}", preset::slot_name(slot));
                    } else {
                        error!("preset: write {} failed", preset::slot_name(slot));
                    }
                }
                Action::Revert(slot) => {
                    critical_section::with(|cs| {
                        preset::set_location(&mut app.borrow_ref_mut(cs).ui.opts, slot);
                    });
                    preset::set_location(&mut *opts, slot);
                }
                Action::None => {}
            }
            let patch_hash = preset::patch_hash(&opts);
            let status = preset_ctl.status(preset::selected_slot(&opts), patch_hash);
            if status != opts.preset.status.value {
                opts.preset.status.value = status.clone();
                critical_section::with(|cs| {
                    app.borrow_ref_mut(cs).ui.opts.preset.status.value = status;
                });
            }
            if preset_ctl.autosave_due(now, patch_hash) {
                if let Some(s) = preset_store.as_mut() {
                    if let Some(n) = preset::encode_edit(&opts, preset_ctl.active, preset_ctl.snap(), rec) {
                        s.save(preset::KEY_EDIT, &rec[..n]);
                    }
                }
                preset_ctl.autosaved(patch_hash);   // also on failure: no retry storm
                #[cfg(feature = "sim_stack")]
                info!("sim_stack: t={}ms autosave", now);
            }

            #[cfg(feature = "stack_watch")]
            {
                tiliqua_fw::stack_watch::scan();
                #[cfg(not(feature = "sim_stack"))]
                tiliqua_fw::stack_watch::report(now, 10_000);
                #[cfg(feature = "sim_stack")]
                tiliqua_fw::stack_watch::report(now, 100);
            }

            let on_help_page = opts.tracker.page.value == Page::Help;

            if draw_options || on_help_page {
                let (x, y) = if on_help_page {
                    (h_active/2-30, v_active-100)
                } else {
                    (h_active/2-30, 70)
                };
                draw::draw_options(&mut display, &*opts, x, y,
                                   HUE).ok();
                draw::draw_name(&mut display, h_active/2, v_active-50, HUE,
                                &bootinfo.manifest.name, &bootinfo.manifest.tag, &modeline).ok();
                if opts.tracker.page.value == Page::Env {
                    use draw::AdsrPhase;
                    let highlight = opts.selected().and_then(|i| {
                        match opts.view().options()[i].key().value() {
                            v if v == opts.adsr.attack.key().value()  => Some(AdsrPhase::Attack),
                            v if v == opts.adsr.decay.key().value()   => Some(AdsrPhase::Decay),
                            v if v == opts.adsr.sustain.key().value() => Some(AdsrPhase::Sustain),
                            v if v == opts.adsr.release.key().value() => Some(AdsrPhase::Release),
                            _ => None,
                        }
                    });
                    draw::draw_adsr(&mut display,
                        h_active/2-190, 75,
                        125, 40,
                        opts.adsr.attack.value,
                        opts.adsr.decay.value,
                        opts.adsr.sustain.value,
                        opts.adsr.release.value,
                        HUE,
                        highlight).ok();
                }
                if opts.tracker.page.value == Page::Dco {
                    const PREVIEW_LEN: usize = 64;
                    let mut preview = [0i16; PREVIEW_LEN];
                    dco::preview(&mut preview, opts.dco.saw.value == Switch::On,
                                 opts.dco.pulse.value == Switch::On, opts.dco.pw.value,
                                 opts.dco.sub.value);
                    draw::draw_waveform_preview(&mut display,
                        h_active/2-190, 80,
                        125, 40,
                        HUE,
                        &preview).ok();
                }
            }

            if on_help_page {
                draw::draw_help_page(&mut display,
                    MODULE_DOCSTRING,
                    bootinfo.manifest.help.as_ref(),
                    h_active,
                    v_active,
                    opts.help.scroll.value,
                    HUE).ok();
                persist.set_persistence(PERSIST_HELP);
            } else {
                persist.set_persistence(PERSIST);
            }

            if !on_help_page {
                for ix in 0usize..N_VOICES {
                    let j = (N_VOICES-1)-ix;
                    draw::draw_voice(&mut display,
                                     ((h_active as f32)/2.0f32 + 330.0f32*f32::cos(2.45f32 + 1.5f32 * j as f32 / (N_VOICES as f32))) as i32,
                                     ((v_active as f32)/2.0f32 + 330.0f32*f32::sin(2.45f32 + 1.5f32 * j as f32 / (N_VOICES as f32))) as u32 - 15,
                                     notes[ix], cutoffs[ix], HUE).ok();
                }
            }
        }
    })
}
