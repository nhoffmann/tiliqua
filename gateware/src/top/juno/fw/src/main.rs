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

pub const TICK_PERIOD_MS: u32 = 5;

// Fixed menu appearance (BEAM page removed).
const HUE: u8 = 10;
const PERSIST: u8 = 15;
const PERSIST_HELP: u8 = 64;

// Mod wheel fully up opens the cutoff by 0.5 (5 octaves). Same as sim_render.py.
const MOD_WHEEL_RANGE: i32 = 16384;

fn adsr_ui_to_rate(ui_value: u16) -> u16 {
    // 0..32768 -> 1ms..2000ms -> hardware rate
    let ms_x32k = 32768u32 + u32::from(ui_value) * 1999;
    let rate = (45_000_000u64 / u64::from(ms_x32k)) as u32;
    rate.min(65535) as u16
}

fn tick_handler(app: &Mutex<RefCell<App>>) {

    critical_section::with(|cs| {

        let mut app = app.borrow_ref_mut(cs);

        //
        // Update UI and options
        //

        app.ui.update();
        let opts = app.ui.opts.clone();

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
            let midi_word = app.synth.midi_read();
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

            // Optionally dump raw MIDI messages out serial port.
            if opts.misc.serial_debug.value == UsbMidiSerialDebug::On {
                info!("midi: 0x{:x} 0x{:x} 0x{:x}",
                      bytes[0], bytes[1], bytes[2]);
            }
        }

        // Highlight the last CC-modified option (once per cycle)
        if opts.misc.cc_highlight.value == CcHighlight::On {
            if let Some(idx) = last_cc_index {
                app.ui.opts.select_global(idx);
                app.ui.external_modify();
            }
        }

        //
        // Update synthesizer
        //

        let drive_smooth = app.drive_smoother.proc_u16(opts.effect.drive.value);
        app.synth.set_drive(drive_smooth);

        // Map 0-1 UI range to 32768-8192 hardware range (inverted)
        let reso_ui = u32::from(opts.vcf.reso.value);
        let reso_hw = (32768 - reso_ui * 24576 / 32768) as u16;
        let reso_smooth = app.reso_smoother.proc_u16(reso_hw);
        app.synth.set_reso(reso_smooth);

        let diffuse_smooth = app.diffusion_smoother.proc_u16(opts.effect.diffuse.value);
        let coeff_dry: i32 = i32::from(32768 - diffuse_smooth);
        let coeff_wet: i32 = i32::from(diffuse_smooth);

        app.synth.set_matrix_coefficient(0, 0, coeff_dry);
        app.synth.set_matrix_coefficient(1, 1, coeff_dry);
        app.synth.set_matrix_coefficient(2, 2, coeff_dry);
        app.synth.set_matrix_coefficient(3, 3, coeff_dry);

        app.synth.set_matrix_coefficient(0, 4, coeff_wet);
        app.synth.set_matrix_coefficient(1, 5, coeff_wet);
        app.synth.set_matrix_coefficient(2, 6, coeff_wet);
        app.synth.set_matrix_coefficient(3, 7, coeff_wet);

        // Exponential cutoff (normalized octaves, 32768 = 1.0 = 10 octaves).
        // Per voice: knob + keyboard tracking (relative to C4) + mod wheel.
        // Env amount (bipolar) and CV in1 are added per sample in gateware.
        {
            let cutoff = i32::from(app.cutoff_smoother.proc_u16(opts.vcf.cutoff.value));
            let kbd = i32::from(opts.vcf.kbd_track.value);
            let mod_off = i32::from(app.mod_wheel) * MOD_WHEEL_RANGE / 127;
            let notes = app.synth.voice_notes();
            for (voice, note) in notes.iter().enumerate() {
                // kbd * (note - 60) / 12 octaves, 1 octave = 1/10 = 3276.8 raw
                let track = kbd * (i32::from(*note) - 60) / 120;
                let base = (cutoff + track + mod_off).clamp(-32768, 32767);
                app.synth.set_cutoff_base(voice, base as i16);
            }
            let amt = i32::from(opts.vcf.env_amt.value).min(32767);
            let amt = match opts.vcf.env_pol.value {
                EnvPolarity::Pos => amt,
                EnvPolarity::Neg => -amt,
            };
            app.synth.set_env_amt(amt as i16);
        }

        app.synth.set_midi_channel_filter(opts.misc.midi_ch.value.to_filter());

        // ADSR params
        app.synth.set_attack_rate(adsr_ui_to_rate(opts.adsr.attack.value));
        app.synth.set_decay_rate(adsr_ui_to_rate(opts.adsr.decay.value));
        app.synth.set_sustain_level((u32::from(opts.adsr.sustain.value) * 65535 / 32768) as u16);
        app.synth.set_release_rate(adsr_ui_to_rate(opts.adsr.release.value));

        // Juno DCO mixer + modes (live; cheap register writes every tick)
        app.synth.set_dco(opts.dco.saw.value == Switch::On,
                          opts.dco.pulse.value == Switch::On,
                          opts.dco.pwm_mode.value == PwmMode::Lfo,
                          opts.adsr.vca_mode.value == VcaMode::Gate);
        app.synth.set_pw_knob(opts.dco.pw.value);
        app.synth.set_mix_levels(dco::sub_gain(opts.dco.sub.value),
                                 dco::level_gain(opts.dco.noise.value));

        // LFO (float conversions only when a knob moves)
        let lfo_knobs = (opts.lfo.rate.value, opts.lfo.delay.value,
                         opts.dco.lfo.value, opts.vcf.lfo.value);
        if app.last_lfo_knobs != Some(lfo_knobs) {
            app.last_lfo_knobs = Some(lfo_knobs);
            app.synth.set_lfo_rate(lfo::rate_inc(opts.lfo.rate.value));
            app.synth.set_lfo_delay(lfo::delay_step(opts.lfo.delay.value));
            app.synth.set_lfo_depths(lfo::dco_depth(opts.dco.lfo.value),
                                     lfo::vcf_depth(opts.vcf.lfo.value));
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
    // LFO page (CC 24-25)
    m.add(24, global_index(opts, &opts.lfo.rate),        CcMapMode::Absolute);
    m.add(25, global_index(opts, &opts.lfo.delay),       CcMapMode::Absolute);
    // DCO page (CC 16-22)
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
    // ADSR page (CC 30-33)
    m.add(30, global_index(opts, &opts.adsr.attack),  CcMapMode::Absolute);
    m.add(31, global_index(opts, &opts.adsr.decay),   CcMapMode::Absolute);
    m.add(32, global_index(opts, &opts.adsr.sustain), CcMapMode::Absolute);
    m.add(33, global_index(opts, &opts.adsr.release), CcMapMode::Absolute);
    m.add(35, global_index(opts, &opts.adsr.vca_mode), CcMapMode::Absolute);
    // Effect page (CC 40-41)
    m.add(40, global_index(opts, &opts.effect.drive),    CcMapMode::Absolute);
    m.add(41, global_index(opts, &opts.effect.diffuse),  CcMapMode::Absolute);
    m
}

static I2C0_BUS: StaticCell<I2cBus<I2c0>> = StaticCell::new();

type Ui = ui::UI<Encoder0, EurorackPmod0, I2cDevice<'static, I2c0>, Opts>;

struct App {
    ui: Ui,
    synth: Polysynth0,
    drive_smoother: OnePoleSmoother,
    reso_smoother: OnePoleSmoother,
    diffusion_smoother: OnePoleSmoother,
    cutoff_smoother: OnePoleSmoother,
    // last mod wheel (CC1) value, 0..127
    mod_wheel: u8,
    // last LFO knobs written (rate, delay, dco depth, vcf depth)
    last_lfo_knobs: Option<(u16, u16, u16, u16)>,
    // midi cc mapper
    cc_mapper: MidiCcMapper,
}

impl App {
    pub fn new(ui: Ui, mut synth: Polysynth0) -> Self {
        let drive_smoother = OnePoleSmoother::new(0.05f32);
        let reso_smoother = OnePoleSmoother::new(0.05f32);
        let diffusion_smoother = OnePoleSmoother::new(0.05f32);
        let cutoff_smoother = OnePoleSmoother::new(0.05f32);
        let cc_mapper = build_cc_mapper(&ui.opts);
        synth.usb_host_midi(1, 1);
        synth.usb_host_vbus(false); // default, but set anyway...
        Self {
            ui,
            synth,
            drive_smoother,
            reso_smoother,
            diffusion_smoother,
            cutoff_smoother,
            mod_wheel: 0,
            last_lfo_knobs: None,
            cc_mapper,
        }
    }
}

#[entry]
fn main() -> ! {
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

    let bootinfo = unsafe { bootinfo::BootInfo::from_addr(BOOTINFO_BASE) }.expect("invalid bootinfo");
    let modeline = bootinfo.modeline.maybe_override_fixed(
        FIXED_MODELINE, CLOCK_DVI_HZ);
    let mut display = hal::new_framebuffer!(peripherals, modeline.clone());

    let mut i2c1 = I2c1::new(peripherals.I2C1);
    let mut pmod = EurorackPmod0::new(peripherals.PMOD0_PERIPH);
    calibration::CalibrationConstants::load_or_default(&mut i2c1, &mut pmod);

    //
    // Create options and maybe load from persistent storage
    //

    let mut opts = Opts::default();
    let flash = FlashMutex::<NoopRawMutex, _>::new(RefCell::new(spiflash));
    let mut flash_persist_opt = FlashOptionsPersistence::open_partition(
        &flash, bootinfo.manifest.get_option_storage_window(), &mut opts);

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

    let ui = ui::UI::new(opts, TICK_PERIOD_MS, Encoder0::new(peripherals.ENCODER0),
                         Pca9635Driver::new(I2cDevice::new(i2c0)), pmod);
    let app = Mutex::new(RefCell::new(App::new(ui, Polysynth0::new(peripherals.SYNTH_PERIPH))));

    handler!(tick_isr = || tick_handler(&app));

    irq::scope(|s| {

        s.register(tick::Interrupt::TICK, tick_isr);

        tick::enable(TICK_PERIOD_MS);

        let h_active = display.size().width;
        let v_active = display.size().height;

        let mut last_attached_state = AttachedState::NotAttached;
        let mut last_opt_host_enabled = false;

        loop {

            let (opts, notes, cutoffs, draw_options, save_opts, wipe_opts) = critical_section::with(|cs| {
                let mut app = app.borrow_ref_mut(cs);
                let save_opts = app.ui.opts.misc.save_opts.poll();
                let wipe_opts = app.ui.opts.misc.wipe_opts.poll();

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

                (app.ui.opts.clone(),
                 app.synth.voice_notes(),
                 app.synth.voice_cutoffs(),
                 app.ui.draw(),
                 save_opts,
                 wipe_opts)
            });

            if save_opts {
                if let Some(ref mut flash_persist) = flash_persist_opt {
                    if let Err(e) = flash_persist.save_options(&opts) {
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

            let on_help_page = opts.tracker.page.value == Page::Help;

            if draw_options || on_help_page {
                let (x, y) = if on_help_page {
                    (h_active/2-30, v_active-100)
                } else {
                    (h_active/2-30, 70)
                };
                draw::draw_options(&mut display, &opts, x, y,
                                   HUE).ok();
                draw::draw_name(&mut display, h_active/2, v_active-50, HUE,
                                &bootinfo.manifest.name, &bootinfo.manifest.tag, &modeline).ok();
                if opts.tracker.page.value == Page::Adsr {
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
