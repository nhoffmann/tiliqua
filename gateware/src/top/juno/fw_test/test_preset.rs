//! Tests for `fw/src/preset.rs`: record codec, flash store on a RAM NOR flash,
//! and the preset control state machine.

use embedded_storage::nor_flash::{
    ErrorType, MultiwriteNorFlash, NorFlash, NorFlashError, NorFlashErrorKind, ReadNorFlash,
};
use opts::{OptionPage, Options, OptionTrait};

use crate::options::*;
use crate::preset::*;

// --- RAM NOR flash ----------------------------------------------------------

const SECTOR: usize = 4096;

/// NOR semantics: erase sets 0xFF, writes can only clear bits.
struct RamFlash {
    mem: Vec<u8>,
    erases: usize,
}

impl RamFlash {
    fn new(sectors: usize) -> Self {
        Self { mem: vec![0xFF; sectors * SECTOR], erases: 0 }
    }
}

#[derive(Debug)]
struct FlashErr;

impl NorFlashError for FlashErr {
    fn kind(&self) -> NorFlashErrorKind {
        NorFlashErrorKind::OutOfBounds
    }
}

impl ErrorType for RamFlash {
    type Error = FlashErr;
}

impl ReadNorFlash for RamFlash {
    const READ_SIZE: usize = 1;
    fn read(&mut self, offset: u32, bytes: &mut [u8]) -> Result<(), FlashErr> {
        let o = offset as usize;
        bytes.copy_from_slice(self.mem.get(o..o + bytes.len()).ok_or(FlashErr)?);
        Ok(())
    }
    fn capacity(&self) -> usize {
        self.mem.len()
    }
}

impl NorFlash for RamFlash {
    const WRITE_SIZE: usize = 1;
    const ERASE_SIZE: usize = SECTOR;
    fn erase(&mut self, from: u32, to: u32) -> Result<(), FlashErr> {
        self.mem.get_mut(from as usize..to as usize).ok_or(FlashErr)?.fill(0xFF);
        self.erases += 1;
        Ok(())
    }
    fn write(&mut self, offset: u32, bytes: &[u8]) -> Result<(), FlashErr> {
        let o = offset as usize;
        let dst = self.mem.get_mut(o..o + bytes.len()).ok_or(FlashErr)?;
        for (d, s) in dst.iter_mut().zip(bytes) {
            *d &= *s;
        }
        Ok(())
    }
}

impl MultiwriteNorFlash for RamFlash {}

// --- helpers ----------------------------------------------------------------

fn record(o: &Opts) -> Vec<u8> {
    let mut buf = [0u8; BLOB_MAX];
    let n = encode(o, &mut buf).expect("record fits");
    buf[..n].to_vec()
}

/// A sound that differs from the defaults on every patch page.
fn edited(seed: usize) -> Opts {
    let mut o = Opts::default();
    for _ in 0..=seed % 5 {
        o.voice.porta.tick_up();
    }
    o.voice.assign.tick_up();
    o.lfo.rate.tick_down();
    o.dco.saw.tick_down();
    o.dco.sub.tick_up();
    for _ in 0..=seed % 7 {
        o.vcf.cutoff.tick_up();
    }
    o.adsr.release.tick_up();
    o.effect.level.tick_down();
    o
}

fn patch_pages_mut(o: &mut Opts) -> impl Iterator<Item = &mut dyn OptionTrait> {
    [o.voice.options_mut(), o.lfo.options_mut(), o.dco.options_mut(),
     o.vcf.options_mut(), o.adsr.options_mut(), o.effect.options_mut()]
        .into_iter().flatten()
}

fn values(o: &Opts) -> Vec<String> {
    patch_options(o).map(|x| x.value().as_str().to_string()).collect()
}

// --- codec ------------------------------------------------------------------

fn record_keys(r: &[u8]) -> Vec<u32> {
    let mut keys = vec![];
    let mut pos = 2;
    for _ in 0..r[1] {
        keys.push(u32::from_le_bytes(r[pos..pos + 4].try_into().unwrap()));
        pos += 5 + r[pos + 4] as usize;
    }
    assert_eq!(pos, r.len());
    keys
}

#[test]
fn record_covers_every_patch_option() {
    let o = Opts::default();
    let mut want: Vec<u32> = patch_options(&o).map(|x| x.key().value()).collect();
    let mut got = record_keys(&record(&o));
    want.sort();
    got.sort();
    assert_eq!(got, want, "for_patch_fields! must list every option of the patch pages");
    assert!(record(&o).len() < BLOB_MAX / 2);
}

#[test]
fn loaded_sound_encodes_like_the_original() {
    // upstream decode() moves `init` to the loaded value; the record must not care
    let src = edited(3);
    let mut a = Opts::default();
    decode(&mut a, &record(&src));
    assert_eq!(record(&a), record(&src));
    // and via the upstream per-option persistence path (boot from system save)
    let mut b = Opts::default();
    for (dst, s) in patch_pages_mut(&mut b).zip(patch_options(&src)) {
        let mut buf = [0u8; 16];
        if let Some(n) = s.encode(&mut buf) {
            dst.decode(&buf[..n]);
        }
    }
    assert_eq!(record(&b), record(&src));
}

#[test]
fn round_trip_restores_patch_options() {
    let src = edited(3);
    let r = record(&src);

    let mut dst = Opts::default();
    assert!(decode(&mut dst, &r));
    assert_eq!(values(&dst), values(&src));
    assert_eq!(patch_hash(&dst), patch_hash(&src));
}

#[test]
fn decode_resets_options_missing_from_the_record() {
    let mut dst = edited(1);
    assert!(decode(&mut dst, &record(&Opts::default())));
    assert_eq!(values(&dst), values(&Opts::default()));
}

#[test]
fn system_pages_are_not_part_of_a_preset() {
    let mut src = Opts::default();
    src.cv.in0.tick_up();
    src.misc.midi_ch.tick_up();
    assert_eq!(record(&src), record(&Opts::default()));

    let mut dst = Opts::default();
    dst.cv.in2.tick_up();
    let cv_before = dst.cv.in2.value();
    decode(&mut dst, &record(&edited(0)));
    assert_eq!(dst.cv.in2.value(), cv_before, "load keeps CV routing");
}

#[test]
fn unknown_keys_are_skipped() {
    let src = edited(2);
    let mut r = record(&src);
    // add an entry for an option that doesn't exist (anymore)
    r[1] += 1;
    r.extend_from_slice(&0x1234_5678u32.to_le_bytes());
    r.extend_from_slice(&[3, 1, 2, 3]);
    let mut dst = Opts::default();
    assert!(decode(&mut dst, &r));
    assert_eq!(values(&dst), values(&src));
}

#[test]
fn bad_records_load_the_default_sound() {
    let good = record(&edited(4));
    let mut wrong_format = good.clone();
    wrong_format[0] = FORMAT + 1;
    for bad in [&good[..good.len() - 1], &wrong_format[..], &[][..]] {
        let mut dst = edited(0);
        assert!(!decode(&mut dst, bad));
        assert_eq!(values(&dst), values(&Opts::default()));
    }
}

#[test]
fn edit_buffer_round_trip() {
    let src = edited(5);
    let mut buf = [0u8; BLOB_MAX + 8];
    let n = encode_edit(&src, Some(19), 0xdead_beef, &mut buf).unwrap();
    let mut dst = Opts::default();
    assert_eq!(decode_edit(&mut dst, &buf[..n]), Some((Some(19), 0xdead_beef)));
    assert_eq!(values(&dst), values(&src));

    let n = encode_edit(&src, None, 7, &mut buf).unwrap();
    assert_eq!(decode_edit(&mut Opts::default(), &buf[..n]), Some((None, 7)));
}

#[test]
fn locations() {
    let mut o = Opts::default();
    assert_eq!(selected_slot(&o), 0);
    for slot in 0..N_SLOTS {
        set_location(&mut o, slot);
        assert_eq!(selected_slot(&o), slot);
    }
    assert_eq!(slot_name(0).as_str(), "A1");
    assert_eq!(slot_name(10).as_str(), "B3");
    assert_eq!(slot_name(31).as_str(), "D8");
    set_location(&mut o, 18);
    assert_eq!(o.preset.bank.value(), "C");
    assert_eq!(o.preset.patch.value(), "3");
}

#[test]
fn preset_page_and_keys() {
    let o = Opts::default();
    o.validate_keys_panic_on_failure();
    // the record keys must not change between builds: same key for the same option
    assert_eq!(Opts::default().vcf.cutoff.key().value(), o.vcf.cutoff.key().value());
}

// --- store ------------------------------------------------------------------

#[test]
fn store_all_slots_and_edit_buffer() {
    let size = 14 * SECTOR as u32;
    let mut store = PresetStore::new(RamFlash::new(14), 0..size).unwrap();
    let mut out = [0u8; BLOB_MAX + 8];
    assert_eq!(store.load(3, &mut out), None, "empty slot");

    for slot in 0..N_SLOTS as u32 {
        assert!(store.save(slot, &record(&edited(slot as usize))));
    }
    // rewrite a few slots and the edit buffer many times (forces garbage collection)
    for i in 0..400 {
        assert!(store.save(i % 4, &record(&edited(i as usize % 4))));
        let mut buf = [0u8; BLOB_MAX + 8];
        let n = encode_edit(&edited(i as usize), Some(5), i, &mut buf).unwrap();
        assert!(store.save(KEY_EDIT, &buf[..n]));
    }
    for slot in 0..N_SLOTS as u32 {
        let n = store.load(slot, &mut out).expect("slot present");
        let mut o = Opts::default();
        assert!(decode(&mut o, &out[..n]));
        assert_eq!(values(&o), values(&edited(slot as usize)), "slot {slot}");
    }
    let n = store.load(KEY_EDIT, &mut out).unwrap();
    let mut o = Opts::default();
    assert_eq!(decode_edit(&mut o, &out[..n]), Some((Some(5), 399)));
}

#[test]
fn split_window_keeps_system_part_in_place() {
    assert_eq!(split_window(None), (None, None));
    // old layout (2 pages): everything stays system, no presets
    assert_eq!(split_window(Some(0x1000..0x3000)), (Some(0x1000..0x3000), None));
    let (sys, pre) = split_window(Some(0xE0000..0xF0000));
    assert_eq!(sys, Some(0xE0000..0xE0000 + SYSTEM_SIZE));
    assert_eq!(pre, Some(0xE0000 + SYSTEM_SIZE..0xF0000));
}

// --- control ----------------------------------------------------------------

const A1: u8 = 0;
const A3: u8 = 2;
const C5: u8 = 20;

fn ctl() -> PresetCtl {
    PresetCtl::new(A1, Some(A1), 100, 100)
}

#[test]
fn selecting_a_location_loads_it() {
    let mut c = ctl();
    assert_eq!(c.step(0, A1, true, false, false, false), Action::None);
    assert_eq!(c.step(5, A3, true, false, false, false), Action::Load(A3));
    assert_eq!(c.step(10, A3, true, false, false, false), Action::None);
    // program change loads too (location already set by the ISR)
    assert_eq!(c.step(15, C5, false, false, false, true), Action::Load(C5));
}

#[test]
fn write_needs_two_presses_and_picks_the_target_in_between() {
    let mut c = ctl();
    assert_eq!(c.step(0, A1, true, true, false, false), Action::None);
    assert!(c.armed());
    assert_eq!(c.status(A1, 100).as_str(), "write A1?");
    // while armed, changing bank / patch selects the target, no load
    assert_eq!(c.step(10, C5, true, false, false, false), Action::None);
    assert_eq!(c.status(C5, 100).as_str(), "write C5?");
    assert_eq!(c.step(20, C5, true, true, false, false), Action::Write(C5));
    assert!(!c.armed());
    c.loaded(Some(C5), 100, false);
    assert_eq!(c.step(30, C5, true, false, false, false), Action::None, "no reload after write");
    assert_eq!(c.status(C5, 100).as_str(), "C5");
}

#[test]
fn write_cancels_on_leaving_the_page_or_timeout() {
    for (now, on_page) in [(1000, false), (ARM_TIMEOUT_MS + 1, true)] {
        let mut c = ctl();
        c.step(0, A1, true, true, false, false);
        c.step(10, C5, true, false, false, false);
        assert_eq!(c.step(now, C5, on_page, false, false, false), Action::Revert(A1));
        assert!(!c.armed());
        assert_eq!(c.step(now + 5, A1, on_page, false, false, false), Action::None);
    }
}

#[test]
fn program_change_cancels_an_armed_write() {
    let mut c = ctl();
    c.step(0, A1, true, true, false, false);
    assert_eq!(c.step(10, C5, true, false, false, true), Action::Load(C5));
    assert!(!c.armed());
}

#[test]
fn init_and_edited_marker() {
    let mut c = ctl();
    assert_eq!(c.status(A1, 100).as_str(), "A1");
    assert_eq!(c.status(A1, 101).as_str(), "A1*");
    assert_eq!(c.step(0, A1, true, false, true, false), Action::Init);
    c.loaded(None, 7, false);
    assert_eq!(c.status(A1, 7).as_str(), "init");
    assert_eq!(c.status(A1, 8).as_str(), "init*");
    c.loaded(Some(A3), 0, true);
    assert_eq!(c.status(A3, 0).as_str(), "A3 empty");
    assert_eq!(c.status(A3, 1).as_str(), "A3*");
}

#[test]
fn autosave_after_the_last_edit() {
    let mut c = ctl();
    assert!(!c.autosave_due(0, 100), "nothing changed");
    assert!(!c.autosave_due(10, 101));
    assert!(!c.autosave_due(1000, 102), "still editing");
    assert!(!c.autosave_due(1000 + AUTOSAVE_MS - 1, 102));
    assert!(c.autosave_due(1000 + AUTOSAVE_MS, 102));
    c.autosaved(102);
    assert!(!c.autosave_due(10_000, 102));
    // a load changes the location: saved as well
    c.loaded(Some(A3), 102, false);
    assert!(c.autosave_due(20_000, 102));
}
