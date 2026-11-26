//! Presets (Phase 6, docs/PRESETS_PLAN.md): 32 slots A1..D8 holding the patch
//! options (VOICE, LFO, DCO, VCF, ENV, FX pages; CV / MISC / HELP are system
//! settings and stay in the upstream option save).
//!
//! - **Record** (`encode` / `decode`): `FORMAT`, entry count, then per patch option
//!   `key u32 LE, len u8, postcard(value)`.
//!   Unknown keys are skipped and missing options load as defaults, so options
//!   can be added / removed without invalidating stored presets.
//! - **Store** (`PresetStore`): own `sequential_storage` map in the option storage
//!   window behind the upstream system part (`split_window`), so wiping the system
//!   settings keeps the presets. Keys: slot 0..31, `KEY_EDIT` = edit buffer.
//! - **Edit buffer**: current patch + active location, saved automatically
//!   `AUTOSAVE_MS` after the last edit and restored at boot.
//! - **Control** (`PresetCtl`): load on bank / patch change, two-press write, init,
//!   edited marker. Pure logic; flash IO and option updates happen in `main.rs`.

use core::fmt::Write as _;
use heapless::String;
use opts::{OptionPage, OptionString, OptionTrait};
use strum::IntoEnumIterator;

use crate::options::{AdsrOpts, Bank, DcoOpts, EffectOpts, LfoOpts, Opts, PatchNum, VcfOpts, VoiceOpts};

pub const N_SLOTS: u8 = 32;
pub const FORMAT: u8 = 1;
/// Max record size (header + entries). Today's 32 patch options need ~230 B.
pub const BLOB_MAX: usize = 512;
const ENTRY_MAX: usize = 16;

/// Size of the upstream system-options part at the start of the window (the
/// former whole window, so existing system settings stay where they are).
pub const SYSTEM_SIZE: u32 = 0x2000;

pub const KEY_EDIT: u32 = 0x100;
const STATE_LEN: usize = 6;
const NO_SLOT: u8 = 0xFF;

/// Write stays armed this long (or until leaving the PRESET page).
pub const ARM_TIMEOUT_MS: u32 = 15_000;
/// Edit buffer is saved this long after the last change.
pub const AUTOSAVE_MS: u32 = 2_000;

// --- patch options --------------------------------------------------------

macro_rules! patch_pages {
    ($o:expr, $method:ident) => {
        [$o.voice.$method(), $o.lfo.$method(), $o.dco.$method(),
         $o.vcf.$method(), $o.adsr.$method(), $o.effect.$method()]
    };
}

// Every patch option, explicitly, per page: the record stores `value` directly.
// (The upstream `OptionTrait::encode` skips values equal to `init`, and `decode`
// moves `init` to the loaded value, so after a load it would drop options.)
// `test_preset.rs` checks that this list covers all options of the patch pages.
macro_rules! patch_layout {
    ($m:ident) => {
        $m!(voice: VoiceOpts [assign, porta, detune, variance, bend, bend_vcf, wheel]);
        $m!(lfo: LfoOpts [rate, delay]);
        $m!(dco: DcoOpts [range, lfo, pw, pwm_mode, pulse, saw, sub, noise]);
        $m!(vcf: VcfOpts [hpf, cutoff, reso, env_pol, env_amt, lfo, kbd_track]);
        $m!(adsr: AdsrOpts [vca_mode, attack, decay, sustain, release]);
        $m!(effect: EffectOpts [chorus, level, drive]);
    };
}

/// The options of the patch pages (for checks; the record uses `patch_layout`).
pub fn patch_options(o: &Opts) -> impl Iterator<Item = &dyn OptionTrait> {
    patch_pages!(o, options).into_iter().flatten()
}

/// Reset all patch options to their defaults (values only; uses the small
/// per-page defaults, no `Opts` temporary: data + stacks share 16 KiB).
pub fn reset(o: &mut Opts) {
    macro_rules! page {
        ($p:ident: $t:ident [$($f:ident),*]) => {{
            let d = $t::default();
            $( o.$p.$f.value = d.$f.value; )*
        }};
    }
    patch_layout!(page);
}

/// Copy the patch option values of `src` into `dst` (other pages untouched).
pub fn copy_patch(dst: &mut Opts, src: &Opts) {
    macro_rules! page {
        ($p:ident: $t:ident [$($f:ident),*]) => { $( dst.$p.$f.value = src.$p.$f.value; )* };
    }
    patch_layout!(page);
}

/// Encode the patch options into `buf` (all of them, defaults included).
/// Returns the record length, or `None` if it does not fit.
pub fn encode(o: &Opts, buf: &mut [u8]) -> Option<usize> {
    if buf.len() < 2 {
        return None;
    }
    buf[0] = FORMAT;
    let mut n: u8 = 0;
    let mut pos = 2;
    macro_rules! put {
        ($opt:expr) => {{
            let mut e = [0u8; ENTRY_MAX];
            let len = postcard::to_slice(&$opt.value, &mut e).ok()?.len();
            let end = pos + 5 + len;
            if end > buf.len() {
                return None;
            }
            buf[pos..pos + 4].copy_from_slice(&$opt.key().value().to_le_bytes());
            buf[pos + 4] = len as u8;
            buf[pos + 5..end].copy_from_slice(&e[..len]);
            pos = end;
            n += 1;
        }};
    }
    macro_rules! page {
        ($p:ident: $t:ident [$($f:ident),*]) => { $( put!(o.$p.$f); )* };
    }
    patch_layout!(page);
    buf[1] = n;
    Some(pos)
}

/// Walk the entries of a record: `f(key, value bytes)`. False if the record is
/// malformed (unknown format, truncated).
fn entries(blob: &[u8], mut f: impl FnMut(u32, &[u8])) -> bool {
    if blob.len() < 2 || blob[0] != FORMAT {
        return false;
    }
    let mut pos = 2;
    for _ in 0..blob[1] {
        if pos + 5 > blob.len() {
            return false;
        }
        let key = u32::from_le_bytes([blob[pos], blob[pos + 1], blob[pos + 2], blob[pos + 3]]);
        let end = pos + 5 + blob[pos + 4] as usize;
        if end > blob.len() {
            return false;
        }
        f(key, &blob[pos + 5..end]);
        pos = end;
    }
    pos == blob.len()
}

/// Load a record into the patch options: defaults first, then every known
/// entry (unknown keys / undecodable values are skipped). A malformed record
/// loads the defaults and returns false.
pub fn decode(o: &mut Opts, blob: &[u8]) -> bool {
    reset(o);
    if !entries(blob, |_, _| ()) {
        return false;
    }
    entries(blob, |key, bytes| {
        macro_rules! get {
            ($opt:expr) => {
                if $opt.key().value() == key {
                    if let Ok(v) = postcard::from_bytes(bytes) {
                        $opt.value = v;
                    }
                    return;
                }
            };
        }
        macro_rules! page {
            ($p:ident: $t:ident [$($f:ident),*]) => { $( get!(o.$p.$f); )* };
        }
        patch_layout!(page);
    })
}

/// FNV-1a over the record: identifies the current sound (edited marker,
/// autosave).
pub fn hash(blob: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in blob {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

pub fn patch_hash(o: &Opts) -> u32 {
    let mut buf = [0u8; BLOB_MAX];
    encode(o, &mut buf).map(|n| hash(&buf[..n])).unwrap_or(0)
}

// --- locations ------------------------------------------------------------

pub fn selected_slot(o: &Opts) -> u8 {
    (o.preset.bank.value as u8) * 8 + o.preset.patch.value as u8
}

pub fn set_location(o: &mut Opts, slot: u8) {
    let slot = slot % N_SLOTS;
    if let Some(b) = Bank::iter().nth((slot / 8) as usize) {
        o.preset.bank.value = b;
    }
    if let Some(p) = PatchNum::iter().nth((slot % 8) as usize) {
        o.preset.patch.value = p;
    }
}

pub fn slot_name(slot: u8) -> String<4> {
    let mut s = String::new();
    write!(&mut s, "{}{}", (b'A' + slot / 8) as char, slot % 8 + 1).ok();
    s
}

// --- edit buffer record ---------------------------------------------------

/// Edit buffer record: `[1, active slot (0xFF = none), loaded hash u32 LE]` + patch record.
pub fn encode_edit(o: &Opts, active: Option<u8>, snap: u32, buf: &mut [u8]) -> Option<usize> {
    if buf.len() < STATE_LEN {
        return None;
    }
    buf[0] = 1;
    buf[1] = active.unwrap_or(NO_SLOT);
    buf[2..STATE_LEN].copy_from_slice(&snap.to_le_bytes());
    encode(o, &mut buf[STATE_LEN..]).map(|n| STATE_LEN + n)
}

/// Restore the edit buffer into `o`. Returns `(active, loaded hash)`.
pub fn decode_edit(o: &mut Opts, rec: &[u8]) -> Option<(Option<u8>, u32)> {
    if rec.len() < STATE_LEN || rec[0] != 1 || !decode(o, &rec[STATE_LEN..]) {
        return None;
    }
    let active = if rec[1] < N_SLOTS { Some(rec[1]) } else { None };
    Some((active, u32::from_le_bytes([rec[2], rec[3], rec[4], rec[5]])))
}

// --- control --------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    /// Load this slot (empty slot: default sound).
    Load(u8),
    /// Store the current sound to this slot.
    Write(u8),
    /// Load the default sound, no active location.
    Init,
    /// Write was cancelled: show this location again (no load).
    Revert(u8),
}

pub struct PresetCtl {
    /// Location of the loaded / written sound, `None` after init.
    pub active: Option<u8>,
    /// The active slot was empty when loaded.
    pub empty: bool,
    /// Hash of the sound when loaded / written (edited marker).
    snap: u32,
    armed_at: Option<u32>,
    last_sel: u8,
    // edit buffer autosave
    saved: (u32, Option<u8>, u32),
    last_hash: u32,
    changed_at: u32,
}

impl PresetCtl {
    /// `sel`: the location shown at boot, `hash`: the restored sound.
    pub fn new(sel: u8, active: Option<u8>, snap: u32, hash: u32) -> Self {
        Self {
            active, empty: false, snap, armed_at: None, last_sel: sel,
            saved: (hash, active, snap), last_hash: hash, changed_at: 0,
        }
    }

    pub fn armed(&self) -> bool {
        self.armed_at.is_some()
    }

    /// One main-loop step. `sel`: selected location, `force`: program change
    /// (loads even while write is armed).
    pub fn step(&mut self, now: u32, sel: u8, on_page: bool, write: bool, init: bool,
                force: bool) -> Action {
        if init {
            self.armed_at = None;
            self.last_sel = sel;
            return Action::Init;
        }
        if force {
            self.armed_at = None;
        }
        if write {
            if self.armed_at.take().is_some() {
                self.last_sel = sel;
                return Action::Write(sel);
            }
            self.armed_at = Some(now);
            return Action::None;
        }
        if let Some(t) = self.armed_at {
            if on_page && now.wrapping_sub(t) < ARM_TIMEOUT_MS {
                return Action::None;    // bank / patch pick the write target
            }
            self.armed_at = None;
            if sel != self.last_sel {
                if let Some(a) = self.active {
                    self.last_sel = a;
                    return Action::Revert(a);
                }
                self.last_sel = sel;
            }
            return Action::None;
        }
        if sel != self.last_sel {
            self.last_sel = sel;
            return Action::Load(sel);
        }
        Action::None
    }

    /// A sound was loaded / written / initialised (`hash` = its patch hash).
    pub fn loaded(&mut self, active: Option<u8>, hash: u32, empty: bool) {
        self.active = active;
        self.snap = hash;
        self.empty = empty;
    }

    pub fn edited(&self, hash: u32) -> bool {
        hash != self.snap
    }

    pub fn snap(&self) -> u32 {
        self.snap
    }

    pub fn status(&self, sel: u8, hash: u32) -> OptionString {
        let mut s = OptionString::new();
        if self.armed() {
            write!(&mut s, "write {}?", slot_name(sel)).ok();
            return s;
        }
        match self.active {
            Some(a) => { write!(&mut s, "{}", slot_name(a)).ok(); }
            None => { write!(&mut s, "init").ok(); }
        }
        if self.edited(hash) {
            write!(&mut s, "*").ok();
        } else if self.empty {
            write!(&mut s, " empty").ok();
        }
        s
    }

    /// True when the edit buffer should be saved now (changed and idle for
    /// `AUTOSAVE_MS`).
    pub fn autosave_due(&mut self, now: u32, hash: u32) -> bool {
        if hash != self.last_hash {
            self.last_hash = hash;
            self.changed_at = now;
        }
        (hash, self.active, self.snap) != self.saved
            && now.wrapping_sub(self.changed_at) >= AUTOSAVE_MS
    }

    pub fn autosaved(&mut self, hash: u32) {
        self.saved = (hash, self.active, self.snap);
    }
}

// --- flash store ----------------------------------------------------------

pub use store::*;

mod store {
    use core::cell::RefCell;
    use embassy_embedded_hal::adapter::BlockingAsync;
    use embassy_embedded_hal::flash::partition::BlockingPartition;
    use embassy_futures::block_on;
    use embassy_sync::blocking_mutex::Mutex;
    use embassy_sync::blocking_mutex::raw::RawMutex;
    use embedded_storage::nor_flash::{MultiwriteNorFlash, NorFlash};
    use sequential_storage::cache::{Cache, Uncached};
    use sequential_storage::map::{MapConfig, MapStorage};

    use super::{BLOB_MAX, SYSTEM_SIZE};

    /// Split the option storage window into the upstream system part and the
    /// preset part. Windows not larger than `SYSTEM_SIZE` (older layout) keep
    /// everything for the system part: no presets.
    pub fn split_window(window: Option<core::ops::Range<u32>>)
        -> (Option<core::ops::Range<u32>>, Option<core::ops::Range<u32>>) {
        match window {
            Some(w) if w.end - w.start > SYSTEM_SIZE => {
                let mid = w.start + SYSTEM_SIZE;
                (Some(w.start..mid), Some(mid..w.end))
            }
            w => (w, None),
        }
    }

    pub struct PresetStore<F: NorFlash + MultiwriteNorFlash> {
        storage: MapStorage<u32, BlockingAsync<F>, Cache<Uncached, Uncached, Uncached, u32>>,
        buf: [u8; BLOB_MAX + 64],
    }

    impl<F: NorFlash + MultiwriteNorFlash> PresetStore<F> {
        pub fn new(flash: F, range: core::ops::Range<u32>) -> Option<Self> {
            let config = MapConfig::try_new(range).ok()?;
            Some(Self {
                storage: MapStorage::new(BlockingAsync::new(flash), config, Cache::new_uncached()),
                buf: [0u8; BLOB_MAX + 64],
            })
        }

        pub fn save(&mut self, key: u32, rec: &[u8]) -> bool {
            match block_on(self.storage.store_item(&mut self.buf, &key, &rec)) {
                Ok(()) => true,
                Err(e) => {
                    log::error!("preset: store {key:x} failed: {e:?}");
                    false
                }
            }
        }

        /// Copy the record for `key` into `out`, returns its length.
        pub fn load(&mut self, key: u32, out: &mut [u8]) -> Option<usize> {
            match block_on(self.storage.fetch_item::<&[u8]>(&mut self.buf, &key)) {
                Ok(Some(data)) => {
                    let n = data.len().min(out.len());
                    out[..n].copy_from_slice(&data[..n]);
                    Some(n)
                }
                Ok(None) => None,
                Err(e) => {
                    log::error!("preset: fetch {key:x} failed: {e:?}");
                    None
                }
            }
        }
    }

    impl<'a, M: RawMutex, F: NorFlash + MultiwriteNorFlash> PresetStore<BlockingPartition<'a, M, F>> {
        pub fn open_partition(flash: &'a Mutex<M, RefCell<F>>,
                              window: Option<core::ops::Range<u32>>) -> Option<Self> {
            let Some(w) = window else {
                log::warn!("preset: no preset storage region: presets disabled");
                return None;
            };
            let erase = F::ERASE_SIZE as u32;
            if !w.start.is_multiple_of(erase) || !w.end.is_multiple_of(erase) {
                log::warn!("preset: window {w:x?} not erase-aligned: presets disabled");
                return None;
            }
            let size = w.end - w.start;
            Self::new(BlockingPartition::new(flash, w.start, size), 0..size)
        }
    }
}
