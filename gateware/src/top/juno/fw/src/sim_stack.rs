//! Simulation only (`--features sim_stack`): drive the firmware through the paths
//! the Verilator SoC harness doesn't reach on its own, so `stack_watch` sees them.
//!
//! - [`PsramFlash`]: NOR flash emulated in PSRAM. The harness's SPI flash can't be
//!   written and its bootinfo has no option storage window, so without this the
//!   preset store and option persistence are disabled in simulation.
//! - [`midi_read`]: scripted MIDI words for the tick ISR (CCs, notes, bend,
//!   program change) instead of the empty hardware FIFO.
//! - [`event`]: scripted main-loop inputs (write presses, slot changes, init,
//!   save / wipe options). The autosave of the edit buffer follows by itself.
//!
//! All times are firmware uptime (ms since the tick started); the harness stops
//! after 5 s simulated time.

use embedded_storage::nor_flash::{
    ErrorType, MultiwriteNorFlash, NorFlash, NorFlashError, NorFlashErrorKind, ReadNorFlash,
};
use portable_atomic::{AtomicUsize, Ordering};

/// Option storage window the firmware uses in simulation (the real one is
/// assigned per slot by the flasher; only the size matters here).
pub const WINDOW: core::ops::Range<u32> = 0x0E_0000..0x0F_0000;
/// MISC `serial-debug` on: the ISR logs every MIDI word (its deepest path).
pub const MIDI_LOG: bool = true;
/// PSRAM backing for flash offset 0 (window lands at 0x20CE_0000, clear of the
/// framebuffer at 0x2000_0000, the firmware at 0x2020_0000 and bootinfo at 0x20FF_F000).
const PSRAM_FLASH_BASE: usize = 0x20C0_0000;

#[derive(Debug)]
pub struct OutOfBounds;

impl NorFlashError for OutOfBounds {
    fn kind(&self) -> NorFlashErrorKind { NorFlashErrorKind::OutOfBounds }
}

pub struct PsramFlash;

impl PsramFlash {
    /// Erase the window (PSRAM starts zeroed in the harness; erased NOR is 0xFF).
    pub fn new() -> Self {
        let mut f = PsramFlash;
        f.erase(WINDOW.start, WINDOW.end).ok();
        f
    }

    fn check(&self, from: u32, len: usize) -> Result<*mut u8, OutOfBounds> {
        if from < WINDOW.start || from as usize + len > WINDOW.end as usize {
            return Err(OutOfBounds);
        }
        Ok((PSRAM_FLASH_BASE + from as usize) as *mut u8)
    }
}

impl ErrorType for PsramFlash {
    type Error = OutOfBounds;
}

impl ReadNorFlash for PsramFlash {
    const READ_SIZE: usize = 1;
    fn read(&mut self, offset: u32, bytes: &mut [u8]) -> Result<(), Self::Error> {
        let p = self.check(offset, bytes.len())?;
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = unsafe { core::ptr::read_volatile(p.add(i)) };
        }
        Ok(())
    }
    fn capacity(&self) -> usize { WINDOW.end as usize }
}

impl NorFlash for PsramFlash {
    const WRITE_SIZE: usize = 1;
    const ERASE_SIZE: usize = 4096;
    fn erase(&mut self, from: u32, to: u32) -> Result<(), Self::Error> {
        let p = self.check(from, (to - from) as usize)? as *mut u32;
        for i in 0..((to - from) as usize / 4) {
            unsafe { core::ptr::write_volatile(p.add(i), 0xFFFF_FFFF) };
        }
        Ok(())
    }
    fn write(&mut self, offset: u32, bytes: &[u8]) -> Result<(), Self::Error> {
        let p = self.check(offset, bytes.len())?;
        for (i, b) in bytes.iter().enumerate() {
            unsafe {
                let old = core::ptr::read_volatile(p.add(i));
                core::ptr::write_volatile(p.add(i), old & b);   // NOR: bits only clear
            }
        }
        Ok(())
    }
}

impl MultiwriteNorFlash for PsramFlash {}

/// MIDI words as the hardware FIFO delivers them: status | d1 << 8 | d2 << 16.
const fn w(s: u8, d1: u8, d2: u8) -> u32 { s as u32 | (d1 as u32) << 8 | (d2 as u32) << 16 }

/// (uptime ms, word). CCs from docs/MIDI_CV.md.
const MIDI: &[(u32, u32)] = &[
    (300, w(0x90, 60, 100)), (300, w(0x90, 64, 100)), (300, w(0x90, 67, 100)),
    (305, w(0xB0, 1, 90)),   // mod wheel
    (305, w(0xB0, 7, 80)),   // FX level
    (310, w(0xB0, 19, 40)),  // DCO pw
    (310, w(0xB0, 5, 20)),   // porta
    (315, w(0xE0, 0, 96)),   // pitch bend
    (320, w(0xB0, 64, 127)), // sustain
    (320, w(0x80, 60, 0)), (320, w(0x80, 64, 0)), (320, w(0x80, 67, 0)),
    (2300, w(0xC0, 5, 0)),   // program change -> A6 (written at 1.4 s)
    (2305, w(0xC0, 40, 0)),  // >= 32: ignored
];

static MIDI_POS: AtomicUsize = AtomicUsize::new(0);

/// Next scripted MIDI word due at `now`, else 0 (= FIFO empty).
pub fn midi_read(now: u32) -> u32 {
    let i = MIDI_POS.load(Ordering::Relaxed);
    match MIDI.get(i) {
        Some(&(t, word)) if now >= t => {
            MIDI_POS.store(i + 1, Ordering::Relaxed);
            word
        }
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// PRESET `write` press (first arms, second writes)
    Write,
    /// Select a slot (bank / patch knobs); loads it unless a write is armed
    Slot(u8),
    Init,
    SaveOpts,
    WipeOpts,
}

const EVENTS: &[(u32, Event)] = &[
    (600, Event::Write), (700, Event::Write),       // write A1
    (1000, Event::Slot(5)),                         // load A6 (empty)
    (1300, Event::Write), (1400, Event::Write),     // write A6
    (1700, Event::Slot(0)),                         // load A1 (decode)
    (2000, Event::Init),
    (2600, Event::SaveOpts),
    // edit buffer autosave ~2 s after the last patch change (PC at 2.3 s)
    (4700, Event::WipeOpts),
];

static EVENT_POS: AtomicUsize = AtomicUsize::new(0);

/// Next scripted main-loop event due at `now`.
pub fn event(now: u32) -> Option<Event> {
    let i = EVENT_POS.load(Ordering::Relaxed);
    match EVENTS.get(i) {
        Some(&(t, e)) if now >= t => {
            EVENT_POS.store(i + 1, Ordering::Relaxed);
            log::info!("sim_stack: t={}ms {:?}", now, e);
            Some(e)
        }
        _ => None,
    }
}
