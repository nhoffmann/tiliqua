//! Stack high-water measurement (`--features stack_watch`).
//!
//! The main loop and the tick ISR share one stack: block RAM from the end of the
//! statics (`__estack`) up to `__sstack` (16 KiB total with data/bss). This module
//! fills the free part with a pattern and later finds the lowest overwritten word.
//!
//! - [`paint`] at the start of `main` (interrupts still off).
//! - [`isr_enter`] / [`isr_exit`] around the tick ISR body: on entry the global
//!   peak is recorded first, then everything below the ISR's own stack pointer is
//!   painted again, so on exit the lowest overwritten word below the entry SP is
//!   the ISR body's own depth. The next scan still sees any deeper main-loop use,
//!   because only free stack (below the live SP) is ever repainted.
//! - [`scan`] in the main loop updates the global peak; [`report`] logs it.
//!
//! Not counted in the ISR depth: the trap entry above the handler (riscv-rt pushes
//! 64 B, plus the IRQ dispatch frames); the global peak includes them.

use core::ptr::{addr_of, read_volatile, write_volatile};
use portable_atomic::{AtomicU32, Ordering};

const PATTERN: u32 = 0x5AC3_5AC3;
/// Bytes kept unpainted below the caller's SP (this function's own frame lives there).
const MARGIN: usize = 128;

extern "C" {
    static __estack: u32;
    static __sstack: u32;
}

static PEAK: AtomicU32 = AtomicU32::new(0);       // lowest address ever used (0 = none yet)
static ISR_DEPTH: AtomicU32 = AtomicU32::new(0);  // max ISR body depth in bytes
static ISR_SP: AtomicU32 = AtomicU32::new(0);     // SP at the last ISR entry

fn bottom() -> usize { addr_of!(__estack) as usize }
/// Top of the stack (initial SP).
pub fn top() -> usize { addr_of!(__sstack) as usize }

#[inline(always)]
fn sp() -> usize {
    let sp: usize;
    unsafe { core::arch::asm!("mv {}, sp", out(reg) sp) };
    sp
}

#[inline(never)]
fn paint_below(limit: usize) {
    let mut p = (bottom() + 3) & !3;
    while p + 4 <= limit {
        unsafe { write_volatile(p as *mut u32, PATTERN) };
        p += 4;
    }
}

/// Lowest overwritten address (searching up from the bottom).
fn lowest_used() -> usize {
    let mut p = (bottom() + 3) & !3;
    let end = top();
    while p < end && unsafe { read_volatile(p as *const u32) } == PATTERN {
        p += 4;
    }
    p
}

/// Free bytes below which `report` warns once (main loop only: no logging
/// from the ISR).
const LOW_WARN: usize = 1024;
static WARNED: portable_atomic::AtomicBool = portable_atomic::AtomicBool::new(false);

fn update_peak() {
    let low = lowest_used() as u32;
    let _ = PEAK.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
                              |p| if p == 0 || low < p { Some(low) } else { None });
}

/// Paint the whole free stack. Call first thing in `main`, interrupts disabled.
pub fn paint() {
    paint_below(sp() - MARGIN);
}

/// Update the global peak (main loop).
pub fn scan() {
    update_peak();
}

/// Tick ISR entry: record the peak, then repaint below the current SP.
pub fn isr_enter() {
    update_peak();
    let s = sp();
    ISR_SP.store(s as u32, Ordering::Relaxed);
    paint_below(s - MARGIN);
}

/// Tick ISR exit: depth of the ISR body below its entry SP.
pub fn isr_exit() {
    let entry = ISR_SP.load(Ordering::Relaxed) as usize;
    let low = lowest_used();
    if low < entry {
        ISR_DEPTH.fetch_max((entry - low) as u32, Ordering::Relaxed);
    }
    update_peak();
}

/// (peak bytes used, bytes never touched, ISR body max depth)
pub fn stats() -> (usize, usize, usize) {
    let peak = match PEAK.load(Ordering::Relaxed) as usize { 0 => top(), p => p };
    (top() - peak, peak - bottom(), ISR_DEPTH.load(Ordering::Relaxed) as usize)
}

/// Log when the peak or the ISR depth grew, and every `every_ms` regardless.
pub fn report(now_ms: u32, every_ms: u32) {
    static LAST: AtomicU32 = AtomicU32::new(0);
    static LAST_MS: AtomicU32 = AtomicU32::new(0);
    let (used, free, isr) = stats();
    if free < LOW_WARN && !WARNED.swap(true, Ordering::Relaxed) {
        log::warn!("stack_watch: LOW, {} B free", free);
    }
    let key = (used as u32) << 16 | isr as u32;
    let due = now_ms.wrapping_sub(LAST_MS.load(Ordering::Relaxed)) >= every_ms;
    if key != LAST.load(Ordering::Relaxed) || due {
        LAST.store(key, Ordering::Relaxed);
        LAST_MS.store(now_ms, Ordering::Relaxed);
        log::info!("stack_watch: t={}ms peak {} B used, {} B free (of {}), isr body {} B",
                   now_ms, used, free, top() - bottom(), isr);
    }
}
