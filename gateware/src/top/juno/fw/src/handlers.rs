//! Logger, panic, exception and default interrupt handlers for the juno
//! firmware. Same as upstream `tiliqua_lib::impl_handlers!` (used by the other
//! examples), except for the logger:
//!
//! The upstream logger guards the UART with a `RefCell` only. A log call from
//! the tick ISR while the main loop is in the middle of a log line then panics
//! ("already borrowed"), and the panic handler's own log call fails the same way:
//! the firmware hangs without any output. Seen in the full-SoC simulation (page
//! logs from the ISR during main-loop logging); on hardware the same could
//! happen with MISC `serial-debug` when the ISR still logged MIDI.
//!
//! Here each log line is written inside a critical section, so ISR and main
//! loop log lines can't interleave. Interrupts stay off while a line goes out
//! (a few ms at 115200 baud): the tick ISR is delayed, not lost; audio is
//! gateware and unaffected. `try_borrow_mut` keeps a panic inside a write from
//! recursing.

use core::cell::RefCell;
use core::fmt::Write;
use critical_section::Mutex;
use log::{Level, LevelFilter, Metadata, Record};

use crate::Serial0;

struct CsLogger {
    writer: Mutex<RefCell<Option<Serial0>>>,
}

impl log::Log for CsLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Trace
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let color = match record.level() {
            Level::Error => "31", // red
            Level::Warn => "33",  // yellow
            _ => "32",            // green
        };
        critical_section::with(|cs| {
            if let Ok(mut writer) = self.writer.borrow(cs).try_borrow_mut() {
                if let Some(w) = writer.as_mut() {
                    // A failed UART write is dropped (nothing else could report it).
                    let _ = writeln!(w, "[\x1B[{}m{}\x1B[0m] {}\r", color, record.level(), record.args());
                }
            }
        });
    }

    fn flush(&self) {}
}

static LOGGER: CsLogger = CsLogger { writer: Mutex::new(RefCell::new(None)) };

pub fn logger_init(writer: Serial0) {
    critical_section::with(|cs| {
        LOGGER.writer.borrow(cs).replace(Some(writer));
    });
    unsafe {
        if log::set_logger_racy(&LOGGER).is_err() {
            panic!("Failed to set logger");
        }
        log::set_max_level_racy(LevelFilter::Trace);
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
    if let Some(location) = panic_info.location() {
        log::error!("panic(): file '{}' at line {}", location.file(), location.line());
    } else {
        log::error!("panic(): no location information");
    }
    log::error!("{:?}", panic_info.message());
    loop {}
}

#[export_name = "ExceptionHandler"]
fn exception_handler(trap_frame: &riscv_rt::TrapFrame) -> ! {
    log::error!("exception_handler(): TrapFrame.ra={:x}", trap_frame.ra);
    loop {}
}

#[export_name = "DefaultHandler"]
fn default_isr_handler() {
    log::error!("default_isr_handler(): unhandled interrupt");
    loop {}
}
