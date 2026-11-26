#![no_std]
#![no_main]

pub use tiliqua_pac as pac;
pub use tiliqua_hal as hal;

hal::impl_tiliqua_soc_pac!();

// Own handlers (critical-section logger), see handlers.rs
pub mod handlers;
tiliqua_lib::impl_tick!(Clint);

pub mod options;
pub mod preset;
pub mod synth;
pub mod dco;
pub mod lfo;
pub mod glide;
pub mod variance;
pub mod out;
pub mod cv;
pub mod vcf;
pub mod env;

#[cfg(feature = "stack_watch")]
pub mod stack_watch;
#[cfg(feature = "sim_stack")]
pub mod sim_stack;

pub use synth::Synth;
