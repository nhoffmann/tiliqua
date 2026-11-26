#![no_std]
#![no_main]

pub use tiliqua_pac as pac;
pub use tiliqua_hal as hal;

hal::impl_tiliqua_soc_pac!();

tiliqua_lib::impl_handlers!(Serial0);
tiliqua_lib::impl_tick!(Clint);

pub mod options;
pub mod polysynth;
pub mod dco;
pub mod lfo;
pub mod glide;
pub mod variance;
pub mod out;
pub mod cv;
pub mod vcf;
pub mod env;

pub use polysynth::Polysynth0;
