//! Host tests for the juno firmware: the modules are compiled from `../fw/src`.

#[path = "../fw/src/options.rs"]
pub mod options;
#[path = "../fw/src/preset.rs"]
pub mod preset;

#[cfg(test)]
mod test_preset;
