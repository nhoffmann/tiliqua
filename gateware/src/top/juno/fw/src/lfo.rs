//! Juno LFO helpers: knob (0..32768) -> `JunoLfo` register values.
//!
//! Physical units -> register values use the same formulas as `sim_render.py`
//! (rate_inc = f / 48000 * 2^32, delay_step = 2^30 / (t * 48000),
//! dco_depth = semitones * 1024, vcf_depth = normalized octaves * 32768).

use micromath::F32Ext;

const KNOB_ONE: f32 = 32768.0;
const FS: f32 = 48000.0;

fn knob(k: u16) -> f32 {
    f32::from(k) / KNOB_ONE
}

/// Rate knob -> LFO frequency in Hz, exponential 0.1 .. 20 Hz.
pub fn rate_hz(k: u16) -> f32 {
    0.1 * 200.0f32.powf(knob(k))
}

/// Rate knob -> phase increment per frame, UQ(0,32).
pub fn rate_inc(k: u16) -> u32 {
    (rate_hz(k) / FS * 4294967296.0) as u32
}

/// Delay knob -> amplitude step per frame (linear fade-in over 0 .. 1.5 s).
pub fn delay_step(k: u16) -> u32 {
    let seconds = 1.5 * knob(k);
    let frames = seconds * FS;
    if frames < 1.0 {
        1 << 30
    } else {
        ((1u32 << 30) as f32 / frames).max(1.0) as u32
    }
}

/// DCO LFO knob -> vibrato depth in 1/1024 semitones. Juno pots are audio
/// taper; 2.5 * k^2.5 semitones (about 0.44 at half travel, 2.5 at full).
pub fn dco_depth(k: u16) -> u16 {
    let semitones = 2.5 * knob(k).powf(2.5);
    (semitones * 1024.0) as u16
}

/// VCF LFO knob -> cutoff depth in normalized octaves (15 bit), up to 3 octaves.
pub fn vcf_depth(k: u16) -> u16 {
    (0.3 * knob(k) * KNOB_ONE) as u16
}
