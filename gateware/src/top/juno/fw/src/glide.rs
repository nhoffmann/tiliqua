//! Juno portamento helper: knob (0..32768) -> `JunoGlide` coefficient register.
//! Same as `juno_glide.py` (`glide_time`, `glide_coef`).
//!
//! Time constant = knob^3 * 3 s (cubic, fine control at the short end);
//! coef = (1 - exp(-1 / (tau * fs))) * 2^24, 18 bit, 0 = portamento off.

use micromath::F32Ext;

const FS: f32 = 48000.0;
const GLIDE_MAX_S: f32 = 3.0;
const COEF_MAX: u32 = (1 << 18) - 1;

/// Portamento knob -> time constant in seconds (0 = off).
pub fn time(k: u16) -> f32 {
    let x = f32::from(k.min(32768)) / 32768.0;
    x * x * x * GLIDE_MAX_S
}

/// Portamento knob -> `glide` register value.
pub fn coef(k: u16) -> u32 {
    let t = time(k);
    if t <= 0.0 {
        return 0;
    }
    let x = 1.0 / (t * FS);
    // 1 - exp(-x); series for small x (f32 exp loses the small difference)
    let c = if x < 0.01 { x - x * x / 2.0 + x * x * x / 6.0 } else { 1.0 - (-x).exp() };
    ((c * 16777216.0 + 0.5) as u32).clamp(1, COEF_MAX)
}
