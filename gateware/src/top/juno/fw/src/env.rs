//! Juno envelope helpers: ADSR knob (0..32768 = Juno-106 slider / 127) -> `MultiADSR`
//! register values. Tables measured on the KR-106; same as `adsr_model.py`
//! (`ATTACK_S`, `DECAY_S`, `RELEASE_S`, `knob_to_time`).

use micromath::F32Ext;

const SLIDERS: [f32; 9] = [0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 127.0];
/// Seconds at the slider points: attack to full level, decay / release to -60 dB.
/// Attack: time to FULL level (measured time to 95 % / 0.95, linear ramp).
pub const ATTACK_S:  [f32; 9] = [0.0016, 0.1347, 0.2632, 0.3979, 0.5347, 0.8168, 1.3263, 2.1537, 3.2063];
pub const DECAY_S:   [f32; 9] = [0.0087, 0.3406, 0.7812, 3.5125, 5.7882, 8.1155, 11.7507, 22.6675, 75.0];
pub const RELEASE_S: [f32; 9] = [0.0120, 0.3553, 0.7442, 3.0000, 4.7017, 6.6780, 11.1710, 24.0906, 75.0];

const FS: f32 = 48000.0;
const L_MAX: f32 = 4294967295.0;
const EPS_SCALE: f32 = 524288.0;          // 2^19
const LN10_X3: f32 = 6.907755;            // 3 ln(10): -60 dB

/// Knob -> seconds, log-interpolated between the slider points.
pub fn knob_to_time(knob: u16, table: &[f32; 9]) -> f32 {
    let sl = f32::from(knob.min(32768)) / 32768.0 * 127.0;
    let mut i = 0;
    while i < 7 && sl > SLIDERS[i + 1] {
        i += 1;
    }
    let f = (sl - SLIDERS[i]) / (SLIDERS[i + 1] - SLIDERS[i]);
    let (a, b) = (table[i].ln(), table[i + 1].ln());
    (a + f * (b - a)).exp()
}

/// Attack time -> 32-bit level increment per sample.
pub fn attack_inc(seconds: f32) -> u32 {
    let inc = L_MAX / (seconds.max(1.0 / FS) * FS);
    if inc >= L_MAX { u32::MAX } else { (inc as u32).max(1) }
}

/// Decay / release time (to -60 dB) -> eps (17 bit, 19 fractional bits).
pub fn eps(seconds: f32) -> u32 {
    let e = LN10_X3 / (seconds.max(1.0 / FS) * FS) * EPS_SCALE + 0.5;
    (e as u32).clamp(1, (1 << 17) - 1)
}
