//! Juno output stage helpers (Phase 4.7): level / drive knobs (0..32768) ->
//! `drive` (pre-gain) and `out_gain` (post-gain) registers. Same formulas as
//! `juno_output.py` (`level_gain`, `drive_gain`, `registers`).
//!
//! level_gain = 4 k^2 (default k = 0.45: a default 6-voice chord peaks at the
//! limiter knee, ~5 V); drive_gain = 1 + 7 d^2 (0 = off); pre = level x drive
//! (< 4); post = 1 / sqrt(drive).

use micromath::F32Ext;

const DRIVE_MAX: f32 = 8.0;
const PRE_MAX: f32 = 4.0 - 1.0 / 32768.0;

fn knob(k: u16) -> f32 {
    f32::from(k.min(32768)) / 32768.0
}

/// (drive register = pre_gain * 8192, out_gain register = post_gain UQ(0,16)).
pub fn registers(level: u16, drive: u16) -> (u16, u16) {
    let (l, d) = (knob(level), knob(drive));
    let g = 1.0 + (DRIVE_MAX - 1.0) * d * d;
    let pre = (4.0 * l * l * g).min(PRE_MAX);
    let pre_raw = ((pre * 8192.0 + 0.5) as u32).min(32767) as u16;
    let post_raw = ((65536.0 / g.sqrt() + 0.5) as u32).min(65535) as u16;
    (pre_raw, post_raw)
}
