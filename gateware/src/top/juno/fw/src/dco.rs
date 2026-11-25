//! Juno DCO helpers: parameter conversion for `MultiJunoOsc` and the menu preview.
//!
//! Keep `pw_knob_to_point` / `sub_gain` in sync with `sim_render.py`.

use micromath::F32Ext;

/// Option range of the DCO knobs (0..32768 = 0..1).
const KNOB_ONE: f32 = 32768.0;

fn sqrt_precise(v: f32) -> f32 {
    // micromath's sqrt is an approximation; refine with two Newton steps.
    let mut x = v.sqrt();
    if x > 0.0 {
        x = 0.5 * (x + v / x);
        x = 0.5 * (x + v / x);
    }
    x
}

/// PW knob (0..32768) -> pulse rising-edge phase, UQ(0,16).
///
/// Juno: the pulse comparator sits on the curved ramp p (1 + 0.15 (1 - p)) at
/// threshold 0.5 + 0.46 * knob (duty ~54 % .. ~4.5 %). The oscillator compares the
/// *linear* phase, so solve the curve for p.
pub fn pw_knob_to_point(knob: u16) -> u16 {
    let k = f32::from(knob) / KNOB_ONE;
    let th = 0.5 + 0.46 * k;
    let p = (1.15 - sqrt_precise(1.15 * 1.15 - 0.6 * th)) / 0.3;
    let v = (p * 65536.0 + 0.5) as u32;
    v.min(65535) as u16
}

/// Sub level knob (0..32768) -> sub gain UQ(0,16), Juno mixer weight 0.67.
pub fn sub_gain(level: u16) -> u16 {
    let g = (u64::from(level) * 65535 * 67 + 32768 * 50) / (32768 * 100);
    g.min(65535) as u16
}

/// Level knob (0..32768) -> gain UQ(0,16).
pub fn level_gain(level: u16) -> u16 {
    (u32::from(level) * 2).min(65535) as u16
}

/// Menu preview: two cycles of the (non band-limited) DCO mix, so the
/// sub-oscillator (one octave down) is visible.
pub fn preview(buf: &mut [i16], saw: bool, pulse: bool, pw: u16, sub_level: u16) {
    let w = f32::from(pw_knob_to_point(pw)) / 65536.0;
    let sub_g = f32::from(sub_gain(sub_level)) / 65536.0;
    let len = buf.len() as f32;
    for (i, b) in buf.iter_mut().enumerate() {
        let x = 2.0 * i as f32 / len;          // 0..2 cycles
        let cycle = if x >= 1.0 { 1 } else { 0 };
        let p = x - cycle as f32;
        let ramp = p * (1.0 + 0.15 * (1.0 - p));
        let mut v = 0.0;
        if saw {
            v += 0.5 * (2.0 * ramp - 1.0);
        }
        if pulse {
            v += 0.5 * if p >= w { 1.0 } else { -1.0 };
        }
        // sub toggles at each wrap: high in the first cycle, low in the second
        v += sub_g * if cycle == 0 { 1.0 } else { -1.0 };
        let v = v.clamp(-1.0, 1.0);
        *b = (v * 32767.0) as i16;
    }
}
