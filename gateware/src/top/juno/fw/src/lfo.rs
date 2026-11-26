//! Juno LFO helpers: knob (0..32768 = Juno-106 slider / 127) -> `JunoLfo` register
//! values. Tables measured on the KR-106; same as `juno_lfo.py` (`LFO_RATE_HZ`,
//! `LFO_DCO_SEMI`, `LFO_VCF_OCT`, `LFO_DELAY95_S`, `lfo_table`).
//!
//! Physical units -> register values use the same formulas as `sim_render.py`
//! (rate_inc = f / 48000 * 2^32, delay_step = 2^30 / (t * 48000),
//! dco_depth = semitones * 1024, vcf_depth = normalized octaves * 32768).

use micromath::F32Ext;

const SLIDERS: [f32; 9] = [0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 127.0];
const RATE_HZ: [f32; 9] = [0.05, 1.28, 2.36, 3.47, 4.54, 6.56, 8.44, 14.77, 29.52];
const DCO_SEMI: [f32; 9] = [0.0, 0.12, 0.21, 0.41, 0.51, 0.64, 0.71, 1.30, 3.80];
const VCF_OCT: [f32; 9] = [0.0, 0.42, 0.84, 1.24, 1.60, 1.92, 2.15, 2.32, 2.42];
const DELAY95_S: [f32; 9] = [0.0, 0.20, 0.30, 0.48, 0.65, 1.00, 1.43, 2.30, 3.38];

const FS: f32 = 48000.0;

fn table(k: u16, t: &[f32; 9], log: bool) -> f32 {
    let sl = f32::from(k.min(32768)) / 32768.0 * 127.0;
    let mut i = 0;
    while i < 7 && sl > SLIDERS[i + 1] {
        i += 1;
    }
    let f = (sl - SLIDERS[i]) / (SLIDERS[i + 1] - SLIDERS[i]);
    if log {
        let (a, b) = (t[i].ln(), t[i + 1].ln());
        (a + f * (b - a)).exp()
    } else {
        t[i] + f * (t[i + 1] - t[i])
    }
}

/// Rate knob -> LFO frequency in Hz (0.05 .. 29.5 Hz, KR-106 curve).
pub fn rate_hz(k: u16) -> f32 {
    table(k, &RATE_HZ, true)
}

/// Rate knob -> phase increment per frame, UQ(0,32).
pub fn rate_inc(k: u16) -> u32 {
    (rate_hz(k) / FS * 4294967296.0 + 0.5) as u32
}

/// Delay knob -> amplitude step per frame. Our fade-in is linear: full time =
/// measured 95 % time / 0.95.
pub fn delay_step(k: u16) -> u32 {
    let seconds = table(k, &DELAY95_S, false) / 0.95;
    let frames = seconds * FS;
    if frames < 1.0 {
        1 << 30
    } else {
        ((1u32 << 30) as f32 / frames).max(1.0) as u32
    }
}

/// DCO LFO knob -> vibrato depth in 1/1024 semitones (peak, KR-106 curve).
pub fn dco_depth(k: u16) -> u16 {
    (table(k, &DCO_SEMI, false) * 1024.0 + 0.5) as u16
}

/// VCF LFO knob -> cutoff depth in normalized octaves (15 bit; 1 octave = 1/10).
pub fn vcf_depth(k: u16) -> u16 {
    (table(k, &VCF_OCT, false) / 10.0 * 32768.0 + 0.5) as u16
}
