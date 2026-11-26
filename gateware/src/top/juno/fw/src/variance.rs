//! Per-voice variance (component tolerances of one analog unit) and unison spread.
//! Same values as `sim_render.py` (`VARIANCE_PITCH_CENTS`, `VARIANCE_CUTOFF_PCT`,
//! `UNISON_SPREAD`).
//!
//! Behaviour reference: KR-106 `docs/DSP_ARCHITECTURE.md` ("Per-Voice Variance":
//! fixed per-voice offsets, pitch +-3 cents, cutoff +-5 %). Our own offsets.
//! Envelope timing / VCA gain variance are not implemented (need per-voice
//! multiplies in gateware; see the plan, Nice to have).

/// Pitch offset per voice at variance 1.0, cents.
pub const PITCH_CENTS: [f32; 6] = [1.8, -2.6, 0.7, -1.2, 2.9, -0.4];
/// Cutoff offset per voice at variance 1.0, normalized octaves raw (1 octave =
/// 3276.8): log2(1 + p) for p = -3.1, +4.6, -0.8, +2.2, -4.7, +1.5 %.
pub const CUTOFF_RAW: [i32; 6] = [-149, 213, -38, 103, -228, 70];
/// Unison spread per voice (x detune knob x `DETUNE_MAX_CENTS`).
pub const UNISON_SPREAD: [f32; 6] = [-1.0, 1.0, -0.6, 0.6, -0.2, 0.2];
pub const DETUNE_MAX_CENTS: f32 = 25.0;

/// Voice pitch offset -> `voice_cfg.detune` (1/1024 semitones, 10 bit signed).
pub fn detune_raw(voice: usize, variance: u16, unison: bool, detune: u16) -> i16 {
    let v = f32::from(variance.min(32768)) / 32768.0;
    let d = f32::from(detune.min(32768)) / 32768.0;
    let mut cents = PITCH_CENTS[voice % 6] * v;
    if unison {
        cents += UNISON_SPREAD[voice % 6] * d * DETUNE_MAX_CENTS;
    }
    let raw = cents / 100.0 * 1024.0;
    let raw = if raw >= 0.0 { raw + 0.5 } else { raw - 0.5 } as i32;
    raw.clamp(-511, 511) as i16
}

/// Voice cutoff offset (normalized octaves raw) for the variance knob.
pub fn cutoff_raw(voice: usize, variance: u16) -> i32 {
    CUTOFF_RAW[voice % 6] * i32::from(variance.min(32768)) / 32768
}
