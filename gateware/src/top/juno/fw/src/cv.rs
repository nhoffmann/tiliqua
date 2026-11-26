//! CV input routing (Phase 5.2) and CV output sources (5.3).
//!
//! Each input jack has a target (`CvTarget`) and a bipolar amount (-1..1, raw
//! +-16384 = SQ(2,14)). Pitch and cutoff are routed in gateware at audio rate
//! (`juno_cv.JunoCvRouter`): the firmware only tells it which jack (first jack with
//! that target) and the amount (1.0 = exactly 1 V/oct). All other targets are
//! offsets on the knob value, applied here at the tick: +-5 V x amount 1.0 = the
//! whole knob range (0..32768). Unpatched jacks contribute nothing.
//! CV samples: calibrated ASQ raw, 1 V = 4000 counts.

use crate::options::CvTarget;

pub const N_JACKS: usize = 4;
pub const AMT_ONE: i32 = 1 << 14;
const COUNTS_5V: i32 = 20000;

/// Gateware route for an audio-rate target: (src 0 = off / 1..4 = jack, amount).
pub fn fast_route(targets: &[CvTarget; N_JACKS], amts: &[i16; N_JACKS], t: CvTarget) -> (u8, i16) {
    for j in 0..N_JACKS {
        if targets[j] == t {
            return ((j + 1) as u8, amts[j]);
        }
    }
    (0, 0)
}

/// Knob offset (0..32768 scale) for target `t`, summed over all patched jacks
/// with that target.
pub fn knob_offset(targets: &[CvTarget; N_JACKS], amts: &[i16; N_JACKS],
                   samples: &[i32; N_JACKS], jack: u8, t: CvTarget) -> i32 {
    let mut off = 0i32;
    for j in 0..N_JACKS {
        if targets[j] == t && (jack >> j) & 1 == 1 {
            let x = (samples[j].clamp(-32768, 32767) * i32::from(amts[j])) >> 14;
            off += x * 32768 / COUNTS_5V;
        }
    }
    off
}

/// Knob value + offset, clamped to the knob range.
pub fn apply(knob: u16, off: i32) -> u16 {
    (i32::from(knob) + off).clamp(0, 32768) as u16
}
