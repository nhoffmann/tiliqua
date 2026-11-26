//! Juno VCF helpers: knob (0..32768) -> `MultiLadder` register values.

/// Cutoff knob -> cutoff c (normalized octaves, 32768 = 1.0 = 10 octaves above
/// 20 Hz), like the Juno-106 VCF FREQ slider (knob = slider / 127), measured on
/// the KR-106: c = 1.40703 knob - 0.17726. Not clamped: env / kbd / mod are added
/// first. Same formula as `ladder_model.cutoff_knob_to_c` (sim_render).
pub fn cutoff_knob_to_c(knob: u16) -> i32 {
    ((i32::from(knob) * 46106) >> 15) - 5809
}

/// Keyboard tracking reference note: C1 (MIDI 24), as the KR-106.
pub const KBD_REF_NOTE: i32 = 24;

/// Resonance knob -> `reso` register (0..32768, gateware k = 4.5 * reso).
/// Piecewise linear at knob = 0, 1/8, ..., 1; calibrated against the KR-106
/// (Juno-106 RES slider). Same table as `ladder_model.RES_CURVE` (sim_render).
pub fn res_knob_to_reg(knob: u16) -> u16 {
    const CURVE: [u32; 9] = [0, 6226, 12288, 15155, 18022, 20316, 23429, 27034, 31130];
    let k = u32::from(knob.min(32768)) * 8;      // 0 .. 8 * 32768
    let i = ((k >> 15) as usize).min(7);
    let frac = k - ((i as u32) << 15);           // 0 .. 32768
    let v = CURVE[i] + ((CURVE[i + 1] - CURVE[i]) * frac + (1 << 14)) / 32768;
    v as u16
}
