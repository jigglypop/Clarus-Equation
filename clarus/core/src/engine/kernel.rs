//! Brain-runtime cell-step kernel.
//!
//! Mirrors the hot path of `runtime.py:BrainRuntime.step()`.
//! All state vectors are flat f32 slices of length `dim`.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ModeParams {
    pub activation_decay: f32,
    pub activation_gain: f32,
    pub refractory_decay: f32,
    pub refractory_gain: f32,
    pub replay_mix: f32,
}

impl ModeParams {
    pub fn wake() -> Self {
        Self {
            activation_decay: 0.18,
            activation_gain: 0.82,
            refractory_decay: 0.12,
            refractory_gain: 0.24,
            replay_mix: 0.08,
        }
    }
    pub fn nrem() -> Self {
        Self {
            activation_decay: 0.34,
            activation_gain: 0.52,
            refractory_decay: 0.26,
            refractory_gain: 0.12,
            replay_mix: 0.28,
        }
    }
    pub fn rem() -> Self {
        Self {
            activation_decay: 0.22,
            activation_gain: 0.68,
            refractory_decay: 0.18,
            refractory_gain: 0.18,
            replay_mix: 0.35,
        }
    }
    pub fn from_mode(mode: super::runtime_types::Mode) -> Self {
        match mode {
            super::runtime_types::Mode::Wake => Self::wake(),
            super::runtime_types::Mode::Nrem => Self::nrem(),
            super::runtime_types::Mode::Rem => Self::rem(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepConfig {
    pub refractory_scale: f32,
    pub goal_gain: f32,
    pub external_gain: f32,
    pub active_threshold: f32,
    pub bit_lower: f32,
    pub bit_upper: f32,
    pub energy_budget: usize,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            refractory_scale: 0.35,
            goal_gain: 0.20,
            external_gain: 0.45,
            active_threshold: 0.22,
            bit_lower: 0.10,
            bit_upper: 0.30,
            energy_budget: 16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub active_count: usize,
    pub energy: f32,
}

/// Sparse CSR matvec: y = A @ x, only over rows where `mask[i]` is true.
fn spmv_masked(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    x: &[f32],
    mask: &[bool],
    out: &mut [f32],
) {
    let dim = out.len();
    out.par_iter_mut().enumerate().for_each(|(i, yi)| {
        if !mask[i] || i >= dim {
            *yi = 0.0;
            return;
        }
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut acc = 0.0_f32;
        for idx in start..end {
            let j = col_idx[idx] as usize;
            acc += values[idx] * x[j];
        }
        *yi = acc;
    });
}

/// Full brain-runtime cell step (one tick).
///
/// Mutates `activation`, `refractory`, `memory_trace`, `bitfield` in place.
/// Returns `StepOutput` with active count and energy estimate.
pub fn brain_step(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    activation: &mut [f32],
    refractory: &mut [f32],
    memory_trace: &mut [f32],
    bitfield: &mut [u8],
    external: &[f32],
    goal: &[f32],
    replay: &[f32],
    mode_params: &ModeParams,
    cfg: &StepConfig,
) -> StepOutput {
    let dim = activation.len();
    if dim == 0 {
        return StepOutput { active_count: 0, energy: 0.0 };
    }

    // 1. active mask from previous step (bitfield==1 as proxy)
    let prev_active: Vec<bool> = bitfield.iter().map(|&b| b > 0).collect();

    // 2. sparse recurrent: recurrent = W @ (activation * prev_active)
    let masked_act: Vec<f32> = activation
        .iter()
        .zip(prev_active.iter())
        .map(|(&a, &m)| if m { a } else { 0.0 })
        .collect();
    let mut recurrent = vec![0.0_f32; dim];
    // full spmv (no row masking for recurrent -- all rows get coupling)
    let all_true = vec![true; dim];
    spmv_masked(values, col_idx, row_ptr, &masked_act, &all_true, &mut recurrent);

    // 3. drive = recurrent + external*gain + goal*gain + replay*mix - refractory*scale
    let ext_g = cfg.external_gain;
    let goal_g = cfg.goal_gain;
    let ref_s = cfg.refractory_scale;
    let rep_m = mode_params.replay_mix;
    let drive: Vec<f32> = (0..dim)
        .map(|i| {
            recurrent[i]
                + ext_g * external[i]
                + goal_g * goal[i]
                + rep_m * replay[i]
                - ref_s * refractory[i]
        })
        .collect();

    // 4. activation update: a' = (1 - decay)*a + gain*tanh(drive)
    let decay = mode_params.activation_decay;
    let gain = mode_params.activation_gain;
    let new_act: Vec<f32> = (0..dim)
        .map(|i| (1.0 - decay) * activation[i] + gain * drive[i].tanh())
        .collect();

    // 5. refractory update: r' = (1 - r_decay)*r + r_gain*a'^2
    let r_decay = mode_params.refractory_decay;
    let r_gain = mode_params.refractory_gain;
    let new_ref: Vec<f32> = (0..dim)
        .map(|i| (1.0 - r_decay) * refractory[i] + r_gain * new_act[i] * new_act[i])
        .collect();

    // 6. memory trace: m' = 0.92*m + 0.08*a'
    let new_mem: Vec<f32> = (0..dim)
        .map(|i| 0.92 * memory_trace[i] + 0.08 * new_act[i])
        .collect();

    // 7. bitfield hysteresis
    let new_bit: Vec<u8> = (0..dim)
        .map(|i| {
            if new_act[i] >= cfg.bit_upper {
                1
            } else if new_act[i] <= cfg.bit_lower {
                0
            } else {
                bitfield[i]
            }
        })
        .collect();

    // 8. salience = |a'| + 0.35*|ext| + 0.25*|replay| + 0.20*|goal| - 0.15*r'
    let salience: Vec<f32> = (0..dim)
        .map(|i| {
            new_act[i].abs()
                + 0.35 * external[i].abs()
                + 0.25 * replay[i].abs()
                + 0.20 * goal[i].abs()
                - 0.15 * new_ref[i]
        })
        .collect();

    // 9. active selection: topk by salience, only where salience >= threshold
    let budget = cfg.energy_budget.min(dim);
    let mut scored: Vec<(f32, usize)> = salience
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= cfg.active_threshold)
        .map(|(i, &s)| (s, i))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(budget);
    let active_count = scored.len();

    // 10. energy estimate
    let coupling_energy = {
        let dot: f32 = new_act.iter().zip(recurrent.iter()).map(|(a, r)| a * r).sum();
        0.5 * dot.abs()
    };
    let local_energy: f32 = new_ref.iter().map(|r| r.abs()).sum::<f32>() / dim as f32
        + 0.25 * new_mem.iter().map(|m| m.abs()).sum::<f32>() / dim as f32;
    let replay_energy = 0.1 * replay.iter().map(|r| r.abs()).sum::<f32>() / dim as f32;
    let energy = coupling_energy + local_energy + replay_energy;

    // commit state
    activation.copy_from_slice(&new_act);
    refractory.copy_from_slice(&new_ref);
    memory_trace.copy_from_slice(&new_mem);
    bitfield.copy_from_slice(&new_bit);

    StepOutput { active_count, energy }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_csr(dim: usize) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        let mut values = Vec::with_capacity(dim);
        let mut col_idx = Vec::with_capacity(dim);
        let mut row_ptr = Vec::with_capacity(dim + 1);
        row_ptr.push(0);
        for i in 0..dim {
            values.push(0.1);
            col_idx.push(i as i32);
            row_ptr.push((i + 1) as i32);
        }
        (values, col_idx, row_ptr)
    }

    #[test]
    fn basic_step_runs() {
        let dim = 16;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.5_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let ext = vec![0.1_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig { energy_budget: 4, ..Default::default() };
        let out = brain_step(
            &vals, &cols, &rows,
            &mut act, &mut refr, &mut mem, &mut bit,
            &ext, &goal, &replay, &mp, &cfg,
        );
        assert!(out.active_count <= 4);
        assert!(out.energy >= 0.0);
        assert!(act.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn energy_decreases_nrem() {
        let dim = 32;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.3_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let ext = vec![0.0_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::nrem();
        let cfg = StepConfig { energy_budget: 8, ..Default::default() };
        let mut energies = Vec::new();
        for _ in 0..20 {
            let out = brain_step(
                &vals, &cols, &rows,
                &mut act, &mut refr, &mut mem, &mut bit,
                &ext, &goal, &replay, &mp, &cfg,
            );
            energies.push(out.energy);
        }
        assert!(energies.last().unwrap() < energies.first().unwrap());
    }
}
