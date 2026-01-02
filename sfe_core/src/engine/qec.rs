use super::noise::{generate_correlated_pink_noise, PinkNoiseGenerator};
use rand::prelude::*;
use rayon::prelude::*;

pub const SFE_PHASE_SCALE: f64 = 0.01_f64;
pub const SFE_EPSILON: f64 = 0.37;

fn generate_sfe_noise_traces(
    steps: usize,
    qubits: usize,
    alpha: f64,
    scale: f64,
    rho: f64,
) -> Vec<Vec<f64>> {
    let traces_base = if qubits == 1 {
        let mut gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);
        let mut buf = vec![0.0; steps];
        let mut v = Vec::with_capacity(qubits);
        for _ in 0..qubits {
            gen.generate(&mut buf);
            v.push(buf.clone());
        }
        v
    } else {
        generate_correlated_pink_noise(steps, qubits, alpha, scale, rho)
    };

    let omega: f64 = std::env::var("SFE_SUPPRESSON_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let amp: f64 = std::env::var("SFE_SUPPRESSON_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let omega2: f64 = std::env::var("SFE_SUPPRESSON_OMEGA2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let amp2: f64 = std::env::var("SFE_SUPPRESSON_AMP2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    if (omega == 0.0 || amp == 0.0) && (omega2 == 0.0 || amp2 == 0.0) {
        return traces_base;
    }

    let mut traces = traces_base;
    for trace in traces.iter_mut() {
        for (t, v) in trace.iter_mut().enumerate() {
            if omega != 0.0 && amp != 0.0 {
                let phase = omega * (t as f64);
                *v += amp * phase.cos();
            }
            if omega2 != 0.0 && amp2 != 0.0 {
                let phase2 = omega2 * (t as f64);
                *v += amp2 * phase2.cos();
            }
        }
    }

    traces
}

pub struct QECResult {
    pub distance: usize,
    pub physical_error_rate: f64,
    pub logical_error_rate: f64,
    pub gain: f64,
}

fn viterbi_final_state(observations: &[bool], p_flips: &[f64], meas_err: f64) -> bool {
    let n = observations.len();
    if n == 0 {
        return false;
    }

    let eps = 1.0e-12_f64;
    let mut m = meas_err;
    if m < eps {
        m = eps;
    } else if m > 1.0 - eps {
        m = 1.0 - eps;
    }
    let ln_m = m.ln();
    let ln_1_m = (1.0 - m).ln();

    let mut p0 = p_flips[0];
    if p0 < eps {
        p0 = eps;
    } else if p0 > 1.0 - eps {
        p0 = 1.0 - eps;
    }
    let ln_p0 = p0.ln();
    let ln_1_p0 = (1.0 - p0).ln();

    let emit0_0 = if observations[0] { ln_m } else { ln_1_m };
    let emit1_0 = if observations[0] { ln_1_m } else { ln_m };

    let mut prev0 = ln_1_p0 + emit0_0;
    let mut prev1 = ln_p0 + emit1_0;

    if n == 1 {
        return prev1 > prev0;
    }

    for i in 1..n {
        let mut p = p_flips[i];
        if p < eps {
            p = eps;
        } else if p > 1.0 - eps {
            p = 1.0 - eps;
        }
        let ln_p = p.ln();
        let ln_1_p = (1.0 - p).ln();

        let emit0 = if observations[i] { ln_m } else { ln_1_m };
        let emit1 = if observations[i] { ln_1_m } else { ln_m };

        let from0_to0 = prev0 + ln_1_p;
        let from1_to0 = prev1 + ln_p;
        let cur0 = emit0
            + if from0_to0 > from1_to0 {
                from0_to0
            } else {
                from1_to0
            };

        let from0_to1 = prev0 + ln_p;
        let from1_to1 = prev1 + ln_1_p;
        let cur1 = emit1
            + if from0_to1 > from1_to1 {
                from0_to1
            } else {
                from1_to1
            };

        prev0 = cur0;
        prev1 = cur1;
    }

    prev1 > prev0
}

pub fn simulate_repetition_code(
    distance: usize,
    pulse_seq: &[usize],
    noise_amp: f64,
    total_time: usize,
    measure_interval: usize,
    trials: usize,
) -> QECResult {
    if distance % 2 == 0 {
        panic!("다수결 보정을 위해 거리는 홀수여야 합니다.");
    }

    let num_cycles = total_time / measure_interval;

    let alpha: f64 = std::env::var("SFE_NOISE_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let scale: f64 = std::env::var("SFE_NOISE_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5 * noise_amp.abs());

    let t1_steps: f64 = std::env::var("SFE_T1_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0e5);
    let gate_err: f64 = std::env::var("SFE_GATE_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0e-3);

    let rho: f64 = std::env::var("SFE_NOISE_RHO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let meas_err: f64 = std::env::var("SFE_MEAS_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0e-3);
    let sup_omega: f64 = std::env::var("SFE_SUPPRESSON_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sup_amp: f64 = std::env::var("SFE_SUPPRESSON_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let anc_flag: i32 = std::env::var("SFE_SUPPRESSON_ANC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let anc_on = anc_flag != 0 && sup_omega != 0.0 && sup_amp != 0.0;

    let dt_cycle = measure_interval as f64;
    let p_t1 = 1.0 - (-dt_cycle / t1_steps).exp();

    // SFE Proposal 17: Gate Fidelity Scaling
    // F_sfe = F_std * exp(-epsilon * t_g / tau_coh)
    // Assuming t_g = 1 step for the effective gate error application
    let sfe_gate_fidelity_factor = (-SFE_EPSILON * (1.0 / t1_steps)).exp();

    let cycle_len = measure_interval;
    let mut cycle_pulses: Vec<usize> = pulse_seq
        .iter()
        .map(|idx| (idx % cycle_len).min(cycle_len - 1))
        .collect();
    cycle_pulses.sort_unstable();
    cycle_pulses.dedup();

    let (logical_errors, physical_errors) = (0..trials)
        .into_par_iter()
        .map(|_| {
            let traces = generate_sfe_noise_traces(total_time, distance, alpha, scale, rho);

            let mut rng = thread_rng();
            let mut phys_err_count = 0_usize;
            let mut states = vec![false; distance];

            let mut obs = vec![vec![false; num_cycles]; distance];
            let mut p_flips = vec![vec![0.0_f64; num_cycles]; distance];

            for cycle in 0..num_cycles {
                let start_idx = cycle * measure_interval;

                for q in 0..distance {
                    let noise = &traces[q];
                    let mut phase = 0.0_f64;
                    let mut sign = 1.0_f64;

                    for t_rel in 0..measure_interval {
                        let t_abs = start_idx + t_rel;
                        if t_abs >= noise.len() {
                            break;
                        }
                        if cycle_pulses.contains(&t_rel) {
                            sign *= -1.0;
                        }
                        let mut val = noise[t_abs];
                        if anc_on {
                            let theta = sup_omega * (t_abs as f64);
                            val -= sup_amp * theta.cos();
                        }
                        phase += sign * val * noise_amp * SFE_PHASE_SCALE;
                    }

                    let p_phase = 0.5 * (1.0 - phase.cos());

                    // Apply SFE Gate Fidelity Correction
                    let f_gate_std = 1.0 - gate_err;
                    let f_gate_sfe = f_gate_std * sfe_gate_fidelity_factor;
                    let p_gate_sfe = 1.0 - f_gate_sfe;

                    let mut p_total = 1.0 - (1.0 - p_phase) * (1.0 - p_t1) * (1.0 - p_gate_sfe);
                    p_total = p_total.clamp(0.0, 1.0);

                    p_flips[q][cycle] = p_total;

                    if rng.gen::<f64>() < p_total {
                        states[q] = !states[q];
                        phys_err_count += 1;
                    }
                }

                for q in 0..distance {
                    let mut meas = states[q];
                    if rng.gen::<f64>() < meas_err {
                        meas = !meas;
                    }
                    obs[q][cycle] = meas;
                }
            }

            let mut logical_failed = false;
            if num_cycles > 0 {
                let mut final_errors = 0_usize;
                for q in 0..distance {
                    if viterbi_final_state(&obs[q], &p_flips[q], meas_err) {
                        final_errors += 1;
                    }
                }
                if final_errors > distance / 2 {
                    logical_failed = true;
                }
            }

            let logical = if logical_failed { 1_usize } else { 0_usize };
            (logical, phys_err_count)
        })
        .reduce(|| (0_usize, 0_usize), |acc, x| (acc.0 + x.0, acc.1 + x.1));

    let total_phys_slots = trials * distance * num_cycles.max(1);
    let phy_rate = if total_phys_slots > 0 {
        physical_errors as f64 / total_phys_slots as f64
    } else {
        0.0
    };
    let log_rate = logical_errors as f64 / trials as f64;

    QECResult {
        distance,
        physical_error_rate: phy_rate,
        logical_error_rate: log_rate,
        gain: if log_rate > 0.0 {
            phy_rate / log_rate
        } else {
            -1.0
        },
    }
}

pub fn simulate_surface_code_d3(
    pulse_seq: &[usize],
    noise_amp: f64,
    total_time: usize,
    measure_interval: usize,
    trials: usize,
) -> QECResult {
    let num_cycles = total_time / measure_interval;

    let alpha: f64 = std::env::var("SFE_NOISE_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let scale: f64 = std::env::var("SFE_NOISE_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5 * noise_amp.abs());

    let t1_steps: f64 = std::env::var("SFE_T1_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0e5);
    // SFE Proposal 17 Correction Factor for Surface Code
    let sfe_gate_fidelity_factor = (-SFE_EPSILON * (1.0 / t1_steps)).exp();

    let meas_err: f64 = std::env::var("SFE_MEAS_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0e-3);

    let rho: f64 = std::env::var("SFE_NOISE_RHO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sup_omega: f64 = std::env::var("SFE_SUPPRESSON_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sup_amp: f64 = std::env::var("SFE_SUPPRESSON_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sup_omega2: f64 = std::env::var("SFE_SUPPRESSON_OMEGA2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sup_amp2: f64 = std::env::var("SFE_SUPPRESSON_AMP2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let anc_flag: i32 = std::env::var("SFE_SUPPRESSON_ANC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let anc_on = anc_flag != 0
        && ((sup_omega != 0.0 && sup_amp != 0.0) || (sup_omega2 != 0.0 && sup_amp2 != 0.0));

    let cycle_len = measure_interval;
    let mut cycle_pulses: Vec<usize> = pulse_seq
        .iter()
        .map(|idx| (idx % cycle_len).min(cycle_len - 1))
        .collect();
    cycle_pulses.sort_unstable();
    cycle_pulses.dedup();

    let data_qubits = 9usize;
    let stabs = [
        [0usize, 1usize, 3usize, 4usize],
        [1usize, 2usize, 4usize, 5usize],
        [3usize, 4usize, 6usize, 7usize],
        [4usize, 5usize, 7usize, 8usize],
    ];

    let logical_string = [1usize, 4usize, 7usize];

    let (logical_errors, physical_errors) = (0..trials)
        .into_par_iter()
        .map(|_| {
            let traces = generate_sfe_noise_traces(total_time, data_qubits, alpha, scale, rho);

            let mut rng = thread_rng();
            let mut phys_err_count = 0_usize;
            let mut z_state = vec![false; data_qubits];

            let mut syndromes = vec![vec![false; num_cycles]; 4];

            for cycle in 0..num_cycles {
                let start_idx = cycle * measure_interval;

                for q in 0..data_qubits {
                    let noise = &traces[q];
                    let mut phase = 0.0_f64;
                    let mut sign = 1.0_f64;

                    for t_rel in 0..measure_interval {
                        let t_abs = start_idx + t_rel;
                        if t_abs >= noise.len() {
                            break;
                        }
                        if cycle_pulses.contains(&t_rel) {
                            sign *= -1.0;
                        }
                        let mut val = noise[t_abs];
                        if anc_on {
                            if sup_omega != 0.0 && sup_amp != 0.0 {
                                let theta = sup_omega * (t_abs as f64);
                                val -= sup_amp * theta.cos();
                            }
                            if sup_omega2 != 0.0 && sup_amp2 != 0.0 {
                                let theta2 = sup_omega2 * (t_abs as f64);
                                val -= sup_amp2 * theta2.cos();
                            }
                        }
                        phase += sign * val * noise_amp * SFE_PHASE_SCALE;
                    }

                    let p_phase = 0.5 * (1.0 - phase.cos());

                    // Apply SFE correction to Z-error probability (acting as effective gate error here)
                    // Assuming p_phase is the base environmental noise, we add the SFE gate scaling effect
                    // F_sfe_gate = exp(-eps * 1/T1). The loss probability is 1 - F.
                    let p_sfe_loss = 1.0 - sfe_gate_fidelity_factor;

                    // Combine Phase noise and SFE suppression loss
                    let mut p_z = 1.0 - (1.0 - p_phase) * (1.0 - p_sfe_loss);
                    p_z = p_z.clamp(0.0, 1.0);

                    if rng.gen::<f64>() < p_z {
                        z_state[q] = !z_state[q];
                        phys_err_count += 1;
                    }
                }

                for s in 0..4 {
                    let mut v = false;
                    let sq = &stabs[s];
                    for idx in sq {
                        if z_state[*idx] {
                            v = !v;
                        }
                    }
                    let mut meas = v;
                    if rng.gen::<f64>() < meas_err {
                        meas = !meas;
                    }
                    syndromes[s][cycle] = meas;
                }
            }

            let mut final_synd = [false; 4];
            if num_cycles > 0 {
                for s in 0..4 {
                    final_synd[s] = syndromes[s][num_cycles - 1];
                }
            }

            let mut cand_patterns = Vec::with_capacity(data_qubits + 1);
            let mut cand_qubits = Vec::with_capacity(data_qubits + 1);

            cand_patterns.push(vec![false; 4]);
            cand_qubits.push(None);

            for q in 0..data_qubits {
                let mut p = vec![false; 4];
                for s in 0..4 {
                    let sq = &stabs[s];
                    if sq.contains(&q) {
                        p[s] = true;
                    }
                }
                cand_patterns.push(p);
                cand_qubits.push(Some(q));
            }

            let mut best_idx = 0usize;
            let mut best_dist = usize::MAX;

            for (i, pat) in cand_patterns.iter().enumerate() {
                let mut d = 0usize;
                for s in 0..4 {
                    if pat[s] != final_synd[s] {
                        d += 1;
                    }
                }
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }

            if let Some(q) = cand_qubits[best_idx] {
                z_state[q] = !z_state[q];
            }

            let mut logical_flip = false;
            for idx in logical_string {
                if z_state[idx] {
                    logical_flip = !logical_flip;
                }
            }

            let logical = if logical_flip { 1_usize } else { 0_usize };
            (logical, phys_err_count)
        })
        .reduce(|| (0_usize, 0_usize), |acc, x| (acc.0 + x.0, acc.1 + x.1));

    let total_phys_slots = trials * data_qubits * num_cycles.max(1);
    let phy_rate = if total_phys_slots > 0 {
        physical_errors as f64 / total_phys_slots as f64
    } else {
        0.0
    };
    let log_rate = logical_errors as f64 / trials as f64;

    QECResult {
        distance: 3,
        physical_error_rate: phy_rate,
        logical_error_rate: log_rate,
        gain: if log_rate > 0.0 {
            phy_rate / log_rate
        } else {
            -1.0
        },
    }
}
