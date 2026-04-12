use super::config::{NoiseConfig, QecConfig, SuppressionConfig};
use super::noise::{generate_correlated_pink_noise, PinkNoiseGenerator};
use rand::prelude::*;
use rayon::prelude::*;

pub const CE_PHASE_SCALE: f64 = 0.01_f64;
pub const CE_EPSILON: f64 = 0.37;

fn generate_ce_noise_traces(
    steps: usize,
    qubits: usize,
    noise_cfg: &NoiseConfig,
    sup_cfg: &SuppressionConfig,
) -> Vec<Vec<f64>> {
    let traces_base = if qubits == 1 {
        let mut gen = PinkNoiseGenerator::new_with_params(steps, noise_cfg.alpha, noise_cfg.scale);
        let mut buf = vec![0.0; steps];
        let mut v = Vec::with_capacity(qubits);
        for _ in 0..qubits {
            gen.generate(&mut buf);
            v.push(buf.clone());
        }
        v
    } else {
        generate_correlated_pink_noise(steps, qubits, noise_cfg.alpha, noise_cfg.scale, noise_cfg.rho)
    };

    if !sup_cfg.has_any() {
        return traces_base;
    }

    let mut traces = traces_base;
    for trace in traces.iter_mut() {
        sup_cfg.apply_to_trace(trace);
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

    let noise_cfg = NoiseConfig::from_env_with_noise(noise_amp);
    let sup_cfg = SuppressionConfig::from_env();
    let qec_cfg = QecConfig::from_env();

    let num_cycles = total_time / measure_interval;
    let dt_cycle = measure_interval as f64;
    let p_t1 = 1.0 - (-dt_cycle / qec_cfg.t1_steps).exp();
    let ce_gate_fidelity_factor = (-CE_EPSILON * (1.0 / qec_cfg.t1_steps)).exp();

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
            let traces = generate_ce_noise_traces(total_time, distance, &noise_cfg, &sup_cfg);

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
                        if sup_cfg.anc_enabled {
                            val = sup_cfg.cancel_from_sample(val, t_abs);
                        }
                        phase += sign * val * noise_amp * CE_PHASE_SCALE;
                    }

                    let p_phase = 0.5 * (1.0 - phase.cos());

                    let f_gate_std = 1.0 - qec_cfg.gate_error;
                    let f_gate_sfe = f_gate_std * ce_gate_fidelity_factor;
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
                    if rng.gen::<f64>() < qec_cfg.meas_error {
                        meas = !meas;
                    }
                    obs[q][cycle] = meas;
                }
            }

            let mut logical_failed = false;
            if num_cycles > 0 {
                let mut final_errors = 0_usize;
                for q in 0..distance {
                    if viterbi_final_state(&obs[q], &p_flips[q], qec_cfg.meas_error) {
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
    let noise_cfg = NoiseConfig::from_env_with_noise(noise_amp);
    let sup_cfg = SuppressionConfig::from_env();
    let qec_cfg = QecConfig::from_env();

    let num_cycles = total_time / measure_interval;
    let ce_gate_fidelity_factor = (-CE_EPSILON * (1.0 / qec_cfg.t1_steps)).exp();

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
            let traces = generate_ce_noise_traces(total_time, data_qubits, &noise_cfg, &sup_cfg);

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
                        if sup_cfg.anc_enabled {
                            val = sup_cfg.cancel_from_sample(val, t_abs);
                        }
                        phase += sign * val * noise_amp * CE_PHASE_SCALE;
                    }

                    let p_phase = 0.5 * (1.0 - phase.cos());
                    let p_ce_loss = 1.0 - ce_gate_fidelity_factor;

                    let mut p_z = 1.0 - (1.0 - p_phase) * (1.0 - p_ce_loss);
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
                    if rng.gen::<f64>() < qec_cfg.meas_error {
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

            let synd_weight: usize = final_synd.iter().filter(|&&s| s).count();

            if synd_weight > 0 {
                let mut single_synd = [[false; 4]; 9];
                for q in 0..data_qubits {
                    for s in 0..4 {
                        if stabs[s].contains(&q) {
                            single_synd[q][s] = true;
                        }
                    }
                }

                let mut best_single_q = 0usize;
                let mut best_single_dist = usize::MAX;

                for q in 0..data_qubits {
                    let mut d = 0usize;
                    for s in 0..4 {
                        if single_synd[q][s] != final_synd[s] {
                            d += 1;
                        }
                    }
                    if d < best_single_dist {
                        best_single_dist = d;
                        best_single_q = q;
                    }
                }

                let mut best_correction: Vec<usize> = vec![best_single_q];

                if best_single_dist >= 2 {
                    for q1 in 0..data_qubits {
                        for q2 in (q1 + 1)..data_qubits {
                            let mut combined = [false; 4];
                            for s in 0..4 {
                                combined[s] = single_synd[q1][s] ^ single_synd[q2][s];
                            }
                            let mut d = 0usize;
                            for s in 0..4 {
                                if combined[s] != final_synd[s] {
                                    d += 1;
                                }
                            }
                            if d == 0 {
                                best_correction = vec![q1, q2];
                                break;
                            }
                        }
                        if best_correction.len() == 2 {
                            break;
                        }
                    }
                }

                for &q in &best_correction {
                    z_state[q] = !z_state[q];
                }
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
