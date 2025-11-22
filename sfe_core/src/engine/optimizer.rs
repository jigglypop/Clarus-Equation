use rayon::prelude::*;
use rand::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use super::noise::PinkNoiseGenerator;
use std::env;

const DT: f64 = 0.01;

pub fn run_pulse_optimizer(steps: usize, n_pulses: usize, generations: usize, noise_amp: f64) -> (Vec<usize>, f64, f64) {
    println!("Starting SFE-Genetic Pulse Optimizer (Island Model)...");
    
    let alpha = env::var("SFE_NOISE_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.8);
    let scale = env::var("SFE_NOISE_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.5);
    let mut qec_distance = env::var("SFE_QEC_DISTANCE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    if qec_distance == 0 {
        qec_distance = 1;
    }
    if qec_distance % 2 == 0 {
        qec_distance += 1;
    }
    let mut moment_order = env::var("SFE_MOMENT_ORDER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    if moment_order == 0 {
        moment_order = 1;
    }

    println!(
        ">> Pre-generating noise pool (alpha={:.3}, scale={:.3})...",
        alpha, scale
    );
    let pool_size = 2000;
    let mut noise_gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);
    let mut noise_pool: Vec<Vec<f64>> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        noise_pool.push(noise_gen.generate_new());
    }
    
    let mut best_sequence: Vec<usize> = Vec::with_capacity(n_pulses);
    for j in 1..=n_pulses {
        let sin_val = (j as f64 * std::f64::consts::PI / (2.0 * (n_pulses as f64 + 1.0))).sin();
        let t_j = (steps as f64 * sin_val.powi(2)).round() as usize;
        if t_j > 0 && t_j < steps { best_sequence.push(t_j); }
    }
    best_sequence.sort();
    best_sequence.dedup();
    
    let udd_score = evaluate_sequence_with_pool(&best_sequence, &noise_pool, noise_amp, 200, moment_order, qec_distance);

    let mut cpmg_seq: Vec<usize> = Vec::with_capacity(n_pulses);
    for k in 0..n_pulses {
        let t = (((k + 1) as f64) / (n_pulses as f64 + 1.0) * steps as f64).round() as usize;
        if t > 0 && t < steps {
            cpmg_seq.push(t);
        }
    }
    cpmg_seq.sort();
    cpmg_seq.dedup();
    let cpmg_score = evaluate_sequence_with_pool(&cpmg_seq, &noise_pool, noise_amp, 200, moment_order, qec_distance);

    let mut global_best_score = udd_score;
    let mut global_best_seq = best_sequence.clone();

    let pb = ProgressBar::new(generations as u64);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [Gen {pos}/{len}] Best: {msg}").unwrap());

    let num_islands = 4;
    let island_pop_size = 50; 
    
    let smart_seeds: Vec<Vec<usize>> = (0..num_islands).map(|i| {
        let sample_idx = i % pool_size;
        let sample_noise = &noise_pool[sample_idx];
        
        let mut gradients: Vec<(usize, f64)> = (0..steps-1).map(|k| {
            (k, (sample_noise[k+1] - sample_noise[k]).abs())
        }).collect();
        gradients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut seed = Vec::new();
        for k in 0..n_pulses {
            if k < gradients.len() {
                seed.push(gradients[k].0);
            }
        }
        seed.sort();
        seed.dedup();
        while seed.len() < n_pulses {
             let mut rng = thread_rng();
             let t = rng.gen_range(1..steps);
             if !seed.contains(&t) { seed.push(t); }
        }
        seed.sort();
        seed
    }).collect();

    let mut islands: Vec<Vec<(Vec<usize>, f64)>> = (0..num_islands).map(|island_idx| {
        let mut pop = Vec::with_capacity(island_pop_size);
        pop.push((best_sequence.clone(), udd_score));
        let smart_score = evaluate_sequence_with_pool(&smart_seeds[island_idx], &noise_pool, noise_amp, 200, moment_order, qec_distance);
        pop.push((smart_seeds[island_idx].clone(), smart_score));
        
        let mut rng = thread_rng();
        while pop.len() < island_pop_size {
             let mut p = best_sequence.clone();
             for t in &mut p {
                 if rng.gen_bool(0.5) {
                     *t = rng.gen_range(1..steps);
                 }
             }
             p.sort(); p.dedup();
             while p.len() < n_pulses {
                 let t = rng.gen_range(1..steps);
                 if !p.contains(&t) { p.push(t); }
             }
             p.sort();
             let score = evaluate_sequence_with_pool(&p, &noise_pool, noise_amp, 200, moment_order, qec_distance);
             pop.push((p, score));
        }
        pop
    }).collect();

    for gen in 0..generations {
        islands.par_iter_mut().for_each(|island| {
            island.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let elites = island[0..5].to_vec();
            let mut new_pop = Vec::with_capacity(island_pop_size);
            new_pop.extend(elites.clone());

            let mut rng = thread_rng();
            while new_pop.len() < island_pop_size {
                let idx1 = rng.gen_range(0..20);
                let idx2 = rng.gen_range(0..20);
                let p1 = &island[idx1].0;
                let p2 = &island[idx2].0;
                
                let mut child = Vec::with_capacity(n_pulses);
                for k in 0..p1.len().min(p2.len()) {
                    if rng.gen_bool(0.5) { child.push(p1[k]); } else { child.push(p2[k]); }
                }
                
                let mutation_rate = 0.4 * (1.0 - gen as f64 / generations as f64) + 0.05;
                let mutation_power = (steps as f64 * 0.1 * (1.0 - gen as f64 / generations as f64)).max(1.0) as i32;

                for t in &mut child {
                    if rng.gen_bool(mutation_rate) {
                        let shift = rng.gen_range(-mutation_power..=mutation_power);
                        let new_val = (*t as i32 + shift).clamp(1, (steps - 1) as i32) as usize;
                        *t = new_val;
                    }
                }
                
                child.sort();
                child.dedup();
                while child.len() < n_pulses {
                    let new_t = rng.gen_range(1..steps);
                    if !child.contains(&new_t) { child.push(new_t); }
                }
                child.sort();

                let score = evaluate_sequence_with_pool(&child, &noise_pool, noise_amp, 200, moment_order, qec_distance);
                new_pop.push((child, score));
            }
            *island = new_pop;
        });

        if gen % 5 == 0 {
            let mut migrants = Vec::new();
            for island in &islands {
                migrants.push(island[0].clone());
            }
            for i in 0..num_islands {
                let target_island = (i + 1) % num_islands;
                let worst_idx = island_pop_size - 1;
                islands[target_island][worst_idx] = migrants[i].clone();
            }
        }

        for island in &islands {
            if island[0].1 > global_best_score {
                global_best_score = island[0].1;
                global_best_seq = island[0].0.clone();
                pb.set_message(format!("{:.5}", global_best_score));
            }
        }
        pb.inc(1);
    }
    pb.finish();

    println!(">> Polishing best solution...");
    let mut polishing_pool = noise_pool.clone();
    for _ in 0..1000 {
        polishing_pool.push(noise_gen.generate_new());
    }

    let polished_seq = local_search_polish(&global_best_seq, steps, noise_amp, &polishing_pool, 500, moment_order, qec_distance);
    let polished_score = evaluate_sequence_with_pool(&polished_seq, &polishing_pool, noise_amp, 1000, moment_order, qec_distance);

    println!(">> Polishing Result: {:.5} -> {:.5}", global_best_score, polished_score);
    println!(">> Baselines (long-time weighted) -> UDD: {:.4}, CPMG: {:.4}", udd_score, cpmg_score);
    println!(">> SFE Long-Time Score: {:.4}", polished_score);

    (polished_seq, udd_score, polished_score)
}

fn local_search_polish(seq: &[usize], steps: usize, noise_amp: f64, noise_pool: &[Vec<f64>], trials: usize, moment_order: usize, qec_distance: usize) -> Vec<usize> {
    let mut current_seq = seq.to_vec();
    let mut current_score = evaluate_sequence_with_pool(&current_seq, noise_pool, noise_amp, trials, moment_order, qec_distance);
    
    let mut improved = true;
    let mut rng = thread_rng();

    let max_iters = 50; 
    for _ in 0..max_iters {
        improved = false;
        let idx = rng.gen_range(0..current_seq.len());
        
        for shift in [-1, 1] {
            let mut neighbor = current_seq.clone();
            let new_val = (neighbor[idx] as i32 + shift).clamp(1, (steps - 1) as i32) as usize;
            
            if !neighbor.contains(&new_val) {
                neighbor[idx] = new_val;
                neighbor.sort(); 
                
                let score = evaluate_sequence_with_pool(&neighbor, noise_pool, noise_amp, trials, moment_order, qec_distance);
                if score > current_score {
                    current_seq = neighbor;
                    current_score = score;
                    improved = true;
                }
            }
        }
        if !improved { break; }
    }
    current_seq
}

pub fn evaluate_sequence_with_pool(seq: &[usize], noise_pool: &[Vec<f64>], noise_amp: f64, trials: usize, moment_order: usize, qec_distance: usize) -> f64 {
    let pool_len = noise_pool.len();
    if pool_len == 0 {
        return 0.0;
    }

    let steps = noise_pool[0].len();
    if steps == 0 {
        return 0.0;
    }

    let safe_trials = trials.min(pool_len);
    if safe_trials == 0 {
        return 0.0;
    }

    let max_order = if moment_order == 0 { 0 } else { moment_order.min(3) };
    let moment_penalty = if max_order == 0 {
        0.0
    } else {
        compute_moment_penalty(seq, steps, max_order)
    };

    let checkpoints = [
        ((steps as f64 * 0.4).round() as usize).min(steps.saturating_sub(1)),
        ((steps as f64 * 0.6).round() as usize).min(steps.saturating_sub(1)),
        ((steps as f64 * 0.8).round() as usize).min(steps.saturating_sub(1)),
        steps.saturating_sub(1),
    ];
    let weights = [1.0_f64, 2.0, 3.0, 4.0];
    let weight_sum: f64 = weights.iter().sum();
    let gamma = (qec_distance as f64 + 1.0) / 2.0;

    let base_score = (0..safe_trials)
        .into_par_iter()
        .map(|i| {
            let pink_noise = &noise_pool[i];
            let mut phase = 0.0_f64;
            let mut sign = 1.0_f64;
            let mut pulse_idx = 0usize;
            let mut values = [0.0_f64; 4];
            let mut ck_idx = 0usize;

            for (t, &noise_val) in pink_noise.iter().enumerate() {
                if pulse_idx < seq.len() && t == seq[pulse_idx] {
                    sign *= -1.0;
                    pulse_idx += 1;
                }
                phase += sign * noise_val * noise_amp * DT;
                if ck_idx < checkpoints.len() && t == checkpoints[ck_idx] {
                    values[ck_idx] = phase.cos();
                    ck_idx += 1;
                }
            }

            let mut loss_num = 0.0_f64;
            for j in 0..values.len() {
                let s = values[j].max(-1.0).min(1.0);
                let mut p = 0.5 * (1.0 - s);
                if p < 0.0 {
                    p = 0.0;
                }
                if p > 1.0 {
                    p = 1.0;
                }
                let l = p.powf(gamma);
                loss_num += weights[j] * l;
            }
            let loss = if weight_sum > 0.0 {
                loss_num / weight_sum
            } else {
                0.0
            };

            let mut mean = 0.0_f64;
            for j in 0..values.len() {
                mean += values[j];
            }
            let n = values.len() as f64;
            if n > 0.0 {
                mean /= n;
            }

            let mut sum_sq_cross = 0.0_f64;
            let mut pairs = 0.0_f64;
            for a in 0..values.len() {
                let da = values[a] - mean;
                for b in 0..values.len() {
                    if a == b {
                        continue;
                    }
                    let db = values[b] - mean;
                    let c = da * db;
                    sum_sq_cross += c * c;
                    pairs += 1.0;
                }
            }
            let corr_penalty = if pairs > 0.0 {
                (sum_sq_cross / pairs).min(1.0)
            } else {
                0.0
            };

            let score = (1.0 - loss) - corr_penalty;
            if score > 0.0 {
                score
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / safe_trials as f64;

    base_score - moment_penalty
}

fn evaluate_sequence(seq: &[usize], steps: usize, noise_amp: f64, trials: usize) -> f64 {
    let mut gen = PinkNoiseGenerator::new(steps);
    let pool: Vec<Vec<f64>> = (0..trials).map(|_| gen.generate_new()).collect();
    evaluate_sequence_with_pool(seq, &pool, noise_amp, trials, 3, 3)
}

fn compute_moment_penalty(seq: &[usize], steps: usize, max_order: usize) -> f64 {
    if max_order == 0 || steps == 0 {
        return 0.0;
    }

    let mut moments = [0.0_f64; 3];
    let mut y = 1.0_f64;
    let mut pulse_idx = 0usize;

    for t in 0..steps {
        while pulse_idx < seq.len() && seq[pulse_idx] == t {
            y *= -1.0;
            pulse_idx += 1;
        }
        let u = t as f64 / steps as f64;
        let mut power = 1.0_f64;
        for k in 0..max_order {
            if k == 0 {
                moments[k] += y * power;
            } else {
                power *= u;
                moments[k] += y * power;
            }
        }
    }

    let norm = steps as f64;
    let mut sum_sq = 0.0_f64;
    for k in 0..max_order {
        let v = moments[k] / norm;
        sum_sq += v * v;
    }
    sum_sq / max_order as f64
}
