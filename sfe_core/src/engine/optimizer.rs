use rayon::prelude::*;
use rand::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use super::noise::PinkNoiseGenerator; // Updated import

const DT: f64 = 0.01;

/// [Advanced] Island Model Genetic Algorithm + Local Polishing + Smart Init
/// - Divides population into 'islands' to maintain diversity.
/// - Migrates elites between islands.
/// - Applies local search (polishing) to the final best solution.
/// - Uses Noise Profiling for Smart Initialization.
pub fn run_pulse_optimizer(steps: usize, n_pulses: usize, generations: usize, noise_amp: f64) -> (Vec<usize>, f64, f64) {
    println!("Starting SFE-Genetic Pulse Optimizer (Island Model)...");
    
    // Pre-generate a noise pool to avoid FFT overhead in inner loops
    // We use a large pool and sample from it or use all of it.
    // For GA stability, using the same noise environment for a generation is good.
    println!(">> Pre-generating noise pool (Hardware Matched: alpha=0.8, scale=1.5)...");
    let pool_size = 2000;
    // IBM Fez Hardware-Matched Noise Model
    let mut noise_gen = PinkNoiseGenerator::new_with_params(steps, 0.8, 1.5);
    let mut noise_pool: Vec<Vec<f64>> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        noise_pool.push(noise_gen.generate_new());
    }
    
    // 1. Initialize with UDD (SOTA) as the seed
    let mut best_sequence: Vec<usize> = Vec::with_capacity(n_pulses);
    for j in 1..=n_pulses {
        let sin_val = (j as f64 * std::f64::consts::PI / (2.0 * (n_pulses as f64 + 1.0))).sin();
        let t_j = (steps as f64 * sin_val.powi(2)).round() as usize;
        if t_j > 0 && t_j < steps { best_sequence.push(t_j); }
    }
    best_sequence.sort();
    best_sequence.dedup();
    
    // Baseline Score (UDD)
    // Use a subset of the pool for evaluation
    let udd_score = evaluate_sequence_with_pool(&best_sequence, &noise_pool, noise_amp, 200);
    
    let mut global_best_score = udd_score;
    let mut global_best_seq = best_sequence.clone();

    let pb = ProgressBar::new(generations as u64);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [Gen {pos}/{len}] Best: {msg}").unwrap());

    // --- Island Model Config ---
    let num_islands = 4;
    let island_pop_size = 50; 
    
    // [NEW] Smart Initialization
    // Create seeds based on noise profiling (random noise sample gradient analysis)
    let smart_seeds: Vec<Vec<usize>> = (0..num_islands).map(|i| {
        // Use a few samples from the pool for profiling
        let sample_idx = i % pool_size;
        let sample_noise = &noise_pool[sample_idx];
        
        // Find regions with high gradient (rapid change) and place pulses there
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
        while seed.len() < n_pulses { // Fill rest randomly
             let mut rng = thread_rng();
             let t = rng.gen_range(1..steps);
             if !seed.contains(&t) { seed.push(t); }
        }
        seed.sort();
        seed
    }).collect();

    // Initialize Islands: UDD + Smart Seeds + Random
    let mut islands: Vec<Vec<(Vec<usize>, f64)>> = (0..num_islands).map(|island_idx| {
        let mut pop = Vec::with_capacity(island_pop_size);
        // Elite Seed (UDD)
        pop.push((best_sequence.clone(), udd_score));
        // Smart Seed
        let smart_score = evaluate_sequence_with_pool(&smart_seeds[island_idx], &noise_pool, noise_amp, 200);
        pop.push((smart_seeds[island_idx].clone(), smart_score));
        
        // Fill rest with mutated UDD or Random
        let mut rng = thread_rng();
        while pop.len() < island_pop_size {
             let mut p = best_sequence.clone();
             // Heavy mutation for diversity
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
             // Initial eval
             let score = evaluate_sequence_with_pool(&p, &noise_pool, noise_amp, 200);
             pop.push((p, score));
        }
        pop
    }).collect();

    for gen in 0..generations {
        // Evolve each island in parallel
        islands.par_iter_mut().for_each(|island| {
            // 1. Evaluate & Sort
            island.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // 2. Evolution Step
            let elites = island[0..5].to_vec(); // Top 5 elites
            let mut new_pop = Vec::with_capacity(island_pop_size);
            new_pop.extend(elites.clone());

            let mut rng = thread_rng();
            while new_pop.len() < island_pop_size {
                // Tournament Selection
                let idx1 = rng.gen_range(0..20);
                let idx2 = rng.gen_range(0..20);
                let p1 = &island[idx1].0;
                let p2 = &island[idx2].0;
                
                // Crossover
                let mut child = Vec::with_capacity(n_pulses);
                for k in 0..p1.len().min(p2.len()) {
                    if rng.gen_bool(0.5) { child.push(p1[k]); } else { child.push(p2[k]); }
                }
                
                // Adaptive Mutation
                let mutation_rate = 0.4 * (1.0 - gen as f64 / generations as f64) + 0.05;
                let mutation_power = (steps as f64 * 0.1 * (1.0 - gen as f64 / generations as f64)).max(1.0) as i32;

                for t in &mut child {
                    if rng.gen_bool(mutation_rate) {
                        let shift = rng.gen_range(-mutation_power..=mutation_power);
                        let new_val = (*t as i32 + shift).clamp(1, (steps - 1) as i32) as usize;
                        *t = new_val;
                    }
                }
                
                // Fix Constraints
                child.sort();
                child.dedup();
                while child.len() < n_pulses {
                    let new_t = rng.gen_range(1..steps);
                    if !child.contains(&new_t) { child.push(new_t); }
                }
                child.sort();

                // Eval using pool
                let score = evaluate_sequence_with_pool(&child, &noise_pool, noise_amp, 200);
                new_pop.push((child, score));
            }
            *island = new_pop;
        });

        // Migration (Every 5 gens)
        if gen % 5 == 0 {
            let mut migrants = Vec::new();
            for island in &islands {
                migrants.push(island[0].clone()); // Copy best
            }
            for i in 0..num_islands {
                let target_island = (i + 1) % num_islands;
                let worst_idx = island_pop_size - 1;
                islands[target_island][worst_idx] = migrants[i].clone();
            }
        }

        // Update Global Best
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

    // --- Final Polishing (Local Search) ---
    println!(">> Polishing best solution...");
    // Generate a fresh, larger pool for polishing to ensure robustness
    let mut polishing_pool = noise_pool.clone();
    for _ in 0..1000 {
        polishing_pool.push(noise_gen.generate_new());
    }

    let polished_seq = local_search_polish(&global_best_seq, steps, noise_amp, &polishing_pool, 500); 
    let polished_score = evaluate_sequence_with_pool(&polished_seq, &polishing_pool, noise_amp, 1000); 
    
    println!(">> Polishing Result: {:.5} -> {:.5}", global_best_score, polished_score);

    (polished_seq, udd_score, polished_score)
}

fn local_search_polish(seq: &[usize], steps: usize, noise_amp: f64, noise_pool: &[Vec<f64>], trials: usize) -> Vec<usize> {
    let mut current_seq = seq.to_vec();
    let mut current_score = evaluate_sequence_with_pool(&current_seq, noise_pool, noise_amp, trials);
    
    let mut improved = true;
    let mut rng = thread_rng();

    // Try small shifts for each pulse
    let max_iters = 50; 
    for _ in 0..max_iters {
        improved = false;
        // Pick random pulse to wiggle
        let idx = rng.gen_range(0..current_seq.len());
        
        // Try shifting left and right
        for shift in [-1, 1] {
            let mut neighbor = current_seq.clone();
            let new_val = (neighbor[idx] as i32 + shift).clamp(1, (steps - 1) as i32) as usize;
            
            // Check collision
            if !neighbor.contains(&new_val) {
                neighbor[idx] = new_val;
                neighbor.sort(); 
                
                let score = evaluate_sequence_with_pool(&neighbor, noise_pool, noise_amp, trials);
                if score > current_score {
                    current_seq = neighbor;
                    current_score = score;
                    improved = true;
                }
            }
        }
        if !improved { break; } // Local optima reached
    }
    current_seq
}

/// Optimized evaluator using pre-generated noise pool
pub fn evaluate_sequence_with_pool(seq: &[usize], noise_pool: &[Vec<f64>], noise_amp: f64, trials: usize) -> f64 {
    let pool_len = noise_pool.len();
    let safe_trials = trials.min(pool_len);

    (0..safe_trials).into_par_iter().map(|i| {
        let pink_noise = &noise_pool[i]; 
        let mut phase = 0.0;
        let mut sign = 1.0;
        let mut pulse_idx = 0;
        let len = pink_noise.len();
        let c0 = ((len as f64 * 0.4).round() as usize).min(len.saturating_sub(1));
        let c1 = ((len as f64 * 0.6).round() as usize).min(len.saturating_sub(1));
        let c2 = ((len as f64 * 0.8).round() as usize).min(len.saturating_sub(1));
        let c3 = len.saturating_sub(1);
        let checkpoints = [c0, c1, c2, c3];
        let weights = [1.0_f64, 2.0, 3.0, 4.0];
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
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..values.len() {
            num += weights[j] * values[j];
            den += weights[j];
        }
        if den > 0.0 { num / den } else { 0.0 }
    }).sum::<f64>() / safe_trials as f64
}

// Deprecated legacy wrapper if needed, but not used here internally
fn evaluate_sequence(seq: &[usize], steps: usize, noise_amp: f64, trials: usize) -> f64 {
    // On-the-fly generation (slow) - kept only if needed for old tests
    let mut gen = PinkNoiseGenerator::new(steps);
    let pool: Vec<Vec<f64>> = (0..trials).map(|_| gen.generate_new()).collect();
    evaluate_sequence_with_pool(seq, &pool, noise_amp, trials)
}
