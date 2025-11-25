use std::f64::consts::PI;
use rayon::prelude::*;
use crate::engine::filter::FilterFunction;

#[derive(Clone, Debug)]
pub struct HardwareConstraints {
    pub min_first_pulse: f64,
    pub min_interval: f64,
    pub max_last_pulse: f64,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            min_first_pulse: 0.10,
            min_interval: 0.05,
            max_last_pulse: 0.98,
        }
    }
}

pub struct SfeOptimizerV2 {
    pub beta: f64,
    pub constraints: HardwareConstraints,
}

impl SfeOptimizerV2 {
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            constraints: HardwareConstraints::default(),
        }
    }
    
    pub fn new_with_constraints(beta: f64, constraints: HardwareConstraints) -> Self {
        Self { beta, constraints }
    }
    
    fn initialize_feasible(&self, n_pulses: usize) -> Vec<f64> {
        let mut seq: Vec<f64> = (1..=n_pulses)
            .map(|j| ((j as f64 * PI) / (2.0 * n_pulses as f64 + 2.0)).sin().powi(2))
            .collect();
        seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if !self.is_feasible(&seq) {
            seq = self.project_to_feasible(seq);
        }
        
        seq
    }

    /// 수치적 구배 계산 (Numerical Gradient for Filter Function)
    fn compute_gradient(&self, seq: &[f64], spectrum_fn: &impl Fn(f64) -> f64, duration: f64) -> Vec<f64> {
        let n = seq.len();
        let mut grad = vec![0.0; n];
        let eps = 1e-5;
        
        // 현재 점수 계산
        let ff_curr = FilterFunction::compute(seq, duration, 512);
        let score_curr = -ff_curr.integrate_with_spectrum(spectrum_fn);
        
        for i in 0..n {
            let mut cand = seq.to_vec();
            cand[i] += eps;
            // 미소 변동
            
            let ff_cand = FilterFunction::compute(&cand, duration, 512);
            let score_cand = -ff_cand.integrate_with_spectrum(spectrum_fn);
            
            grad[i] = (score_cand - score_curr) / eps;
        }
        
        grad
    }

    /// 리만 기하학적 최적화 (Riemannian Gradient Descent)
    /// 8장 이론의 Reality_Stone 개념 적용: 다양체(Manifold) 위에서의 구배 하강
    pub fn optimize_riemannian(
        &self,
        n_pulses: usize,
        spectrum_fn: impl Fn(f64) -> f64 + Send + Sync,
    ) -> Vec<f64> {
        let mut seq = self.initialize_feasible(n_pulses);
        let duration = 100.0;
        
        let mut step_size = 0.01;
        let min_step = 1e-6;
        let max_iter = 200;
        
        for _iter in 0..max_iter {
            if step_size < min_step { break; }
            
            // 1. 리만 구배 근사 (유클리드 구배 계산)
            let grad = self.compute_gradient(&seq, &spectrum_fn, duration);
            
            // 2. 접공간 업데이트 (Tangent Space Step)
            let mut new_seq = seq.clone();
            let mut norm_grad = 0.0;
            for g in &grad { norm_grad += g * g; }
            norm_grad = norm_grad.sqrt();
            
            if norm_grad < 1e-9 { break; }

            for i in 0..n_pulses {
                new_seq[i] += step_size * grad[i] / norm_grad; // 정규화된 구배 방향 이동
            }
            
            // 3. Retraction (다양체 위로 투영)
            new_seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !self.is_feasible(&new_seq) {
                new_seq = self.project_to_feasible(new_seq);
            }
            
            // 4. 라인 서치 (Line Search)와 유사한 수락 조건
            let ff_prev = FilterFunction::compute(&seq, duration, 512);
            let score_prev = -ff_prev.integrate_with_spectrum(&spectrum_fn);
            
            let ff_new = FilterFunction::compute(&new_seq, duration, 512);
            let score_new = -ff_new.integrate_with_spectrum(&spectrum_fn);
            
            if score_new > score_prev {
                seq = new_seq;
                step_size *= 1.05; // 가속
            } else {
                step_size *= 0.5; // 감속
            }
        }
        
        seq
    }
    
    fn is_feasible(&self, seq: &[f64]) -> bool {
        if seq.is_empty() {
            return true;
        }
        
        if seq[0] < self.constraints.min_first_pulse {
            return false;
        }
        
        for i in 0..seq.len()-1 {
            if seq[i+1] - seq[i] < self.constraints.min_interval {
                return false;
            }
        }
        
        if let Some(&last) = seq.last() {
            if last > self.constraints.max_last_pulse {
                return false;
            }
        }
        
        true
    }
    
    fn project_to_feasible(&self, mut seq: Vec<f64>) -> Vec<f64> {
        if seq.is_empty() {
            return seq;
        }
        
        // Stack Overflow 방지를 위한 반복문 구조 변경 (최대 10회 재시도)
        for _ in 0..10 {
            seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // 1. 첫 펄스 보정
            seq[0] = seq[0].max(self.constraints.min_first_pulse);
            
            // 2. 간격 보정 (밀어내기)
            let mut interval_violated = false;
            for i in 0..seq.len()-1 {
                if seq[i+1] - seq[i] < self.constraints.min_interval {
                    seq[i+1] = seq[i] + self.constraints.min_interval;
                    interval_violated = true;
                }
            }
            
            // 3. 마지막 펄스 체크 및 스케일링
            let mut range_violated = false;
            if let Some(&last) = seq.last() {
                if last > self.constraints.max_last_pulse {
                    let scale = self.constraints.max_last_pulse / last;
                    for x in seq.iter_mut() {
                        *x *= scale;
                    }
                    range_violated = true;
                }
            }
            
            if !interval_violated && !range_violated {
                return seq;
            }
        }
        
        // 수렴하지 못한 경우 안전장치: 마지막으로 강제 스케일링만 하고 반환
        if let Some(&last) = seq.last() {
            if last > self.constraints.max_last_pulse {
                 let scale = self.constraints.max_last_pulse / last;
                 for x in seq.iter_mut() { *x *= scale; }
            }
        }
        
        seq
    }
    
    fn project_single_move(&self, seq: &[f64], idx: usize, new_val: f64) -> f64 {
        let mut val = new_val;
        
        if idx == 0 {
            val = val.max(self.constraints.min_first_pulse);
        }
        
        if idx > 0 {
            val = val.max(seq[idx-1] + self.constraints.min_interval);
        }
        
        if idx < seq.len() - 1 {
            val = val.min(seq[idx+1] - self.constraints.min_interval);
        }
        
        val = val.clamp(0.0, self.constraints.max_last_pulse);
        
        val
    }
    
    pub fn optimize_constrained(
        &self,
        steps: usize,
        n_pulses: usize,
        noise_level: f64,
        noise_pool: &[Vec<f64>],
    ) -> Vec<f64> {
        let mut seq = self.initialize_feasible(n_pulses);
        
        let best_idx: Vec<usize> = seq
            .iter()
            .map(|&t| (t * steps as f64).round() as usize)
            .collect();
        let mut best_score = evaluate_sequence_with_pool(&best_idx, noise_level, noise_pool);
        
        let mut step_size = 0.02;
        let min_step = 1e-3;
        let max_iterations = 200;
        let mut iteration = 0;
        
        while step_size > min_step && iteration < max_iterations {
            let mut improved = false;
            
            for i in 0..n_pulses {
                for dir in [-1.0, 1.0] {
                    let mut cand = seq.clone();
                    let new_val = cand[i] + dir * step_size;
                    
                    cand[i] = self.project_single_move(&cand, i, new_val);
                    
                    if (cand[i] - seq[i]).abs() < 1e-6 {
                        continue;
                    }
                    
                    cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    
                    if !self.is_feasible(&cand) {
                        cand = self.project_to_feasible(cand);
                    }
                    
                    let cand_idx: Vec<usize> = cand
                        .iter()
                        .map(|&t| (t * steps as f64).round() as usize)
                        .collect();
                    
                    let score = evaluate_sequence_with_pool(&cand_idx, noise_level, noise_pool);
                    
                    if score > best_score {
                        best_score = score;
                        seq = cand;
                        improved = true;
                        break;
                    }
                }
                
                if improved {
                    break;
                }
            }
            
            if !improved {
                step_size *= 0.7;
            }
            
            iteration += 1;
        }
        
        seq
    }
    
    pub fn optimize_with_filter_function(
        &self,
        _steps: usize,
        n_pulses: usize,
        spectrum_fn: impl Fn(f64) -> f64 + Send + Sync,
    ) -> Vec<f64> {
        let mut seq = self.initialize_feasible(n_pulses);
        
        let duration = 100.0;
        let ff = FilterFunction::compute(&seq, duration, 512);
        let mut best_score = -ff.integrate_with_spectrum(&spectrum_fn);
        
        let mut step_size = 0.02;
        let min_step = 1e-3;
        
        while step_size > min_step {
            let mut improved = false;
            
            for i in 0..n_pulses {
                for dir in [-1.0, 1.0] {
                    let mut cand = seq.clone();
                    let new_val = cand[i] + dir * step_size;
                    
                    cand[i] = self.project_single_move(&cand, i, new_val);
                    
                    if (cand[i] - seq[i]).abs() < 1e-6 {
                        continue;
                    }
                    
                    cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    
                    if !self.is_feasible(&cand) {
                        cand = self.project_to_feasible(cand);
                    }
                    
                    let ff_cand = FilterFunction::compute(&cand, duration, 512);
                    let score = -ff_cand.integrate_with_spectrum(&spectrum_fn);
                    
                    if score > best_score {
                        best_score = score;
                        seq = cand;
                        improved = true;
                        break;
                    }
                }
            }
            
            if !improved {
                step_size *= 0.7;
            }
        }
        
        seq
    }
}

fn evaluate_sequence_with_pool(
    pulses: &[usize],
    noise_amp: f64,
    noise_pool: &[Vec<f64>],
) -> f64 {
    if noise_pool.is_empty() {
        return 0.0;
    }

    let steps = noise_pool[0].len();
    let ratios: [f64; 4] = [0.4, 0.6, 0.8, 1.0];
    let weights: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let weight_sum: f64 = weights.iter().sum();

    let mut pulses_sorted = pulses.to_vec();
    pulses_sorted.sort_unstable();
    let n_pulses = pulses_sorted.len();

    let dt: f64 = 1.0;

    let mut y = vec![1.0_f64; steps];
    let mut current_sign = 1.0_f64;
    let mut pulse_idx = 0_usize;

    for t in 0..steps {
        if pulse_idx < n_pulses && t == pulses_sorted[pulse_idx] {
            current_sign *= -1.0;
            pulse_idx += 1;
        }
        y[t] = current_sign;
    }

    let moment_order: usize = std::env::var("SFE_MOMENT_ORDER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
        .min(3);

    let mut m = [0.0_f64; 3];
    if moment_order > 0 {
        let steps_f = (steps.saturating_sub(1)) as f64;
        if steps_f > 0.0 {
            let dt_norm = 1.0 / steps_f;
            for (t_idx, &y_val) in y.iter().enumerate() {
                let t_norm = t_idx as f64 / steps_f;
                if moment_order >= 1 {
                    m[0] += y_val * dt_norm;
                }
                if moment_order >= 2 {
                    m[1] += t_norm * y_val * dt_norm;
                }
                if moment_order >= 3 {
                    m[2] += t_norm * t_norm * y_val * dt_norm;
                }
            }
        }
    }

    let mut moment_penalty = 0.0_f64;
    if moment_order > 0 {
        let mut acc = 0.0_f64;
        for k in 0..moment_order {
            acc += m[k] * m[k];
        }
        moment_penalty = acc / (moment_order as f64);
    }

    let tls_omega: f64 = std::env::var("SFE_TLS_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let tls_weight: f64 = std::env::var("SFE_TLS_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let mut tls_penalty = 0.0_f64;
    if tls_omega > 0.0 && tls_weight > 0.0 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for t in 0..steps {
            let phase = tls_omega * (t as f64);
            let c = phase.cos();
            let s = phase.sin();
            let v = y[t];
            re += v * c;
            im += v * s;
        }
        let norm = steps as f64;
        let y2 = (re * re + im * im) / (norm * norm);
        tls_penalty = tls_weight * y2;
    }

    let scores: Vec<f64> = noise_pool
        .par_iter()
        .map(|noise| {
            let check_indices: Vec<usize> = ratios
                .iter()
                .map(|r| ((steps as f64 - 1.0) * r).round() as usize)
                .collect();

            let mut s_vals = [0.0_f64; 4];
            let mut phase = 0.0_f64;
            let mut next_check = 0_usize;

            for t in 0..steps {
                phase += y[t] * noise[t] * noise_amp * dt;

                if next_check < check_indices.len() && t == check_indices[next_check] {
                    s_vals[next_check] = phase.cos();
                    next_check += 1;
                }
            }

            let mut acc = 0.0_f64;
            for k in 0..4 {
                acc += weights[k] * s_vals[k];
            }
            acc / weight_sum
        })
        .collect();

    let sum: f64 = scores.iter().sum();
    let avg_s = sum / scores.len() as f64;

    avg_s - moment_penalty - tls_penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::noise::PinkNoiseGenerator;
    
    #[test]
    fn test_feasibility_projection() {
        let optimizer = SfeOptimizerV2::new(50.0);
        
        let infeasible = vec![0.05, 0.06, 0.99];
        let feasible = optimizer.project_to_feasible(infeasible);
        
        assert!(optimizer.is_feasible(&feasible));
        assert!(feasible[0] >= 0.10);
        assert!(feasible[1] - feasible[0] >= 0.05);
    }
    
    #[test]
    fn test_constrained_optimization() {
        let optimizer = SfeOptimizerV2::new(50.0);
        
        let mut gen = PinkNoiseGenerator::new_with_params(2000, 0.8, 1.5);
        let mut noise_pool = Vec::with_capacity(50);
        for _ in 0..50 {
            noise_pool.push(gen.generate_new());
        }
        
        let seq = optimizer.optimize_constrained(2000, 8, 0.15, &noise_pool);
        
        assert!(optimizer.is_feasible(&seq));
        assert_eq!(seq.len(), 8);
        assert!(seq[0] >= 0.10);
    }
}

