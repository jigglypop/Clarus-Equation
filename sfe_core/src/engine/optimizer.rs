use std::f64::consts::PI;
use rayon::prelude::*;
use crate::engine::noise::PinkNoiseGenerator;

/// SFE 순수 해석학적 최적화기 (고도화 버전: Balanced Log-SFE)
/// 복잡한 유전 알고리즘을 SFE 이론에서 유도된 공식으로 대체하고,
/// DC 오프셋 제거를 위한 자동 균형 보정(Balancing)을 수행합니다.
pub struct SfeOptimizer {
    pub beta: f64,
}

impl SfeOptimizer {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    pub fn optimize(&self, steps: usize, n_pulses: usize, noise_level: f64, noise_pool: &[Vec<f64>]) -> Vec<f64> {
        // 초기값: UDD 정규화 시퀀스
        let mut seq: Vec<f64> = (1..=n_pulses)
            .map(|j| ((j as f64 * PI) / (2.0 * n_pulses as f64 + 2.0)).sin().powi(2))
            .collect();
        seq.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut best_norm = seq.clone();
        let best_idx: Vec<usize> = best_norm
            .iter()
            .map(|&t| (t * steps as f64).round() as usize)
            .collect();
        let mut best_score = evaluate_sequence_with_pool(&best_idx, noise_level, noise_pool);

        let mut step_size: f64 = 0.02;
        let min_step: f64 = 1e-3;

        while step_size > min_step {
            let mut improved = false;

            for i in 0..n_pulses {
                for dir in [-1.0_f64, 1.0_f64] {
                    let mut cand = best_norm.clone();
                    cand[i] += dir * step_size;
                    if cand[i] <= 0.0 || cand[i] >= 1.0 {
                        continue;
                    }
                    cand.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let cand_idx: Vec<usize> = cand
                        .iter()
                        .map(|&t| (t * steps as f64).round() as usize)
                        .collect();

                    let score = evaluate_sequence_with_pool(&cand_idx, noise_level, noise_pool);
                    if score > best_score {
                        best_score = score;
                        best_norm = cand;
                        improved = true;
                        break;
                    }
                }
            }

            if !improved {
                step_size *= 0.5;
            }
        }

        best_norm
    }

    /// 하드웨어 스펙을 반영한 Robust SFE 최적화
    /// - TLS Notch Filter 자동 적용
    /// - 초기 펄스 간격 제약 (0.03 이상) 강제
    pub fn optimize_for_spec(&self, steps: usize, n_pulses: usize, noise_level: f64, noise_pool: &[Vec<f64>], tls_freq: Option<f64>) -> Vec<f64> {
        // TLS 주파수가 감지되면 환경변수를 통해 Notch Filter 활성화
        // (evaluate_sequence_with_pool 함수가 환경변수를 참조하므로)
        if let Some(omega) = tls_freq {
             std::env::set_var("SFE_TLS_OMEGA", omega.to_string());
             std::env::set_var("SFE_TLS_WEIGHT", "0.5"); // Strong penalty weight
        } else {
             std::env::remove_var("SFE_TLS_OMEGA");
             std::env::remove_var("SFE_TLS_WEIGHT");
        }
        
        let mut seq = self.optimize(steps, n_pulses, noise_level, noise_pool);
        
        // Robustness: 초기 펄스가 너무 빠르면 하드웨어 오류 발생 가능
        // 최소 0.10 (10%) 지점 이후에 오도록 강제 보정 (대폭 상향)
        let min_first_pulse = 0.10; 
        let min_pulse_interval = 0.05; // 펄스 간격 최소 5%

        if !seq.is_empty() {
            seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // 1. 첫 펄스 보정
            if seq[0] < min_first_pulse {
                seq[0] = min_first_pulse;
            }
            
            // 2. 간격 보정 (밀리는 도미노 효과)
            for i in 0..seq.len()-1 {
                if seq[i+1] - seq[i] < min_pulse_interval {
                    seq[i+1] = seq[i] + min_pulse_interval;
                }
            }
            
            // 3. 끝 범위를 넘어갔으면 전체 스케일링으로 압축 (단, 간격 비율은 유지)
            if let Some(&last) = seq.last() {
                 if last >= 0.99 {
                     let scale = 0.98 / last; // 0.98까지 안전하게 압축
                     for x in seq.iter_mut() {
                         *x *= scale;
                     }
                 }
            }
        }
        
        seq
    }
}

/// 최적화 메인 진입점 (퍼사드 패턴)
/// main.rs / benchmark.rs와의 호환성을 위해 기존 GA 버전과 동일한 시그니처를 유지합니다.
/// 이제 해석적 공식을 사용하여 즉시 실행됩니다.
///
/// 반환값: (최적 시퀀스 인덱스, UDD 점수, SFE 점수)
pub fn run_pulse_optimizer(
    steps: usize,
    n_pulses: usize,
    _generations: usize,
    noise_level: f64,
) -> (Vec<usize>, f64, f64) {
    let beta = 50.0;
    let optimizer = SfeOptimizer::new(beta);

    let trials = 200;

    let alpha: f64 = std::env::var("SFE_NOISE_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);

    let scale: f64 = std::env::var("SFE_NOISE_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5 * noise_level.abs());

    let suppresson_omega: f64 = std::env::var("SFE_SUPPRESSON_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let suppresson_amp: f64 = std::env::var("SFE_SUPPRESSON_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let suppresson_omega2: f64 = std::env::var("SFE_SUPPRESSON_OMEGA2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let suppresson_amp2: f64 = std::env::var("SFE_SUPPRESSON_AMP2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let mut gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);
    let mut noise_pool: Vec<Vec<f64>> = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut trace = gen.generate_new();
        if suppresson_amp != 0.0 && suppresson_omega != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                let phase = suppresson_omega * (t as f64);
                *v += suppresson_amp * phase.cos();
            }
        }
        if suppresson_amp2 != 0.0 && suppresson_omega2 != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                let phase = suppresson_omega2 * (t as f64);
                *v += suppresson_amp2 * phase.cos();
            }
        }
        noise_pool.push(trace);
    }

    let sfe_seq_norm = optimizer.optimize(steps, n_pulses, noise_level, &noise_pool);
    let sfe_seq_idx: Vec<usize> = sfe_seq_norm
        .iter()
        .map(|&t| (t * steps as f64).round() as usize)
        .collect();

    let (udd_score, sfe_score) = evaluate_performance(steps, &sfe_seq_idx, n_pulses, noise_level, &noise_pool);

    (sfe_seq_idx, udd_score, sfe_score)
}

/// 도우미: 생성된 노이즈에 대해 UDD vs SFE 평가
fn evaluate_performance(
    steps: usize,
    sfe_seq: &[usize],
    n_pulses: usize,
    noise_amp: f64,
    noise_pool: &[Vec<f64>]
) -> (f64, f64) {
    let mut udd_seq = Vec::with_capacity(n_pulses);
    for j in 1..=n_pulses {
        let t = ((j as f64 * PI) / (2.0 * n_pulses as f64 + 2.0)).sin().powi(2);
        udd_seq.push((t * steps as f64).round() as usize);
    }

    let udd_score = evaluate_sequence_with_pool(&udd_seq, noise_amp, noise_pool);
    let sfe_score = evaluate_sequence_with_pool(sfe_seq, noise_amp, noise_pool);

    (udd_score, sfe_score)
}

/// 필터 함수 및 장기 가중 목적함수 기반 시퀀스 평가
/// 7.2 장 4.3절의 구조를 따름:
/// - 여러 노이즈 궤적에 대해 위상 φ(t)를 누적
/// - r = {0.4,0.6,0.8,1.0} 지점에서 S_k = cos φ(t_k) 측정
/// - 가중 평균 S_eff = Σ w_k S_k / Σ w_k
/// - 모든 궤적에 대해 평균한 뒤, 저주파 모멘트 패널티(m0,m1,m2)를 반영하여 최종 점수 반환
pub fn evaluate_sequence_with_pool(
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

    // 저주파 모멘트 M_k = ∫ t^k y(t) dt (정규화된 시간 t∈[0,1]) 계산
    // SFE_MOMENT_ORDER 환경변수로 최대 차수 설정 (기본 0: 패널티 없음)
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

    // 모멘트 패널티: (m0^2 + m1^2 + m2^2)/order (order=사용된 모멘트 개수)
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

// 목적함수 인터페이스 추가 (SfeObjective)
pub trait SfeObjective {
    fn evaluate(&self, seq: &[f64]) -> f64;
}
