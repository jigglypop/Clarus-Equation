use anyhow::Result;
use rayon::prelude::*;

/// RBESoftmax - 푸앵카레 볼 기하학 기반 소프트맥스
/// 
/// 핵심 혁신:
/// 1. 푸앵카레 메트릭을 활용한 거리 기반 가중치
/// 2. 11비트 사이클 상태와 동기화된 온도 조절
/// 3. 수치적 안정성을 위한 쌍곡함수 활용
#[derive(Debug, Clone)]
pub struct RBESoftmax {
    /// 소프트맥스 차원 (-1은 마지막 차원)
    pub dim: isize,
    
    /// 온도 매개변수 (기본값: 1.0)
    pub temperature: f32,
    
    /// 푸앵카레 곡률 매개변수
    pub curvature: f32,
    
    /// 11비트 사이클 동기화 여부
    pub cycle_sync: bool,
    
    /// 수치적 안정성을 위한 epsilon
    pub epsilon: f32,
}

impl RBESoftmax {
    pub fn new(dim: isize) -> Self {
        Self {
            dim,
            temperature: 1.0,
            curvature: 1.0,
            cycle_sync: false,
            epsilon: 1e-9,
        }
    }
    
    /// 푸앵카레 거리 기반 가중치 계산
    fn poincare_distance_weight(&self, logit: f32, max_logit: f32) -> f32 {
        // 로짓을 푸앵카레 볼 좌표로 변환
        let r = (logit - max_logit).abs() / (logit.abs() + max_logit.abs() + self.epsilon);
        let r = r.min(0.9999); // 경계 안정성
        
        // 푸앵카레 메트릭 가중치
        // ds² = 4/(1-r²)² dr²
        let metric_factor = 2.0 / (1.0 - r * r);
        
        // 쌍곡탄젠트로 부드러운 전이
        metric_factor.tanh()
    }
    
    /// 11비트 사이클 동기화 온도 조절
    fn cycle_adjusted_temperature(&self, cycle_state: u16) -> f32 {
        if !self.cycle_sync {
            return self.temperature;
        }
        
        // 11비트 사이클을 [0, 1] 범위로 정규화
        let cycle_phase = (cycle_state & 0x7FF) as f32 / 2048.0;
        
        // 사인파로 온도 변조 (0.5 ~ 1.5 범위)
        let temp_modulation = 1.0 + 0.5 * (2.0 * std::f32::consts::PI * cycle_phase).sin();
        
        self.temperature * temp_modulation
    }
    
    /// 표준 소프트맥스 (비교 검증용)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // 수치적 안정성을 위해 최댓값 빼기
        let exp_vals: Vec<f32> = input
            .iter()
            .map(|&x| ((x - max_val) / self.temperature).exp())
            .collect();
        
        let sum: f32 = exp_vals.iter().sum();
        
        exp_vals.iter().map(|&x| x / sum).collect()
    }
    
    /// 푸앵카레 볼 소프트맥스
    pub fn forward_poincare(&self, input: &[f32], cycle_state: Option<u16>) -> Vec<f32> {
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // 사이클 동기화 온도
        let temp = if let Some(cycle) = cycle_state {
            self.cycle_adjusted_temperature(cycle)
        } else {
            self.temperature
        };
        
        // 푸앵카레 거리 가중치 적용
        let weighted_exp_vals: Vec<f32> = input
            .iter()
            .map(|&x| {
                let weight = self.poincare_distance_weight(x, max_val);
                let exp_val = ((x - max_val) / temp).exp();
                exp_val * weight
            })
            .collect();
        
        let sum: f32 = weighted_exp_vals.iter().sum();
        
        weighted_exp_vals.iter().map(|&x| x / sum).collect()
    }
    
    /// 2D 배치 소프트맥스 (병렬 처리)
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input
            .par_iter()
            .map(|row| self.forward(row))
            .collect()
    }
    
    /// 로그 소프트맥스 (수치적으로 안정)
    pub fn log_forward(&self, input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // log(exp(x - max) / sum(exp(x - max))) = x - max - log(sum(exp(x - max)))
        let shifted: Vec<f32> = input.iter().map(|&x| x - max_val).collect();
        let log_sum_exp = shifted
            .iter()
            .map(|&x| (x / self.temperature).exp())
            .sum::<f32>()
            .ln();
        
        shifted
            .iter()
            .map(|&x| x / self.temperature - log_sum_exp)
            .collect()
    }
    
    /// 소프트맥스의 야코비안 계산 (역전파용)
    pub fn jacobian(&self, softmax_output: &[f32]) -> Vec<Vec<f32>> {
        let n = softmax_output.len();
        let mut jacobian = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
                } else {
                    jacobian[i][j] = -softmax_output[i] * softmax_output[j];
                }
            }
        }
        
        jacobian
    }
    
    /// Gumbel-Softmax (미분 가능한 샘플링)
    pub fn gumbel_softmax(&self, input: &[f32], tau: f32) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Gumbel noise 추가
        let gumbel_noise: Vec<f32> = (0..input.len())
            .map(|_| {
                let u: f32 = rng.gen_range(self.epsilon..1.0);
                -(-u.ln()).ln()
            })
            .collect();
        
        // Gumbel-Max trick
        let perturbed: Vec<f32> = input
            .iter()
            .zip(&gumbel_noise)
            .map(|(&x, &g)| (x + g) / tau)
            .collect();
        
        self.forward(&perturbed)
    }
} 