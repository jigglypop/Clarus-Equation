use crate::packed_params::Packed128;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// 4.3.2 상태-전이 미분: 이산 공간의 "미분"
/// 
/// 이산 상태에 대한 그래디언트를 상태 전이 확률로 근사
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// 온도 파라미터 (보통 β = 10)
    pub temperature: f32,
    /// 빔 서치 크기 (K = 16)
    pub beam_size: usize,
    /// 상태 전이 히스토리
    pub transition_history: HashMap<u64, Vec<f32>>,
}

impl StateTransition {
    pub fn new() -> Self {
        Self {
            temperature: 10.0,
            beam_size: 16,
            transition_history: HashMap::new(),
        }
    }
    
    /// 4.3.2.1 상태-전이 미분의 수학적 정의
    /// 
    /// ∂L/∂s ≈ L(s) - min_{s'≠s} L(s')
    /// 현재 상태와 최적 대안 상태 사이의 손실 차이로 그래디언트 근사
    pub fn compute_state_gradients(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
    ) -> HashMap<String, f32> {
        let mut state_gradients = HashMap::new();
        
        // 현재 손실 계산
        let current_loss = self.compute_loss(target, predicted);
        
        // 주요 상태 비트들에 대한 전이 확률 계산
        let quadrant_gradients = self.compute_quadrant_transitions(
            target, params, rows, cols, current_loss
        );
        
        let frequency_gradients = self.compute_frequency_transitions(
            target, params, rows, cols, current_loss
        );
        
        let amplitude_gradients = self.compute_amplitude_transitions(
            target, params, rows, cols, current_loss
        );
        
        // 상태 그래디언트 결합
        state_gradients.insert("quadrant".to_string(), quadrant_gradients);
        state_gradients.insert("frequency".to_string(), frequency_gradients);
        state_gradients.insert("amplitude".to_string(), amplitude_gradients);
        
        state_gradients
    }
    
    /// 4.3.2.2 확률적 상태 전이 규칙
    /// 
    /// P(s → s') = softmax(-β · ΔL_{s→s'})
    /// 여기서 ΔL_{s→s'} = L(s') - L(s)
    pub fn apply_probabilistic_transition(
        &mut self,
        params: &mut Packed128,
        state_gradients: &HashMap<String, f32>,
        epoch: i32,
    ) {
        // 온도 감소 (simulated annealing)
        let current_temperature = self.temperature / (1.0 + 0.1 * epoch as f32);
        
        // 각 상태 그룹에 대해 확률적 전이 수행
        if let Some(&quadrant_grad) = state_gradients.get("quadrant") {
            self.update_quadrant_probabilistic(params, quadrant_grad, current_temperature);
        }
        
        if let Some(&frequency_grad) = state_gradients.get("frequency") {
            self.update_frequency_probabilistic(params, frequency_grad, current_temperature);
        }
        
        if let Some(&amplitude_grad) = state_gradients.get("amplitude") {
            self.update_amplitude_probabilistic(params, amplitude_grad, current_temperature);
        }
    }
    
    /// 4.3.2.3 멀티-비트 동시 업데이트 (빔 서치)
    /// 
    /// arg min_{(quad,freq,amp)} L(quad,freq,amp,basis,cordic)
    /// 67M 조합을 빔 서치로 근사하여 상위 K=16개 후보만 유지
    pub fn beam_search_optimization(
        &self,
        params: &Packed128,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> Packed128 {
        let mut candidates = vec![*params];
        let mut _rng = StdRng::from_entropy();
        
        // 빔 서치 반복
        for _iteration in 0..4 {
            let mut new_candidates = Vec::new();
            
            for candidate in &candidates {
                // 각 후보에서 1-비트 변경 시도
                for bit_pos in 0..20 {
                    let mut new_candidate = *candidate;
                    new_candidate.hi ^= 1u64 << bit_pos;
                    new_candidates.push(new_candidate);
                }
            }
            
            // 상위 K개 후보 선택
            new_candidates.sort_by(|a, b| {
                let loss_a = self.evaluate_candidate_loss(a, target, rows, cols);
                let loss_b = self.evaluate_candidate_loss(b, target, rows, cols);
                loss_a.partial_cmp(&loss_b).unwrap()
            });
            
            candidates = new_candidates.into_iter().take(self.beam_size).collect();
        }
        
        candidates[0]
    }
    
    // Helper methods
    fn compute_loss(&self, target: &[f32], predicted: &[f32]) -> f32 {
        target.iter().zip(predicted.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>() / target.len() as f32
    }
    
    fn compute_quadrant_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 4가지 사분면에 대한 전이 확률 계산
        let mut best_loss_diff = 0.0;
        
        for quadrant in 0..4 {
            let mut test_params = *params;
            // 사분면 비트 수정 (상위 2비트)
            test_params.hi = (test_params.hi & !0xC000000000000000) | ((quadrant as u64) << 62);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn compute_frequency_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 12비트 주파수 필드 (비트 50-61) 에 대한 전이 계산
        let mut best_loss_diff = 0.0;
        let current_freq = (params.hi >> 50) & 0xFFF;
        
        // 주파수를 ±10% 변경해보며 최적 전이 찾기
        let freq_variations = [
            (current_freq + 100).min(0xFFF),
            current_freq.saturating_sub(100),
            (current_freq + 200).min(0xFFF),
            current_freq.saturating_sub(200),
        ];
        
        for &new_freq in &freq_variations {
            let mut test_params = *params;
            // 주파수 비트 수정
            test_params.hi = (test_params.hi & !(0xFFFu64 << 50)) | (new_freq << 50);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn compute_amplitude_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 12비트 진폭 필드 (비트 38-49) 에 대한 전이 계산
        let mut best_loss_diff = 0.0;
        let current_amp = (params.hi >> 38) & 0xFFF;
        
        // 진폭을 ±15% 변경해보며 최적 전이 찾기
        let amp_variations = [
            (current_amp + 150).min(0xFFF),
            current_amp.saturating_sub(150),
            (current_amp + 300).min(0xFFF),
            current_amp.saturating_sub(300),
            current_amp ^ 0x800, // 상위 비트 플립
        ];
        
        for &new_amp in &amp_variations {
            let mut test_params = *params;
            // 진폭 비트 수정
            test_params.hi = (test_params.hi & !(0xFFFu64 << 38)) | (new_amp << 38);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn update_quadrant_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.1 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 사분면 전이
                let current_quadrant = (params.hi >> 62) & 0x3;
                let new_quadrant = (current_quadrant + 1) % 4;
                params.hi = (params.hi & !0xC000000000000000) | (new_quadrant << 62);
            }
        }
    }
    
    fn update_frequency_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.05 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal.abs() / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 주파수 전이
                let current_freq = (params.hi >> 50) & 0xFFF;
                let direction = if gradient_signal > 0.0 { 1 } else { -1 };
                let freq_change = (rng.gen::<u64>() % 100) * direction as u64;
                
                let new_freq = if direction > 0 {
                    (current_freq + freq_change).min(0xFFF)
                } else {
                    current_freq.saturating_sub(freq_change)
                };
                
                params.hi = (params.hi & !(0xFFFu64 << 50)) | (new_freq << 50);
            }
        }
    }
    
    fn update_amplitude_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.08 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal.abs() / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 진폭 전이
                let current_amp = (params.hi >> 38) & 0xFFF;
                
                // 세 가지 전이 모드 중 선택
                let transition_mode = rng.gen::<u32>() % 3;
                let new_amp = match transition_mode {
                    0 => {
                        // 선형 증감
                        let direction = if gradient_signal > 0.0 { 1 } else { -1 };
                        let amp_change = (rng.gen::<u64>() % 200) * direction as u64;
                        
                        if direction > 0 {
                            (current_amp + amp_change).min(0xFFF)
                        } else {
                            current_amp.saturating_sub(amp_change)
                        }
                    },
                    1 => {
                        // 비트 플립
                        let flip_bit = rng.gen::<u64>() % 12;
                        current_amp ^ (1u64 << flip_bit)
                    },
                    _ => {
                        // 극값으로 점프
                        if rng.gen::<bool>() { 0xFFF } else { 0x000 }
                    }
                };
                
                params.hi = (params.hi & !(0xFFFu64 << 38)) | (new_amp << 38);
            }
        }
    }
    
    fn evaluate_candidate_loss(
        &self,
        params: &Packed128,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 후보 파라미터로 예측값 생성 후 손실 계산
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = params.fused_forward(i, j, rows, cols);
            }
        }
        self.compute_loss(target, &predicted)
    }
}
 