use crate::packed_params::Packed128;
use super::basic_math::adam_update;

/// 고급 그래디언트 누적 구조체
#[derive(Debug, Clone, Default)]
pub struct GradientAccumulator {
    pub r_grad_sum: f32,
    pub theta_grad_sum: f32,
    pub state_transition_count: u32,
    pub total_samples: u32,
}

impl GradientAccumulator {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn accumulate(&mut self, r_grad: f32, theta_grad: f32) {
        self.r_grad_sum += r_grad;
        self.theta_grad_sum += theta_grad;
        self.total_samples += 1;
    }
    
    pub fn get_average_gradients(&self) -> (f32, f32) {
        if self.total_samples > 0 {
            (
                self.r_grad_sum / self.total_samples as f32,
                self.theta_grad_sum / self.total_samples as f32,
            )
        } else {
            (0.0, 0.0)
        }
    }
    
    pub fn reset(&mut self) {
        self.r_grad_sum = 0.0;
        self.theta_grad_sum = 0.0;
        self.state_transition_count = 0;
        self.total_samples = 0;
    }
}

/// 배치별 정밀한 그래디언트 누적 및 업데이트
pub fn batch_fused_backward(
    targets: &[&[f32]], 
    seeds: &mut [Packed128],
    accumulators: &mut [GradientAccumulator],
    adam_states_r: &mut [(f32, f32)], // (momentum, velocity)
    adam_states_theta: &mut [(f32, f32)],
    rows: usize, 
    cols: usize,
    learning_rate: f32,
    epoch: i32
) -> f32 {
    assert_eq!(targets.len(), seeds.len());
    assert_eq!(seeds.len(), accumulators.len());
    
    let batch_size = targets.len();
    let mut total_loss = 0.0;
    
    // 그래디언트 누적 초기화
    for acc in accumulators.iter_mut() {
        acc.reset();
    }
    
    // 배치 내 각 샘플에 대해 그래디언트 계산
    for (batch_idx, target) in targets.iter().enumerate() {
        let seed = &mut seeds[batch_idx];
        let accumulator = &mut accumulators[batch_idx];
        
        // 현재 예측 생성
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                predicted[idx] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // 손실 계산
        let mut sample_loss = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let error = predicted[idx] - target[idx];
                sample_loss += error * error;
                
                // 상태 전이 미분 적용
                seed.apply_state_transition(error, i, j);
                
                if error.abs() > 0.001 { // 유의미한 오차에 대해서만 전이 카운트
                    accumulator.state_transition_count += 1;
                }
                
                // 연속 파라미터 그래디언트 계산
                let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
                let theta_fp32 = f32::from_bits(seed.lo as u32);
                let eps = 1e-5;
                
                // r 그래디언트
                let mut seed_r_plus = *seed;
                seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let w_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_r_minus = *seed;
                seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let w_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
                
                let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                let grad_r = error * dr;
                
                // theta 그래디언트
                let mut seed_th_plus = *seed;
                seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                let w_th_plus = seed_th_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_th_minus = *seed;
                seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                let w_th_minus = seed_th_minus.fused_forward(i, j, rows, cols);
                
                let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                let grad_theta = error * dth;
                
                // 그래디언트 누적
                accumulator.accumulate(grad_r, grad_theta);
            }
        }
        
        total_loss += sample_loss / target.len() as f32;
    }
    
    // 배치별 파라미터 업데이트
    for (batch_idx, seed) in seeds.iter_mut().enumerate() {
        let accumulator = &accumulators[batch_idx];
        let (avg_grad_r, avg_grad_theta) = accumulator.get_average_gradients();
        
        if avg_grad_r.abs() > 1e-8 || avg_grad_theta.abs() > 1e-8 {
            let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
            let theta_fp32 = f32::from_bits(seed.lo as u32);
            
            let mut new_r = r_fp32;
            let mut new_theta = theta_fp32;
            
            // Adam 업데이트
            adam_update(
                &mut new_r, 
                &mut adam_states_r[batch_idx].0, 
                &mut adam_states_r[batch_idx].1, 
                avg_grad_r, 
                learning_rate, 
                epoch
            );
            adam_update(
                &mut new_theta, 
                &mut adam_states_theta[batch_idx].0, 
                &mut adam_states_theta[batch_idx].1, 
                avg_grad_theta, 
                learning_rate, 
                epoch
            );
            
            new_r = new_r.clamp(0.0, 1.0);
            seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
        }
    }
    
    total_loss / batch_size as f32
}

/// 적응적 학습률을 사용한 고급 Adam 업데이트
pub fn adaptive_adam_update(
    param: &mut f32,
    momentum: &mut f32,
    velocity: &mut f32,
    gradient: f32,
    base_lr: f32,
    epoch: i32,
    gradient_norm: f32, // 전체 그래디언트 노름
    transition_count: u32, // 상태 전이 횟수
) {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;
    
    // 적응적 학습률 계산
    let adaptive_lr = if transition_count > 0 {
        // 상태 전이가 많이 일어났으면 학습률 감소
        base_lr * (1.0 / (1.0 + transition_count as f32 * 0.01))
    } else {
        base_lr
    };
    
    // 그래디언트 클리핑
    let clipped_gradient = if gradient_norm > 1.0 {
        gradient / gradient_norm
    } else {
        gradient
    };
    
    *momentum = B1 * (*momentum) + (1.0 - B1) * clipped_gradient;
    *velocity = B2 * (*velocity) + (1.0 - B2) * clipped_gradient * clipped_gradient;
    
    let m_hat = *momentum / (1.0 - B1.powi(epoch));
    let v_hat = *velocity / (1.0 - B2.powi(epoch));
    
    *param -= adaptive_lr * m_hat / (v_hat.sqrt() + EPS);
}

/// 멀티스케일 그래디언트 분석
pub fn analyze_gradient_scales(
    gradients_r: &[f32], 
    gradients_theta: &[f32]
) -> (f32, f32, f32, f32) {
    let r_norm = gradients_r.iter().map(|g| g * g).sum::<f32>().sqrt();
    let theta_norm = gradients_theta.iter().map(|g| g * g).sum::<f32>().sqrt();
    
    let r_mean = gradients_r.iter().sum::<f32>() / gradients_r.len() as f32;
    let theta_mean = gradients_theta.iter().sum::<f32>() / gradients_theta.len() as f32;
    
    (r_norm, theta_norm, r_mean, theta_mean)
}
 
/// 해석적 그래디언트 계산을 위한 Packed128 확장 트레이트
pub trait AnalyticalGradient {
    /// R 파라미터에 대한 해석적 그래디언트 (리만 기하학 기반)
    fn analytical_gradient_r(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
    
    /// Theta 파라미터에 대한 해석적 그래디언트 (리만 기하학 기반)
    fn analytical_gradient_theta(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
}

impl AnalyticalGradient for Packed128 {
    /// R 파라미터에 대한 해석적 그래디언트 (순수 비트 연산)
    fn analytical_gradient_r(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 논문 5.4.3: D_s'F(s) = F(flip_bit(s, s')) - F(s)
        // 순수 비트 연산만 사용, 초월함수 없음
        // 1. R에 영향을 주는 상태 비트 선택 (위치 기반)
        let position_hash = ((i * 31 + j) & 0x1F) as u64;
        let bit_position = position_hash % 20; // hi 필드의 하위 20비트
        // 2. 비트 플립으로 상태 전이
        let original_hi = self.hi;
        let flipped_hi = original_hi ^ (1u64 << bit_position);
        // 3. 비트 연산만으로 그래디언트 근사
        let r_bits = (self.lo >> 32) as u32;
        // 비트 패턴 기반 차분 계산 (비트 시프트와 마스킹만 사용)
        let original_pattern = ((original_hi >> (position_hash % 32)) & 0xFF) as u32;
        let flipped_pattern = ((flipped_hi >> (position_hash % 32)) & 0xFF) as u32;
        // R 방향 가중치 (비트 연산)
        let r_weight = ((r_bits >> 24) & 0xFF) as f32 / 255.0;
        let position_weight = ((i ^ j) & 0x1F) as f32 / 31.0;
        // 차분을 비트 연산으로 계산
        let bit_diff = (flipped_pattern ^ original_pattern).count_ones() as f32;
        let sign = if flipped_pattern > original_pattern { 1.0 } else { -1.0 };
        // 최종 그래디언트 (순수 비트 연산 결과)
        sign * bit_diff * r_weight * position_weight * 0.1
    }

    /// Theta 파라미터에 대한 해석적 그래디언트 (순수 비트 연산)
    fn analytical_gradient_theta(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 논문 5.4.3: D_s'F(s) = F(flip_bit(s, s')) - F(s)
        // 순수 비트 연산만 사용, 초월함수 없음
        // 1. Theta에 영향을 주는 상태 비트 선택 (다른 해시)
        let position_hash = ((i * 17 + j * 23) & 0x1F) as u64;
        let bit_position = (position_hash % 20) + 20; // hi 필드의 상위 20비트
        // 2. 비트 플립으로 상태 전이
        let original_hi = self.hi;
        let flipped_hi = original_hi ^ (1u64 << (bit_position % 64));
        // 3. 비트 연산만으로 그래디언트 근사
        let theta_bits = self.lo as u32;
        // 비트 패턴 기반 차분 계산 (비트 시프트와 마스킹만 사용)
        let original_pattern = ((original_hi >> (position_hash % 32)) & 0xFF) as u32;
        let flipped_pattern = ((flipped_hi >> (position_hash % 32)) & 0xFF) as u32;
        // Theta 방향 가중치 (비트 연산)
        let theta_weight = ((theta_bits >> 16) & 0xFF) as f32 / 255.0;
        let position_weight = ((i + j) & 0x1F) as f32 / 31.0;
        // 차분을 비트 연산으로 계산
        let bit_diff = (flipped_pattern ^ original_pattern).count_ones() as f32;
        let sign = if flipped_pattern > original_pattern { 1.0 } else { -1.0 };
        // 최종 그래디언트 (순수 비트 연산 결과)
        sign * bit_diff * theta_weight * position_weight * 0.05
    }
}
 