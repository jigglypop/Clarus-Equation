use crate::packed_params::Packed128;
use std::collections::HashMap;

/// 4.3.3 하이브리드 그래디언트 적용
/// 
/// 연속 파라미터와 이산 상태의 그래디언트를 조합적으로 적용
#[derive(Debug, Clone)]
pub struct HybridOptimizer {
    /// Adam 파라미터들
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// 모멘텀 상태
    pub momentum_r: f32,
    pub momentum_theta: f32,
    /// 속도 상태
    pub velocity_r: f32,
    pub velocity_theta: f32,
    /// 적응적 학습률 파라미터
    pub learning_rate_decay: f32,
    pub discrete_transition_decay: f32,
}

impl HybridOptimizer {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum_r: 0.0,
            momentum_theta: 0.0,
            velocity_r: 0.0,
            velocity_theta: 0.0,
            learning_rate_decay: 0.95,
            discrete_transition_decay: 0.8,
        }
    }
    
    /// 하이브리드 파라미터 업데이트
    /// 
    /// 업데이트 순서:
    /// 1. 연속 파라미터 업데이트 (Adam)
    /// 2. 이산 상태 확률적 업데이트
    /// 3. 전체 성능 검증 및 롤백
    pub fn update_parameters(
        &mut self,
        params: &mut Packed128,
        grad_r: f32,
        grad_theta: f32,
        _state_gradients: &HashMap<String, f32>,
        learning_rate: f32,
        epoch: i32,
    ) -> f32 {
        // 백업
        let backup_params = *params;
        
        // 1단계: 연속 파라미터 업데이트 (Adam)
        self.update_continuous_parameters(params, grad_r, grad_theta, learning_rate, epoch);
        
        // 2단계: 이산 상태 확률적 업데이트는 StateTransition에서 수행
        
        // 3단계: 성능 검증 (간소화된 버전)
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // 기본적인 수치 안정성 체크
        if !r_fp32.is_finite() || !theta_fp32.is_finite() {
            *params = backup_params;
        }
        
        // MSE 근사 계산
        let error_estimate = grad_r.powi(2) + grad_theta.powi(2);
        error_estimate
    }
    
    /// Adam 옵티마이저를 사용한 연속 파라미터 업데이트
    fn update_continuous_parameters(
        &mut self,
        params: &mut Packed128,
        grad_r: f32,
        grad_theta: f32,
        learning_rate: f32,
        epoch: i32,
    ) {
        // 현재 파라미터 추출
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // Adam 업데이트 for r
        self.momentum_r = self.beta1 * self.momentum_r + (1.0 - self.beta1) * grad_r;
        self.velocity_r = self.beta2 * self.velocity_r + (1.0 - self.beta2) * grad_r.powi(2);
        
        let bias_correction_1_r = 1.0 - self.beta1.powi(epoch + 1);
        let bias_correction_2_r = 1.0 - self.beta2.powi(epoch + 1);
        
        let corrected_momentum_r = self.momentum_r / bias_correction_1_r;
        let corrected_velocity_r = self.velocity_r / bias_correction_2_r;
        
        let new_r = r_fp32 - learning_rate * corrected_momentum_r / (corrected_velocity_r.sqrt() + self.epsilon);
        
        // Adam 업데이트 for theta
        self.momentum_theta = self.beta1 * self.momentum_theta + (1.0 - self.beta1) * grad_theta;
        self.velocity_theta = self.beta2 * self.velocity_theta + (1.0 - self.beta2) * grad_theta.powi(2);
        
        let bias_correction_1_theta = 1.0 - self.beta1.powi(epoch + 1);
        let bias_correction_2_theta = 1.0 - self.beta2.powi(epoch + 1);
        
        let corrected_momentum_theta = self.momentum_theta / bias_correction_1_theta;
        let corrected_velocity_theta = self.velocity_theta / bias_correction_2_theta;
        
        let new_theta = theta_fp32 - learning_rate * corrected_momentum_theta / (corrected_velocity_theta.sqrt() + self.epsilon);
        
        // 파라미터 업데이트
        params.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    }
    
    /// 4.6.2 적응적 학습률 스케줄링
    /// 
    /// 연속 파라미터: α_r(t) = α_0 · (t_0/(t_0+t))^0.5
    /// 이산 상태 전이: P_transition(t) = P_0 · (t_0/(t_0+t))^2.0
    pub fn get_adaptive_learning_rate(&self, base_rate: f32, epoch: i32) -> f32 {
        let t_0 = 100.0;
        let t = epoch as f32;
        base_rate * (t_0 / (t_0 + t)).powf(0.5)
    }
    
    pub fn get_discrete_transition_probability(&self, base_prob: f32, epoch: i32) -> f32 {
        let t_0 = 100.0;
        let t = epoch as f32;
        base_prob * (t_0 / (t_0 + t)).powf(2.0)
    }
} 