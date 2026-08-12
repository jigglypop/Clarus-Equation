use crate::packed_params::Packed128;

/// 4.4.2 정규화 항
/// 
/// L2 정규화와 복잡도 패널티를 결합
#[derive(Debug, Clone)]
pub struct RegularizationTerms {
    /// L2 정규화 강도
    pub l2_weight: f32,
    /// 복잡도 패널티 강도
    pub complexity_weight: f32,
    /// 진폭 분산 패널티
    pub variance_penalty: f32,
}

impl RegularizationTerms {
    pub fn new() -> Self {
        Self {
            l2_weight: 0.001,
            complexity_weight: 0.0001,
            variance_penalty: 0.01,
        }
    }
    
    /// 정규화 손실 계산
    /// 
    /// L_reg = λ₁‖θ‖² + λ₂·complexity(s) + λ₃·variance(θ)
    pub fn compute_regularization_loss(&mut self, params: &Packed128) -> f32 {
        let mut total_loss = 0.0;
        
        // 1. L2 정규화
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        let l2_loss = self.l2_weight * (r_fp32.powi(2) + theta_fp32.powi(2));
        total_loss += l2_loss;
        
        // 2. 복잡도 패널티 (이산 상태의 다양성)
        let complexity_score = self.compute_discrete_complexity(params);
        let complexity_loss = self.complexity_weight * complexity_score;
        total_loss += complexity_loss;
        
        // 3. 진폭 분산 패널티
        let variance_score = self.compute_parameter_variance(params);
        let variance_loss = self.variance_penalty * variance_score;
        total_loss += variance_loss;
        
        total_loss
    }
    
    /// 이산 상태 복잡도 계산
    fn compute_discrete_complexity(&self, params: &Packed128) -> f32 {
        // 상위 20비트의 복잡도 측정 (해밍 가중치)
        let active_bits = params.hi.count_ones() as f32;
        let total_bits = 20.0;
        
        // 너무 단순(모든 0)하거나 너무 복잡(모든 1)한 경우 패널티
        let normalized_complexity = active_bits / total_bits;
        let optimal_complexity = 0.5; // 50% 활성화가 이상적
        
        (normalized_complexity - optimal_complexity).abs()
    }
    
    /// 파라미터 분산 계산
    fn compute_parameter_variance(&self, params: &Packed128) -> f32 {
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // 간단한 분산 근사: 평균으로부터의 편차
        let param_mean = (r_fp32 + theta_fp32) / 2.0;
        let variance = ((r_fp32 - param_mean).powi(2) + (theta_fp32 - param_mean).powi(2)) / 2.0;
        
        variance
    }
} 