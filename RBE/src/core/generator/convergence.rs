use crate::packed_params::Packed128;

/// 4.5 수렴성 분석
/// 
/// 하이브리드 학습의 수렴 조건 모니터링
#[derive(Debug, Clone)]
pub struct ConvergenceAnalyzer {
    /// 손실 히스토리
    pub loss_history: Vec<f32>,
    /// 그래디언트 노름 히스토리
    pub gradient_norm_history: Vec<f32>,
    /// 수렴 임계값
    pub convergence_threshold: f32,
    /// 패턴 안정성 윈도우
    pub stability_window: usize,
}

impl ConvergenceAnalyzer {
    pub fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            gradient_norm_history: Vec::new(),
            convergence_threshold: 1e-6,
            stability_window: 10,
        }
    }
    
    /// 수렴성 검사
    /// 
    /// 수렴 조건:
    /// 1. 손실 변화율 < threshold
    /// 2. 그래디언트 노름 < threshold
    /// 3. stability_window 기간 동안 안정성 유지
    pub fn check_convergence(&mut self, loss: f32, gradient_norm: f32) -> bool {
        // 히스토리 업데이트
        self.loss_history.push(loss);
        self.gradient_norm_history.push(gradient_norm);
        
        // 윈도우 크기 제한
        if self.loss_history.len() > 100 {
            self.loss_history.remove(0);
        }
        if self.gradient_norm_history.len() > 100 {
            self.gradient_norm_history.remove(0);
        }
        
        // 충분한 데이터가 없으면 수렴하지 않음
        if self.loss_history.len() < self.stability_window {
            return false;
        }
        
        // 1. 손실 변화율 검사
        let recent_losses = &self.loss_history[self.loss_history.len() - self.stability_window..];
        let loss_variance = self.compute_variance(recent_losses);
        
        // 2. 그래디언트 노름 검사
        let recent_gradients = &self.gradient_norm_history[self.gradient_norm_history.len() - self.stability_window..];
        let avg_gradient_norm = recent_gradients.iter().sum::<f32>() / recent_gradients.len() as f32;
        
        // 수렴 조건 확인
        loss_variance < self.convergence_threshold && avg_gradient_norm < self.convergence_threshold
    }
    
    /// 푸앵카레 볼 수렴 조건 검증
    /// 
    /// 추가 검증: 파라미터가 유효한 푸앵카레 볼 영역에 있는지 확인
    pub fn verify_convergence_conditions(&self, params: &Packed128) -> bool {
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // 기본 유효성 검사
        if !r_fp32.is_finite() || !theta_fp32.is_finite() {
            return false;
        }
        
        // 푸앵카레 볼 제약 확인
        if r_fp32 < 0.0 || r_fp32 >= 1.0 {
            return false;
        }
        
        // 수치적 안정성 확인
        if r_fp32.abs() < 1e-10 && theta_fp32.abs() < 1e-10 {
            return false; // 너무 작은 값들
        }
        
        true
    }
    
    /// 분산 계산 헬퍼
    fn compute_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return f32::INFINITY;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance
    }
} 