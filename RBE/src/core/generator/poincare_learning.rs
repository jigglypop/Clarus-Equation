//! 4장: 푸앵카레 볼 학습 - 압축된 공간에서의 직접 학습
//! 
//! 본 모듈은 RBE의 핵심인 "압축된 기하학적 공간에서의 직접 학습"을 구현합니다.
//! 128비트 Packed128 파라미터로부터 즉석에서 가중치를 생성하며 동시에 학습하는
//! 하이브리드 학습 전략을 제공합니다.

use crate::packed_params::Packed128;
use crate::math::AnalyticalGradient;
use super::{HybridOptimizer, StateTransition, ConstraintProjection, RegularizationTerms, ConvergenceAnalyzer};

/// 4.1 푸앵카레 볼 학습의 핵심 시스템
/// 
/// 이중 파라미터 학습 전략:
/// - hi (이산 상태): 상태-전이 미분을 통한 확률적 비트 플립
/// - lo (연속 파라미터): 해석적 그래디언트를 통한 표준 그래디언트 하강
#[derive(Debug, Clone)]
pub struct PoincareLearning {
    /// 하이브리드 최적화기
    pub hybrid_optimizer: HybridOptimizer,
    /// 상태 전이 시스템
    pub state_transition: StateTransition,
    /// 제약 투영 시스템
    pub constraint_projection: ConstraintProjection,
    /// 정규화 시스템
    pub regularization: RegularizationTerms,
    /// 수렴성 분석기
    pub convergence_analyzer: ConvergenceAnalyzer,
}

impl PoincareLearning {
    /// 새로운 푸앵카레 볼 학습 시스템 생성
    pub fn new() -> Self {
        Self {
            hybrid_optimizer: HybridOptimizer::new(),
            state_transition: StateTransition::new(),
            constraint_projection: ConstraintProjection::new(),
            regularization: RegularizationTerms::new(),
            convergence_analyzer: ConvergenceAnalyzer::new(),
        }
    }
    
    /// 4.3 융합 역전파: 이중 파라미터 그래디언트 계산
    /// 
    /// 연속 파라미터와 이산 상태의 그래디언트를 조합적으로 계산
    pub fn fused_backward_hybrid(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        params: &mut Packed128,
        rows: usize,
        cols: usize,
        learning_rate: f32,
        epoch: i32,
    ) -> (f32, f32) {
        // 1. 연속 파라미터 그래디언트 계산 (해석적)
        let (grad_r, grad_theta) = self.compute_continuous_gradients(
            target, predicted, params, rows, cols
        );
        
        // 2. 이산 상태 그래디언트 계산 (상태-전이)
        let state_gradients = self.state_transition.compute_state_gradients(
            target, predicted, params, rows, cols
        );
        
        // 3. 하이브리드 파라미터 업데이트
        let mse = self.hybrid_optimizer.update_parameters(
            params, 
            grad_r, 
            grad_theta, 
            &state_gradients,
            learning_rate,
            epoch
        );
        
        // 4. 제약 투영 적용
        self.constraint_projection.project_to_poincare_ball(params);
        
        // 5. 정규화 항 계산
        let regularized_loss = mse + self.regularization.compute_regularization_loss(params);
        
        (regularized_loss, regularized_loss.sqrt())
    }
    
    /// 연속 파라미터 해석적 그래디언트 계산
    fn compute_continuous_gradients(
        &self,
        target: &[f32],
        predicted: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
    ) -> (f32, f32) {
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let error = predicted[idx] - target[idx];
                
                // 해석적 그래디언트 사용 (4배 성능 향상)
                let dr = params.analytical_gradient_r(i, j, rows, cols);
                let dtheta = params.analytical_gradient_theta(i, j, rows, cols);
                
                grad_r_sum += error * dr;
                grad_theta_sum += error * dtheta;
            }
        }
        
        let batch_size = (rows * cols) as f32;
        (grad_r_sum / batch_size, grad_theta_sum / batch_size)
    }
} 