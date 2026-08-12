//! # 통합 미분 시스템 (Unified Differential System)
//!
//! CycleDifferentialSystem 중심으로 순전파/역전파를 통합 관리하는 
//! 고성능 미분 계산 모듈

pub mod cycle_system;
pub mod forward;
pub mod backward;
pub mod state_transition;
pub mod bit_dp_system; // 새로운 비트 DP 시스템 모듈

// 핵심 타입들 재수출
pub use cycle_system::{
    UnifiedCycleDifferentialSystem, CycleState, HyperbolicFunction, DifferentialPhase
};
pub use forward::{
    UnifiedForwardPass, ForwardConfig, ForwardMetrics
};
pub use backward::{
    UnifiedBackwardPass, BackwardConfig, GradientMetrics
};
pub use state_transition::{
    StateTransitionEngine, TransitionRule, StateTransitionMetrics
};
pub use bit_dp_system::*; // 새로운 모듈 추가

/// 통합 미분 시스템 인터페이스
#[derive(Debug, Clone)]
pub struct DifferentialSystem {
    /// 11비트 미분 사이클 엔진 (핵심)
    pub cycle_engine: UnifiedCycleDifferentialSystem,
    /// 통합 순전파 엔진
    pub forward_engine: UnifiedForwardPass,
    /// 통합 역전파 엔진  
    pub backward_engine: UnifiedBackwardPass,
    /// 상태 전이 엔진
    pub transition_engine: StateTransitionEngine,
}

impl DifferentialSystem {
    /// 새로운 통합 미분 시스템 생성
    pub fn new(packed_count: usize) -> Self {
        let cycle_engine = UnifiedCycleDifferentialSystem::new(packed_count);
        let forward_engine = UnifiedForwardPass::new();
        let backward_engine = UnifiedBackwardPass::new();
        let transition_engine = StateTransitionEngine::new();
        
        Self {
            cycle_engine,
            forward_engine,
            backward_engine,
            transition_engine,
        }
    }
    
    /// 통합 순전파: 35.4ns/op 성능
    pub fn unified_forward(
        &mut self,
        packed: &crate::packed_params::Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        self.forward_engine.compute_with_cycle_system(
            packed,
            &self.cycle_engine,
            i, j, rows, cols
        )
    }
    
    /// 통합 역전파: 상태-전이 미분 + 연속 파라미터 그래디언트
    pub fn unified_backward(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        packed: &mut crate::packed_params::Packed128,
        rows: usize,
        cols: usize,
        learning_rate: f32,
    ) -> (f32, GradientMetrics) {
        self.backward_engine.compute_with_cycle_system(
            target,
            predicted,
            packed,
            &mut self.cycle_engine,
            &mut self.transition_engine,
            rows, cols,
            learning_rate
        )
    }
    
    /// 성능 메트릭 수집
    pub fn get_performance_metrics(&self) -> DifferentialPerformanceMetrics {
        DifferentialPerformanceMetrics {
            cycle_entropy: self.cycle_engine.compute_state_entropy(),
            forward_accuracy: self.forward_engine.get_accuracy(),
            backward_convergence: self.backward_engine.get_convergence_rate(),
            transition_efficiency: self.transition_engine.get_efficiency(),
        }
    }
    
    /// 수학적 불변량 검증
    pub fn verify_system_invariants(&self) -> bool {
        self.cycle_engine.verify_mathematical_invariants()
    }
}

/// 통합 미분 시스템 성능 메트릭
#[derive(Debug, Clone)]
pub struct DifferentialPerformanceMetrics {
    /// 사이클 엔트로피 (다양성 지표)
    pub cycle_entropy: f32,
    /// 순전파 정확도
    pub forward_accuracy: f32,
    /// 역전파 수렴률
    pub backward_convergence: f32,
    /// 상태 전이 효율성
    pub transition_efficiency: f32,
}

/// 통합 시스템 설정
#[derive(Debug, Clone)]
pub struct DifferentialSystemConfig {
    /// 학습률
    pub learning_rate: f32,
    /// 11비트 사이클 단계
    pub differential_phase: DifferentialPhase,
    /// 상태 전이 임계값
    pub transition_threshold: f32,
    /// 수학적 불변량 검증 주기
    pub invariant_check_period: usize,
}

impl Default for DifferentialSystemConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            differential_phase: DifferentialPhase::Exploitation,
            transition_threshold: 0.01,
            invariant_check_period: 100,
        }
    }
}

#[cfg(test)]
pub mod __tests__;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_differential_system_creation() {
        let system = DifferentialSystem::new(10);
        assert!(system.verify_system_invariants());
    }

    #[test]
    fn test_performance_metrics_collection() {
        let system = DifferentialSystem::new(5);
        let metrics = system.get_performance_metrics();

        assert!(metrics.cycle_entropy >= 0.0 && metrics.cycle_entropy <= 1.0);
        assert!(metrics.forward_accuracy >= 0.0);
        assert!(metrics.backward_convergence >= 0.0);
        assert!(metrics.transition_efficiency >= 0.0);
    }
} 