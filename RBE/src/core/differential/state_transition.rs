//! # 상태 전이 엔진 (State Transition Engine)
//!
//! 11비트 미분 사이클의 상태 전이 규칙과 효율성을 관리하는 시스템

use super::cycle_system::{CycleState, HyperbolicFunction, DifferentialPhase};
use std::collections::HashMap;

/// 상태 전이 엔진
#[derive(Debug, Clone)]
pub struct StateTransitionEngine {
    /// 전이 규칙 테이블
    transition_rules: TransitionRuleTable,
    /// 효율성 메트릭
    pub efficiency_tracker: EfficiencyTracker,
    /// 전이 통계
    transition_stats: TransitionStatistics,
}

/// 전이 규칙
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionRule {
    /// 그래디언트 크기 기반
    GradientMagnitude,
    /// 함수 종류 기반
    FunctionType,
    /// 학습 단계 기반
    LearningPhase,
    /// 하이브리드 (모든 요소 고려)
    Hybrid,
}

/// 상태 전이 메트릭
#[derive(Debug, Clone)]
pub struct StateTransitionMetrics {
    /// 전이 효율성 (0.0 ~ 1.0)
    pub efficiency: f32,
    /// 전이 빈도 (Hz)
    pub transition_frequency: f32,
    /// 상태 다양성 (엔트로피)
    pub state_diversity: f32,
    /// 수렴 기여도
    pub convergence_contribution: f32,
}

/// 전이 규칙 테이블
#[derive(Debug, Clone)]
struct TransitionRuleTable {
    /// 함수별 전이 임계값
    function_thresholds: HashMap<HyperbolicFunction, f32>,
    /// 학습 단계별 임계값
    phase_thresholds: HashMap<DifferentialPhase, f32>,
    /// 하이브리드 가중치
    hybrid_weights: HybridWeights,
}

/// 하이브리드 가중치
#[derive(Debug, Clone)]
struct HybridWeights {
    gradient_weight: f32,
    function_weight: f32,
    phase_weight: f32,
    history_weight: f32,
}

/// 효율성 추적기
#[derive(Debug, Clone)]
pub struct EfficiencyTracker {
    pub successful_transitions: u64,
    pub total_attempts: u64,
    pub efficiency_history: Vec<f32>,
    pub current_efficiency: f32,
}

/// 전이 통계
#[derive(Debug, Clone)]
struct TransitionStatistics {
    /// 함수별 전이 횟수
    function_transitions: HashMap<HyperbolicFunction, u64>,
    /// 단계별 전이 횟수
    phase_transitions: HashMap<DifferentialPhase, u64>,
    /// 전이 시간 통계
    transition_times: Vec<u64>, // nanoseconds
    /// 평균 전이 시간
    avg_transition_time: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            gradient_weight: 0.4,
            function_weight: 0.3,
            phase_weight: 0.2,
            history_weight: 0.1,
        }
    }
}

impl StateTransitionEngine {
    /// 새로운 상태 전이 엔진 생성
    pub fn new() -> Self {
        // 기본 전이 임계값 설정
        let mut function_thresholds = HashMap::new();
        function_thresholds.insert(HyperbolicFunction::Sinh, 0.01);
        function_thresholds.insert(HyperbolicFunction::Cosh, 0.008);
        function_thresholds.insert(HyperbolicFunction::Tanh, 0.012);
        function_thresholds.insert(HyperbolicFunction::Sech2, 0.006);
        
        let mut phase_thresholds = HashMap::new();
        phase_thresholds.insert(DifferentialPhase::Exploration, 0.01);
        phase_thresholds.insert(DifferentialPhase::Exploitation, 0.05);
        phase_thresholds.insert(DifferentialPhase::Convergence, 0.1);
        
        Self {
            transition_rules: TransitionRuleTable {
                function_thresholds,
                phase_thresholds,
                hybrid_weights: HybridWeights::default(),
            },
            efficiency_tracker: EfficiencyTracker {
                successful_transitions: 0,
                total_attempts: 0,
                efficiency_history: Vec::with_capacity(1000),
                current_efficiency: 1.0, // 초기값을 1.0으로 복원
            },
            transition_stats: TransitionStatistics {
                function_transitions: HashMap::new(),
                phase_transitions: HashMap::new(),
                transition_times: Vec::with_capacity(1000),
                avg_transition_time: 0.0,
            },
        }
    }
    
    /// **핵심: 상태 전이 결정** (11비트 미분 사이클 최적화)
    pub fn should_transition(
        &mut self,
        current_state: &CycleState,
        gradient_signal: f32,
        learning_phase: DifferentialPhase,
        rule: TransitionRule,
    ) -> bool {
        let start_time = std::time::Instant::now();
        self.efficiency_tracker.total_attempts += 1;
        
        let should_transition = match rule {
            TransitionRule::GradientMagnitude => {
                self.gradient_magnitude_rule(gradient_signal)
            },
            TransitionRule::FunctionType => {
                self.function_type_rule(current_state, gradient_signal)
            },
            TransitionRule::LearningPhase => {
                self.learning_phase_rule(learning_phase, gradient_signal)
            },
            TransitionRule::Hybrid => {
                self.hybrid_rule(current_state, gradient_signal, learning_phase)
            },
        };
        
        // 통계 업데이트
        if should_transition {
            self.efficiency_tracker.successful_transitions += 1;
            
            // 함수별 통계
            let current_function = current_state.get_active_function();
            *self.transition_stats.function_transitions
                .entry(current_function)
                .or_insert(0) += 1;
            
            // 단계별 통계
            *self.transition_stats.phase_transitions
                .entry(learning_phase)
                .or_insert(0) += 1;
        }
        
        // 전이 시간 기록 (배치 최적화)
        let transition_time = start_time.elapsed().as_nanos() as u64;
        if self.transition_stats.transition_times.len() < 1000 {
            self.transition_stats.transition_times.push(transition_time);
        } else {
            // 순환 버퍼 방식으로 최적화 (remove(0) 제거)
            let index = (self.efficiency_tracker.total_attempts % 1000) as usize;
            if index < self.transition_stats.transition_times.len() {
                self.transition_stats.transition_times[index] = transition_time;
            }
        }
        
        // 배치 업데이트 (성능 향상을 위해 100회마다만 실행)
        if self.efficiency_tracker.total_attempts % 100 == 0 {
            self.update_efficiency();
            self.update_avg_transition_time();
        } else {
            // 가벼운 즉시 효율성 업데이트
            if self.efficiency_tracker.total_attempts > 0 {
                self.efficiency_tracker.current_efficiency = 
                    self.efficiency_tracker.successful_transitions as f32 / 
                    self.efficiency_tracker.total_attempts as f32;
            }
        }
        
        should_transition
    }
    
    /// 그래디언트 크기 기반 전이 규칙
    fn gradient_magnitude_rule(&self, gradient_signal: f32) -> bool {
        let magnitude = gradient_signal.abs();
        magnitude > 0.01 // 기본 임계값
    }
    
    /// 함수 종류 기반 전이 규칙
    fn function_type_rule(&self, current_state: &CycleState, gradient_signal: f32) -> bool {
        let current_function = current_state.get_active_function();
        let threshold = self.transition_rules.function_thresholds
            .get(&current_function)
            .copied()
            .unwrap_or(0.01);
        
        gradient_signal.abs() > threshold
    }
    
    /// 학습 단계 기반 전이 규칙
    fn learning_phase_rule(&self, learning_phase: DifferentialPhase, gradient_signal: f32) -> bool {
        let threshold = self.transition_rules.phase_thresholds
            .get(&learning_phase)
            .copied()
            .unwrap_or(0.05);
        
        gradient_signal.abs() > threshold
    }
    
    /// **하이브리드 전이 규칙** (모든 요소 통합) - 성능 최적화
    fn hybrid_rule(
        &self,
        current_state: &CycleState,
        gradient_signal: f32,
        learning_phase: DifferentialPhase,
    ) -> bool {
        // 빠른 경로: 극단적인 경우 즉시 결정 (임계값 조정)
        let abs_gradient = gradient_signal.abs();
        if abs_gradient < 0.005 { return false; }  // 0.001 -> 0.005로 완화 (더 많은 빠른 경로)
        if abs_gradient > 0.1 { return true; }     // 0.2 -> 0.1로 강화 (더 많은 빠른 경로)
        
        let weights = &self.transition_rules.hybrid_weights;
        
        // 1. 그래디언트 점수 (나눗셈 최소화)
        let gradient_score = if abs_gradient > 0.05 { weights.gradient_weight } else { 0.0 };
        
        // 2. 함수 점수 (lookup table 방식, 분기 최소화)
        let function_score = match current_state.state_bits {
            0 => weights.function_weight,      // Sinh
            1 => weights.function_weight * 0.8, // Cosh  
            2 => weights.function_weight * 1.2, // Tanh
            _ => weights.function_weight * 0.6, // Sech2
        };
        
        // 3. 단계 점수 (분기 최소화)
        let phase_score = match learning_phase {
            DifferentialPhase::Exploration => weights.phase_weight * 1.2,
            DifferentialPhase::Exploitation => weights.phase_weight,
            DifferentialPhase::Convergence => weights.phase_weight * 0.6,
        };
        
        // 4. 히스토리 점수 (미리 계산된 값 사용)
        let history_score = self.efficiency_tracker.current_efficiency * weights.history_weight;
        
        // 5. 가중합 (단순 덧셈으로 최적화)
        let total_score = gradient_score + function_score + phase_score + history_score;
        
        // 6. 고정 임계값 (조건문 최소화)
        total_score > 0.6
    }
    
    /// 전이 규칙 최적화 (성능 기반 자동 조정)
    pub fn optimize_transition_rules(&mut self) {
        if self.efficiency_tracker.efficiency_history.len() < 100 {
            return; // 충분한 데이터가 없으면 최적화 건너뛰기
        }
        
        let recent_efficiency = self.efficiency_tracker.efficiency_history
            .iter().rev().take(50).sum::<f32>() / 50.0;
        
        let older_efficiency = self.efficiency_tracker.efficiency_history
            .iter().rev().skip(50).take(50).sum::<f32>() / 50.0;
        
        // 효율성이 감소하면 임계값 조정
        if recent_efficiency < older_efficiency {
            // 임계값을 약간 낮춰서 더 많은 전이 허용
            for threshold in self.transition_rules.function_thresholds.values_mut() {
                *threshold *= 0.95;
                *threshold = threshold.max(0.001); // 최소값 보장
            }
            
            for threshold in self.transition_rules.phase_thresholds.values_mut() {
                *threshold *= 0.95;
                *threshold = threshold.max(0.001);
            }
        } else if recent_efficiency > older_efficiency + 0.05 {
            // 효율성이 크게 향상되면 임계값을 약간 높여서 선택적 전이
            for threshold in self.transition_rules.function_thresholds.values_mut() {
                *threshold *= 1.02;
                *threshold = threshold.min(0.1); // 최대값 제한
            }
        }
    }
    
    /// 평균 전이 시간 업데이트
    fn update_avg_transition_time(&mut self) {
        if !self.transition_stats.transition_times.is_empty() {
            let sum: u64 = self.transition_stats.transition_times.iter().sum();
            self.transition_stats.avg_transition_time = 
                sum as f32 / self.transition_stats.transition_times.len() as f32;
        }
    }
    
    /// 효율성 업데이트
    fn update_efficiency(&mut self) {
        if self.efficiency_tracker.total_attempts > 0 {
            self.efficiency_tracker.current_efficiency = 
                self.efficiency_tracker.successful_transitions as f32 / 
                self.efficiency_tracker.total_attempts as f32;
            
            self.efficiency_tracker.efficiency_history.push(self.efficiency_tracker.current_efficiency);
            
            // 히스토리 크기 제한
            if self.efficiency_tracker.efficiency_history.len() > 1000 {
                self.efficiency_tracker.efficiency_history.remove(0);
            }
        }
    }
    
    /// 상태 다양성 계산 (엔트로피 기반)
    fn compute_state_diversity(&self) -> f32 {
        if self.transition_stats.function_transitions.is_empty() {
            return 0.0;
        }
        
        let total_transitions: u64 = self.transition_stats.function_transitions.values().sum();
        if total_transitions == 0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &count in self.transition_stats.function_transitions.values() {
            if count > 0 {
                let p = count as f32 / total_transitions as f32;
                entropy -= p * p.ln();
            }
        }
        
        // 정규화 (최대 엔트로피는 ln(4) = 1.386)
        entropy / 4.0_f32.ln()
    }
    
    /// 수렴 기여도 계산
    fn compute_convergence_contribution(&self) -> f32 {
        // 초기 상태에서는 0.0 반환 (아직 전이가 없음)
        if self.efficiency_tracker.total_attempts == 0 {
            return 0.0;
        }
        
        // 전이 효율성과 다양성의 균형
        let efficiency_component = self.efficiency_tracker.current_efficiency;
        let diversity_component = self.compute_state_diversity();
        
        // 최적 균형: 높은 효율성 + 적당한 다양성
        let balance_score = efficiency_component * (1.0 - (diversity_component - 0.7).abs());
        
        balance_score.clamp(0.0, 1.0)
    }
    
    /// 효율성 반환
    pub fn get_efficiency(&self) -> f32 {
        self.efficiency_tracker.current_efficiency
    }
    
    /// 메트릭 수집
    pub fn get_metrics(&self) -> StateTransitionMetrics {
        StateTransitionMetrics {
            efficiency: self.efficiency_tracker.current_efficiency,
            transition_frequency: if !self.transition_stats.transition_times.is_empty() && self.transition_stats.avg_transition_time > 0.0 {
                1_000_000_000.0 / self.transition_stats.avg_transition_time // Hz
            } else {
                0.0
            },
            state_diversity: self.compute_state_diversity(),
            convergence_contribution: self.compute_convergence_contribution(),
        }
    }
    
    /// 통계 리셋
    pub fn reset_stats(&mut self) {
        self.efficiency_tracker.successful_transitions = 0;
        self.efficiency_tracker.total_attempts = 0;
        self.efficiency_tracker.efficiency_history.clear();
        self.efficiency_tracker.current_efficiency = 1.0;
        
        self.transition_stats.function_transitions.clear();
        self.transition_stats.phase_transitions.clear();
        self.transition_stats.transition_times.clear();
        self.transition_stats.avg_transition_time = 0.0;
    }
    
    /// 규칙 테이블 내보내기 (디버깅용)
    pub fn export_rules(&self) -> String {
        format!(
            "Function Thresholds: {:?}\nPhase Thresholds: {:?}\nHybrid Weights: {:?}",
            self.transition_rules.function_thresholds,
            self.transition_rules.phase_thresholds,
            self.transition_rules.hybrid_weights
        )
    }
} 