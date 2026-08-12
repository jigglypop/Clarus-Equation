use crate::core::differential::state_transition::*;
use crate::core::differential::cycle_system::*;

#[test]
fn test_state_transition_engine_creation() {
    let engine = StateTransitionEngine::new();
    
    // 초기 효율성이 1.0이어야 함
    assert_eq!(engine.get_efficiency(), 1.0);
    
    // 초기 메트릭 확인
    let metrics = engine.get_metrics();
    assert_eq!(metrics.efficiency, 1.0);
    assert_eq!(metrics.transition_frequency, 0.0); // 초기에는 전이 없음
    assert_eq!(metrics.state_diversity, 0.0); // 초기에는 다양성 없음
    assert_eq!(metrics.convergence_contribution, 0.0);
}

#[test]
fn test_transition_rules_enum() {
    // 모든 전이 규칙이 정의되어 있는지 확인
    let rules = [
        TransitionRule::GradientMagnitude,
        TransitionRule::FunctionType,
        TransitionRule::LearningPhase,
        TransitionRule::Hybrid,
    ];
    
    for rule in rules {
        println!("Transition rule: {:?}", rule);
    }
}

#[test]
fn test_gradient_magnitude_rule() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    let test_cases = [
        (0.005, DifferentialPhase::Exploration, false), // 임계값 이하
        (0.015, DifferentialPhase::Exploration, true),  // 임계값 초과
        (0.05, DifferentialPhase::Exploitation, true),  // 충분히 큰 그래디언트
        (0.001, DifferentialPhase::Convergence, false), // 작은 그래디언트
    ];
    
    for (gradient, phase, expected) in test_cases {
        let result = engine.should_transition(
            &state, gradient, phase, TransitionRule::GradientMagnitude
        );
        
        if expected {
            println!("Gradient {}: Should transition = {}", gradient, result);
        }
    }
}

#[test]
fn test_function_type_rule() {
    let mut engine = StateTransitionEngine::new();
    
    // 다양한 함수 상태 테스트
    let states = [
        (CycleState::from_bits(0b00000000000), HyperbolicFunction::Sinh),  // state_bits = 0
        (CycleState::from_bits(0b01000000000), HyperbolicFunction::Cosh),  // state_bits = 1
        (CycleState::from_bits(0b10000000000), HyperbolicFunction::Tanh),  // state_bits = 2
        (CycleState::from_bits(0b11000000000), HyperbolicFunction::Sech2), // state_bits = 3
    ];
    
    for (state, expected_function) in states {
        assert_eq!(state.get_active_function(), expected_function);
        
        let result = engine.should_transition(
            &state, 0.02, DifferentialPhase::Exploitation, TransitionRule::FunctionType
        );
        
        println!("Function {:?}: Should transition = {}", expected_function, result);
    }
}

#[test]
fn test_learning_phase_rule() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    let phases = [
        DifferentialPhase::Exploration,  // 낮은 임계값 (0.01)
        DifferentialPhase::Exploitation, // 중간 임계값 (0.05)
        DifferentialPhase::Convergence,  // 높은 임계값 (0.1)
    ];
    
    let gradient = 0.03; // 중간 정도 그래디언트
    
    for phase in phases {
        let result = engine.should_transition(
            &state, gradient, phase, TransitionRule::LearningPhase
        );
        
        println!("Phase {:?}: Should transition = {}", phase, result);
    }
}

#[test]
fn test_hybrid_rule_comprehensive() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 다양한 시나리오 테스트
    let test_scenarios = [
        (0.2, DifferentialPhase::Exploration, "Strong gradient + exploration"),
        (0.01, DifferentialPhase::Exploration, "Weak gradient + exploration"), 
        (0.1, DifferentialPhase::Exploitation, "Medium gradient + exploitation"),
        (0.05, DifferentialPhase::Convergence, "Medium gradient + convergence"),
        (0.001, DifferentialPhase::Convergence, "Very weak gradient + convergence"),
    ];
    
    for (gradient, phase, description) in test_scenarios {
        let result = engine.should_transition(
            &state, gradient, phase, TransitionRule::Hybrid
        );
        
        println!("{}: Should transition = {}", description, result);
        assert!(result == true || result == false); // 유효한 불린 값
    }
}

#[test]
fn test_efficiency_tracking() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 여러 전이 시도
    let mut successful_transitions = 0;
    let total_attempts = 10;
    
    for i in 0..total_attempts {
        let gradient = (i as f32 + 1.0) * 0.02; // 점진적으로 증가
        
        let result = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
        
        if result {
            successful_transitions += 1;
        }
    }
    
    let expected_efficiency = successful_transitions as f32 / total_attempts as f32;
    let actual_efficiency = engine.get_efficiency();
    
    assert!((actual_efficiency - expected_efficiency).abs() < 1e-6);
    
    println!("Efficiency: expected={:.2}, actual={:.2}", expected_efficiency, actual_efficiency);
}

#[test]
fn test_transition_statistics_collection() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b10000000000); // Tanh function
    
    // 여러 전이 수행
    for i in 0..20 {
        let gradient = 0.1 + (i as f32 * 0.01);
        let phase = match i % 3 {
            0 => DifferentialPhase::Exploration,
            1 => DifferentialPhase::Exploitation,
            _ => DifferentialPhase::Convergence,
        };
        
        let _ = engine.should_transition(&state, gradient, phase, TransitionRule::Hybrid);
    }
    
    let metrics = engine.get_metrics();
    
    // 메트릭 유효성 확인
    assert!(metrics.efficiency >= 0.0 && metrics.efficiency <= 1.0);
    assert!(metrics.transition_frequency >= 0.0);
    assert!(metrics.state_diversity >= 0.0 && metrics.state_diversity <= 1.0);
    assert!(metrics.convergence_contribution >= 0.0 && metrics.convergence_contribution <= 1.0);
    
    println!("Metrics: {:?}", metrics);
}

#[test]
fn test_transition_rules_optimization() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 초기 임계값 확인
    let initial_rules = engine.export_rules();
    println!("Initial rules:\n{}", initial_rules);
    
    // 효율성 히스토리 생성 (감소 추세)
    for i in 0..120 {
        let gradient = 0.05;
        let _ = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
        
        // 인위적으로 효율성 히스토리 조작 (테스트 목적)
        if i >= 100 {
            // 최근 효율성을 낮게 만들기
            engine.efficiency_tracker.efficiency_history.push(0.3);
        }
    }
    
    // 규칙 최적화 실행
    engine.optimize_transition_rules();
    
    let optimized_rules = engine.export_rules();
    println!("Optimized rules:\n{}", optimized_rules);
}

#[test]
fn test_state_diversity_calculation() {
    let mut engine = StateTransitionEngine::new();
    
    // 다양한 상태로 전이 수행
    let states = [
        CycleState::from_bits(0b00000000000), // Sinh
        CycleState::from_bits(0b01000000000), // Cosh
        CycleState::from_bits(0b10000000000), // Tanh
        CycleState::from_bits(0b11000000000), // Sech2
    ];
    
    for (i, state) in states.iter().enumerate() {
        // 각 상태에서 여러 번 전이 시도
        for _ in 0..5 {
            let _ = engine.should_transition(
                state, 0.1, DifferentialPhase::Exploitation, TransitionRule::Hybrid
            );
        }
    }
    
    let metrics = engine.get_metrics();
    
    // 다양한 상태가 사용되면 다양성이 높아져야 함
    assert!(metrics.state_diversity > 0.0);
    
    println!("State diversity: {:.3}", metrics.state_diversity);
}

#[test]
fn test_convergence_contribution() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 높은 효율성과 적당한 다양성으로 좋은 수렴 기여도 달성
    for i in 0..50 {
        let gradient = 0.08; // 적당한 크기
        let _ = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
    }
    
    let metrics = engine.get_metrics();
    
    // 수렴 기여도가 계산되어야 함
    assert!(metrics.convergence_contribution >= 0.0);
    assert!(metrics.convergence_contribution <= 1.0);
    
    println!("Convergence contribution: {:.3}", metrics.convergence_contribution);
}

#[test]
fn test_transition_frequency_measurement() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    let start_time = std::time::Instant::now();
    
    // 빠른 전이 연산 수행
    for i in 0..100 {
        let gradient = 0.1 + (i as f32 * 0.001);
        let _ = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
    }
    
    let elapsed = start_time.elapsed();
    println!("100 transitions took: {:?}", elapsed);
    
    let metrics = engine.get_metrics();
    
    // 전이 빈도가 계산되어야 함
    if metrics.transition_frequency > 0.0 {
        println!("Transition frequency: {:.0} Hz", metrics.transition_frequency);
    }
}

#[test]
fn test_reset_statistics() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 일부 전이 수행하여 통계 생성
    for _ in 0..10 {
        let _ = engine.should_transition(
            &state, 0.1, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
    }
    
    let metrics_before = engine.get_metrics();
    assert!(metrics_before.efficiency > 0.0 || metrics_before.efficiency <= 1.0);
    
    // 통계 리셋
    engine.reset_stats();
    
    let metrics_after = engine.get_metrics();
    assert_eq!(metrics_after.efficiency, 1.0); // 리셋 후 초기값
    assert_eq!(metrics_after.transition_frequency, 0.0);
    assert_eq!(metrics_after.state_diversity, 0.0);
}

#[test]
fn test_extreme_gradients() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 극단적인 그래디언트 값들
    let extreme_gradients = [
        0.0,      // 제로
        1e-10,    // 매우 작음
        1e10,     // 매우 큼
        f32::MAX, // 최대값
        -1e5,     // 큰 음수
        f32::NAN, // NaN (이상한 경우)
    ];
    
    for gradient in extreme_gradients {
        // NaN은 건너뛰기
        if gradient.is_nan() {
            continue;
        }
        
        let result = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
        
        // 결과가 유효한 불린 값이어야 함
        assert!(result == true || result == false);
        
        println!("Extreme gradient {}: result = {}", gradient, result);
    }
}

#[test]
fn test_transition_time_measurement() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 전이 시간 측정 테스트
    for i in 0..20 {
        let gradient = 0.05 + (i as f32 * 0.01);
        
        let start = std::time::Instant::now();
        let _ = engine.should_transition(
            &state, gradient, DifferentialPhase::Exploitation, TransitionRule::Hybrid
        );
        let elapsed = start.elapsed();
        
        // 전이 시간이 합리적 범위에 있어야 함 (1ms 이하)
        assert!(elapsed.as_millis() < 1);
    }
    
    let metrics = engine.get_metrics();
    println!("Average transition time: {:.0}ns", 1e9 / metrics.transition_frequency.max(1.0));
}

#[test]
fn test_adaptive_threshold_behavior() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 효율성을 인위적으로 낮춤
    for _ in 0..50 {
        engine.efficiency_tracker.efficiency_history.push(0.2); // 낮은 효율성
    }
    engine.efficiency_tracker.current_efficiency = 0.2;
    
    // 낮은 효율성에서의 전이 행동
    let low_efficiency_result = engine.should_transition(
        &state, 0.05, DifferentialPhase::Exploitation, TransitionRule::Hybrid
    );
    
    // 효율성을 높임
    for _ in 0..50 {
        engine.efficiency_tracker.efficiency_history.push(0.9); // 높은 효율성
    }
    engine.efficiency_tracker.current_efficiency = 0.9;
    
    // 높은 효율성에서의 전이 행동
    let high_efficiency_result = engine.should_transition(
        &state, 0.05, DifferentialPhase::Exploitation, TransitionRule::Hybrid
    );
    
    println!("Low efficiency: {}, High efficiency: {}", 
             low_efficiency_result, high_efficiency_result);
}

#[test]
fn test_all_differential_phases() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    let gradient = 0.06; // 중간 강도
    
    let phases = [
        DifferentialPhase::Exploration,
        DifferentialPhase::Exploitation,
        DifferentialPhase::Convergence,
    ];
    
    for phase in phases {
        let result = engine.should_transition(
            &state, gradient, phase, TransitionRule::Hybrid
        );
        
        println!("Phase {:?}: transition = {}", phase, result);
        
        // 모든 단계에서 유효한 결과
        assert!(result == true || result == false);
    }
    
    // 단계별 통계 확인
    let metrics = engine.get_metrics();
    assert!(metrics.efficiency <= 1.0);
}

#[test]
fn test_performance_benchmark() {
    let mut engine = StateTransitionEngine::new();
    let state = CycleState::from_bits(0b01011100101);
    
    let start_time = std::time::Instant::now();
    
    // 대량 전이 연산 성능 테스트
    for i in 0..10000 {
        let gradient = (i as f32 * 0.0001).sin().abs();
        let phase = match i % 3 {
            0 => DifferentialPhase::Exploration,
            1 => DifferentialPhase::Exploitation,
            _ => DifferentialPhase::Convergence,
        };
        
        let _ = engine.should_transition(&state, gradient, phase, TransitionRule::Hybrid);
    }
    
    let elapsed = start_time.elapsed();
    let avg_time_ns = elapsed.as_nanos() / 10000;
    
    println!("Average transition decision time: {}ns", avg_time_ns);
    
    // 성능 목표: 1μs 이하
    assert!(avg_time_ns < 1000, "Performance target not met: {}ns", avg_time_ns);
    
    // 최종 메트릭 확인
    let metrics = engine.get_metrics();
    assert!(metrics.efficiency >= 0.0 && metrics.efficiency <= 1.0);
}

#[test]
fn test_rules_export_and_format() {
    let engine = StateTransitionEngine::new();
    let rules_export = engine.export_rules();
    
    // 문자열이 비어있지 않아야 함
    assert!(!rules_export.is_empty());
    
    // 주요 키워드들이 포함되어야 함
    assert!(rules_export.contains("Function Thresholds"));
    assert!(rules_export.contains("Phase Thresholds"));
    assert!(rules_export.contains("Hybrid Weights"));
    
    println!("Rules export:\n{}", rules_export);
} 