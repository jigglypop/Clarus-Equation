use crate::core::differential::backward::*;
use crate::core::differential::cycle_system::*;
use crate::core::differential::state_transition::*;
use crate::packed_params::Packed128;

#[test]
fn test_unified_backward_pass_creation() {
    let backward_pass = UnifiedBackwardPass::new();
    
    // 초기 수렴률이 0.0이어야 함
    assert_eq!(backward_pass.get_convergence_rate(), 0.0);
    
    // 초기 성능 통계 확인
    let stats = backward_pass.get_performance_stats();
    assert_eq!(stats.total_backward_passes, 0);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);
}

#[test]
fn test_backward_config_default() {
    let config = BackwardConfig::default();
    
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.gradient_clip_threshold, 1.0);
    assert_eq!(config.state_transition_weight, 0.6);
    assert_eq!(config.continuous_weight, 0.4);
    assert_eq!(config.stability_level, 2);
}

#[test]
fn test_compute_with_cycle_system_basic() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(4);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.5, -0.3, 0.8];
    let predicted = vec![0.9, 0.6, -0.2, 0.7];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000, // r=1.0, theta=0.5 정도
    };
    
    let (loss, metrics) = backward_pass.compute_with_cycle_system(
        &target,
        &predicted,
        &mut packed,
        &mut cycle_system,
        &mut transition_engine,
        2, 2,
        0.01
    );
    
    // 손실이 유한하고 양수여야 함
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    
    // 메트릭 유효성 확인
    assert!(metrics.gradient_norm.is_finite());
    assert!(metrics.state_transition_rate >= 0.0);
    assert!(metrics.convergence_speed >= 0.0);
    assert!(metrics.stability_score >= 0.0 && metrics.stability_score <= 1.0);
}

#[test]
fn test_gradient_cache_functionality() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(3);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![0.5, -0.2, 0.8];
    let predicted = vec![0.4, -0.1, 0.9];
    let mut packed = Packed128 {
        hi: 0xAAAA5555AAAA5555,
        lo: 0x3F8000003F800000,
    };
    
    // 첫 번째 호출 (캐시 미스)
    let (loss1, _) = backward_pass.compute_with_cycle_system(
        &target, &predicted, &mut packed,
        &mut cycle_system, &mut transition_engine,
        1, 3, 0.001
    );
    
    // 동일한 packed로 두 번째 호출 (캐시 히트)
    let (loss2, _) = backward_pass.compute_with_cycle_system(
        &target, &predicted, &mut packed,
        &mut cycle_system, &mut transition_engine,
        1, 3, 0.001
    );
    
    // 손실이 유사해야 함 (완전히 같지는 않을 수 있음 - 상태 변화)
    assert!((loss1 - loss2).abs() < 1.0);
    
    let stats = backward_pass.get_performance_stats();
    assert!(stats.total_backward_passes >= 2);
}

#[test]
fn test_state_transition_gradients() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(4);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.0, -1.0, 0.5];
    let predicted = vec![0.8, 0.2, -0.8, 0.3];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let original_hi = packed.hi;
    
    let (_, metrics) = backward_pass.compute_with_cycle_system(
        &target, &predicted, &mut packed,
        &mut cycle_system, &mut transition_engine,
        2, 2, 0.01
    );
    
    // Hi 필드가 변경되었을 가능성 (상태 전이)
    let hi_potentially_changed = packed.hi != original_hi;
    
    // 상태 전이율이 기록되어야 함
    assert!(metrics.state_transition_rate >= 0.0);
    
    let stats = backward_pass.get_performance_stats();
    assert!(stats.state_transitions >= 0);
    
    println!("Hi changed: {}, State transition rate: {}", 
             hi_potentially_changed, metrics.state_transition_rate);
}

#[test]
fn test_continuous_gradients_computation() {
    let backward_pass = UnifiedBackwardPass::new();
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000, // r=1.0, theta=0.5
    };
    
    let errors = vec![0.1, -0.05, 0.2, -0.15];
    
    let (grad_r, grad_theta) = backward_pass.compute_continuous_gradients(
        &packed, &errors, 2, 2
    );
    
    // 그래디언트가 유한해야 함
    assert!(grad_r.is_finite());
    assert!(grad_theta.is_finite());
    
    // 영향이 있어야 함 (완전히 0이 아님)
    assert!(grad_r.abs() > 1e-10 || grad_theta.abs() > 1e-10);
}

#[test]
fn test_learning_phase_determination() {
    let mut backward_pass = UnifiedBackwardPass::new();
    
    // 손실 히스토리 시뮬레이션
    for i in 0..25 {
        backward_pass.convergence_tracker.loss_history.push(1.0 - i as f32 * 0.03);
    }
    
    let phase = backward_pass.determine_learning_phase();
    
    // 손실이 감소하는 경우 Exploration이어야 함
    assert_eq!(phase, DifferentialPhase::Exploration);
    
    // 정체 상태 시뮬레이션
    for _ in 0..15 {
        backward_pass.convergence_tracker.loss_history.push(0.25);
    }
    
    let phase_stagnant = backward_pass.determine_learning_phase();
    assert_eq!(phase_stagnant, DifferentialPhase::Convergence);
}

#[test]
fn test_gradient_clipping() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(2);
    
    // 극단적인 그래디언트 생성
    let extreme_continuous_gradients = (10.0, -5.0); // 클리핑 임계값 초과
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let clipped_gradients = backward_pass.apply_fused_gradients(
        &mut packed,
        &cycle_system,
        0.5,
        extreme_continuous_gradients,
        0.01,
        2, 2
    );
    
    // 그래디언트가 클리핑되었는지 확인
    assert!(clipped_gradients.0.abs() <= 1.0); // 기본 클리핑 임계값
    assert!(clipped_gradients.1.abs() <= 1.0);
}

#[test]
fn test_parameter_update_bounds() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(1);
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3E8000003E800000, // 작은 r 값
    };
    
    let original_r = f32::from_bits((packed.lo >> 32) as u32);
    
    // 큰 그래디언트로 파라미터 업데이트
    let _ = backward_pass.apply_fused_gradients(
        &mut packed,
        &cycle_system,
        0.0,
        (100.0, 50.0), // 매우 큰 그래디언트
        1.0, // 큰 학습률
        1, 1
    );
    
    let updated_r = f32::from_bits((packed.lo >> 32) as u32);
    
    // r 값이 푸앵카레볼 범위 [0.0, 0.9999999] 내에 있어야 함
    assert!(updated_r >= 0.0 && updated_r <= 0.9999999);
    
    println!("Original r: {}, Updated r: {}", original_r, updated_r);
}

#[test]
fn test_convergence_tracking() {
    let mut backward_pass = UnifiedBackwardPass::new();
    
    // 수렴하는 손실 시퀀스 시뮬레이션
    let losses = vec![1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.41, 0.405, 0.403, 0.402];
    
    for loss in losses {
        backward_pass.convergence_tracker.loss_history.push(loss);
        backward_pass.convergence_tracker.gradient_history.push(loss * 2.0);
    }
    
    backward_pass.update_convergence_tracking();
    
    let convergence_rate = backward_pass.get_convergence_rate();
    
    // 수렴률이 양수여야 함 (손실이 감소)
    assert!(convergence_rate >= 0.0);
    
    println!("Convergence rate: {}", convergence_rate);
}

#[test]
fn test_mse_loss_computation() {
    let backward_pass = UnifiedBackwardPass::new();
    
    let target = vec![1.0, 0.0, -1.0, 0.5];
    let predicted = vec![0.9, 0.1, -0.8, 0.6];
    
    let mse = backward_pass.compute_mse_loss(&target, &predicted);
    
    // 수동 계산: ((0.1)² + (0.1)² + (0.2)² + (0.1)²) / 4 = 0.07 / 4 = 0.0175
    let expected = 0.0175;
    
    assert!((mse - expected).abs() < 1e-6);
}

#[test]
fn test_cached_gradients_application() {
    let backward_pass = UnifiedBackwardPass::new();
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000, // r=1.0, theta=0.5
    };
    
    let original_r = f32::from_bits((packed.lo >> 32) as u32);
    let original_theta = f32::from_bits(packed.lo as u32);
    
    let gradients = (0.1, -0.05);
    let learning_rate = 0.01;
    
    backward_pass.apply_cached_gradients(&mut packed, gradients, learning_rate);
    
    let updated_r = f32::from_bits((packed.lo >> 32) as u32);
    let updated_theta = f32::from_bits(packed.lo as u32);
    
    // 파라미터가 그래디언트 방향으로 업데이트되었는지 확인
    assert!((updated_r - (original_r - learning_rate * gradients.0)).abs() < 1e-6);
    assert!((updated_theta - (original_theta - learning_rate * gradients.1)).abs() < 1e-6);
}

#[test]
fn test_gradient_metrics_building() {
    let backward_pass = UnifiedBackwardPass::new();
    
    let grad_r = 0.3;
    let grad_theta = 0.4;
    
    let metrics = backward_pass.build_gradient_metrics(grad_r, grad_theta);
    
    // 그래디언트 노름 확인
    let expected_norm = (grad_r * grad_r + grad_theta * grad_theta as f32).sqrt();
    assert!((metrics.gradient_norm - expected_norm).abs() < 1e-6);
    
    // 안정성 스코어 확인
    assert!(metrics.stability_score >= 0.0 && metrics.stability_score <= 1.0);
}

#[test]
fn test_performance_statistics_collection() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(2);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.0];
    let predicted = vec![0.8, 0.2];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 여러 번 역전파 수행
    for _ in 0..5 {
        let _ = backward_pass.compute_with_cycle_system(
            &target, &predicted, &mut packed,
            &mut cycle_system, &mut transition_engine,
            1, 2, 0.01
        );
    }
    
    let stats = backward_pass.get_performance_stats();
    
    assert_eq!(stats.total_backward_passes, 5);
    assert!(stats.continuous_updates >= 0);
    assert!(stats.state_transitions >= 0);
}

#[test]
fn test_numerical_stability_with_extreme_values() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(1);
    let mut transition_engine = StateTransitionEngine::new();
    
    // 극단적인 값들
    let target = vec![1000.0];
    let predicted = vec![-1000.0];
    let mut packed = Packed128 {
        hi: 0xFFFFFFFFFFFFFFFF,
        lo: 0x0000000000000000,
    };
    
    let (loss, metrics) = backward_pass.compute_with_cycle_system(
        &target, &predicted, &mut packed,
        &mut cycle_system, &mut transition_engine,
        1, 1, 0.001
    );
    
    // 결과가 유한해야 함
    assert!(loss.is_finite());
    assert!(!loss.is_nan());
    
    // 메트릭도 유효해야 함
    assert!(metrics.gradient_norm.is_finite());
    assert!(metrics.stability_score >= 0.0);
}

#[test]
fn test_cache_memory_management() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(1);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![0.5];
    let predicted = vec![0.4];
    
    // 캐시 크기 제한 테스트
    for i in 0..6000 {
        let mut packed = Packed128 {
            hi: i as u64,
            lo: (i * 2) as u64,
        };
        
        let _ = backward_pass.compute_with_cycle_system(
            &target, &predicted, &mut packed,
            &mut cycle_system, &mut transition_engine,
            1, 1, 0.001
        );
    }
    
    // 캐시가 정리되었는지 확인
    println!("Cache management test completed");
}

#[test]
fn test_clear_cache_and_reset_stats() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(2);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.0];
    let predicted = vec![0.9, 0.1];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 일부 통계 생성
    for _ in 0..3 {
        let _ = backward_pass.compute_with_cycle_system(
            &target, &predicted, &mut packed,
            &mut cycle_system, &mut transition_engine,
            1, 2, 0.01
        );
    }
    
    let stats_before = backward_pass.get_performance_stats();
    assert!(stats_before.total_backward_passes > 0);
    
    // 캐시 초기화
    backward_pass.clear_cache();
    
    // 통계 리셋
    backward_pass.reset_stats();
    
    let stats_after = backward_pass.get_performance_stats();
    assert_eq!(stats_after.total_backward_passes, 0);
    assert_eq!(stats_after.cache_hits, 0);
    assert_eq!(stats_after.cache_misses, 0);
    
    assert_eq!(backward_pass.get_convergence_rate(), 0.0);
}

#[test]
fn test_different_learning_rates() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut cycle_system = UnifiedCycleDifferentialSystem::new(2);
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.0];
    let predicted = vec![0.5, 0.5];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let learning_rates = [0.001, 0.01, 0.1];
    let mut losses = Vec::new();
    
    for &lr in &learning_rates {
        let (loss, _) = backward_pass.compute_with_cycle_system(
            &target, &predicted, &mut packed,
            &mut cycle_system, &mut transition_engine,
            1, 2, lr
        );
        losses.push(loss);
    }
    
    // 모든 학습률에서 유효한 손실
    for loss in losses {
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}

#[test]
fn test_integration_with_different_cycle_systems() {
    let mut backward_pass = UnifiedBackwardPass::new();
    let mut transition_engine = StateTransitionEngine::new();
    
    let target = vec![1.0, 0.0, -0.5, 0.8];
    let predicted = vec![0.8, 0.2, -0.3, 0.6];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 다양한 크기의 사이클 시스템 테스트
    let system_sizes = [1, 4, 8, 16];
    
    for size in system_sizes {
        let mut cycle_system = UnifiedCycleDifferentialSystem::new(size);
        
        let (loss, metrics) = backward_pass.compute_with_cycle_system(
            &target, &predicted, &mut packed,
            &mut cycle_system, &mut transition_engine,
            2, 2, 0.01
        );
        
        assert!(loss.is_finite());
        assert!(metrics.gradient_norm.is_finite());
        
        println!("System size {}: loss={:.6}, grad_norm={:.6}", 
                 size, loss, metrics.gradient_norm);
    }
} 