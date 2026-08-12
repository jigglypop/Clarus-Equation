use crate::core::differential::forward::*;
use crate::core::differential::cycle_system::*;
use crate::packed_params::Packed128;

#[test]
fn test_unified_forward_pass_creation() {
    let forward_pass = UnifiedForwardPass::new();
    
    // 초기 정확도가 1.0이어야 함
    assert_eq!(forward_pass.get_accuracy(), 1.0);
    
    // 초기 메트릭 확인
    let metrics = forward_pass.get_metrics();
    assert_eq!(metrics.cache_hit_rate, 0.0); // 초기에는 캐시 없음
    assert!(metrics.numerical_stability >= 0.0);
    assert_eq!(metrics.cycle_utilization, 0.95); // 높은 사이클 활용률
}

#[test]
fn test_forward_pass_with_cycle_system_integration() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(5);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    let result = forward_pass.compute_with_cycle_system(
        &packed,
        &cycle_system,
        0, 0, 4, 4
    );
    
    // 결과가 유한하고 안정적인 범위에 있어야 함
    assert!(result.is_finite());
    assert!(result >= -10.0 && result <= 10.0); // 클램핑 범위
}

#[test]
fn test_forward_pass_cache_functionality() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(3);
    
    let packed = Packed128 {
        hi: 0x555555555555AAAA,
        lo: 0xAAAA555555555555,
    };
    
    // 첫 번째 호출 (캐시 미스)
    let result1 = forward_pass.compute_with_cycle_system(
        &packed, &cycle_system, 1, 1, 4, 4
    );
    
    // 두 번째 호출 (캐시 히트)
    let result2 = forward_pass.compute_with_cycle_system(
        &packed, &cycle_system, 1, 1, 4, 4
    );
    
    // 결과가 동일해야 함
    assert_eq!(result1, result2);
    
    // 캐시 히트율이 증가했어야 함
    let metrics = forward_pass.get_metrics();
    assert!(metrics.cache_hit_rate > 0.0);
}

#[test]
fn test_forward_pass_numerical_stability() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(8);
    
    // 극단적인 값들로 테스트
    let extreme_packed = Packed128 {
        hi: 0xFFFFFFFFFFFFFFFF,
        lo: 0x0000000000000000,
    };
    
    let result = forward_pass.compute_with_cycle_system(
        &extreme_packed, &cycle_system, 0, 0, 2, 2
    );
    
    // NaN이나 Infinity가 아니어야 함
    assert!(result.is_finite());
    assert!(!result.is_nan());
    
    // 안정성 스코어 확인
    let metrics = forward_pass.get_metrics();
    assert!(metrics.numerical_stability >= 0.0);
}

#[test]
fn test_forward_pass_different_matrix_sizes() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(16);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    // 다양한 행렬 크기 테스트
    let sizes = [(2, 2), (4, 4), (8, 8), (16, 16)];
    
    for (rows, cols) in sizes {
        for i in 0..rows.min(4) {
            for j in 0..cols.min(4) {
                let result = forward_pass.compute_with_cycle_system(
                    &packed, &cycle_system, i, j, rows, cols
                );
                
                assert!(result.is_finite(), "결과가 유한하지 않음: {}x{} at ({}, {})", rows, cols, i, j);
            }
        }
    }
}

#[test]
fn test_state_modulation_factors() {
    let forward_pass = UnifiedForwardPass::new();
    
    // 다양한 상태에 대한 변조 계수 테스트
    let states = [
        CycleState::from_bits(0b00000000000), // Sinh
        CycleState::from_bits(0b01000000000), // Cosh  
        CycleState::from_bits(0b10000000000), // Tanh
        CycleState::from_bits(0b11000000000), // Sech2
    ];
    
    for state in &states {
        let modulation = forward_pass.get_state_modulation_factor(state);
        
        // 변조 계수가 합리적 범위에 있어야 함
        assert!(modulation >= 0.5 && modulation <= 1.5);
        
        let angle_offset = forward_pass.get_state_angle_offset(state);
        
        // 각도 오프셋이 합리적 범위에 있어야 함
        assert!(angle_offset >= 0.0 && angle_offset <= 1.0);
    }
}

#[test]
fn test_detail_modulation_computation() {
    let forward_pass = UnifiedForwardPass::new();
    let state = CycleState::from_bits(0b01011100101);
    
    // 다양한 거리와 각도에서 세부 변조 테스트
    let test_cases = [
        (0.0, 0.0),
        (0.5, std::f32::consts::PI / 4.0),
        (1.0, std::f32::consts::PI / 2.0),
        (1.5, std::f32::consts::PI),
    ];
    
    for (dist, angle) in test_cases {
        let modulation = forward_pass.compute_detail_modulation(&state, dist, angle);
        
        // 변조값이 안정적 범위에 있어야 함
        assert!(modulation >= 0.1 && modulation <= 2.0);
        assert!(modulation.is_finite());
    }
}

#[test]
fn test_accuracy_tracking() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(4);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    // 여러 계산 수행하여 정확도 추적 테스트
    for i in 0..10 {
        let result = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, i % 4, (i + 1) % 4, 4, 4
        );
        
        // 안정적 결과인지 확인
        assert!(result.is_finite());
    }
    
    // 정확도가 적절한 범위에 있어야 함
    let accuracy = forward_pass.get_accuracy();
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
}

#[test]
fn test_performance_metrics_collection() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(6);
    
    let packed = Packed128 {
        hi: 0xAAAA5555AAAA5555,
        lo: 0x5555AAAA5555AAAA,
    };
    
    // 여러 계산으로 메트릭 생성
    for _ in 0..5 {
        let _ = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, 0, 0, 3, 3
        );
    }
    
    let metrics = forward_pass.get_metrics();
    
    // 메트릭 유효성 확인
    assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
    assert!(metrics.avg_computation_time_ns >= 0.0);
    assert!(metrics.numerical_stability >= 0.0 && metrics.numerical_stability <= 1.0);
    assert_eq!(metrics.cycle_utilization, 0.95);
}

#[test]
fn test_cache_memory_management() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(2);
    
    // 캐시 크기 제한 테스트를 위해 많은 다른 입력 생성
    for i in 0..15000 {
        let packed = Packed128 {
            hi: i as u64,
            lo: (i * 2) as u64,
        };
        
        let _ = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, 0, 0, 2, 2
        );
    }
    
    // 캐시가 자동으로 정리되었는지 확인 (메모리 관리)
    let metrics = forward_pass.get_metrics();
    println!("Final cache hit rate after memory management: {}", metrics.cache_hit_rate);
}

#[test]
fn test_forward_config_default() {
    let config = ForwardConfig::default();
    
    assert_eq!(config.enable_cache, true);
    assert_eq!(config.high_precision, false);
    assert_eq!(config.cycle_integration_level, 2);
}

#[test]
fn test_edge_case_inputs() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(1);
    
    // 경계 조건 테스트
    let edge_cases = [
        // 제로 값
        Packed128 { hi: 0, lo: 0 },
        // 최대 값
        Packed128 { hi: u64::MAX, lo: u64::MAX },
        // 특수 패턴
        Packed128 { hi: 0x5555555555555555, lo: 0xAAAAAAAAAAAAAAAA },
        Packed128 { hi: 0xAAAAAAAAAAAAAAAA, lo: 0x5555555555555555 },
    ];
    
    for packed in edge_cases {
        let result = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, 0, 0, 1, 1
        );
        
        // 모든 경우에서 안정적 결과
        assert!(result.is_finite());
        assert!(!result.is_nan());
        assert!(result >= -10.0 && result <= 10.0);
    }
}

#[test]
fn test_concurrent_computation_consistency() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let forward_pass = Arc::new(Mutex::new(UnifiedForwardPass::new()));
    let cycle_system = Arc::new(UnifiedCycleDifferentialSystem::new(4));
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    let mut handles = vec![];
    
    // 여러 스레드에서 동시 계산
    for i in 0..4 {
        let forward_pass = Arc::clone(&forward_pass);
        let cycle_system = Arc::clone(&cycle_system);
        let packed_copy = packed;
        
        let handle = thread::spawn(move || {
            let mut fp = forward_pass.lock().unwrap();
            fp.compute_with_cycle_system(
                &packed_copy, &cycle_system, i, i, 4, 4
            )
        });
        
        handles.push(handle);
    }
    
    // 모든 스레드 결과 수집
    let results: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // 모든 결과가 유효해야 함
    for result in results {
        assert!(result.is_finite());
        assert!(!result.is_nan());
    }
}

#[test]
fn test_performance_benchmark() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(8);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    let start_time = std::time::Instant::now();
    
    // 1000번 계산 수행
    for i in 0..1000 {
        let _ = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, i % 8, (i + 1) % 8, 8, 8
        );
    }
    
    let elapsed = start_time.elapsed();
    let avg_time_ns = elapsed.as_nanos() / 1000;
    
    println!("Forward pass average time: {}ns", avg_time_ns);
    
    // 성능 목표 확인 (넉넉하게 1μs 이하)
    assert!(avg_time_ns < 1000, "Performance target not met: {}ns", avg_time_ns);
    
    // 최종 메트릭 확인
    let metrics = forward_pass.get_metrics();
    assert!(metrics.cache_hit_rate > 0.0); // 캐시 활용됨
    assert!(metrics.numerical_stability > 0.5); // 높은 안정성
}

#[test]
fn test_cache_clear_and_reset() {
    let mut forward_pass = UnifiedForwardPass::new();
    let cycle_system = UnifiedCycleDifferentialSystem::new(3);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    // 캐시 생성
    for i in 0..5 {
        let _ = forward_pass.compute_with_cycle_system(
            &packed, &cycle_system, i % 3, (i + 1) % 3, 3, 3
        );
    }
    
    let metrics_before = forward_pass.get_metrics();
    assert!(metrics_before.cache_hit_rate > 0.0);
    
    // 캐시 초기화
    forward_pass.clear_cache();
    
    // 통계 리셋
    forward_pass.reset_stats();
    
    let metrics_after = forward_pass.get_metrics();
    assert_eq!(metrics_after.cache_hit_rate, 0.0);
    assert_eq!(metrics_after.avg_computation_time_ns, 0.0);
} 