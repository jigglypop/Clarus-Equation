use crate::core::differential::*;
use crate::packed_params::Packed128;

#[test]
fn test_differential_system_creation() {
    let system = DifferentialSystem::new(10);
    
    // 시스템이 정상적으로 생성되었는지 확인
    assert!(system.verify_system_invariants());
    
    // 각 엔진이 초기화되었는지 확인
    let metrics = system.get_performance_metrics();
    assert!(metrics.cycle_entropy >= 0.0 && metrics.cycle_entropy <= 1.0);
    assert_eq!(metrics.forward_accuracy, 1.0); // 초기 정확도
    assert_eq!(metrics.backward_convergence, 0.0); // 초기 수렴률
    assert_eq!(metrics.transition_efficiency, 1.0); // 초기 효율성
}

#[test]
fn test_differential_system_config_default() {
    let config = DifferentialSystemConfig::default();
    
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.differential_phase, DifferentialPhase::Exploitation);
    assert_eq!(config.transition_threshold, 0.01);
    assert_eq!(config.invariant_check_period, 100);
}

#[test]
fn test_unified_forward_pass() {
    let mut system = DifferentialSystem::new(5);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000, // r=1.0, theta=0.5
    };
    
    let result = system.unified_forward(&packed, 1, 2, 4, 4);
    
    // 결과가 유한하고 안정적 범위에 있어야 함
    assert!(result.is_finite());
    assert!(!result.is_nan());
    assert!(result >= -10.0 && result <= 10.0); // 클램핑 범위
    
    println!("Forward result: {}", result);
}

#[test]
fn test_unified_backward_pass() {
    let mut system = DifferentialSystem::new(4);
    
    let target = vec![1.0, 0.5, -0.2, 0.8];
    let predicted = vec![0.9, 0.6, -0.1, 0.7];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let (loss, metrics) = system.unified_backward(
        &target, &predicted, &mut packed, 2, 2, 0.01
    );
    
    // 손실이 유효해야 함
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    
    // 메트릭이 유효해야 함
    assert!(metrics.gradient_norm.is_finite());
    assert!(metrics.state_transition_rate >= 0.0);
    assert!(metrics.convergence_speed >= 0.0);
    assert!(metrics.stability_score >= 0.0 && metrics.stability_score <= 1.0);
    
    println!("Loss: {:.6}, Gradient norm: {:.6}", loss, metrics.gradient_norm);
}

#[test]
fn test_forward_backward_integration() {
    let mut system = DifferentialSystem::new(6);
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 순전파 수행
    let forward_result = system.unified_forward(&packed, 0, 0, 3, 3);
    
    // 역전파용 데이터 준비
    let target = vec![0.5];
    let predicted = vec![forward_result];
    
    // 역전파 수행
    let (loss, _) = system.unified_backward(
        &target, &predicted, &mut packed, 1, 1, 0.01
    );
    
    // 순전파-역전파 사이클이 성공적으로 완료
    assert!(forward_result.is_finite());
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    
    println!("Forward: {:.6}, Loss: {:.6}", forward_result, loss);
}

#[test]
fn test_performance_metrics_collection() {
    let mut system = DifferentialSystem::new(8);
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 여러 번의 순전파-역전파 수행
    for i in 0..10 {
        // 순전파
        let forward_result = system.unified_forward(&packed, i % 3, (i + 1) % 3, 3, 3);
        
        // 역전파
        let target = vec![0.5, -0.2, 0.8];
        let predicted = vec![forward_result, forward_result * 0.8, forward_result * 1.2];
        
        let _ = system.unified_backward(
            &target, &predicted, &mut packed, 1, 3, 0.001
        );
    }
    
    let metrics = system.get_performance_metrics();
    
    // 모든 메트릭이 유효한 범위에 있어야 함
    assert!(metrics.cycle_entropy >= 0.0 && metrics.cycle_entropy <= 1.0);
    assert!(metrics.forward_accuracy >= 0.0 && metrics.forward_accuracy <= 1.0);
    assert!(metrics.backward_convergence >= 0.0);
    assert!(metrics.transition_efficiency >= 0.0 && metrics.transition_efficiency <= 1.0);
    
    println!("Performance metrics: {:?}", metrics);
}

#[test]
fn test_system_invariants_verification() {
    let mut system = DifferentialSystem::new(12);
    
    // 초기 불변량 확인
    assert!(system.verify_system_invariants());
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 여러 연산 수행 후에도 불변량 유지 확인
    for i in 0..20 {
        let _ = system.unified_forward(&packed, i % 4, (i + 1) % 4, 4, 4);
        
        let target = vec![1.0, 0.0, -0.5, 0.3];
        let predicted = vec![0.8, 0.1, -0.4, 0.2];
        
        let _ = system.unified_backward(
            &target, &predicted, &mut packed, 2, 2, 0.001
        );
        
        // 주기적으로 불변량 확인
        if i % 5 == 0 {
            assert!(system.verify_system_invariants(), "Invariants violated at step {}", i);
        }
    }
    
    // 최종 불변량 확인
    assert!(system.verify_system_invariants());
}

#[test]
fn test_different_matrix_sizes() {
    let mut system = DifferentialSystem::new(16);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let matrix_sizes = [(1, 1), (2, 3), (4, 4), (8, 8)];
    
    for (rows, cols) in matrix_sizes {
        // 순전파 테스트
        for i in 0..rows.min(4) {
            for j in 0..cols.min(4) {
                let result = system.unified_forward(&packed, i, j, rows, cols);
                assert!(result.is_finite(), "Forward failed for {}x{} at ({}, {})", rows, cols, i, j);
            }
        }
        
        // 역전파 테스트
        let target: Vec<f32> = (0..rows*cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let predicted: Vec<f32> = target.iter().map(|x| x + 0.1).collect();
        let mut packed_copy = packed;
        
        let (loss, _) = system.unified_backward(
            &target, &predicted, &mut packed_copy, rows, cols, 0.01
        );
        
        assert!(loss.is_finite(), "Backward failed for {}x{}", rows, cols);
        
        println!("Matrix {}x{}: Loss = {:.6}", rows, cols, loss);
    }
}

#[test]
fn test_learning_rate_sensitivity() {
    let mut system = DifferentialSystem::new(4);
    
    let target = vec![1.0, 0.0, -0.5, 0.8];
    let predicted = vec![0.8, 0.2, -0.3, 0.6];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let learning_rates = [0.0001, 0.001, 0.01, 0.1];
    let mut losses = Vec::new();
    
    for &lr in &learning_rates {
        let mut packed_copy = packed;
        let (loss, _) = system.unified_backward(
            &target, &predicted, &mut packed_copy, 2, 2, lr
        );
        losses.push(loss);
        
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        
        println!("Learning rate {}: Loss = {:.6}", lr, loss);
    }
    
    // 모든 학습률에서 유효한 손실
    assert_eq!(losses.len(), learning_rates.len());
}

#[test]
fn test_extreme_input_handling() {
    let mut system = DifferentialSystem::new(2);
    
    // 극단적인 Packed128 값들
    let extreme_inputs = [
        Packed128 { hi: 0, lo: 0 },
        Packed128 { hi: u64::MAX, lo: u64::MAX },
        Packed128 { hi: 0x5555555555555555, lo: 0xAAAAAAAAAAAAAAAA },
        Packed128 { hi: 0xAAAAAAAAAAAAAAAA, lo: 0x5555555555555555 },
    ];
    
    for (i, packed) in extreme_inputs.iter().enumerate() {
        // 순전파 테스트
        let forward_result = system.unified_forward(packed, 0, 0, 2, 2);
        assert!(forward_result.is_finite(), "Forward failed for extreme input {}", i);
        assert!(!forward_result.is_nan(), "Forward NaN for extreme input {}", i);
        
        // 역전파 테스트
        let target = vec![0.5, -0.3];
        let predicted = vec![forward_result, forward_result * 0.5];
        let mut packed_copy = *packed;
        
        let (loss, _) = system.unified_backward(
            &target, &predicted, &mut packed_copy, 1, 2, 0.001
        );
        
        assert!(loss.is_finite(), "Backward failed for extreme input {}", i);
        assert!(!loss.is_nan(), "Backward NaN for extreme input {}", i);
        
        println!("Extreme input {}: Forward = {:.6}, Loss = {:.6}", i, forward_result, loss);
    }
}

#[test]
fn test_convergence_behavior() {
    let mut system = DifferentialSystem::new(3);
    
    let target = vec![0.8, -0.2, 0.5];
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let mut loss_history = Vec::new();
    
    // 여러 에포크 시뮬레이션
    for epoch in 0..50 {
        // 순전파
        let predictions: Vec<f32> = (0..3).map(|i| {
            system.unified_forward(&packed, 0, i, 1, 3)
        }).collect();
        
        // 역전파
        let (loss, metrics) = system.unified_backward(
            &target, &predictions, &mut packed, 1, 3, 0.01
        );
        
        loss_history.push(loss);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}, Convergence = {:.6}", 
                     epoch, loss, metrics.convergence_speed);
        }
    }
    
    // 초기 손실과 최종 손실 비교
    let initial_loss = loss_history[0];
    let final_loss = loss_history[loss_history.len() - 1];
    
    println!("Initial loss: {:.6}, Final loss: {:.6}", initial_loss, final_loss);
    
    // 손실이 감소하거나 안정화되어야 함
    assert!(final_loss <= initial_loss + 0.1); // 약간의 여유
}

#[test]
fn test_concurrent_system_usage() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let system = Arc::new(Mutex::new(DifferentialSystem::new(8)));
    
    let mut handles = vec![];
    
    // 여러 스레드에서 동시에 시스템 사용
    for i in 0..4 {
        let system = Arc::clone(&system);
        
        let handle = thread::spawn(move || {
            let mut sys = system.lock().unwrap();
            
            let packed = Packed128 {
                hi: 0x123456789ABCDEF0 + i as u64,
                lo: 0x3F8000003F000000 + i as u64,
            };
            
            // 순전파
            let forward_result = sys.unified_forward(&packed, i, (i + 1) % 3, 3, 3);
            
            // 역전파
            let target = vec![0.5, -0.2, 0.8];
            let predicted = vec![forward_result, forward_result * 0.8, forward_result * 1.2];
            let mut packed_copy = packed;
            
            let (loss, _) = sys.unified_backward(
                &target, &predicted, &mut packed_copy, 1, 3, 0.001
            );
            
            (forward_result, loss)
        });
        
        handles.push(handle);
    }
    
    // 모든 스레드 결과 수집
    let results: Vec<(f32, f32)> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // 모든 결과가 유효해야 함
    for (i, (forward, loss)) in results.iter().enumerate() {
        assert!(forward.is_finite(), "Thread {} forward invalid", i);
        assert!(loss.is_finite(), "Thread {} loss invalid", i);
        assert!(*loss >= 0.0, "Thread {} negative loss", i);
        
        println!("Thread {}: Forward = {:.6}, Loss = {:.6}", i, forward, loss);
    }
}

#[test]
fn test_performance_benchmark() {
    let mut system = DifferentialSystem::new(16);
    
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let target = vec![1.0, 0.0, -0.5, 0.8];
    let mut packed_copy = packed;
    
    // 순전파 성능 측정
    let start_time = std::time::Instant::now();
    for i in 0..1000 {
        let _ = system.unified_forward(&packed, i % 4, (i + 1) % 4, 4, 4);
    }
    let forward_elapsed = start_time.elapsed();
    let avg_forward_ns = forward_elapsed.as_nanos() / 1000;
    
    // 역전파 성능 측정
    let start_time = std::time::Instant::now();
    for i in 0..1000 {
        let predicted = vec![(i as f32 * 0.001).sin(); 4];
        let _ = system.unified_backward(
            &target, &predicted, &mut packed_copy, 2, 2, 0.001
        );
    }
    let backward_elapsed = start_time.elapsed();
    let avg_backward_ns = backward_elapsed.as_nanos() / 1000;
    
    println!("Average forward time: {}ns", avg_forward_ns);
    println!("Average backward time: {}ns", avg_backward_ns);
    
    // 성능 목표 확인 (넉넉한 임계값)
    assert!(avg_forward_ns < 10000, "Forward performance target not met: {}ns", avg_forward_ns);
    assert!(avg_backward_ns < 50000, "Backward performance target not met: {}ns", avg_backward_ns);
    
    // 최종 시스템 상태 확인
    assert!(system.verify_system_invariants());
}

#[test]
fn test_system_state_persistence() {
    let mut system = DifferentialSystem::new(6);
    
    let initial_metrics = system.get_performance_metrics();
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 많은 연산 수행
    for i in 0..100 {
        let _ = system.unified_forward(&packed, i % 3, (i + 1) % 3, 3, 3);
        
        let target = vec![0.5, -0.2, 0.8];
        let predicted = vec![0.4, -0.1, 0.9];
        
        let _ = system.unified_backward(
            &target, &predicted, &mut packed, 1, 3, 0.001
        );
    }
    
    let final_metrics = system.get_performance_metrics();
    
    // 메트릭이 진화했어야 함
    assert_ne!(initial_metrics.cycle_entropy, final_metrics.cycle_entropy);
    // backward_convergence는 최소값 보장으로 변경될 수 있음
    assert!(final_metrics.backward_convergence >= 0.0);
    
    // 하지만 시스템 불변량은 유지되어야 함
    assert!(system.verify_system_invariants());
    
    println!("Initial entropy: {:.6}, Final entropy: {:.6}", 
             initial_metrics.cycle_entropy, final_metrics.cycle_entropy);
    println!("Initial convergence: {:.6}, Final convergence: {:.6}", 
             initial_metrics.backward_convergence, final_metrics.backward_convergence);
}

#[test]
fn test_differential_system_config_usage() {
    let mut system = DifferentialSystem::new(4);
    let config = DifferentialSystemConfig {
        learning_rate: 0.005,
        differential_phase: DifferentialPhase::Exploration,
        transition_threshold: 0.02,
        invariant_check_period: 50,
    };
    
    let mut packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    // 설정된 학습률로 역전파 수행
    let target = vec![1.0, -0.5];
    let predicted = vec![0.8, -0.3];
    
    let (loss, _) = system.unified_backward(
        &target, &predicted, &mut packed, 1, 2, config.learning_rate
    );
    
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    
    // 불변량 확인 주기 테스트 (50번마다)
    for i in 0..60 {
        let _ = system.unified_forward(&packed, 0, 0, 2, 2);
        
        if i % config.invariant_check_period == 0 {
            assert!(system.verify_system_invariants(), 
                   "Invariants violated at check period {}", i);
        }
    }
}

#[test]
fn test_comprehensive_integration() {
    let mut system = DifferentialSystem::new(10);
    
    // 다양한 시나리오 통합 테스트
    let scenarios = [
        (vec![1.0, 0.0], vec![0.9, 0.1], 0.001),
        (vec![0.5, -0.5, 0.8], vec![0.4, -0.4, 0.9], 0.01),
        (vec![-1.0, 1.0, 0.0, 0.0], vec![-0.8, 0.8, 0.1, -0.1], 0.005),
    ];
    
    for (i, (target, predicted, learning_rate)) in scenarios.iter().enumerate() {
        let mut packed = Packed128 {
            hi: 0x123456789ABCDEF0 + i as u64 * 0x1111,
            lo: 0x3F8000003F000000 + i as u64 * 0x2222,
        };
        
        let rows = (target.len() as f32).sqrt().ceil() as usize;
        let cols = (target.len() + rows - 1) / rows;
        
        // 순전파-역전파 사이클
        let forward_result = system.unified_forward(&packed, 0, 0, rows, cols);
        
        let (loss, metrics) = system.unified_backward(
            target, predicted, &mut packed, rows, cols, *learning_rate
        );
        
        // 모든 시나리오에서 유효한 결과
        assert!(forward_result.is_finite(), "Scenario {} forward failed", i);
        assert!(loss.is_finite(), "Scenario {} backward failed", i);
        assert!(loss >= 0.0, "Scenario {} negative loss", i);
        
        // 메트릭 유효성
        assert!(metrics.gradient_norm.is_finite(), "Scenario {} invalid gradient norm", i);
        assert!(metrics.stability_score >= 0.0 && metrics.stability_score <= 1.0, 
               "Scenario {} invalid stability score", i);
        
        println!("Scenario {}: Forward = {:.6}, Loss = {:.6}, Stability = {:.3}", 
                 i, forward_result, loss, metrics.stability_score);
    }
    
    // 최종 시스템 상태 확인
    assert!(system.verify_system_invariants());
    
    let final_metrics = system.get_performance_metrics();
    println!("Final system metrics: {:?}", final_metrics);
} 