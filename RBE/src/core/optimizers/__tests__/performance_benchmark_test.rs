use crate::core::optimizers::{AdamState, RiemannianAdamState};
use std::time::Instant;

/// Adam 성능 벤치마크
#[test]
fn adam_performance_benchmark() {
    let mut adam = AdamState::new();
    let mut param = 1.0;
    let gradient = 0.01;
    let learning_rate = 0.001;
    
    let iterations = 1000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        adam.update(&mut param, gradient, learning_rate);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("Adam 평균 업데이트 시간: {}ns", avg_time_ns);
    println!("최종 파라미터 값: {:.6}", param);
    
    // 성능 목표: 70ns 이하 (인라인 최적화)
    assert!(avg_time_ns < 70, "Adam 업데이트 시간이 너무 깁니다: {}ns", avg_time_ns);
}

/// Riemannian Adam 성능 벤치마크  
#[test]
fn riemannian_adam_performance_benchmark() {
    let mut riemannian_adam = RiemannianAdamState::new();
    let mut r = 0.5;
    let mut theta = 1.0;
    let grad_r = 0.01;
    let grad_theta = 0.005;
    let learning_rate = 0.001;
    
    let iterations = 1000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        riemannian_adam.update(&mut r, &mut theta, grad_r, grad_theta, learning_rate);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("Riemannian Adam 평균 업데이트 시간: {}ns", avg_time_ns);
    println!("최종 r: {:.6}, θ: {:.6}", r, theta);
    
    // 성능 목표: 220ns 이하 (복잡한 계산 고려)
    assert!(avg_time_ns < 220, "Riemannian Adam 업데이트 시간이 너무 깁니다: {}ns", avg_time_ns);
}

/// 배치 업데이트 성능 벤치마크
#[test] 
fn adam_batch_performance_benchmark() {
    let mut adam = AdamState::new();
    let mut params = vec![1.0; 100];
    let gradients = vec![0.01; 100];
    let learning_rate = 0.001;
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        adam.update_batch(&mut params, &gradients, learning_rate);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("Adam 배치 업데이트 평균 시간: {}ns (100개 파라미터)", avg_time_ns);
    println!("배치당 파라미터 평균 시간: {}ns", avg_time_ns / 100);
    
    // 성능 목표: 15000ns 이하 (100개 파라미터, 150ns/파라미터)
    assert!(avg_time_ns < 15000, "Adam 배치 업데이트 시간이 너무 깁니다: {}ns", avg_time_ns);
}

/// 조기 종료 최적화 효과 테스트
#[test]
fn adam_early_termination_optimization() {
    let mut adam = AdamState::new();
    let mut param = 1.0;
    let small_gradient = 1e-15; // 조기 종료 임계값 이하
    let learning_rate = 0.001;
    
    let original_param = param;
    let iterations = 1000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        adam.update(&mut param, small_gradient, learning_rate);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("조기 종료 시 평균 시간: {}ns", avg_time_ns);
    println!("파라미터 변화: {:.15} → {:.15}", original_param, param);
    
    // 조기 종료로 파라미터가 변하지 않아야 함
    assert_eq!(param, original_param, "조기 종료가 제대로 작동하지 않습니다");
    
    // 조기 종료 시 매우 빨라야 함 (35ns 이하)
    assert!(avg_time_ns < 35, "조기 종료 최적화가 효과적이지 않습니다: {}ns", avg_time_ns);
}

/// Riemannian Adam 경계값 최적화 효과 테스트
#[test]
fn riemannian_adam_boundary_optimization() {
    let mut riemannian_adam = RiemannianAdamState::new();
    let mut r = 0.9999998; // 경계에 매우 가까운 값
    let mut theta = 0.0;
    let grad_r = 0.001;
    let grad_theta = 0.001;
    let learning_rate = 0.01;
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        riemannian_adam.update(&mut r, &mut theta, grad_r, grad_theta, learning_rate);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("경계 근처 업데이트 평균 시간: {}ns", avg_time_ns);
    println!("경계값 유지: r = {:.10}", r);
    
    // 경계값이 올바르게 제한되어야 함
    assert!(r < 0.9999999, "푸앵카레볼 경계값 제한이 작동하지 않습니다");
    assert!(r >= 0.0, "r 값이 음수가 되었습니다");
    
    // 경계 근처에서도 합리적인 성능이어야 함 (500ns 이하)
    assert!(avg_time_ns < 500, "경계 근처 업데이트가 너무 느립니다: {}ns", avg_time_ns);
}

/// NaN/Inf 처리 성능 테스트
#[test]
fn nan_inf_handling_performance() {
    let mut adam = AdamState::new();
    let mut riemannian_adam = RiemannianAdamState::new();
    
    let mut param = 1.0;
    let mut r = 0.5;
    let mut theta = 1.0;
    
    let iterations = 1000;
    
    // Adam NaN 처리 테스트
    let start = Instant::now();
    for _ in 0..iterations {
        adam.update(&mut param, f32::NAN, 0.001);
    }
    let adam_duration = start.elapsed();
    let adam_avg_ns = adam_duration.as_nanos() / iterations as u128;
    
    // Riemannian Adam NaN 처리 테스트  
    let start = Instant::now();
    for _ in 0..iterations {
        riemannian_adam.update(&mut r, &mut theta, f32::NAN, f32::INFINITY, 0.001);
    }
    let riemannian_duration = start.elapsed();
    let riemannian_avg_ns = riemannian_duration.as_nanos() / iterations as u128;
    
    println!("Adam NaN 처리 평균 시간: {}ns", adam_avg_ns);
    println!("Riemannian Adam NaN 처리 평균 시간: {}ns", riemannian_avg_ns);
    
    // NaN 입력 시 파라미터가 변하지 않아야 함
    assert_eq!(param, 1.0, "Adam이 NaN을 제대로 처리하지 못했습니다");
    assert_eq!(r, 0.5, "Riemannian Adam이 NaN을 제대로 처리하지 못했습니다");
    assert_eq!(theta, 1.0, "Riemannian Adam이 Inf를 제대로 처리하지 못했습니다");
    
    // NaN 처리도 빨라야 함
    assert!(adam_avg_ns < 30, "Adam NaN 처리가 너무 느립니다: {}ns", adam_avg_ns);
    assert!(riemannian_avg_ns < 25, "Riemannian Adam NaN 처리가 너무 느립니다: {}ns", riemannian_avg_ns);
} 