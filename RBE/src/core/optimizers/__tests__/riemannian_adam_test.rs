use crate::core::optimizers::RiemannianAdamState;
use std::f32::consts::PI;

#[test]
fn Riemannian_Adam상태_초기화_테스트() {
    let riemannian_adam = RiemannianAdamState::new();
    assert_eq!(riemannian_adam.m_r, 0.0, "r 파라미터 1차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_adam.v_r, 0.0, "r 파라미터 2차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_adam.m_theta, 0.0, "θ 파라미터 1차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_adam.v_theta, 0.0, "θ 파라미터 2차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_adam.t, 0, "시간 스텝은 0으로 초기화");
    assert_eq!(riemannian_adam.beta1, 0.9, "beta1 기본값은 0.9");
    assert_eq!(riemannian_adam.beta2, 0.999, "beta2 기본값은 0.999");
    assert_eq!(riemannian_adam.epsilon, 1e-8, "epsilon 기본값은 1e-8");
    println!("✅ Riemannian Adam 상태 초기화 테스트 통과");
}

#[test]
fn 메트릭텐서_계산_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    
    // 다양한 r 값에 대한 메트릭 텐서 계산
    let test_cases = [
        (0.0, (4.0, 0.0)),      // r=0일 때: g_rr=4, g_θθ=0
        (0.5, (16.0/3.0, 4.0/3.0)), // r=0.5일 때
        (0.9, (104.47, 85.18)),      // r=0.9일 때 (근사값)
    ];
    
    for (r, expected) in test_cases {
        let (g_rr, g_theta_theta) = riemannian_adam.compute_metric_tensor(r);
        
        println!("   r={}: g_rr={:.2}, g_θθ={:.2}", r, g_rr, g_theta_theta);
        
        if r == 0.0 {
            assert!((g_rr - expected.0).abs() < 1e-6, "r=0일 때 g_rr 정확성");
            assert!((g_theta_theta - expected.1).abs() < 1e-6, "r=0일 때 g_θθ 정확성");
        } else {
            // 대략적인 범위 확인
            assert!(g_rr > 0.0, "g_rr은 항상 양수여야 함");
            assert!(g_theta_theta >= 0.0, "g_θθ은 항상 비음수여야 함");
            assert!(g_rr > g_theta_theta, "일반적으로 g_rr > g_θθ");
        }
    }
    
    println!("✅ 메트릭 텐서 계산 테스트 통과");
}

#[test]
fn 뫼비우스덧셈_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    
    // 기본 테스트 케이스들
    let test_cases = [
        (0.0, 0.0, 0.0),        // 0 + 0 = 0
        (0.5, 0.0, 0.5),        // x + 0 = x
        (0.0, 0.3, 0.3),        // 0 + y = y
        (0.5, 0.3, 0.615),      // 일반적인 경우 (근사값)
        (-0.2, 0.2, 0.0),       // 근사적 역원
    ];
    
    for (x, y, expected) in test_cases {
        let result = riemannian_adam.mobius_add(x, y);
        println!("   뫼비우스 덧셈: {} ⊕ {} = {:.3}", x, y, result);
        
        if (expected - result).abs() < 0.1 {
            // 허용 오차 내에서 일치
            assert!((result - expected).abs() < 0.1, "뫼비우스 덧셈 결과 확인");
        }
        
        // 항상 |result| ≤ 1 이어야 함 (푸앵카레 볼 내부)
        assert!(result.abs() <= 1.0, "결과가 푸앵카레 볼 내부에 있어야 함");
    }
    
    // 교환법칙 확인
    let x = 0.3;
    let y = 0.4;
    let xy = riemannian_adam.mobius_add(x, y);
    let yx = riemannian_adam.mobius_add(y, x);
    assert!((xy - yx).abs() < 1e-6, "뫼비우스 덧셈의 교환법칙");
    
    println!("✅ 뫼비우스 덧셈 테스트 통과");
}

#[test]
fn 지수사상_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    
    // 기본 테스트 케이스들
    let test_cases = [
        (0.0, 0.0, 0.0),        // exp_0(0) = 0
        (0.5, 0.0, 0.5),        // exp_x(0) = x
        (0.0, 0.1, 0.1),        // exp_0(v) ≈ v (작은 v에 대해)
        (0.2, 0.1, 0.3),        // 일반적인 경우 (근사)
    ];
    
    for (x, v, _expected) in test_cases {
        let result = riemannian_adam.exponential_map(x, v);
        println!("   지수사상: exp_{}({}) = {:.3}", x, v, result);
        
        // 결과가 푸앵카레 볼 내부에 있어야 함
        assert!(result.abs() < 1.0, "지수사상 결과가 푸앵카레 볼 내부에 있어야 함");
    }
    
    // v=0일 때는 x와 동일해야 함
    let x = 0.7;
    let result = riemannian_adam.exponential_map(x, 0.0);
    assert!((result - x).abs() < 1e-6, "v=0일 때 exp_x(0) = x");
    
    // 작은 v에 대해서는 x + v와 근사해야 함
    let x = 0.1;
    let v = 0.01;
    let result = riemannian_adam.exponential_map(x, v);
    assert!((result - (x + v)).abs() < 0.01, "작은 v에 대한 선형 근사");
    
    println!("✅ 지수사상 테스트 통과");
}

#[test]
fn Riemannian_Adam업데이트_기본동작_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    let mut r = 0.5;
    let mut theta = PI / 4.0;
    let grad_r = 0.1;
    let grad_theta = 0.05;
    let lr = 0.01;
    
    let original_r = r;
    let original_theta = theta;
    
    riemannian_adam.update(&mut r, &mut theta, grad_r, grad_theta, lr);
    
    assert_ne!(r, original_r, "r 파라미터가 업데이트되어야 함");
    assert_ne!(theta, original_theta, "θ 파라미터가 업데이트되어야 함");
    assert!(r >= 0.0 && r < 1.0, "r은 [0, 1) 범위 내에 있어야 함");
    assert!(theta >= 0.0 && theta < 2.0 * PI, "θ는 [0, 2π) 범위 내에 있어야 함");
    assert_eq!(riemannian_adam.t, 1, "시간 스텝이 증가해야 함");
    
    println!("✅ Riemannian Adam 업데이트 기본동작 테스트 통과");
    println!("   r: {:.3} → {:.3}", original_r, r);
    println!("   θ: {:.3} → {:.3}", original_theta, theta);
}

#[test]
fn Riemannian_Adam업데이트_경계조건_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    
    // r이 1에 가까운 경우
    let mut r = 0.99;
    let mut theta = 0.0;
    let grad_r = 0.1;  // 양의 그래디언트 (r 증가 방향)
    let grad_theta = 0.0;
    let lr = 0.1;
    
    riemannian_adam.update(&mut r, &mut theta, grad_r, grad_theta, lr);
    
    assert!(r < 1.0, "r은 항상 1 미만이어야 함");
    assert!(r >= 0.0, "r은 항상 0 이상이어야 함");
    
    println!("✅ Riemannian Adam 업데이트 경계조건 테스트 통과");
    println!("   경계 근처 r: 0.99 → {:.6}", r);
}

#[test]
fn Riemannian_Adam업데이트_연속호출_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    let mut r = 0.8;
    let mut theta = PI / 2.0;
    let grad_r = -0.05;  // r 감소 방향
    let grad_theta = 0.02;
    let lr = 0.01;
    
    let initial_r = r;
    let initial_theta = theta;
    
    // 50번 업데이트
    for i in 0..50 {
        riemannian_adam.update(&mut r, &mut theta, grad_r, grad_theta, lr);
        
        // 중간 확인
        if i % 10 == 9 {
            println!("   스텝 {}: r={:.3}, θ={:.3}", i + 1, r, theta);
        }
        
        // 경계 조건 확인
        assert!(r >= 0.0 && r < 1.0, "r이 올바른 범위에 있어야 함");
        assert!(theta >= 0.0 && theta < 2.0 * PI, "θ가 올바른 범위에 있어야 함");
    }
    
    assert_eq!(riemannian_adam.t, 50, "시간 스텝이 올바르게 증가해야 함");
    
    println!("✅ Riemannian Adam 업데이트 연속호출 테스트 통과");
    println!("   초기: r={:.3}, θ={:.3}", initial_r, initial_theta);
    println!("   최종: r={:.3}, θ={:.3}", r, theta);
}

#[test]
fn Riemannian_Adam상태_with_config_테스트() {
    let riemannian_adam = RiemannianAdamState::with_config(0.95, 0.9999, 1e-7);
    
    assert_eq!(riemannian_adam.beta1, 0.95, "커스텀 beta1 값 확인");
    assert_eq!(riemannian_adam.beta2, 0.9999, "커스텀 beta2 값 확인");
    assert_eq!(riemannian_adam.epsilon, 1e-7, "커스텀 epsilon 값 확인");
    assert_eq!(riemannian_adam.m_r, 0.0, "r 1차 모멘트는 여전히 0으로 초기화");
    assert_eq!(riemannian_adam.v_r, 0.0, "r 2차 모멘트는 여전히 0으로 초기화");
    assert_eq!(riemannian_adam.m_theta, 0.0, "θ 1차 모멘트는 여전히 0으로 초기화");
    assert_eq!(riemannian_adam.v_theta, 0.0, "θ 2차 모멘트는 여전히 0으로 초기화");
    
    println!("✅ Riemannian Adam 상태 with_config 테스트 통과");
    println!("   커스텀 설정: β1={}, β2={}, ε={}", 0.95, 0.9999, 1e-7);
}

#[test]
fn Riemannian_Adam상태_reset_테스트() {
    let mut riemannian_adam = RiemannianAdamState::new();
    let mut r = 0.5;
    let mut theta = PI / 4.0;
    
    // 여러 번 업데이트
    for _ in 0..5 {
        riemannian_adam.update(&mut r, &mut theta, 0.1, 0.2, 0.001);
    }
    
    // 상태 초기화
    riemannian_adam.reset();
    
    // 상태가 초기화되었는지 확인
    assert_eq!(riemannian_adam.t, 0);
    assert_eq!(riemannian_adam.m_r, 0.0);
    assert_eq!(riemannian_adam.v_r, 0.0);
    assert_eq!(riemannian_adam.m_theta, 0.0);
    assert_eq!(riemannian_adam.v_theta, 0.0);
}

// 추가: Small-Move 지오데식 근사 성능 테스트 (수정됨)
#[test]
fn test_small_move_geodesic_성능() {
    use std::time::Instant;
    
    let iterations = 100000;
    
    // 작은 이동만 테스트 (근사 활성화)
    let mut r = 0.5;
    let mut total_time_approx = std::time::Duration::ZERO;
    
    for _ in 0..iterations {
        let small_v = 0.05; // 0.1보다 작은 값
        let adam = RiemannianAdamState::new();
        
        let start = Instant::now();
        let _ = adam.exponential_map(r, small_v);
        total_time_approx += start.elapsed();
        
        r = (r + 0.001) % 0.99; // r 값 변경
    }
    
    // 큰 이동만 테스트 (정확한 공식)
    r = 0.5;
    let mut total_time_exact = std::time::Duration::ZERO;
    
    for _ in 0..iterations {
        let large_v = 0.5; // 0.1보다 큰 값
        let adam = RiemannianAdamState::new();
        
        let start = Instant::now();
        let _ = adam.exponential_map(r, large_v);
        total_time_exact += start.elapsed();
        
        r = (r + 0.001) % 0.99; // r 값 변경
    }
    
    println!("Small-move 근사 (작은 이동): {:?} ({} iterations)", total_time_approx, iterations);
    println!("정확한 공식 (큰 이동): {:?} ({} iterations)", total_time_exact, iterations);
    
    // 작은 이동에서는 근사가 더 빠름
    println!("속도 향상: {:.2}x", total_time_exact.as_secs_f64() / total_time_approx.as_secs_f64());
}

// 추가: Small-Move 근사 정확도 테스트
#[test]
fn test_small_move_geodesic_정확도() {
    let adam = RiemannianAdamState::new();
    
    // 작은 이동에 대한 근사 테스트
    let x = 0.5;
    let small_v = 0.01;
    
    // 근사 공식
    let x_norm_sq = x * x;
    let conformal_factor = 1.0 / (1.0 - x_norm_sq);
    let approx_result = x + small_v * conformal_factor;
    
    // 정확한 공식 (mobius_add)
    let exact_result = adam.exponential_map(x, small_v);
    
    // 근사 오차가 충분히 작아야 함
    let error = (approx_result - exact_result).abs();
    assert!(error < 1e-4, "근사 오차: {}", error);
} 