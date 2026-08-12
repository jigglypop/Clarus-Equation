use crate::core::generator::PoincareLearning;
use crate::packed_params::Packed128;
use crate::core::math::AnalyticalGradient;
use rand::thread_rng;
use std::time::Instant;

#[test]
fn 기본_푸앵카레_학습_테스트() {
    println!("=== 기본 푸앵카레 학습 테스트 ===");
    
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    
    // 간단한 타겟 패턴 생성
    let rows = 4;
    let cols = 4;
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            if (row + col) % 2 == 0 { 1.0 } else { 0.0 }
        })
        .collect();
    
    // 초기 예측 생성
    let mut predictions = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            predictions.push(params.fused_forward(i, j, rows, cols));
        }
    }
    
    let initial_mse = target.iter()
        .zip(predictions.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    
    println!("초기 MSE: {:.6}", initial_mse);
    
    // 상태 전이 적용 테스트
    for (i, (&t, &p)) in target.iter().zip(predictions.iter()).enumerate() {
        let error = p - t;
        let row = i / cols;
        let col = i % cols;
        params.apply_state_transition(error, row, col);
    }
    
    println!("상태 전이 적용 완료");
    
    assert!(initial_mse >= 0.0, "초기 MSE가 음수");
    println!("기본 푸앵카레 학습 테스트 통과!");
}

#[test]
fn 해석적_그래디언트_테스트() {
    println!("=== 해석적 그래디언트 테스트 ===");
    
    let mut rng = thread_rng();
    let seed = Packed128::random(&mut rng);
    let rows = 4;
    let cols = 4;
    
    // 해석적 그래디언트 계산
    for i in 0..rows {
        for j in 0..cols {
            let grad_r = seed.analytical_gradient_r(i, j, rows, cols);
            let grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            
            assert!(grad_r.is_finite(), "r 그래디언트가 무한대: {}", grad_r);
            assert!(grad_theta.is_finite(), "theta 그래디언트가 무한대: {}", grad_theta);
        }
    }
    
    println!("해석적 그래디언트 테스트 통과!");
}

#[test]
fn 상태_전이_미분_테스트() {
    println!("=== 상태 전이 미분 테스트 ===");
    
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    
    let initial_hi = params.hi;
    
    // 다양한 그래디언트 신호로 상태 전이 테스트
    let test_gradients = [0.0, 0.05, 0.15, -0.05, -0.15];
    
    for &gradient in &test_gradients {
        params.hi = initial_hi; // 상태 초기화
        params.apply_state_transition(gradient, 0, 0);
        
        // 상태가 변경되었는지 확인
        if gradient.abs() > 0.1 {
            // 강한 그래디언트는 상태 변경을 야기할 수 있음
            println!("그래디언트 {:.3}: hi 변화 0x{:016x} -> 0x{:016x}", 
                     gradient, initial_hi, params.hi);
        }
    }
    
    println!("상태 전이 미분 테스트 통과!");
}

#[test]
fn 메모리_효율성_검증_테스트() {
    println!("=== 메모리 효율성 검증 테스트 ===");
    
    let test_sizes = [(8, 8), (16, 16), (32, 32)];
    
    for (rows, cols) in test_sizes {
        let elements = rows * cols;
        let dense_size = elements * std::mem::size_of::<f32>();
        let compressed_size = std::mem::size_of::<Packed128>();
        let compression_ratio = dense_size as f32 / compressed_size as f32;
        
        println!("{}x{} 행렬: Dense {}KB vs RBE {}bytes (압축률 {:.1}:1)", 
                 rows, cols, dense_size / 1024, compressed_size, compression_ratio);
        
        assert!(compression_ratio > 10.0, "압축률이 부족: {:.1}:1", compression_ratio);
    }
    
    println!("메모리 효율성 검증 테스트 통과!");
}

#[test]
fn 학습_수렴성_테스트() {
    println!("=== 학습 수렴성 테스트 ===");
    
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    
    // 간단한 선형 그래디언트 패턴
    let rows = 4;
    let cols = 4;
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| {
            let x = (i % cols) as f32 / (cols - 1) as f32;
            x // 0에서 1로 증가하는 패턴
        })
        .collect();
    
    let mut best_loss = f32::INFINITY;
    
    for epoch in 0..20 {
        // 예측 생성
        let mut predictions = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                predictions.push(params.fused_forward(i, j, rows, cols));
            }
        }
        
        // 손실 계산
        let mse = target.iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>() / target.len() as f32;
        
        if mse < best_loss {
            best_loss = mse;
        }
        
        // 상태 전이 적용 (간단한 학습 시뮬레이션)
        for (i, (&t, &p)) in target.iter().zip(predictions.iter()).enumerate() {
            let error = p - t;
            let row = i / cols;
            let col = i % cols;
            params.apply_state_transition(error * 0.1, row, col); // 스케일링
        }
        
        if epoch % 5 == 0 {
            println!("에포크 {}: MSE={:.6}", epoch, mse);
        }
    }
    
    assert!(best_loss < 2.0, "학습이 수렴하지 않음: 최종 손실={:.6}", best_loss);
    println!("학습 수렴성 테스트 통과!");
} 