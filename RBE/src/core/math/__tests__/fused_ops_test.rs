use crate::packed_params::Packed128;
use crate::core::math::fused_ops::{fused_backward, fused_backward_fast, fused_backward_gemv, fused_backward_adam};
use std::time::Instant;
use rand::{thread_rng, Rng};

#[test]
fn 빠른_역전파_정확성_검증() {
    let mut rng = thread_rng();
    let rows = 8;
    let cols = 8;
    
    // 동일한 시드로 두 함수 테스트
    let mut seed_original = Packed128::random(&mut rng);
    let mut seed_fast = seed_original;
    
    // 간단한 타겟 패턴 생성
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| (i as f32 / (rows*cols) as f32).sin())
        .collect();
    
    // 초기 예측 생성
    let mut predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = seed_original.fused_forward(i, j, rows, cols);
        }
    }
    
    let learning_rate = 0.01;
    
    // 기존 함수 실행
    let (mse_original, rmse_original) = fused_backward(
        &target, &predicted, &mut seed_original, rows, cols, learning_rate
    );
    
    // 빠른 함수 실행
    let (mse_fast, rmse_fast) = fused_backward_fast(
        &target, &predicted, &mut seed_fast, rows, cols, learning_rate
    );
    
    // 결과가 유사해야 함 (상대 오차 10% 이내)
    let mse_error = (mse_original - mse_fast).abs() / mse_original;
    let rmse_error = (rmse_original - rmse_fast).abs() / rmse_original;
    
    assert!(
        mse_error < 0.1,
        "MSE 차이가 너무 큼: 기존={:.6}, 빠른={:.6}, 오차={:.3}%", 
        mse_original, mse_fast, mse_error * 100.0
    );
    
    assert!(
        rmse_error < 0.1,
        "RMSE 차이가 너무 큼: 기존={:.6}, 빠른={:.6}, 오차={:.3}%", 
        rmse_original, rmse_fast, rmse_error * 100.0
    );
    
    println!("정확성 검증 성공: MSE={:.6}, RMSE={:.6}", mse_fast, rmse_fast);
}

#[test]
fn 역전파_성능_비교() {
    let mut rng = thread_rng();
    let rows = 16;
    let cols = 16;
    
    let mut seed1 = Packed128::random(&mut rng);
    let mut seed2 = seed1;
    
    // 복잡한 타겟 패턴 생성
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| {
            let x = (i % cols) as f32 / cols as f32;
            let y = (i / cols) as f32 / rows as f32;
            (x * 3.14159).sin() * (y * 3.14159).cos()
        })
        .collect();
    
    let mut predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = seed1.fused_forward(i, j, rows, cols);
        }
    }
    
    let learning_rate = 0.005;
    
    // 기존 역전파 성능 측정
    let start_time = Instant::now();
    let (mse1, _) = fused_backward(&target, &predicted, &mut seed1, rows, cols, learning_rate);
    let time_original = start_time.elapsed();
    
    // 빠른 역전파 성능 측정
    let start_time = Instant::now();
    let (mse2, _) = fused_backward_fast(&target, &predicted, &mut seed2, rows, cols, learning_rate);
    let time_fast = start_time.elapsed();
    
    let speedup = time_original.as_secs_f64() / time_fast.as_secs_f64();
    
    println!("성능 비교:");
    println!("  기존 역전파: {:.2}ms, MSE: {:.6}", time_original.as_secs_f64() * 1000.0, mse1);
    println!("  빠른 역전파: {:.2}ms, MSE: {:.6}", time_fast.as_secs_f64() * 1000.0, mse2);
    println!("  속도 향상: {:.2}x", speedup);
    
    // 빠른 버전이 더 빨라야 함
    assert!(speedup > 1.5, "빠른 역전파가 충분히 빠르지 않음: {:.2}x", speedup);
    
    // 정확도는 유사해야 함
    let accuracy_error = (mse1 - mse2).abs() / mse1;
    assert!(accuracy_error < 0.2, "정확도 차이가 너무 큼: {:.3}%", accuracy_error * 100.0);
}

#[test]
fn 수치적_안정성_테스트() {
    let mut rng = thread_rng();
    let rows = 4;
    let cols = 4;
    
    // 극값 테스트 케이스들
    let extreme_targets = vec![
        vec![1000.0; rows * cols],    // 매우 큰 값
        vec![-1000.0; rows * cols],   // 매우 작은 값
        vec![0.0001; rows * cols],    // 매우 작은 양수
        vec![-0.0001; rows * cols],   // 매우 작은 음수
    ];
    
    for (case_idx, target) in extreme_targets.iter().enumerate() {
        let mut seed = Packed128::random(&mut rng);
        
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i*cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        let learning_rate = 0.001; // 안전한 학습률
        
        let (mse, rmse) = fused_backward_fast(target, &predicted, &mut seed, rows, cols, learning_rate);
        
        // 결과가 유한해야 함
        assert!(mse.is_finite(), "케이스 {}에서 MSE가 무한대: {}", case_idx, mse);
        assert!(rmse.is_finite(), "케이스 {}에서 RMSE가 무한대: {}", case_idx, rmse);
        assert!(mse >= 0.0, "케이스 {}에서 MSE가 음수: {}", case_idx, mse);
        assert!(rmse >= 0.0, "케이스 {}에서 RMSE가 음수: {}", case_idx, rmse);
        
        // 업데이트된 파라미터들도 유한해야 함
        let updated_r = f32::from_bits((seed.lo >> 32) as u32);
        let updated_theta = f32::from_bits(seed.lo as u32);
        
        assert!(updated_r.is_finite(), "케이스 {}에서 업데이트된 r이 무한대: {}", case_idx, updated_r);
        assert!(updated_theta.is_finite(), "케이스 {}에서 업데이트된 theta가 무한대: {}", case_idx, updated_theta);
        
        println!("케이스 {}: MSE={:.6}, RMSE={:.6}, r={:.4}, theta={:.4}", 
                 case_idx, mse, rmse, updated_r, updated_theta);
    }
}

#[test]
fn 학습률_민감도_테스트() {
    let mut rng = thread_rng();
    let rows = 6;
    let cols = 6;
    
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| (i as f32 / (rows*cols) as f32) * 2.0 - 1.0) // [-1, 1] 범위
        .collect();
    
    let learning_rates = vec![0.001, 0.01, 0.1, 1.0];
    
    for lr in learning_rates {
        let mut seed = Packed128::random(&mut rng);
        
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i*cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        let (mse, rmse) = fused_backward_fast(&target, &predicted, &mut seed, rows, cols, lr);
        
        // 모든 학습률에서 안정적인 결과
        assert!(mse.is_finite(), "학습률 {}에서 MSE가 무한대: {}", lr, mse);
        assert!(rmse.is_finite(), "학습률 {}에서 RMSE가 무한대: {}", lr, rmse);
        assert!(mse >= 0.0, "학습률 {}에서 MSE가 음수: {}", lr, mse);
        
        // 너무 큰 학습률에서는 발산할 수 있지만, 기본적으로 클램핑이 있어야 함
        if lr <= 0.1 {
            assert!(mse < 100.0, "학습률 {}에서 MSE가 너무 큼: {}", lr, mse);
        }
        
        println!("학습률 {}: MSE={:.6}, RMSE={:.6}", lr, mse, rmse);
    }
}

#[test]
fn 배치_크기_확장성_테스트() {
    let mut rng = thread_rng();
    
    let batch_sizes = vec![(2, 2), (4, 4), (8, 8), (16, 16)];
    
    for (rows, cols) in batch_sizes {
        let mut seed = Packed128::random(&mut rng);
        
        // 체스판 패턴
        let target: Vec<f32> = (0..rows*cols)
            .map(|i| {
                let row = i / cols;
                let col = i % cols;
                if (row + col) % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i*cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        let learning_rate = 0.01;
        let start_time = Instant::now();
        let (mse, rmse) = fused_backward_fast(&target, &predicted, &mut seed, rows, cols, learning_rate);
        let elapsed = start_time.elapsed();
        
        // 결과 검증
        assert!(mse.is_finite(), "{}x{} 배치에서 MSE가 무한대", rows, cols);
        assert!(rmse.is_finite(), "{}x{} 배치에서 RMSE가 무한대", rows, cols);
        
        println!("배치 {}x{}: MSE={:.6}, RMSE={:.6}, 시간={:.2}ms", 
                 rows, cols, mse, rmse, elapsed.as_secs_f64() * 1000.0);
    }
}

#[test]
fn 융합_연산_메모리_효율성_테스트() {
    let rows = 32;
    let cols = 32;
    let elements = rows * cols;
    
    // Dense 행렬 메모리 사용량 추정
    let dense_memory = elements * std::mem::size_of::<f32>(); // 4KB
    
    // 융합 연산 메모리 사용량 (Packed128 1개)
    let fused_memory = std::mem::size_of::<Packed128>(); // 16 bytes
    
    let memory_ratio = dense_memory as f32 / fused_memory as f32;
    
    println!("메모리 효율성:");
    println!("  Dense 행렬: {} KB", dense_memory / 1024);
    println!("  융합 연산: {} bytes", fused_memory);
    println!("  메모리 절약: {:.1}:1", memory_ratio);
    
    // 최소 50:1 압축률 달성
    assert!(memory_ratio > 50.0, "메모리 압축률이 부족: {:.1}:1", memory_ratio);
    
    // 실제 기능성 테스트
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    let target: Vec<f32> = (0..elements).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut predicted = vec![0.0; target.len()];
    
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = seed.fused_forward(i, j, rows, cols);
        }
    }
    
    let (mse, _) = fused_backward_fast(&target, &predicted, &mut seed, rows, cols, 0.01);
    
    assert!(mse.is_finite(), "대용량 데이터에서 MSE가 무한대");
    println!("  대용량 테스트: MSE={:.6}", mse);
} 
 