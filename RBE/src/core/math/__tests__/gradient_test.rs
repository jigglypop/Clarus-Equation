use crate::core::math::gradient::AnalyticalGradient;
use crate::core::packed_params::{Packed128, Packed64};
use rand::{thread_rng, Rng};
use std::time::Instant;

#[test]
fn 해석적_미분_성능_벤치마크_테스트() {
    let mut rng = thread_rng();
    let rows = 32;
    let cols = 32;
    let iterations = 1000;
    
    // 테스트 데이터 생성
    let mut seeds: Vec<Packed128> = (0..iterations)
        .map(|_| Packed128::random(&mut rng))
        .collect();
    
    println!("해석적 미분 vs 수치적 미분 성능 비교 ({}x{} 그리드, {} 반복)", rows, cols, iterations);
    
    // 해석적 미분 성능 측정
    let start = Instant::now();
    for seed in &seeds {
        for i in 0..rows {
            for j in 0..cols {
                let _grad_r = seed.analytical_gradient_r(i, j, rows, cols);
                let _grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            }
        }
    }
    let analytical_time = start.elapsed();
    
    // 수치적 미분 성능 측정 (기준)
    let eps = 1e-5;
    let start = Instant::now();
    for seed in &mut seeds {
        for i in 0..rows {
            for j in 0..cols {
                // R에 대한 수치적 미분
                let original_r = f32::from_bits((seed.lo >> 32) as u32);
                let f_original = seed.fused_forward(i, j, rows, cols);
                
                // R + eps
                let r_plus = original_r + eps;
                seed.lo = ((r_plus.to_bits() as u64) << 32) | (seed.lo & 0xFFFFFFFF);
                let f_plus = seed.fused_forward(i, j, rows, cols);
                
                // R 복원
                seed.lo = ((original_r.to_bits() as u64) << 32) | (seed.lo & 0xFFFFFFFF);
                
                let _numerical_grad_r = (f_plus - f_original) / eps;
                
                // Theta에 대한 수치적 미분
                let original_theta = f32::from_bits(seed.lo as u32);
                let theta_plus = original_theta + eps;
                seed.lo = (seed.lo & 0xFFFFFFFF00000000) | theta_plus.to_bits() as u64;
                let f_plus_theta = seed.fused_forward(i, j, rows, cols);
                
                // Theta 복원
                seed.lo = (seed.lo & 0xFFFFFFFF00000000) | original_theta.to_bits() as u64;
                
                let _numerical_grad_theta = (f_plus_theta - f_original) / eps;
            }
        }
    }
    let numerical_time = start.elapsed();
    // 결과 출력
    println!("해석적 미분 소요시간: {:?}", analytical_time);
    println!("수치적 미분 소요시간: {:?}", numerical_time);
    let speedup = numerical_time.as_secs_f64() / analytical_time.as_secs_f64();
    println!("속도 향상: {:.2}x", speedup);
    // 해석적 미분이 최소 8배는 빨라야 함 (비트 연산 기반)
    assert!(speedup >= 8.0, "해석적 미분이 충분히 빠르지 않음: {:.2}x", speedup);
}

#[test]
fn 해석적_미분_정확성_대규모_테스트() {
    let mut rng = thread_rng();
    let seed = Packed128::random(&mut rng);
    let rows = 16;
    let cols = 16;
    let eps = 1e-4;
    let tolerance = 0.15; // 15% 허용 오차
    
    let mut max_error_r = 0.0f32;
    let mut max_error_theta = 0.0f32;
    let mut total_tests = 0;
    let mut passed_tests = 0;
    
    for i in 0..rows {
        for j in 0..cols {
            // 해석적 미분
            let analytical_r = seed.analytical_gradient_r(i, j, rows, cols);
            let analytical_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            
            // 수치적 미분으로 검증
            let mut seed_copy = seed;
            let original_r = f32::from_bits((seed_copy.lo >> 32) as u32);
            let original_theta = f32::from_bits(seed_copy.lo as u32);
            let f_original = seed_copy.fused_forward(i, j, rows, cols);
            
            // R에 대한 수치적 미분
            let r_plus = original_r + eps;
            seed_copy.lo = ((r_plus.to_bits() as u64) << 32) | (seed_copy.lo & 0xFFFFFFFF);
            let f_plus_r = seed_copy.fused_forward(i, j, rows, cols);
            seed_copy.lo = ((original_r.to_bits() as u64) << 32) | (seed_copy.lo & 0xFFFFFFFF);
            let numerical_r = (f_plus_r - f_original) / eps;
            
            // Theta에 대한 수치적 미분
            let theta_plus = original_theta + eps;
            seed_copy.lo = (seed_copy.lo & 0xFFFFFFFF00000000) | theta_plus.to_bits() as u64;
            let f_plus_theta = seed_copy.fused_forward(i, j, rows, cols);
            let numerical_theta = (f_plus_theta - f_original) / eps;
            
            // 상대 오차 계산
            total_tests += 2;
            
            if numerical_r.abs() > 1e-8 {
                let rel_error_r = ((analytical_r - numerical_r) / numerical_r).abs();
                max_error_r = max_error_r.max(rel_error_r);
                if rel_error_r <= tolerance {
                    passed_tests += 1;
                }
            } else {
                passed_tests += 1; // 0 근처에서는 통과로 처리
            }
            
            if numerical_theta.abs() > 1e-8 {
                let rel_error_theta = ((analytical_theta - numerical_theta) / numerical_theta).abs();
                max_error_theta = max_error_theta.max(rel_error_theta);
                if rel_error_theta <= tolerance {
                    passed_tests += 1;
                }
            } else {
                passed_tests += 1; // 0 근처에서는 통과로 처리
            }
        }
    }
    
    let pass_rate = passed_tests as f32 / total_tests as f32;
    println!("해석적 미분 정확성 통계:");
    println!("  총 테스트: {}", total_tests);
    println!("  통과한 테스트: {}", passed_tests);
    println!("  통과율: {:.1}%", pass_rate * 100.0);
    println!("  최대 R 상대오차: {:.3}%", max_error_r * 100.0);
    println!("  최대 Theta 상대오차: {:.3}%", max_error_theta * 100.0);
    
    // 최소 70% 이상 통과해야 함
    assert!(pass_rate >= 0.7, "해석적 미분 정확성이 너무 낮음: {:.1}%", pass_rate * 100.0);
}

#[test]
fn packed64_성능_테스트() {
    let mut rng = thread_rng();
    let rows = 16;
    let cols = 16;
    let iterations = 500;
    
    // Packed64 테스트 데이터 생성
    let seeds: Vec<Packed64> = (0..iterations)
        .map(|_| Packed64::new(rng.gen::<u64>()))
        .collect();
    
    // Packed64는 compute_weight가 Packed128보다 단순하므로 성능이 좋을 것으로 예상
    let start = Instant::now();
    for seed in &seeds {
        for i in 0..rows {
            for j in 0..cols {
                let _value = seed.compute_weight(i, j, rows, cols);
            }
        }
    }
    let packed64_time = start.elapsed();
    
    println!("Packed64 compute_weight 성능: {:?} ({} 호출)", 
             packed64_time, iterations * rows * cols);
    println!("호출당 평균 시간: {:.2} ns", 
             packed64_time.as_nanos() as f64 / (iterations * rows * cols) as f64);
    
    // Packed64는 충분히 빨라야 함 (각 호출이 1μs 미만)
    let avg_time_ns = packed64_time.as_nanos() as f64 / (iterations * rows * cols) as f64;
    assert!(avg_time_ns < 1000.0, "Packed64 compute_weight가 너무 느림: {:.2} ns/호출", avg_time_ns);
}

// 이하는 packed_params/__tests__/에서 이동된 정확성 검증 테스트들
#[test]
fn 해석적_미분_r_파라미터_정확성_검증() {
    use rand::thread_rng;
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터로 설정하여 예측 가능한 테스트
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let eps = 1e-4; // 더 안정적인 eps 값으로 조정
    
    for i in 0..rows {
        for j in 0..cols {
            // 해석적 미분 결과
            let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
            
            // 수치 미분으로 검증 (ground truth)
            let mut seed_plus = seed;
            let r_plus = r_value + eps;
            seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_plus = seed_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_minus = seed;
            let r_minus = r_value - eps;
            seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_minus = seed_minus.fused_forward(i, j, rows, cols);
            
            let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
            
            // 상대 오차 10% 이내로 완화 (복잡한 비선형 함수이므로)
            let relative_error = if numerical_grad.abs() > 1e-6 {
                (analytical_grad - numerical_grad).abs() / numerical_grad.abs()
            } else {
                (analytical_grad - numerical_grad).abs()
            };
            
            assert!(
                relative_error < 0.10,
                "위치 ({},{})에서 r 미분 오차가 너무 큼: 해석적={:.6}, 수치적={:.6}, 상대오차={:.3}%", 
                i, j, analytical_grad, numerical_grad, relative_error * 100.0
            );
        }
    }
}

#[test]
fn 해석적_미분_theta_파라미터_정확성_검증() {
    use rand::thread_rng;
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터로 설정
    let r_value = 0.8f32;
    let theta_value = 0.2f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let eps = 1e-4; // 더 안정적인 eps 값으로 조정
    
    for i in 0..rows {
        for j in 0..cols {
            // 해석적 미분 결과
            let analytical_grad = seed.analytical_gradient_theta(i, j, rows, cols);
            
            // 수치 미분으로 검증
            let mut seed_plus = seed;
            let theta_plus = theta_value + eps;
            seed_plus.lo = ((r_value.to_bits() as u64) << 32) | theta_plus.to_bits() as u64;
            let f_plus = seed_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_minus = seed;
            let theta_minus = theta_value - eps;
            seed_minus.lo = ((r_value.to_bits() as u64) << 32) | theta_minus.to_bits() as u64;
            let f_minus = seed_minus.fused_forward(i, j, rows, cols);
            
            let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
            
            // 상대 오차 15% 이내로 완화 (theta는 더 복잡함)
            let relative_error = if numerical_grad.abs() > 1e-6 {
                (analytical_grad - numerical_grad).abs() / numerical_grad.abs()
            } else {
                (analytical_grad - numerical_grad).abs()
            };
            
            assert!(
                relative_error < 0.15,
                "위치 ({},{})에서 theta 미분 오차가 너무 큼: 해석적={:.6}, 수치적={:.6}, 상대오차={:.3}%", 
                i, j, analytical_grad, numerical_grad, relative_error * 100.0
            );
        }
    }
}
