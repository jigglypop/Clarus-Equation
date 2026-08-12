use crate::core::math::gradient::AnalyticalGradient;
use crate::core::packed_params::Packed128;
use rand::{thread_rng, Rng};

#[test]
fn 해석적_미분_디버그_상세_분석() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터로 설정하여 예측 가능한 테스트
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let i = 2; // 실패한 위치
    let j = 2; // 실패한 위치
    let eps = 1e-5;
    
    println!("=== 해석적 미분 상세 분석 (위치: {},{}) ===", i, j);
    println!("R 파라미터: {}", r_value);
    println!("Theta 파라미터: {}", theta_value);
    println!("상태 비트 (hi): 0x{:X}", seed.hi);
    
    // 좌표 정규화 확인
    let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
    let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
    let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
    let base_angle = y_norm.atan2(x_norm);
    
    println!("좌표 정규화:");
    println!("  x_norm: {}, y_norm: {}", x_norm, y_norm);
    println!("  dist: {}, base_angle: {}", dist, base_angle);
    
    // base_pattern 확인
    let base_pattern_unclamped = r_value - dist * r_value + theta_value;
    let base_pattern = base_pattern_unclamped.clamp(0.0, 1.0);
    println!("Base pattern:");
    println!("  unclamped: {}, clamped: {}", base_pattern_unclamped, base_pattern);
    
    // 상태 비트 추출
    let state_bits = seed.hi & 0xFFFFF;
    let primary_hash = ((i * 31 + j) & 0x7) as u64;
    let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
    let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
    let detail_bits = (state_bits >> 16) & 0xF;
    
    println!("상태 정보:");
    println!("  primary_hash: {}, primary_state: {}", primary_hash, primary_state);
    println!("  secondary_state: {}, detail_bits: {}", secondary_state, detail_bits);
    
    // primary_value 계산
    let input_angle = base_angle + theta_value * 0.5;
    let primary_value = seed.compute_state_function(primary_state, input_angle, r_value);
    println!("Primary value:");
    println!("  input_angle: {}, primary_value: {}", input_angle, primary_value);
    
    // modulation_factor와 detail_factor
    let modulation_factor = match secondary_state {
        0 => 1.0,
        1 => 0.8 + 0.4 * (dist * 3.14159).sin(),
        2 => 1.0 - 0.3 * dist,
        3 => (1.0 + (base_angle * 2.0).cos()) * 0.5,
        _ => 1.0,
    };
    let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
    
    println!("변조 팩터:");
    println!("  modulation_factor: {}, detail_factor: {}", modulation_factor, detail_factor);
    
    // 최종 값 계산
    let final_value = seed.fused_forward(i, j, rows, cols);
    println!("최종 fused_forward 값: {}", final_value);
    
    // 해석적 미분 계산
    let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
    
    // 수치적 미분 계산
    let mut seed_plus = seed;
    let r_plus = r_value + eps;
    seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_plus = seed_plus.fused_forward(i, j, rows, cols);
    
    let mut seed_minus = seed;
    let r_minus = r_value - eps;
    seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_minus = seed_minus.fused_forward(i, j, rows, cols);
    
    let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
    
    println!("=== 미분 결과 비교 ===");
    println!("해석적 미분: {}", analytical_grad);
    println!("수치적 미분: {}", numerical_grad);
    println!("f_plus: {}, f_minus: {}", f_plus, f_minus);
    
    let relative_error = if numerical_grad.abs() > 1e-8 {
        ((analytical_grad - numerical_grad) / numerical_grad).abs()
    } else {
        (analytical_grad - numerical_grad).abs()
    };
    
    println!("상대오차: {:.6}%", relative_error * 100.0);
    
    // 각 단계별 미분 분석
    println!("\n=== 각 단계별 미분 분석 ===");
    
    // base_pattern 미분
    let base_pattern_grad_r = if base_pattern_unclamped > 0.0 && base_pattern_unclamped < 1.0 {
        1.0 - dist
    } else {
        0.0
    };
    println!("base_pattern 미분: {}", base_pattern_grad_r);
    
    // primary_value 미분
    let scaled_input = input_angle * r_value;
    let primary_value_grad_r = match primary_state {
        0 => input_angle * scaled_input.cos(),
        1 => -input_angle * scaled_input.sin(),
        2 => {
            let sech = 1.0 / scaled_input.cosh();
            input_angle * sech * sech
        },
        3 => {
            let cosh_val = scaled_input.cosh();
            let sech = 1.0 / cosh_val;
            let tanh_val = scaled_input.tanh();
            -2.0 * input_angle * sech * sech * tanh_val
        },
        4 => {
            let exp_val = (scaled_input * 0.1).exp().min(10.0);
            if exp_val < 10.0 {
                input_angle * 0.1 * exp_val
            } else {
                0.0
            }
        },
        5 => {
            let abs_val = scaled_input.abs() + 1e-6;
            input_angle * scaled_input.signum() / abs_val
        },
        6 => {
            let denom = scaled_input + 1e-6;
            -input_angle / (denom * denom)
        },
        7 => {
            input_angle / (1.0 + scaled_input * scaled_input)
        },
        _ => 0.0,
    };
    println!("primary_value 미분 (상태 {}): {}", primary_state, primary_value_grad_r);
    
    // Chain rule 최종 결과
    let abs_primary_grad = if primary_value >= 0.0 { 1.0 } else { -1.0 };
    let final_grad = 
        base_pattern_grad_r * primary_value.abs() * modulation_factor * detail_factor +
        base_pattern * abs_primary_grad * primary_value_grad_r * modulation_factor * detail_factor;
        
    println!("Chain rule 결과: {}", final_grad);
    println!("차이 (chain vs analytical): {}", (final_grad - analytical_grad).abs());
}

#[test]
fn 실패한_위치_1_1_분석() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 실제 테스트와 동일한 파라미터
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let i = 1; // 실패한 위치
    let j = 1; // 실패한 위치
    let eps = 1e-3;
    
    println!("=== 실패한 위치 (1,1) 상세 분석 ===");
    println!("R 파라미터: {}", r_value);
    println!("Theta 파라미터: {}", theta_value);
    println!("상태 비트 (hi): 0x{:X}", seed.hi);
    
    // 좌표 정규화
    let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
    let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
    let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
    let base_angle = y_norm.atan2(x_norm);
    
    println!("좌표 정규화:");
    println!("  x_norm: {}, y_norm: {}", x_norm, y_norm);
    println!("  dist: {}, base_angle: {}", dist, base_angle);
    
    // 상태 비트 추출
    let state_bits = seed.hi & 0xFFFFF;
    let primary_hash = ((i * 31 + j) & 0x7) as u64;
    let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
    let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
    let detail_bits = (state_bits >> 16) & 0xF;
    
    println!("상태 정보:");
    println!("  primary_hash: {}, primary_state: {}", primary_hash, primary_state);
    println!("  secondary_state: {}, detail_bits: {}", secondary_state, detail_bits);
    
    // fused_forward 값들 계산
    let actual_value = seed.fused_forward(i, j, rows, cols);
    println!("실제 fused_forward 값: {}", actual_value);
    
    // 해석적 미분
    let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
    
    // 수치적 미분
    let mut seed_plus = seed;
    let r_plus = r_value + eps;
    seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_plus = seed_plus.fused_forward(i, j, rows, cols);
    
    let mut seed_minus = seed;
    let r_minus = r_value - eps;
    seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_minus = seed_minus.fused_forward(i, j, rows, cols);
    
    let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
    
    println!("=== 미분 결과 비교 ===");
    println!("해석적 미분: {}", analytical_grad);
    println!("수치적 미분: {}", numerical_grad);
    println!("f_plus: {}, f_minus: {}", f_plus, f_minus);
    
    let relative_error = if numerical_grad.abs() > 1e-6 {
        ((analytical_grad - numerical_grad) / numerical_grad).abs()
    } else {
        (analytical_grad - numerical_grad).abs()
    };
    
    println!("상대오차: {:.6}%", relative_error * 100.0);
    
    // fused_forward 단계별 재계산으로 검증
    let base_pattern_unclamped = r_value - dist * r_value + theta_value;
    let base_pattern = base_pattern_unclamped.clamp(0.0, 1.0);
    
    let input_angle = base_angle + theta_value * 0.5;
    let primary_value = seed.compute_state_function(primary_state, input_angle, r_value);
    
    let modulation_factor = match secondary_state {
        0 => 1.0,
        1 => 0.8 + 0.4 * (dist * 3.14159).sin(),
        2 => 1.0 - 0.3 * dist,
        3 => (1.0 + (base_angle * 2.0).cos()) * 0.5,
        _ => 1.0,
    };
    let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
    
    let intermediate_value = base_pattern * primary_value.abs() * modulation_factor * detail_factor;
    let computed_value = intermediate_value.clamp(0.0, 1.0);
    
    println!("\n=== fused_forward 단계별 검증 ===");
    println!("base_pattern_unclamped: {}", base_pattern_unclamped);
    println!("base_pattern: {}", base_pattern);
    println!("input_angle: {}", input_angle);
    println!("primary_value: {}", primary_value);
    println!("modulation_factor: {}", modulation_factor);
    println!("detail_factor: {}", detail_factor);
    println!("intermediate_value: {}", intermediate_value);
    println!("computed_value: {}", computed_value);
    println!("actual_value: {}", actual_value);
    println!("차이: {}", (computed_value - actual_value).abs());
    
    // 세부 미분 단계 분석
    println!("\n=== 세부 미분 단계 분석 ===");
    let base_pattern_grad_r = if base_pattern_unclamped > 0.0 && base_pattern_unclamped < 1.0 {
        1.0 - dist
    } else {
        0.0
    };
    println!("base_pattern_grad_r: {}", base_pattern_grad_r);
    
    let scaled_input = input_angle * r_value;
    let primary_value_grad_r = match primary_state {
        0 => input_angle * scaled_input.cos(),
        1 => -input_angle * scaled_input.sin(),
        2 => {
            let sech = 1.0 / scaled_input.cosh();
            input_angle * sech * sech
        },
        3 => {
            let cosh_val = scaled_input.cosh();
            let sech = 1.0 / cosh_val;
            let tanh_val = scaled_input.tanh();
            -2.0 * input_angle * sech * sech * tanh_val
        },
        4 => {
            let exp_val = (scaled_input * 0.1).exp().min(10.0);
            if exp_val < 10.0 {
                input_angle * 0.1 * exp_val
            } else {
                0.0
            }
        },
        5 => {
            let abs_val = scaled_input.abs() + 1e-6;
            input_angle * scaled_input.signum() / abs_val
        },
        6 => {
            let denom = scaled_input + 1e-6;
            -input_angle / (denom * denom)
        },
        7 => {
            input_angle / (1.0 + scaled_input * scaled_input)
        },
        _ => 0.0,
    };
    println!("primary_value_grad_r (상태 {}): {}", primary_state, primary_value_grad_r);
    
    let abs_primary_grad = if primary_value >= 0.0 { 1.0 } else { -1.0 };
    let intermediate_grad_r = 
        base_pattern_grad_r * primary_value.abs() * modulation_factor * detail_factor +
        base_pattern * abs_primary_grad * primary_value_grad_r * modulation_factor * detail_factor;
    
    let final_clamp_grad = if intermediate_value > 0.0 && intermediate_value < 1.0 {
        1.0
    } else {
        0.0
    };
    
    let manual_gradient = final_clamp_grad * intermediate_grad_r;
    
    println!("abs_primary_grad: {}", abs_primary_grad);
    println!("intermediate_grad_r: {}", intermediate_grad_r);
    println!("final_clamp_grad: {}", final_clamp_grad);
    println!("manual_gradient: {}", manual_gradient);
    println!("차이 (manual vs analytical): {}", (manual_gradient - analytical_grad).abs());
}

#[test]
fn 실패한_위치_3_0_분석() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 실제 테스트와 동일한 파라미터
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let i = 3; // 실패한 위치
    let j = 0; // 실패한 위치
    let eps = 1e-3;
    
    println!("=== 실패한 위치 (3,0) 상세 분석 ===");
    println!("R 파라미터: {}", r_value);
    println!("Theta 파라미터: {}", theta_value);
    println!("상태 비트 (hi): 0x{:X}", seed.hi);
    
    // 좌표 정규화
    let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
    let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
    let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
    let base_angle = y_norm.atan2(x_norm);
    
    println!("좌표 정규화:");
    println!("  x_norm: {}, y_norm: {}", x_norm, y_norm);
    println!("  dist: {}, base_angle: {}", dist, base_angle);
    
    // 상태 비트 추출
    let state_bits = seed.hi & 0xFFFFF;
    let primary_hash = ((i * 31 + j) & 0x7) as u64;
    let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
    let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
    let detail_bits = (state_bits >> 16) & 0xF;
    
    println!("상태 정보:");
    println!("  primary_hash: {}, primary_state: {}", primary_hash, primary_state);
    println!("  secondary_state: {}, detail_bits: {}", secondary_state, detail_bits);
    
    // fused_forward 값 계산
    let actual_value = seed.fused_forward(i, j, rows, cols);
    println!("실제 fused_forward 값: {}", actual_value);
    
    // 해석적 미분
    let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
    
    // 수치적 미분
    let mut seed_plus = seed;
    let r_plus = r_value + eps;
    seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_plus = seed_plus.fused_forward(i, j, rows, cols);
    
    let mut seed_minus = seed;
    let r_minus = r_value - eps;
    seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    let f_minus = seed_minus.fused_forward(i, j, rows, cols);
    
    let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
    
    println!("=== 미분 결과 비교 ===");
    println!("해석적 미분: {}", analytical_grad);
    println!("수치적 미분: {}", numerical_grad);
    println!("f_plus: {}, f_minus: {}", f_plus, f_minus);
    
    let relative_error = if numerical_grad.abs() > 1e-6 {
        ((analytical_grad - numerical_grad) / numerical_grad).abs()
    } else {
        (analytical_grad - numerical_grad).abs()
    };
    
    println!("상대오차: {:.6}%", relative_error * 100.0);
    
    // fused_forward 단계별 재계산으로 검증
    let base_pattern_unclamped = r_value - dist * r_value + theta_value;
    let base_pattern = base_pattern_unclamped.clamp(0.0, 1.0);
    
    let input_angle = base_angle + theta_value * 0.5;
    let primary_value = seed.compute_state_function(primary_state, input_angle, r_value);
    
    let modulation_factor = match secondary_state {
        0 => 1.0,
        1 => 0.8 + 0.4 * (dist * 3.14159).sin(),
        2 => 1.0 - 0.3 * dist,
        3 => (1.0 + (base_angle * 2.0).cos()) * 0.5,
        _ => 1.0,
    };
    let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
    
    let computed_value = (base_pattern * primary_value.abs() * modulation_factor * detail_factor).clamp(0.0, 1.0);
    
    println!("\n=== fused_forward 단계별 검증 ===");
    println!("base_pattern: {}", base_pattern);
    println!("primary_value: {}", primary_value);
    println!("modulation_factor: {}", modulation_factor);
    println!("detail_factor: {}", detail_factor);
    println!("computed_value: {}", computed_value);
    println!("actual_value: {}", actual_value);
    println!("차이: {}", (computed_value - actual_value).abs());
} 