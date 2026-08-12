use crate::core::math::gradient::AnalyticalGradient;
use crate::core::packed_params::Packed128;
use rand::{thread_rng};

#[test]
fn eps_최적화_테스트() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let i = 2;
    let j = 2;
    
    // 해석적 미분
    let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
    
    println!("=== EPS 최적화 테스트 ===");
    println!("해석적 미분: {}", analytical_grad);
    
    // 다양한 eps 값 테스트
    let eps_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
    
    for &eps in &eps_values {
        let mut seed_plus = seed;
        let r_plus = r_value + eps;
        seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
        let f_plus = seed_plus.fused_forward(i, j, rows, cols);
        
        let mut seed_minus = seed;
        let r_minus = r_value - eps;
        seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
        let f_minus = seed_minus.fused_forward(i, j, rows, cols);
        
        let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
        
        let relative_error = if numerical_grad.abs() > 1e-8 {
            ((analytical_grad - numerical_grad) / numerical_grad).abs()
        } else {
            (analytical_grad - numerical_grad).abs()
        };
        
        println!("eps={:.0e}: 수치적={:.8}, 상대오차={:.4}%", 
                 eps, numerical_grad, relative_error * 100.0);
    }
}

#[test]
fn 고정밀_해석적_미분_검증() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let eps = 1e-3; // 최적화된 eps 값
    
    let mut passed = 0;
    let mut total = 0;
    let mut max_error = 0.0f32;
    
    println!("=== 고정밀 해석적 미분 검증 (eps={:.0e}) ===", eps);
    
    for i in 0..rows {
        for j in 0..cols {
            let analytical_grad_r = seed.analytical_gradient_r(i, j, rows, cols);
            let analytical_grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            
            // R 파라미터 검증
            let mut seed_plus = seed;
            let r_plus = r_value + eps;
            seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_plus = seed_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_minus = seed;
            let r_minus = r_value - eps;
            seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_minus = seed_minus.fused_forward(i, j, rows, cols);
            
            let numerical_grad_r = (f_plus - f_minus) / (2.0 * eps);
            
            total += 1;
            let relative_error_r = if numerical_grad_r.abs() > 1e-8 {
                ((analytical_grad_r - numerical_grad_r) / numerical_grad_r).abs()
            } else {
                (analytical_grad_r - numerical_grad_r).abs()
            };
            
            max_error = max_error.max(relative_error_r);
            
            if relative_error_r < 0.02 { // 2% 허용 오차
                passed += 1;
            }
            
            // Theta 파라미터 검증
            let mut seed_plus_theta = seed;
            let theta_plus = theta_value + eps;
            seed_plus_theta.lo = ((r_value.to_bits() as u64) << 32) | theta_plus.to_bits() as u64;
            let f_plus_theta = seed_plus_theta.fused_forward(i, j, rows, cols);
            
            let mut seed_minus_theta = seed;
            let theta_minus = theta_value - eps;
            seed_minus_theta.lo = ((r_value.to_bits() as u64) << 32) | theta_minus.to_bits() as u64;
            let f_minus_theta = seed_minus_theta.fused_forward(i, j, rows, cols);
            
            let numerical_grad_theta = (f_plus_theta - f_minus_theta) / (2.0 * eps);
            
            total += 1;
            let relative_error_theta = if numerical_grad_theta.abs() > 1e-8 {
                ((analytical_grad_theta - numerical_grad_theta) / numerical_grad_theta).abs()
            } else {
                (analytical_grad_theta - numerical_grad_theta).abs()
            };
            
            max_error = max_error.max(relative_error_theta);
            
            if relative_error_theta < 0.02 { // 2% 허용 오차
                passed += 1;
            }
        }
    }
    
    let pass_rate = passed as f32 / total as f32;
    println!("통과율: {:.1}% ({}/{})", pass_rate * 100.0, passed, total);
    println!("최대 상대오차: {:.3}%", max_error * 100.0);
    
    // 98% 이상 통과하면 충분히 정확함
    assert!(pass_rate >= 0.98, "고정밀 검증 실패: 통과율 {:.1}%", pass_rate * 100.0);
} 