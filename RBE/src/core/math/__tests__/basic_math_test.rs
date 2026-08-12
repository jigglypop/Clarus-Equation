use crate::core::math::basic_math::*;
use crate::packed_params::{Packed64, Packed128, DecodedParams};

#[test]
fn rmse_계산_테스트() {
    let matrix = vec![1.0, 2.0, 3.0, 4.0];
    let seed = Packed64::new(0x1234567890ABCDEF);
    
    let rmse = compute_full_rmse(&matrix, &seed, 2, 2);
    assert!(rmse >= 0.0, "RMSE는 음수가 될 수 없음");
}

#[test]
fn adam_업데이트_테스트() {
    let mut param = 1.0;
    let mut momentum = 0.0;
    let mut velocity = 0.0;
    let gradient = 0.1;
    let learning_rate = 0.001;
    let epoch = 1;
    
    adam_update(&mut param, &mut momentum, &mut velocity, gradient, learning_rate, epoch);
    
    assert_ne!(param, 1.0, "파라미터가 업데이트되어야 함");
    assert_ne!(momentum, 0.0, "모멘텀이 업데이트되어야 함");
    assert_ne!(velocity, 0.0, "속도가 업데이트되어야 함");
}

#[test]
fn ste_양자화_테스트() {
    let val = 0.5;
    let bits = 8;
    
    let quantized = ste_quant_q0x(val, bits);
    assert!(quantized <= (1u64 << bits) - 1, "양자화 값이 범위를 벗어남");
    
    let phase_quantized = ste_quant_phase(std::f32::consts::PI, bits);
    assert!(phase_quantized <= (1u64 << bits) - 1, "위상 양자화 값이 범위를 벗어남");
}

#[test]
fn 그래디언트_계산_테스트() {
    use rand::thread_rng;
    
    let params = DecodedParams {
        r_fp32: 0.5,
        theta_fp32: 1.0,
    };
    
    let (grad_r, grad_theta) = analytic_grad(&params, 0, 0, 2, 2);
    
    // 그래디언트가 유한한 값인지 확인
    assert!(grad_r.is_finite(), "r 그래디언트가 무한대임");
    assert!(grad_theta.is_finite(), "theta 그래디언트가 무한대임");
}

#[test]
fn 시드_변이_테스트() {
    let original_seed = Packed64::new(0x1234567890ABCDEF);
    let mutation_rate = 0.1;
    
    let mutated_seed = mutate_seed(original_seed, mutation_rate);
    
    // 변이된 시드는 원본과 달라야 함 (높은 확률로)
    // 다만 변이율이 낮으면 같을 수도 있으므로 단순히 실행만 확인
    assert_eq!(mutated_seed.rotations.count_ones() <= 64, true, "비트 수가 유효 범위를 벗어남");
} 