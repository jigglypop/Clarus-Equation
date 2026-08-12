use crate::packed_params::{Packed64, Packed128, DecodedParams};

/// RMSE (Root Mean Square Error)를 계산합니다. (이름 변경)
pub fn compute_full_rmse(matrix: &[f32], seed: &Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let original_value = matrix[i * cols + j];
            let reconstructed_value = seed.compute_weight(i, j, rows, cols);
            error += (original_value - reconstructed_value).powi(2);
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

/// 해석적 그래디언트 계산 (r, theta) -> 수치적 미분으로 변경
/// `p` 파라미터의 r, theta 값 주변에서 f(x) = compute_weight의 변화율을 계산합니다.
pub fn analytic_grad(p: &DecodedParams, i: usize, j: usize, rows: usize, cols: usize) -> (f32, f32) {
    let eps = 1e-4;

    // r에 대한 그래디언트 계산
    let mut p_r_plus = p.clone();
    p_r_plus.r_fp32 += eps;
    let seed_r_plus = Packed128::from_continuous(&p_r_plus);
    let weight_r_plus = Packed64 { rotations: seed_r_plus.hi }.compute_weight(i, j, rows, cols);

    let mut p_r_minus = p.clone();
    p_r_minus.r_fp32 -= eps;
    let seed_r_minus = Packed128::from_continuous(&p_r_minus);
    let weight_r_minus = Packed64 { rotations: seed_r_minus.hi }.compute_weight(i, j, rows, cols);

    let dr = (weight_r_plus - weight_r_minus) / (2.0 * eps);

    // theta에 대한 그래디언트 계산
    let mut p_th_plus = p.clone();
    p_th_plus.theta_fp32 += eps;
    let seed_th_plus = Packed128::from_continuous(&p_th_plus);
    let weight_th_plus = Packed64 { rotations: seed_th_plus.hi }.compute_weight(i, j, rows, cols);

    let mut p_th_minus = p.clone();
    p_th_minus.theta_fp32 -= eps;
    let seed_th_minus = Packed128::from_continuous(&p_th_minus);
    let weight_th_minus = Packed64 { rotations: seed_th_minus.hi }.compute_weight(i, j, rows, cols);

    let dth = (weight_th_plus - weight_th_minus) / (2.0 * eps);

    (dr, dth)
}

/// Adam 옵티마이저 업데이트
#[inline]
pub fn adam_update(p:&mut f32, m:&mut f32, v:&mut f32, g:f32, lr:f32, t:i32){
    const B1:f32=0.9; const B2:f32=0.999; const EPS:f32=1e-8;
    *m = B1*(*m)+(1.0-B1)*g;
    *v = B2*(*v)+(1.0-B2)*g*g;
    let m_hat=*m/(1.0-B1.powi(t));
    let v_hat=*v/(1.0-B2.powi(t));
    *p -= lr*m_hat/(v_hat.sqrt()+EPS);
}

/// Float to Q-format (STE placeholder)
pub fn ste_quant_q0x(val: f32, bits: u8) -> u64 {
    (val.clamp(0.0, 1.0) * ((1u64 << bits) - 1) as f32) as u64
}

/// Float to Phase (STE placeholder)
pub fn ste_quant_phase(val: f32, bits: u8) -> u64 {
    (val.rem_euclid(2.0 * std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * ((1u64 << bits) - 1) as f32) as u64
}

/// 유전 알고리즘의 변이(mutation) 연산을 수행합니다.
///
/// 주어진 시드의 각 비트를 `mutation_rate` 확률로 뒤집습니다.
///
/// # Arguments
/// * `seed` - 변이를 적용할 원본 Packed64 시드
/// * `mutation_rate` - 각 비트가 변이될 확률 (0.0 ~ 1.0)
///
/// # Returns
/// * 변이된 새로운 Packed64 시드
pub fn mutate_seed(seed: Packed64, mutation_rate: f32) -> Packed64 {
    use crate::packed_params::Packed64;
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let mut new_rotations = seed.rotations;
    for i in 0..64 {
        if rng.gen_bool(mutation_rate as f64) {
            new_rotations ^= 1 << i;
        }
    }
    Packed64::new(new_rotations)
} 