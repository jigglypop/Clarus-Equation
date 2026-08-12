#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn fast_rsqrt8_ps(x: __m256) -> __m256 {
    // 첫 단계: rsqrt 근사
    let approx = _mm256_rsqrt_ps(x);
    // 뉴턴–랩슨 두 번으로 정밀도 향상 (1/sqrt)
    // y = y * (1.5 - 0.5 * x * y^2)
    let half = _mm256_set1_ps(0.5);
    let three = _mm256_set1_ps(3.0);
    let mut y = approx;
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let prod = _mm256_mul_ps(x, y2);
        let half_prod = _mm256_mul_ps(half, prod);
        let three_minus = _mm256_sub_ps(three, half_prod);
        y = _mm256_mul_ps(y, three_minus);
    }
    y
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub unsafe fn adam_update_avx2(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    epsilon: f32,
    t: i32,
) {
    let len = params.len();
    let mut idx = 0;
    let beta1_vec = _mm256_set1_ps(beta1);
    let beta1_comp = _mm256_set1_ps(1.0 - beta1);
    let beta2_vec = _mm256_set1_ps(beta2);
    let beta2_comp = _mm256_set1_ps(1.0 - beta2);
    let lr_vec = _mm256_set1_ps(lr);
    let eps_vec = _mm256_set1_ps(epsilon);

    // 편향 보정 계수 (스칼라) – 소량 비용
    let beta1_pow = beta1.powi(t);
    let beta2_pow = beta2.powi(t);
    let one = 1.0;
    let m_corr_den = one - beta1_pow;
    let v_corr_den = one - beta2_pow;
    let m_corr_den_vec = _mm256_set1_ps(m_corr_den);
    let v_corr_den_vec = _mm256_set1_ps(v_corr_den);

    while idx + 8 <= len {
        // load
        let p = _mm256_loadu_ps(params[idx..].as_ptr());
        let g = _mm256_loadu_ps(grads[idx..].as_ptr());
        let m_old = _mm256_loadu_ps(m[idx..].as_ptr());
        let v_old = _mm256_loadu_ps(v[idx..].as_ptr());

        // m, v 업데이트
        let m_new = _mm256_add_ps(_mm256_mul_ps(beta1_vec, m_old), _mm256_mul_ps(beta1_comp, g));
        let g2 = _mm256_mul_ps(g, g);
        let v_new = _mm256_add_ps(_mm256_mul_ps(beta2_vec, v_old), _mm256_mul_ps(beta2_comp, g2));

        // 편향 보정
        let m_hat = _mm256_div_ps(m_new, m_corr_den_vec);
        let v_hat = _mm256_div_ps(v_new, v_corr_den_vec);

        // denom = sqrt(v_hat) + eps (직접 sqrt 사용)
        let v_sqrt = _mm256_sqrt_ps(v_hat);
        let denom = _mm256_add_ps(v_sqrt, eps_vec);
        let raw_update = _mm256_mul_ps(_mm256_div_ps(_mm256_mul_ps(lr_vec, m_hat), denom), _mm256_set1_ps(-1.0));
        
        // 그래디언트 클리핑 (±5.0)
        let clip_max = _mm256_set1_ps(5.0);
        let clip_min = _mm256_set1_ps(-5.0);
        let clipped_update = _mm256_min_ps(_mm256_max_ps(raw_update, clip_min), clip_max);

        // 파라미터 업데이트
        let p_new = _mm256_add_ps(p, clipped_update);

        // store back
        _mm256_storeu_ps(params[idx..].as_mut_ptr(), p_new);
        _mm256_storeu_ps(m[idx..].as_mut_ptr(), m_new);
        _mm256_storeu_ps(v[idx..].as_mut_ptr(), v_new);

        idx += 8;
    }
    // 남은 요소 스칼라 처리
    for i in idx..len {
        // 동일 로직
        let g = grads[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / (one - beta1_pow);
        let v_hat = v[i] / (one - beta2_pow);
        let denom = v_hat.sqrt() + epsilon;
        let raw_update = -lr * m_hat / denom;
        // 클리핑
        let clipped_update = if raw_update.abs() > 5.0 {
            5.0 * raw_update.signum()
        } else {
            raw_update
        };
        params[i] += clipped_update;
    }
} 