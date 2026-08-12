/// Bessel J0 함수 계산
pub fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 
            + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164; // PI/4
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (std::f32::consts::PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

// CORDIC 특수 함수 헬퍼 (CORDIC.md 기반)
// 이 함수들은 현재 compute_weight에서 직접 사용되지는 않지만,
// 향후 CORDIC 알고리즘 확장을 위해 남겨둡니다.
pub fn apply_bessel_cordic(result: (f32, f32), special: u16) -> f32 {
    let r = (result.0 * result.0 + result.1 * result.1).sqrt();
    bessel_j0(r * (special as f32 / 63.0))
}

pub fn apply_elliptic_cordic(result: (f32, f32), special: u16) -> f32 {
    let angle = result.1.atan2(result.0);
    (angle * (special as f32 / 63.0 - 1.0)).tanh()
}

pub fn apply_theta_cordic(result: (f32, f32), special: u16) -> f32 {
    let theta = result.1.atan2(result.0);
    theta * (special as f32 / 63.0)
} 