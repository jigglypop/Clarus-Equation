use crate::core::math::bessel::*;

#[test]
fn bessel_j0_기본_테스트() {
    // J0(0) = 1 확인
    let result = bessel_j0(0.0);
    assert!((result - 1.0).abs() < 0.1, "J0(0)는 1에 가까워야 함: {}", result);
}

#[test]
fn bessel_j0_값_범위_테스트() {
    // 다양한 입력에 대해 출력이 유한한지 확인
    let test_values = [0.0, 1.0, 5.0, 10.0, -1.0, -5.0];
    
    for &x in &test_values {
        let result = bessel_j0(x);
        assert!(result.is_finite(), "J0({})가 무한대임: {}", x, result);
        assert!(result.abs() <= 1.5, "J0({})의 절댓값이 너무 큼: {}", x, result);
    }
}

#[test]
fn cordic_함수_테스트() {
    let result = (1.0, 0.5);
    let special = 32;
    
    // CORDIC 함수들이 유한한 값을 반환하는지 확인
    let bessel_result = apply_bessel_cordic(result, special);
    assert!(bessel_result.is_finite(), "Bessel CORDIC 결과가 무한대임");
    
    let elliptic_result = apply_elliptic_cordic(result, special);
    assert!(elliptic_result.is_finite(), "Elliptic CORDIC 결과가 무한대임");
    
    let theta_result = apply_theta_cordic(result, special);
    assert!(theta_result.is_finite(), "Theta CORDIC 결과가 무한대임");
}

#[test]
fn cordic_특수값_테스트() {
    // 원점에서 테스트
    let zero_result = (0.0, 0.0);
    let special = 10;
    
    let bessel_zero = apply_bessel_cordic(zero_result, special);
    assert!((bessel_zero - 1.0).abs() < 0.1, "원점에서 Bessel J0는 1에 가까워야 함");
    
    let theta_zero = apply_theta_cordic(zero_result, special);
    assert!(theta_zero.abs() < 0.1, "원점에서 theta는 0에 가까워야 함");
} 