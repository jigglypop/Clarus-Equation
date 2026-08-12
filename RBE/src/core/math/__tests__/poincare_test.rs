use crate::core::math::poincare::*;
use std::f32::consts::PI;

#[test]
fn poincare_점_생성_테스트() {
    let point = PoincareBallPoint::new(0.5, PI / 4.0);
    assert_eq!(point.r, 0.5);
    assert_eq!(point.theta, PI / 4.0);
    
    // 경계값 클램핑 확인
    let boundary_point = PoincareBallPoint::new(1.5, 0.0);
    assert!(boundary_point.r < 1.0, "반지름이 1 미만으로 클램핑되어야 함");
}

#[test]
fn 원점_테스트() {
    let origin = PoincareBallPoint::origin();
    assert_eq!(origin.r, 0.0);
    assert_eq!(origin.theta, 0.0);
}

#[test]
fn 좌표_변환_테스트() {
    let point = PoincareBallPoint::new(0.5, PI / 2.0);
    let (x, y) = point.to_cartesian();
    
    // π/2에서는 x≈0, y≈0.5
    assert!((x).abs() < 0.01, "x 좌표가 0에 가까워야 함: {}", x);
    assert!((y - 0.5).abs() < 0.01, "y 좌표가 0.5에 가까워야 함: {}", y);
    
    // 역변환 테스트
    let converted_back = PoincareBallPoint::from_cartesian(x, y);
    assert!((converted_back.r - point.r).abs() < 0.01, "반지름 역변환 오류");
}

#[test]
fn 경계_거리_테스트() {
    let point = PoincareBallPoint::new(0.8, 0.0);
    let distance = point.distance_to_boundary();
    
    assert!((distance - 0.2).abs() < 0.01, "경계 거리 계산 오류: {}", distance);
}

#[test]
fn 리만_메트릭_테스트() {
    let point = PoincareBallPoint::new(0.5, 0.0);
    
    let metric_factor = RiemannianGeometry::metric_factor(&point);
    assert!(metric_factor > 0.0, "메트릭 인수는 양수여야 함");
    assert!(metric_factor.is_finite(), "메트릭 인수가 무한대임");
    
    let inverse_metric = RiemannianGeometry::inverse_metric_factor(&point);
    assert!(inverse_metric > 0.0, "역메트릭 인수는 양수여야 함");
    assert!(inverse_metric.is_finite(), "역메트릭 인수가 무한대임");
}

#[test]
fn mobius_덧셈_테스트() {
    let p1 = PoincareBallPoint::new(0.3, 0.0);
    let p2 = PoincareBallPoint::new(0.2, PI / 2.0);
    
    let result = RiemannianGeometry::mobius_addition(&p1, &p2);
    
    assert!(result.r < 1.0, "Möbius 덧셈 결과가 푸앵카레 볼 내부에 있어야 함");
    assert!(result.r.is_finite(), "결과 반지름이 무한대임");
    assert!(result.theta.is_finite(), "결과 각도가 무한대임");
}

#[test]
fn 스칼라_곱셈_테스트() {
    let point = PoincareBallPoint::new(0.4, PI / 3.0);
    let scalar = 2.0;
    
    let result = RiemannianGeometry::scalar_multiplication(scalar, &point);
    
    assert!(result.r < 1.0, "스칼라 곱셈 결과가 푸앵카레 볼 내부에 있어야 함");
    assert!(result.r.is_finite(), "결과가 유한해야 함");
}

#[test]
fn 지수_사상_테스트() {
    let base = PoincareBallPoint::new(0.2, 0.0);
    let tangent = PoincareBallPoint::new(0.1, PI / 4.0);
    
    let result = RiemannianGeometry::exponential_map(&base, &tangent);
    
    assert!(result.r < 1.0, "지수 사상 결과가 푸앵카레 볼 내부에 있어야 함");
    assert!(result.r.is_finite(), "결과가 유한해야 함");
}

#[test]
fn 쌍곡_거리_테스트() {
    let p1 = PoincareBallPoint::new(0.3, 0.0);
    let p2 = PoincareBallPoint::new(0.4, PI);
    
    let distance = RiemannianGeometry::hyperbolic_distance(&p1, &p2);
    
    assert!(distance >= 0.0, "거리는 음수가 될 수 없음");
    assert!(distance.is_finite(), "거리가 무한대임");
    
    // 같은 점까지의 거리는 0
    let same_distance = RiemannianGeometry::hyperbolic_distance(&p1, &p1);
    assert!(same_distance.abs() < 0.01, "같은 점까지의 거리는 0이어야 함");
} 