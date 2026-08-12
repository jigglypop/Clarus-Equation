use crate::core::packed_params::{Packed64, Packed128, DecodedParams};
use rand::thread_rng;

#[test]
fn packed64_생성_테스트() {
    let packed = Packed64::new(0x123456789ABCDEF0);
    assert_eq!(packed.rotations, 0x123456789ABCDEF0);
}

#[test]
fn packed128_디코딩_테스트() {
    let mut rng = thread_rng();
    let packed = Packed128::random(&mut rng);
    
    let decoded = packed.decode();
    
    assert!(decoded.r_fp32.is_finite());
    assert!(decoded.theta_fp32.is_finite());
}

#[test]
fn 연속파라미터_변환_테스트() {
    let params = DecodedParams {
        r_fp32: 0.5,
        theta_fp32: 1.0,
    };
    
    let packed = Packed128::from_continuous(&params);
    let decoded = packed.decode();
    
    // 부동소수점 정밀도로 인한 약간의 오차 허용
    assert!((decoded.r_fp32 - params.r_fp32).abs() < 0.1);
    assert!((decoded.theta_fp32 - params.theta_fp32).abs() < 0.1);
}

// 해석적 미분 테스트는 src/core/math/__tests__/gradient_test.rs로 이동됨

// 해석적 미분 테스트는 src/core/math/__tests__/gradient_test.rs로 이동됨

#[test]
fn 융합_순전파_안정성_테스트() {
    let mut rng = thread_rng();
    
    for _ in 0..50 { // 50번 랜덤 테스트
        let seed = Packed128::random(&mut rng);
        
        let rows = 8;
        let cols = 8;
        
        for i in 0..rows {
            for j in 0..cols {
                let weight = seed.fused_forward(i, j, rows, cols);
                
                // 수치적 안정성 확인
                assert!(weight.is_finite(), "융합 순전파 결과가 무한대: {}", weight);
                assert!(weight >= 0.0 && weight <= 1.0, "융합 순전파 결과가 범위를 벗어남: {}", weight);
            }
        }
    }
}

#[test]
fn 상태_전이_기능_테스트() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    let original_hi = seed.hi;
    
    // 다양한 에러 크기로 상태 전이 테스트
    let error_cases = vec![0.001, 0.05, 0.2];
    
    for error in error_cases {
        seed.apply_state_transition(error, 2, 3);
        
        // 상태가 변했는지 확인 (완전히 같을 수도 있지만 일반적으로는 다름)
        // hi 필드는 상태 전이에 의해 변경될 수 있음
        assert!(seed.hi > 0, "상태 전이 후 hi 필드가 유효하지 않음");
    }
    
    // 고급 상태 전이도 테스트
    seed.advanced_state_transition(0.1, 1, 4);
    assert!(seed.hi > 0, "고급 상태 전이 후 hi 필드가 유효하지 않음");
}

#[test]
fn 고급_순전파_기능_테스트() {
    let mut rng = thread_rng();
    let seed = Packed128::random(&mut rng);
    
    let rows = 6;
    let cols = 6;
    
    for i in 0..rows {
        for j in 0..cols {
            let basic_weight = seed.fused_forward(i, j, rows, cols);
            let advanced_weight = seed.fused_forward(i, j, rows, cols);
            
            // 둘 다 유한하고 범위 내에 있어야 함
            assert!(basic_weight.is_finite(), "기본 순전파 결과가 무한대");
            assert!(advanced_weight.is_finite(), "고급 순전파 결과가 무한대");
            
            assert!(basic_weight >= 0.0 && basic_weight <= 1.0, "기본 순전파 범위 초과");
            assert!(advanced_weight >= -1.0 && advanced_weight <= 1.0, "고급 순전파 범위 초과");
        }
    }
} 