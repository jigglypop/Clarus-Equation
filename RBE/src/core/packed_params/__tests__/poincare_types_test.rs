use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use rand::thread_rng;

#[test]
fn 푸앵카레_패킹_기본_테스트() {
    println!("=== 푸앵카레 패킹 기본 테스트 ===");
    
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::First,
        1024,  // frequency
        2048,  // amplitude  
        16,    // basis_func
        0x12345678,  // cordic_rotation_sequence
        0.7,   // r_poincare
        0.3    // theta_poincare
    );
    
    // 기본 생성 확인
    assert_eq!(packed.get_quadrant(), PoincareQuadrant::First);
    assert_eq!(packed.get_hyperbolic_frequency(), 1024);
    assert_eq!(packed.get_geodesic_amplitude(), 2048);
    assert_eq!(packed.get_cordic_rotation_sequence(), 0x12345678);
    
    println!("기본 푸앵카레 패킹 테스트 통과!");
}

#[test]
fn 다양한_사분면_테스트() {
    println!("=== 다양한 사분면 테스트 ===");
    
    let quadrants = [
        PoincareQuadrant::First,
        PoincareQuadrant::Second,
        PoincareQuadrant::Third,
        PoincareQuadrant::Fourth,
    ];
    
    for (i, &quadrant) in quadrants.iter().enumerate() {
        let packed = PoincarePackedBit128::new(
            quadrant,
            (1024 + i * 512) as u16,
            (2048 + i * 256) as u16,
            (8 + i * 4) as u8,
            0x11111111 + i as u32 * 0x11111111,
            0.4 + i as f32 * 0.1,
            0.2 + i as f32 * 0.05
        );
        
        assert_eq!(packed.get_quadrant(), quadrant);
        println!("사분면 {:?} 테스트 통과", quadrant);
    }
    
    println!("다양한 사분면 테스트 통과!");
}

#[test]
fn 무작위_생성_테스트() {
    println!("=== 무작위 생성 테스트 ===");
    
    let mut rng = thread_rng();
    
    for _ in 0..10 {
        let packed = PoincarePackedBit128::random(&mut rng);
        
        // 생성된 값들이 유효한 범위에 있는지 확인
        let freq = packed.get_hyperbolic_frequency();
        let amp = packed.get_geodesic_amplitude();
        let seq = packed.get_cordic_rotation_sequence();
        
        assert!(freq > 0, "주파수가 0");
        assert!(amp > 0, "진폭이 0");
        assert!(seq > 0, "CORDIC 시퀀스가 0");
        
        println!("무작위 생성: freq={}, amp={}, seq=0x{:08x}", freq, amp, seq);
    }
    
    println!("무작위 생성 테스트 통과!");
}

#[test]
fn 유효성_검증_테스트() {
    println!("=== 유효성 검증 테스트 ===");
    
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::Third,
        2048,
        1024,
        32,
        0xDEADBEEF,
        0.5,
        1.0
    );
    
    // 푸앵카레 볼 유효성 검증 (기본 검증)
    assert_eq!(packed.get_quadrant(), PoincareQuadrant::Third);
    assert!(packed.get_hyperbolic_frequency() > 0, "주파수가 0");
    
    println!("유효성 검증 테스트 통과!");
}

#[test]
fn 재현성_테스트() {
    println!("=== 재현성 테스트 ===");
    
    // 동일한 파라미터로 두 번 생성
    let params = (PoincareQuadrant::Second, 1500, 3000, 20, 0x87654321, 0.8, 1.2);
    
    let packed1 = PoincarePackedBit128::new(
        params.0, params.1, params.2, params.3, params.4, params.5, params.6
    );
    let packed2 = PoincarePackedBit128::new(
        params.0, params.1, params.2, params.3, params.4, params.5, params.6
    );
    
    // 완전히 동일해야 함
    assert_eq!(packed1.get_quadrant(), packed2.get_quadrant());
    assert_eq!(packed1.get_hyperbolic_frequency(), packed2.get_hyperbolic_frequency());
    assert_eq!(packed1.get_geodesic_amplitude(), packed2.get_geodesic_amplitude());
    assert_eq!(packed1.get_cordic_rotation_sequence(), packed2.get_cordic_rotation_sequence());
    
    println!("재현성 테스트 통과!");
}

#[test]
fn 경계값_테스트() {
    println!("=== 경계값 테스트 ===");
    
    // 최소값 테스트
    let min_packed = PoincarePackedBit128::new(
        PoincareQuadrant::First,
        1,      // 최소 주파수
        1,      // 최소 진폭
        0,      // 최소 기저 함수
        1,      // 최소 CORDIC 시퀀스
        0.01,   // 최소 r
        -10.0   // 최소 theta
    );
    
    assert_eq!(min_packed.get_quadrant(), PoincareQuadrant::First);
    assert!(min_packed.get_hyperbolic_frequency() > 0, "최소값 주파수가 0");
    
    // 최대값 테스트
    let max_packed = PoincarePackedBit128::new(
        PoincareQuadrant::Fourth,
        4095,       // 최대 주파수 (12비트)
        4095,       // 최대 진폭 (12비트)  
        63,         // 최대 기저 함수 (6비트)
        0xFFFFFFFF, // 최대 CORDIC 시퀀스
        0.99,       // 최대 r
        10.0        // 최대 theta
    );
    
    assert_eq!(max_packed.get_quadrant(), PoincareQuadrant::Fourth);
    assert!(max_packed.get_hyperbolic_frequency() > 0, "최대값 주파수가 0");
    
    println!("경계값 테스트 통과!");
} 