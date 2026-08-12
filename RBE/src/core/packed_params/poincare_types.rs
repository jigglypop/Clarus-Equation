//! 푸앵카레 볼 관련 타입들

use super::packed_types::Packed128;
use rand::Rng;

/// 행렬 압축 및 복원 (PoincareMatrix가 Packed128을 사용하도록 변경)
pub struct PoincareMatrix {
    pub seed: Packed128,
    pub rows: usize,
    pub cols: usize,
}

/// 푸앵카레 볼 128비트 인코딩 (1장 문서 정확한 설계)
/// hi: 푸앵카레 상태 코어, lo: 연속 파라미터 코어
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincarePackedBit128 {
    /// 푸앵카레 상태 코어 (64비트)
    /// [63:62] poincare_quadrant (2비트)
    /// [61:50] hyperbolic_frequency (12비트) 
    /// [49:38] geodesic_amplitude (12비트)
    /// [37:32] basis_function_selector (6비트)
    /// [31:0]  cordic_rotation_sequence (32비트)
    pub hi: u64,
    
    /// 연속 파라미터 코어 (64비트)
    /// [63:32] r_poincare (32비트 IEEE754 float)
    /// [31:0]  theta_poincare (32비트 IEEE754 float)
    pub lo: u64,
}

/// 푸앵카레 사분면 열거형 (2비트 인코딩)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoincareQuadrant {
    First,  // 00: sinh(x) - 양의 지수 증가
    Second, // 01: cosh(x) - 대칭적 지수 증가  
    Third,  // 10: tanh(x) - S자형 포화 함수
    Fourth, // 11: sech²(x) - 종 모양 함수
}

impl PoincarePackedBit128 {
    /// 새로운 푸앵카레 볼 인코딩 생성
    pub fn new(
        quadrant: PoincareQuadrant,
        frequency: u16,  // 12비트
        amplitude: u16,  // 12비트  
        basis_func: u8,  // 6비트
        cordic_seq: u32, // 32비트
        r_poincare: f32, // [0, 1)
        theta_poincare: f32, // [0, 2π]
    ) -> Self {
        let hi = Self::encode_hi_field(quadrant, frequency, amplitude, basis_func, cordic_seq);
        let lo = Self::encode_lo_field(r_poincare, theta_poincare);
        
        Self { hi, lo }
    }
    
    /// hi 필드 인코딩 (푸앵카레 상태 코어)
    fn encode_hi_field(
        quadrant: PoincareQuadrant,
        frequency: u16,
        amplitude: u16, 
        basis_func: u8,
        cordic_seq: u32,
    ) -> u64 {
        let quadrant_bits = quadrant as u64;
        let freq_bits = (frequency as u64) & 0xFFF; // 12비트 마스킹
        let amp_bits = (amplitude as u64) & 0xFFF;  // 12비트 마스킹
        let basis_bits = (basis_func as u64) & 0x3F; // 6비트 마스킹
        let cordic_bits = cordic_seq as u64;
        
        (quadrant_bits << 62) | (freq_bits << 50) | (amp_bits << 38) | 
        (basis_bits << 32) | cordic_bits
    }
    
    /// lo 필드 인코딩 (연속 파라미터 코어)
    fn encode_lo_field(r: f32, theta: f32) -> u64 {
        let r_bits = r.to_bits() as u64;
        let theta_bits = theta.to_bits() as u64;
        (r_bits << 32) | theta_bits
    }
    
    /// 사분면 추출
    pub fn get_quadrant(&self) -> PoincareQuadrant {
        match (self.hi >> 62) & 0x3 {
            0 => PoincareQuadrant::First,
            1 => PoincareQuadrant::Second,
            2 => PoincareQuadrant::Third,
            _ => PoincareQuadrant::Fourth,
        }
    }
    
    /// 주파수 추출
    pub fn get_hyperbolic_frequency(&self) -> u16 {
        ((self.hi >> 50) & 0xFFF) as u16
    }
    
    /// 진폭 추출
    pub fn get_geodesic_amplitude(&self) -> u16 {
        ((self.hi >> 38) & 0xFFF) as u16
    }
    
    /// 기저함수 선택자 추출
    pub fn get_basis_function_selector(&self) -> u8 {
        ((self.hi >> 32) & 0x3F) as u8
    }
    
    /// CORDIC 회전 시퀀스 추출
    pub fn get_cordic_rotation_sequence(&self) -> u32 {
        (self.hi & 0xFFFFFFFF) as u32
    }
    
    /// r 푸앵카레 추출
    pub fn get_r_poincare(&self) -> f32 {
        f32::from_bits((self.lo >> 32) as u32)
    }
    
    /// theta 푸앵카레 추출
    pub fn get_theta_poincare(&self) -> f32 {
        f32::from_bits(self.lo as u32)
    }
    
    /// 무작위 생성
    pub fn random(rng: &mut impl Rng) -> Self {
        let quadrant = match rng.gen::<u8>() % 4 {
            0 => PoincareQuadrant::First,
            1 => PoincareQuadrant::Second,
            2 => PoincareQuadrant::Third,
            _ => PoincareQuadrant::Fourth,
        };
        
        let frequency = rng.gen::<u16>() & 0xFFF;
        let amplitude = rng.gen::<u16>() & 0xFFF;
        let basis_func = rng.gen::<u8>() & 0x3F;
        let cordic_seq = rng.gen::<u32>();
        let r_poincare = rng.gen::<f32>() * 0.99; // [0, 0.99)
        let theta_poincare = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        
        Self::new(quadrant, frequency, amplitude, basis_func, cordic_seq, r_poincare, theta_poincare)
    }
} 