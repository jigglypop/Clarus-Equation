//! 패킹된 파라미터 타입들

use rand::Rng;
use crate::math::{ste_quant_q0x, ste_quant_phase};

/// 64-bit Packed Poincaré 시드 표현 (CORDIC 통합)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64 {
    pub rotations: u64,  // CORDIC 회전 시퀀스
}

impl Packed64 {
    pub fn new(rotations: u64) -> Self {
        Packed64 { rotations }
    }

    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. hi 비트필드에서 r, theta 디코딩
        let r_quant = (self.rotations >> 44) & 0xFFFFF; // 20 bits
        let theta_quant = (self.rotations >> 20) & 0xFFFFFF; // 24 bits
        
        let r_val = r_quant as f32 / ((1u64 << 20) - 1) as f32; // [0, 1] 범위로 정규화
        let theta_val = (theta_quant as f32 / ((1u64 << 24) - 1) as f32) * 2.0 * std::f32::consts::PI; // [0, 2PI] 범위로 정규화

        let rotations = self.rotations;

        // 2. 좌표 기반 초기 각도 계산
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        let base_angle = y_norm.atan2(x_norm);
        
        // 3. r, theta를 적용하여 초기 벡터 (x, y) 설정
        let mut x = r_val * (base_angle + theta_val).cos();
        let mut y = r_val * (base_angle + theta_val).sin();

        for k in 0..20 { // CORDIC 반복 횟수를 r, theta를 제외한 나머지 비트(20)만큼으로 조정
            let sigma = if (rotations >> k) & 1 == 1 { 1.0 } else { -1.0 };
            
            let power_of_2 = (2.0f32).powi(-(k as i32));

            let x_new = x - sigma * y * power_of_2;
            let y_new = y + sigma * x * power_of_2;
            
            x = x_new;
            y = y_new;

            // 쌍곡 변환 추가
            if k % 4 == 0 {
                let r = (x*x + y*y).sqrt();
                if r > 1e-9 { // 0에 가까운 값 방지
                    let tanh_r = r.tanh();
                    x *= tanh_r;
                    y *= tanh_r;
                }
            }
        }
        
        // CORDIC 게인 보정.
        let gain = 1.64676; 
        x / gain
    }
}

/// 128-bit 시드 (Seed0: 비트필드, Seed1: 연속 FP32×2)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Packed128 {
    pub hi: u64,   // Seed0 : 기존 Packed64 비트필드
    pub lo: u64,   // Seed1 : [63:32] r_fp32  |  [31:0] θ_fp32
}

/// 연속 파라미터까지 포함해 디코딩
#[derive(Debug, Clone, Default)]
pub struct DecodedParams {
    pub r_fp32: f32,
    pub theta_fp32: f32,
}

impl Packed128 {
    /// Seed0+1 디코딩
    pub fn decode(&self) -> DecodedParams {
        let r_fp32     = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        DecodedParams { r_fp32, theta_fp32, ..Default::default() }
    }
    
    /// 연속 파라미터 → 128 bit 시드
    pub fn from_continuous(p: &DecodedParams) -> Self {
        // new.md의 비트 레이아웃에 따라 hi 필드를 구성합니다.
        // r (Q0.20) -> [63:44], theta (Q0.24) -> [43:20]
        let r_quant = ste_quant_q0x(p.r_fp32, 20);
        let theta_quant = ste_quant_phase(p.theta_fp32, 24);

        let hi = (r_quant << 44) | (theta_quant << 20); // 다른 필드는 0으로 가정
        let lo = ((p.r_fp32.to_bits() as u64) << 32) | p.theta_fp32.to_bits() as u64;
        
        Packed128 { hi, lo }
    }

    /// 무작위 초기화
    pub fn random(rng: &mut impl Rng) -> Self {
        let r = 0.8 + rng.gen::<f32>() * 0.2; // [0.8, 1.0] 범위로 증가
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        
        let r_quant = ste_quant_q0x(r, 20);
        let theta_quant = ste_quant_phase(theta, 24);
        
        let random_bits = rng.gen::<u64>() & 0xFFFFF; // 하위 20비트
        
        let hi = (r_quant << 44) | (theta_quant << 20) | random_bits;
        let lo = ((r.to_bits() as u64) << 32) | theta.to_bits() as u64;
        
        Packed128 { hi, lo }
    }
    
    /// 정밀한 순전파: hi(상태 전이) + lo(연속) 융합 (고급 8개 상태 함수)
    #[inline(always)]
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출 (lo)
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 상태 비트 추출 (hi) - 고급 버전
        let state_bits = self.hi & 0xFFFFF; // 하위 20비트
        
        // 3. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 4. 연속 기저 패턴 계산
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        let base_pattern = (r_fp32 - dist * r_fp32 + theta_fp32).clamp(0.0, 1.0);
        
        // 5. 다중 상태 기반 함수 선택 및 변조 (8개 상태 함수)
        let primary_hash = ((i * 31 + j) & 0x7) as u64;
        let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
        let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
        
        // 주요 함수 계산 (8개 상태)
        let primary_value = self.compute_state_function(
            primary_state, 
            base_angle + theta_fp32 * 0.5, 
            r_fp32
        );
        
        // 보조 변조 적용
        let modulation_factor = match secondary_state {
            0 => 1.0,                                    // 변조 없음
            1 => 0.8 + 0.4 * (dist * 3.14159).sin(),    // 사인 변조
            2 => 1.0 - 0.3 * dist,                      // 거리 기반 감쇠
            3 => (1.0 + (base_angle * 2.0).cos()) * 0.5, // 각도 기반 변조
            _ => 1.0,
        };
        
        let modulated_value = base_pattern * primary_value.abs() * modulation_factor;
        
        // 6. 고주파 세부사항 추가 (나머지 상태 비트 활용)
        let detail_bits = (state_bits >> 16) & 0xF; // 상위 4비트
        let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
        
        (modulated_value * detail_factor).clamp(-1.0, 1.0)
    }
    
    /// 상태 전이 적용 (이산 미분)
    pub fn apply_state_transition(&mut self, error: f32, i: usize, j: usize) {
        let coord_hash = (i * 31 + j) & 0x1F; // 5비트 해시
        
        // 에러 크기에 따른 상태 전이 강도 결정
        let transition_strength = if error.abs() > 0.1 {
            3 // 큰 에러: 강한 전이
        } else if error.abs() > 0.01 {
            2 // 중간 에러: 중간 전이
        } else {
            1 // 작은 에러: 약한 전이
        };
        
        // 에러 부호에 따른 전이 방향
        let transition_direction = if error > 0.0 { 1u64 } else { 0u64 };
        
        // 상태 비트 업데이트
        let bit_position = coord_hash % 20;
        let mask = !(1u64 << bit_position);
        self.hi = (self.hi & mask) | (transition_direction << bit_position);
        
        // 추가 강도에 따른 비트 확산
        for s in 1..transition_strength {
            let spread_pos = (bit_position + s) % 20;
            let spread_mask = !(1u64 << spread_pos);
            self.hi = (self.hi & spread_mask) | (transition_direction << spread_pos);
        }
    }
    
    /// 고급 상태 전이 (다단계 전이)
    pub fn advanced_state_transition(&mut self, error: f32, i: usize, j: usize) {
        let primary_hash = (i * 31 + j) & 0x1F;
        let secondary_hash = (i * 17 + j * 13) & 0x1F;
        
        // 에러 크기에 따른 전이 패턴
        let error_magnitude = error.abs();
        let transition_pattern = if error_magnitude > 0.5 {
            0b111 // 강한 패턴
        } else if error_magnitude > 0.1 {
            0b101 // 중간 패턴
        } else {
            0b001 // 약한 패턴
        };
        
        // 주요 전이
        let primary_pos = primary_hash % 20;
        self.hi ^= (transition_pattern as u64) << primary_pos;
        
        // 보조 전이 (약간의 확산)
        let secondary_pos = secondary_hash % 20;
        self.hi ^= ((transition_pattern >> 1) as u64) << secondary_pos;
    }
    

    
    /// 8가지 상태 함수 계산
    pub fn compute_state_function(&self, state: u64, input: f32, scale: f32) -> f32 {
        let scaled_input = input * scale;
        match state {
            0 => scaled_input.sin(),                           // sin 상태
            1 => scaled_input.cos(),                           // cos 상태
            2 => scaled_input.tanh(),                          // tanh 상태
            3 => {                                             // sech² 상태
                let cosh_val = scaled_input.cosh();
                1.0 / (cosh_val * cosh_val)
            },
            4 => (scaled_input * 0.1).exp().min(10.0),        // exp 상태 (폭발 방지)
            5 => (scaled_input.abs() + 1e-6).ln(),            // log 상태 (0 방지)
            6 => 1.0 / (scaled_input + 1e-6),                 // 1/x 상태 (무한대 방지)
            7 => scaled_input + 0.1 * scaled_input * scaled_input, // 다항식 상태
            _ => scaled_input,
        }
    }
} 