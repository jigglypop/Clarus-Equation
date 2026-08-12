use crate::packed_params::Packed128;

/// 4.4.1 푸앵카레 볼 제약 투영
/// 
/// 연속 파라미터 업데이트 시 푸앵카레 볼 내부를 유지
#[derive(Debug, Clone)]
pub struct ConstraintProjection {
    /// 하드 제약: 절대 경계
    pub hard_boundary: f32,
    /// 소프트 제약: 경고 영역
    pub soft_boundary: f32,
    /// 소프트 제약 강도 (복원력)
    pub penalty_strength: f32,
}

impl ConstraintProjection {
    pub fn new() -> Self {
        Self {
            hard_boundary: 0.99,  // 푸앵카레 볼 단위원 내부
            soft_boundary: 0.95,  // 소프트 경고 영역
            penalty_strength: 1.0,
        }
    }
    
    /// 푸앵카레 볼 제약 투영
    /// 
    /// r ∈ [0, 0.99), θ ∈ R (무제한)
    /// r ≥ 0.95 시 소프트 패널티로 중심으로 끌어당김
    pub fn project_to_poincare_ball(&self, params: &mut Packed128) {
        // 연속 파라미터 추출
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // r 제약 처리
        let mut new_r = r_fp32;
        if r_fp32 >= self.hard_boundary {
            // 하드 클램핑
            new_r = self.hard_boundary - 0.001;
        } else if r_fp32 >= self.soft_boundary {
            // 소프트 패널티로 중심쪽으로 이동
            let excess = r_fp32 - self.soft_boundary;
            let penalty = self.penalty_strength * excess;
            new_r = r_fp32 - penalty;
        }
        
        // 음수 방지
        new_r = new_r.max(0.0);
        
        // theta는 제약 없음 (주기적 함수이므로)
        let new_theta = theta_fp32;
        
        // 업데이트된 파라미터 적용
        params.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    }
} 