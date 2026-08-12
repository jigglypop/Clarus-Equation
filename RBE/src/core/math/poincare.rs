use std::f32::consts::PI;

// f64 정밀도에 맞는 푸앵카레볼 경계값 상수
const POINCARE_BOUNDARY_F32: f32 = 0.9999999; // f32용 안전한 경계값
const POINCARE_BOUNDARY_F64: f64 = 0.9999999999999999; // f64용 안전한 경계값

/// 5.2 푸앵카레 볼 위의 점
/// 
/// 푸앵카레 볼 D^n = {x ∈ R^n : ||x|| < 1}에서 점을 나타냅니다.
/// 리만 메트릭: g(x) = 4/(1-||x||²)² I_n
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareBallPoint {
    /// 좌표 (r, θ)
    pub r: f32,      // 반지름 [0, 1)
    pub theta: f32,  // 각도 [-∞, ∞)
}

impl PoincareBallPoint {
    /// 새로운 푸앵카레 볼 점 생성
    pub fn new(r: f32, theta: f32) -> Self {
        Self {
            r: r.clamp(0.0, POINCARE_BOUNDARY_F32), // f32 정밀도에 맞는 경계값 사용
            theta,
        }
    }
    
    /// 원점 (r=0, θ=0)
    pub fn origin() -> Self {
        Self { r: 0.0, theta: 0.0 }
    }
    
    /// 데카르트 좌표로 변환
    pub fn to_cartesian(&self) -> (f32, f32) {
        (self.r * libm::cosf(self.theta), self.r * libm::sinf(self.theta))
    }
    
    /// 데카르트 좌표에서 변환
    pub fn from_cartesian(x: f32, y: f32) -> Self {
        let r = (x*x + y*y).sqrt().min(POINCARE_BOUNDARY_F32);
        let theta = libm::atan2f(y, x);
        Self { r, theta }
    }
    
    /// 푸앵카레 볼 경계까지의 거리
    pub fn distance_to_boundary(&self) -> f32 {
        1.0 - self.r
    }
}

/// 5.2 리만 기하학 연산
/// 
/// 푸앵카레 볼의 리만 메트릭 연산을 제공합니다.
#[derive(Debug, Clone)]
pub struct RiemannianGeometry;

impl RiemannianGeometry {
    /// 5.2.1 리만 메트릭 텐서 계산
    /// 
    /// g(x) = 4/(1-||x||²)² I_n
    /// 점 x에서의 메트릭 인수를 반환합니다.
    pub fn metric_factor(point: &PoincareBallPoint) -> f32 {
        let norm_sq = point.r * point.r;
        let denominator = 1.0 - norm_sq;
        
        if denominator < 1e-8 {
            // 경계 근처에서 안정성 보장
            1e8
        } else {
            4.0 / (denominator * denominator)
        }
    }
    
    /// 5.2.1 리만 메트릭의 역행렬 인수
    /// 
    /// g^(-1)(x) = (1-||x||²)²/4 I_n
    pub fn inverse_metric_factor(point: &PoincareBallPoint) -> f32 {
        let norm_sq = point.r * point.r;
        let numerator = 1.0 - norm_sq;
        
        if numerator < 1e-8 {
            // 경계 근처에서 안정성 보장
            1e-8
        } else {
            numerator * numerator / 4.0
        }
    }
    
    /// 5.2.2 크리스토펠 기호 계산
    /// 
    /// Γ^k_ij = (2δ^k_i x_j + 2δ^k_j x_i - 2δ_ij x^k)/(1-||x||²)
    pub fn christoffel_symbols(point: &PoincareBallPoint) -> (f32, f32) {
        let (x, y) = point.to_cartesian();
        let norm_sq = point.r * point.r;
        let denominator = 1.0 - norm_sq;
        
        if denominator < 1e-8 {
            return (0.0, 0.0);
        }
        
        let factor = 2.0 / denominator;
        
        // r 방향과 θ 방향 크리스토펠 기호
        let gamma_r = factor * x;
        let gamma_theta = factor * y;
        
        (gamma_r, gamma_theta)
    }
    
    /// 5.2.4 Möbius 덧셈 (푸앵카레 볼의 덧셈)
    /// 
    /// x ⊕ y = (x + y + 2⟨x,y⟩/(1+||x||²) x) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
    pub fn mobius_addition(p1: &PoincareBallPoint, p2: &PoincareBallPoint) -> PoincareBallPoint {
        let (x1, y1) = p1.to_cartesian();
        let (x2, y2) = p2.to_cartesian();
        
        let dot_product = x1 * x2 + y1 * y2;
        let norm1_sq = p1.r * p1.r;
        let norm2_sq = p2.r * p2.r;
        
        let numerator_factor = 1.0 + 2.0 * dot_product / (1.0 + norm1_sq);
        let denominator = 1.0 + 2.0 * dot_product + norm1_sq * norm2_sq;
        
        if denominator < 1e-8 {
            return *p1; // 안전한 기본값
        }
        
        let result_x = (x1 + x2 * numerator_factor) / denominator;
        let result_y = (y1 + y2 * numerator_factor) / denominator;
        
        PoincareBallPoint::from_cartesian(result_x, result_y)
    }
    
    /// 5.2.4 스칼라 곱 (푸앵카레 볼의 스칼라 곱)
    /// 
    /// t ⊙ v = (t||v||/artanh(||v||)) · v/||v||
    pub fn scalar_multiplication(t: f32, point: &PoincareBallPoint) -> PoincareBallPoint {
        if point.r < 1e-8 {
            return PoincareBallPoint::origin();
        }
        
        let norm = point.r;
        let artanh_norm = if norm < POINCARE_BOUNDARY_F32 {
            0.5 * libm::logf((1.0 + norm) / (1.0 - norm))
        } else {
            10.0 // 경계 근처에서 클램핑
        };
        
        let scale_factor = if artanh_norm > 1e-8 {
            t * norm / artanh_norm
        } else {
            t
        };
        
        let new_r = (scale_factor * norm).clamp(0.0, POINCARE_BOUNDARY_F32);
        
        PoincareBallPoint::new(new_r, point.theta)
    }
    
    /// 5.2.4 지수 사상 (Exponential Map)
    /// 
    /// exp_x(v) = x ⊕ (tanh(||v||_x/2) · v/||v||_x)
    pub fn exponential_map(base: &PoincareBallPoint, tangent: &PoincareBallPoint) -> PoincareBallPoint {
        if tangent.r < 1e-8 {
            return *base;
        }
        
        // 리만 노름 계산
        let inverse_metric = Self::inverse_metric_factor(base);
        let riemannian_norm = tangent.r * inverse_metric.sqrt();
        
        // tanh(||v||_x/2) 계산
        let tanh_half_norm = libm::tanhf(riemannian_norm / 2.0);
        
        // 방향 벡터 정규화
        let direction = if riemannian_norm > 1e-8 {
            PoincareBallPoint::new(tanh_half_norm, tangent.theta)
        } else {
            PoincareBallPoint::origin()
        };
        
        Self::mobius_addition(base, &direction)
    }
    
    /// 5.3.4 쌍곡 거리 계산
    /// 
    /// d_hyp(x,y) = artanh(||(-x) ⊕ y||)
    pub fn hyperbolic_distance(p1: &PoincareBallPoint, p2: &PoincareBallPoint) -> f32 {
        // 같은 점인지 확인 (수치적 오차 고려)
        let eps = 1e-6;
        if (p1.r - p2.r).abs() < eps && ((p1.theta - p2.theta).abs() < eps || (p1.theta - p2.theta).abs() > 2.0 * PI - eps) {
            return 0.0;
        }
        
        // -p1 계산 (Möbius 역원)
        // 원점이면 원점, 아니면 데카르트 좌표에서 -z/|z|²
        let neg_p1 = if p1.r < eps {
            PoincareBallPoint::origin()
        } else {
            let (x1, y1) = p1.to_cartesian();
            let norm_sq = x1 * x1 + y1 * y1;
            if norm_sq > eps {
                let neg_x = -x1 / norm_sq;
                let neg_y = -y1 / norm_sq;
                PoincareBallPoint::from_cartesian(neg_x, neg_y)
            } else {
                PoincareBallPoint::origin()
            }
        };
        
        // (-p1) ⊕ p2 계산
        let diff = Self::mobius_addition(&neg_p1, p2);
        
        // artanh(||diff||) 계산
        let norm = diff.r;
        if norm < eps {
            0.0
        } else if norm < POINCARE_BOUNDARY_F32 {
            0.5 * libm::logf((1.0 + norm) / (1.0 - norm))
        } else {
            10.0 // 경계에서 클램핑
        }
    }
} 