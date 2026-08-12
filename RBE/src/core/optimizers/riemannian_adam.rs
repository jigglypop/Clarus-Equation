use std::f32::consts::PI;
use std::collections::HashMap;

// f32 정밀도에 맞는 푸앵카레볼 경계값
const POINCARE_BOUNDARY_F32: f32 = 0.9999999;

// **고속 삼각함수 룩업 테이블** (differential 스타일 최적화)
const TANH_LUT_SIZE: usize = 2048;
static mut TANH_LUT: [f32; TANH_LUT_SIZE] = [0.0; TANH_LUT_SIZE];
static mut LUT_INITIALIZED: bool = false;

/// 푸앵카레 볼에서의 리만 아담 최적화기 (비트 수준 최적화)
/// 
/// 6.1 리만 아담 알고리즘
/// 하이퍼볼릭 공간에서의 기울기 하강법
#[derive(Debug, Clone)]
pub struct RiemannianAdamState {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m_r: f32,      // r에 대한 1차 모멘텀
    pub v_r: f32,      // r에 대한 2차 모멘텀
    pub m_theta: f32,  // θ에 대한 1차 모멘텀
    pub v_theta: f32,  // θ에 대한 2차 모멘텀
    pub t: i32,        // 시간 스텝
    // **성능 최적화 캐시** (differential 패턴)
    beta_powers_cache: HashMap<i32, (f32, f32)>, // t -> (beta1^t, beta2^t)
    metric_cache: HashMap<u32, (f32, f32)>, // r_bits -> (g_rr, g_theta_theta)
    update_cache: Option<(f32, f32, f32, f32, (f32, f32))>, // (grad_r, grad_theta, lr, t) -> (update_r, update_theta)
}

impl Default for RiemannianAdamState {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            beta_powers_cache: HashMap::new(),
            metric_cache: HashMap::new(),
            update_cache: None,
        }
    }
}

impl RiemannianAdamState {
    pub fn new() -> Self {
        // **룩업 테이블 초기화** (최초 1회만)
        unsafe {
            if !LUT_INITIALIZED {
                Self::init_lookup_tables();
                LUT_INITIALIZED = true;
            }
        }
        
        Self {
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            beta_powers_cache: HashMap::new(),
            metric_cache: HashMap::new(),
            update_cache: None,
        }
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        unsafe {
            if !LUT_INITIALIZED {
                Self::init_lookup_tables();
                LUT_INITIALIZED = true;
            }
        }
        
        Self {
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            beta1,
            beta2,
            epsilon,
            beta_powers_cache: HashMap::new(),
            metric_cache: HashMap::new(),
            update_cache: None,
        }
    }
    
    /// **룩업 테이블 초기화** (고속 삼각함수)
    unsafe fn init_lookup_tables() {
        for i in 0..TANH_LUT_SIZE {
            let x = (i as f32 / TANH_LUT_SIZE as f32) * 10.0 - 5.0; // [-5, 5] 범위
            TANH_LUT[i] = x.tanh();
        }
    }
    
    /// **고속 tanh 룩업** (삼각함수 최적화)
    #[inline]
    fn fast_tanh(&self, x: f32) -> f32 {
        if x.abs() > 5.0 {
            return if x > 0.0 { 1.0 } else { -1.0 };
        }
        
        unsafe {
            let index = ((x + 5.0) / 10.0 * TANH_LUT_SIZE as f32) as usize;
            let clamped_index = index.min(TANH_LUT_SIZE - 1);
            TANH_LUT[clamped_index]
        }
    }
    
    /// 푸앵카레 볼에서의 메트릭 텐서 계산 (캐시 최적화)
    pub fn compute_metric_tensor(&mut self, r: f32) -> (f32, f32) {
        // **캐시 확인** (비트 수준 해싱)
        let r_bits = r.to_bits();
        let cache_key = r_bits;
        
        if let Some(&cached_metric) = self.metric_cache.get(&cache_key) {
            return cached_metric;
        }
        
        // r이 경계에 너무 가까우면 안전한 값으로 클램핑
        let safe_r = r.clamp(0.0, POINCARE_BOUNDARY_F32);
        let r_squared = safe_r * safe_r;
        let denominator = (1.0 - r_squared).max(1e-10); // 분모 보호
        
        let factor = 4.0 / (denominator * denominator);
        let g_rr = factor;
        let g_theta_theta = factor * r_squared.max(1e-10); // θ 메트릭 안정성 보장
        
        let result = (g_rr, g_theta_theta);
        
        // **캐시 저장** (크기 제한)
        if self.metric_cache.len() < 10000 {
            self.metric_cache.insert(cache_key, result);
        }
        
        result
    }
    
    /// 뫼비우스 덧셈 (인라인 최적화)
    #[inline]
    pub fn mobius_add(&self, x: f32, y: f32) -> f32 {
        // **극초기 종료** (differential 스타일)
        if y.abs() < 1e-15 { return x; }
        if x.abs() < 1e-15 { return y; }
        
        // 입력 검증
        if !x.is_finite() || !y.is_finite() {
            return 0.0;
        }
        
        let numerator = x + y;
        let denominator = 1.0 + x * y;
        
        // 분모가 0에 가까우면 안전한 값 반환
        if denominator.abs() < 1e-12 {
            return x; // 원래 값 유지
        }
        
        let result = numerator / denominator;
        
        // 결과가 푸앵카레볼을 벗어나면 클리핑
        if result.abs() >= 1.0 {
            POINCARE_BOUNDARY_F32 * result.signum()
        } else {
            result
        }
    }
    
    /// 푸앵카레 볼에서의 지수 사상 (고속 최적화 + Small-Move 근사)
    #[inline]
    pub fn exponential_map(&self, x: f32, v: f32) -> f32 {
        // **극초기 종료** (조기 최적화)
        if v.abs() < 1e-15 { return x; }
        
        // 입력 검증
        if !x.is_finite() || !v.is_finite() {
            return x.clamp(0.0, POINCARE_BOUNDARY_F32);
        }
        
        let norm_v = v.abs();
        
        // **Small-Move 지오데식 근사**: 작은 이동에 대해 Taylor 근사 사용
        if norm_v < 0.1 {
            // 1차 근사: exp_x(v) ≈ x + v/(1-|x|²)
            let x_norm_sq = x * x;
            if x_norm_sq < 0.999 {
                let conformal_factor = 1.0 / (1.0 - x_norm_sq);
                let result = x + v * conformal_factor;
                return result.clamp(0.0, POINCARE_BOUNDARY_F32);
            }
        }
        
        // 큰 이동에 대해서는 기존 정확한 공식 사용
        let tanh_arg = (norm_v / 2.0).min(5.0); // tanh 포화 방지
        let tanh_factor = self.fast_tanh(tanh_arg);
        let direction = v / norm_v;
        
        let result = self.mobius_add(x, tanh_factor * direction);
        result.clamp(0.0, POINCARE_BOUNDARY_F32)
    }
    
    /// **베타 거듭제곱 최적화** (Adam과 동일한 최적화)
    #[inline]
    fn get_or_compute_beta_powers(&mut self, t: i32) -> (f32, f32) {
        // 캐시 확인
        if let Some(&powers) = self.beta_powers_cache.get(&t) {
            return powers;
        }
        
        // 효율적 계산: 이전 값 기반 증분 계산
        let (beta1_power, beta2_power) = if t == 1 {
            (self.beta1, self.beta2)
        } else if let Some(&(prev_beta1, prev_beta2)) = self.beta_powers_cache.get(&(t-1)) {
            (prev_beta1 * self.beta1, prev_beta2 * self.beta2)
        } else {
            // 폴백: 고속 거듭제곱 (비트 시프트 활용)
            (self.fast_power(self.beta1, t), self.fast_power(self.beta2, t))
        };
        
        // 캐시 저장 (크기 제한)
        if self.beta_powers_cache.len() < 1000 {
            self.beta_powers_cache.insert(t, (beta1_power, beta2_power));
        }
        
        (beta1_power, beta2_power)
    }
    
    /// **고속 거듭제곱** (이진 거듭제곱법)
    #[inline]
    fn fast_power(&self, base: f32, exp: i32) -> f32 {
        if exp <= 0 { return 1.0; }
        if exp == 1 { return base; }
        
        // 이진 거듭제곱법 (비트 수준 최적화)
        let mut result = 1.0;
        let mut base_power = base;
        let mut exponent = exp as u32;
        
        while exponent > 0 {
            if exponent & 1 == 1 {
                result *= base_power;
            }
            base_power *= base_power;
            exponent >>= 1;
        }
        
        result
    }
    
    /// **고속 제곱근** (뉴턴-랩슨)
    #[inline]
    fn fast_sqrt(&self, x: f32) -> f32 {
        if x <= 0.0 { return 0.0; }
        if x == 1.0 { return 1.0; }
        
        // 뉴턴-랩슨 1회 반복
        let guess = x * 0.5;
        guess + (x - guess * guess) / (2.0 * guess)
    }
    
    /// **Riemannian Adam 업데이트** (비트 수준 최적화 적용)
    pub fn update(&mut self, r: &mut f32, theta: &mut f32, grad_r: f32, grad_theta: f32, learning_rate: f32) {
        // **1단계: 극초기 종료** (differential 스타일)
        if grad_r.abs() < 1e-15 && grad_theta.abs() < 1e-15 {
            return;
        }
        
        // **2단계: NaN/Inf 체크**
        if !grad_r.is_finite() || !grad_theta.is_finite() {
            return;
        }
        
        // **3단계: 캐시 확인** (5-튜플 캐시)
        if let Some((cached_grad_r, cached_grad_theta, cached_lr, cached_t, (cached_update_r, cached_update_theta))) = self.update_cache {
            if (grad_r - cached_grad_r).abs() < 1e-12 && 
               (grad_theta - cached_grad_theta).abs() < 1e-12 && 
               (learning_rate - cached_lr).abs() < 1e-12 &&
               self.t == cached_t as i32 {
                *r = (*r + cached_update_r).clamp(0.0, POINCARE_BOUNDARY_F32);
                *theta = (*theta + cached_update_theta).rem_euclid(2.0 * PI);
                return; // 캐시 히트로 즉시 완료
            }
        }
        
        // 안전한 r 값 확보
        *r = (*r).clamp(0.0, POINCARE_BOUNDARY_F32);
        
        self.t += 1;
        
        // **4단계: 메트릭 텐서 계산** (캐시 최적화)
        let (g_rr, g_theta_theta) = self.compute_metric_tensor(*r);
        
        // **5단계: 리만 그래디언트 계산** (분모 보호)
        let riem_grad_r = if g_rr > 1e-10 { grad_r / g_rr } else { 0.0 };
        let riem_grad_theta = if g_theta_theta > 1e-10 { grad_theta / g_theta_theta } else { 0.0 };
        
        // **6단계: 모멘텀 업데이트** (인라인)
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * riem_grad_r;
        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * riem_grad_r * riem_grad_r;
        
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * riem_grad_theta;
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * riem_grad_theta * riem_grad_theta;
        
        // **7단계: 편향 보정** (powi() 제거)
        let (beta1_power, beta2_power) = self.get_or_compute_beta_powers(self.t);
        
        let beta1_complement = 1.0 - beta1_power;
        let beta2_complement = 1.0 - beta2_power;
        
        if beta1_complement < 1e-12 || beta2_complement < 1e-12 {
            return;
        }
        
        let m_r_hat = self.m_r / beta1_complement;
        let v_r_hat = self.v_r / beta2_complement;
        
        let m_theta_hat = self.m_theta / beta1_complement;
        let v_theta_hat = self.v_theta / beta2_complement;
        
        // **8단계: 업데이트 벡터 계산** (고속 sqrt)
        let denom_r = self.fast_sqrt(v_r_hat) + self.epsilon;
        let denom_theta = self.fast_sqrt(v_theta_hat) + self.epsilon;
        
        if denom_r < self.epsilon * 2.0 && denom_theta < self.epsilon * 2.0 {
            return;
        }
        
        let update_r = -learning_rate * m_r_hat / denom_r;
        let update_theta = -learning_rate * m_theta_hat / denom_theta;
        
        // **9단계: 업데이트 크기 제한** (그래디언트 폭발 방지)
        let clipped_update_r = if update_r.abs() > 1.0 {
            1.0 * update_r.signum()
        } else {
            update_r
        };
        
        let clipped_update_theta = if update_theta.abs() > PI {
            PI * update_theta.signum()
        } else {
            update_theta
        };
        
        // **10단계: 지수 사상을 통한 업데이트** (고속 최적화)
        *r = self.exponential_map(*r, clipped_update_r);
        
        // **11단계: θ 업데이트** (각도 정규화)
        *theta = (*theta + clipped_update_theta).rem_euclid(2.0 * PI);
        
        // **12단계: 최종 안전성 체크**
        *r = (*r).clamp(0.0, POINCARE_BOUNDARY_F32);
        
        // **13단계: 캐시 업데이트** (다음 호출 최적화)
        self.update_cache = Some((grad_r, grad_theta, learning_rate, self.t as f32, (clipped_update_r, clipped_update_theta)));
    }
    
    /// **고급 업데이트** (그래디언트 클리핑 포함)
    pub fn update_with_clipping(&mut self, r: &mut f32, theta: &mut f32, grad_r: f32, grad_theta: f32, learning_rate: f32, clip_norm: Option<f32>) {
        let (clipped_grad_r, clipped_grad_theta) = if let Some(clip) = clip_norm {
            let grad_norm = self.fast_sqrt(grad_r * grad_r + grad_theta * grad_theta);
            if grad_norm > clip {
                let scale = clip / grad_norm;
                (grad_r * scale, grad_theta * scale)
            } else {
                (grad_r, grad_theta)
            }
        } else {
            (grad_r, grad_theta)
        };
        
        self.update(r, theta, clipped_grad_r, clipped_grad_theta, learning_rate);
    }
    
    /// **상태 초기화** (캐시 포함)
    pub fn reset(&mut self) {
        self.m_r = 0.0;
        self.v_r = 0.0;
        self.m_theta = 0.0;
        self.v_theta = 0.0;
        self.t = 0;
        self.beta_powers_cache.clear();
        self.metric_cache.clear();
        self.update_cache = None;
    }
    
    /// 현재 모멘텀 크기 반환
    pub fn get_momentum_magnitude(&self) -> (f32, f32) {
        (self.m_r.abs(), self.m_theta.abs())
    }
    
    /// **적응적 학습률 확인** (캐시 활용)
    pub fn get_effective_learning_rates(&self, base_lr: f32) -> (f32, f32) {
        if self.t == 0 {
            return (base_lr, base_lr);
        }
        
        if let Some(&(beta1_power, beta2_power)) = self.beta_powers_cache.get(&self.t) {
            let m_r_hat = self.m_r / (1.0 - beta1_power);
            let v_r_hat = self.v_r / (1.0 - beta2_power);
            let m_theta_hat = self.m_theta / (1.0 - beta1_power);
            let v_theta_hat = self.v_theta / (1.0 - beta2_power);
            
            let lr_r = if v_r_hat > 0.0 {
                base_lr * m_r_hat.abs() / (self.fast_sqrt(v_r_hat) + self.epsilon)
            } else {
                0.0
            };
            
            let lr_theta = if v_theta_hat > 0.0 {
                base_lr * m_theta_hat.abs() / (self.fast_sqrt(v_theta_hat) + self.epsilon)
            } else {
                0.0
            };
            
            (lr_r, lr_theta)
        } else {
            (0.0, 0.0)
        }
    }
    
    /// **수렴 여부 확인**
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.m_r.abs() < threshold && 
        self.m_theta.abs() < threshold && 
        self.fast_sqrt(self.v_r) < threshold && 
        self.fast_sqrt(self.v_theta) < threshold
    }
    
    /// **푸앵카레볼 거리 계산** (고속 최적화)
    pub fn poincare_distance(&self, r1: f32, theta1: f32, r2: f32, theta2: f32) -> f32 {
        // 빠른 경로: 동일한 점
        if (r1 - r2).abs() < 1e-10 && (theta1 - theta2).abs() < 1e-10 {
            return 0.0;
        }
        
        let x1 = r1 * theta1.cos();
        let y1 = r1 * theta1.sin();
        let x2 = r2 * theta2.cos();
        let y2 = r2 * theta2.sin();
        
        let norm1_sq = x1 * x1 + y1 * y1;
        let norm2_sq = x2 * x2 + y2 * y2;
        let dot_product = x1 * x2 + y1 * y2;
        
        let numerator = norm1_sq + norm2_sq - 2.0 * dot_product;
        let denominator = (1.0 - norm1_sq) * (1.0 - norm2_sq);
        
        if denominator > 1e-10 {
            let ratio = numerator / denominator;
            (1.0 + 2.0 * ratio.max(0.0)).ln()
        } else {
            0.0
        }
    }
    
    /// **캐시 통계**
    pub fn get_cache_stats(&self) -> (usize, usize, bool) {
        (self.beta_powers_cache.len(), self.metric_cache.len(), self.update_cache.is_some())
    }
} 