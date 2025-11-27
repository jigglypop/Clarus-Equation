use std::collections::VecDeque;

/// SFE-ARC (Adaptive Riemannian Cancellation) 컨트롤러
pub struct SfeArcController {
    pub alpha: f64, 
    pub beta: f64,  
    pub latency: usize, 

    predictor: CurvaturePredictor,
}

impl SfeArcController {
    pub fn new(alpha: f64, beta: f64, latency: usize) -> Self {
        Self {
            alpha,
            beta,
            latency,
            predictor: CurvaturePredictor::new(),
        }
    }

    pub fn step(&mut self, measured_r: f64, dt: f64) -> f64 {
        // 1. 상태 업데이트
        self.predictor.update(measured_r, dt);

        // 2. 정밀 미래 예측 (Analytic Solution)
        let lookahead = self.latency as f64 * dt;
        let (pred_r, pred_v) = self.predictor.predict_exact(lookahead);

        // 3. 펄스 생성
        let sqrt_r = if pred_r > 0.0 { pred_r.sqrt() } else { 0.0 };
        let cancel_pulse = -(self.alpha * sqrt_r + self.beta * pred_v);

        cancel_pulse
    }
}

/// 고정밀 곡률 예측 엔진
struct CurvaturePredictor {
    est_r: f64,      
    est_v: f64,      
    
    // 물리 파라미터
    omega_0: f64,    
    gamma: f64,      
    
    // 유도된 파라미터 (캐싱)
    omega_prime: f64, // 감쇠 진동수
    
    kalman_gain: f64, 
}

impl CurvaturePredictor {
    fn new() -> Self {
        let omega_0: f64 = 5.0;
        let gamma: f64 = 0.5;
        // Under-damped condition check: omega_0 > gamma/2
        let omega_prime = (omega_0.powi(2) - (gamma / 2.0).powi(2)).sqrt();

        Self {
            est_r: 1.0, 
            est_v: 0.0,
            omega_0,
            gamma,
            omega_prime,
            kalman_gain: 0.5, // 반응 속도 상향
        }
    }

    fn update(&mut self, measured_r: f64, dt: f64) {
        // A priori prediction (짧은 구간은 Euler로 충분)
        let pred_r = self.est_r + self.est_v * dt;
        let pred_v = self.est_v + (-self.gamma * self.est_v - self.omega_0.powi(2) * self.est_r) * dt;

        // Correction
        let innovation = measured_r - pred_r;
        self.est_r = pred_r + self.kalman_gain * innovation;
        // 속도 보정 계수 튜닝: 위치 오차가 크면 속도 추정도 크게 틀렸을 가능성
        self.est_v = pred_v + (self.kalman_gain * 2.0) * innovation / dt; 
    }

    /// 감쇠 진동자의 해석적 해를 이용한 미래 예측
    /// R(t) = e^(-gamma*t/2) * (C1*cos(w't) + C2*sin(w't))
    fn predict_exact(&self, t: f64) -> (f64, f64) {
        let c1 = self.est_r;
        let c2 = (self.est_v + self.gamma / 2.0 * self.est_r) / self.omega_prime;

        let decay = (-self.gamma * t / 2.0).exp();
        let cos_wt = (self.omega_prime * t).cos();
        let sin_wt = (self.omega_prime * t).sin();

        let r_pred = decay * (c1 * cos_wt + c2 * sin_wt);
        
        // 미분값(속도) 예측
        // v(t) = R'(t)
        // 식 유도 생략, 결과식 적용
        let v_pred = decay * (
            (c2 * self.omega_prime - c1 * self.gamma / 2.0) * cos_wt - 
            (c1 * self.omega_prime + c2 * self.gamma / 2.0) * sin_wt
        );

        (r_pred, v_pred)
    }
}

pub struct ArcSimulationEnv {
    true_r: f64,
    true_v: f64,
    
    process_noise: f64,
    measure_noise: f64,
    
    pub controller: SfeArcController,
    pulse_queue: VecDeque<f64>,
}

impl ArcSimulationEnv {
    pub fn new() -> Self {
        let mut env = Self {
            true_r: 1.0,
            true_v: 0.0,
            process_noise: 0.005, // 극저온 환경 반영 (0.5% Noise)
            measure_noise: 0.01,  // 정밀 측정 (1.0% Noise)
            controller: SfeArcController::new(1.0, 0.1, 2), 
            pulse_queue: VecDeque::with_capacity(2),
        };
        
        for _ in 0..2 {
            env.pulse_queue.push_back(0.0);
        }
        
        env
    }

    pub fn step(&mut self, dt: f64) -> (f64, f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let noise_impulse: f64 = rng.gen_range(-1.0..1.0);
        
        let gamma: f64 = 0.5;
        let omega: f64 = 5.0;
        
        let force = -gamma * self.true_v - omega.powi(2) * self.true_r;
        self.true_v += force * dt + self.process_noise * noise_impulse * dt.sqrt();
        self.true_r += self.true_v * dt;

        let r_clamped = if self.true_r > 0.0 { self.true_r } else { 0.0 };
        let actual_noise_influence = self.controller.alpha * r_clamped.sqrt() + self.controller.beta * self.true_v;

        let measure_noise_val: f64 = rng.gen_range(-1.0..1.0);
        let measured_r = self.true_r + self.measure_noise * measure_noise_val;

        let new_pulse = self.controller.step(measured_r, dt);
        
        self.pulse_queue.push_back(new_pulse);
        let delayed_pulse = self.pulse_queue.pop_front().unwrap_or(0.0);

        let residual = actual_noise_influence + delayed_pulse;

        (actual_noise_influence, residual)
    }
}
