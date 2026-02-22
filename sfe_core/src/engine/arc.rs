use std::collections::VecDeque;

const MAX_MODES: usize = 6;

/// Riemannian geodesic deviation mode tracker (Jacobi equation discretization).
/// d^2 r / dt^2 = -omega^2 r - gamma dr/dt + noise
/// Symplectic Euler: v_new = v + f(r)*dt, r_new = r + v_new*dt
#[derive(Clone, Debug)]
struct CurvatureMode {
    omega: f64,
    gamma: f64,
    est_r: f64,
    est_v: f64,
    p_rr: f64,
    p_rv: f64,
    p_vv: f64,
    q_vv: f64,
}

impl CurvatureMode {
    fn new(omega: f64, gamma: f64, q_vv: f64) -> Self {
        Self {
            omega,
            gamma,
            est_r: 0.0,
            est_v: 0.0,
            p_rr: 2.0,
            p_rv: 0.0,
            p_vv: 2.0,
            q_vv,
        }
    }

    /// Symplectic Euler predict -- matches the simulation integrator exactly.
    fn predict(&mut self, dt: f64) {
        let force_coeff_r = -self.omega * self.omega * dt;
        let force_coeff_v = 1.0 - self.gamma * dt;

        let v_new = force_coeff_r * self.est_r + force_coeff_v * self.est_v;
        let r_new = self.est_r + v_new * dt;

        // State transition for symplectic Euler:
        // v_new = a21 * r + a22 * v          =>  A_v = [a21, a22]
        // r_new = r + v_new * dt = (1 + a21*dt) * r + a22*dt * v   =>  A_r = [1+a21*dt, a22*dt]
        let a_rr = 1.0 + force_coeff_r * dt;
        let a_rv = force_coeff_v * dt;
        let a_vr = force_coeff_r;
        let a_vv = force_coeff_v;

        let p11 = a_rr * a_rr * self.p_rr + 2.0 * a_rr * a_rv * self.p_rv + a_rv * a_rv * self.p_vv;
        let p12 = a_rr * a_vr * self.p_rr + (a_rr * a_vv + a_rv * a_vr) * self.p_rv + a_rv * a_vv * self.p_vv;
        let p22 = a_vr * a_vr * self.p_rr + 2.0 * a_vr * a_vv * self.p_rv + a_vv * a_vv * self.p_vv;

        // Process noise enters through velocity only: Q = [[q_vv*dt^2, q_vv*dt], [q_vv*dt, q_vv]]
        let q_rr = self.q_vv * dt * dt;
        let q_rv = self.q_vv * dt;

        self.est_r = r_new;
        self.est_v = v_new;
        self.p_rr = p11 + q_rr;
        self.p_rv = p12 + q_rv;
        self.p_vv = p22 + self.q_vv;
    }

    fn update_shared(&mut self, innovation: f64, s_inv: f64) {
        let k_r = self.p_rr * s_inv;
        let k_v = self.p_rv * s_inv;

        self.est_r += k_r * innovation;
        self.est_v += k_v * innovation;

        let p_rv_old = self.p_rv;
        self.p_rr -= k_r * self.p_rr;
        self.p_rv -= k_r * self.p_rv;
        self.p_vv -= k_v * p_rv_old;
        if self.p_rr < 1e-15 {
            self.p_rr = 1e-15;
        }
        if self.p_vv < 1e-15 {
            self.p_vv = 1e-15;
        }
    }

    /// Analytic prediction for the deterministic damped oscillator.
    fn predict_future(&self, t: f64) -> (f64, f64) {
        let disc = self.omega * self.omega - (self.gamma / 2.0).powi(2);

        if disc < 1e-10 {
            let decay = (-self.gamma * t / 2.0).exp();
            return (decay * self.est_r, decay * self.est_v);
        }

        let omega_d = disc.sqrt();
        let c1 = self.est_r;
        let c2 = (self.est_v + self.gamma / 2.0 * self.est_r) / omega_d;

        let decay = (-self.gamma * t / 2.0).exp();
        let cos_wt = (omega_d * t).cos();
        let sin_wt = (omega_d * t).sin();

        let r_pred = decay * (c1 * cos_wt + c2 * sin_wt);
        let v_pred = decay
            * ((c2 * omega_d - c1 * self.gamma / 2.0) * cos_wt
                - (c1 * omega_d + c2 * self.gamma / 2.0) * sin_wt);

        (r_pred, v_pred)
    }
}

/// SFE-ARC Controller: Riemannian holonomy coupling, joint Kalman filter.
/// noise_influence = alpha * R + beta * V  (smooth, linear in curvature)
pub struct SfeArcController {
    pub alpha: f64,
    pub beta: f64,
    pub latency: usize,

    modes: Vec<CurvatureMode>,
    r_meas: f64,
}

impl SfeArcController {
    pub fn new(alpha: f64, beta: f64, latency: usize) -> Self {
        // q_vv = process_noise^2 * dt / 3  (uniform distribution variance correction)
        // Inflated 5x for robustness against model mismatch
        let q_vv_base = 0.08_f64.powi(2) * 0.01 / 3.0;
        let q_vv = q_vv_base * 5.0;

        let modes = vec![
            CurvatureMode::new(5.0, 0.5, q_vv),
            CurvatureMode::new(2.5, 0.3, q_vv),
            CurvatureMode::new(10.0, 0.8, q_vv),
        ];

        // r_meas = measure_noise^2 / 3, inflated 5x
        let r_meas = 0.005_f64.powi(2) / 3.0 * 5.0;

        Self {
            alpha,
            beta,
            latency,
            modes,
            r_meas,
        }
    }

    pub fn new_with_modes(
        alpha: f64,
        beta: f64,
        latency: usize,
        mode_params: &[(f64, f64, f64)],
    ) -> Self {
        let modes = mode_params
            .iter()
            .take(MAX_MODES)
            .map(|&(omega, gamma, qvv)| CurvatureMode::new(omega, gamma, qvv))
            .collect();

        let r_meas = 0.005_f64.powi(2) / 3.0 * 5.0;

        Self {
            alpha,
            beta,
            latency,
            modes,
            r_meas,
        }
    }

    pub fn step(&mut self, measured_r: f64, dt: f64) -> f64 {
        for mode in self.modes.iter_mut() {
            mode.predict(dt);
        }

        let composite_pred: f64 = self.modes.iter().map(|m| m.est_r).sum();
        let innovation = measured_r - composite_pred;

        // Joint Kalman: S = sum(P_rr_i) + R_meas, each mode updates with full innovation
        let total_p_rr: f64 = self.modes.iter().map(|m| m.p_rr).sum();
        let s = total_p_rr + self.r_meas;
        let s_inv = 1.0 / s;

        for mode in self.modes.iter_mut() {
            mode.update_shared(innovation, s_inv);
        }

        let lookahead = self.latency as f64 * dt;
        let mut pred_r = 0.0_f64;
        let mut pred_v = 0.0_f64;

        for mode in &self.modes {
            let (pr, pv) = mode.predict_future(lookahead);
            pred_r += pr;
            pred_v += pv;
        }

        -(self.alpha * pred_r + self.beta * pred_v)
    }

    pub fn mode_energies(&self) -> Vec<f64> {
        self.modes.iter().map(|m| m.est_r * m.est_r).collect()
    }

    pub fn total_estimation_uncertainty(&self) -> f64 {
        self.modes.iter().map(|m| m.p_rr).sum()
    }
}

/// Simulation: 3 true curvature modes, Riemannian holonomy coupling
pub struct ArcSimulationEnv {
    true_modes: Vec<(f64, f64, f64, f64)>,
    process_noise: f64,
    measure_noise: f64,
    pub controller: SfeArcController,
    pulse_queue: VecDeque<f64>,
}

impl Default for ArcSimulationEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl ArcSimulationEnv {
    pub fn new() -> Self {
        let true_modes = vec![
            (1.0, 0.0, 5.0, 0.5),
            (0.3, 0.0, 2.5, 0.3),
            (0.15, 0.0, 10.0, 0.8),
        ];

        let mut env = Self {
            true_modes,
            process_noise: 0.08,
            measure_noise: 0.005,
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

        // Symplectic Euler: v first, then r with new v
        for mode in self.true_modes.iter_mut() {
            let (ref mut r, ref mut v, omega, gamma) = *mode;
            let noise_impulse: f64 = rng.gen_range(-1.0..1.0);

            let force = -gamma * *v - omega * omega * *r;
            *v += force * dt + self.process_noise * noise_impulse * dt.sqrt();
            *r += *v * dt;
        }

        let true_r: f64 = self.true_modes.iter().map(|(r, _, _, _)| *r).sum();
        let true_v: f64 = self.true_modes.iter().map(|(_, v, _, _)| *v).sum();

        let actual_noise_influence =
            self.controller.alpha * true_r + self.controller.beta * true_v;

        let measure_noise_val: f64 = rng.gen_range(-1.0..1.0);
        let measured_r = true_r + self.measure_noise * measure_noise_val;

        let new_pulse = self.controller.step(measured_r, dt);

        self.pulse_queue.push_back(new_pulse);
        let delayed_pulse = self.pulse_queue.pop_front().unwrap_or(0.0);

        let residual = actual_noise_influence + delayed_pulse;

        (actual_noise_influence, residual)
    }
}
