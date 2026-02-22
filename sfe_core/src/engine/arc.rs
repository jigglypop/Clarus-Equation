use rand::Rng;
use std::collections::VecDeque;
use std::f64::consts::E;

/// Self-consistent strong coupling from alpha_total = 1/(2*pi)
const ALPHA_S: f64 = 0.11789;

/// Error feedback measurement variance.
/// Signal is computed from true state (noise-free); only staleness uncertainty remains.
const R_MEAS_ERR: f64 = 1e-10;

fn sfe_physics_params() -> (f64, f64, f64, f64, f64) {
    let xi = ALPHA_S.powf(1.0 / 3.0);  // coupling per dimension: alpha_s^{1/d}
    let survival = (-1.0_f64).exp();     // e^{-1}: fundamental 1D damping
    (E, survival, xi, 1.0, survival)
    // (alpha_lapse, beta_damp, xi, m_phi, gamma_phi)
}

/// SFE 3+1 state: [R, K, Phi, Pi] coupled through the suppression field equation.
/// R: Ricci scalar (spatial curvature)
/// K: extrinsic curvature (geometry velocity, from ADM lapse)
/// Phi: suppression field amplitude
/// Pi: field momentum
///
/// ADM + SFE evolution:
///   dR/dt  = -2*alpha*K          (ADM lapse coupling)
///   dK/dt  = alpha*R/2 - beta*K + xi*Phi^2   (Raychaudhuri + SFE restoring)
///   dPhi/dt = Pi
///   dPi/dt  = -(m^2 + xi*R)*Phi - gamma_phi*Pi   (field equation in curved bg)
#[derive(Clone, Debug)]
struct SfeState {
    x: [f64; 4],
    p: [[f64; 4]; 4],
    q: [[f64; 4]; 4],
}

impl SfeState {
    fn new(process_noise: f64, dt: f64) -> Self {
        let mut p = [[0.0; 4]; 4];
        for i in 0..4 {
            p[i][i] = 2.0;
        }

        // noise in K per step: process_noise * U[-1,1] * sqrt(dt) * dt
        // variance = process_noise^2 * dt^3 / 3
        let q_k = process_noise * process_noise * dt * dt * dt / 3.0;
        // noise in Pi has 0.5 factor
        let q_pi = q_k * 0.25;

        let mut q = [[0.0; 4]; 4];
        q[1][1] = q_k * 2.0;
        q[3][3] = q_pi * 2.0;

        Self {
            x: [0.0; 4],
            p,
            q,
        }
    }

    fn transition_matrix(&self, dt: f64, alpha_lapse: f64, beta_damp: f64,
                         xi: f64, m_phi: f64, gamma_phi: f64) -> [[f64; 4]; 4] {
        let r = self.x[0];
        let phi = self.x[2];

        // Jacobian of symplectic Euler scheme:
        // Step 1 (velocities): K' = K + (alpha*R/2 - beta*K + xi*Phi^2)*dt
        //   dK'/dR = alpha/2 * dt
        //   dK'/dK = 1 - beta*dt
        //   dK'/dPhi = 2*xi*Phi*dt
        //   dK'/dPi = 0
        let dk_dr = alpha_lapse / 2.0 * dt;
        let dk_dk = 1.0 - beta_damp * dt;
        let dk_dphi = 2.0 * xi * phi * dt;

        // Pi' = Pi + (-(m^2+xi*R)*Phi - gamma*Pi)*dt
        //   dPi'/dR = -xi*Phi*dt
        //   dPi'/dK = 0
        //   dPi'/dPhi = -(m^2+xi*R)*dt
        //   dPi'/dPi = 1 - gamma*dt
        let dpi_dr = -xi * phi * dt;
        let dpi_dphi = -(m_phi * m_phi + xi * r) * dt;
        let dpi_dpi = 1.0 - gamma_phi * dt;

        // Step 2 (positions using updated velocities):
        // R' = R + (-2*alpha*K')*dt = R - 2*alpha*(K + dK)*dt
        //   dR'/dR = 1 - 2*alpha*dK'/dR*dt = 1 - 2*alpha*(alpha/2*dt)*dt
        //   dR'/dK = -2*alpha*dK'/dK*dt = -2*alpha*(1-beta*dt)*dt
        //   dR'/dPhi = -2*alpha*dK'/dPhi*dt
        //   dR'/dPi = 0
        let dr_dr = 1.0 - 2.0 * alpha_lapse * dk_dr * dt;
        let dr_dk = -2.0 * alpha_lapse * dk_dk * dt;
        let dr_dphi = -2.0 * alpha_lapse * dk_dphi * dt;

        // Phi' = Phi + Pi'*dt
        //   dPhi'/dR = dPi'/dR*dt
        //   dPhi'/dK = 0
        //   dPhi'/dPhi = 1 + dPi'/dPhi*dt
        //   dPhi'/dPi = dPi'/dPi*dt
        let dphi_dr = dpi_dr * dt;
        let dphi_dphi = 1.0 + dpi_dphi * dt;
        let dphi_dpi = dpi_dpi * dt;

        //             R        K        Phi        Pi
        [
            [dr_dr,    dr_dk,   dr_dphi,   0.0      ],  // R'
            [dk_dr,    dk_dk,   dk_dphi,   0.0      ],  // K'
            [dphi_dr,  0.0,     dphi_dphi, dphi_dpi ],  // Phi'
            [dpi_dr,   0.0,     dpi_dphi,  dpi_dpi  ],  // Pi'
        ]
    }

    fn predict(&mut self, dt: f64, alpha_lapse: f64, beta_damp: f64,
               xi: f64, m_phi: f64, gamma_phi: f64) {
        // Jacobian at OLD state (before evolution) -- EKF requires this
        let a = self.transition_matrix(dt, alpha_lapse, beta_damp, xi, m_phi, gamma_phi);

        let r = self.x[0];
        let kk = self.x[1];
        let phi = self.x[2];
        let pi = self.x[3];

        let dk = alpha_lapse * r / 2.0 - beta_damp * kk + xi * phi * phi;
        let dpi = -(m_phi * m_phi + xi * r) * phi - gamma_phi * pi;

        let k_new = kk + dk * dt;
        let pi_new = pi + dpi * dt;

        let r_new = r + (-2.0 * alpha_lapse * k_new) * dt;
        let phi_new = phi + pi_new * dt;

        self.x = [r_new, k_new, phi_new, pi_new];

        // P = A * P * A^T + Q * q_factor
        let mut ap = [[0.0_f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    ap[i][j] += a[i][k] * self.p[k][j];
                }
            }
        }

        let q_factor = (-2.0 * xi * self.x[0].abs()).exp();
        let mut p_new = [[0.0_f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    p_new[i][j] += ap[i][k] * a[j][k];
                }
                p_new[i][j] += self.q[i][j] * q_factor;
            }
        }

        self.p = p_new;
    }

    /// General scalar Kalman update for observation z with H vector and noise r_meas.
    fn update_observation(&mut self, z: f64, h: [f64; 4], r_meas: f64) {
        let mut z_pred = 0.0;
        for i in 0..4 {
            z_pred += h[i] * self.x[i];
        }

        // S = H * P * H^T + R
        let mut hph = 0.0_f64;
        for i in 0..4 {
            for j in 0..4 {
                hph += h[i] * self.p[i][j] * h[j];
            }
        }
        let s = hph + r_meas;
        if s.abs() < 1e-20 {
            return;
        }
        let s_inv = 1.0 / s;
        let innovation = z - z_pred;

        // K = P * H^T / S
        let mut k = [0.0_f64; 4];
        for i in 0..4 {
            for j in 0..4 {
                k[i] += self.p[i][j] * h[j];
            }
            k[i] *= s_inv;
        }

        for i in 0..4 {
            self.x[i] += k[i] * innovation;
        }

        // P = (I - K*H) * P
        let mut hp = [0.0_f64; 4];
        for j in 0..4 {
            for l in 0..4 {
                hp[j] += h[l] * self.p[l][j];
            }
        }

        for i in 0..4 {
            for j in 0..4 {
                self.p[i][j] -= k[i] * hp[j];
            }
        }

        // Symmetrize + floor
        for i in 0..4 {
            for j in (i + 1)..4 {
                let avg = 0.5 * (self.p[i][j] + self.p[j][i]);
                self.p[i][j] = avg;
                self.p[j][i] = avg;
            }
            if self.p[i][i] < 1e-15 {
                self.p[i][i] = 1e-15;
            }
        }
    }

    fn predict_future(&self, t: f64, alpha_lapse: f64, beta_damp: f64,
                      xi: f64, m_phi: f64, gamma_phi: f64) -> [f64; 4] {
        let steps = (t / 0.001).ceil() as usize;
        if steps == 0 {
            return self.x;
        }
        let sub_dt = t / steps as f64;

        let mut x = self.x;

        for _ in 0..steps {
            // Symplectic Euler: velocities first, then positions
            let dk = alpha_lapse * x[0] / 2.0 - beta_damp * x[1] + xi * x[2] * x[2];
            let dpi = -(m_phi * m_phi + xi * x[0]) * x[2] - gamma_phi * x[3];

            x[1] += dk * sub_dt;
            x[3] += dpi * sub_dt;

            x[0] += (-2.0 * alpha_lapse * x[1]) * sub_dt;
            x[2] += x[3] * sub_dt;
        }

        x
    }
}

/// SFE 3+1 ARC Controller
pub struct SfeArcController {
    pub alpha: f64,
    pub beta: f64,
    pub latency: usize,

    state: SfeState,
    pub r_meas: f64,
    prev_z: Option<f64>,

    alpha_lapse: f64,
    beta_damp: f64,
    xi: f64,
    m_phi: f64,
    gamma_phi: f64,
}

impl SfeArcController {
    pub fn new(alpha: f64, beta: f64, latency: usize,
               process_noise: f64, measure_noise: f64) -> Self {
        let (alpha_lapse, beta_damp, xi, m_phi, gamma_phi) = sfe_physics_params();
        let dt = 0.01;

        let r_meas = measure_noise * measure_noise / 3.0;

        Self {
            alpha,
            beta,
            latency,
            state: SfeState::new(process_noise, dt),
            r_meas,
            prev_z: None,
            alpha_lapse,
            beta_damp,
            xi,
            m_phi,
            gamma_phi,
        }
    }

    pub fn step(&mut self, measured_r: f64, dt: f64) -> f64 {
        self.state.predict(dt, self.alpha_lapse, self.beta_damp,
                           self.xi, self.m_phi, self.gamma_phi);

        // Obs 1: R (curvature-dependent noise)
        let r_meas = self.r_meas * (-2.0 * self.xi * self.state.x[0].abs()).exp();
        self.state.update_observation(measured_r, [1.0, 0.0, 0.0, 0.0], r_meas);

        // Obs 2: dR/dt = -2*alpha*K (velocity from forward difference)
        if let Some(prev) = self.prev_z {
            let dr_dt = (measured_r - prev) / dt;
            let h1 = -2.0 * self.alpha_lapse;
            let r_meas_vel = 2.0 * self.r_meas / (dt * dt);
            self.state.update_observation(dr_dt, [0.0, h1, 0.0, 0.0], r_meas_vel);
        }
        self.prev_z = Some(measured_r);

        let lookahead = self.latency as f64 * dt;
        let x_future = self.state.predict_future(
            lookahead, self.alpha_lapse, self.beta_damp,
            self.xi, self.m_phi, self.gamma_phi,
        );

        -(self.alpha * x_future[0] + self.beta * x_future[1])
    }

    /// Obs 3: error feedback. residual - pulse = alpha*R + beta*K.
    pub fn update_error(&mut self, error_signal: f64, r_meas_err: f64) {
        self.state.update_observation(
            error_signal,
            [self.alpha, self.beta, 0.0, 0.0],
            r_meas_err,
        );
    }

    pub fn mode_energies(&self) -> Vec<f64> {
        vec![
            self.state.x[0] * self.state.x[0],
            self.state.x[2] * self.state.x[2],
        ]
    }

    pub fn total_estimation_uncertainty(&self) -> f64 {
        self.state.p[0][0]
    }
}

/// Simulation: 3+1 SFE coupled dynamics
pub struct ArcSimulationEnv {
    true_state: [f64; 4],
    process_noise: f64,
    measure_noise: f64,

    alpha_lapse: f64,
    beta_damp: f64,
    xi: f64,
    m_phi: f64,
    gamma_phi: f64,

    pub controller: SfeArcController,
    pulse_queue: VecDeque<f64>,
    prev_residual: f64,
    prev_delayed_pulse: f64,
}

impl Default for ArcSimulationEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl ArcSimulationEnv {
    pub fn new() -> Self {
        Self::with_params(0.08, 0.005, 2)
    }

    pub fn with_params(process_noise: f64, measure_noise: f64, latency: usize) -> Self {
        // Initial: R=1.0, K=0.3, Phi=0.5, Pi=0.0
        let true_state = [1.0, 0.3, 0.5, 0.0];

        let (alpha_lapse, beta_damp, xi, m_phi, gamma_phi) = sfe_physics_params();

        let mut env = Self {
            true_state,
            process_noise,
            measure_noise,
            alpha_lapse,
            beta_damp,
            xi,
            m_phi,
            gamma_phi,
            controller: SfeArcController::new(1.0, 0.1, latency, process_noise, measure_noise),
            pulse_queue: VecDeque::with_capacity(latency),
            prev_residual: 0.0,
            prev_delayed_pulse: 0.0,
        };

        for _ in 0..latency {
            env.pulse_queue.push_back(0.0);
        }

        env
    }

    pub fn step(&mut self, dt: f64) -> (f64, f64) {
        let mut rng = rand::thread_rng();

        let r = self.true_state[0];
        let k = self.true_state[1];
        let phi = self.true_state[2];
        let pi = self.true_state[3];

        let noise_k: f64 = rng.gen_range(-1.0..1.0);
        let noise_pi: f64 = rng.gen_range(-1.0..1.0);

        // Process noise suppressed by curvature: high |R| -> more folding -> less random
        let noise_suppression = (-self.xi * r.abs()).exp();

        let dk = self.alpha_lapse * r / 2.0 - self.beta_damp * k
            + self.xi * phi * phi
            + self.process_noise * noise_k * dt.sqrt() * noise_suppression;
        let dpi = -(self.m_phi * self.m_phi + self.xi * r) * phi
            - self.gamma_phi * pi
            + self.process_noise * 0.5 * noise_pi * dt.sqrt() * noise_suppression;

        let k_new = k + dk * dt;
        let pi_new = pi + dpi * dt;

        let r_new = r + (-2.0 * self.alpha_lapse * k_new) * dt;
        let phi_new = phi + pi_new * dt;

        self.true_state = [r_new, k_new, phi_new, pi_new];

        let actual_noise_influence =
            self.controller.alpha * r_new + self.controller.beta * k_new;

        let meas_noise: f64 = rng.gen_range(-1.0..1.0);
        let curvature_suppression = (-self.xi * r_new.abs()).exp();
        let measured_r = r_new + self.measure_noise * meas_noise * curvature_suppression;

        // Feed back previous residual as error observation before controller step
        let error_signal = self.prev_residual - self.prev_delayed_pulse;
        self.controller.update_error(error_signal, R_MEAS_ERR);

        let new_pulse = self.controller.step(measured_r, dt);

        self.pulse_queue.push_back(new_pulse);
        let delayed_pulse = self.pulse_queue.pop_front().unwrap_or(0.0);

        let residual = actual_noise_influence + delayed_pulse;

        self.prev_residual = residual;
        self.prev_delayed_pulse = delayed_pulse;

        (actual_noise_influence, residual)
    }
}
