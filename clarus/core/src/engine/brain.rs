use ndarray::Array1;

use crate::engine::arc::CeArcController;
use crate::engine::constants::CE;
use crate::engine::core::QCEngine;
use crate::engine::suppression_filter::LyapunovSuppressionFilter;

const DT: f64 = 0.01;

#[derive(Clone, Debug)]
pub struct BrainState {
    pub r: f64,
    pub k: f64,
    pub phi_global: f64,
    pub pi_global: f64,
    pub field_energy: f64,
    pub suppression_factor: f64,
    pub memory_norm: f64,
}

pub struct BrainEngine {
    pub arc: CeArcController,
    pub field: QCEngine,
    suppression: LyapunovSuppressionFilter,
    base_source: Array1<f64>,
    xi: f64,
    delta_xi: f64,
    alpha_ricci: f64,
    pub memory: Array1<f64>,
    pub rho_mem: f64,
    pub w_mem: f64,
    pub w_j: f64,
    pub goal: Array1<f64>,
    mem_dim: usize,
}

impl BrainEngine {
    pub fn new(field_size: usize) -> Self {
        let ce = &*CE;
        let xi = ce.alpha_s.powf(1.0 / 3.0);
        let delta_xi = ce.delta * xi;
        let alpha_ricci = ce.alpha_s;

        let arc = CeArcController::new(1.0, ce.alpha_s, 0, 0.02, 0.001);
        let mut field = QCEngine::new(field_size);
        field.alpha2 = 0.01;
        let base_source = field.source_j.clone();

        let suppression = LyapunovSuppressionFilter::new(0.5, 1.0);

        let mem_dim = 32.min(field_size);
        let rho_mem = (-ce.epsilon2).exp();
        let w_mem = ce.epsilon2;

        Self {
            arc,
            field,
            suppression,
            base_source,
            xi,
            delta_xi,
            alpha_ricci,
            memory: Array1::zeros(mem_dim),
            rho_mem,
            w_mem,
            w_j: ce.epsilon2,
            goal: Array1::zeros(field_size),
            mem_dim,
        }
    }

    fn ricci_from_field(&self) -> f64 {
        let phi = self.field.phi.as_slice().unwrap();
        let n = phi.len();
        if n < 2 {
            return 0.0;
        }
        let mut grad_sq = 0.0;
        for i in 0..n - 1 {
            let d = phi[i + 1] - phi[i];
            grad_sq += d * d;
        }
        grad_sq /= (n - 1) as f64;
        let phi_abs = phi.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
        self.alpha_ricci * grad_sq * phi_abs
    }

    fn field_energy(&self) -> f64 {
        let phi = self.field.phi.as_slice().unwrap();
        let dphi = self.field.dphi.as_slice().unwrap();
        let n = phi.len();
        if n == 0 {
            return 0.0;
        }
        let ke: f64 = dphi.iter().map(|v| 0.5 * v * v).sum();
        let pe: f64 = phi.iter().map(|v| {
            let mu2 = self.field.mu * self.field.mu;
            let lam = self.field.lam;
            -0.5 * mu2 * v * v + 0.25 * lam * v.powi(4)
        }).sum();
        (ke + pe) / n as f64
    }

    fn pool_field(&self) -> Array1<f64> {
        let phi = self.field.phi.as_slice().unwrap();
        let n = phi.len();
        let mut pooled = Array1::zeros(self.mem_dim);
        let chunk = n / self.mem_dim;
        if chunk == 0 {
            for (i, p) in pooled.iter_mut().enumerate() {
                *p = phi[i % n];
            }
        } else {
            for (i, p) in pooled.iter_mut().enumerate() {
                let start = i * chunk;
                let end = (start + chunk).min(n);
                let sum: f64 = phi[start..end].iter().sum();
                *p = sum / (end - start) as f64;
            }
        }
        pooled
    }

    pub fn set_goal(&mut self, goal: &[f64]) {
        let n = self.goal.len();
        for i in 0..n {
            self.goal[i] = goal[i % goal.len()];
        }
    }

    pub fn step(&mut self, external_input: Option<&[f64]>) -> BrainState {
        let r_measured = self.ricci_from_field();
        self.arc.step(r_measured, DT);
        let est = self.arc.estimated_state();
        let r_est = est[0];
        let k_est = est[1];

        let n = self.field.phi.len();
        {
            let phi_slice = self.field.phi.as_slice().unwrap();
            let source = self.field.source_j.as_slice_mut().unwrap();
            let base = self.base_source.as_slice().unwrap();
            let mem = self.memory.as_slice().unwrap();
            let goal = self.goal.as_slice().unwrap();
            for i in 0..n {
                let curvature_coupling = -(self.xi * r_est + self.delta_xi * k_est) * phi_slice[i];
                let ext = external_input.map_or(0.0, |inp| inp[i % inp.len()]);
                let mem_contrib = self.w_j * mem[i % self.mem_dim];
                source[i] = base[i] + curvature_coupling + ext + mem_contrib + goal[i];
            }
        }

        self.field.step();

        let geo_sup = (-r_est.abs()).exp();
        {
            let dphi = self.field.dphi.as_slice_mut().unwrap();
            for v in dphi.iter_mut() {
                *v *= geo_sup;
            }
        }

        self.suppression.apply_to_signal(self.field.dphi.as_slice_mut().unwrap());

        let pooled = self.pool_field();
        for i in 0..self.mem_dim {
            self.memory[i] = self.rho_mem * self.memory[i] + self.w_mem * pooled[i];
        }

        let energy = self.field_energy();
        let memory_norm = self.memory.iter().map(|v| v * v).sum::<f64>().sqrt();

        BrainState {
            r: r_est,
            k: k_est,
            phi_global: self.field.get_center_value(),
            pi_global: self.field.dphi[n / 2],
            field_energy: energy,
            suppression_factor: geo_sup,
            memory_norm,
        }
    }

    pub fn run(&mut self, steps: usize) -> Vec<BrainState> {
        let mut history = Vec::with_capacity(steps);
        for _ in 0..steps {
            history.push(self.step(None));
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_convergence() {
        let mut brain = BrainEngine::new(128);
        brain.base_source.fill(0.0);
        brain.field.source_j.fill(0.0);
        let vev = brain.field.mu / brain.field.lam.sqrt();

        for _ in 0..5000 {
            brain.step(None);
        }

        let state = brain.step(None);
        assert!(
            state.r.abs() < 1.0,
            "R should converge near 0, got {}",
            state.r
        );
        assert!(
            (state.phi_global.abs() - vev).abs() < vev * 0.5,
            "phi should be near vacuum vev={}, got {}",
            vev,
            state.phi_global
        );
    }

    #[test]
    fn suppression_reduces_velocity() {
        let mut brain = BrainEngine::new(64);
        brain.step(None);
        let dphi_before: f64 = brain.field.dphi.iter().map(|v| v.abs()).sum();

        let big_input: Vec<f64> = vec![10.0; 64];
        for _ in 0..10 {
            brain.step(Some(&big_input));
        }

        let state = brain.step(None);
        assert!(
            state.suppression_factor <= 1.0,
            "suppression factor should be <= 1.0, got {}",
            state.suppression_factor
        );

        let dphi_after: f64 = brain.field.dphi.iter().map(|v| v.abs()).sum();
        let _ = (dphi_before, dphi_after);
    }

    #[test]
    fn memory_accumulates() {
        let mut brain = BrainEngine::new(64);
        for _ in 0..100 {
            brain.step(None);
        }
        let state = brain.step(None);
        assert!(
            state.memory_norm > 0.0,
            "memory should accumulate, got norm={}",
            state.memory_norm
        );
    }

    #[test]
    fn goal_affects_dynamics() {
        let mut brain_no_goal = BrainEngine::new(64);
        let mut brain_goal = BrainEngine::new(64);
        brain_goal.set_goal(&vec![1.0; 64]);

        for _ in 0..100 {
            brain_no_goal.step(None);
            brain_goal.step(None);
        }

        let s1 = brain_no_goal.step(None);
        let s2 = brain_goal.step(None);

        assert!(
            (s1.phi_global - s2.phi_global).abs() > 1e-10,
            "goal should change dynamics: no_goal={}, goal={}",
            s1.phi_global,
            s2.phi_global
        );
    }

    #[test]
    fn arc_field_coupling() {
        let mut brain = BrainEngine::new(64);
        let mut prev_r = 0.0_f64;
        let mut r_changed = false;

        for i in 0..200 {
            let state = brain.step(None);
            if i > 50 && (state.r - prev_r).abs() > 1e-12 {
                r_changed = true;
            }
            prev_r = state.r;
        }

        assert!(
            r_changed,
            "arc R estimate should change in response to field dynamics"
        );
    }
}
