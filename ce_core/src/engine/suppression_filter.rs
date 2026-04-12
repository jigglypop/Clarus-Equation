use rayon::prelude::*;

pub struct LyapunovSuppressionFilter {
    pub alpha: f64,
    pub gamma: f64,
    pub e_scale: f64,
    pub sigma_c: f64,
}

impl LyapunovSuppressionFilter {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        Self {
            alpha,
            gamma,
            e_scale: 1.0,
            sigma_c: 1.5,
        }
    }

    pub fn new_with_scale(alpha: f64, gamma: f64, e_scale: f64) -> Self {
        Self {
            alpha,
            gamma,
            e_scale,
            sigma_c: 1.5,
        }
    }

    #[inline]
    pub fn suppression_factor(&self, energy: f64) -> f64 {
        self.alpha * (-self.gamma * energy / self.e_scale).exp()
    }

    #[inline]
    pub fn effective_sigma(&self, energy: f64) -> f64 {
        self.suppression_factor(energy) / self.sigma_c
    }

    pub fn apply_to_signal(&self, signal: &mut [f64]) {
        let n = signal.len();
        if n == 0 {
            return;
        }

        let energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let sup = self.suppression_factor(energy);

        signal.par_iter_mut().for_each(|x| {
            *x *= 1.0 - sup;
        });
    }

    pub fn apply_adaptive(&self, signal: &mut [f64], window_size: usize) {
        let n = signal.len();
        if n == 0 || window_size == 0 {
            return;
        }

        let half_window = window_size / 2;

        let start = 0usize;
        let end = half_window.min(n);
        let mut window_sum: f64 = signal[start..end].iter().map(|x| x * x).sum();
        let mut window_count = end - start;

        let mut suppression_factors = Vec::with_capacity(n);

        for i in 0..n {
            let new_end = (i + half_window).min(n);
            let new_start = i.saturating_sub(half_window);

            if i > 0 {
                if new_end > (i - 1 + half_window).min(n) {
                    let added = new_end - 1;
                    if added < n {
                        window_sum += signal[added] * signal[added];
                        window_count += 1;
                    }
                }
                let old_start = (i - 1).saturating_sub(half_window);
                if new_start > old_start {
                    window_sum -= signal[old_start] * signal[old_start];
                    window_count -= 1;
                }
            }

            let local_energy = if window_count > 0 {
                window_sum / window_count as f64
            } else {
                0.0
            };
            suppression_factors.push(self.suppression_factor(local_energy));
        }

        signal
            .par_iter_mut()
            .zip(suppression_factors.par_iter())
            .for_each(|(x, sup)| {
                *x *= 1.0 - sup;
            });
    }

    pub fn compute_lyapunov_reduction(&self, signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }

        let energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        let sup = self.suppression_factor(energy);

        sup * self.sigma_c
    }

    pub fn is_in_stable_regime(&self, signal: &[f64]) -> bool {
        let sigma = self.compute_lyapunov_reduction(signal);
        sigma > 1.0
    }
}

pub struct AdaptiveSuppressionController {
    pub filter: LyapunovSuppressionFilter,
    pub target_sigma: f64,
    pub learning_rate: f64,
    pub history: Vec<f64>,
    integral_error: f64,
    ki: f64,
}

impl AdaptiveSuppressionController {
    pub fn new(target_sigma: f64) -> Self {
        Self {
            filter: LyapunovSuppressionFilter::new(1.0, 1.0),
            target_sigma,
            learning_rate: 0.1,
            history: Vec::new(),
            integral_error: 0.0,
            ki: 0.02,
        }
    }

    pub fn update(&mut self, signal: &[f64]) {
        let current_sigma = self.filter.compute_lyapunov_reduction(signal);
        self.history.push(current_sigma);

        let error = self.target_sigma - current_sigma;
        self.integral_error += error;
        self.integral_error = self.integral_error.clamp(-50.0, 50.0);

        let correction = self.learning_rate * error + self.ki * self.integral_error;
        self.filter.alpha += correction;
        self.filter.alpha = self.filter.alpha.clamp(0.1, 10.0);
    }

    pub fn apply(&self, signal: &mut [f64]) {
        self.filter.apply_to_signal(signal);
    }

    pub fn get_convergence_status(&self) -> ConvergenceStatus {
        if self.history.len() < 10 {
            return ConvergenceStatus::Initializing;
        }

        let recent: Vec<f64> = self.history.iter().rev().take(10).cloned().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 =
            recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        let std = variance.sqrt();

        if std < 0.05 * mean.abs().max(1e-12) {
            if mean > self.target_sigma * 0.9 && mean < self.target_sigma * 1.1 {
                ConvergenceStatus::Converged
            } else {
                ConvergenceStatus::Stable
            }
        } else {
            ConvergenceStatus::Adapting
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceStatus {
    Initializing,
    Adapting,
    Stable,
    Converged,
}

pub fn apply_lyapunov_suppression(noise: &mut [f64], alpha: f64, gamma: f64) {
    let filter = LyapunovSuppressionFilter::new(alpha, gamma);
    filter.apply_to_signal(noise);
}

pub fn apply_adaptive_lyapunov_suppression(
    noise: &mut [f64],
    alpha: f64,
    gamma: f64,
    window_size: usize,
) {
    let filter = LyapunovSuppressionFilter::new(alpha, gamma);
    filter.apply_adaptive(noise, window_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suppression_factor() {
        let filter = LyapunovSuppressionFilter::new(1.0, 1.0);

        let sup_low = filter.suppression_factor(0.1);
        let sup_high = filter.suppression_factor(10.0);

        assert!(sup_low > sup_high);
        assert!(sup_low > 0.0);
        assert!(sup_high > 0.0);
    }

    #[test]
    fn test_signal_suppression() {
        let filter = LyapunovSuppressionFilter::new(0.5, 1.0);

        let mut signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let original_energy: f64 = signal.iter().map(|x| x * x).sum();

        filter.apply_to_signal(&mut signal);
        let new_energy: f64 = signal.iter().map(|x| x * x).sum();

        assert!(new_energy < original_energy);
    }

    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveSuppressionController::new(1.5);

        let signal = vec![1.0; 100];
        for _ in 0..20 {
            controller.update(&signal);
        }

        assert!(controller.history.len() == 20);
    }

    #[test]
    fn test_stability_check() {
        let filter = LyapunovSuppressionFilter::new(2.0, 0.5);

        let low_energy_signal = vec![0.1; 100];
        let _high_energy_signal = vec![10.0; 100];

        assert!(filter.is_in_stable_regime(&low_energy_signal));
    }
}
