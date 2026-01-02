use rand::prelude::*;
use rand_distr::StandardNormal;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LorentzianPeak {
    pub omega_center: f64,
    pub gamma: f64,
    pub amplitude: f64,
}

pub struct MultiPeakNoiseGenerator {
    steps: usize,
    pink_alpha: f64,
    pink_scale: f64,
    lorentzian_peaks: Vec<LorentzianPeak>,
    white_level: f64,
    fft: Arc<dyn Fft<f64>>,
    spectrum_buffer: Vec<Complex<f64>>,
}

impl MultiPeakNoiseGenerator {
    pub fn new_with_peaks(
        steps: usize,
        alpha: f64,
        scale: f64,
        peaks: Vec<LorentzianPeak>,
        white: f64,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(steps);
        Self {
            steps,
            pink_alpha: alpha,
            pink_scale: scale,
            lorentzian_peaks: peaks,
            white_level: white,
            fft,
            spectrum_buffer: vec![Complex::zero(); steps],
        }
    }

    pub fn spectrum_density(&self, omega: f64) -> f64 {
        if omega < 1e-10 {
            return self.white_level;
        }

        let pink = self.pink_scale / omega.powf(self.pink_alpha);

        let lorentz: f64 = self
            .lorentzian_peaks
            .iter()
            .map(|p| {
                let denom = (omega - p.omega_center).powi(2) + p.gamma.powi(2);
                p.amplitude * p.gamma / denom
            })
            .sum();

        pink + lorentz + self.white_level
    }

    pub fn generate(&mut self, output: &mut [f64]) {
        if output.len() != self.steps {
            panic!("출력 버퍼 길이는 생성기 스텝과 일치해야 합니다");
        }

        let mut rng = thread_rng();
        let steps = self.steps;

        for i in 0..steps {
            let f = if i <= steps / 2 {
                i as f64
            } else {
                (steps - i) as f64
            };
            let omega = 2.0 * PI * f / (steps as f64);

            let s_omega = self.spectrum_density(omega).sqrt();

            let real: f64 = rng.sample(StandardNormal);
            let imag: f64 = rng.sample(StandardNormal);

            self.spectrum_buffer[i] = Complex::new(real * s_omega, imag * s_omega);
        }

        self.fft.process(&mut self.spectrum_buffer);

        for (i, val) in self.spectrum_buffer.iter().enumerate() {
            output[i] = val.re;
        }
    }

    pub fn generate_new(&mut self) -> Vec<f64> {
        let mut out = vec![0.0; self.steps];
        self.generate(&mut out);
        out
    }
}

pub fn diagnose_tls_spectrum(t1: f64, t2: f64, dt_us: f64) -> Vec<LorentzianPeak> {
    let rate_t2 = 1.0 / t2;
    let rate_t1_limit = 1.0 / (2.0 * t1);
    let gamma_phi = rate_t2 - rate_t1_limit;

    if gamma_phi < 0.01 {
        return vec![];
    }

    let omega_tls_physical = 0.2713;
    let omega_tls_sim = omega_tls_physical * dt_us;

    let amp = gamma_phi.sqrt() * 2.0;

    vec![
        LorentzianPeak {
            omega_center: omega_tls_sim,
            gamma: 0.05 * dt_us,
            amplitude: amp,
        },
        LorentzianPeak {
            omega_center: omega_tls_sim * 0.5,
            gamma: 0.03 * dt_us,
            amplitude: amp * 0.3,
        },
    ]
}

pub fn generate_correlated_multi_peak_noise(
    steps: usize,
    qubits: usize,
    alpha: f64,
    scale: f64,
    peaks: Vec<LorentzianPeak>,
    white: f64,
    rho: f64,
) -> Vec<Vec<f64>> {
    let r = rho.clamp(0.0, 1.0);
    let w_common = r.sqrt();
    let w_indiv = (1.0 - r).sqrt();

    let mut common_gen =
        MultiPeakNoiseGenerator::new_with_peaks(steps, alpha, scale, peaks.clone(), white);
    let mut indiv_gen = MultiPeakNoiseGenerator::new_with_peaks(steps, alpha, scale, peaks, white);

    let common = common_gen.generate_new();

    let mut traces = Vec::with_capacity(qubits);
    for _ in 0..qubits {
        let indiv = indiv_gen.generate_new();
        let mut v = Vec::with_capacity(steps);
        for t in 0..steps {
            let val = w_common * common[t] + w_indiv * indiv[t];
            v.push(val);
        }
        traces.push(v);
    }

    traces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_peak_generation() {
        let peaks = vec![LorentzianPeak {
            omega_center: 0.1,
            gamma: 0.01,
            amplitude: 1.0,
        }];

        let mut gen = MultiPeakNoiseGenerator::new_with_peaks(1000, 0.8, 1.0, peaks, 0.01);

        let noise = gen.generate_new();
        assert_eq!(noise.len(), 1000);

        let mean: f64 = noise.iter().sum::<f64>() / noise.len() as f64;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_tls_diagnosis() {
        let peaks = diagnose_tls_spectrum(60.0, 40.0, 0.05);
        assert_eq!(peaks.len(), 2);
        assert!(peaks[0].omega_center > 0.0);
    }
}
