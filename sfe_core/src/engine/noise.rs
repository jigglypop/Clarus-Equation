use rand::prelude::*;
use rand_distr::StandardNormal;
use rustfft::{FftPlanner, num_complex::Complex, Fft};
use rustfft::num_traits::Zero;
use std::sync::Arc;

/// caching the FFT planner to avoid re-initialization overhead.
pub struct PinkNoiseGenerator {
    planner: FftPlanner<f64>,
    fft: Arc<dyn Fft<f64>>,
    steps: usize,
    spectrum_buffer: Vec<Complex<f64>>,
    alpha: f64, // 1/f^alpha (Default 1.0)
    scale: f64, // Overall amplitude scaling
}

impl PinkNoiseGenerator {
    pub fn new(steps: usize) -> Self {
        Self::new_with_params(steps, 1.0, 1.0)
    }

    pub fn new_with_params(steps: usize, alpha: f64, scale: f64) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(steps);
        Self {
            planner, 
            fft,
            steps,
            spectrum_buffer: vec![Complex::zero(); steps],
            alpha,
            scale,
        }
    }

    /// Generates 1/f^alpha Noise using IFFT
    pub fn generate(&mut self, output: &mut [f64]) {
        if output.len() != self.steps {
            panic!("Output buffer length must match generator steps");
        }

        let mut rng = thread_rng();
        let steps = self.steps;
        
        // DC component
        let dc_real: f64 = rng.sample(StandardNormal);
        self.spectrum_buffer[0] = Complex::new(dc_real * (steps as f64).sqrt() * self.scale, 0.0);
        
        for i in 1..steps {
            let f = if i <= steps/2 { i as f64 } else { (steps - i) as f64 };
            // Power ~ 1/f^alpha => Amp ~ 1/f^(alpha/2)
            let amplitude = self.scale / f.powf(self.alpha / 2.0); 
            
            let real: f64 = rng.sample(StandardNormal);
            let imag: f64 = rng.sample(StandardNormal);
            
            self.spectrum_buffer[i] = Complex::new(real * amplitude, imag * amplitude);
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

pub fn generate_pink_noise(steps: usize) -> Vec<f64> {
    let mut gen = PinkNoiseGenerator::new(steps);
    gen.generate_new()
}
