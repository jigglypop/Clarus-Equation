use rand::prelude::*;
use rand_distr::StandardNormal;
use rustfft::{FftPlanner, num_complex::Complex, Fft};
use rustfft::num_traits::Zero;
use std::sync::Arc;

pub struct PinkNoiseGenerator {
    fft: Arc<dyn Fft<f64>>,
    steps: usize,
    spectrum_buffer: Vec<Complex<f64>>,
    alpha: f64, // 1/f^alpha (기본값 1.0)
    scale: f64, // 전체 진폭 스케일링
}

impl PinkNoiseGenerator {
    pub fn new(steps: usize) -> Self {
        Self::new_with_params(steps, 1.0, 1.0)
    }

    pub fn new_with_params(steps: usize, alpha: f64, scale: f64) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(steps);
        Self {
            fft,
            steps,
            spectrum_buffer: vec![Complex::zero(); steps],
            alpha,
            scale,
        }
    }

    /// IFFT를 사용하여 1/f^alpha 노이즈 생성
    pub fn generate(&mut self, output: &mut [f64]) {
        if output.len() != self.steps {
            panic!("출력 버퍼 길이는 생성기 스텝과 일치해야 합니다");
        }

        let mut rng = thread_rng();
        let steps = self.steps;
        
        // DC 성분
        let dc_real: f64 = rng.sample(StandardNormal);
        self.spectrum_buffer[0] = Complex::new(dc_real * (steps as f64).sqrt() * self.scale, 0.0);
        
        for i in 1..steps {
            let f = if i <= steps/2 { i as f64 } else { (steps - i) as f64 };
            // 전력 ~ 1/f^alpha => 진폭 ~ 1/f^(alpha/2)
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

pub fn generate_pink_noise_with_params(steps: usize, alpha: f64, scale: f64) -> Vec<f64> {
    let mut gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);
    gen.generate_new()
}

pub fn generate_correlated_pink_noise(
    steps: usize,
    qubits: usize,
    alpha: f64,
    scale: f64,
    rho: f64,
) -> Vec<Vec<f64>> {
    let r = if rho < 0.0 {
        0.0
    } else if rho > 1.0 {
        1.0
    } else {
        rho
    };

    let w_common = r.sqrt();
    let w_indiv = (1.0 - r).sqrt();

    let mut common_gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);
    let mut indiv_gen = PinkNoiseGenerator::new_with_params(steps, alpha, scale);

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
