use crate::packed_params::TransformType;
use std::f32::consts::PI;

/// 변환 알고리즘 성능 분석기
#[derive(Debug, Clone)]
pub struct TransformAnalyzer {
    pub dct_performance: f32,
    pub wavelet_performance: f32,
    pub smoothness_threshold: f32,
    pub frequency_concentration_threshold: f32,
}

impl TransformAnalyzer {
    pub fn new() -> Self {
        Self {
            dct_performance: 0.0,
            wavelet_performance: 0.0,
            smoothness_threshold: 0.1,
            frequency_concentration_threshold: 2.0,
        }
    }
    
    /// 신호의 평활도 측정
    pub fn measure_smoothness(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        let mut total_variation = 0.0;
        for i in 1..signal.len() {
            total_variation += (signal[i] - signal[i-1]).abs();
        }
        
        total_variation / (signal.len() - 1) as f32
    }
    
    /// 주파수 집중도 측정 (DCT 기반 에너지 집중도)
    pub fn measure_frequency_concentration(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        
        // 간단한 DCT 근사
        let mut dct_coeffs = vec![0.0; signal.len()];
        for k in 0..signal.len() {
            let mut sum = 0.0;
            for n in 0..signal.len() {
                let angle = PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * signal.len() as f32);
                sum += signal[n] * angle.cos();
            }
            dct_coeffs[k] = sum;
        }
        
        // 에너지 기반 집중도 계산
        let mut energy_coeffs: Vec<f32> = dct_coeffs.iter().map(|x| x * x).collect();
        energy_coeffs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let total_energy: f32 = energy_coeffs.iter().sum();
        if total_energy < 1e-8 {
            return 0.0;
        }
        
        // 상위 25% 계수들이 차지하는 에너지 비율
        let top_count = (signal.len() / 4).max(1);
        let top_energy: f32 = energy_coeffs.iter().take(top_count).sum();
        
        top_energy / total_energy
    }
    
    /// 최적 변환 방법 선택
    pub fn select_optimal_transform(&self, signal: &[f32]) -> TransformType {
        let smoothness = self.measure_smoothness(signal);
        let concentration = self.measure_frequency_concentration(signal);
        
        if smoothness < self.smoothness_threshold && concentration > self.frequency_concentration_threshold {
            TransformType::Dct
        } else {
            TransformType::Dwt
        }
    }
    
    /// DCT 압축 성능 측정
    pub fn measure_dct_performance(&mut self, original: &[f32], compression_ratio: f32) -> f32 {
        if original.is_empty() {
            return 0.0;
        }
        
        // DCT 변환
        let mut dct_coeffs = vec![0.0; original.len()];
        for k in 0..original.len() {
            let mut sum = 0.0;
            let weight = if k == 0 { (1.0 / original.len() as f32).sqrt() } else { (2.0 / original.len() as f32).sqrt() };
            
            for n in 0..original.len() {
                let angle = PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * original.len() as f32);
                sum += original[n] * angle.cos();
            }
            dct_coeffs[k] = weight * sum;
        }
        
        // 압축 (상위 계수만 유지)
        let keep_count = ((1.0 - compression_ratio) * original.len() as f32) as usize;
        let keep_count = keep_count.max(1).min(original.len());
        
        for i in keep_count..dct_coeffs.len() {
            dct_coeffs[i] = 0.0;
        }
        
        // 역변환
        let mut reconstructed = vec![0.0; original.len()];
        for n in 0..original.len() {
            let mut sum = 0.0;
            for k in 0..original.len() {
                let weight = if k == 0 { (1.0 / original.len() as f32).sqrt() } else { (2.0 / original.len() as f32).sqrt() };
                let angle = PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * original.len() as f32);
                sum += weight * dct_coeffs[k] * angle.cos();
            }
            reconstructed[n] = sum;
        }
        
        // RMSE 계산
        let mse = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        let rmse = mse.sqrt();
        self.dct_performance = rmse;
        rmse
    }
    
    /// 웨이블릿 압축 성능 측정 (간단한 Haar 웨이블릿)
    pub fn measure_wavelet_performance(&mut self, original: &[f32], compression_ratio: f32) -> f32 {
        if original.is_empty() || original.len() < 2 {
            return 0.0;
        }
        
        let mut data = original.to_vec();
        let mut temp = vec![0.0; data.len()];
        
        // 단순한 Haar 웨이블릿 변환
        let mut n = data.len();
        while n > 1 {
            for i in 0..n/2 {
                temp[i] = (data[2*i] + data[2*i + 1]) / 2.0;           // 평균 (저주파)
                temp[n/2 + i] = (data[2*i] - data[2*i + 1]) / 2.0;     // 차이 (고주파)
            }
            
            for i in 0..n {
                data[i] = temp[i];
            }
            n /= 2;
        }
        
        // 압축 (고주파 계수 제거)
        let keep_count = ((1.0 - compression_ratio) * data.len() as f32) as usize;
        let keep_count = keep_count.max(1).min(data.len());
        
        for i in keep_count..data.len() {
            data[i] = 0.0;
        }
        
        // 역변환
        n = 2;
        while n <= data.len() {
            for i in 0..n/2 {
                temp[2*i] = data[i] + data[n/2 + i];     // 복원
                temp[2*i + 1] = data[i] - data[n/2 + i];
            }
            
            for i in 0..n {
                data[i] = temp[i];
            }
            n *= 2;
        }
        
        // RMSE 계산
        let mse = original.iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        let rmse = mse.sqrt();
        self.wavelet_performance = rmse;
        rmse
    }
    
    /// 두 변환 방법 비교
    pub fn compare_transforms(&mut self, signal: &[f32], compression_ratio: f32) -> (TransformType, f32, f32) {
        let dct_rmse = self.measure_dct_performance(signal, compression_ratio);
        let wavelet_rmse = self.measure_wavelet_performance(signal, compression_ratio);
        
        let winner = if dct_rmse < wavelet_rmse {
            TransformType::Dct
        } else {
            TransformType::Dwt
        };
        
        (winner, dct_rmse, wavelet_rmse)
    }
} 