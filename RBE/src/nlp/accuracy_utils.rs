//! 정확도 측정 유틸리티

use std::f32::consts::PI;

/// RMSE (Root Mean Square Error) 계산
pub fn rmse(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    let sum_sq_diff: f32 = reference.iter()
        .zip(approximation.iter())
        .map(|(r, a)| (r - a).powi(2))
        .sum();
    
    (sum_sq_diff / reference.len() as f32).sqrt()
}

/// 상대 오차 계산
pub fn relative_error(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    
    for (&r, &a) in reference.iter().zip(approximation.iter()) {
        let diff = (r - a).abs();
        let ref_abs = r.abs();
        
        num += diff * diff;
        den += ref_abs * ref_abs;
    }
    
    if den < 1e-10 {
        return if num < 1e-10 { 0.0 } else { f32::INFINITY };
    }
    
    (num / den).sqrt()
}

/// 코사인 유사도 계산
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    
    let mut dot_product = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// 최대 절대 오차 계산
pub fn max_absolute_error(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    reference.iter()
        .zip(approximation.iter())
        .map(|(r, a)| (r - a).abs())
        .fold(0.0f32, f32::max)
}

/// 평균 절대 오차 계산
pub fn mean_absolute_error(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    let sum: f32 = reference.iter()
        .zip(approximation.iter())
        .map(|(r, a)| (r - a).abs())
        .sum();
    
    sum / reference.len() as f32
}

/// 신호 대 잡음비 (SNR) 계산
pub fn signal_to_noise_ratio(signal: &[f32], noise: &[f32]) -> f32 {
    let signal_power: f32 = signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32;
    let noise_power: f32 = noise.iter().map(|x| x * x).sum::<f32>() / noise.len() as f32;
    
    if noise_power < 1e-10 {
        return f32::INFINITY;
    }
    
    10.0 * (signal_power / noise_power).log10()
}

/// 정확도 메트릭 구조체
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    pub rmse: f32,
    pub relative_error: f32,
    pub cosine_similarity: f32,
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    pub snr_db: f32,
}

impl AccuracyMetrics {
    /// 두 배열 간의 모든 메트릭 계산
    pub fn calculate(reference: &[f32], approximation: &[f32]) -> Self {
        let noise: Vec<f32> = reference.iter()
            .zip(approximation.iter())
            .map(|(r, a)| r - a)
            .collect();
        
        Self {
            rmse: rmse(reference, approximation),
            relative_error: relative_error(reference, approximation),
            cosine_similarity: cosine_similarity(reference, approximation),
            max_absolute_error: max_absolute_error(reference, approximation),
            mean_absolute_error: mean_absolute_error(reference, approximation),
            snr_db: signal_to_noise_ratio(reference, &noise),
        }
    }
    
    /// 품질 등급 판정
    pub fn quality_grade(&self) -> &'static str {
        match self.rmse {
            x if x < 0.0001 => "S",
            x if x < 0.001 => "A",
            x if x < 0.01 => "B",
            x if x < 0.1 => "C",
            _ => "D",
        }
    }
    
    /// 상세 리포트 출력
    pub fn report(&self, name: &str) {
        println!("\n=== {} 정확도 분석 ===", name);
        println!("RMSE: {:.6}", self.rmse);
        println!("상대 오차: {:.2}%", self.relative_error * 100.0);
        println!("코사인 유사도: {:.6}", self.cosine_similarity);
        println!("최대 절대 오차: {:.6}", self.max_absolute_error);
        println!("평균 절대 오차: {:.6}", self.mean_absolute_error);
        println!("SNR: {:.2} dB", self.snr_db);
        println!("품질 등급: {}", self.quality_grade());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn 정확도_메트릭_기본_테스트() {
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let approximation = vec![1.01, 1.99, 3.02, 3.98, 5.01];
        
        let metrics = AccuracyMetrics::calculate(&reference, &approximation);
        
        assert!(metrics.rmse < 0.02);
        assert!(metrics.relative_error < 0.01);
        assert!(metrics.cosine_similarity > 0.999);
        assert_eq!(metrics.quality_grade(), "B");
    }
    
    #[test]
    fn 완벽한_일치_테스트() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = AccuracyMetrics::calculate(&data, &data);
        
        assert_eq!(metrics.rmse, 0.0);
        assert_eq!(metrics.relative_error, 0.0);
        assert_eq!(metrics.cosine_similarity, 1.0);
        assert_eq!(metrics.max_absolute_error, 0.0);
        assert_eq!(metrics.quality_grade(), "S");
    }
} 