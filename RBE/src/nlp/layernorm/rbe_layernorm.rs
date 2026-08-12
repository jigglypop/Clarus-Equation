//! RBE 기반 Layer Normalization
//! 수치적 안정성을 위한 Kahan summation과 융합 연산 구현

use anyhow::{Result, bail};
use std::sync::Arc;

/// Layer Normalization 설정
#[derive(Debug, Clone)]
pub struct RBELayerNormConfig {
    /// 정규화할 차원의 크기
    pub normalized_shape: Vec<usize>,
    /// 수치적 안정성을 위한 epsilon
    pub eps: f64,
    /// 학습 가능한 scale parameter 사용 여부
    pub elementwise_affine: bool,
    /// 융합 연산 사용 여부
    pub use_fused_ops: bool,
}

impl Default for RBELayerNormConfig {
    fn default() -> Self {
        Self {
            normalized_shape: vec![768],  // GPT-2 기본값
            eps: 1e-5,
            elementwise_affine: true,
            use_fused_ops: true,
        }
    }
}

/// RBE Layer Normalization
#[derive(Debug)]
pub struct RBELayerNorm {
    /// 설정
    config: RBELayerNormConfig,
    /// Scale parameter (gamma)
    gamma: Option<Vec<f32>>,
    /// Shift parameter (beta)
    beta: Option<Vec<f32>>,
    /// 정규화할 원소 개수
    normalized_size: usize,
}

impl RBELayerNorm {
    /// 새로운 Layer Normalization 생성
    pub fn new(config: RBELayerNormConfig) -> Result<Self> {
        let normalized_size: usize = config.normalized_shape.iter().product();
        
        if normalized_size == 0 {
            bail!("Normalized shape cannot be empty");
        }
        
        let (gamma, beta) = if config.elementwise_affine {
            (
                Some(vec![1.0; normalized_size]),  // 초기값 1.0
                Some(vec![0.0; normalized_size]),  // 초기값 0.0
            )
        } else {
            (None, None)
        };
        
        Ok(Self {
            config,
            gamma,
            beta,
            normalized_size,
        })
    }
    
    /// 사전 학습된 가중치로 초기화
    pub fn from_pretrained(
        gamma: Option<Vec<f32>>,
        beta: Option<Vec<f32>>,
        config: RBELayerNormConfig,
    ) -> Result<Self> {
        let normalized_size: usize = config.normalized_shape.iter().product();
        
        // 검증
        if let Some(ref g) = gamma {
            if g.len() != normalized_size {
                bail!("Gamma size mismatch: expected {}, got {}", normalized_size, g.len());
            }
        }
        if let Some(ref b) = beta {
            if b.len() != normalized_size {
                bail!("Beta size mismatch: expected {}, got {}", normalized_size, b.len());
            }
        }
        
        Ok(Self {
            config,
            gamma,
            beta,
            normalized_size,
        })
    }
    
    /// 순전파
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() % self.normalized_size != 0 {
            bail!("Input size {} is not divisible by normalized size {}", 
                  input.len(), self.normalized_size);
        }
        
        let batch_size = input.len() / self.normalized_size;
        let mut output = vec![0.0f32; input.len()];
        
        if self.config.use_fused_ops {
            self.forward_fused(input, &mut output, batch_size)?;
        } else {
            self.forward_standard(input, &mut output, batch_size)?;
        }
        
        Ok(output)
    }
    
    /// 표준 순전파 (디버깅용)
    fn forward_standard(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
    ) -> Result<()> {
        for b in 0..batch_size {
            let offset = b * self.normalized_size;
            let input_slice = &input[offset..offset + self.normalized_size];
            let output_slice = &mut output[offset..offset + self.normalized_size];
            
            // 평균 계산
            let mean = self.compute_mean(input_slice);
            
            // 분산 계산
            let variance = self.compute_variance(input_slice, mean);
            
            // 정규화 및 affine 변환
            self.normalize_and_affine(input_slice, output_slice, mean, variance)?;
        }
        
        Ok(())
    }
    
    /// 융합 순전파 (최적화)
    fn forward_fused(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
    ) -> Result<()> {
        use rayon::prelude::*;
        
        // 병렬 처리
        output.par_chunks_mut(self.normalized_size)
            .zip(input.par_chunks(self.normalized_size))
            .try_for_each(|(out_chunk, in_chunk)| {
                // Kahan summation으로 평균 계산
                let mean = self.compute_mean_kahan(in_chunk);
                
                // Two-pass 알고리즘으로 분산 계산 (수치적 안정성)
                let variance = self.compute_variance_stable(in_chunk, mean);
                
                // 정규화 및 affine 변환 (융합)
                self.normalize_and_affine_fused(in_chunk, out_chunk, mean, variance)
            })?;
        
        Ok(())
    }
    
    /// Kahan summation을 사용한 평균 계산
    fn compute_mean_kahan(&self, input: &[f32]) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64;  // 보정값
        
        for &x in input {
            let y = x as f64 - c;  // 보정된 값
            let t = sum + y;       // 새로운 합
            c = (t - sum) - y;     // 새로운 보정값
            sum = t;
        }
        
        sum / input.len() as f64
    }
    
    /// 표준 평균 계산
    fn compute_mean(&self, input: &[f32]) -> f64 {
        let sum: f64 = input.iter().map(|&x| x as f64).sum();
        sum / input.len() as f64
    }
    
    /// 수치적으로 안정한 분산 계산 (Two-pass)
    fn compute_variance_stable(&self, input: &[f32], mean: f64) -> f64 {
        let mut sum_sq = 0.0f64;
        let mut c = 0.0f64;  // Kahan 보정값
        
        for &x in input {
            let diff = x as f64 - mean;
            let sq = diff * diff;
            let y = sq - c;
            let t = sum_sq + y;
            c = (t - sum_sq) - y;
            sum_sq = t;
        }
        
        sum_sq / input.len() as f64
    }
    
    /// 표준 분산 계산
    fn compute_variance(&self, input: &[f32], mean: f64) -> f64 {
        let sum_sq: f64 = input.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum();
        
        sum_sq / input.len() as f64
    }
    
    /// 정규화 및 affine 변환 (표준)
    fn normalize_and_affine(
        &self,
        input: &[f32],
        output: &mut [f32],
        mean: f64,
        variance: f64,
    ) -> Result<()> {
        let std_inv = 1.0 / (variance + self.config.eps).sqrt();
        
        for (i, (&x, out)) in input.iter().zip(output.iter_mut()).enumerate() {
            let normalized = ((x as f64 - mean) * std_inv) as f32;
            
            *out = if self.config.elementwise_affine {
                let gamma = self.gamma.as_ref().unwrap()[i];
                let beta = self.beta.as_ref().unwrap()[i];
                gamma * normalized + beta
            } else {
                normalized
            };
        }
        
        Ok(())
    }
    
    /// 정규화 및 affine 변환 (융합, SIMD 최적화 가능)
    fn normalize_and_affine_fused(
        &self,
        input: &[f32],
        output: &mut [f32],
        mean: f64,
        variance: f64,
    ) -> Result<()> {
        let std_inv = 1.0 / (variance + self.config.eps).sqrt();
        let mean_f32 = mean as f32;
        let std_inv_f32 = std_inv as f32;
        
        if self.config.elementwise_affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();
            
            // SIMD 친화적 루프
            for i in 0..input.len() {
                let normalized = (input[i] - mean_f32) * std_inv_f32;
                output[i] = gamma[i] * normalized + beta[i];
            }
        } else {
            // Affine 없는 경우
            for i in 0..input.len() {
                output[i] = (input[i] - mean_f32) * std_inv_f32;
            }
        }
        
        Ok(())
    }
    
    /// 파라미터 업데이트 (학습용)
    pub fn update_parameters(&mut self, gamma: Option<Vec<f32>>, beta: Option<Vec<f32>>) -> Result<()> {
        if let Some(g) = gamma {
            if g.len() != self.normalized_size {
                bail!("Gamma size mismatch");
            }
            self.gamma = Some(g);
        }
        
        if let Some(b) = beta {
            if b.len() != self.normalized_size {
                bail!("Beta size mismatch");
            }
            self.beta = Some(b);
        }
        
        Ok(())
    }
    
    /// 통계 정보 반환 (디버깅용)
    pub fn compute_statistics(&self, input: &[f32]) -> Result<LayerNormStats> {
        if input.len() % self.normalized_size != 0 {
            bail!("Input size mismatch");
        }
        
        let batch_size = input.len() / self.normalized_size;
        let mut all_means = Vec::with_capacity(batch_size);
        let mut all_vars = Vec::with_capacity(batch_size);
        
        for b in 0..batch_size {
            let offset = b * self.normalized_size;
            let input_slice = &input[offset..offset + self.normalized_size];
            
            let mean = self.compute_mean_kahan(input_slice);
            let variance = self.compute_variance_stable(input_slice, mean);
            
            all_means.push(mean as f32);
            all_vars.push(variance as f32);
        }
        
        Ok(LayerNormStats {
            means: all_means,
            variances: all_vars,
            eps: self.config.eps as f32,
        })
    }
}

/// Layer Normalization 통계
#[derive(Debug, Clone)]
pub struct LayerNormStats {
    pub means: Vec<f32>,
    pub variances: Vec<f32>,
    pub eps: f32,
}

/// 배치 Layer Normalization (여러 샘플 동시 처리)
impl RBELayerNorm {
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        inputs.iter()
            .map(|input| self.forward(input))
            .collect()
    }
} 