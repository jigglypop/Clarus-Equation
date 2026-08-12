//! RBERMSNorm - 압축 도메인에서의 Root Mean Square 정규화
//! 
//! LayerNorm의 간소화된 버전으로, 평균을 빼지 않고 RMS로만 정규화
//! LLaMA, T5 등의 모델에서 사용

use anyhow::Result;
use rayon::prelude::*;
use crate::{
    core::{
        decoder::WeightGenerator,
        packed_params::HybridEncodedBlock,
    },
    QualityGrade,
};

/// RBERMSNorm 설정
#[derive(Debug, Clone)]
pub struct RBERMSNormConfig {
    /// 정규화 차원
    pub normalized_shape: usize,
    
    /// 엡실론 (수치적 안정성)
    pub epsilon: f32,
    
    /// 품질 등급
    pub quality_grade: QualityGrade,
    
    /// 병렬 처리 활성화
    pub enable_parallel: bool,
}

impl Default for RBERMSNormConfig {
    fn default() -> Self {
        Self {
            normalized_shape: 768,
            epsilon: 1e-5,
            quality_grade: QualityGrade::A,
            enable_parallel: true,
        }
    }
}

/// RBERMSNorm - Root Mean Square 정규화
#[derive(Clone)]
pub struct RBERMSNorm {
    /// 설정
    pub config: RBERMSNormConfig,
    
    /// 압축된 가중치 (gamma)
    pub gamma_blocks: Vec<HybridEncodedBlock>,
    
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    
    /// 디코딩 캐시
    decoded_cache: Option<Vec<f32>>,
}

impl RBERMSNorm {
    /// 새로운 RMSNorm 레이어 생성
    pub fn new(config: RBERMSNormConfig) -> Self {
        Self {
            config,
            gamma_blocks: Vec::new(),
            weight_generator: WeightGenerator::new(),
            decoded_cache: None,
        }
    }
    
    /// 가중치 초기화
    pub fn init_weights(&mut self) -> Result<()> {
        use rand::distributions::{Distribution, Uniform};
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.9, 1.1); // RMSNorm은 보통 1 근처로 초기화
        
        let gamma: Vec<f32> = (0..self.config.normalized_shape)
            .map(|_| uniform.sample(&mut rng))
            .collect();
        
        // 압축
        let mut encoder = match self.config.quality_grade {
            QualityGrade::S => crate::RBEEncoder::new_s_grade(),
            QualityGrade::A => crate::RBEEncoder::new_a_grade(),
            QualityGrade::B => crate::RBEEncoder::new_b_grade(),
            QualityGrade::C => crate::RBEEncoder::new_b_grade(), // C는 B로 대체
        };
        
        // 1차원 벡터로 압축
        let block = encoder.encode_vector(&gamma);
        self.gamma_blocks = vec![block];
        
        Ok(())
    }
    
    /// 디코딩된 가중치 가져오기
    fn get_gamma(&mut self) -> Result<&[f32]> {
        if self.decoded_cache.is_none() && !self.gamma_blocks.is_empty() {
            let decoded = self.weight_generator.decode_block(&self.gamma_blocks[0]);
            self.decoded_cache = Some((*decoded).clone());
        }
        
        self.decoded_cache.as_deref()
            .ok_or_else(|| anyhow::anyhow!("가중치가 초기화되지 않음"))
    }
    
    /// RMS 계산 (Kahan summation 사용)
    fn compute_rms(&self, input: &[f32]) -> f32 {
        let mut sum = 0.0;
        let mut compensation = 0.0;
        
        for &x in input {
            let y = x * x - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        
        (sum / input.len() as f32 + self.config.epsilon).sqrt()
    }
    
    /// 순전파
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let size = input.len();
        let normalized_shape = self.config.normalized_shape;
        if size % normalized_shape != 0 {
            anyhow::bail!("입력 크기가 normalized_shape의 배수가 아님");
        }
        
        let batch_size = size / normalized_shape;
        let norm_size = normalized_shape;
        let enable_parallel = self.config.enable_parallel;
        let epsilon = self.config.epsilon;
        let gamma = self.get_gamma()?;
        
        let mut output = vec![0.0; size];
        
        if enable_parallel && batch_size > 1 {
            // 병렬 처리
            output.par_chunks_mut(norm_size)
                .enumerate()
                .for_each(|(batch_idx, out_chunk)| {
                    let start = batch_idx * norm_size;
                    let input_chunk = &input[start..start + norm_size];
                    
                    // RMS 계산 (직접 인라인)
                    let mut sum = 0.0;
                    let mut compensation = 0.0;
                    for &x in input_chunk {
                        let y = x * x - compensation;
                        let t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    }
                    let rms = (sum / input_chunk.len() as f32 + epsilon).sqrt();
                    let scale = 1.0 / rms;
                    
                    // 정규화 및 스케일링
                    for i in 0..norm_size {
                        out_chunk[i] = input_chunk[i] * scale * gamma[i];
                    }
                });
        } else {
            // 순차 처리
            for batch_idx in 0..batch_size {
                let start = batch_idx * norm_size;
                let input_chunk = &input[start..start + norm_size];
                let output_chunk = &mut output[start..start + norm_size];
                
                // RMS 계산 (직접 인라인)
                let mut sum = 0.0;
                let mut compensation = 0.0;
                for &x in input_chunk {
                    let y = x * x - compensation;
                    let t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }
                let rms = (sum / input_chunk.len() as f32 + epsilon).sqrt();
                let scale = 1.0 / rms;
                
                // 정규화 및 스케일링
                for i in 0..norm_size {
                    output_chunk[i] = input_chunk[i] * scale * gamma[i];
                }
            }
        }
        
        Ok(output)
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let compressed_size = self.gamma_blocks.iter()
            .map(|block| std::mem::size_of::<HybridEncodedBlock>())
            .sum::<usize>();
            
        let original_size = self.config.normalized_shape * std::mem::size_of::<f32>();
        let compression_ratio = original_size as f32 / compressed_size.max(1) as f32;
        
        (compressed_size, compression_ratio)
    }
} 