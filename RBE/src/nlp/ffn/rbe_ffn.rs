//! RBE 기반 Feed-Forward Network (FFN)
//! Transformer의 FFN 블록을 압축된 형태로 구현

use crate::{
    core::{
        decoder::WeightGenerator,
        encoder::RBEEncoder,
        packed_params::HybridEncodedBlock,
    },
    nlp::linear::{RBELinear, RBELinearConfig},
    QualityGrade,
};
use anyhow::{Result, bail};
use std::sync::Arc;
use rayon::prelude::*;

/// FFN 설정
#[derive(Debug, Clone)]
pub struct RBEFFNConfig {
    /// 입력/출력 차원
    pub hidden_dim: usize,
    /// 중간 레이어 차원 (일반적으로 4 * hidden_dim)
    pub intermediate_dim: usize,
    /// 활성화 함수 타입
    pub activation: ActivationType,
    /// 드롭아웃 확률
    pub dropout: f32,
    /// 블록 크기
    pub block_size: usize,
    /// 압축 품질
    pub quality_grade: QualityGrade,
    /// 병렬 처리 여부
    pub enable_parallel: bool,
    /// 캐시 크기
    pub cache_size: usize,
}

/// 활성화 함수 타입
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    Gelu,
    GeluNew,  // GPT-2 스타일 GELU
    Relu,
    Swish,
}

impl Default for RBEFFNConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            intermediate_dim: 3072,  // 4 * 768
            activation: ActivationType::Gelu,
            dropout: 0.0,
            block_size: 256,  // 큰 블록으로 효율성 증대
            quality_grade: QualityGrade::B,
            enable_parallel: true,
            cache_size: 32,
        }
    }
}

/// RBE 기반 Feed-Forward Network
#[derive(Debug)]
pub struct RBEFFN {
    /// 첫 번째 선형 레이어 (hidden -> intermediate)
    up_proj: RBELinear,
    /// 두 번째 선형 레이어 (intermediate -> hidden)
    down_proj: RBELinear,
    /// 활성화 함수
    activation: ActivationType,
    /// 드롭아웃 확률
    dropout: f32,
    /// 설정
    config: RBEFFNConfig,
}

impl RBEFFN {
    /// 새로운 FFN 생성
    pub fn new(config: RBEFFNConfig) -> Result<Self> {
        println!("RBEFFN 초기화:");
        println!("  - 입력/출력 차원: {}", config.hidden_dim);
        println!("  - 중간 차원: {}", config.intermediate_dim);
        println!("  - 압축 품질: {:?}", config.quality_grade);
        
        // 선형 레이어 설정
        let linear_config = RBELinearConfig {
            enable_parallel: config.enable_parallel,
            cache_size: config.cache_size,
        };
        
        // 빈 블록으로 초기화 (나중에 압축된 가중치 로드)
        let up_proj = RBELinear::with_config(
            Vec::new(),
            config.hidden_dim,
            config.intermediate_dim,
            None,  // bias 없음
            linear_config.clone(),
        );
        
        let down_proj = RBELinear::with_config(
            Vec::new(),
            config.intermediate_dim,
            config.hidden_dim,
            None,  // bias 없음
            linear_config,
        );
        
        Ok(Self {
            up_proj,
            down_proj,
            activation: config.activation,
            dropout: config.dropout,
            config,
        })
    }
    
    /// 랜덤 가중치로 초기화하고 압축
    pub fn init_random(&mut self) -> Result<()> {
        use rand::thread_rng;
        use rand::distributions::{Distribution, Uniform};
        
        println!("FFN 랜덤 초기화 및 압축 시작...");
        
        // RBE 인코더 생성
        let mut encoder = match self.config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_b_grade(), // C급도 B급으로 처리
        };
        
        // up projection 압축 (hidden -> intermediate)
        self.compress_up_projection(&mut encoder)?;
        
        // down projection 압축 (intermediate -> hidden)
        self.compress_down_projection(&mut encoder)?;
        
        println!("FFN 초기화 완료!");
        Ok(())
    }
    
    /// Up projection 압축
    fn compress_up_projection(&mut self, encoder: &mut RBEEncoder) -> Result<()> {
        use rand::{thread_rng, Rng};
        use rand::distributions::Uniform;
        
        let mut rng = thread_rng();
        let scale = (2.0 / self.config.hidden_dim as f32).sqrt();
        let dist = Uniform::new(-scale, scale);
        
        let mut blocks = Vec::new();
        
        // 행별로 블록 단위 압축
        let blocks_per_row = (self.config.intermediate_dim + self.config.block_size - 1) 
                           / self.config.block_size;
        let blocks_per_col = (self.config.hidden_dim + self.config.block_size - 1) 
                           / self.config.block_size;
        
        for row_block in 0..blocks_per_row {
            for col_block in 0..blocks_per_col {
                // 블록 데이터 생성
                let row_start = row_block * self.config.block_size;
                let row_end = ((row_block + 1) * self.config.block_size)
                    .min(self.config.intermediate_dim);
                let col_start = col_block * self.config.block_size;
                let col_end = ((col_block + 1) * self.config.block_size)
                    .min(self.config.hidden_dim);
                
                let block_rows = row_end - row_start;
                let block_cols = col_end - col_start;
                
                let block_data: Vec<f32> = (0..block_rows * block_cols)
                    .map(|_| rng.sample(dist))
                    .collect();
                
                // 압축
                let encoded = encoder.encode_block(&block_data, block_rows, block_cols);
                blocks.push(encoded);
            }
        }
        
        self.up_proj.blocks = blocks;
        println!("  Up projection 압축 완료: {} -> {}", 
                 self.config.hidden_dim, self.config.intermediate_dim);
        
        Ok(())
    }
    
    /// Down projection 압축
    fn compress_down_projection(&mut self, encoder: &mut RBEEncoder) -> Result<()> {
        use rand::{thread_rng, Rng};
        use rand::distributions::Uniform;
        
        let mut rng = thread_rng();
        let scale = (2.0 / self.config.intermediate_dim as f32).sqrt();
        let dist = Uniform::new(-scale, scale);
        
        let mut blocks = Vec::new();
        
        // 행별로 블록 단위 압축
        let blocks_per_row = (self.config.hidden_dim + self.config.block_size - 1) 
                           / self.config.block_size;
        let blocks_per_col = (self.config.intermediate_dim + self.config.block_size - 1) 
                           / self.config.block_size;
        
        for row_block in 0..blocks_per_row {
            for col_block in 0..blocks_per_col {
                // 블록 데이터 생성
                let row_start = row_block * self.config.block_size;
                let row_end = ((row_block + 1) * self.config.block_size)
                    .min(self.config.hidden_dim);
                let col_start = col_block * self.config.block_size;
                let col_end = ((col_block + 1) * self.config.block_size)
                    .min(self.config.intermediate_dim);
                
                let block_rows = row_end - row_start;
                let block_cols = col_end - col_start;
                
                let block_data: Vec<f32> = (0..block_rows * block_cols)
                    .map(|_| rng.sample(dist))
                    .collect();
                
                // 압축
                let encoded = encoder.encode_block(&block_data, block_rows, block_cols);
                blocks.push(encoded);
            }
        }
        
        self.down_proj.blocks = blocks;
        println!("  Down projection 압축 완료: {} -> {}", 
                 self.config.intermediate_dim, self.config.hidden_dim);
        
        Ok(())
    }
    
    /// 순전파
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() % self.config.hidden_dim != 0 {
            bail!("Input size {} not divisible by hidden_dim {}", 
                  input.len(), self.config.hidden_dim);
        }
        
        // 1. Up projection (hidden -> intermediate)
        let intermediate = self.up_proj.forward(input);
        
        // 2. 활성화 함수
        let activated = self.apply_activation(&intermediate);
        
        // 3. 드롭아웃 (학습 모드에서만)
        let dropped = if self.dropout > 0.0 {
            self.apply_dropout(&activated)
        } else {
            activated
        };
        
        // 4. Down projection (intermediate -> hidden)
        let output = self.down_proj.forward(&dropped);
        
        Ok(output)
    }
    
    /// 활성화 함수 적용
    fn apply_activation(&self, input: &[f32]) -> Vec<f32> {
        match self.activation {
            ActivationType::Gelu => self.gelu(input),
            ActivationType::GeluNew => self.gelu_new(input),
            ActivationType::Relu => self.relu(input),
            ActivationType::Swish => self.swish(input),
        }
    }
    
    /// GELU 활성화 함수
    fn gelu(&self, input: &[f32]) -> Vec<f32> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        
        input.iter()
            .map(|&x| {
                let cdf = 0.5 * (1.0 + ((SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh()));
                x * cdf
            })
            .collect()
    }
    
    /// GPT-2 스타일 GELU (더 정확한 근사)
    fn gelu_new(&self, input: &[f32]) -> Vec<f32> {
        use std::f32::consts::PI;
        
        input.iter()
            .map(|&x| {
                0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            })
            .collect()
    }
    
    /// ReLU 활성화 함수
    fn relu(&self, input: &[f32]) -> Vec<f32> {
        input.iter()
            .map(|&x| x.max(0.0))
            .collect()
    }
    
    /// Swish 활성화 함수
    fn swish(&self, input: &[f32]) -> Vec<f32> {
        input.iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect()
    }
    
    /// 드롭아웃 적용 (학습 모드)
    fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
        use rand::{thread_rng, Rng};
        
        let mut rng = thread_rng();
        let scale = 1.0 / (1.0 - self.dropout);
        
        input.iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.dropout {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect()
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let (up_size, up_ratio) = self.up_proj.memory_usage();
        let (down_size, down_ratio) = self.down_proj.memory_usage();
        
        let total_compressed = up_size + down_size;
        let total_original = (self.config.hidden_dim * self.config.intermediate_dim * 2) * 4;
        let total_ratio = total_original as f32 / total_compressed as f32;
        
        (total_compressed, total_ratio)
    }
    
    /// 캐시 초기화
    pub fn clear_cache(&mut self) {
        self.up_proj.clear_cache();
        self.down_proj.clear_cache();
    }
}

/// 사전 학습된 가중치에서 로드
impl RBEFFN {
    /// 특정 projection 압축 헬퍼
    fn compress_projection(
        weights: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        encoder: &mut RBEEncoder,
    ) -> Result<Vec<HybridEncodedBlock>> {
        let mut blocks = Vec::new();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let blocks_per_col = (rows + block_size - 1) / block_size;
        
        for row_block in 0..blocks_per_col {
            for col_block in 0..blocks_per_row {
                let row_start = row_block * block_size;
                let row_end = ((row_block + 1) * block_size).min(rows);
                let col_start = col_block * block_size;
                let col_end = ((col_block + 1) * block_size).min(cols);
                
                let block_rows = row_end - row_start;
                let block_cols = col_end - col_start;
                
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; block_rows * block_cols];
                for r in 0..block_rows {
                    for c in 0..block_cols {
                        let src_idx = (row_start + r) * cols + (col_start + c);
                        let dst_idx = r * block_cols + c;
                        block_data[dst_idx] = weights[src_idx];
                    }
                }
                
                // 압축
                let encoded = encoder.encode_block(&block_data, block_rows, block_cols);
                blocks.push(encoded);
            }
        }
        
        Ok(blocks)
    }

    /// 사전 학습된 가중치로부터 RBEFFN 생성
    pub fn from_pretrained_weights(
        up_weights: &[f32],
        down_weights: &[f32],
        config: RBEFFNConfig,
    ) -> Result<Self> {
        // 검증
        let expected_up_size = config.hidden_dim * config.intermediate_dim;
        let expected_down_size = config.intermediate_dim * config.hidden_dim;
        
        if up_weights.len() != expected_up_size {
            bail!("Up projection weights size mismatch: expected {}, got {}", 
                  expected_up_size, up_weights.len());
        }
        
        if down_weights.len() != expected_down_size {
            bail!("Down projection weights size mismatch: expected {}, got {}", 
                  expected_down_size, down_weights.len());
        }
        
        // RBE 인코더 생성
        let mut encoder = match config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_b_grade(), // C급도 B급으로 처리
        };
        
        println!("사전 학습된 FFN 가중치 압축 시작...");
        
        // up projection 압축
        let up_blocks = Self::compress_projection(
            up_weights,
            config.hidden_dim,
            config.intermediate_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        // down projection 압축
        let down_blocks = Self::compress_projection(
            down_weights,
            config.intermediate_dim,
            config.hidden_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        // RBELinear 생성
        let up_proj = RBELinear::with_config(
            up_blocks,
            config.hidden_dim,
            config.intermediate_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let down_proj = RBELinear::with_config(
            down_blocks,
            config.intermediate_dim,
            config.hidden_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let ffn = Self {
            up_proj,
            down_proj,
            activation: config.activation,
            dropout: config.dropout,
            config,
        };
        
        let (compressed_size, ratio) = ffn.memory_usage();
        println!("FFN 압축 완료! 압축률: {:.1}:1, 압축 크기: {:.2} MB", 
                 ratio, compressed_size as f32 / 1024.0 / 1024.0);
        
        Ok(ffn)
    }
} 