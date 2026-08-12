//! RBE 기반 Multi-Head Self-Attention
//! Transformer의 핵심인 attention 메커니즘을 압축된 형태로 구현

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

/// Attention 설정
#[derive(Debug, Clone)]
pub struct RBEAttentionConfig {
    /// 히든 차원
    pub hidden_dim: usize,
    /// 어텐션 헤드 수
    pub num_heads: usize,
    /// 헤드당 차원 (일반적으로 hidden_dim / num_heads)
    pub head_dim: usize,
    /// 어텐션 드롭아웃
    pub attention_dropout: f32,
    /// 출력 드롭아웃
    pub output_dropout: f32,
    /// 블록 크기
    pub block_size: usize,
    /// 압축 품질
    pub quality_grade: QualityGrade,
    /// 병렬 처리 여부
    pub enable_parallel: bool,
    /// 캐시 크기
    pub cache_size: usize,
}

impl Default for RBEAttentionConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,  // 768 / 12
            attention_dropout: 0.0,
            output_dropout: 0.0,
            block_size: 128,
            quality_grade: QualityGrade::B,
            enable_parallel: true,
            cache_size: 32,
        }
    }
}

/// RBE 기반 Multi-Head Attention
#[derive(Debug)]
pub struct RBEAttention {
    /// Query projection
    q_proj: RBELinear,
    /// Key projection
    k_proj: RBELinear,
    /// Value projection
    v_proj: RBELinear,
    /// Output projection
    out_proj: RBELinear,
    /// 설정
    config: RBEAttentionConfig,
    /// 스케일 팩터 (1/sqrt(head_dim))
    scale: f32,
}

impl RBEAttention {
    /// 새로운 Attention 레이어 생성
    pub fn new(config: RBEAttentionConfig) -> Result<Self> {
        if config.hidden_dim % config.num_heads != 0 {
            bail!("hidden_dim {} must be divisible by num_heads {}", 
                  config.hidden_dim, config.num_heads);
        }
        
        let actual_head_dim = config.hidden_dim / config.num_heads;
        if config.head_dim != actual_head_dim {
            bail!("head_dim {} doesn't match hidden_dim/num_heads {}", 
                  config.head_dim, actual_head_dim);
        }
        
        println!("RBEAttention 초기화:");
        println!("  - 히든 차원: {}", config.hidden_dim);
        println!("  - 헤드 수: {}", config.num_heads);
        println!("  - 헤드당 차원: {}", config.head_dim);
        
        // 선형 레이어 설정
        let linear_config = RBELinearConfig {
            enable_parallel: config.enable_parallel,
            cache_size: config.cache_size,
        };
        
        // Q, K, V, O projections
        let q_proj = RBELinear::with_config(
            Vec::new(),
            config.hidden_dim,
            config.hidden_dim,
            None,
            linear_config.clone(),
        );
        
        let k_proj = RBELinear::with_config(
            Vec::new(),
            config.hidden_dim,
            config.hidden_dim,
            None,
            linear_config.clone(),
        );
        
        let v_proj = RBELinear::with_config(
            Vec::new(),
            config.hidden_dim,
            config.hidden_dim,
            None,
            linear_config.clone(),
        );
        
        let out_proj = RBELinear::with_config(
            Vec::new(),
            config.hidden_dim,
            config.hidden_dim,
            None,
            linear_config,
        );
        
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            config,
            scale,
        })
    }
    
    /// 랜덤 가중치로 초기화하고 압축
    pub fn init_random(&mut self) -> Result<()> {
        use rand::thread_rng;
        use rand::distributions::{Distribution, Uniform};
        
        println!("Attention 랜덤 초기화 및 압축 시작...");
        
        // RBE 인코더 생성
        let mut encoder = match self.config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_b_grade(), // C급도 B급으로 처리
        };
        
        // 각 projection 압축
        self.compress_projection(&mut encoder, "query")?;
        self.compress_projection(&mut encoder, "key")?;
        self.compress_projection(&mut encoder, "value")?;
        self.compress_projection(&mut encoder, "output")?;
        
        println!("Attention 초기화 완료!");
        Ok(())
    }
    
    /// 특정 projection 압축
    fn compress_projection(&mut self, encoder: &mut RBEEncoder, proj_type: &str) -> Result<()> {
        use rand::{thread_rng, Rng};
        use rand::distributions::Uniform;
        
        let mut rng = thread_rng();
        let scale = (2.0 / self.config.hidden_dim as f32).sqrt();
        let dist = Uniform::new(-scale, scale);
        
        let mut blocks = Vec::new();
        
        // 블록 단위 압축
        let blocks_per_row = (self.config.hidden_dim + self.config.block_size - 1) 
                           / self.config.block_size;
        let blocks_per_col = blocks_per_row;  // 정방 행렬
        
        for row_block in 0..blocks_per_row {
            for col_block in 0..blocks_per_col {
                let row_start = row_block * self.config.block_size;
                let row_end = ((row_block + 1) * self.config.block_size)
                    .min(self.config.hidden_dim);
                let col_start = col_block * self.config.block_size;
                let col_end = ((col_block + 1) * self.config.block_size)
                    .min(self.config.hidden_dim);
                
                let block_rows = row_end - row_start;
                let block_cols = col_end - col_start;
                
                let block_data: Vec<f32> = (0..block_rows * block_cols)
                    .map(|_| rng.sample(dist))
                    .collect();
                
                let encoded = encoder.encode_block(&block_data, block_rows, block_cols);
                blocks.push(encoded);
            }
        }
        
        // 해당 projection에 블록 할당
        match proj_type {
            "query" => self.q_proj.blocks = blocks,
            "key" => self.k_proj.blocks = blocks,
            "value" => self.v_proj.blocks = blocks,
            "output" => self.out_proj.blocks = blocks,
            _ => bail!("Unknown projection type: {}", proj_type),
        }
        
        println!("  {} projection 압축 완료", proj_type);
        Ok(())
    }
    
    /// 순전파
    pub fn forward(
        &mut self,
        hidden_states: &[f32],
        attention_mask: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_dim;
        
        if hidden_states.len() % hidden_size != 0 {
            bail!("Hidden states size {} not divisible by hidden_dim {}", 
                  hidden_states.len(), hidden_size);
        }
        
        let seq_len = hidden_states.len() / hidden_size;
        
        // 1. Q, K, V projections
        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);
        
        // 2. Reshape for multi-head attention
        // [seq_len, hidden_dim] -> [seq_len, num_heads, head_dim]
        let q_heads = self.reshape_for_heads(&q, seq_len);
        let k_heads = self.reshape_for_heads(&k, seq_len);
        let v_heads = self.reshape_for_heads(&v, seq_len);
        
        // 3. Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            seq_len,
            attention_mask,
        )?;
        
        // 4. Reshape back
        // [seq_len, num_heads, head_dim] -> [seq_len, hidden_dim]
        let concatenated = self.reshape_from_heads(&attention_output, seq_len);
        
        // 5. Output projection
        let output = self.out_proj.forward(&concatenated);
        
        // 6. Dropout (if enabled)
        if self.config.output_dropout > 0.0 {
            Ok(self.apply_dropout(&output))
        } else {
            Ok(output)
        }
    }
    
    /// Reshape for multi-head attention
    fn reshape_for_heads(&self, x: &[f32], seq_len: usize) -> Vec<Vec<Vec<f32>>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        
        let mut heads = vec![vec![vec![0.0; head_dim]; seq_len]; num_heads];
        
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = s * self.config.hidden_dim + h * head_dim + d;
                    heads[h][s][d] = x[idx];
                }
            }
        }
        
        heads
    }
    
    /// Reshape from multi-head format back to flat
    fn reshape_from_heads(&self, heads: &[Vec<Vec<f32>>], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        
        let mut output = vec![0.0; seq_len * hidden_dim];
        
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = s * hidden_dim + h * head_dim + d;
                    output[idx] = heads[h][s][d];
                }
            }
        }
        
        output
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q_heads: &[Vec<Vec<f32>>],  // [num_heads, seq_len, head_dim]
        k_heads: &[Vec<Vec<f32>>],
        v_heads: &[Vec<Vec<f32>>],
        seq_len: usize,
        attention_mask: Option<&[f32]>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        
        let mut output_heads = vec![vec![vec![0.0; head_dim]; seq_len]; num_heads];
        
        // 각 헤드별로 attention 계산
        for h in 0..num_heads {
            // QK^T 계산
            let mut scores = vec![vec![0.0f32; seq_len]; seq_len];
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_heads[h][i][d] * k_heads[h][j][d];
                    }
                    scores[i][j] = score * self.scale;
                }
            }
            
            // Apply attention mask if provided
            if let Some(mask) = attention_mask {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if mask[i * seq_len + j] == 0.0 {
                            scores[i][j] = -1e10;  // 매우 작은 값
                        }
                    }
                }
            }
            
            // Softmax
            for i in 0..seq_len {
                self.softmax_inplace(&mut scores[i]);
            }
            
            // Dropout on attention weights (if enabled)
            if self.config.attention_dropout > 0.0 {
                self.apply_dropout_2d(&mut scores);
            }
            
            // Attention output: scores @ V
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        sum += scores[i][j] * v_heads[h][j][d];
                    }
                    output_heads[h][i][d] = sum;
                }
            }
        }
        
        Ok(output_heads)
    }
    
    /// In-place softmax
    fn softmax_inplace(&self, scores: &mut [f32]) {
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let mut sum = 0.0;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            sum += *score;
        }
        
        if sum > 0.0 {
            for score in scores.iter_mut() {
                *score /= sum;
            }
        }
    }
    
    /// Apply dropout
    fn apply_dropout(&self, x: &[f32]) -> Vec<f32> {
        use rand::{thread_rng, Rng};
        
        let mut rng = thread_rng();
        let scale = 1.0 / (1.0 - self.config.output_dropout);
        
        x.iter()
            .map(|&val| {
                if rng.gen::<f32>() < self.config.output_dropout {
                    0.0
                } else {
                    val * scale
                }
            })
            .collect()
    }
    
    /// Apply dropout to 2D attention scores
    fn apply_dropout_2d(&self, scores: &mut Vec<Vec<f32>>) {
        use rand::{thread_rng, Rng};
        
        let mut rng = thread_rng();
        let scale = 1.0 / (1.0 - self.config.attention_dropout);
        
        for row in scores.iter_mut() {
            for val in row.iter_mut() {
                if rng.gen::<f32>() < self.config.attention_dropout {
                    *val = 0.0;
                } else {
                    *val *= scale;
                }
            }
        }
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let (q_size, _) = self.q_proj.memory_usage();
        let (k_size, _) = self.k_proj.memory_usage();
        let (v_size, _) = self.v_proj.memory_usage();
        let (o_size, _) = self.out_proj.memory_usage();
        
        let total_compressed = q_size + k_size + v_size + o_size;
        let total_original = 4 * self.config.hidden_dim * self.config.hidden_dim * 4;  // 4 projections
        let total_ratio = total_original as f32 / total_compressed as f32;
        
        (total_compressed, total_ratio)
    }
    
    /// 캐시 초기화
    pub fn clear_cache(&mut self) {
        self.q_proj.clear_cache();
        self.k_proj.clear_cache();
        self.v_proj.clear_cache();
        self.out_proj.clear_cache();
    }
}

/// 사전 학습된 가중치에서 로드
impl RBEAttention {
    /// 사전 학습된 가중치로부터 RBEAttention 생성
    pub fn from_pretrained_weights(
        q_weights: &[f32],
        k_weights: &[f32],
        v_weights: &[f32],
        o_weights: &[f32],
        config: RBEAttentionConfig,
    ) -> Result<Self> {
        // 검증
        let weight_size = config.hidden_dim * config.hidden_dim;
        
        if q_weights.len() != weight_size || k_weights.len() != weight_size ||
           v_weights.len() != weight_size || o_weights.len() != weight_size {
            bail!("Weight size mismatch: expected {}, got Q:{}, K:{}, V:{}, O:{}", 
                  weight_size, q_weights.len(), k_weights.len(), 
                  v_weights.len(), o_weights.len());
        }
        
        // RBE 인코더 생성
        let mut encoder = match config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_b_grade(), // C급도 B급으로 처리
        };
        
        println!("사전 학습된 Attention 가중치 압축 시작...");
        
        // 각 projection 압축
        let q_blocks = Self::compress_weights(
            q_weights,
            config.hidden_dim,
            config.hidden_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        let k_blocks = Self::compress_weights(
            k_weights,
            config.hidden_dim,
            config.hidden_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        let v_blocks = Self::compress_weights(
            v_weights,
            config.hidden_dim,
            config.hidden_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        let o_blocks = Self::compress_weights(
            o_weights,
            config.hidden_dim,
            config.hidden_dim,
            config.block_size,
            &mut encoder,
        )?;
        
        // RBELinear 생성
        let q_proj = RBELinear::with_config(
            q_blocks,
            config.hidden_dim,
            config.hidden_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let k_proj = RBELinear::with_config(
            k_blocks,
            config.hidden_dim,
            config.hidden_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let v_proj = RBELinear::with_config(
            v_blocks,
            config.hidden_dim,
            config.hidden_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let out_proj = RBELinear::with_config(
            o_blocks,
            config.hidden_dim,
            config.hidden_dim,
            None,
            RBELinearConfig {
                enable_parallel: config.enable_parallel,
                cache_size: config.cache_size,
            },
        );
        
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        
        let attention = Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
            config,
        };
        
        let (compressed_size, ratio) = attention.memory_usage();
        println!("Attention 압축 완료! 압축률: {:.1}:1, 압축 크기: {:.2} MB", 
                 ratio, compressed_size as f32 / 1024.0 / 1024.0);
        
        Ok(attention)
    }
    
    /// 특정 가중치 압축 헬퍼
    fn compress_weights(
        weights: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        encoder: &mut RBEEncoder,
    ) -> Result<Vec<HybridEncodedBlock>> {
        let mut blocks = Vec::new();
        let dim = rows; // Use rows for block_size
        let block_size = block_size;
        
        let blocks_per_row = (dim + block_size - 1) / block_size;
        
        for row_block in 0..blocks_per_row {
            for col_block in 0..blocks_per_row {
                let row_start = row_block * block_size;
                let row_end = ((row_block + 1) * block_size).min(dim);
                let col_start = col_block * block_size;
                let col_end = ((col_block + 1) * block_size).min(dim);
                
                let block_rows = row_end - row_start;
                let block_cols = col_end - col_start;
                
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; block_rows * block_cols];
                for r in 0..block_rows {
                    for c in 0..block_cols {
                        let src_idx = (row_start + r) * dim + (col_start + c);
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
} 