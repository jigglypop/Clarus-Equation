//! RBE 기반 선형 레이어

use crate::core::{
    decoder::weight_generator::WeightGenerator,
    packed_params::HybridEncodedBlock,
};
use std::sync::Arc;
use anyhow::{Result, bail};

/// RBE 선형 레이어 설정
#[derive(Debug, Clone)]
pub struct RBELinearConfig {
    pub enable_parallel: bool,
    pub cache_size: usize,
}

impl Default for RBELinearConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            cache_size: 16,
}
    }
}

/// RBE 기반 선형 레이어
#[derive(Debug, Clone)]
pub struct RBELinear {
    /// 압축된 가중치 블록들
    pub blocks: Vec<HybridEncodedBlock>,
    /// 디코딩된 블록 캐시 (lazy)
    decoded_cache: Vec<Option<Arc<Vec<f32>>>>,
    /// 입력 크기
    pub in_features: usize,
    /// 출력 크기
    pub out_features: usize,
    /// 편향 (옵션)
    pub bias: Option<Vec<f32>>,
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    /// 설정
    config: RBELinearConfig,
    /// 행 당 블록 수
    blocks_per_row: usize,
}

impl RBELinear {
    /// 새로운 RBE 선형 레이어 생성
    pub fn new(
        blocks: Vec<HybridEncodedBlock>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        Self::with_config(blocks, in_features, out_features, bias, RBELinearConfig::default())
    }
    
    /// 설정과 함께 새로운 RBE 선형 레이어 생성
    pub fn with_config(
        blocks: Vec<HybridEncodedBlock>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
        config: RBELinearConfig,
    ) -> Self {
        let weight_generator = WeightGenerator::new();
        let cache = vec![None; blocks.len()];
        
        // blocks_per_row 계산
        let blocks_per_row = if blocks.is_empty() { 
            1 
        } else {
            // 첫 번째 블록의 cols가 블록 크기를 나타냄
            let block_size = blocks[0].cols;
            (in_features + block_size - 1) / block_size
        };

        Self {
            blocks,
            decoded_cache: cache,
            in_features,
            out_features,
            bias,
            weight_generator,
            config,
            blocks_per_row,
        }
    }
    
    /// 순전파
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features, "입력 크기 불일치");
        
        let mut output = vec![0.0; self.out_features];
        
        // 블록별로 디코딩하고 GEMV 수행
        for (idx, block) in self.blocks.iter().enumerate() {
            // 행렬 내 블록 위치 계산
            let blocks_per_row = (self.in_features + block.cols - 1) / block.cols;
            let block_row = idx / blocks_per_row;
            let block_col = idx % blocks_per_row;

            // 블록 디코딩 결과 캐싱
            let weights_arc = if let Some(ref cached) = self.decoded_cache[idx] {
                cached.clone()
            } else {
                let decoded = self.weight_generator.decode_block(block);
                self.decoded_cache[idx] = Some(decoded.clone());
                decoded
            };
            let weights = &*weights_arc;

            // GEMV 수행 (SIMD 최적화)
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use std::arch::x86_64::*;
                for local_r in 0..block.rows {
                    let out_row = block_row * block.rows + local_r;
                    if out_row >= self.out_features { break; }
                    
                    let mut acc_vec = _mm256_setzero_ps();
                    let weight_offset = local_r * block.cols;
                    
                    // 8개씩 SIMD 처리
                    let simd_chunks = block.cols / 8;
                    for chunk in 0..simd_chunks {
                        let local_c = chunk * 8;
                        let in_col = block_col * block.cols + local_c;
                        if in_col + 7 >= self.in_features { break; }
                        
                        let w_vec = _mm256_loadu_ps(weights[weight_offset + local_c..].as_ptr());
                        let in_vec = _mm256_loadu_ps(input[in_col..].as_ptr());
                        acc_vec = _mm256_fmadd_ps(w_vec, in_vec, acc_vec);
                    }
                    
                    // 누적값 합산
                    let mut acc_arr = [0.0f32; 8];
                    _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc_vec);
                    let mut acc = acc_arr.iter().sum::<f32>();
                    
                    // 나머지 처리
                    for local_c in (simd_chunks * 8)..block.cols {
                        let in_col = block_col * block.cols + local_c;
                        if in_col >= self.in_features { break; }
                        let w = weights[weight_offset + local_c];
                        acc += w * input[in_col];
                    }
                    
                    output[out_row] += acc;
                }
            }
            
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                // 기존 스칼라 코드
                for local_r in 0..block.rows {
                    let out_row = block_row * block.rows + local_r;
                    if out_row >= self.out_features { break; }
                    let mut acc = 0.0f32;
                    for local_c in 0..block.cols {
                        let in_col = block_col * block.cols + local_c;
                        if in_col >= self.in_features { break; }
                        let w = weights[local_r * block.cols + local_c];
                        acc += w * input[in_col];
                    }
                    output[out_row] += acc;
                }
            }
        }
        
        // 편향 추가 (있는 경우)
        if let Some(ref bias) = self.bias {
            for (o, b) in output.iter_mut().zip(bias.iter()) {
                *o += *b;
            }
        }
        
        output
    }
    
    /// 배치 순전파
    pub fn forward_batch(&mut self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter()
            .map(|input| self.forward(input))
            .collect()
    }

    /// 메모리 재사용을 위한 순전파 - output 버퍼에 직접 쓰기
    pub fn forward_into(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.in_features {
            bail!("입력 크기 불일치: expected {}, got {}", self.in_features, input.len());
        }
        if output.len() != self.out_features {
            bail!("출력 크기 불일치: expected {}, got {}", self.out_features, output.len());
        }
        
        // 출력 버퍼 초기화
        output.fill(0.0);
        
        // 블록별 처리
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let block_row = block_idx / self.blocks_per_row;
            let block_col = block_idx % self.blocks_per_row;
            
            // 캐시된 가중치 사용 또는 디코딩
            let weights = if let Some(ref cached) = self.decoded_cache[block_idx] {
                cached
            } else {
                let decoded = self.weight_generator.decode_block(block);
                self.decoded_cache[block_idx] = Some(decoded.clone());
                self.decoded_cache[block_idx].as_ref().unwrap()
            };
            
            // SIMD 또는 스칼라 연산
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use std::arch::x86_64::*;
                for local_r in 0..block.rows {
                    let out_row = block_row * block.rows + local_r;
                    if out_row >= self.out_features { break; }
                    
                    let mut acc_vec = _mm256_setzero_ps();
                    let weight_offset = local_r * block.cols;
                    
                    let mut local_c = 0;
                    while local_c + 8 <= block.cols {
                        let in_base = block_col * block.cols + local_c;
                        if in_base + 8 > self.in_features { break; }
                        
                        let w = _mm256_loadu_ps(&weights[weight_offset + local_c]);
                        let inp = _mm256_loadu_ps(&input[in_base]);
                        acc_vec = _mm256_fmadd_ps(w, inp, acc_vec);
                        local_c += 8;
                    }
                    
                    let mut acc_arr = [0.0f32; 8];
                    _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc_vec);
                    let mut acc = acc_arr.iter().sum::<f32>();
                    
                    // 나머지 처리
                    for local_c in (local_c..block.cols) {
                        let in_col = block_col * block.cols + local_c;
                        if in_col >= self.in_features { break; }
                        let w = weights[weight_offset + local_c];
                        acc += w * input[in_col];
                    }
                    
                    output[out_row] += acc;
                }
            }
            
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                for local_r in 0..block.rows {
                    let out_row = block_row * block.rows + local_r;
                    if out_row >= self.out_features { break; }
                    let mut acc = 0.0f32;
                    for local_c in 0..block.cols {
                        let in_col = block_col * block.cols + local_c;
                        if in_col >= self.in_features { break; }
                        let w = weights[local_r * block.cols + local_c];
                        acc += w * input[in_col];
                    }
                    output[out_row] += acc;
                }
            }
        }
        
        // 편향 추가
        if let Some(ref bias) = self.bias {
            for (o, b) in output.iter_mut().zip(bias.iter()) {
                *o += *b;
            }
        }
        
        Ok(())
    }

    /// 캐시 초기화
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
        for entry in &mut self.decoded_cache {
            *entry = None;
        }
    }
    
    /// 메모리 사용량 추정
    pub fn memory_usage(&self) -> (usize, f32) {
        let compressed_size = self.blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let original_size = self.in_features * self.out_features * std::mem::size_of::<f32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        (compressed_size, compression_ratio)
                }
            }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::packed_params::TransformType;

    #[test]
    fn test_basic_creation() {
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params: [1.0; 8],
            residuals: vec![],
            transform_type: TransformType::Dct,
        };
        
        let layer = RBELinear::new(vec![block], 4, 4, None);
        
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 4);
        assert!(layer.bias.is_none());
    }
} 