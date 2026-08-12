//! 융합 순전파 (Fused Forward Pass) - RBE 기반 최적화

use crate::packed_params::HybridEncodedBlock;
use super::weight_generator::WeightGenerator;
use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;

/// 융합 순전파 처리기
#[derive(Clone)]
pub struct FusedForwardPass {
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    /// 병렬 처리 설정
    enable_parallel: bool,
}

impl FusedForwardPass {
    /// 새로운 융합 순전파 처리기 생성
    pub fn new() -> Self {
        Self {
            weight_generator: WeightGenerator::new(),
            enable_parallel: true,
        }
    }
    
    /// 블록 기반 행렬-벡터 곱셈 (GEMV)
    pub fn block_gemv(
        &self,
        blocks: &[HybridEncodedBlock],
        input: &[f32],
        output: &mut [f32],
        block_layout: &BlockLayout,
    ) {
        if self.enable_parallel {
            self.parallel_block_gemv(blocks, input, output, block_layout);
        } else {
            self.sequential_block_gemv(blocks, input, output, block_layout);
        }
    }
    
    /// 순차 처리 버전
    fn sequential_block_gemv(
        &self,
        blocks: &[HybridEncodedBlock],
        input: &[f32],
        output: &mut [f32],
        block_layout: &BlockLayout,
    ) {
        output.fill(0.0);
        
        for (block_idx, block) in blocks.iter().enumerate() {
            let (block_row, block_col) = block_layout.get_block_position(block_idx);
            let start_row = block_row * block_layout.block_size;
            let start_col = block_col * block_layout.block_size;
            
            // 블록 디코딩
            let weights = self.weight_generator.decode_block(block);
            
            // 블록 GEMV
            for local_row in 0..block.rows {
                let global_row = start_row + local_row;
                if global_row >= output.len() {
                    break;
                }
                
                let mut sum = 0.0f32;
                for local_col in 0..block.cols {
                    let global_col = start_col + local_col;
                    if global_col >= input.len() {
                        break;
                    }
                    
                    let weight_idx = local_row * block.cols + local_col;
                    if weight_idx < weights.len() {
                        sum += weights[weight_idx] * input[global_col];
                    }
                }
                
                output[global_row] += sum;
            }
        }
    }
    
    /// 병렬 처리 버전
    fn parallel_block_gemv(
        &self,
        blocks: &[HybridEncodedBlock],
        input: &[f32],
        output: &mut [f32],
        block_layout: &BlockLayout,
    ) {
        // 출력을 Arc<RwLock>으로 감싸서 병렬 쓰기 가능하게
        let output_arc = Arc::new(RwLock::new(output));
        
        // 블록별 부분 결과 계산
        let partial_results: Vec<(usize, Vec<(usize, f32)>)> = blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, block)| {
                let (block_row, block_col) = block_layout.get_block_position(block_idx);
                let start_row = block_row * block_layout.block_size;
                let start_col = block_col * block_layout.block_size;
                
                // 블록 디코딩
                let weights = self.weight_generator.decode_block(block);
                
                // 블록별 결과 계산
                let mut block_results = Vec::new();
                
                for local_row in 0..block.rows {
                    let global_row = start_row + local_row;
                    
                    let mut sum = 0.0f32;
                    for local_col in 0..block.cols {
                        let global_col = start_col + local_col;
                        if global_col < input.len() {
                            let weight_idx = local_row * block.cols + local_col;
                            if weight_idx < weights.len() {
                                sum += weights[weight_idx] * input[global_col];
                            }
                        }
                    }
                    
                    if sum != 0.0 {
                        block_results.push((global_row, sum));
                    }
                }
                
                (block_idx, block_results)
            })
            .collect();
        
        // 결과 병합
        let mut output_guard = output_arc.write();
        output_guard.fill(0.0);
        
        for (_block_idx, results) in partial_results {
            for (row, value) in results {
                if row < output_guard.len() {
                    output_guard[row] += value;
                }
            }
        }
    }
    
    /// 배치 처리
    pub fn batch_forward(
        &self,
        blocks: &[HybridEncodedBlock],
        inputs: &[Vec<f32>],
        block_layout: &BlockLayout,
    ) -> Vec<Vec<f32>> {
        inputs
            .par_iter()
            .map(|input| {
                let mut output = vec![0.0; block_layout.total_rows];
                self.block_gemv(blocks, input, &mut output, block_layout);
                output
            })
            .collect()
    }
    
    /// 캐시 정리
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
    }
    
    /// 병렬 처리 설정
    pub fn set_parallel(&mut self, enable: bool) {
        self.enable_parallel = enable;
    }
}

/// 블록 레이아웃 정보
#[derive(Debug, Clone)]
pub struct BlockLayout {
    pub total_rows: usize,
    pub total_cols: usize,
    pub block_size: usize,
    pub grid_rows: usize,
    pub grid_cols: usize,
}

impl BlockLayout {
    pub fn new(total_rows: usize, total_cols: usize, block_size: usize) -> Self {
        let grid_rows = (total_rows + block_size - 1) / block_size;
        let grid_cols = (total_cols + block_size - 1) / block_size;
        
        Self {
            total_rows,
            total_cols,
            block_size,
            grid_rows,
            grid_cols,
        }
    }
    
    pub fn get_block_position(&self, block_idx: usize) -> (usize, usize) {
        let block_row = block_idx / self.grid_cols;
        let block_col = block_idx % self.grid_cols;
        (block_row, block_col)
    }
    
    pub fn get_block_index(&self, block_row: usize, block_col: usize) -> usize {
        block_row * self.grid_cols + block_col
    }
}

impl Default for FusedForwardPass {
    fn default() -> Self {
        Self::new()
    }
}

/// 공유 가중치 캐시 (더미 구현 - 호환성용)
#[derive(Debug, Clone)]
pub struct SharedWeightCache;

impl SharedWeightCache {
    pub fn new() -> Self {
        Self
    }
    
    pub fn get_weights(&self, _key: &(u64, u64, usize, usize)) -> Option<Vec<f32>> {
        None
    }
    
    pub fn store_weights(&self, _key: (u64, u64, usize, usize), _weights: Vec<f32>) {
        // 더미 구현
    }
} 