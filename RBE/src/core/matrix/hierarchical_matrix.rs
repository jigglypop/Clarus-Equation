use crate::packed_params::Packed128;

use super::{QualityLevel, ErrorController, L1Block, L2Block, QualityStats};

/// 계층적 블록 분할 시스템
/// 4단계 분할 구조로 대규모 행렬을 효율적으로 처리합니다.
/// L1: 4096×4096 → L2: 1024×1024 → L3: 256×256 → L4: 64×64
#[derive(Debug, Clone)]
pub struct HierarchicalBlockMatrix {
    /// 전체 행렬 크기
    pub total_rows: usize,
    pub total_cols: usize,
    /// 4단계 블록 구조
    pub l1_blocks: Vec<Vec<L1Block>>,
    /// 품질 등급별 설정
    pub quality_level: QualityLevel,
    /// 오차 제어 시스템
    pub error_controller: ErrorController,
}

impl HierarchicalBlockMatrix {
    /// 새로운 계층적 블록 행렬 생성
    pub fn new(rows: usize, cols: usize, quality: QualityLevel) -> Self {
        let error_threshold = match quality {
            QualityLevel::Ultra => 1e-4,
            QualityLevel::High => 1e-3,
            QualityLevel::Medium => 1e-2,
            QualityLevel::Low => 1e-1,
        };
        
        Self {
            total_rows: rows,
            total_cols: cols,
            l1_blocks: Vec::new(),
            quality_level: quality,
            error_controller: ErrorController::new(error_threshold),
        }
    }
    
    /// 적응적 블록 분할 수행
    pub fn adaptive_partition(&mut self, source_matrix: &[f32]) {
        let l1_block_size = 4096;
        
        // L1 블록들 생성
        for i in (0..self.total_rows).step_by(l1_block_size) {
            let mut l1_row = Vec::new();
            
            for j in (0..self.total_cols).step_by(l1_block_size) {
                let rows = (l1_block_size).min(self.total_rows - i);
                let cols = (l1_block_size).min(self.total_cols - j);
                
                let l1_block = self.create_l1_block(source_matrix, i, j, rows, cols);
                l1_row.push(l1_block);
            }
            
            self.l1_blocks.push(l1_row);
        }
    }
    
    /// L1 블록 생성
    fn create_l1_block(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                       rows: usize, cols: usize) -> L1Block {
        // 전역 파라미터 추정
        let global_params = self.estimate_global_parameters(source_matrix, row_start, col_start, rows, cols);
        
        // 압축 오차 계산
        let compression_error = self.compute_compression_error(source_matrix, &global_params, 
                                                             row_start, col_start, rows, cols);
        
        // 오차 기록
        self.error_controller.update_block_error((row_start, col_start), compression_error);
        
        // L2 블록들 생성 (재귀적 분할)
        let l2_blocks = if self.error_controller.should_subdivide((row_start, col_start), 1) {
            self.create_l2_blocks(source_matrix, row_start, col_start, rows, cols)
        } else {
            Vec::new() // 분할 불필요
        };
        
        L1Block {
            row_start,
            col_start,
            rows,
            cols,
            l2_blocks,
            global_params,
        }
    }
    
    /// L2 블록들 생성
    fn create_l2_blocks(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                        rows: usize, cols: usize) -> Vec<Vec<L2Block>> {
        let l2_block_size = 1024;
        let mut l2_blocks = Vec::new();
        
        for i in (0..rows).step_by(l2_block_size) {
            let mut l2_row = Vec::new();
            
            for j in (0..cols).step_by(l2_block_size) {
                let sub_rows = l2_block_size.min(rows - i);
                let sub_cols = l2_block_size.min(cols - j);
                
                let l2_block = self.create_l2_block(source_matrix, 
                                                  row_start + i, col_start + j, 
                                                  sub_rows, sub_cols);
                l2_row.push(l2_block);
            }
            
            l2_blocks.push(l2_row);
        }
        
        l2_blocks
    }
    
    /// L2 블록 생성 (단순화된 버전)
    fn create_l2_block(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                       rows: usize, cols: usize) -> L2Block {
        let macro_params = self.estimate_local_parameters(source_matrix, row_start, col_start, rows, cols);
        
        L2Block {
            row_start,
            col_start,
            rows,
            cols,
            l3_blocks: Vec::new(), // 단순화
            macro_params,
        }
    }
    
    /// 전역 파라미터 추정
    fn estimate_global_parameters(&self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                                  rows: usize, cols: usize) -> Packed128 {
        // 간단한 샘플링 기반 파라미터 추정
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..rows.min(100) { // 샘플링
            for j in 0..cols.min(100) {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        sum += source_matrix[idx];
                        count += 1;
                    }
                }
            }
        }
        
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
        
        // Packed128으로 변환 (기본값 사용)
        Packed128::default()
    }
    
    /// 지역 파라미터 추정
    fn estimate_local_parameters(&self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                                 rows: usize, cols: usize) -> Packed128 {
        // 전역 파라미터와 동일한 로직 사용 (단순화)
        self.estimate_global_parameters(source_matrix, row_start, col_start, rows, cols)
    }
    
    /// 압축 오차 계산
    fn compute_compression_error(&self, source_matrix: &[f32], params: &Packed128,
                                 row_start: usize, col_start: usize, rows: usize, cols: usize) -> f32 {
        let mut error_sum = 0.0;
        let mut count = 0;
        
        for i in 0..rows.min(50) { // 샘플링
            for j in 0..cols.min(50) {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        let original = source_matrix[idx];
                        let reconstructed = params.fused_forward(i, j, rows, cols);
                        error_sum += (original - reconstructed).powi(2);
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 { (error_sum / count as f32).sqrt() } else { 0.0 }
    }
    
    /// 병렬 GEMV 연산 (단순화된 버전)
    pub fn parallel_gemv(&self, input: &[f32], output: &mut [f32], _num_threads: usize) {
        output.fill(0.0);
        
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                // 간단한 순차 처리
                for i in 0..l1_block.rows {
                    for j in 0..l1_block.cols {
                        if l1_block.col_start + j < input.len() && l1_block.row_start + i < output.len() {
                            let weight = l1_block.global_params.fused_forward(i, j, l1_block.rows, l1_block.cols);
                            output[l1_block.row_start + i] += weight * input[l1_block.col_start + j];
                        }
                    }
                }
            }
        }
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let mut total_memory = 0;
        let mut total_blocks = 0;
        
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                total_memory += std::mem::size_of::<L1Block>();
                total_blocks += 1;
                
                for l2_row in &l1_block.l2_blocks {
                    for l2_block in l2_row {
                        total_memory += std::mem::size_of::<L2Block>();
                        total_blocks += 1;
                    }
                }
            }
        }
        
        let original_memory = self.total_rows * self.total_cols * std::mem::size_of::<f32>();
        let compression_ratio = if total_memory > 0 {
            original_memory as f32 / total_memory as f32
        } else {
            1.0
        };
        
        (total_memory, compression_ratio)
    }
    
    /// 품질 통계 계산
    pub fn quality_statistics(&self) -> QualityStats {
        let total_error = self.error_controller.compute_total_error();
        let (memory_bytes, compression_ratio) = self.memory_usage();
        
        // PSNR 계산 (간단화)
        let psnr = if total_error > 0.0 {
            20.0 * (1.0 / total_error).log10()
        } else {
            60.0 // 완벽한 복원
        };
        
        let mut total_blocks = 0;
        for l1_row in &self.l1_blocks {
            total_blocks += l1_row.len();
        }
        
        QualityStats {
            total_error,
            psnr,
            compression_ratio,
            memory_usage_bytes: memory_bytes,
            total_blocks,
        }
    }
} 