//! 블록 단위 디코딩 모듈

use crate::packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient};
use crate::core::math::{compute_rbe_basis_1d, compute_rbe_basis, normalize_coords, compute_pixel_value};

impl HybridEncodedBlock {
    /// 압축된 블록을 원본 데이터로 디코딩
    pub fn decode(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;
        let total_size = rows * cols;
        
        // RBE 기본 패턴 복원
        let mut reconstruction = vec![0.0f32; total_size];
        
        // 1차원 벡터인 경우 (임베딩 등)
        if rows == 1 {
            // 1차원에서는 y=0으로 고정하고 x만 변화
            for col in 0..cols {
                let x = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                
                // RBE 기저 함수 계산
                let basis_values = compute_rbe_basis_1d(x);
                
                // RBE 파라미터 적용
                reconstruction[col] = compute_pixel_value(&self.rbe_params, &basis_values);
            }
        } else {
            // 2차원 이상인 경우 기존 로직 사용
            for idx in 0..total_size {
                let row = idx / cols;
                let col = idx % cols;
                
                // 픽셀 좌표를 [-1, 1] 범위로 정규화
                let (x, y) = normalize_coords(row, col, rows, cols);
                
                // RBE 기저 함수 계산
                let basis_values = compute_rbe_basis(x, y);
                
                // RBE 파라미터 적용
                reconstruction[idx] = compute_pixel_value(&self.rbe_params, &basis_values);
            }
        }
        
        reconstruction
    }
} 