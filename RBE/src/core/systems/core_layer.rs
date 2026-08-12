//! # 핵심 레이어 구현
//!
//! EncodedLayer와 FusedEncodedLayer의 구현

use crate::packed_params::{
    HybridEncodedBlock, RbeParameters, EncodedBlockGradients,
    Packed128
};
use nalgebra::{DMatrix, DVector};
use std::ops::AddAssign;

#[derive(Debug)]
pub struct EncodedLayer {
    pub blocks: Vec<Vec<HybridEncodedBlock>>,
    pub block_rows: usize,
    pub block_cols: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

/// 융합 레이어: 디코딩 없는 정밀한 순전파/역전파
pub struct FusedEncodedLayer {
    /// 각 블록을 Packed128 파라미터로 표현
    pub weight_seeds: Vec<Vec<Packed128>>,
    pub block_rows: usize,
    pub block_cols: usize,
    pub block_height: usize,
    pub block_width: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl FusedEncodedLayer {
    pub fn new(
        weight_seeds: Vec<Vec<Packed128>>,
        block_height: usize,
        block_width: usize,
        total_rows: usize,
        total_cols: usize,
    ) -> Self {
        let block_rows = weight_seeds.len();
        let block_cols = if block_rows > 0 { weight_seeds[0].len() } else { 0 };
        
        Self {
            weight_seeds,
            block_rows,
            block_cols,
            block_height,
            block_width,
            total_rows,
            total_cols,
        }
    }

    /// 디코딩 없는 정밀한 순전파 (Fused RBE-GEMM)
    pub fn fused_forward_precise(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(x.nrows(), self.total_cols, "Input vector dimension mismatch");

        let mut y = DVector::from_element(self.total_rows, 0.0);

        for (block_i, block_row) in self.weight_seeds.iter().enumerate() {
            for (block_j, weight_seed) in block_row.iter().enumerate() {
                let y_start = block_i * self.block_height;
                let x_start = block_j * self.block_width;
                
                let x_slice = x.rows(x_start, self.block_width);

                // 블록 내 각 행에 대해 융합 연산
                for row_idx in 0..self.block_height {
                    if y_start + row_idx >= self.total_rows { break; }
                    
                    let mut dot_product = 0.0;

                    // 융합 내적 계산: 가중치를 즉석에서 생성하며 곱셈
                    for col_idx in 0..self.block_width {
                        if x_start + col_idx >= self.total_cols { break; }
                        
                        // 핵심: 디코딩 없이 가중치 생성 및 곱셈
                        let weight = weight_seed.fused_forward(
                            row_idx, 
                            col_idx, 
                            self.block_height, 
                            self.block_width
                        ) as f64;
                        
                        dot_product += weight * x_slice[col_idx];
                    }

                    y[y_start + row_idx] += dot_product;
                }
            }
        }
        y
    }

    /// 디코딩 없는 정밀한 역전파 (Fused Backpropagation)
    pub fn fused_backward_precise(
        &mut self,
        x: &DVector<f64>,
        d_loss_d_y: &DVector<f64>,
        learning_rate: f32,
    ) -> DVector<f64> {
        assert_eq!(d_loss_d_y.nrows(), self.total_rows, "Output gradient dimension mismatch");

        let mut d_loss_d_x = DVector::from_element(self.total_cols, 0.0);

        for (block_i, block_row) in self.weight_seeds.iter_mut().enumerate() {
            for (block_j, weight_seed) in block_row.iter_mut().enumerate() {
                let y_start = block_i * self.block_height;
                let x_start = block_j * self.block_width;

                let d_loss_d_y_slice = d_loss_d_y.rows(y_start, self.block_height);
                let x_slice = x.rows(x_start, self.block_width);

                // 블록별 융합 역전파
                for row_idx in 0..self.block_height {
                    if y_start + row_idx >= self.total_rows { break; }
                    
                    let output_grad = d_loss_d_y_slice[row_idx] as f32;

                    for col_idx in 0..self.block_width {
                        if x_start + col_idx >= self.total_cols { break; }
                        
                        let input_val = x_slice[col_idx] as f32;
                        
                        // 가중치에 대한 그래디언트
                        let weight_grad = output_grad * input_val;
                        
                        // 상태 전이 미분 적용 (hi 비트 업데이트)
                        weight_seed.apply_state_transition(weight_grad, row_idx, col_idx);
                        
                        // 연속 파라미터 그래디언트 계산 및 업데이트
                        let r_fp32 = f32::from_bits((weight_seed.lo >> 32) as u32);
                        let theta_fp32 = f32::from_bits(weight_seed.lo as u32);
                        
                        let eps = 1e-5;
                        
                        // r 파라미터 그래디언트
                        let mut seed_r_plus = *weight_seed;
                        seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                        let w_r_plus = seed_r_plus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let mut seed_r_minus = *weight_seed;
                        seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                        let w_r_minus = seed_r_minus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                        let grad_r = weight_grad * dr;
                        
                        // theta 파라미터 그래디언트
                        let mut seed_th_plus = *weight_seed;
                        seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                        let w_th_plus = seed_th_plus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let mut seed_th_minus = *weight_seed;
                        seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                        let w_th_minus = seed_th_minus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                        let grad_theta = weight_grad * dth;
                        
                        // 연속 파라미터 업데이트
                        let new_r = (r_fp32 - learning_rate * grad_r).clamp(0.0, 1.0);
                        let new_theta = theta_fp32 - learning_rate * grad_theta;
                        weight_seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
                        
                        // 입력에 대한 그래디언트: dx[j] += dy[i] * w[i][j]
                        let current_weight = weight_seed.fused_forward(row_idx, col_idx, self.block_height, self.block_width) as f64;
                        d_loss_d_x[x_start + col_idx] += output_grad as f64 * current_weight;
                    }
                }
            }
        }

        d_loss_d_x
    }

    /// Adam 옵티마이저를 사용한 고급 융합 역전파
    pub fn fused_backward_adam(
        &mut self,
        x: &DVector<f64>,
        d_loss_d_y: &DVector<f64>,
        momentum_r: &mut Vec<Vec<f32>>,
        velocity_r: &mut Vec<Vec<f32>>,
        momentum_theta: &mut Vec<Vec<f32>>,
        velocity_theta: &mut Vec<Vec<f32>>,
        epoch: i32,
        learning_rate: f32,
    ) -> DVector<f64> {
        let mut d_loss_d_x = DVector::from_element(self.total_cols, 0.0);

        for (block_i, block_row) in self.weight_seeds.iter_mut().enumerate() {
            for (block_j, weight_seed) in block_row.iter_mut().enumerate() {
                let y_start = block_i * self.block_height;
                let x_start = block_j * self.block_width;

                let d_loss_d_y_slice = d_loss_d_y.rows(y_start, self.block_height);
                let x_slice = x.rows(x_start, self.block_width);

                let mut grad_r_sum = 0.0;
                let mut grad_theta_sum = 0.0;

                // 블록 내 그래디언트 누적
                for row_idx in 0..self.block_height {
                    if y_start + row_idx >= self.total_rows { break; }
                    
                    let output_grad = d_loss_d_y_slice[row_idx] as f32;

                    for col_idx in 0..self.block_width {
                        if x_start + col_idx >= self.total_cols { break; }
                        
                        let input_val = x_slice[col_idx] as f32;
                        let weight_grad = output_grad * input_val;
                        
                        // 상태 전이 미분
                        weight_seed.apply_state_transition(weight_grad, row_idx, col_idx);
                        
                        // 연속 파라미터 그래디언트 계산
                        let r_fp32 = f32::from_bits((weight_seed.lo >> 32) as u32);
                        let theta_fp32 = f32::from_bits(weight_seed.lo as u32);
                        let eps = 1e-5;
                        
                        // r 그래디언트
                        let mut seed_r_plus = *weight_seed;
                        seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                        let w_r_plus = seed_r_plus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let mut seed_r_minus = *weight_seed;
                        seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                        let w_r_minus = seed_r_minus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                        grad_r_sum += weight_grad * dr;
                        
                        // theta 그래디언트
                        let mut seed_th_plus = *weight_seed;
                        seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                        let w_th_plus = seed_th_plus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let mut seed_th_minus = *weight_seed;
                        seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                        let w_th_minus = seed_th_minus.fused_forward(row_idx, col_idx, self.block_height, self.block_width);
                        
                        let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                        grad_theta_sum += weight_grad * dth;
                        
                        // 입력 그래디언트 계산
                        let current_weight = weight_seed.fused_forward(row_idx, col_idx, self.block_height, self.block_width) as f64;
                        d_loss_d_x[x_start + col_idx] += output_grad as f64 * current_weight;
                    }
                }

                // Adam 업데이트 적용
                let r_fp32 = f32::from_bits((weight_seed.lo >> 32) as u32);
                let theta_fp32 = f32::from_bits(weight_seed.lo as u32);
                
                let mut new_r = r_fp32;
                let mut new_theta = theta_fp32;
                
                crate::math::adam_update(
                    &mut new_r, 
                    &mut momentum_r[block_i][block_j], 
                    &mut velocity_r[block_i][block_j], 
                    grad_r_sum, 
                    learning_rate, 
                    epoch
                );
                crate::math::adam_update(
                    &mut new_theta, 
                    &mut momentum_theta[block_i][block_j], 
                    &mut velocity_theta[block_i][block_j], 
                    grad_theta_sum, 
                    learning_rate, 
                    epoch
                );
                
                new_r = new_r.clamp(0.0, 1.0);
                weight_seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
            }
        }

        d_loss_d_x
    }
}

impl EncodedLayer {
    pub fn new(blocks: Vec<Vec<HybridEncodedBlock>>, total_rows: usize, total_cols: usize) -> Self {
        let block_rows = blocks.len();
        let block_cols = if block_rows > 0 { blocks[0].len() } else { 0 };
        Self {
            blocks,
            block_rows,
            block_cols,
            total_rows,
            total_cols,
        }
    }

    pub fn fused_forward(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(x.nrows(), self.total_cols, "Input vector dimension mismatch");

        let mut y = DVector::from_element(self.total_rows, 0.0);
        let block_height = self.total_rows / self.block_rows;
        let block_width = self.total_cols / self.block_cols;

        for (i, block_row) in self.blocks.iter().enumerate() {
            for (j, block) in block_row.iter().enumerate() {
                let y_start = i * block_height;
                let x_start = j * block_width;
                
                let x_slice = x.rows(x_start, block_width).into_owned();

                for row_idx in 0..block_height {
                    let mut dot_product = 0.0;

                    // 1. RBE Base Pattern Contribution
                    let row_pattern_contribution = self.calculate_rbe_row_dot_product(
                        &block.rbe_params,
                        row_idx,
                        block_height,
                        block_width,
                        &x_slice,
                    );
                    dot_product += row_pattern_contribution;
                    
                    // 2. DCT/DWT Residual Contribution
                    let residual_row = self.calculate_residual_row_dot_product(block, row_idx, &x_slice);
                    dot_product += residual_row;

                    y[y_start + row_idx] += dot_product;
                }
            }
        }
        y
    }

    pub fn fused_backward(
        &self,
        _: &DVector<f64>,      // Original input from the forward pass
        d_loss_d_y: &DVector<f64>, // Gradient from the next layer
    ) -> (DVector<f64>, Vec<Vec<EncodedBlockGradients>>) {
        assert_eq!(d_loss_d_y.nrows(), self.total_rows, "Output gradient dimension mismatch");

        let mut d_loss_d_x = DVector::from_element(self.total_cols, 0.0);
        let mut param_grads = Vec::new();

        let block_height = self.total_rows / self.block_rows;
        let block_width = self.total_cols / self.block_cols;

        for (i, block_row) in self.blocks.iter().enumerate() {
            let mut grad_row = Vec::new();
            for (j, block) in block_row.iter().enumerate() {
                // To ensure correctness, we first decode the full weight matrix for the block.
                // A performant implementation would not do this.
                let w_matrix = DMatrix::from_vec(
                    block.rows,
                    block.cols,
                    block.decode()
                ).map(|e| e as f64);
                let y_start = i * block_height;
                let x_start = j * block_width;
                let d_loss_d_y_slice = d_loss_d_y.rows(y_start, block_height);
                // 1. Calculate d_loss/d_x for this block: W^T * d_loss_d_y_slice
                let d_loss_d_x_block = w_matrix.transpose() * d_loss_d_y_slice;
                d_loss_d_x.rows_mut(x_start, block_width).add_assign(&d_loss_d_x_block);
                // 2. Calculate d_loss/d_params for this block
                // d_loss/d_W = d_loss_d_y_slice * x_slice^T (outer product)
                // This is the hardest part: projecting d_loss/d_W back onto the gradients
                // of the RBE params and residual coefficients. This requires the chain rule
                // through the decoding process. For now, we'll leave it as a placeholder.
                let rbe_params_grad = [0.0f32; 8]; // Placeholder
                let residuals_grad = Vec::new();   // Placeholder

                grad_row.push(EncodedBlockGradients {
                    rbe_params_grad,
                    residuals_grad,
                });
            }
            param_grads.push(grad_row);
        }

        (d_loss_d_x, param_grads)
    }
    
    fn calculate_rbe_row_dot_product(
        &self,
        params: &RbeParameters,
        row_idx: usize,
        block_height: usize,
        block_width: usize,
        x_slice: &DVector<f64>,
    ) -> f64 {
        let mut dot_product = 0.0;
        let y_norm = (row_idx as f32 / (block_height.saturating_sub(1)) as f32) * 2.0 - 1.0;
        
        for col_idx in 0..block_width {
            let x_norm = (col_idx as f32 / (block_width.saturating_sub(1)) as f32) * 2.0 - 1.0;
            let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
            let pi = std::f32::consts::PI;
            
            let basis = [
                1.0, d, d * d, (pi * x_norm).cos(), (pi * y_norm).cos(),
                (2.0 * pi * x_norm).cos(), (2.0 * pi * y_norm).cos(),
                (pi * x_norm).cos() * (pi * y_norm).cos(),
            ];
            
            let val: f32 = params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
            dot_product += val as f64 * x_slice[col_idx];
        }
        dot_product
    }
    
    fn calculate_residual_row_dot_product(
        &self,
        block: &HybridEncodedBlock,
        row_idx: usize,
        x_slice: &DVector<f64>,
    ) -> f64 {
        if block.residuals.is_empty() {
            return 0.0;
        }
        let decoded_block_data = block.decode();
        let decoded_matrix = DMatrix::from_vec(block.rows, block.cols, decoded_block_data);
        let mut rbe_pattern_vec = DVector::<f32>::zeros(block.rows * block.cols);
        for r in 0..block.rows {
            for c in 0..block.cols {
                let y_norm = (r as f32 / (block.rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let x_norm = (c as f32 / (block.cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
                let pi = std::f32::consts::PI;
                
                let basis = [
                    1.0, d, d * d, (pi * x_norm).cos(), (pi * y_norm).cos(),
                    (2.0 * pi * x_norm).cos(), (2.0 * pi * y_norm).cos(),
                    (pi * x_norm).cos() * (pi * y_norm).cos(),
                ];
                
                let val: f32 = block.rbe_params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
                rbe_pattern_vec[r * block.cols + c] = val;
            }
        }
        let rbe_matrix = DMatrix::from_vec(block.rows, block.cols, rbe_pattern_vec.data.into());
        let residual_matrix = decoded_matrix - rbe_matrix;
        residual_matrix.row(row_idx).transpose().dot(&x_slice.map(|v| v as f32)) as f64
    }
} 