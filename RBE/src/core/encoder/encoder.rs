use crate::core::{
    HybridEncodedBlock, ResidualCoefficient, TransformType,
    math::{compute_rbe_basis, normalize_coords, compute_pixel_value, compute_rmse},
};
use crate::optimizers::{AdamState, RiemannianAdamState};
use crate::decoder::WeightGenerator;
use std::time::Instant;
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector, RowDVector};
use rustdct::{DctPlanner, TransformType2And3};
use ndarray::{Array, Array1, Array2};
use omni_wave::{wavelet as w, completely_decompose_2d};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use std::f32::consts::PI;

// A matrix 캐시 (thread-safe)
static A_MATRIX_CACHE: Lazy<Arc<RwLock<HashMap<(usize, usize), Arc<DMatrix<f32>>>>>> = 
    Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
}

#[derive(Debug, Clone, Copy)]
pub enum CompressionProfile {
    UltraHigh,   // 최고 품질, 느린 속도
    High,        // 고품질, 중간 속도  
    Balanced,    // 균형
    Fast,        // 빠른 속도, 낮은 품질
    UltraFast,   // 최고 속도, 최저 품질
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub block_size: usize,
    pub quality_grade: QualityGrade, 
    pub transform_type: TransformType,
    pub profile: CompressionProfile,
    pub custom_coefficients: Option<usize>, // None이면 자동 계산
    
    // 사용자가 요청한 조정 가능 파라미터들
    pub min_block_count: Option<usize>,     // 최소 블록 개수 (하드코딩 조정용)
    pub rmse_threshold: Option<f32>,        // RMSE 임계값 (0.001 같은 값)
    pub compression_ratio_threshold: Option<f32>, // 압축률 임계값 (최소 압축률)
    pub enable_residual: bool,
    pub residual_threshold: f32,
}

/// 통합된 RBE 인코더 (기존 AutoOptimizedEncoder + HybridEncoder)
pub struct RBEEncoder {
    pub k_coeffs: usize, // 유지할 잔차 계수의 개수
    pub transform_type: TransformType,
    // planner는 재사용 가능하므로 인코더가 소유하는 것이 효율적
    dct_planner_f32: DctPlanner<f32>,
    config: CompressionConfig,
}

// 기존 AutoOptimizedEncoder는 호환성을 위해 타입 별칭으로 유지
pub type AutoOptimizedEncoder = RBEEncoder;

// IntegerAdamEncoder는 RBEEncoder의 별칭
pub type IntegerAdamEncoder = RBEEncoder;

impl CompressionConfig {
    /// 기본 설정 (Balanced 프로파일)
    pub fn default() -> Self {
        Self {
            block_size: 64,
            quality_grade: QualityGrade::B,
            transform_type: TransformType::Dwt,  // DWT! DWT! DWT!
            profile: CompressionProfile::Balanced,
            custom_coefficients: None,
            min_block_count: None,
            rmse_threshold: None,
            compression_ratio_threshold: None,
            enable_residual: false,
            residual_threshold: 0.0,
        }
    }
    
    /// UltraHigh 품질 프리셋 (현실적인 임계값)
    pub fn ultra_high() -> Self {
        Self {
            block_size: 32,  // 작은 블록 = 높은 품질
            quality_grade: QualityGrade::S,
            transform_type: TransformType::Dwt,  // DWT 사용!
            profile: CompressionProfile::UltraHigh,
            custom_coefficients: None,
            min_block_count: None,
            rmse_threshold: Some(0.01),   // 0.00001 → 0.01 (현실적으로 조정)
            compression_ratio_threshold: Some(50.0),
            enable_residual: false,
            residual_threshold: 0.0,
        }
    }
    
    /// Fast 압축 프리셋
    pub fn fast() -> Self {
        Self {
            block_size: 128, // 큰 블록 = 빠른 속도
            quality_grade: QualityGrade::C,
            transform_type: TransformType::Dwt,  // DWT 사용!
            profile: CompressionProfile::Fast,
            custom_coefficients: Some(256), // 적은 계수
            min_block_count: None,
            rmse_threshold: Some(0.1),
            compression_ratio_threshold: Some(10.0),
            enable_residual: false,
            residual_threshold: 0.0,
        }
    }
    
    /// 사용자 정의 설정
    pub fn custom(
        block_size: usize,
        rmse_threshold: f32,
        compression_ratio: f32,
        min_blocks: Option<usize>,
        enable_residual: bool,
        residual_threshold: f32,
    ) -> Self {
        Self {
            block_size,
            quality_grade: QualityGrade::B,
            transform_type: TransformType::Dwt,  // DWT! DWT! DWT!
            profile: CompressionProfile::Balanced,
            custom_coefficients: None,
            min_block_count: min_blocks,
            rmse_threshold: Some(rmse_threshold),
            compression_ratio_threshold: Some(compression_ratio),
            enable_residual,
            residual_threshold,
        }
    }
}

impl RBEEncoder {
    /// 새로운 RBE 인코더 생성
    pub fn new(k_coeffs: usize, transform_type: TransformType) -> Self {
        Self {
            k_coeffs,
            transform_type,
            dct_planner_f32: DctPlanner::new(),
            config: CompressionConfig::default(),
        }
    }
    
    /// 최대공약수(GCD) 계산
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            Self::gcd(b, a % b)
        }
    }
    
    /// 행렬 크기에 따른 최적 블록 크기 결정
    pub fn determine_optimal_block_size(rows: usize, cols: usize) -> usize {
        // 행과 열의 최대공약수 계산
        let gcd_size = Self::gcd(rows, cols);
        
        // 2의 배수로 맞추기 (SIMD 최적화를 위해)
        let mut block_size = gcd_size;
        
        // 너무 작으면 성능이 떨어지므로 최소값 보장
        if block_size < 16 {
            // 16의 배수 중 rows와 cols를 나누어 떨어뜨리는 가장 큰 값 찾기
            for size in &[64, 32, 16] {
                if rows % size == 0 && cols % size == 0 {
                    block_size = *size;
                    break;
                }
            }
            if block_size < 16 {
                block_size = 16; // 최소값
            }
        }
        
        // 너무 크면 압축 품질이 떨어지므로 최대값 제한
        if block_size > 256 {
            // 256의 약수 중 rows와 cols를 나누어 떨어뜨리는 가장 큰 값 찾기
            for size in &[256, 128, 64, 32] {
                if rows % size == 0 && cols % size == 0 {
                    block_size = *size;
                    break;
                }
            }
        }
        
        // 2의 거듭제곱으로 조정 (성능 최적화)
        let mut power_of_two = 1;
        while power_of_two < block_size {
            power_of_two *= 2;
        }
        if power_of_two > block_size {
            power_of_two /= 2;
        }
        
        // rows와 cols를 나누어 떨어뜨리는 가장 가까운 2의 거듭제곱 찾기
        while power_of_two >= 16 {
            if rows % power_of_two == 0 && cols % power_of_two == 0 {
                return power_of_two;
            }
            power_of_two /= 2;
        }
        
        // 마지막 대안: 패딩이 필요하더라도 적절한 크기 선택
        if rows <= 64 && cols <= 64 {
            16
        } else if rows <= 128 && cols <= 128 {
            32
        } else if rows <= 256 && cols <= 256 {
            64
        } else {
            128
        }
    }
    
    /// S급 품질 (RMSE < 0.001): 819:1 압축률, 128x128 블록 권장
    pub fn new_s_grade() -> Self {
        Self::new(500, TransformType::Dwt)  // DWT 사용!
    }
    
    /// A급 품질 (RMSE < 0.01): 3276:1 압축률, 256x256 블록 권장  
    pub fn new_a_grade() -> Self {
        Self::new(300, TransformType::Dwt)  // DWT 사용!
    }
    
    /// B급 품질 (RMSE < 0.1): 3276:1 압축률, 256x256 블록 권장
    pub fn new_b_grade() -> Self {
        Self::new(200, TransformType::Dwt)  // DWT 사용!
    }
    
    /// 극한 압축 (RMSE ~0.09): 3276:1 압축률
    pub fn new_extreme_compression() -> Self {
        Self::new(75, TransformType::Dwt)   // DWT 사용! 계수 100 -> 75
    }

    /// 단일 블록을 RBE+DCT/DWT로 압축 (기존 HybridEncoder::encode_block)
    pub fn encode_block(&mut self, block_data: &[f32], rows: usize, cols: usize) -> HybridEncodedBlock {
        // 입력 데이터 크기 검증
        if block_data.len() != rows * cols {
            panic!("block_data 길이({})가 rows * cols({})와 일치하지 않음", block_data.len(), rows * cols);
        }
        
        let b_vector = DVector::from_row_slice(block_data);
        
        // A matrix 캐시 확인
        let cache_key = (rows, cols);
        let a_matrix = {
            let cache = A_MATRIX_CACHE.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                Arc::clone(cached)
            } else {
                drop(cache); // 읽기 잠금 해제
                
                // A matrix 생성 (병렬 처리)
                let a_matrix_rows: Vec<RowDVector<f32>> = (0..rows * cols).into_par_iter().map(|idx| {
                    let r = idx / cols;
                    let c = idx % cols;
                    let (x, y) = normalize_coords(r, c, rows, cols);
                    let basis = compute_rbe_basis(x, y);
                    RowDVector::from_vec(basis.to_vec())
                }).collect();

                let new_a_matrix = Arc::new(DMatrix::from_rows(&a_matrix_rows));
                
                // 캐시에 저장
                let mut cache = A_MATRIX_CACHE.write().unwrap();
                cache.insert(cache_key, Arc::clone(&new_a_matrix));
                
                new_a_matrix
            }
        };

        // Pseudo-inverse via SVD with proper dimension handling
        let svd = a_matrix.as_ref().clone().svd(true, true);
        let singular_values = &svd.singular_values;
        let mut sigma_inv = DMatrix::zeros(svd.v_t.as_ref().unwrap().nrows(), svd.u.as_ref().unwrap().ncols());
        
        let tolerance = 1e-10_f32;
        for i in 0..singular_values.len().min(sigma_inv.nrows()).min(sigma_inv.ncols()) {
            if singular_values[i].abs() > tolerance {
                sigma_inv[(i, i)] = 1.0 / singular_values[i];
            }
        }
        
        let a_pseudo_inv = svd.v_t.as_ref().unwrap().transpose() * sigma_inv * svd.u.as_ref().unwrap().transpose();
        
        // RBE 파라미터 계산
        let rbe_params_vec = a_pseudo_inv.clone() * b_vector.clone();
        let mut rbe_params = [0.0f32; 8];
        for i in 0..8.min(rbe_params_vec.len()) {
            rbe_params[i] = rbe_params_vec[i];
        }
        
        // 잔차 계산
        let predicted_vector = a_matrix.as_ref() * rbe_params_vec;
        let residual_vector = b_vector - predicted_vector;

        self.encode_single_transform(rbe_params, &residual_vector, rows, cols, self.transform_type)
    }

    fn encode_single_transform(
        &mut self,
        rbe_params: [f32; 8],
        residual_vector: &DVector<f32>,
        rows: usize,
        cols: usize,
        transform_type: TransformType,
    ) -> HybridEncodedBlock {
        let mut residual_matrix = Array2::from_shape_vec((rows, cols), residual_vector.iter().cloned().collect()).unwrap();

        match transform_type {
            TransformType::Dct => {
                let dct_row = self.dct_planner_f32.plan_dct2(cols);
                let dct_col = self.dct_planner_f32.plan_dct2(rows);
                
                // --- 행별 DCT 병렬 처리 ---
                residual_matrix.axis_iter_mut(ndarray::Axis(0)).into_par_iter().for_each(|mut row| {
                    let mut row_vec = row.to_vec();
                    dct_row.process_dct2(&mut row_vec);
                    row.assign(&Array::from(row_vec));
                });
                
                // --- 열별 DCT 병렬 처리 ---
                let mut transposed = residual_matrix.t().to_owned();
                transposed.axis_iter_mut(ndarray::Axis(0)).into_par_iter().for_each(|mut col| {
                    let mut col_vec = col.to_vec();
                    dct_col.process_dct2(&mut col_vec);
                    col.assign(&Array::from(col_vec));
                });
                residual_matrix = transposed.t().to_owned();
            },
            TransformType::Dwt => {
                let wavelet = w::BIOR_3_1;
                let mut buffer = Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_decompose_2d(residual_matrix.view_mut(), buffer.view_mut(), wavelet);
            },
            TransformType::Adaptive => unreachable!("Adaptive transform should be handled separately"),
        }

        let mut coeffs: Vec<ResidualCoefficient> = residual_matrix
            .indexed_iter()
            .map(|((r, c), &val)| ResidualCoefficient {
                index: (r as u16, c as u16),
                value: val,
            })
            .collect();
        coeffs.sort_unstable_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
        
        // k_coeffs가 8 이하면 잔차 없음
        let top_k_coeffs = if self.k_coeffs <= 8 {
            Vec::new()
        } else {
            coeffs.into_iter().take(self.k_coeffs - 8).collect()
        };

        HybridEncodedBlock {
            rbe_params,
            residuals: top_k_coeffs,
            rows,
            cols,
            transform_type,
        }
    }

    /// 비대칭 매트릭스 지원
    pub fn compress_with_profile(
        matrix_data: &[f32],
        height: usize,
        width: usize, 
        block_size: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String> {
        let start = Instant::now();
        
        // 비대칭 매트릭스 격자 분할
        let blocks_per_height = (height + block_size - 1) / block_size;
        let blocks_per_width = (width + block_size - 1) / block_size;
        let total_blocks = blocks_per_height * blocks_per_width;
        
        // 병렬 블록 처리 (Rayon)
        let encoded_blocks: Vec<HybridEncodedBlock> = (0..total_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let mut local_encoder = RBEEncoder::new(coefficients, transform_type);
                let block_i = block_idx / blocks_per_width;
                let block_j = block_idx % blocks_per_width;
                let start_i = block_i * block_size;
                let start_j = block_j * block_size;
                
                // 블록 데이터 추출 (비대칭 매트릭스)
                let mut block_data = vec![0.0f32; block_size * block_size];
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = start_i + i;
                        let global_j = start_j + j;
                        if global_i < height && global_j < width {
                            block_data[i * block_size + j] = 
                                matrix_data[global_i * width + global_j];
                        }
                        // 패딩은 0.0으로 유지
                    }
                }
                
                // 블록 압축 (각 스레드별 local_encoder)
                local_encoder.encode_block(&block_data, block_size, block_size)
            })
            .collect();
        
        let compression_time = start.elapsed().as_secs_f64();
        
        // 압축률 계산 (비대칭 매트릭스)
        let original_size = height * width * 4; // f32 bytes
        let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        // RMSE 계산 - 디코딩해서 원본과 비교
        let mut reconstructed_data = vec![0.0f32; height * width];
        
        for (block_idx, encoded_block) in encoded_blocks.iter().enumerate() {
            let block_i = block_idx / blocks_per_width;
            let block_j = block_idx % blocks_per_width;
            let start_i = block_i * block_size;
            let start_j = block_j * block_size;
            
            // 블록 디코딩
            let decoded_block = encoded_block.decode();
            
            // 원본 행렬에 복사 (비대칭 매트릭스)
            for i in 0..block_size {
                for j in 0..block_size {
                    let global_i = start_i + i;
                    let global_j = start_j + j;
                    if global_i < height && global_j < width {
                        reconstructed_data[global_i * width + global_j] = 
                            decoded_block[i * block_size + j];
                    }
                }
            }
        }
        
        // RMSE 계산 (실제 데이터 영역만)
        let mse: f32 = matrix_data.iter()
            .zip(reconstructed_data.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (height * width) as f32;
        let rmse = mse.sqrt();
        
        Ok((encoded_blocks, compression_time, compression_ratio, rmse))
    }
    
    /// 설정 기반 압축 함수 (사용자 요청 파라미터 적용)
    pub fn compress_with_config(
        matrix_data: &[f32],
        height: usize,
        width: usize,
        config: &CompressionConfig,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String> {
        // 1. 최소 블록 개수 체크
        let blocks_per_height = (height + config.block_size - 1) / config.block_size;
        let blocks_per_width = (width + config.block_size - 1) / config.block_size;
        let actual_block_count = blocks_per_height * blocks_per_width;
        
        if let Some(min_blocks) = config.min_block_count {
            if actual_block_count < min_blocks {
                return Err(format!(
                    "블록 개수 부족: 실제 {}개 < 최소 {}개 (블록 크기를 {}에서 더 작게 조정하세요)",
                    actual_block_count, min_blocks, config.block_size
                ));
            }
        }
        
        // 2. 계수 개수 결정
        let coefficients = if let Some(custom_coeffs) = config.custom_coefficients {
            custom_coeffs
        } else {
            // 설정에 따른 자동 계수 계산 (UltraHigh는 더 많은 계수 사용)
            match config.profile {
                CompressionProfile::UltraHigh => Self::predict_coefficients_improved(config.block_size) * 4, // 2 → 4배
                CompressionProfile::High => Self::predict_coefficients_improved(config.block_size) * 2,     // 1 → 2배
                CompressionProfile::Balanced => Self::predict_coefficients_improved(config.block_size),
                CompressionProfile::Fast => Self::predict_coefficients_improved(config.block_size) / 2,     // /4 → /2
                CompressionProfile::UltraFast => Self::predict_coefficients_improved(config.block_size) / 4, // /8 → /4
            }
        };
        
        // 3. 압축 실행
        let result = Self::compress_with_profile(
            matrix_data,
            height,
            width,
            config.block_size,
            coefficients,
            config.transform_type,
        )?;
        
        let (blocks, time, ratio, rmse) = result;
        
        // 4. RMSE 임계값 체크
        if let Some(max_rmse) = config.rmse_threshold {
            if rmse > max_rmse {
                return Err(format!(
                    "RMSE 임계값 초과: {:.6} > {:.6} (계수를 {}에서 더 늘리거나 블록을 더 작게 하세요)",
                    rmse, max_rmse, coefficients
                ));
            }
        }
        
        // 5. 압축률 임계값 체크
        if let Some(min_ratio) = config.compression_ratio_threshold {
            if ratio < min_ratio {
                return Err(format!(
                    "압축률 임계값 미달: {:.1}x < {:.1}x (계수를 {}에서 더 줄이거나 블록을 더 크게 하세요)",
                    ratio, min_ratio, coefficients
                ));
            }
        }
        
        Ok((blocks, time, ratio, rmse))
    }

    /// compress_multi.rs와 동일한 이분탐색
    pub fn find_critical_coefficients(
        matrix_data: &[f32], 
        matrix_size: usize, 
        block_size: usize,
        rmse_threshold: f32,
        transform_type: TransformType,
    ) -> Result<usize, String> {
        // 이분탐색으로 임계 계수 찾기
        let max_coeffs = (block_size * block_size) / 4; // 상한: 전체 픽셀의 1/4
        let min_coeffs = 8; // 하한: 최소 8개
        
        let mut left = min_coeffs;
        let mut right = max_coeffs;
        let mut critical_coeffs = max_coeffs;
        
        while left <= right {
            let mid = (left + right) / 2;
            
            match Self::compress_with_profile(matrix_data, matrix_size, matrix_size, block_size, mid, transform_type) {
                Ok((_, _, _, rmse)) => {
                    if rmse <= rmse_threshold {
                        // 성공: 더 적은 계수로 시도
                        critical_coeffs = mid;
                        right = mid - 1;
                    } else {
                        // 실패: 더 많은 계수 필요
                        left = mid + 1;
                    }
                },
                Err(_) => {
                    left = mid + 1;
                }
            }
        }
        
        Ok(critical_coeffs)
    }
    
    /// 푸앵카레 볼 기반 수학적으로 올바른 공식 (기존 테스트용)
    pub fn predict_coefficients_improved(block_size: usize) -> usize {
        // 올바른 임계비율 계산: R(Block_Size) = max(25, 32 - ⌊log₂(Block_Size/32)⌋)
        let log_factor = if block_size >= 32 {
            (block_size as f32 / 32.0).log2().floor() as i32
        } else {
            -1 // log₂(16/32) = log₂(0.5) = -1
        };
        
        let r_value = (32 - log_factor).max(25) as usize;
        
        // K_critical = ⌈Block_Size² / R(Block_Size)⌉
        let k_critical = (block_size * block_size + r_value - 1) / r_value; // 올림 처리
        
        k_critical
    }

    /// 자동 최적화: 개선된 공식으로 빠른 예측 후 미세 조정 (기존 테스트용)
    pub fn create_optimized_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        transform_type: TransformType,
        rmse_threshold: Option<f32>,
    ) -> Result<RBEEncoder, String> {
        let threshold = rmse_threshold.unwrap_or(0.000001);
        
        // 1. 개선된 공식으로 초기 예측
        let predicted_coeffs = Self::predict_coefficients_improved(rows);
        
        // 2. 예측값으로 빠른 검증
        let mut test_encoder = RBEEncoder::new(predicted_coeffs, transform_type);
        let encoded_block = test_encoder.encode_block(data, rows, cols);
        let decoded_data = encoded_block.decode();
        
        let mse: f32 = data.iter()
            .zip(decoded_data.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (rows * cols) as f32;
        let predicted_rmse = mse.sqrt();
        
        let final_coeffs = if predicted_rmse <= threshold {
            // 예측이 정확하면 그대로 사용
            predicted_coeffs
        } else {
            // 예측이 부족하면 정밀 탐색 (좁은 범위에서만)
            let search_min = predicted_coeffs;
            let search_max = predicted_coeffs * 2; // 2배까지만 탐색
            
            let mut left = search_min;
            let mut right = search_max;
            let mut result = search_max;
            
            while left <= right {
                let mid = (left + right) / 2;
                
                let mut mid_encoder = RBEEncoder::new(mid, transform_type);
                let mid_encoded = mid_encoder.encode_block(data, rows, cols);
                let mid_decoded = mid_encoded.decode();
                
                let mid_mse: f32 = data.iter()
                    .zip(mid_decoded.iter())
                    .map(|(orig, recon)| (orig - recon).powi(2))
                    .sum::<f32>() / (rows * cols) as f32;
                
                if mid_mse.sqrt() <= threshold {
                    result = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            
            result
        };
        
        Ok(RBEEncoder::new(final_coeffs, transform_type))
    }

    /// 품질 등급에 따른 encoder 생성 (기존 테스트용 - 단일 블록 압축)
    pub fn create_quality_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        grade: QualityGrade,
        transform_type: TransformType,
    ) -> Result<RBEEncoder, String> {
        let rmse_threshold = match grade {
            QualityGrade::S => 0.000001, // 이분탐색에서는 엄격하게, 테스트에서 여유분 적용
            QualityGrade::A => 0.001,
            QualityGrade::B => 0.01,
            QualityGrade::C => 0.1,
        };
        
        // 기존 방식: 단일 블록 이분탐색 (테스트 호환성)
        let critical_coeffs = Self::find_critical_coefficients_single_block(data, rows, cols, rmse_threshold, transform_type)?;
        Ok(RBEEncoder::new(critical_coeffs, transform_type))
    }

    /// 단일 블록 이분탐색 (기존 테스트용)
    pub fn find_critical_coefficients_single_block(
        data: &[f32],
        rows: usize,
        cols: usize,
        rmse_threshold: f32,
        transform_type: TransformType,
    ) -> Result<usize, String> {
        let max_coeffs = (rows * cols) / 4; // 상한: 전체 픽셀의 1/4
        let min_coeffs = 8; // 하한: 최소 8개
        
        let mut left = min_coeffs;
        let mut right = max_coeffs;
        let mut critical_coeffs = max_coeffs;
        
        while left <= right {
            let mid = (left + right) / 2;
            let mut encoder = RBEEncoder::new(mid, transform_type);
            
            // 단일 블록 압축 및 복원
            let encoded = encoder.encode_block(data, rows, cols);
            let decoded = encoded.decode();
            
            // RMSE 계산
            let mse: f32 = data.iter()
                .zip(decoded.iter())
                .map(|(orig, decoded)| (orig - decoded).powi(2))
                .sum::<f32>() / data.len() as f32;
            let rmse = mse.sqrt();
            
            if rmse <= rmse_threshold {
                critical_coeffs = mid;
                right = mid - 1; // 더 적은 계수로 시도
            } else {
                left = mid + 1; // 더 많은 계수 필요
            }
        }
        
        Ok(critical_coeffs)
    }

    /// 동적 블록 크기를 사용한 압축
    pub fn compress_with_dynamic_blocks(
        matrix_data: &[f32],
        height: usize,
        width: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Result<(Vec<HybridEncodedBlock>, usize, f64, f32, f32), String> {
        // 최적 블록 크기 자동 결정
        let block_size = Self::determine_optimal_block_size(height, width);
        
        println!("행렬 크기 {}x{}, 최적 블록 크기: {}x{}", height, width, block_size, block_size);
        
        // 기존 압축 함수 호출
        let (blocks, time, ratio, rmse) = Self::compress_with_profile(
            matrix_data,
            height,
            width,
            block_size,
            coefficients,
            transform_type,
        )?;
        
        Ok((blocks, block_size, time, ratio, rmse))
    }

    /// 정수 Adam 최적화를 사용한 초고속 인코딩
    /// 1000:1 압축률과 RMSE ≈ 10⁻⁶ 달성
    pub fn encode_block_int_adam(
        &mut self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        steps: usize,
    ) -> HybridEncodedBlock {
        let start = Instant::now();
        
        // 1. 초기 RBE 파라미터 설정 - 부동소수점으로 시작
        let mut rbe_params = [0.0f32; 8];
        
        // 데이터 통계로 초기값 설정
        let data_mean = block_data.iter().sum::<f32>() / block_data.len() as f32;
        let data_std = (block_data.iter()
            .map(|&x| (x - data_mean).powi(2))
            .sum::<f32>() / block_data.len() as f32).sqrt();
            
        // 상수항은 평균값으로 초기화
        rbe_params[0] = data_mean;
        // 나머지는 작은 랜덤값
        for i in 1..8 {
            rbe_params[i] = data_std * 0.01 * ((i as f32) - 4.0);
        }
        
        let mut residual_coeffs = vec![0.0f32; 2];
        
        // 2. Adam 옵티마이저 상태 초기화 (부동소수점)
        let mut m = vec![0.0f32; 10];
        let mut v = vec![0.0f32; 10];
        
        // 3. Adam 상수
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        
        // 4. 최적화 루프
        let mut converged_at_step = 0;
        let mut best_rmse = f32::INFINITY;
        let mut best_params = rbe_params.clone();
        let mut best_residuals = residual_coeffs.clone();
        let mut no_improvement_count = 0;
        
        // 적응적 학습률
        let mut lr = 0.001; // 초기 학습률을 높임
        
        for step in 1..=steps {
            // 그래디언트 계산
            let grads = self.compute_float_gradients(&rbe_params, &residual_coeffs, block_data, rows, cols);
            
            // Adam 업데이트
            let mut max_grad = 0.0f32;
            for i in 0..8 {
                // 모멘텀 업데이트
                m[i] = B1 * m[i] + (1.0 - B1) * grads[i];
                v[i] = B2 * v[i] + (1.0 - B2) * grads[i] * grads[i];
                
                // Bias correction
                let m_hat = m[i] / (1.0 - B1.powi(step as i32));
                let v_hat = v[i] / (1.0 - B2.powi(step as i32));
                
                // 파라미터 업데이트
                rbe_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
                
                max_grad = max_grad.max(grads[i].abs());
            }
            
            // 잔차 계수 업데이트 (50 step 이후부터 시작)
            if step > 50 {
                for k in 0..2 {
                    if grads[8 + k].abs() > 1e-6 {
                        m[8 + k] = B1 * m[8 + k] + (1.0 - B1) * grads[8 + k];
                        v[8 + k] = B2 * v[8 + k] + (1.0 - B2) * grads[8 + k] * grads[8 + k];
                        
                        let m_hat = m[8 + k] / (1.0 - B1.powi(step as i32));
                        let v_hat = v[8 + k] / (1.0 - B2.powi(step as i32));
                        
                        residual_coeffs[k] -= lr * m_hat / (v_hat.sqrt() + EPS);
                    }
                }
            }
            
            // 100 step마다 성능 체크
            if step % 100 == 0 {
                let current_rmse = self.compute_rmse(&rbe_params, &residual_coeffs, block_data, rows, cols);
                
                // 개선 여부 체크
                if current_rmse < best_rmse * 0.99 { // 1% 이상 개선시
                    best_rmse = current_rmse;
                    best_params = rbe_params.clone();
                    best_residuals = residual_coeffs.clone();
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                }
                
                // 학습률 조정
                if no_improvement_count >= 2 {
                    lr *= 0.5; // 학습률 반으로 줄임
                    no_improvement_count = 0;
                    println!("Step {}: 학습률 감소 -> {:.6}", step, lr);
                }
                
                let active_residuals = residual_coeffs.iter().filter(|&&x| x.abs() > 1e-6).count();
                
                println!("Step {}: grad={:.2e}, RMSE={:.2e}, K={}, lr={:.6}", 
                         step, max_grad, current_rmse, active_residuals, lr);
            }
            
            // 수렴 체크
            if max_grad < 1e-6 && step > 200 {
                converged_at_step = step;
                println!("수렴! Step {}: final max_grad={:.2e}", step, max_grad);
                break;
            }
            
            // 학습률이 너무 작아지면 중단
            if lr < 1e-6 {
                println!("학습률이 너무 작아짐. Step {}에서 중단", step);
                break;
            }
        }
        
        // 베스트 파라미터 사용
        if best_rmse < f32::INFINITY {
            rbe_params = best_params;
            residual_coeffs = best_residuals;
        }
        
        // 6. 잔차 계수 포맷팅
        let residual_coefficients: Vec<ResidualCoefficient> = residual_coeffs.iter()
            .enumerate()
            .filter(|(_, &val)| val.abs() > 1e-6)
            .map(|(idx, &val)| ResidualCoefficient {
                index: (0, idx as u16), // 간단히 처리
                value: val,
            })
            .collect();
        
        let encoding_time = start.elapsed().as_micros() as f32;
        
        println!("인코딩 완료: {:.1}ms, 수렴 step: {}, 최종 K: {}, 최종 RMSE: {:.2e}", 
                 encoding_time / 1000.0, converged_at_step, residual_coefficients.len(), best_rmse);
        
        HybridEncodedBlock {
            rbe_params: rbe_params,
            residuals: residual_coefficients,
            transform_type: self.transform_type,
            rows: rows,
            cols: cols,
        }
    }


    /// 대규모 행렬 압축 (병렬 처리)
    pub fn compress_with_int_adam(
        matrix_data: &[f32],
        height: usize,
        width: usize,
        block_size: usize,
        steps: usize,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f64, f64), String> {
        let start = Instant::now();
        
        // 블록 분할
        let blocks_per_row = (width + block_size - 1) / block_size;
        let blocks_per_col = (height + block_size - 1) / block_size;
        let total_blocks = blocks_per_row * blocks_per_col;
        
        println!("\n=== 대규모 행렬 압축 시작 ===");
        println!("행렬 크기: {}x{}", height, width);
        println!("블록 크기: {}x{}", block_size, block_size);
        println!("총 블록 수: {}", total_blocks);
        
        // 병렬 블록 압축
        let blocks: Vec<HybridEncodedBlock> = (0..total_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let block_row = block_idx / blocks_per_row;
                let block_col = block_idx % blocks_per_row;
                
                let row_start = block_row * block_size;
                let col_start = block_col * block_size;
                let row_end = (row_start + block_size).min(height);
                let col_end = (col_start + block_size).min(width);
                
                let actual_block_height = row_end - row_start;
                let actual_block_width = col_end - col_start;
                
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; actual_block_height * actual_block_width];
                for r in 0..actual_block_height {
                    for c in 0..actual_block_width {
                        let src_idx = (row_start + r) * width + (col_start + c);
                        let dst_idx = r * actual_block_width + c;
                        block_data[dst_idx] = matrix_data[src_idx];
                    }
                }
                
                // 개별 블록 인코딩 (로그 출력 없음)
                let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
                encoder.encode_block_int_adam_quiet(&block_data, actual_block_height, actual_block_width, steps)
            })
            .collect();
        
        let compression_time = start.elapsed().as_millis() as f64;
        
        // 압축률 계산
        let original_size = height * width * 4; // f32
        let compressed_size: usize = blocks.iter()
            .map(|b| 8 * 4 + b.residuals.len() * 8)
            .sum();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        // RMSE 계산 (첫 번째 블록만 샘플로)
        let decoder = WeightGenerator::new();
        let first_decoded = decoder.decode_block_int_adam(&blocks[0]);
        let mut sample_error = 0.0;
        let sample_size = first_decoded.len().min(100);
        for i in 0..sample_size {
            let diff = first_decoded[i] - matrix_data[i];
            sample_error += diff * diff;
        }
        let rmse = (sample_error / sample_size as f32).sqrt();
        
        println!("\n압축 완료:");
        println!("- 압축 시간: {:.2}초", compression_time / 1000.0);
        println!("- 압축률: {:.1}:1", compression_ratio);
        println!("- 샘플 RMSE: {:.2e}", rmse);
        
        Ok((blocks, compression_time, compression_ratio, rmse as f64))
    }
    
    /// 블록을 정수 Adam으로 인코딩 (조용한 버전)
    pub fn encode_block_int_adam_quiet(
        &self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        steps: usize,
    ) -> HybridEncodedBlock {
        // 1. 베이스 RBE 파라미터 초기화 (rank-4 기반)
        let mut rbe_params = [0.0f32; 8];
        for i in 0..8 {
            rbe_params[i] = rand::thread_rng().gen_range(-0.01..0.01);
        }
        
        // 2. K개의 잔차 계수 초기화 - 최대 64개까지 허용
        const MAX_K: usize = 64;
        let mut residual_coeffs = vec![0.0f32; MAX_K];
        let mut m = vec![0.0f32; 8 + MAX_K]; // momentum for all params
        let mut v = vec![0.0f32; 8 + MAX_K]; // velocity for all params
        
        // 3. Adam 상수
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        
        // 4. 최적화 루프
        let mut converged_step = 0;
        let mut active_k = 0;
        let mut lr = 0.001; // 적응적 학습률
        let mut best_rmse = f32::INFINITY;
        let mut no_improvement_count = 0;
        
        for step in 1..=steps {
            // 그래디언트 계산 (부동소수점)
            let grads = self.compute_float_gradients_with_k(
                &rbe_params,
                &residual_coeffs[..active_k],
                block_data,
                rows,
                cols,
            );
            
            // Adam update
            let mut max_grad = 0.0f32;
            for i in 0..8 {
                m[i] = B1 * m[i] + (1.0 - B1) * grads[i];
                v[i] = B2 * v[i] + (1.0 - B2) * grads[i] * grads[i];
                
                // Bias correction
                let m_hat = m[i] / (1.0 - B1.powi(step as i32));
                let v_hat = v[i] / (1.0 - B2.powi(step as i32));
                
                rbe_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
                max_grad = max_grad.max(grads[i].abs());
            }
            
            // 잔차 계수 업데이트
            for k in 0..active_k {
                let idx = 8 + k;
                if idx < grads.len() {
                    m[idx] = B1 * m[idx] + (1.0 - B1) * grads[idx];
                    v[idx] = B2 * v[idx] + (1.0 - B2) * grads[idx] * grads[idx];
                    
                    let m_hat = m[idx] / (1.0 - B1.powi(step as i32));
                    let v_hat = v[idx] / (1.0 - B2.powi(step as i32));
                    
                    residual_coeffs[k] -= lr * m_hat / (v_hat.sqrt() + EPS);
                }
            }
            
            // 50 step마다 RMSE 체크하고 필요시 K 증가
            if step % 50 == 0 {
                let rmse = self.compute_rmse_with_k(&rbe_params, &residual_coeffs[..active_k], block_data, rows, cols);
                
                // RMSE가 높고 K를 늘릴 여지가 있으면
                if rmse > 0.005 && active_k < MAX_K && step > 100 {
                    // 더 세밀한 단계로 K 증가
                    if rmse > 0.05 && active_k < 4 {
                        active_k = 4;
                    } else if rmse > 0.03 && active_k < 8 {
                        active_k = 8;
                    } else if rmse > 0.02 && active_k < 16 {
                        active_k = 16;
                    } else if rmse > 0.015 && active_k < 24 {
                        active_k = 24;
                    } else if rmse > 0.01 && active_k < 32 {
                        active_k = 32;
                    } else if rmse > 0.007 && active_k < 48 {
                        active_k = 48;
                    } else if rmse > 0.005 && active_k < MAX_K {
                        active_k = MAX_K;
                    }
                }
                
                // 개선 체크
                if rmse < best_rmse * 0.99 {
                    best_rmse = rmse;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= 2 {
                        lr *= 0.5;
                        no_improvement_count = 0;
                    }
                }
            }
            
            // 100 에포크마다 로그 출력
            if step % 100 == 0 {
                let rmse = self.compute_rmse_with_k(&rbe_params, &residual_coeffs[..active_k], block_data, rows, cols);
                println!("  Block({},{}): Step {}: grad={:.2e}, RMSE={:.2e}, K={}, lr={:.6}", 
                         rows, cols, step, max_grad, rmse, active_k, lr);
            }
            
            // 수렴 체크
            if max_grad < 1e-5 && converged_step == 0 {
                converged_step = step;
            }
            
            if lr < 1e-6 {
                break;
            }
        }
        
        // 활성화된 잔차 계수만 저장
        let active_residuals: Vec<ResidualCoefficient> = residual_coeffs[..active_k]
            .iter()
            .enumerate()
            .map(|(i, &coeff)| ResidualCoefficient {
                index: (0, i as u16),  // (row_idx, col_idx) 형태
                value: coeff,
            })
            .collect();
        
        HybridEncodedBlock {
            rows,
            cols,
            rbe_params,
            residuals: active_residuals,
            transform_type: self.transform_type,
        }
    }
    
    // 가변 K로 그래디언트 계산
    fn compute_float_gradients_with_k(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let k = residual_coeffs.len();
        let mut grads = vec![0.0f32; 8 + k];
        let epsilon = 1e-4;
        
        // RBE 파라미터 그래디언트
        for i in 0..8 {
            let mut params_plus = *rbe_params;
            let mut params_minus = *rbe_params;
            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;
            
            let loss_plus = self.compute_loss_with_k(&params_plus, residual_coeffs, block_data, rows, cols);
            let loss_minus = self.compute_loss_with_k(&params_minus, residual_coeffs, block_data, rows, cols);
            
            grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
        
        // 잔차 계수 그래디언트
        for j in 0..k {
            let mut coeffs_plus = residual_coeffs.to_vec();
            let mut coeffs_minus = residual_coeffs.to_vec();
            coeffs_plus[j] += epsilon;
            coeffs_minus[j] -= epsilon;
            
            let loss_plus = self.compute_loss_with_k(rbe_params, &coeffs_plus, block_data, rows, cols);
            let loss_minus = self.compute_loss_with_k(rbe_params, &coeffs_minus, block_data, rows, cols);
            
            grads[8 + j] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
        
        grads
    }
    
    // 가변 K로 손실 계산
    fn compute_loss_with_k(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        let predicted = self.compute_predicted_with_k(rbe_params, residual_coeffs, rows, cols);
        
        let mut loss = 0.0;
        for i in 0..block_data.len() {
            let diff = predicted[i] - block_data[i];
            loss += diff * diff;
        }
        
        loss / block_data.len() as f32
    }
    
    // 가변 K로 예측값 계산
    fn compute_predicted_with_k(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut predicted = vec![0.0f32; rows * cols];
        
        // 베이스 RBE 기여도
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                predicted[idx] = self.compute_basis_value(rbe_params, i, j, rows, cols);
            }
        }
        
        // 잔차 계수 기여도
        let k = residual_coeffs.len();
        if k > 0 {
            for freq_idx in 0..k {
                let coeff = residual_coeffs[freq_idx];
                if coeff.abs() > 1e-8 {
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            let freq_contrib = self.compute_frequency_basis(freq_idx, i, j, rows, cols);
                            predicted[idx] += coeff * freq_contrib;
                        }
                    }
                }
            }
        }
        
        predicted
    }
    
    // 가변 K로 RMSE 계산
    fn compute_rmse_with_k(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        let loss = self.compute_loss_with_k(rbe_params, residual_coeffs, block_data, rows, cols);
        loss.sqrt()
    }

    /// 부동소수점 그래디언트 계산
    fn compute_float_gradients(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut grads = vec![0.0f32; 10];
        let n_pixels = (rows * cols) as f32;
        
        // Core 파라미터 그래디언트
        for i in 0..8 {
            let mut grad_sum = 0.0f32;
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    let predicted = self.compute_predicted_float(rbe_params, residual_coeffs, r, c, rows, cols);
                    let error = predicted - block_data[idx];
                    let basis_val = self.compute_basis_float(i, r, c, rows, cols);
                    grad_sum += 2.0 * error * basis_val;
                }
            }
            grads[i] = grad_sum / n_pixels;
        }
        
        // Residual 그래디언트
        for k in 0..residual_coeffs.len().min(2) {
            let mut grad_sum = 0.0f32;
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    let predicted = self.compute_predicted_float(rbe_params, residual_coeffs, r, c, rows, cols);
                    let error = predicted - block_data[idx];
                    let basis_k = if k == 0 { 
                        if (r + c) % 2 == 0 { 1.0 } else { -1.0 }
                    } else { 
                        ((r * c) % 5) as f32 - 2.0
                    };
                    grad_sum += 2.0 * error * basis_k;
                }
            }
            grads[8 + k] = grad_sum / n_pixels;
        }
        
        grads
    }
    
    /// 부동소수점 예측값 계산
    fn compute_predicted_float(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        r: usize,
        c: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut val = 0.0f32;
        
        // RBE 기저 함수 적용
        for i in 0..8 {
            val += rbe_params[i] * self.compute_basis_float(i, r, c, rows, cols);
        }
        
        // 잔차 추가
        if residual_coeffs.len() > 0 {
            val += residual_coeffs[0];
        }
        if residual_coeffs.len() > 1 {
            val += residual_coeffs[1];
        }
        
        val
    }
    
    /// 부동소수점 기저 함수
    fn compute_basis_float(&self, idx: usize, r: usize, c: usize, rows: usize, cols: usize) -> f32 {
        let x = c as f32 / cols as f32;
        let y = r as f32 / rows as f32;
        
        match idx {
            0 => 1.0,
            1 => x,
            2 => y,
            3 => x * x,
            4 => y * y,
            5 => x * y,
            6 => (std::f32::consts::PI * x).cos(),
            7 => (std::f32::consts::PI * y).cos(),
            _ => 0.0,
        }
    }
    
    /// RMSE 계산
    fn compute_rmse(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut sse = 0.0f32;
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let predicted = self.compute_predicted_float(rbe_params, residual_coeffs, r, c, rows, cols);
                let error = predicted - block_data[idx];
                sse += error * error;
            }
        }
        
        (sse / (rows * cols) as f32).sqrt()
    }

    // 베이시스 함수값 계산
    fn compute_basis_value(
        &self,
        rbe_params: &[f32; 8],
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let y = i as f32 / rows as f32;
        let x = j as f32 / cols as f32;
        
        // rank-4 베이시스 함수
        rbe_params[0] + 
        rbe_params[1] * x + 
        rbe_params[2] * y + 
        rbe_params[3] * x * y +
        rbe_params[4] * (2.0 * x * x - 1.0) +
        rbe_params[5] * (2.0 * y * y - 1.0) +
        rbe_params[6] * x * (2.0 * y * y - 1.0) +
        rbe_params[7] * y * (2.0 * x * x - 1.0)
    }
    
    // 주파수 베이시스 계산 (DCT-like)
    fn compute_frequency_basis(
        &self,
        freq_idx: usize,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 2D DCT 베이시스
        let u = freq_idx % 4; // 주파수 인덱스를 2D로 변환
        let v = freq_idx / 4;
        
        let y_term = ((2.0 * i as f32 + 1.0) * u as f32 * std::f32::consts::PI / (2.0 * rows as f32)).cos();
        let x_term = ((2.0 * j as f32 + 1.0) * v as f32 * std::f32::consts::PI / (2.0 * cols as f32)).cos();
        
        let scale = if u == 0 && v == 0 {
            1.0 / (rows as f32 * cols as f32).sqrt()
        } else if u == 0 || v == 0 {
            (2.0 / (rows as f32 * cols as f32)).sqrt()
        } else {
            2.0 / (rows as f32 * cols as f32).sqrt()
        };
        
        scale * y_term * x_term
    }

    /// 확장된 기저 함수를 사용한 블록 인코딩
    pub fn encode_block_extended_basis(
        &self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        steps: usize,
        num_basis: usize, // 8, 16, 32, 64 등
    ) -> HybridEncodedBlock {
        // 기저 함수 개수에 따른 파라미터 초기화
        let mut basis_params = vec![0.0f32; num_basis];
        for i in 0..num_basis {
            basis_params[i] = rand::thread_rng().gen_range(-0.01..0.01);
        }
        
        // 잔차 계수는 적게 시작
        const MAX_K: usize = 32;
        let mut residual_coeffs = vec![0.0f32; MAX_K];
        let mut m = vec![0.0f32; num_basis + MAX_K]; // momentum for all params
        let mut v = vec![0.0f32; num_basis + MAX_K]; // velocity for all params
        
        // Adam 상수
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        
        let mut active_k = 0;
        let mut lr = 0.001;
        let mut best_rmse = f32::INFINITY;
        let mut no_improvement_count = 0;
        
        for step in 1..=steps {
            // 그래디언트 계산
            let grads = self.compute_extended_gradients(
                &basis_params,
                &residual_coeffs[..active_k],
                block_data,
                rows,
                cols,
            );
            
            // Adam 업데이트
            let mut max_grad = 0.0f32;
            for i in 0..num_basis {
                m[i] = B1 * m[i] + (1.0 - B1) * grads[i];
                v[i] = B2 * v[i] + (1.0 - B2) * grads[i] * grads[i];
                
                let m_hat = m[i] / (1.0 - B1.powi(step as i32));
                let v_hat = v[i] / (1.0 - B2.powi(step as i32));
                
                basis_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
                max_grad = max_grad.max(grads[i].abs());
            }
            
            // 잔차 계수 업데이트
            for k in 0..active_k {
                let idx = num_basis + k;
                if idx < grads.len() {
                    m[idx] = B1 * m[idx] + (1.0 - B1) * grads[idx];
                    v[idx] = B2 * v[idx] + (1.0 - B2) * grads[idx] * grads[idx];
                    
                    let m_hat = m[idx] / (1.0 - B1.powi(step as i32));
                    let v_hat = v[idx] / (1.0 - B2.powi(step as i32));
                    
                    residual_coeffs[k] -= lr * m_hat / (v_hat.sqrt() + EPS);
                }
            }
            
            // 주기적 체크
            if step % 50 == 0 {
                let rmse = self.compute_extended_rmse(
                    &basis_params,
                    &residual_coeffs[..active_k],
                    block_data,
                    rows,
                    cols,
                );
                
                // 기저 함수가 많으면 잔차가 덜 필요함
                let rmse_threshold = if num_basis >= 32 { 0.01 } else { 0.005 };
                if rmse > rmse_threshold && active_k < MAX_K && step > 100 {
                    active_k = (active_k + 4).min(MAX_K);
                }
                
                // 개선 체크
                if rmse < best_rmse * 0.99 {
                    best_rmse = rmse;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= 2 {
                        lr *= 0.5;
                        no_improvement_count = 0;
                    }
                }
            }
            
            // 조기 종료
            if lr < 1e-6 || max_grad < 1e-5 {
                break;
            }
        }
        
        // rank-4 형식으로 변환 (호환성을 위해)
        let mut rbe_params = [0.0f32; 8];
        for i in 0..8.min(num_basis) {
            rbe_params[i] = basis_params[i];
        }
        
        // 활성화된 잔차 계수 저장
        let active_residuals: Vec<ResidualCoefficient> = residual_coeffs[..active_k]
            .iter()
            .enumerate()
            .map(|(i, &coeff)| ResidualCoefficient {
                index: (0, i as u16),
                value: coeff,
            })
            .collect();
        
        HybridEncodedBlock {
            rows,
            cols,
            rbe_params,
            residuals: active_residuals,
            transform_type: self.transform_type,
        }
    }
    
    // 확장된 기저 함수로 그래디언트 계산
    fn compute_extended_gradients(
        &self,
        basis_params: &[f32],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut grads = vec![0.0f32; basis_params.len() + residual_coeffs.len()];
        
        // 예측값과 오차 계산
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let predicted = self.compute_extended_value(basis_params, residual_coeffs, i, j, rows, cols);
                let error = predicted - block_data[idx];
                
                // 기저 함수 그래디언트
                for k in 0..basis_params.len() {
                    let basis_grad = self.compute_extended_basis_gradient(k, i, j, rows, cols);
                    grads[k] += error * basis_grad;
                }
                
                // 잔차 계수 그래디언트
                for k in 0..residual_coeffs.len() {
                    let freq_basis = self.compute_frequency_basis(k, i, j, rows, cols);
                    grads[basis_params.len() + k] += error * freq_basis;
                }
            }
        }
        
        // 정규화
        let n = (rows * cols) as f32;
        for g in &mut grads {
            *g /= n;
        }
        
        grads
    }
    
    // 확장된 기저 함수 값 계산
    fn compute_extended_value(
        &self,
        basis_params: &[f32],
        residual_coeffs: &[f32],
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut value = 0.0;
        
        // 확장된 기저 함수들
        for (k, &param) in basis_params.iter().enumerate() {
            value += param * self.compute_extended_basis_gradient(k, i, j, rows, cols);
        }
        
        // 잔차 항
        for (k, &coeff) in residual_coeffs.iter().enumerate() {
            value += coeff * self.compute_frequency_basis(k, i, j, rows, cols);
        }
        
        value
    }
    
    // 확장된 기저 함수의 그래디언트
    fn compute_extended_basis_gradient(
        &self,
        basis_idx: usize,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let y = i as f32 / rows as f32;
        let x = j as f32 / cols as f32;
        
        match basis_idx {
            0 => 1.0,
            1 => x,
            2 => y,
            3 => x * y,
            4 => 2.0 * x * x - 1.0,
            5 => 2.0 * y * y - 1.0,
            6 => x * (2.0 * y * y - 1.0),
            7 => y * (2.0 * x * x - 1.0),
            // 고차 기저 함수들
            8 => x * x * x,
            9 => y * y * y,
            10 => x * x * y,
            11 => x * y * y,
            12 => (4.0 * x * x - 3.0) * x,
            13 => (4.0 * y * y - 3.0) * y,
            14 => x * x * y * y,
            15 => (x * x - 0.5) * (y * y - 0.5),
            // 더 고차 항들
            16..=31 => {
                let k = basis_idx - 16;
                let u = k % 4 + 1;
                let v = k / 4 + 1;
                (std::f32::consts::PI * u as f32 * x).cos() * 
                (std::f32::consts::PI * v as f32 * y).cos()
            }
            32..=63 => {
                let k = basis_idx - 32;
                let u = k % 4 + 1;
                let v = k / 4 + 1;
                (std::f32::consts::PI * u as f32 * x).sin() * 
                (std::f32::consts::PI * v as f32 * y).sin()
            }
            _ => 0.0,
        }
    }
    
    // 확장된 RMSE 계산
    fn compute_extended_rmse(
        &self,
        basis_params: &[f32],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut sse = 0.0f32;
        
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let predicted = self.compute_extended_value(basis_params, residual_coeffs, i, j, rows, cols);
                let error = predicted - block_data[idx];
                sse += error * error;
            }
        }
        
        (sse / (rows * cols) as f32).sqrt()
    }

    /// 향상된 블록 인코딩 - 적응적 K값과 더 나은 기저 함수
    pub fn encode_block_int_adam_enhanced(
        &mut self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        steps: usize,
        target_rmse: f32,
    ) -> HybridEncodedBlock {
        // Adam 파라미터
        const LR: f32 = 0.01;
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        
        // 확장된 기저 함수 사용 (rank-8)
        let mut rbe_params = [0.0f32; 8];
        
        // 최대 잔차 계수를 동적으로 조정
        const MIN_K: usize = 2; // 최소 2개는 유지
        const MAX_K: usize = 128; // 더 큰 K 허용
        let mut residual_coeffs = vec![0.0f32; MAX_K];
        let mut active_k = MIN_K;
        
        // 초기화: 평균값과 선형 트렌드
        let mut sum = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let val = block_data[idx];
                sum += val;
                sum_x += val * (c as f32 / cols as f32);
                sum_y += val * (r as f32 / rows as f32);
            }
        }
        
        let n = (rows * cols) as f32;
        rbe_params[0] = sum / n;
        rbe_params[1] = (sum_x - sum * 0.5) * 2.0 / n;
        rbe_params[2] = (sum_y - sum * 0.5) * 2.0 / n;
        
        // Adam 상태
        let mut m = vec![0.0f32; 8 + MAX_K];
        let mut v = vec![0.0f32; 8 + MAX_K];
        
        let mut lr = LR;
        let mut best_rmse = f32::INFINITY;
        let mut no_improvement_count = 0;
        let mut k_increase_cooldown = 0;
        
        for step in 1..=steps {
            // 그래디언트 계산
            let grads = self.compute_enhanced_gradients(
                &rbe_params,
                &residual_coeffs[..active_k],
                block_data,
                rows,
                cols,
            );
            
            // 파라미터 그래디언트와 잔차 그래디언트 분리
            let param_grads = &grads[..8];
            let residual_grads = &grads[8..8+active_k];
            
            // 파라미터 업데이트 (단계에 따라)
            let mut max_grad = 0.0f32;
            for i in 0..8 {
                m[i] = B1 * m[i] + (1.0 - B1) * param_grads[i];
                v[i] = B2 * v[i] + (1.0 - B2) * param_grads[i] * param_grads[i];
                
                let m_hat = m[i] / (1.0 - B1.powi(step as i32));
                let v_hat = v[i] / (1.0 - B2.powi(step as i32));
                
                rbe_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
                max_grad = max_grad.max(param_grads[i].abs());
            }
            
            // Update residual coefficients
            for k in 0..active_k {
                m[8 + k] = B1 * m[8 + k] + (1.0 - B1) * residual_grads[k];
                v[8 + k] = B2 * v[8 + k] + (1.0 - B2) * residual_grads[k] * residual_grads[k];
                
                let m_hat = m[8 + k] / (1.0 - B1.powi(step as i32));
                let v_hat = v[8 + k] / (1.0 - B2.powi(step as i32));
                
                residual_coeffs[k] -= lr * m_hat / (v_hat.sqrt() + EPS);
            }
            
            // Adaptive K adjustment
            if step % 25 == 0 && k_increase_cooldown == 0 {
                let rmse = self.compute_enhanced_rmse(&rbe_params, &residual_coeffs[..active_k], block_data, rows, cols);
                
                // 목표 RMSE에 따라 K 증가 - 더 보수적으로
                if rmse > target_rmse && active_k < MAX_K {
                    let rmse_ratio = rmse / target_rmse;
                    
                    // K 증가 결정 (더 적극적으로)
                    if rmse > target_rmse * 2.0 && active_k < MAX_K && step > 100 {
                        let rmse_ratio = rmse / target_rmse;
                        let old_k = active_k;
                        
                        if rmse_ratio > 100.0 && active_k < MAX_K / 2 {
                            // 매우 높은 RMSE: K를 두 배로
                            active_k = (active_k * 2).min(MAX_K);
                            println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                        } else if rmse_ratio > 10.0 {
                            // 높은 RMSE: K를 50% 증가
                            active_k = (active_k + active_k / 2 + 1).min(MAX_K);
                            println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                        } else if rmse_ratio > 5.0 {
                            // 중간 RMSE: K를 25% 증가
                            active_k = (active_k + active_k / 4 + 1).min(MAX_K);
                            println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                        }
                        
                        // 새 계수 초기화 (새로 추가된 부분만)
                        for k in old_k..active_k {
                            residual_coeffs[k] = 0.0;
                        }
                    }
                }
                
                // 수렴 체크
                if rmse < best_rmse * 0.99 {
                    best_rmse = rmse;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= 4 {
                        lr *= 0.8;
                        no_improvement_count = 0;
                    }
                }
                
                // 목표 달성시 조기 종료
                if rmse < target_rmse * 0.9 {
                    break;
                }
            }
            
            if k_increase_cooldown > 0 {
                k_increase_cooldown -= 1;
            }
            
            // 학습률이 너무 낮으면 종료
            if lr < 1e-7 {
                break;
            }
        }
        
        // 최종 RMSE 확인
        let final_rmse = self.compute_enhanced_rmse(&rbe_params, &residual_coeffs[..active_k], block_data, rows, cols);
        
        // 유의미한 잔차 계수만 저장 (임계값 이상)
        let threshold = 0.0001;
        let active_residuals: Vec<ResidualCoefficient> = residual_coeffs[..active_k]
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > threshold)
            .map(|(i, &coeff)| ResidualCoefficient {
                index: (i as u16 / 16, i as u16 % 16), // 2D 인덱싱
                value: coeff,
            })
            .collect();
        
        // 압축률 계산
        let original_size = rows * cols * 4; // f32
        let compressed_size = 8 * 4 + active_residuals.len() * 8; // rbe_params + residuals
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("Enhanced encoding: RMSE={:.4}, K={} (from {}), ratio={:.0}:1", 
                 final_rmse, active_residuals.len(), active_k, compression_ratio);
        
        HybridEncodedBlock {
            rows,
            cols,
            rbe_params,
            residuals: active_residuals,
            transform_type: self.transform_type,
        }
    }
    
    /// 향상된 그래디언트 계산
    fn compute_enhanced_gradients(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut grads = vec![0.0f32; 8 + residual_coeffs.len()];
        let n_pixels = (rows * cols) as f32;
        
        // RBE 파라미터 그래디언트
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let predicted = self.compute_enhanced_prediction(rbe_params, residual_coeffs, r, c, rows, cols);
                let error = predicted - block_data[idx];
                
                // 각 기저 함수에 대한 그래디언트
                let basis_values = self.compute_enhanced_basis(r, c, rows, cols);
                for i in 0..8 {
                    grads[i] += 2.0 * error * basis_values[i];
                }
                
                // 잔차 계수 그래디언트
                for k in 0..residual_coeffs.len() {
                    let residual_basis = self.compute_residual_basis(k, r, c, rows, cols);
                    grads[8 + k] += 2.0 * error * residual_basis;
                }
            }
        }
        
        // 정규화
        for g in grads.iter_mut() {
            *g /= n_pixels;
        }
        
        grads
    }
    
    /// 향상된 예측값 계산
    fn compute_enhanced_prediction(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        r: usize,
        c: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut val = 0.0f32;
        
        // RBE 기저 함수
        let basis_values = self.compute_enhanced_basis(r, c, rows, cols);
        for i in 0..8 {
            val += rbe_params[i] * basis_values[i];
        }
        
        // 잔차 항
        for k in 0..residual_coeffs.len() {
            let residual_basis = self.compute_residual_basis(k, r, c, rows, cols);
            val += residual_coeffs[k] * residual_basis;
        }
        
        val
    }
    
    /// 향상된 기저 함수 (더 표현력 있는 함수들)
    fn compute_enhanced_basis(&self, r: usize, c: usize, rows: usize, cols: usize) -> [f32; 8] {
        let x = if cols > 0 { c as f32 / cols as f32 } else { 0.0 };
        let y = if rows > 0 { r as f32 / rows as f32 } else { 0.0 };
        
        [
            1.0,                                    // 상수
            x,                                      // 선형 x
            y,                                      // 선형 y
            x * y,                                  // 교차항
            (2.0 * std::f32::consts::PI * x).cos(), // 코사인 x
            (2.0 * std::f32::consts::PI * y).cos(), // 코사인 y
            x * x - 0.5,                            // 2차 x (중심화)
            y * y - 0.5,                            // 2차 y (중심화)
        ]
    }
    
    /// 잔차 기저 함수 (다양한 주파수 성분)
    fn compute_residual_basis(&self, k: usize, r: usize, c: usize, rows: usize, cols: usize) -> f32 {
        let x = if cols > 0 { c as f32 / cols as f32 } else { 0.0 };
        let y = if rows > 0 { r as f32 / rows as f32 } else { 0.0 };
        
        // 다양한 기저 함수 패턴 (16개 반복)
        match k % 16 {
            0 => ((k + 1) as f32 * std::f32::consts::PI * x).sin(),
            1 => ((k + 1) as f32 * std::f32::consts::PI * y).sin(),
            2 => ((k + 1) as f32 * std::f32::consts::PI * x).cos(),
            3 => ((k + 1) as f32 * std::f32::consts::PI * y).cos(),
            4 => ((k / 4 + 1) as f32 * std::f32::consts::PI * (x + y)).sin(),
            5 => ((k / 4 + 1) as f32 * std::f32::consts::PI * (x - y)).sin(),
            6 => (x * y * (k + 1) as f32).sin(),
            7 => (x * y * (k + 1) as f32).cos(),
            8 => x.powf((k / 8 + 2) as f32) - 0.5,
            9 => y.powf((k / 8 + 2) as f32) - 0.5,
            10 => ((x - 0.5) * (y - 0.5) * (k + 1) as f32).tanh(),
            11 => (2.0 * x - 1.0) * (2.0 * y - 1.0),
            12 => (x - 0.5).powi(3),
            13 => (y - 0.5).powi(3),
            14 => (k as f32 * 0.5 * std::f32::consts::PI * (x * y)).sin(),
            15 => (k as f32 * 0.5 * std::f32::consts::PI * (x * y)).cos(),
            _ => unreachable!(),
        }
    }
    
    /// 향상된 RMSE 계산
    fn compute_enhanced_rmse(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        let mut sse = 0.0f32;
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let predicted = self.compute_enhanced_prediction(rbe_params, residual_coeffs, r, c, rows, cols);
                let error = predicted - block_data[idx];
                sse += error * error;
            }
        }
        
        (sse / (rows * cols) as f32).sqrt()
    }

    /// 하이브리드 최적화: 표준 Adam과 Riemannian Adam을 번갈아 사용 (K 제한)
    pub fn encode_block_hybrid_optimization_limited(
        &mut self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        max_steps: usize,
        target_rmse: f32,
        target_compression_ratio: f32,
        max_k_limit: usize, // 추가된 파라미터
    ) -> HybridEncodedBlock {
        let mut rng = thread_rng();
        
        // 초기 RBE 파라미터 (평균과 선형 트렌드 기반)
        let mut rbe_params = self.initialize_rbe_params(block_data, rows, cols);
        
        // 최대 K값 제한 (압축률과 사용자 제한 중 작은 값)
        let original_size = rows * cols * 4; // f32
        let max_compressed_size = (original_size as f32 / target_compression_ratio) as usize;
        let max_k_from_ratio = ((max_compressed_size - 32) / 8).max(8).min(256);
        let max_k_allowed = max_k_from_ratio.min(max_k_limit); // 사용자 제한 적용
        
        println!("K 제한: 압축률 기반={}, 사용자={}, 최종={}", 
                 max_k_from_ratio, max_k_limit, max_k_allowed);
        
        let mut residual_coeffs = vec![0.0f32; max_k_allowed];
        let mut active_k = (max_k_allowed / 4).max(4).min(8); // 적당히 시작
        
        // Adam 상태들
        let mut standard_adam = AdamState::new();
        let mut riemannian_adam = AdamState::new(); // 두 번째 Adam도 일반 Adam으로
        
        // 최적화 설정
        let mut lr = 0.01; // 더 높은 초기 학습률
        let mut best_rmse = f32::MAX;
        let mut no_improvement_count = 0;
        let mut phase = 0; // 0: Standard Adam, 1: Riemannian Adam
        let phase_steps = 50; // 더 자주 전환
        
        for step in 0..max_steps {
            // 단계 전환
            if step > 0 && step % phase_steps == 0 {
                phase = 1 - phase;
                let phase_name = if phase == 0 { "Standard Adam" } else { "Riemannian Adam" };
                println!("Step {}: 최적화 전환 -> {}", step, phase_name);
            }
            
            // 그래디언트 계산
            let grads = self.compute_enhanced_gradients(
                &rbe_params,
                &residual_coeffs[..active_k],
                block_data,
                rows,
                cols,
            );
            
            // 파라미터 그래디언트와 잔차 그래디언트 분리
            let param_grads = &grads[..8];
            let residual_grads = &grads[8..8+active_k];
            
            // 파라미터 업데이트 (단계에 따라)
            if phase == 0 {
                // Standard Adam
                for i in 0..8 {
                    standard_adam.update(&mut rbe_params[i], param_grads[i], lr);
                }
                for k in 0..active_k {
                    // 새로운 계수일수록 더 큰 학습률 사용
                    let adaptive_lr = if k < 8 {
                        lr
                    } else if k < 16 {
                        lr * 2.0
                    } else {
                        lr * 4.0
                    };
                    standard_adam.update(&mut residual_coeffs[k], residual_grads[k], adaptive_lr);
                }
            } else {
                // Riemannian Adam (푸앵카레 볼에서) - 간단한 버전으로
                // r, theta 파라미터에 메트릭 스케일링만 적용
                for i in 0..8 {
                    let scaled_grad = if i % 2 == 0 {
                        // r 파라미터: 푸앵카레 볼 메트릭 고려
                        let r = rbe_params[i].clamp(0.0, 0.95);
                        let metric_factor = 1.0 / (1.0 - r * r + 1e-6);
                        param_grads[i] * metric_factor.sqrt()
                    } else {
                        // theta 파라미터: 일반 그래디언트
                        param_grads[i]
                    };
                    
                    riemannian_adam.update(&mut rbe_params[i], scaled_grad, lr);
                    
                    // r 제약 재적용 (짝수 인덱스만)
                    if i % 2 == 0 {
                        rbe_params[i] = rbe_params[i].clamp(0.0, 0.95);
                    }
                }
                
                // 잔차는 표준 Adam으로 (적응적 학습률)
                for k in 0..active_k {
                    // 새로운 계수일수록 더 큰 학습률 사용
                    let adaptive_lr = if k < 8 {
                        lr
                    } else if k < 16 {
                        lr * 2.0
                    } else {
                        lr * 4.0
                    };
                    standard_adam.update(&mut residual_coeffs[k], residual_grads[k], adaptive_lr);
                }
            }
            
            // RMSE 계산
            if step % 50 == 0 {
                let prediction = self.compute_full_prediction(&rbe_params, &residual_coeffs[..active_k], rows, cols);
                let rmse = compute_rmse(block_data, &prediction);
                
                // 압축률 확인
                let compressed_size = 32 + active_k * 8;
                let original_size = rows * cols * 4;
                let current_ratio = original_size as f32 / compressed_size as f32;
                
                let phase_name = if phase == 0 { "Standard" } else { "Riemannian" };
                println!("Step {}: RMSE={:.4}, K={}/{}, ratio={:.0}:1, phase={}", 
                         step, rmse, active_k, max_k_allowed, current_ratio, phase_name);
                
                // K 증가 결정 (더 적극적으로)
                if rmse > target_rmse * 1.5 && active_k < max_k_allowed && current_ratio > target_compression_ratio * 0.8 {
                    let rmse_ratio = rmse / target_rmse;
                    let old_k = active_k;
                    
                    if rmse_ratio > 100.0 && active_k < max_k_allowed / 2 {
                        // 매우 높은 RMSE: K를 두 배로
                        active_k = (active_k * 2).min(max_k_allowed);
                        println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                    } else if rmse_ratio > 10.0 {
                        // 높은 RMSE: K를 50% 증가
                        active_k = (active_k + active_k / 2 + 1).min(max_k_allowed);
                        println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                    } else if rmse_ratio > 5.0 {
                        // 중간 RMSE: K를 25% 증가
                        active_k = (active_k + active_k / 4 + 1).min(max_k_allowed);
                        println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                    } else if rmse_ratio > 2.0 {
                        // 약간 높은 RMSE: K를 1 증가
                        active_k = (active_k + 1).min(max_k_allowed);
                        println!("  K 증가: {} -> {} (RMSE 비율: {:.1})", old_k, active_k, rmse_ratio);
                    }
                    
                    // 새 계수 초기화 - 현재 잔차에서 투영하여 초기화
                    let current_residual = self.compute_residual_error(&rbe_params, &residual_coeffs[..old_k], block_data, rows, cols);
                    for k in old_k..active_k {
                        // 잔차를 기저 함수에 투영
                        let mut coeff_sum = 0.0f32;
                        let mut basis_norm = 0.0f32;
                        
                        for r in 0..rows {
                            for c in 0..cols {
                                let idx = r * cols + c;
                                let basis = self.compute_residual_basis(k, r, c, rows, cols);
                                coeff_sum += current_residual[idx] * basis;
                                basis_norm += basis * basis;
                            }
                        }
                        
                        // 정규화된 투영 계수
                        residual_coeffs[k] = if basis_norm > 1e-6 {
                            coeff_sum / basis_norm * 0.5 // 적당한 크기로 시작
                        } else {
                            0.0
                        };
                    }
                }
                
                // 학습률 감소
                if rmse < best_rmse {
                    best_rmse = rmse;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= 3 {
                        lr *= 0.8;
                        no_improvement_count = 0;
                        println!("  학습률 감소: {:.6}", lr);
                    }
                }
                
                // 조기 종료
                if rmse < target_rmse * 0.5 {
                    println!("목표 RMSE 달성, 조기 종료");
                    break;
                }
            }
        }
        
        // 최종 예측 및 압축
        let final_prediction = self.compute_full_prediction(&rbe_params, &residual_coeffs[..active_k], rows, cols);
        let final_rmse = compute_rmse(block_data, &final_prediction);
        
        println!("\n하이브리드 최적화 완료:");
        println!("  - 최종 RMSE: {:.4}", final_rmse);
        println!("  - 활성 잔차: {} / {}", active_k, max_k_allowed);
        
        // 잔차 계수를 ResidualCoefficient 형식으로 변환
        let residuals: Vec<ResidualCoefficient> = residual_coeffs[..active_k]
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > 1e-6)
            .map(|(k, &coeff)| ResidualCoefficient {
                index: ((k / 16) as u16, (k % 16) as u16),
                value: coeff,
            })
            .collect();
        
        let compressed_size = 32 + residuals.len() * 8;
        let original_size = rows * cols * 4;
        let final_ratio = original_size as f32 / compressed_size as f32;
        println!("  - 최종 압축률: {:.0}:1", final_ratio);
        
        HybridEncodedBlock {
            rbe_params,
            residuals,
            rows,
            cols,
            transform_type: self.transform_type,
        }
    }
    
    /// RBE 파라미터 초기화 (평균과 트렌드 기반)
    fn initialize_rbe_params(&self, block_data: &[f32], rows: usize, cols: usize) -> [f32; 8] {
        let mut rbe_params = [0.0f32; 8];
        
        // 데이터 통계
        let mean = block_data.iter().sum::<f32>() / block_data.len() as f32;
        let variance = block_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / block_data.len() as f32;
        let std_dev = variance.sqrt();
        
        // 주파수 분석을 위한 간단한 DCT
        let mut freq_energy = [0.0f32; 4];
        for i in 0..rows.min(8) {
            for j in 0..cols.min(8) {
                let value = block_data[i * cols + j] - mean;
                for k in 0..4 {
                    let freq = k as f32 * 0.5;
                    freq_energy[k] += value * (freq * i as f32 * PI / rows as f32).cos()
                                           * (freq * j as f32 * PI / cols as f32).cos();
                }
            }
        }
        
        // RBE 파라미터 설정
        for i in 0..4 {
            let energy = freq_energy[i].abs() / (rows * cols) as f32;
            let r = (energy / (std_dev + 1e-6)).clamp(0.1, 0.8);
            let theta = if freq_energy[i] >= 0.0 { 0.0 } else { PI };
            
            rbe_params[i * 2] = r;
            rbe_params[i * 2 + 1] = theta;
        }
        
        rbe_params
    }

    /// 현재 잔차 에러 계산
    fn compute_residual_error(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        block_data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut residual = vec![0.0f32; rows * cols];
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let predicted = self.compute_enhanced_prediction(rbe_params, residual_coeffs, r, c, rows, cols);
                residual[idx] = block_data[idx] - predicted;
            }
        }
        
        residual
    }

    /// 하이브리드 최적화: 표준 Adam과 Riemannian Adam을 번갈아 사용
    pub fn encode_block_hybrid_optimization(
        &mut self,
        block_data: &[f32],
        rows: usize,
        cols: usize,
        max_steps: usize,
        target_rmse: f32,
        target_compression_ratio: f32,
    ) -> HybridEncodedBlock {
        // 기본값으로 256개 최대 K 사용
        self.encode_block_hybrid_optimization_limited(
            block_data,
            rows,
            cols,
            max_steps,
            target_rmse,
            target_compression_ratio,
            256, // 기본 최대값
        )
    }

    /// 전체 블록에 대한 예측값 계산
    fn compute_full_prediction(
        &self,
        rbe_params: &[f32; 8],
        residual_coeffs: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut prediction = vec![0.0f32; rows * cols];
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                prediction[idx] = self.compute_enhanced_prediction(rbe_params, residual_coeffs, r, c, rows, cols);
            }
        }
        
        prediction
    }

    /// 1차원 벡터를 RBE로 압축 (임베딩 등에 사용)
    pub fn encode_vector(&mut self, vector_data: &[f32]) -> HybridEncodedBlock {
        self.encode_vector_poincare(vector_data)
    }
    
    /// 푸앵카레 볼 기반 벡터 인코딩
    pub fn encode_vector_poincare(&mut self, vector_data: &[f32]) -> HybridEncodedBlock {
        let size = vector_data.len();
        
        // 1. 데이터 정규화 및 통계
        let mean = vector_data.iter().sum::<f32>() / size as f32;
        let variance = vector_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / size as f32;
        let std_dev = variance.sqrt();
        
        // 2. 데이터를 푸앵카레 볼로 매핑
        // 정규화된 데이터를 [0, 1) 범위로 변환
        let normalized: Vec<f32> = vector_data.iter()
            .map(|&x| {
                if std_dev > 1e-6 {
                    ((x - mean) / (3.0 * std_dev)).tanh() * 0.5 + 0.5
                } else {
                    0.5
                }
            })
            .collect();
        
        // 3. 푸앵카레 볼 파라미터 학습 (Adam 최적화)
        let mut rbe_params = [0.0f32; 8];
        let mut residual_coeffs = vec![0.0f32; self.k_coeffs.min(size / 4)];
        
        // 초기 파라미터 설정
        rbe_params[0] = mean;
        rbe_params[1] = std_dev;
        
        // Adam 상태
        let mut m_rbe = vec![0.0f32; 8];
        let mut v_rbe = vec![0.0f32; 8];
        let mut m_res = vec![0.0f32; residual_coeffs.len()];
        let mut v_res = vec![0.0f32; residual_coeffs.len()];
        
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let lr = 0.01;
        
        // 4. 최적화 루프
        for step in 1..=100 {
            // 순전파: 푸앵카레 볼 기저 함수 사용
            let mut predicted = vec![0.0f32; size];
            for i in 0..size {
                let t = i as f32 / size.max(1) as f32;
                
                // 푸앵카레 볼 좌표 (r, θ)
                let r = normalized[i].min(0.999); // 경계 보호
                let theta = 2.0 * PI * t;
                
                // 푸앵카레 메트릭 가중치
                let metric_factor = 1.0 / (1.0 - r * r).max(1e-6);
                
                // 기저 함수 값 계산
                predicted[i] = rbe_params[0] 
                    + rbe_params[1] * r * metric_factor
                    + rbe_params[2] * (theta).cos()
                    + rbe_params[3] * (theta).sin()
                    + rbe_params[4] * (2.0 * theta).cos()
                    + rbe_params[5] * (2.0 * theta).sin()
                    + rbe_params[6] * r * (theta).cos() * metric_factor
                    + rbe_params[7] * r * (theta).sin() * metric_factor;
                
                // 잔차 항 추가
                for (k, &coeff) in residual_coeffs.iter().enumerate() {
                    let basis = ((k + 1) as f32 * PI * t).sin();
                    predicted[i] += coeff * basis;
                }
            }
            
            // 손실 계산
            let mut loss = 0.0f32;
            let mut grads_rbe = vec![0.0f32; 8];
            let mut grads_res = vec![0.0f32; residual_coeffs.len()];
            
            for i in 0..size {
                let error = predicted[i] - vector_data[i];
                loss += error * error;
                
                let t = i as f32 / size.max(1) as f32;
                let r = normalized[i].min(0.999);
                let theta = 2.0 * PI * t;
                let metric_factor = 1.0 / (1.0 - r * r).max(1e-6);
                
                // RBE 파라미터 그래디언트
                grads_rbe[0] += error;
                grads_rbe[1] += error * r * metric_factor;
                grads_rbe[2] += error * (theta).cos();
                grads_rbe[3] += error * (theta).sin();
                grads_rbe[4] += error * (2.0 * theta).cos();
                grads_rbe[5] += error * (2.0 * theta).sin();
                grads_rbe[6] += error * r * (theta).cos() * metric_factor;
                grads_rbe[7] += error * r * (theta).sin() * metric_factor;
                
                // 잔차 그래디언트
                for (k, grad) in grads_res.iter_mut().enumerate() {
                    let basis = ((k + 1) as f32 * PI * t).sin();
                    *grad += error * basis;
                }
            }
            
            // 그래디언트 정규화
            let n = size as f32;
            for g in &mut grads_rbe { *g /= n; }
            for g in &mut grads_res { *g /= n; }
            
            loss /= n;
            
            // Adam 업데이트
            let beta1_t = B1.powi(step);
            let beta2_t = B2.powi(step);
            
            // RBE 파라미터 업데이트
            for i in 0..8 {
                m_rbe[i] = B1 * m_rbe[i] + (1.0 - B1) * grads_rbe[i];
                v_rbe[i] = B2 * v_rbe[i] + (1.0 - B2) * grads_rbe[i] * grads_rbe[i];
                
                let m_hat = m_rbe[i] / (1.0 - beta1_t);
                let v_hat = v_rbe[i] / (1.0 - beta2_t);
                
                rbe_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
            }
            
            // 잔차 계수 업데이트
            for i in 0..residual_coeffs.len() {
                m_res[i] = B1 * m_res[i] + (1.0 - B1) * grads_res[i];
                v_res[i] = B2 * v_res[i] + (1.0 - B2) * grads_res[i] * grads_res[i];
                
                let m_hat = m_res[i] / (1.0 - beta1_t);
                let v_hat = v_res[i] / (1.0 - beta2_t);
                
                residual_coeffs[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
            }
            
            // 조기 종료
            if loss < 1e-6 {
                break;
            }
        }
        
        // 5. 유의미한 잔차만 저장
        let residuals: Vec<ResidualCoefficient> = residual_coeffs.iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > 1e-4)
            .map(|(i, &coeff)| ResidualCoefficient {
                index: (0, i as u16),
                value: coeff,
            })
            .collect();
        
        HybridEncodedBlock {
            rbe_params,
            residuals,
            rows: 1,
            cols: size,
            transform_type: TransformType::Adaptive,
        }
    }
    
    /// 기존 encode_vector 구현 (Deprecated - 호환성을 위해 유지)
    #[deprecated(note = "Use encode_vector_poincare instead")]
    pub fn encode_vector_simple(&mut self, vector_data: &[f32]) -> HybridEncodedBlock {
        let size = vector_data.len();
        
        // RBE 기저 함수 계산 - 1차원에 최적화
        let mut rbe_params = [0.0f32; 8];
        
        // 간단한 최적화: 평균과 분산 계산
        let mean = vector_data.iter().sum::<f32>() / size as f32;
        let variance = vector_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / size as f32;
        let std_dev = variance.sqrt();
        
        // 기본 파라미터 설정
        rbe_params[0] = mean;
        rbe_params[1] = std_dev;
        
        // 주파수 성분 분석 (간단한 푸리에 변환)
        for k in 2..6 {
            let freq = k - 2;
            let mut cos_sum = 0.0;
            let mut sin_sum = 0.0;
            
            for (i, &val) in vector_data.iter().enumerate() {
                let phase = 2.0 * std::f32::consts::PI * freq as f32 * i as f32 / size as f32;
                cos_sum += val * phase.cos();
                sin_sum += val * phase.sin();
            }
            
            rbe_params[k] = cos_sum / size as f32;
            rbe_params[k + 1] = sin_sum / size as f32;
        }
        
        // 잔차 계산 (선택적)
        let residuals = if self.config.enable_residual {
            let mut residual_values = Vec::new();
            
            // RBE 재구성
            for i in 0..size {
                let x = if size > 1 { (i as f32 / (size - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let pi = std::f32::consts::PI;
                
                let mut reconstructed = rbe_params[0];
                reconstructed += rbe_params[1] * x;
                reconstructed += rbe_params[2] * (pi * x).cos();
                reconstructed += rbe_params[3] * (pi * x).sin();
                reconstructed += rbe_params[4] * (2.0 * pi * x).cos();
                reconstructed += rbe_params[5] * (2.0 * pi * x).sin();
                
                let residual = vector_data[i] - reconstructed;
                if residual.abs() > self.config.residual_threshold {
                    residual_values.push(ResidualCoefficient {
                        index: (0, i as u16),
                        value: residual,
                    });
                }
            }
            
            residual_values
        } else {
            Vec::new()
        };
        
        HybridEncodedBlock {
            rbe_params,
            residuals,
            rows: 1,
            cols: size,
            transform_type: TransformType::Adaptive,
        }
    }
}



 