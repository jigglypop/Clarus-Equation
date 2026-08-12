//! RBE 기반 가중치 생성기 - HybridEncodedBlock의 decode와 일치하는 방식

use crate::packed_params::{
    HybridEncodedBlock, TransformType, HippocampalMemory, DeltaPatch, PatchTag, 
    apply_delta_scalar, apply_delta_rank1
};
use crate::core::decoder::caching_strategy::{CachingStrategy, DecoderCache, create_cache};
use std::sync::Arc;
use parking_lot::RwLock;

/// RBE 디코더 설정
#[derive(Debug, Clone)]
pub struct RBEDecoderConfig {
    /// 캐싱 전략
    pub caching_strategy: CachingStrategy,
    /// 병렬 처리 활성화
    pub enable_parallel: bool,
    /// SIMD 최적화 활성화
    pub enable_simd: bool,
}

impl RBEDecoderConfig {
    /// 최소 메모리 설정
    pub fn minimal_memory() -> Self {
        Self {
            caching_strategy: CachingStrategy::NoCache,
            enable_parallel: false,
            enable_simd: true,
        }
    }
    
    /// 균형잡힌 설정 (기본값)
    pub fn balanced() -> Self {
        Self {
            caching_strategy: CachingStrategy::PercentageBased { percentage: 0.1 },
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 최대 성능 설정
    pub fn max_performance() -> Self {
        Self {
            caching_strategy: CachingStrategy::PrecomputeAll,
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 적응형 설정
    pub fn adaptive() -> Self {
        Self {
            caching_strategy: CachingStrategy::Adaptive {
                min_size: 8,
                max_size: 1024,
                target_hit_rate: 0.9,
            },
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 레거시 호환성을 위한 생성자
    pub fn legacy(cache_size: usize) -> Self {
        Self {
            caching_strategy: CachingStrategy::FixedLRU { size: cache_size },
            enable_parallel: true,
            enable_simd: true,
        }
    }
}

impl Default for RBEDecoderConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// RBE 기반 가중치 생성기
#[derive(Clone)]
pub struct WeightGenerator {
    config: RBEDecoderConfig,
    /// 디코딩 캐시 - 블록 해시를 키로 사용
    cache: Arc<Box<dyn DecoderCache>>,
    /// 성능 통계
    stats: Arc<RwLock<DecoderStats>>,
}

impl std::fmt::Debug for WeightGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightGenerator")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

#[derive(Debug, Default, Clone)]
pub struct DecoderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_decodes: u64,
    pub total_decode_time_ns: u64,
}

impl WeightGenerator {
    /// 새로운 WeightGenerator 생성
    pub fn new() -> Self {
        Self::with_config(RBEDecoderConfig::default())
    }
    
    /// 설정 기반 생성
    pub fn with_config(config: RBEDecoderConfig) -> Self {
        let cache = create_cache(config.caching_strategy, None);
        Self {
            config,
            cache: Arc::new(cache),
            stats: Arc::new(RwLock::new(DecoderStats::default())),
        }
    }
    
    /// 총 블록 수를 알고 있을 때 생성
    pub fn with_config_and_blocks(config: RBEDecoderConfig, total_blocks: usize) -> Self {
        let cache = create_cache(config.caching_strategy, Some(total_blocks));
        Self {
            config,
            cache: Arc::new(cache),
            stats: Arc::new(RwLock::new(DecoderStats::default())),
        }
    }
    
    /// 블록 해시 계산 (캐시 키로 사용)
    fn compute_block_hash(block: &HybridEncodedBlock) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // RBE 파라미터 해시
        for &param in &block.rbe_params {
            param.to_bits().hash(&mut hasher);
        }
        
        // 블록 크기 정보
        block.rows.hash(&mut hasher);
        block.cols.hash(&mut hasher);
        
        // 잔차 계수 개수
        block.residuals.len().hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// 블록 디코딩 (HybridEncodedBlock::decode와 동일한 로직)
    pub fn decode_block(&self, block: &HybridEncodedBlock) -> Arc<Vec<f32>> {
        let start_time = std::time::Instant::now();
        
        // 캐시 확인
        let block_hash = Self::compute_block_hash(block);
        
        if let Some(cached) = self.cache.get(block_hash) {
            let mut stats = self.stats.write();
            stats.cache_hits += 1;
            stats.total_decodes += 1;
            return cached;
        }
        
        // 캐시 미스 - 디코딩 수행
        let decoded = if self.config.enable_simd {
            self.decode_block_simd(block)
        } else {
            self.decode_block_scalar(block)
        };
        
        let decoded_arc = Arc::new(decoded);
        
        // 캐시에 저장
        self.cache.put(block_hash, decoded_arc.clone());
        
        // 통계 업데이트
        {
            let mut stats = self.stats.write();
            stats.cache_misses += 1;
            stats.total_decodes += 1;
            stats.total_decode_time_ns += start_time.elapsed().as_nanos() as u64;
        }
        
        decoded_arc
    }
    
    /// 스칼라 버전 디코딩
    fn decode_block_scalar(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // HybridEncodedBlock::decode 호출
        block.decode()
    }
    
    /// SIMD 버전 디코딩 (x86_64)
    #[cfg(target_arch = "x86_64")]
    fn decode_block_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        let rows = block.rows;
        let cols = block.cols;
        let total_size = rows * cols;
        
        // SIMD 사용 불가능한 경우 스칼라 버전 사용
        if !is_x86_feature_detected!("avx2") || total_size < 8 {
            return self.decode_block_scalar(block);
        }
        
        let mut reconstruction = vec![0.0f32; total_size];
        
        unsafe {
            // RBE 파라미터를 SIMD 레지스터에 로드
            let rbe_params_ptr = block.rbe_params.as_ptr();
            let params_low = _mm256_loadu_ps(rbe_params_ptr);
            
            // 상수들
            let pi = std::f32::consts::PI;
            let two = _mm256_set1_ps(2.0);
            let one = _mm256_set1_ps(1.0);
            let pi_vec = _mm256_set1_ps(pi);
            let two_pi_vec = _mm256_set1_ps(2.0 * pi);
            
            // 8개씩 벡터화 처리
            let mut idx = 0;
            while idx + 8 <= total_size {
                // 좌표 계산
                let mut x_vals = [0.0f32; 8];
                let mut y_vals = [0.0f32; 8];
                
                for i in 0..8 {
                    let row = (idx + i) / cols;
                    let col = (idx + i) % cols;
                    x_vals[i] = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                    y_vals[i] = if rows > 1 { (row as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                }
                
                let x_vec = _mm256_loadu_ps(x_vals.as_ptr());
                let y_vec = _mm256_loadu_ps(y_vals.as_ptr());
                
                // d = sqrt(x*x + y*y)
                let x_sq = _mm256_mul_ps(x_vec, x_vec);
                let y_sq = _mm256_mul_ps(y_vec, y_vec);
                let d_sq = _mm256_add_ps(x_sq, y_sq);
                let d_vec = _mm256_sqrt_ps(d_sq);
                
                // 기저 함수 계산 (처음 4개만 벡터화)
                let basis0 = one; // 1.0
                let basis1 = d_vec; // d
                let basis2 = _mm256_mul_ps(d_vec, d_vec); // d*d
                
                // cos 계산은 스칼라로 (AVX2에는 cos 명령어가 없음)
                let mut result = _mm256_setzero_ps();
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[0]), basis0, result);
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[1]), basis1, result);
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[2]), basis2, result);
                
                // 결과 저장
                _mm256_storeu_ps(&mut reconstruction[idx], result);
                
                // 나머지 기저 함수들은 스칼라로 처리
                for i in 0..8 {
                    let x = x_vals[i];
                    let y = y_vals[i];
                    reconstruction[idx + i] += 
                        block.rbe_params[3] * (pi * x).cos() +
                        block.rbe_params[4] * (pi * y).cos() +
                        block.rbe_params[5] * (2.0 * pi * x).cos() +
                        block.rbe_params[6] * (2.0 * pi * y).cos() +
                        block.rbe_params[7] * (pi * x).cos() * (pi * y).cos();
                }
                
                idx += 8;
            }
            
            // 나머지 요소들은 스칼라로 처리
            while idx < total_size {
                let row = idx / cols;
                let col = idx % cols;
                let x = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let y = if rows > 1 { (row as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let d = (x * x + y * y).sqrt();
                
                reconstruction[idx] = 
                    block.rbe_params[0] * 1.0 +
                    block.rbe_params[1] * d +
                    block.rbe_params[2] * d * d +
                    block.rbe_params[3] * (pi * x).cos() +
                    block.rbe_params[4] * (pi * y).cos() +
                    block.rbe_params[5] * (2.0 * pi * x).cos() +
                    block.rbe_params[6] * (2.0 * pi * y).cos() +
                    block.rbe_params[7] * (pi * x).cos() * (pi * y).cos();
                
                idx += 1;
            }
        }
        
        // DWT 역변환은 block.decode()에서 처리하도록 위임
        if !block.residuals.is_empty() && matches!(block.transform_type, crate::packed_params::TransformType::Dwt) {
            // 잔차가 있으면 전체 디코딩 수행
            return block.decode();
        }
        
        reconstruction
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn decode_block_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // 다른 아키텍처에서는 스칼라 버전 사용
        self.decode_block_scalar(block)
    }
    
    /// 초고속 정수 기반 디코딩 (0.15μs 목표)
    /// 비트필드 직접 연산으로 복원
    #[inline(always)]
    pub fn decode_int_adam_fast(
        &self,
        block: &HybridEncodedBlock,
        i: usize,
        j: usize,
    ) -> f32 {
        // 0. 사전 계산된 상수
        const INV_ROWS: i32 = 1024; // 1/64 in Q16.16 (64x64 블록 가정)
        const INV_COLS: i32 = 1024;
        
        // 1. 좌표 정규화 (정수 연산)
        let x_int = (j as i32 * INV_COLS) >> 6; // Q10.10
        let y_int = (i as i32 * INV_ROWS) >> 6;
        
        // 2. RBE 파라미터로부터 직접 계산 (비트필드 연산)
        let mut val = 0i32;
        
        // 기저 함수 0: 상수항
        val += (block.rbe_params[0] * 65536.0) as i32;
        
        // 기저 함수 1,2: 선형항
        val += ((block.rbe_params[1] * 65536.0) as i32 * x_int) >> 10;
        val += ((block.rbe_params[2] * 65536.0) as i32 * y_int) >> 10;
        
        // 기저 함수 3,4: 이차항 (비트시프트로 제곱 근사)
        let x2 = (x_int * x_int) >> 10;
        let y2 = (y_int * y_int) >> 10;
        val += ((block.rbe_params[3] * 65536.0) as i32 * x2) >> 10;
        val += ((block.rbe_params[4] * 65536.0) as i32 * y2) >> 10;
        
        // 기저 함수 5: 교차항
        val += ((block.rbe_params[5] * 65536.0) as i32 * x_int * y_int) >> 20;
        
        // 기저 함수 6,7: 삼각함수 (근사)
        // cos(πx) ≈ 1 - 2(πx/2)² for small x
        let px = (x_int * 3217) >> 10; // π*x
        let cos_px = 65536 - ((px * px) >> 15);
        val += ((block.rbe_params[6] * 65536.0) as i32 * cos_px) >> 16;
        
        let py = (y_int * 3217) >> 10;
        let cos_py = 65536 - ((py * py) >> 15);
        val += ((block.rbe_params[7] * 65536.0) as i32 * cos_py) >> 16;
        
        // 3. 잔차 추가 (최대 2개, 평균 0.8개)
        for coeff in &block.residuals {
            if coeff.index == (i as u16, j as u16) {
                val += (coeff.value * 65536.0) as i32;
                break; // 위치당 최대 1개
            }
        }
        
        // 4. 정수 → 부동소수점 변환
        (val as f32) / 65536.0
    }
    
    /// 벡터화된 초고속 디코딩 (SIMD 최적화)
    #[cfg(target_arch = "x86_64")]
    pub fn decode_int_adam_simd(
        &self,
        block: &HybridEncodedBlock,
        positions: &[(usize, usize)],
    ) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        // RBE 파라미터를 SIMD 레지스터로 로드
        let params_int: [i32; 8] = block.rbe_params.iter()
            .map(|&p| (p * 65536.0) as i32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        
        let mut results = vec![0f32; positions.len()];
        
        // 8개씩 병렬 처리
        for (chunk_idx, chunk) in positions.chunks(8).enumerate() {
            unsafe {
                // 좌표 로드 및 정규화
                let mut x_vals = [0i32; 8];
                let mut y_vals = [0i32; 8];
                
                for (i, &(row, col)) in chunk.iter().enumerate() {
                    x_vals[i] = (col as i32 * 1024) >> 6;
                    y_vals[i] = (row as i32 * 1024) >> 6;
                }
                
                let x_vec = _mm256_loadu_si256(x_vals.as_ptr() as *const __m256i);
                let y_vec = _mm256_loadu_si256(y_vals.as_ptr() as *const __m256i);
                
                // 상수항
                let mut val_vec = _mm256_set1_epi32(params_int[0]);
                
                // 선형항
                let p1 = _mm256_set1_epi32(params_int[1]);
                let p2 = _mm256_set1_epi32(params_int[2]);
                val_vec = _mm256_add_epi32(val_vec, _mm256_srai_epi32(_mm256_mullo_epi32(p1, x_vec), 10));
                val_vec = _mm256_add_epi32(val_vec, _mm256_srai_epi32(_mm256_mullo_epi32(p2, y_vec), 10));
                
                // 이차항
                let x2_vec = _mm256_srai_epi32(_mm256_mullo_epi32(x_vec, x_vec), 10);
                let y2_vec = _mm256_srai_epi32(_mm256_mullo_epi32(y_vec, y_vec), 10);
                let p3 = _mm256_set1_epi32(params_int[3]);
                let p4 = _mm256_set1_epi32(params_int[4]);
                val_vec = _mm256_add_epi32(val_vec, _mm256_srai_epi32(_mm256_mullo_epi32(p3, x2_vec), 10));
                val_vec = _mm256_add_epi32(val_vec, _mm256_srai_epi32(_mm256_mullo_epi32(p4, y2_vec), 10));
                
                // 결과 저장
                let mut temp_results = [0i32; 8];
                _mm256_storeu_si256(temp_results.as_mut_ptr() as *mut __m256i, val_vec);
                
                for (i, &val) in temp_results.iter().enumerate() {
                    if chunk_idx * 8 + i < results.len() {
                        results[chunk_idx * 8 + i] = (val as f32) / 65536.0;
                    }
                }
            }
        }
        
        // 잔차 추가 (SIMD 이후 처리)
        for (idx, &(i, j)) in positions.iter().enumerate() {
            for coeff in &block.residuals {
                if coeff.index == (i as u16, j as u16) {
                    results[idx] += coeff.value;
                    break;
                }
            }
        }
        
        results
    }
    
    /// 블록 전체 디코딩 (초고속 버전)
    pub fn decode_block_int_adam(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        let rows = block.rows;
        let cols = block.cols;
        let mut result = vec![0.0f32; rows * cols];
        
        // SIMD 가능한 경우 벡터화 사용
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let positions: Vec<(usize, usize)> = (0..rows)
                    .flat_map(|r| (0..cols).map(move |c| (r, c)))
                    .collect();
                
                let decoded = self.decode_int_adam_simd(block, &positions);
                result.copy_from_slice(&decoded);
                return result;
            }
        }
        
        // Fallback: 스칼라 버전
        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] = self.decode_int_adam_fast(block, i, j);
            }
        }
        
        result
    }
    
    /// 향상된 블록 디코딩 - 다양한 기저 함수 지원
    pub fn decode_block_enhanced(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        let size = block.rows * block.cols;
        let mut result = vec![0.0f32; size];
        
        // 병렬 디코딩
        use rayon::prelude::*;
        result.par_chunks_mut(block.cols)
            .enumerate()
            .for_each(|(r, row)| {
                for c in 0..block.cols {
                    row[c] = self.decode_pixel_enhanced(block, r, c);
                }
            });
        
        result
    }
    
    /// 향상된 픽셀 디코딩
    fn decode_pixel_enhanced(&self, block: &HybridEncodedBlock, r: usize, c: usize) -> f32 {
        let x = if block.cols > 0 { c as f32 / block.cols as f32 } else { 0.0 };
        let y = if block.rows > 0 { r as f32 / block.rows as f32 } else { 0.0 };
        
        // RBE 기저 함수 적용
        let basis = [
            1.0,                                    // 상수
            x,                                      // 선형 x
            y,                                      // 선형 y
            x * y,                                  // 교차항
            (2.0 * std::f32::consts::PI * x).cos(), // 코사인 x
            (2.0 * std::f32::consts::PI * y).cos(), // 코사인 y
            x * x - 0.5,                            // 2차 x (중심화)
            y * y - 0.5,                            // 2차 y (중심화)
        ];
        
        let mut val = 0.0f32;
        for i in 0..8 {
            val += block.rbe_params[i] * basis[i];
        }
        
        // 잔차 계수 적용
        for residual in &block.residuals {
            let (k_high, k_low) = residual.index;
            let k = (k_high as usize) * 16 + (k_low as usize);
            
            let residual_basis = match k % 16 {
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
                14 => (x + y) * ((k + 1) as f32).sqrt(),
                15 => (x - y).abs() * ((k + 1) as f32).sqrt(),
                _ => 0.0,
            };
            
            val += residual.value * residual_basis;
        }
        
        val
    }
    
    /// SIMD 가속 향상된 디코딩
    #[cfg(target_arch = "x86_64")]
    pub fn decode_block_enhanced_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        if !is_x86_feature_detected!("avx2") {
            return self.decode_block_enhanced(block);
        }
        
        let size = block.rows * block.cols;
        let mut result = vec![0.0f32; size];
        
        unsafe {
            use std::arch::x86_64::*;
            
            // RBE 파라미터를 SIMD 레지스터에 로드
            let params: [__m256; 1] = [
                _mm256_loadu_ps(block.rbe_params.as_ptr()),
            ];
            
            // 8픽셀씩 병렬 처리
            for chunk_start in (0..size).step_by(8) {
                if chunk_start + 8 <= size {
                    let mut vals = _mm256_setzero_ps();
                    
                    // 각 픽셀의 좌표 계산
                    let indices = _mm256_set_epi32(
                        (chunk_start + 7) as i32,
                        (chunk_start + 6) as i32,
                        (chunk_start + 5) as i32,
                        (chunk_start + 4) as i32,
                        (chunk_start + 3) as i32,
                        (chunk_start + 2) as i32,
                        (chunk_start + 1) as i32,
                        chunk_start as i32,
                    );
                    
                    let cols_vec = _mm256_set1_ps(block.cols as f32);
                    let indices_f32 = _mm256_cvtepi32_ps(indices);
                    let rows_vec = _mm256_div_ps(indices_f32, cols_vec);
                    let cols_indices = _mm256_sub_ps(indices_f32, _mm256_mul_ps(_mm256_floor_ps(rows_vec), cols_vec));
                    
                    let inv_cols = _mm256_set1_ps(1.0 / block.cols.max(1) as f32);
                    let inv_rows = _mm256_set1_ps(1.0 / block.rows.max(1) as f32);
                    
                    let x = _mm256_mul_ps(cols_indices, inv_cols);
                    let y = _mm256_mul_ps(rows_vec, inv_rows);
                    
                    // 기저 함수 계산 및 적용
                    vals = _mm256_add_ps(vals, _mm256_mul_ps(_mm256_set1_ps(block.rbe_params[0]), _mm256_set1_ps(1.0)));
                    vals = _mm256_add_ps(vals, _mm256_mul_ps(_mm256_set1_ps(block.rbe_params[1]), x));
                    vals = _mm256_add_ps(vals, _mm256_mul_ps(_mm256_set1_ps(block.rbe_params[2]), y));
                    vals = _mm256_add_ps(vals, _mm256_mul_ps(_mm256_set1_ps(block.rbe_params[3]), _mm256_mul_ps(x, y)));
                    
                    // 결과 저장
                    _mm256_storeu_ps(&mut result[chunk_start], vals);
                } else {
                    // 나머지 픽셀은 스칼라로 처리
                    for i in chunk_start..size {
                        let r = i / block.cols;
                        let c = i % block.cols;
                        result[i] = self.decode_pixel_enhanced(block, r, c);
                    }
                }
            }
            
            // 잔차 계수는 스칼라로 처리 (복잡한 기저 함수 때문)
            for i in 0..size {
                let r = i / block.cols;
                let c = i % block.cols;
                let x = if block.cols > 0 { c as f32 / block.cols as f32 } else { 0.0 };
                let y = if block.rows > 0 { r as f32 / block.rows as f32 } else { 0.0 };
                
                for residual in &block.residuals {
                    let (k_high, k_low) = residual.index;
                    let k = (k_high as usize) * 16 + (k_low as usize);
                    
                    let residual_basis = match k % 16 {
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
                        14 => (x + y) * ((k + 1) as f32).sqrt(),
                        15 => (x - y).abs() * ((k + 1) as f32).sqrt(),
                        _ => 0.0,
                    };
                    
                    result[i] += residual.value * residual_basis;
                }
            }
        }
        
        result
    }
    
    /// 패치를 적용한 블록 디코딩
    pub fn decode_block_with_patches(
        &self,
        block: &HybridEncodedBlock,
        hippocampal_memory: Option<&HippocampalMemory>,
    ) -> Vec<f32> {
        // 1. 기본 디코딩
        let mut weights = self.decode_block_int_adam(block);
        
        // 2. 패치 적용 (있는 경우)
        if let Some(memory) = hippocampal_memory {
            let block_id = self.compute_block_id(block.rows, block.cols);
            let patches = memory.get_patches(block_id);
            
            for patch in patches {
                self.apply_patch(&mut weights, &patch, block.rows, block.cols);
            }
        }
        
        weights
    }
    
    /// 단일 패치 적용
    fn apply_patch(&self, weights: &mut [f32], patch: &DeltaPatch, rows: usize, cols: usize) {
        match patch.tag {
            PatchTag::DeltaScalar => {
                if patch.payload.len() >= 1 {
                    let alpha = f32::from_bits(patch.payload[0]);
                    apply_delta_scalar(weights, alpha);
                }
            }
            PatchTag::DeltaRank1 => {
                if patch.payload.len() >= 3 {
                    let u_idx = (patch.payload[0] >> 16) as usize;
                    let v_idx = (patch.payload[0] & 0xFFFF) as usize;
                    let scale = f32::from_bits(patch.payload[1]);
                    apply_delta_rank1(weights, rows, cols, u_idx, v_idx, scale);
                }
            }
            PatchTag::MaskDrop => {
                // 잔차를 0으로 만드는 효과
                // 이미 디코딩된 weights에서는 적용 불가, 인코딩 단계에서 처리 필요
            }
            _ => {}
        }
    }
    
    /// 블록 ID 계산 (해시 기반)
    fn compute_block_id(&self, rows: usize, cols: usize) -> u16 {
        // 간단한 해시 함수로 블록 ID 생성
        let hash = (rows * 31 + cols) as u32;
        (hash % 1024) as u16
    }
    
    /// 캐시 정리
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// 블록들을 미리 디코딩하여 캐시에 로드
    pub fn preload_blocks(&self, blocks: &[HybridEncodedBlock]) -> usize {
        let mut loaded = 0;
        for block in blocks {
            let _ = self.decode_block(block);
            loaded += 1;
        }
        loaded
    }
    
    /// 워밍업 - 캐시를 미리 채우기
    pub fn warmup(&self, blocks: &[HybridEncodedBlock], warmup_ratio: f32) {
        let num_to_load = (blocks.len() as f32 * warmup_ratio.min(1.0)) as usize;
        let start = std::time::Instant::now();
        
        println!("워밍업 시작: {} / {} 블록 사전 로딩", num_to_load, blocks.len());
        
        // 병렬로 블록 디코딩
        use rayon::prelude::*;
        blocks.par_iter()
            .take(num_to_load)
            .for_each(|block| {
                let _ = self.decode_block(block);
            });
            
        let elapsed = start.elapsed();
        println!("워밍업 완료: {:.2}ms 소요", elapsed.as_millis());
    }
    
    /// 캐시 크기 동적 조정
    pub fn resize_cache(&mut self, new_size: usize) -> bool {
        if new_size == 0 {
            return false;
        }
        
        self.cache.resize(new_size)
    }
    
    /// 통계 반환
    pub fn get_stats(&self) -> DecoderStats {
        let stats = self.stats.read();
        let cache_stats = self.cache.stats();
        
        DecoderStats {
            cache_hits: cache_stats.hits,
            cache_misses: cache_stats.misses,
            total_decodes: stats.total_decodes,
            total_decode_time_ns: stats.total_decode_time_ns,
        }
    }
    
    /// 캐시 히트율 계산
    pub fn get_cache_hit_rate(&self) -> f32 {
        let cache_stats = self.cache.stats();
        cache_stats.hit_rate()
    }
}

/// 전역 가중치 생성기 (후위 호환성)
pub fn generate_weight_global(
    block: &HybridEncodedBlock,
    _row: usize,
    _col: usize,
    _rows: usize,
    _cols: usize,
) -> f32 {
    // 단순히 블록을 디코딩하고 평균값 반환 (호환성용)
    let decoded = block.decode();
    decoded.iter().sum::<f32>() / decoded.len() as f32
}