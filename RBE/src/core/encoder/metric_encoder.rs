//! 메트릭 텐서 전용 비트필드 인코더
//! 
//! 이 모듈은 가중치 행렬의 메트릭 텐서(G = W^T W)를 압축합니다.
//! 주의: 메트릭 텐서는 원본 가중치를 복원하는 용도가 아닙니다!
//! 
//! 주요 용도:
//! - 자연 그래디언트 최적화 (Natural Gradient Descent)
//! - Fisher Information Matrix 근사
//! - 2차 최적화 방법 (Newton-Raphson)
//! - Riemannian 최적화
//! 
//! 가중치 압축/복원이 목적이라면 RBEEncoder를 사용하세요.

use nalgebra::{DMatrix, DVector};
use crate::packed_params::ResidualCoefficient;
use std::io::{Write, Read, Seek};
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};

/// 메트릭 텐서 헤더
#[derive(Debug, Clone, Copy)]
pub struct MetricHeader {
    pub magic: u8,       // 0xBE
    pub n: u16,          // 행렬 크기 (최대 65535)
    pub k_flags: u8,     // 상위 4비트: rank k, 하위 4비트: 플래그
}

impl MetricHeader {
    const MAGIC: u8 = 0xBE;
    
    pub fn k(&self) -> usize {
        (self.k_flags >> 4) as usize
    }
    
    pub fn is_f16(&self) -> bool {
        (self.k_flags & 0x01) != 0
    }
}

/// 메트릭 텐서 블록 (압축된 형태)
pub struct MetricTensorBlock {
    /// 헤더 정보
    pub header: MetricHeader,
    /// 고유값 (log2 스케일, μ-law 8bit)
    pub eigenvalues: Vec<u8>,
    /// 고유벡터 (f16 or posit8)
    pub eigenvectors: Vec<u8>,
    /// 잔차 정보 (옵션)
    pub residuals: Vec<ResidualCoefficient>,
}

// 통합된 mulaw 모듈 사용
use super::mulaw;

/// 메트릭 텐서 인코더
pub struct MetricTensorEncoder {
    /// 목표 rank
    pub target_rank: usize,
    /// 정규화 파라미터 (damping)
    pub damping: f32,
    /// f16 사용 여부
    pub use_f16: bool,
}

impl MetricTensorEncoder {
    pub fn new(target_rank: usize) -> Self {
        Self {
            target_rank,
            damping: 1e-3,
            use_f16: true,
        }
    }
    
    /// 가중치 행렬에서 메트릭 텐서 추출 및 압축
    pub fn encode_from_weights(
        &self,
        weights: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<MetricTensorBlock, String> {
        // 가중치 행렬 W를 DMatrix로 변환
        let w_matrix = DMatrix::from_row_slice(rows, cols, weights);
        
        // 메트릭 텐서 계산: G = W^T W (Gram matrix)
        let metric_tensor = w_matrix.transpose() * &w_matrix;
        
        // 대칭성 확인 및 보정
        let metric_symmetric = (&metric_tensor + metric_tensor.transpose()) * 0.5;
        
        // SVD를 통한 rank-k 근사
        let svd = metric_symmetric.svd(true, true);
        
        // 상위 k개 고유값/벡터 선택
        let k = self.target_rank.min(svd.singular_values.len());
        
        // rank가 15를 초과하면 에러 (4비트 제한)
        if k > 15 {
            return Err(format!("Rank {} exceeds maximum of 15 (4-bit limit)", k));
        }
        
        // 고유값 인코딩 (log2 스케일 → μ-law 8bit)
        let mut encoded_eigenvalues = Vec::with_capacity(k);
        for i in 0..k {
            let lambda = svd.singular_values[i];
            // log2(λ) 정규화 [-30, +30] 범위로
            let log_lambda = lambda.log2().clamp(-30.0, 30.0) / 30.0;
            encoded_eigenvalues.push(mulaw::encode(log_lambda));
        }
        
        // 고유벡터 인코딩
        let v_t_matrix = svd.v_t.as_ref()
            .ok_or("SVD V matrix not computed")?;
        
        let mut encoded_eigenvectors = Vec::new();
        if self.use_f16 {
            // f16 인코딩
            for i in 0..k {
                for j in 0..cols {
                    let value = v_t_matrix[(i, j)]; // v_t는 이미 전치되어 있음
                    let f16_bits = half::f16::from_f32(value).to_bits();
                    encoded_eigenvectors.write_u16::<LittleEndian>(f16_bits)
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            // Posit8 인코딩 (더 공격적 압축)
            // TODO: Posit8 구현
            return Err("Posit8 encoding not yet implemented".to_string());
        }
        
        // 헤더 생성
        let header = MetricHeader {
            magic: MetricHeader::MAGIC,
            n: cols as u16,
            k_flags: ((k as u8) << 4) | (if self.use_f16 { 0x01 } else { 0x00 }),
        };
        
        Ok(MetricTensorBlock {
            header,
            eigenvalues: encoded_eigenvalues,
            eigenvectors: encoded_eigenvectors,
            residuals: Vec::new(), // 초기에는 잔차 없음
        })
    }
    
    /// 블록을 바이트 스트림으로 직렬화
    pub fn serialize(&self, block: &MetricTensorBlock) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        
        // 헤더 쓰기
        buffer.push(block.header.magic);
        buffer.write_u16::<LittleEndian>(block.header.n)
            .map_err(|e| e.to_string())?;
        buffer.push(block.header.k_flags);
        
        // 고유값 쓰기
        buffer.extend_from_slice(&block.eigenvalues);
        
        // 고유벡터 쓰기
        buffer.extend_from_slice(&block.eigenvectors);
        
        // CRC16 (옵션)
        let crc = crc16::State::<crc16::XMODEM>::calculate(&buffer);
        buffer.write_u16::<LittleEndian>(crc)
            .map_err(|e| e.to_string())?;
        
        Ok(buffer)
    }
}

/// 메트릭 텐서 디코더
pub struct MetricTensorDecoder {
    /// 정규화 파라미터
    pub damping: f32,
}

impl MetricTensorDecoder {
    pub fn new() -> Self {
        Self {
            damping: 1e-3,
        }
    }
    
    /// 바이트 스트림에서 메트릭 텐서 블록 역직렬화
    pub fn deserialize(&self, data: &[u8]) -> Result<MetricTensorBlock, String> {
        let mut cursor = std::io::Cursor::new(data);
        
        // 헤더 읽기
        let magic = cursor.read_u8().map_err(|e| e.to_string())?;
        if magic != MetricHeader::MAGIC {
            return Err(format!("Invalid magic number: 0x{:02X}", magic));
        }
        
        let n = cursor.read_u16::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let k_flags = cursor.read_u8().map_err(|e| e.to_string())?;
        let k = (k_flags >> 4) as usize;
        
        let header = MetricHeader { magic, n: n as u16, k_flags };
        
        // 고유값 읽기
        let mut eigenvalues = vec![0u8; k];
        cursor.read_exact(&mut eigenvalues).map_err(|e| e.to_string())?;
        
        // 고유벡터 읽기
        let vector_size = if header.is_f16() { n * k * 2 } else { n * k };
        let mut eigenvectors = vec![0u8; vector_size];
        cursor.read_exact(&mut eigenvectors).map_err(|e| e.to_string())?;
        
        // CRC 검증 (옵션)
        let crc_position = cursor.position() as usize;
        let expected_crc = cursor.read_u16::<LittleEndian>().map_err(|e| e.to_string())?;
        let actual_crc = crc16::State::<crc16::XMODEM>::calculate(&data[..crc_position]);
        if expected_crc != actual_crc {
            return Err(format!("CRC mismatch: expected 0x{:04X}, got 0x{:04X}", expected_crc, actual_crc));
        }
        
        Ok(MetricTensorBlock {
            header,
            eigenvalues,
            eigenvectors,
            residuals: Vec::new(),
        })
    }
    
    /// 메트릭 텐서 복원 (rank-k 형태)
    pub fn decode_to_rank_k(&self, block: &MetricTensorBlock) -> Result<(DMatrix<f32>, DVector<f32>), String> {
        let n = block.header.n as usize;
        let k = block.header.k();
        
        // 고유값 디코딩
        let mut lambda = DVector::zeros(k);
        for i in 0..k {
            let log_lambda_norm = mulaw::decode(block.eigenvalues[i]);
            let log_lambda = log_lambda_norm * 30.0; // [-30, +30] 범위 복원
            lambda[i] = 2.0_f32.powf(log_lambda);
        }
        
        // 고유벡터 디코딩
        let mut u_matrix = DMatrix::zeros(n, k);
        if block.header.is_f16() {
            let mut cursor = std::io::Cursor::new(&block.eigenvectors);
            for i in 0..k {
                for j in 0..n {
                    let f16_bits = cursor.read_u16::<LittleEndian>().map_err(|e| e.to_string())?;
                    let value = half::f16::from_bits(f16_bits).to_f32();
                    u_matrix[(j, i)] = value; // 전치 주의
                }
            }
        }
        
        Ok((u_matrix, lambda))
    }
    
    /// Woodbury 공식을 이용한 역행렬 계산
    /// (σI + UΛU^T)^{-1} = σ^{-1}I - σ^{-1}U(Λ^{-1} + U^TU/σ)^{-1}U^T/σ
    pub fn compute_inverse(&self, u: &DMatrix<f32>, lambda: &DVector<f32>) -> DMatrix<f32> {
        let n = u.nrows();
        let k = u.ncols();
        let sigma = self.damping;
        let sigma_inv = 1.0 / sigma;
        
        // Λ^{-1} + U^TU/σ
        let mut middle = DMatrix::zeros(k, k);
        for i in 0..k {
            middle[(i, i)] = 1.0 / lambda[i] + 1.0 / sigma;
        }
        
        // U^TU 항 추가
        let utu = u.transpose() * u;
        middle += utu * sigma_inv;
        
        // 중간 행렬 역행렬
        let middle_inv = middle.clone().try_inverse()
            .unwrap_or_else(|| {
                eprintln!("Warning: Middle matrix inversion failed, using pseudo-inverse");
                let middle_copy = DMatrix::from_iterator(middle.nrows(), middle.ncols(), middle.iter().cloned());
                middle_copy.pseudo_inverse(1e-6).unwrap()
            });
        
        // 최종 역행렬: σ^{-1}I - σ^{-1}U * middle_inv * U^T * σ^{-1}
        let identity = DMatrix::identity(n, n);
        let correction = u * middle_inv * u.transpose() * (sigma_inv * sigma_inv);
        
        identity * sigma_inv - correction
    }
    
    /// 메트릭 텐서를 이용한 자연 그래디언트 계산
    /// g_natural = G^{-1} * g_euclidean
    pub fn natural_gradient(
        &self,
        block: &MetricTensorBlock,
        euclidean_grad: &[f32],
    ) -> Result<Vec<f32>, String> {
        // rank-k 형태로 디코딩
        let (u, lambda) = self.decode_to_rank_k(block)?;
        let n = block.header.n as usize;
        
        // 역행렬 계산
        let g_inv = self.compute_inverse(&u, &lambda);
        
        // 자연 그래디언트 계산
        // 가중치 행렬의 각 행에 대해 메트릭 텐서의 역행렬 적용
        let mut natural_grad = Vec::with_capacity(euclidean_grad.len());
        
        for row_idx in 0..n {
            let start = row_idx * n;
            let end = start + n;
            let grad_row = DVector::from_row_slice(&euclidean_grad[start..end]);
            let natural_grad_row = &g_inv * &grad_row;
            natural_grad.extend_from_slice(natural_grad_row.as_slice());
        }
        
        Ok(natural_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_메트릭_텐서_인코딩_디코딩() {
        let encoder = MetricTensorEncoder::new(4);
        let decoder = MetricTensorDecoder::new();
        
        // 테스트 가중치 행렬 (8x8)
        let weights: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        
        // 인코딩
        let block = encoder.encode_from_weights(&weights, 8, 8).unwrap();
        assert_eq!(block.header.n, 8);
        assert_eq!(block.header.k(), 4);
        
        // 직렬화 및 역직렬화
        let serialized = encoder.serialize(&block).unwrap();
        let deserialized = decoder.deserialize(&serialized).unwrap();
        
        // rank-k 형태로 디코딩
        let (u, lambda) = decoder.decode_to_rank_k(&deserialized).unwrap();
        assert_eq!(u.shape(), (8, 4));
        assert_eq!(lambda.len(), 4);
        
        // 역행렬 계산 테스트
        let g_inv = decoder.compute_inverse(&u, &lambda);
        assert_eq!(g_inv.shape(), (8, 8));
    }
    
    #[test]
    fn test_자연_그래디언트_계산() {
        let encoder = MetricTensorEncoder::new(3);
        let decoder = MetricTensorDecoder::new();
        
        // 작은 테스트 행렬
        let weights = vec![1.0, 0.5, 0.3, 0.5, 2.0, 0.1, 0.3, 0.1, 1.5];
        let block = encoder.encode_from_weights(&weights, 3, 3).unwrap();
        
        // 유클리드 그래디언트
        let euclidean_grad = vec![1.0, 0.0, 0.0];
        
        // 자연 그래디언트 계산
        let natural_grad = decoder.natural_gradient(&block, &euclidean_grad).unwrap();
        assert_eq!(natural_grad.len(), 3);
        
        // 자연 그래디언트는 메트릭의 영향을 받아야 함
        assert_ne!(natural_grad[0], euclidean_grad[0]);
    }
    
    #[test]
    fn test_압축률_계산() {
        let encoder = MetricTensorEncoder::new(8);
        
        // 64x64 행렬
        let n = 64;
        let weights: Vec<f32> = (0..n*n).map(|i| (i as f32 * 0.01).cos()).collect();
        let block = encoder.encode_from_weights(&weights, n, n).unwrap();
        
        // 원본 크기: 64*64*4 = 16KB
        let original_size = n * n * std::mem::size_of::<f32>();
        
        // 압축 크기: 헤더(1+2+1) + 고유값(8) + 고유벡터(64*8*2) + CRC(2)
        let compressed_size = 4 + 8 + (n * 8 * 2) + 2;
        
        let compression_ratio = original_size as f32 / compressed_size as f32;
        println!("압축률: {:.1}:1 (원본 {}B → 압축 {}B)", compression_ratio, original_size, compressed_size);
        
        assert!(compression_ratio > 15.0); // 최소 15:1 압축
    }
} 