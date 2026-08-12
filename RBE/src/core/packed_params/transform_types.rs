//! 변환 타입 정의 (DCT, Wavelet 등)

use serde::{Serialize, Deserialize};

/// 변환 타입
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransformType {
    Dct,      // Discrete Cosine Transform
    Dwt,      // Discrete Wavelet Transform
    Adaptive, // 적응형 (현재는 DCT)
}

/// 잔차 계수 구조체
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualCoefficient {
    /// 계수의 2차원 인덱스 (row, col)
    pub index: (u16, u16),
    /// 계수 값
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridEncodedBlock {
    /// RBE 파라미터 (8개)
    pub rbe_params: [f32; 8],
    /// 잔차 계수들 (스파스 저장)
    pub residuals: Vec<ResidualCoefficient>,
    /// 블록 크기 정보
    pub rows: usize,
    pub cols: usize,
    /// 사용된 변환 타입
    pub transform_type: TransformType,
}

/// 단일 인코딩 블록의 그래디언트를 저장하는 구조체
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBlockGradients {
    pub rbe_params_grad: RbeParameters,
    pub residuals_grad: Vec<ResidualCoefficient>,
}

/// RBE 기본 패턴 파라미터 타입 별칭
pub type RbeParameters = [f32; 8]; 