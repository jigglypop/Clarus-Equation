//! # NLP 모듈 - RBE 기반 언어 모델 구성요소들
//!
//! core 구현체를 활용한 실제 NLP 레이어 구현

pub mod linear;
pub mod model_tools;
pub mod bert_inference;
pub mod embedding;
pub mod layernorm;
pub mod ffn;
pub mod attention;
pub mod dropout;
pub mod softmax;
pub mod rmsnorm;
pub mod accuracy_utils;

// candle 비교 테스트
#[cfg(test)]
pub mod candle_comparison_test;

// tensor 모듈 제거 - WeightGenerator 직접 사용으로 대체

// 재-export for convenience
pub use linear::*;
pub use model_tools::*;

/// 두 벡터 간의 상대 오차 계산
pub fn compute_relative_error(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    
    for (&r, &a) in reference.iter().zip(approximation.iter()) {
        let diff = (r - a).abs();
        let ref_abs = r.abs();
        
        num += diff * diff;
        den += ref_abs * ref_abs;
    }
    
    if den < 1e-10 {
        return if num < 1e-10 { 0.0 } else { f32::INFINITY };
    }
    
    (num / den).sqrt()
} 