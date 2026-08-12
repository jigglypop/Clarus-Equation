//! 통합된 RBE 기저 함수 모듈
//! 
//! encoder와 decoder에서 중복 구현된 기저 함수들을 통합

use std::f32::consts::PI;

/// RBE 기저 함수 계산 (8개 기본 함수)
#[inline(always)]
pub fn compute_rbe_basis(x: f32, y: f32) -> [f32; 8] {
    let d = (x * x + y * y).sqrt();
    
    [
        1.0,                        // 상수항
        d,                          // 거리
        d * d,                      // 거리 제곱
        (PI * x).cos(),            // cos(πx)
        (PI * y).cos(),            // cos(πy)
        (2.0 * PI * x).cos(),      // cos(2πx)
        (2.0 * PI * y).cos(),      // cos(2πy)
        (PI * x).cos() * (PI * y).cos(), // cos(πx)cos(πy)
    ]
}

/// 1차원 벡터용 기저 함수
#[inline(always)]
pub fn compute_rbe_basis_1d(x: f32) -> [f32; 8] {
    [
        1.0,                        // 상수항
        x.abs(),                    // |x|
        x * x,                      // x²
        (PI * x).cos(),            // cos(πx)
        1.0,                        // cos(0) = 1
        (2.0 * PI * x).cos(),      // cos(2πx)
        1.0,                        // cos(0) = 1
        (PI * x).cos(),            // cos(πx) * 1
    ]
}

/// 확장된 기저 함수 (enhanced 버전용)
#[inline(always)]
pub fn compute_enhanced_basis(x: f32, y: f32) -> [f32; 8] {
    [
        1.0,                        // 상수
        x,                          // 선형 x
        y,                          // 선형 y
        x * y,                      // 교차항
        (2.0 * PI * x).cos(),      // 코사인 x
        (2.0 * PI * y).cos(),      // 코사인 y
        x * x - 0.5,                // 2차 x (중심화)
        y * y - 0.5,                // 2차 y (중심화)
    ]
}

/// 좌표 정규화
#[inline(always)]
pub fn normalize_coords(i: usize, j: usize, rows: usize, cols: usize) -> (f32, f32) {
    let x = if cols > 1 { 
        (j as f32 / (cols - 1) as f32) * 2.0 - 1.0 
    } else { 
        0.0 
    };
    
    let y = if rows > 1 { 
        (i as f32 / (rows - 1) as f32) * 2.0 - 1.0 
    } else { 
        0.0 
    };
    
    (x, y)
}

/// 픽셀 값 계산
#[inline(always)]
pub fn compute_pixel_value(rbe_params: &[f32; 8], basis: &[f32; 8]) -> f32 {
    let mut val = 0.0f32;
    for i in 0..8 {
        val += rbe_params[i] * basis[i];
    }
    val
}

/// RMSE 계산 (통합 버전)
pub fn compute_rmse(actual: &[f32], predicted: &[f32]) -> f32 {
    if actual.len() != predicted.len() {
        return f32::INFINITY;
    }
    
    let sum_sq_error: f32 = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    
    (sum_sq_error / actual.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basis_functions() {
        let (x, y) = normalize_coords(32, 32, 64, 64);
        let basis = compute_rbe_basis(x, y);
        
        // 기본 검증
        assert_eq!(basis[0], 1.0);
        assert!(basis[1] >= 0.0); // 거리는 항상 양수
        
        // 1D 버전 테스트
        let basis_1d = compute_rbe_basis_1d(0.5);
        assert_eq!(basis_1d[0], 1.0);
        assert_eq!(basis_1d[4], 1.0); // cos(0) = 1
    }
} 