//! μ-law 인코딩/디코딩 유틸리티

const MU: f32 = 255.0;

/// μ-law 인코딩
pub fn encode(x: f32) -> u8 {
    let x_norm = x.clamp(-1.0, 1.0);
    let sign = if x_norm >= 0.0 { 0u8 } else { 128u8 };
    let compressed = (x_norm.abs().ln_1p() / (1.0 + MU).ln()) * 127.0;
    sign | (compressed as u8)
}

/// μ-law 디코딩
pub fn decode(byte: u8) -> f32 {
    let sign = if byte & 128 != 0 { -1.0 } else { 1.0 };
    let value = (byte & 127) as f32 / 127.0;
    sign * ((1.0 + MU).powf(value) - 1.0) / MU
}

/// 값이 μ-law로 인코딩 가능한지 확인
pub fn is_encodable(value: f32) -> bool {
    value >= -1.0 && value <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mulaw_roundtrip() {
        let test_values = [-1.0, -0.5, 0.0, 0.5, 1.0];
        
        for &value in &test_values {
            let encoded = encode(value);
            let decoded = decode(encoded);
            assert!((value - decoded).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_mulaw_sign_preservation() {
        let positive = encode(0.5);
        assert!(positive < 128);
        
        let negative = encode(-0.5);
        assert!(negative >= 128);
    }
} 