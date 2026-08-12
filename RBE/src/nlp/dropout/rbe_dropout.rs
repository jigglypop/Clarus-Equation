use anyhow::Result;
use rand::{thread_rng, Rng, rngs::ThreadRng};
use std::sync::Arc;

/// RBEDropout - 푸앵카레 볼 기하학적 구조를 활용한 드롭아웃
/// 
/// 기존 드롭아웃과의 차이점:
/// 1. 비트 레벨 마스킹으로 메모리 효율성 극대화
/// 2. 푸앵카레 볼의 거리 기반 확률적 드롭
/// 3. 11비트 사이클과 동기화된 드롭아웃 패턴
#[derive(Clone)]
pub struct RBEDropout {
    /// 드롭아웃 확률 (0.0 ~ 1.0)
    pub dropout_prob: f32,
    
    /// 훈련 모드 여부
    pub training: bool,
    
    /// 푸앵카레 볼 중심으로부터의 거리 임계값
    /// 중심에서 멀수록 드롭아웃 확률 증가
    pub distance_threshold: f32,
    
    /// 11비트 사이클 마스크 (2048 상태)
    pub cycle_mask: u16,
    
    /// 캐시된 비트 마스크 패턴
    pub cached_masks: Option<Arc<Vec<u128>>>,
    
    /// 캐시된 RNG (성능 개선)
    rng: Option<ThreadRng>,
}

impl RBEDropout {
    pub fn new(dropout_prob: f32) -> Result<Self> {
        if dropout_prob < 0.0 || dropout_prob > 1.0 {
            return Err(anyhow::anyhow!("Dropout probability must be in [0, 1]"));
        }
        
        Ok(Self {
            dropout_prob,
            training: true,
            distance_threshold: 0.95, // 푸앵카레 볼 경계 근처
            cycle_mask: 0x7FF, // 11비트 마스크
            cached_masks: None,
            rng: Some(thread_rng()),
        })
    }
    
    /// 푸앵카레 볼 거리 기반 드롭아웃 마스크 생성
    pub fn generate_poincare_mask(&mut self, size: usize) -> Vec<bool> {
        let rng = self.rng.as_mut().unwrap();
        let mut mask = vec![false; size];
        
        for i in 0..size {
            // 인덱스를 푸앵카레 볼 좌표로 변환
            let normalized_pos = i as f32 / size as f32;
            let r = normalized_pos * self.distance_threshold;
            
            // 중심에서 멀수록 드롭아웃 확률 증가
            // 푸앵카레 메트릭: ds² = 4/(1-r²)² dr²
            let poincare_factor = 1.0 / (1.0 - r * r).max(0.01);
            let adjusted_prob = self.dropout_prob * poincare_factor.min(2.0);
            
            mask[i] = rng.gen::<f32>() < adjusted_prob;
        }
        
        mask
    }
    
    /// 11비트 사이클과 동기화된 드롭아웃
    pub fn cycle_synchronized_dropout(&mut self, input: &[f32], cycle_state: u16) -> Vec<f32> {
        if !self.training || self.dropout_prob == 0.0 {
            return input.to_vec();
        }
        
        let size = input.len();
        let mut output = vec![0.0; size];
        
        // 11비트 사이클 상태를 기반으로 드롭아웃 패턴 생성
        let cycle_pattern = (cycle_state & self.cycle_mask) as usize;
        let pattern_shift = cycle_pattern % 64; // 64비트 시프트
        
        // 비트 마스크 생성 (128비트 단위)
        let rng = self.rng.as_mut().unwrap();
        let scale = 1.0 / (1.0 - self.dropout_prob);
        
        for i in 0..size {
            // 각 요소에 대해 128비트 마스크의 특정 비트 확인
            let mask_idx = i / 128;
            let bit_idx = i % 128;
            
            // 사이클 패턴에 따른 비트 시프트
            let shifted_bit_idx = (bit_idx + pattern_shift) % 128;
            
            // 확률적 드롭 결정
            let drop = if shifted_bit_idx < (128.0 * self.dropout_prob) as usize {
                true
            } else {
                // 추가 랜덤성
                rng.gen::<f32>() < self.dropout_prob * 0.1
            };
            
            output[i] = if drop { 0.0 } else { input[i] * scale };
        }
        
        output
    }
    
    /// 표준 forward (비교 검증용)
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        if !self.training || self.dropout_prob == 0.0 {
            return input.to_vec();
        }
        
        let rng = self.rng.as_mut().unwrap();
        let scale = 1.0 / (1.0 - self.dropout_prob);
        
        input.iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.dropout_prob {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect()
    }
    
    /// 2D 텐서용 드롭아웃 (어텐션 스코어 등)
    pub fn forward_2d(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if !self.training || self.dropout_prob == 0.0 {
            return input.to_vec();
        }
        
        let poincare_mask = self.generate_poincare_mask(input.len() * input[0].len());
        let scale = 1.0 / (1.0 - self.dropout_prob);
        let mut idx = 0;
        
        input.iter()
            .map(|row| {
                row.iter()
                    .map(|&val| {
                        let result = if poincare_mask[idx] { 0.0 } else { val * scale };
                        idx += 1;
                        result
                    })
                    .collect()
            })
            .collect()
    }
    
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
} 