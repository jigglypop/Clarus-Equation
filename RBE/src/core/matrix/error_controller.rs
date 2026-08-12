use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ErrorController {
    /// 전체 오차 임계값
    pub global_error_threshold: f32,
    /// 블록별 오차 맵
    pub block_errors: HashMap<(usize, usize), f32>,
    /// 오차 가중치
    pub error_weights: Vec<f32>,
}

impl ErrorController {
    /// 새로운 오차 제어기 생성
    pub fn new(error_threshold: f32) -> Self {
        Self {
            global_error_threshold: error_threshold,
            block_errors: HashMap::new(),
            error_weights: Vec::new(),
        }
    }
    
    /// 전체 오차 계산
    /// E_total = √(Σ w_i² E_i²)
    pub fn compute_total_error(&self) -> f32 {
        let mut weighted_error_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (block_id, &error) in &self.block_errors {
            let weight = self.get_block_weight(block_id);
            weighted_error_sum += weight * weight * error * error;
            total_weight += weight * weight;
        }
        
        if total_weight > 0.0 {
            (weighted_error_sum / total_weight).sqrt()
        } else {
            0.0
        }
    }
    
    /// 블록 가중치 계산 (블록 크기에 비례)
    fn get_block_weight(&self, _block_id: &(usize, usize)) -> f32 {
        1.0
    }
    
    /// 블록 오차 업데이트
    pub fn update_block_error(&mut self, block_id: (usize, usize), error: f32) {
        self.block_errors.insert(block_id, error);
    }
    
    /// 블록 분할 필요성 판단
    pub fn should_subdivide(&self, block_id: (usize, usize), current_level: usize) -> bool {
        if current_level >= 4 {
            return false; // 최대 깊이 도달
        }
        
        if let Some(&error) = self.block_errors.get(&block_id) {
            error > self.global_error_threshold
        } else {
            true // 오차 정보가 없으면 분할
        }
    }
} 