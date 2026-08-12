//! 상태 전이 그래프 구현

/// 상태 전이 그래프 (네트워크 토폴로지 관리)
#[derive(Debug, Clone)]
pub struct StateTransitionGraph {
    pub adjacency_matrix: Vec<Vec<f32>>,
    pub node_count: usize,
    pub edge_weights: Vec<f32>,
}

impl StateTransitionGraph {
    /// 새로운 상태 전이 그래프 생성
    pub fn new(node_count: usize) -> Self {
        let adjacency_matrix = vec![vec![0.0; node_count]; node_count];
        let edge_weights = vec![0.0; node_count * node_count];
        
        Self {
            adjacency_matrix,
            node_count,
            edge_weights,
        }
    }
    
    /// 엣지 가중치 설정
    pub fn set_edge_weight(&mut self, from: usize, to: usize, weight: f32) {
        if from < self.node_count && to < self.node_count {
            self.adjacency_matrix[from][to] = weight;
            self.edge_weights[from * self.node_count + to] = weight;
        }
    }
    
    /// 엣지 가중치 가져오기
    pub fn get_edge_weight(&self, from: usize, to: usize) -> f32 {
        if from < self.node_count && to < self.node_count {
            self.adjacency_matrix[from][to]
        } else {
            0.0
        }
    }
    
    /// 다음 상태 계산
    pub fn compute_next_state(&self, current_state: usize, input: f32) -> usize {
        let mut max_weight = f32::NEG_INFINITY;
        let mut next_state = current_state;
        
        for i in 0..self.node_count {
            let weight = self.get_edge_weight(current_state, i) * input;
            if weight > max_weight {
                max_weight = weight;
                next_state = i;
            }
        }
        
        next_state
    }
    
    /// 그래프 초기화 (기본 토폴로지)
    pub fn initialize_default(&mut self) {
        // 간단한 순환 그래프 생성
        for i in 0..self.node_count {
            let next = (i + 1) % self.node_count;
            self.set_edge_weight(i, next, 1.0);
        }
    }
    
    /// 완전 연결 그래프로 초기화
    pub fn initialize_fully_connected(&mut self, default_weight: f32) {
        for i in 0..self.node_count {
            for j in 0..self.node_count {
                if i != j {
                    self.set_edge_weight(i, j, default_weight);
                }
            }
        }
    }
    
    /// 그래프 정규화
    pub fn normalize_weights(&mut self) {
        for i in 0..self.node_count {
            let mut sum = 0.0;
            for j in 0..self.node_count {
                sum += self.adjacency_matrix[i][j].abs();
            }
            
            if sum > 0.0 {
                for j in 0..self.node_count {
                    self.adjacency_matrix[i][j] /= sum;
                    self.edge_weights[i * self.node_count + j] = self.adjacency_matrix[i][j];
                }
            }
        }
    }
} 