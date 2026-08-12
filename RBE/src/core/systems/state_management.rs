//! # 학습 상태 관리
//!
//! 7.4 학습 상태 관리
//! 하이브리드 학습 시스템의 상태 추적 및 관리

use crate::math::{RiemannianGeometry, StateTransitionGraph};
use std::collections::HashMap;

/// 7.4 학습 상태 관리
#[derive(Debug, Clone)]
pub struct LearningState {
    /// 현재 에포크
    pub current_epoch: usize,
    /// 현재 배치
    pub current_batch: usize,
    /// 학습률 히스토리
    pub learning_rate_history: Vec<f32>,
    /// 손실 히스토리
    pub loss_history: Vec<LossComponents>,
    /// 수렴 상태
    pub convergence_status: ConvergenceStatus,
}

/// 7.4.1 손실 구성요소
#[derive(Debug, Clone, Copy)]
pub struct LossComponents {
    /// 데이터 손실
    pub data_loss: f32,
    /// 푸앵카레 정규화 손실
    pub poincare_loss: f32,
    /// 상태 분포 손실
    pub state_loss: f32,
    /// 희소성 손실
    pub sparsity_loss: f32,
    /// 총 손실
    pub total_loss: f32,
}

/// 7.4.2 수렴 상태
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceStatus {
    /// 학습 시작
    Training,
    /// 수렴 중
    Converging,
    /// 수렴 완료
    Converged,
    /// 발산
    Diverged,
    /// 정체
    Stagnant,
}

/// 상태 관리자
#[derive(Debug, Clone)]
pub struct StateManager {
    /// 8가지 기저 함수 상태 분포
    pub state_distribution: [f32; 8],
    /// 상태 전이 그래프
    pub transition_graph: StateTransitionGraph,
    /// 상태 사용 히스토리
    pub usage_history: Vec<HashMap<usize, usize>>,
}

/// 파라미터 관리자
#[derive(Debug, Clone)]
pub struct ParameterManager {
    /// 연속 파라미터 값들
    pub continuous_params: Vec<(f32, f32)>, // (r, theta) 쌍들
    /// 리만 기하학 구조
    pub riemannian_geometry: RiemannianGeometry,
    /// 파라미터 업데이트 히스토리
    pub update_history: Vec<Vec<(f32, f32)>>,
}

// 구현들

impl LearningState {
    pub fn new() -> Self {
        Self {
            current_epoch: 0,
            current_batch: 0,
            learning_rate_history: Vec::new(),
            loss_history: Vec::new(),
            convergence_status: ConvergenceStatus::Training,
        }
    }
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            state_distribution: [0.125; 8], // 균등 분포로 초기화
            transition_graph: StateTransitionGraph::new(1024),
            usage_history: Vec::new(),
        }
    }
}

impl ParameterManager {
    pub fn new(input_dim: usize, output_dim: usize, _block_size: usize) -> Self {
        let num_params = (input_dim * output_dim) / 64; // 블록당 파라미터 개수 추정
        let continuous_params = vec![(0.5, 0.0); num_params]; // (r=0.5, theta=0.0) 초기값
        
        Self {
            continuous_params,
            riemannian_geometry: RiemannianGeometry,
            update_history: Vec::new(),
        }
    }
} 