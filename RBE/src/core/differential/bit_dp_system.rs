//! # 비트 DP 시스템 (Bit Dynamic Programming System)
//!
//! 11비트 상태 전이에 대한 진정한 동적 프로그래밍 메모이제이션 테이블

use crate::packed_params::Packed128;
use super::cycle_system::{CycleState, HyperbolicFunction, DifferentialPhase};
use std::collections::HashMap;
use rayon::prelude::*;

/// **메모리 효율성을 위한 상수들**
const MAX_DP_STATES: usize = 128;  // 최대 상태 수 (메모리 제한)
const MAX_GRADIENT_LEVELS: usize = 8;  // 최대 그래디언트 레벨
const MAX_POSITIONS: usize = 1024;  // 최대 위치 수

/// **비트 DP 테이블** - 진정한 동적 프로그래밍 메모이제이션
#[derive(Debug, Clone)]
pub struct BitDPTable {
    /// DP 테이블: [상태][gradient_level][position] -> 최적값
    dp_table: Vec<Vec<Vec<f32>>>, // [2048 states][16 gradient_levels][positions] 
    /// 상태 전이 테이블: [from_state][to_state] -> 전이 비용
    transition_table: [[f32; MAX_DP_STATES]; MAX_DP_STATES],
    /// 최적 부분구조 추적
    optimal_substructure: HashMap<(u16, u8, usize), (f32, u16)>, // (state, grad_level, pos) -> (value, next_state)
    /// 중복 부분문제 캐시
    subproblem_cache: HashMap<u64, f32>, // problem_hash -> optimal_value
    /// DP 테이블 크기
    state_count: usize,
    gradient_levels: usize,
    max_positions: usize,
}

/// **비트 DP 문제 정의**
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BitDPProblem {
    /// 현재 상태 (11비트)
    pub current_state: u16,
    /// 그래디언트 레벨 (0-15)
    pub gradient_level: u8,
    /// 현재 위치
    pub position: usize,
    /// 남은 단계 수
    pub remaining_steps: u8,
}

/// **DP 최적화 결과**
#[derive(Debug, Clone)]
pub struct DPOptimizationResult {
    /// 최적 값
    pub optimal_value: f32,
    /// 최적 경로
    pub optimal_path: Vec<u16>,
    /// 상태 전이 시퀀스
    pub transition_sequence: Vec<(u16, u16, f32)>, // (from, to, cost)
    /// 총 계산 단계
    pub total_steps: usize,
}

/// **병렬 DP 처리기**
#[derive(Debug, Clone)]
pub struct ParallelDPProcessor {
    /// 스레드별 DP 테이블
    thread_local_tables: Vec<BitDPTable>,
    /// 전역 결과 병합 캐시
    global_merge_cache: HashMap<u64, f32>,
    /// 병렬 처리 설정
    num_threads: usize,
    chunk_size: usize,
}

impl BitDPTable {
    /// 새로운 비트 DP 테이블 생성 (메모리 효율적)
    pub fn new(state_count: usize, gradient_levels: usize, max_positions: usize) -> Self {
        // **메모리 사용량 제한** - 최대 64MB로 제한
        let safe_state_count = state_count.min(128);  // 2048 -> 128
        let safe_gradient_levels = gradient_levels.min(8);  // 16 -> 8
        let safe_max_positions = max_positions.min(1024);  // 무제한 -> 1024
        
        // 작은 DP 테이블 초기화 (메모리 효율적)
        let mut dp_table = vec![vec![vec![f32::INFINITY; safe_max_positions]; safe_gradient_levels]; safe_state_count];
        
        // 기저 상태 초기화 (position = 0)
        for state in 0..safe_state_count {
            for grad_level in 0..safe_gradient_levels {
                if safe_max_positions > 0 {
                    dp_table[state][grad_level][0] = 0.0; // 시작점은 비용 0
                }
            }
        }
        
        Self {
            dp_table,
            transition_table: [[f32::INFINITY; MAX_DP_STATES]; MAX_DP_STATES],
            optimal_substructure: HashMap::new(),
            subproblem_cache: HashMap::new(),
            state_count: safe_state_count,
            gradient_levels: safe_gradient_levels,
            max_positions: safe_max_positions,
        }
    }
    
    /// **핵심: 비트 DP 최적화** (진정한 동적 프로그래밍)
    pub fn optimize_bit_sequence(
        &mut self,
        problem: &BitDPProblem,
        packed: &Packed128,
        errors: &[f32],
        rows: usize,
        cols: usize,
    ) -> DPOptimizationResult {
        // 1. 문제 해시 계산 (중복 부분문제 확인)
        let problem_hash = self.compute_problem_hash(problem, packed);
        if let Some(&cached_value) = self.subproblem_cache.get(&problem_hash) {
            return DPOptimizationResult {
                optimal_value: cached_value,
                optimal_path: vec![problem.current_state],
                transition_sequence: vec![],
                total_steps: 1,
            };
        }
        
        // 2. 기저 조건 확인
        if problem.remaining_steps == 0 || problem.position >= self.max_positions {
            let final_value = self.compute_final_cost(problem, packed, errors);
            self.subproblem_cache.insert(problem_hash, final_value);
            return DPOptimizationResult {
                optimal_value: final_value,
                optimal_path: vec![problem.current_state],
                transition_sequence: vec![],
                total_steps: 1,
            };
        }
        
        // 3. **최적 부분구조 활용**: 모든 가능한 다음 상태 탐색
        let mut best_value = f32::INFINITY;
        let mut best_next_state = problem.current_state;
        let mut best_path = vec![problem.current_state];
        let mut best_transitions = vec![];
        let mut total_steps = 1;
        
        // 가능한 모든 상태 전이 탐색 (11비트 = 2048 상태)
        for next_state in 0..self.state_count {
            let next_state_u16 = next_state as u16;
            
            // 전이 비용 계산
            let transition_cost = self.compute_transition_cost(
                problem.current_state, 
                next_state_u16, 
                problem.gradient_level,
                packed,
                errors,
                problem.position
            );
            
            // 불가능한 전이는 스킵
            if transition_cost == f32::INFINITY {
                continue;
            }
            
            // 다음 단계 문제 정의
            let next_problem = BitDPProblem {
                current_state: next_state_u16,
                gradient_level: self.update_gradient_level(problem.gradient_level, &errors[problem.position.min(errors.len() - 1)]),
                position: problem.position + 1,
                remaining_steps: problem.remaining_steps - 1,
            };
            
            // **재귀적 DP 호출** (최적 부분구조)
            let sub_result = self.optimize_bit_sequence(&next_problem, packed, errors, rows, cols);
            let total_cost = transition_cost + sub_result.optimal_value;
            
            // 최적값 업데이트
            if total_cost < best_value {
                best_value = total_cost;
                best_next_state = next_state_u16;
                best_path = [vec![problem.current_state], sub_result.optimal_path].concat();
                best_transitions = [
                    vec![(problem.current_state, next_state_u16, transition_cost)],
                    sub_result.transition_sequence
                ].concat();
                total_steps = sub_result.total_steps + 1;
            }
        }
        
        // 4. DP 테이블 업데이트
        if problem.position < self.max_positions && 
           (problem.gradient_level as usize) < self.gradient_levels &&
           (problem.current_state as usize) < self.state_count {
            self.dp_table[problem.current_state as usize][problem.gradient_level as usize][problem.position] = best_value;
        }
        
        // 5. 최적 부분구조 저장
        let key = (problem.current_state, problem.gradient_level, problem.position);
        self.optimal_substructure.insert(key, (best_value, best_next_state));
        
        // 6. 중복 부분문제 캐시 업데이트
        self.subproblem_cache.insert(problem_hash, best_value);
        
        DPOptimizationResult {
            optimal_value: best_value,
            optimal_path: best_path,
            transition_sequence: best_transitions,
            total_steps,
        }
    }
    
    /// 전이 비용 계산 (상태간 전이 비용)
    fn compute_transition_cost(
        &mut self,
        from_state: u16,
        to_state: u16,
        gradient_level: u8,
        packed: &Packed128,
        errors: &[f32],
        position: usize,
    ) -> f32 {
        // 캐시된 전이 비용 확인
        if self.transition_table[from_state as usize][to_state as usize] != f32::INFINITY {
            return self.transition_table[from_state as usize][to_state as usize];
        }
        
        // 전이 가능성 검사
        let hamming_distance = (from_state ^ to_state).count_ones();
        if hamming_distance > 3 {
            // 3비트 이상 변화는 불가능한 전이
            self.transition_table[from_state as usize][to_state as usize] = f32::INFINITY;
            return f32::INFINITY;
        }
        
        // 실제 전이 비용 계산
        let error = if position < errors.len() { errors[position] } else { 0.0 };
        let gradient_weight = (gradient_level as f32 + 1.0) / 16.0;
        
        // 상태 차이 기반 비용
        let state_diff_cost = hamming_distance as f32 * 0.1;
        
        // 그래디언트 정렬 비용
        let gradient_alignment_cost = (error.abs() * gradient_weight - 0.5).abs();
        
        // 수치적 안정성 비용
        let stability_cost = if from_state == to_state { 0.0 } else { 0.01 };
        
        let total_cost = state_diff_cost + gradient_alignment_cost + stability_cost;
        
        // 전이 테이블 캐시 업데이트
        self.transition_table[from_state as usize][to_state as usize] = total_cost;
        
        total_cost
    }
    
    /// 최종 비용 계산
    fn compute_final_cost(&self, problem: &BitDPProblem, packed: &Packed128, errors: &[f32]) -> f32 {
        let position = problem.position.min(errors.len() - 1);
        let error = errors[position];
        
        // 상태-에러 정렬도 계산
        let state_error_alignment = (problem.current_state as f32 / 2048.0 - error.abs()).abs();
        
        // 최종 비용
        state_error_alignment + 0.01 * problem.gradient_level as f32
    }
    
    /// 그래디언트 레벨 업데이트
    fn update_gradient_level(&self, current_level: u8, error: &f32) -> u8 {
        let error_magnitude = error.abs();
        
        if error_magnitude > 0.1 {
            15 // 최대 레벨
        } else if error_magnitude > 0.01 {
            ((error_magnitude * 150.0) as u8).min(15)
        } else {
            (current_level / 2).max(1) // 감소
        }
    }
    
    /// 문제 해시 계산 (중복 부분문제 식별)
    fn compute_problem_hash(&self, problem: &BitDPProblem, packed: &Packed128) -> u64 {
        let mut hash = 0u64;
        hash ^= (problem.current_state as u64) << 48;
        hash ^= (problem.gradient_level as u64) << 40;
        hash ^= (problem.position as u64) << 32;
        hash ^= (problem.remaining_steps as u64) << 24;
        hash ^= (packed.hi >> 32) & 0xFFFFFF; // 상위 24비트만 사용
        hash
    }
    
    /// DP 테이블 클리어 (메모리 관리)
    pub fn clear_cache(&mut self) {
        self.subproblem_cache.clear();
        self.optimal_substructure.clear();
        
        // DP 테이블 재초기화 (선택적)
        if self.subproblem_cache.len() > 10000 {
            for state in 0..self.state_count {
                for grad_level in 0..self.gradient_levels {
                    for pos in 0..self.max_positions {
                        self.dp_table[state][grad_level][pos] = f32::INFINITY;
                    }
                }
            }
            
            // 기저 상태 재설정
            for state in 0..self.state_count {
                for grad_level in 0..self.gradient_levels {
                    self.dp_table[state][grad_level][0] = 0.0;
                }
            }
        }
    }
    
    /// DP 통계 반환
    pub fn get_dp_stats(&self) -> (usize, usize, usize) {
        (
            self.subproblem_cache.len(),
            self.optimal_substructure.len(),
            self.dp_table.iter().map(|state_table| 
                state_table.iter().map(|grad_table| 
                    grad_table.iter().filter(|&&x| x != f32::INFINITY).count()
                ).sum::<usize>()
            ).sum::<usize>()
        )
    }
}

impl ParallelDPProcessor {
    /// 새로운 병렬 DP 처리기
    pub fn new(num_threads: usize, state_count: usize, gradient_levels: usize, max_positions: usize) -> Self {
        let thread_local_tables = (0..num_threads)
            .map(|_| BitDPTable::new(state_count, gradient_levels, max_positions))
            .collect();
            
        Self {
            thread_local_tables,
            global_merge_cache: HashMap::new(),
            num_threads,
            chunk_size: max_positions / num_threads + 1,
        }
    }
    
    /// **병렬 비트 DP 최적화** (rayon 기반)
    pub fn parallel_optimize(
        &mut self,
        problems: &[BitDPProblem],
        packed_params: &[Packed128],
        error_batches: &[Vec<f32>],
        rows: usize,
        cols: usize,
    ) -> Vec<DPOptimizationResult> {
        // 문제들을 청크로 분할
        let problem_chunks: Vec<_> = problems.chunks(self.chunk_size).collect();
        
        // 병렬 처리
        let results: Vec<Vec<DPOptimizationResult>> = problem_chunks
            .into_par_iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let thread_id = chunk_idx % self.num_threads;
                let mut local_table = self.thread_local_tables[thread_id].clone();
                
                // 청크 내 문제들 순차 처리
                chunk.iter().enumerate().map(|(local_idx, problem)| {
                    let global_idx = chunk_idx * self.chunk_size + local_idx;
                    let packed = if global_idx < packed_params.len() { 
                        &packed_params[global_idx] 
                    } else { 
                        &packed_params[0] 
                    };
                    let errors = if global_idx < error_batches.len() { 
                        &error_batches[global_idx] 
                    } else { 
                        &error_batches[0] 
                    };
                    
                    local_table.optimize_bit_sequence(problem, packed, errors, rows, cols)
                }).collect()
            })
            .collect();
        
        // 결과 병합
        results.into_iter().flatten().collect()
    }
    
    /// 병렬 캐시 병합
    pub fn merge_caches(&mut self) {
        // 각 스레드의 캐시를 전역 캐시로 병합
        for table in &self.thread_local_tables {
            for (&key, &value) in &table.subproblem_cache {
                self.global_merge_cache.entry(key)
                    .and_modify(|existing| *existing = existing.min(value))
                    .or_insert(value);
            }
        }
        
        // 전역 캐시를 각 스레드로 배포
        for table in &mut self.thread_local_tables {
            for (&key, &value) in &self.global_merge_cache {
                table.subproblem_cache.insert(key, value);
            }
        }
    }
    
    /// 캐시 통계
    pub fn get_parallel_stats(&self) -> (usize, usize, Vec<(usize, usize, usize)>) {
        let thread_stats: Vec<_> = self.thread_local_tables.iter()
            .map(|table| table.get_dp_stats())
            .collect();
            
        (
            self.global_merge_cache.len(),
            self.num_threads,
            thread_stats,
        )
    }
} 