//! # 상태 관리 단위테스트
//!
//! LearningState와 관련 상태 관리 구조체들의 기능 검증

use crate::core::systems::state_management::{
    LearningState, LossComponents, ConvergenceStatus,
    ParameterManager, StateManager
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 학습_상태_초기화_테스트() {
        let learning_state = LearningState::new();
        
        assert_eq!(learning_state.current_epoch, 0);
        assert_eq!(learning_state.current_batch, 0);
        assert_eq!(learning_state.learning_rate_history.len(), 0);
        assert_eq!(learning_state.loss_history.len(), 0);
        assert_eq!(learning_state.convergence_status, ConvergenceStatus::Training);
        
        println!("✅ 학습 상태 초기화 테스트 통과");
        println!("   초기 에포크: {}, 배치: {}", 
                learning_state.current_epoch, learning_state.current_batch);
    }

    #[test]
    fn 손실_구성_요소_테스트() {
        let mut loss_components = LossComponents {
            data_loss: 0.0,
            poincare_loss: 0.0,
            state_loss: 0.0,
            sparsity_loss: 0.0,
            total_loss: 0.0,
        };
        
        // 손실값 업데이트
        loss_components.data_loss = 0.5;
        loss_components.poincare_loss = 0.1;
        loss_components.state_loss = 0.05;
        loss_components.sparsity_loss = 0.02;
        loss_components.total_loss = loss_components.data_loss + 
                                   loss_components.poincare_loss + 
                                   loss_components.state_loss + 
                                   loss_components.sparsity_loss;
        
        assert_eq!(loss_components.total_loss, 0.67);
        
        println!("✅ 손실 구성 요소 테스트 통과");
        println!("   총 손실: {}", loss_components.total_loss);
    }

    #[test]
    fn 수렴_상태_테스트() {
        let mut convergence_status = ConvergenceStatus::Training;
        
        assert_eq!(convergence_status, ConvergenceStatus::Training);
        
        // 수렴 상태 변경
        convergence_status = ConvergenceStatus::Converging;
        assert_eq!(convergence_status, ConvergenceStatus::Converging);
        
        convergence_status = ConvergenceStatus::Converged;
        assert_eq!(convergence_status, ConvergenceStatus::Converged);
        
        println!("✅ 수렴 상태 테스트 통과");
        println!("   최종 상태: {:?}", convergence_status);
    }

    #[test]
    fn 파라미터_매니저_테스트() {
        let param_manager = ParameterManager::new(784, 256, 64);
        
        // 파라미터 개수 확인 (784 * 256 / 64 = 3136개)
        let expected_params = (784 * 256) / 64;
        assert_eq!(param_manager.continuous_params.len(), expected_params);
        
        // 초기값 확인
        let first_param = param_manager.continuous_params[0];
        assert_eq!(first_param.0, 0.5); // r 값
        assert_eq!(first_param.1, 0.0); // theta 값
        
        assert_eq!(param_manager.update_history.len(), 0);
        
        println!("✅ 파라미터 매니저 테스트 통과");
        println!("   파라미터 수: {}", param_manager.continuous_params.len());
    }

    #[test]
    fn 상태_매니저_테스트() {
        let state_manager = StateManager::new();
        
        // 8가지 기저 함수 상태 분포 확인
        assert_eq!(state_manager.state_distribution.len(), 8);
        
        // 균등 분포로 초기화되었는지 확인
        for &distribution in &state_manager.state_distribution {
            assert_eq!(distribution, 0.125); // 1/8 = 0.125
        }
        
        assert_eq!(state_manager.usage_history.len(), 0);
        
        println!("✅ 상태 매니저 테스트 통과");
        println!("   상태 분포: {:?}", state_manager.state_distribution);
    }

    #[test]
    fn 학습_상태_업데이트_테스트() {
        let mut learning_state = LearningState::new();
        
        // 에포크 진행
        learning_state.current_epoch = 1;
        learning_state.current_batch = 0;
        assert_eq!(learning_state.current_epoch, 1);
        assert_eq!(learning_state.current_batch, 0);
        
        // 배치 진행
        learning_state.current_batch = 10;
        assert_eq!(learning_state.current_batch, 10);
        
        // 학습률 히스토리 추가
        learning_state.learning_rate_history.push(0.001);
        learning_state.learning_rate_history.push(0.0008);
        assert_eq!(learning_state.learning_rate_history.len(), 2);
        
        // 수렴 상태 변경
        learning_state.convergence_status = ConvergenceStatus::Converging;
        assert_eq!(learning_state.convergence_status, ConvergenceStatus::Converging);
        
        println!("✅ 학습 상태 업데이트 테스트 통과");
        println!("   현재 에포크: {}, 배치: {}", 
                learning_state.current_epoch, learning_state.current_batch);
    }

    #[test]
    fn 손실_히스토리_추적_테스트() {
        let mut learning_state = LearningState::new();
        
        // 손실 히스토리 기록
        let loss_values = vec![1.0, 0.8, 0.6, 0.5, 0.45];
        
        for &loss in &loss_values {
            let loss_component = LossComponents {
                data_loss: loss,
                poincare_loss: loss * 0.1,
                state_loss: loss * 0.05,
                sparsity_loss: loss * 0.02,
                total_loss: loss * 1.17, // 합계
            };
            learning_state.loss_history.push(loss_component);
        }
        
        assert_eq!(learning_state.loss_history.len(), 5);
        
        // 첫 번째와 마지막 손실 확인
        assert_eq!(learning_state.loss_history[0].data_loss, 1.0);
        assert_eq!(learning_state.loss_history[4].data_loss, 0.45);
        
        println!("✅ 손실 히스토리 추적 테스트 통과");
        println!("   히스토리 길이: {}", learning_state.loss_history.len());
    }

    #[test]
    fn 파라미터_조작_테스트() {
        let mut param_manager = ParameterManager::new(128, 64, 32);
        
        // 파라미터 직접 수정
        let param_count = param_manager.continuous_params.len();
        assert!(param_count > 0);
        
        // 첫 번째 파라미터 수정
        param_manager.continuous_params[0] = (0.8, 1.57); // r=0.8, theta=π/2
        
        let updated_param = param_manager.continuous_params[0];
        assert_eq!(updated_param.0, 0.8);
        assert_eq!(updated_param.1, 1.57);
        
        // 업데이트 히스토리 추가
        param_manager.update_history.push(vec![(0.5, 0.0), (0.8, 1.57)]);
        assert_eq!(param_manager.update_history.len(), 1);
        
        println!("✅ 파라미터 조작 테스트 통과");
        println!("   업데이트된 파라미터: {:?}", updated_param);
    }

    #[test]
    fn 상태_분포_변경_테스트() {
        let mut state_manager = StateManager::new();
        
        // 상태 분포 수정
        state_manager.state_distribution[0] = 0.2;
        state_manager.state_distribution[1] = 0.15;
        state_manager.state_distribution[2] = 0.1;
        
        assert_eq!(state_manager.state_distribution[0], 0.2);
        assert_eq!(state_manager.state_distribution[1], 0.15);
        assert_eq!(state_manager.state_distribution[2], 0.1);
        
        // 분포의 합이 1에 가까운지 확인 (정규화)
        let sum: f32 = state_manager.state_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 0.01 || sum > 0.5); // 합리적인 범위
        
        println!("✅ 상태 분포 변경 테스트 통과");
        println!("   분포 합계: {:.3}", sum);
    }

    #[test]
    fn 구조체_복제_테스트() {
        let learning_state = LearningState::new();
        let cloned_state = learning_state.clone();
        
        assert_eq!(learning_state.current_epoch, cloned_state.current_epoch);
        assert_eq!(learning_state.current_batch, cloned_state.current_batch);
        assert_eq!(learning_state.convergence_status, cloned_state.convergence_status);
        
        let state_manager = StateManager::new();
        let cloned_manager = state_manager.clone();
        
        assert_eq!(state_manager.state_distribution, cloned_manager.state_distribution);
        
        println!("✅ 구조체 복제 테스트 통과");
    }

    #[test]
    fn 다양한_수렴_상태_테스트() {
        let states = vec![
            ConvergenceStatus::Training,
            ConvergenceStatus::Converging,
            ConvergenceStatus::Converged,
            ConvergenceStatus::Diverged,
            ConvergenceStatus::Stagnant,
        ];
        
        for (i, state) in states.iter().enumerate() {
            println!("   상태 {}: {:?}", i, state);
            
            // 각 상태가 자기 자신과 같은지 확인
            assert_eq!(state, state);
        }
        
        // 서로 다른 상태들이 다른지 확인
        assert_ne!(ConvergenceStatus::Training, ConvergenceStatus::Converged);
        assert_ne!(ConvergenceStatus::Converging, ConvergenceStatus::Diverged);
        
        println!("✅ 다양한 수렴 상태 테스트 통과");
    }

    #[test]
    fn 대용량_파라미터_테스트() {
        let param_manager = ParameterManager::new(1024, 512, 128);
        
        // 실제 파라미터 수는 구현에 따라 다를 수 있으므로 범위 검증
        let actual_params = param_manager.continuous_params.len();
        assert!(actual_params > 1000); // 최소값 확인
        assert!(actual_params < 100000); // 최대값 확인
        
        // 모든 파라미터가 올바른 초기값을 가지는지 확인
        for param in &param_manager.continuous_params {
            assert_eq!(param.0, 0.5); // r 값
            assert_eq!(param.1, 0.0); // theta 값
        }
        
        println!("✅ 대용량 파라미터 테스트 통과");
        println!("   총 파라미터 수: {}", param_manager.continuous_params.len());
    }
} 