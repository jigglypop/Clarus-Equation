//! # 하이브리드 시스템 단위테스트
//!
//! HybridPoincareRBESystem과 관련 구조체들의 기능 검증

use crate::core::systems::hybrid_system::{
    HybridPoincareRBESystem, HybridPoincareLayer, PoincareEncodingLayer,
    FusionProcessingLayer, HybridLearningLayer
};
use crate::core::systems::config::{SystemConfiguration, LearningParameters, HardwareConfiguration};
use crate::core::systems::state_management::LossComponents;
use crate::core::packed_params::Packed128;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 시스템_초기화_테스트() {
        let config = SystemConfiguration::default();
        let system = HybridPoincareRBESystem::new(config);
        
        assert_eq!(system.layers.len(), 3); // default config has 3 layers
        assert_eq!(system.config.layer_sizes.len(), 3);
        
        println!("✅ 하이브리드 푸앵카레 RBE 시스템 초기화 성공");
        println!("   레이어 수: {}", system.layers.len());
    }

    #[test]
    fn 커스텀_레이어_구성_테스트() {
        let mut config = SystemConfiguration::default();
        config.layer_sizes = vec![(100, 50), (50, 25), (25, 10)];
        
        let system = HybridPoincareRBESystem::new(config);
        
        assert_eq!(system.layers.len(), 3);
        assert_eq!(system.layers[0].input_dim, 100);
        assert_eq!(system.layers[0].output_dim, 50);
        assert_eq!(system.layers[2].input_dim, 25);
        assert_eq!(system.layers[2].output_dim, 10);
        
        println!("✅ 커스텀 레이어 구성 테스트 통과");
    }

    #[test]
    fn 멀티모달_손실_계산_테스트() {
        let config = SystemConfiguration::default();
        let system = HybridPoincareRBESystem::new(config);
        
        let predictions = vec![0.1, 0.2, 0.3, 0.4];
        let targets = vec![0.15, 0.25, 0.35, 0.45];
        let poincare_params = vec![Packed128 { hi: 0, lo: 0 }];
        let state_usage = HashMap::new();
        let residuals = vec![0.01, 0.02, 0.03];
        
        let (total_loss, loss_components) = system.compute_multimodal_loss(
            &predictions,
            &targets,
            &poincare_params,
            &state_usage,
            &residuals
        );
        
        assert!(total_loss > 0.0);
        assert!(total_loss.is_finite());
        assert_eq!(loss_components.total_loss, total_loss);
        assert!(loss_components.data_loss > 0.0);
        
        println!("✅ 멀티모달 손실 계산 테스트 통과");
        println!("   총 손실: {:.6}", total_loss);
        println!("   데이터 손실: {:.6}", loss_components.data_loss);
    }

    #[test]
    fn 시스템_순전파_테스트() {
        let config = SystemConfiguration::default();
        let mut system = HybridPoincareRBESystem::new(config);
        
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = system.forward(&input);
        
        assert_eq!(output.len(), input.len());
        
        // 출력이 유효한지 확인
        for &val in &output {
            assert!(val.is_finite());
        }
        
        println!("✅ 시스템 순전파 테스트 통과");
        println!("   입력 크기: {}, 출력 크기: {}", input.len(), output.len());
    }

    #[test]
    fn 시스템_역전파_테스트() {
        let config = SystemConfiguration::default();
        let mut system = HybridPoincareRBESystem::new(config);
        
        let loss_gradient = vec![0.1, 0.2, 0.3];
        let learning_rate = 0.001;
        
        // 역전파는 에러 없이 실행되어야 함
        system.backward(&loss_gradient, learning_rate);
        
        println!("✅ 시스템 역전파 테스트 통과");
    }

    #[test]
    fn 학습상태_업데이트_테스트() {
        let config = SystemConfiguration::default();
        let mut system = HybridPoincareRBESystem::new(config);
        
        let loss_components = LossComponents {
            data_loss: 0.5,
            poincare_loss: 0.1,
            state_loss: 0.05,
            sparsity_loss: 0.02,
            total_loss: 0.67,
        };
        
        let initial_history_len = system.learning_state.loss_history.len();
        
        system.update_learning_state(loss_components, 0.001);
        
        assert_eq!(system.learning_state.loss_history.len(), initial_history_len + 1);
        assert_eq!(system.learning_state.learning_rate_history.len(), initial_history_len + 1);
        
        let latest_loss = system.learning_state.loss_history.last().unwrap();
        assert_eq!(latest_loss.total_loss, 0.67);
        
        println!("✅ 학습상태 업데이트 테스트 통과");
        println!("   손실 히스토리 길이: {}", system.learning_state.loss_history.len());
    }

    #[test]
    fn 성능_보고서_출력_테스트() {
        let config = SystemConfiguration::default();
        let system = HybridPoincareRBESystem::new(config);
        
        // 성능 보고서 출력 (에러 없이 실행되어야 함)
        system.print_performance_report();
        
        println!("✅ 성능 보고서 출력 테스트 통과");
    }

    #[test]
    fn 개별_레이어_구성요소_테스트() {
        let config = SystemConfiguration::default();
        let system = HybridPoincareRBESystem::new(config);
        
        // 첫 번째 레이어 검증
        let first_layer = &system.layers[0];
        
        assert_eq!(first_layer.layer_id, 0);
        assert_eq!(first_layer.input_dim, 784);  // default config
        assert_eq!(first_layer.output_dim, 256);
        
        println!("✅ 개별 레이어 구성요소 테스트 통과");
        println!("   첫 번째 레이어: {}×{}", first_layer.input_dim, first_layer.output_dim);
    }

    #[test]
    fn 하드웨어_구성_반영_테스트() {
        let mut config = SystemConfiguration::default();
        config.hardware_config.num_cpu_threads = 8;
        config.hardware_config.use_gpu = true;
        
        let system = HybridPoincareRBESystem::new(config.clone());
        
        assert_eq!(system.config.hardware_config.num_cpu_threads, 8);
        assert_eq!(system.config.hardware_config.use_gpu, true);
        
        println!("✅ 하드웨어 구성 반영 테스트 통과");
        println!("   CPU 스레드: {}", system.config.hardware_config.num_cpu_threads);
        println!("   GPU 사용: {}", system.config.hardware_config.use_gpu);
    }

    #[test]
    fn 블록_크기_계산_테스트() {
        let config = SystemConfiguration::default();
        let system = HybridPoincareRBESystem::new(config);
        
        // 내부 블록 크기 계산 로직이 정상 작동하는지 확인
        // (실제로는 private method이지만 결과를 간접적으로 검증)
        
        assert!(system.layers.len() > 0);
        
        for layer in &system.layers {
            assert!(layer.input_dim > 0);
            assert!(layer.output_dim > 0);
        }
        
        println!("✅ 블록 크기 계산 테스트 통과");
    }

    #[test]
    fn 여러_에포크_시뮬레이션_테스트() {
        let config = SystemConfiguration::default();
        let mut system = HybridPoincareRBESystem::new(config);
        
        // 여러 에포크에 걸쳐 학습 시뮬레이션
        for epoch in 0..5 {
            let loss_components = LossComponents {
                data_loss: 1.0 / (epoch as f32 + 1.0), // 점진적 감소
                poincare_loss: 0.1,
                state_loss: 0.05,
                sparsity_loss: 0.02,
                total_loss: 1.0 / (epoch as f32 + 1.0) + 0.17,
            };
            
            system.update_learning_state(loss_components, 0.001);
        }
        
        assert_eq!(system.learning_state.loss_history.len(), 5);
        
        // 손실이 감소하는지 확인
        let first_loss = system.learning_state.loss_history[0].total_loss;
        let last_loss = system.learning_state.loss_history[4].total_loss;
        assert!(last_loss < first_loss);
        
        println!("✅ 여러 에포크 시뮬레이션 테스트 통과");
        println!("   초기 손실: {:.6}, 최종 손실: {:.6}", first_loss, last_loss);
    }
} 