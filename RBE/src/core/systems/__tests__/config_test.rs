//! # 구성 설정 단위테스트
//!
//! SystemConfiguration과 관련 config 구조체들의 기능 검증

use crate::core::systems::config::{
    SystemConfiguration, LearningParameters, HardwareConfiguration,
    OptimizationConfiguration, MemoryConfiguration, AdaptiveLearningRateConfig, LossWeights
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 시스템_구성_기본값_테스트() {
        let config = SystemConfiguration::default();
        
        assert_eq!(config.layer_sizes.len(), 3);
        assert_eq!(config.layer_sizes[0], (784, 256)); // MNIST input
        assert_eq!(config.layer_sizes[1], (256, 128));
        assert_eq!(config.layer_sizes[2], (128, 10));  // 10 classes
        
        println!("✅ 시스템 구성 기본값 테스트 통과");
        println!("   레이어 구조: {:?}", config.layer_sizes);
    }

    #[test]
    fn 커스텀_레이어_구성_테스트() {
        let mut config = SystemConfiguration::default();
        config.layer_sizes = vec![(512, 256), (256, 128), (128, 64), (64, 10)];
        
        assert_eq!(config.layer_sizes.len(), 4);
        assert_eq!(config.layer_sizes[0], (512, 256));
        assert_eq!(config.layer_sizes[3], (64, 10));
        
        println!("✅ 커스텀 레이어 구성 테스트 통과");
        println!("   사용자 정의 레이어: {:?}", config.layer_sizes);
    }

    #[test]
    fn 학습_매개변수_기본값_테스트() {
        let learning_params = LearningParameters::default();
        
        assert_eq!(learning_params.base_learning_rate, 0.001);
        assert_eq!(learning_params.batch_size, 32);
        assert_eq!(learning_params.max_epochs, 100);
        assert_eq!(learning_params.adaptive_lr_config.initial_lr, 0.001);
        assert_eq!(learning_params.loss_weights.data_loss_weight, 1.0);
        
        println!("✅ 학습 매개변수 기본값 테스트 통과");
        println!("   학습률: {}, 배치 크기: {}", learning_params.base_learning_rate, learning_params.batch_size);
    }

    #[test]
    fn 학습_매개변수_수정_테스트() {
        let mut learning_params = LearningParameters::default();
        learning_params.learning_rate = 0.01;
        learning_params.batch_size = 64;
        learning_params.epochs = 200;
        
        assert_eq!(learning_params.learning_rate, 0.01);
        assert_eq!(learning_params.batch_size, 64);
        assert_eq!(learning_params.epochs, 200);
        
        println!("✅ 학습 매개변수 수정 테스트 통과");
        println!("   새로운 학습률: {}, 새로운 배치 크기: {}", learning_params.learning_rate, learning_params.batch_size);
    }

    #[test]
    fn 하드웨어_구성_기본값_테스트() {
        let hardware_config = HardwareConfiguration::default();
        
        // CPU 스레드 수는 시스템에 따라 다를 수 있으므로 범위 검증
        assert!(hardware_config.num_cpu_threads >= 1);
        assert!(hardware_config.num_cpu_threads <= 128); // 실용적인 상한선
        assert_eq!(hardware_config.use_gpu, false);
        assert_eq!(hardware_config.gpu_device_id, 0);
        assert_eq!(hardware_config.enable_mixed_precision, false);
        
        println!("✅ 하드웨어 구성 기본값 테스트 통과");
        println!("   CPU 스레드: {}, GPU 사용: {}", hardware_config.num_cpu_threads, hardware_config.use_gpu);
    }

    #[test]
    fn GPU_활성화_구성_테스트() {
        let mut hardware_config = HardwareConfiguration::default();
        hardware_config.use_gpu = true;
        hardware_config.gpu_device_id = 1;
        hardware_config.enable_mixed_precision = true;
        
        assert_eq!(hardware_config.use_gpu, true);
        assert_eq!(hardware_config.gpu_device_id, 1);
        assert_eq!(hardware_config.enable_mixed_precision, true);
        
        println!("✅ GPU 활성화 구성 테스트 통과");
        println!("   GPU 장치 ID: {}, 혼합 정밀도: {}", hardware_config.gpu_device_id, hardware_config.enable_mixed_precision);
    }

    #[test]
    fn 최적화_구성_기본값_테스트() {
        let opt_config = OptimizationConfiguration::default();
        
        assert_eq!(opt_config.block_size_threshold, 64);
        assert_eq!(opt_config.enable_sparsity, true);
        assert_eq!(opt_config.sparsity_threshold, 0.01);
        assert_eq!(opt_config.enable_quantization, false);
        assert_eq!(opt_config.quantization_bits, 8);
        
        println!("✅ 최적화 구성 기본값 테스트 통과");
        println!("   블록 크기 임계값: {}, 희소성 활성화: {}", opt_config.block_size_threshold, opt_config.enable_sparsity);
    }

    #[test]
    fn 양자화_최적화_구성_테스트() {
        let mut opt_config = OptimizationConfiguration::default();
        opt_config.enable_quantization = true;
        opt_config.quantization_bits = 4;
        opt_config.sparsity_threshold = 0.05;
        
        assert_eq!(opt_config.enable_quantization, true);
        assert_eq!(opt_config.quantization_bits, 4);
        assert_eq!(opt_config.sparsity_threshold, 0.05);
        
        println!("✅ 양자화 최적화 구성 테스트 통과");
        println!("   양자화 비트: {}, 희소성 임계값: {}", opt_config.quantization_bits, opt_config.sparsity_threshold);
    }

    #[test]
    fn 메모리_구성_기본값_테스트() {
        let memory_config = MemoryConfiguration::default();
        
        assert_eq!(memory_config.cache_size_mb, 256);
        assert_eq!(memory_config.enable_memory_mapping, false);
        assert_eq!(memory_config.preload_weights, true);
        assert_eq!(memory_config.memory_pool_size_mb, 512);
        
        println!("✅ 메모리 구성 기본값 테스트 통과");
        println!("   캐시 크기: {}MB, 메모리 풀: {}MB", memory_config.cache_size_mb, memory_config.memory_pool_size_mb);
    }

    #[test]
    fn 대용량_메모리_구성_테스트() {
        let mut memory_config = MemoryConfiguration::default();
        memory_config.cache_size_mb = 1024;
        memory_config.enable_memory_mapping = true;
        memory_config.memory_pool_size_mb = 2048;
        
        assert_eq!(memory_config.cache_size_mb, 1024);
        assert_eq!(memory_config.enable_memory_mapping, true);
        assert_eq!(memory_config.memory_pool_size_mb, 2048);
        
        println!("✅ 대용량 메모리 구성 테스트 통과");
        println!("   대용량 캐시: {}MB, 메모리 매핑: {}", memory_config.cache_size_mb, memory_config.enable_memory_mapping);
    }

    #[test]
    fn 통합_시스템_구성_테스트() {
        let mut config = SystemConfiguration::default();
        
        // 모든 하위 구성 요소 수정
        config.learning_params.learning_rate = 0.005;
        config.hardware_config.use_gpu = true;
        config.optimization_config.enable_quantization = true;
        config.memory_config.cache_size_mb = 512;
        
        assert_eq!(config.learning_params.learning_rate, 0.005);
        assert_eq!(config.hardware_config.use_gpu, true);
        assert_eq!(config.optimization_config.enable_quantization, true);
        assert_eq!(config.memory_config.cache_size_mb, 512);
        
        println!("✅ 통합 시스템 구성 테스트 통과");
        println!("   모든 구성 요소가 올바르게 설정됨");
    }

    #[test]
    fn 구성_검증_로직_테스트() {
        let config = SystemConfiguration::default();
        
        // 레이어 크기 연결성 검증
        for i in 0..config.layer_sizes.len() - 1 {
            let current_layer = config.layer_sizes[i];
            let next_layer = config.layer_sizes[i + 1];
            
            // 현재 레이어의 출력이 다음 레이어의 입력과 같아야 함
            assert_eq!(current_layer.1, next_layer.0);
        }
        
        println!("✅ 구성 검증 로직 테스트 통과");
        println!("   레이어 크기 연결성 검증 완료");
    }

    #[test]
    fn 학습률_범위_검증_테스트() {
        let learning_params = LearningParameters::default();
        
        // 학습률이 적절한 범위에 있는지 확인
        assert!(learning_params.learning_rate > 0.0);
        assert!(learning_params.learning_rate <= 1.0);
        
        // 학습률 관련 매개변수 검증
        assert!(learning_params.learning_rate > 0.0 && learning_params.learning_rate <= 1.0);
        assert!(learning_params.base_learning_rate > 0.0 && learning_params.base_learning_rate <= 1.0);
        
        println!("✅ 학습률 범위 검증 테스트 통과");
        println!("   모든 매개변수가 유효한 범위 내에 있음");
    }

    #[test]
    fn 배치_크기_유효성_테스트() {
        let learning_params = LearningParameters::default();
        
        // 배치 크기가 유효한지 확인
        assert!(learning_params.batch_size > 0);
        assert!(learning_params.batch_size <= 1024); // 실용적인 상한선
        
        // 2의 거듭제곱인지 확인 (선택적, 하지만 성능에 유리)
        let is_power_of_two = learning_params.batch_size.is_power_of_two();
        
        println!("✅ 배치 크기 유효성 테스트 통과");
        println!("   배치 크기: {}, 2의 거듭제곱: {}", learning_params.batch_size, is_power_of_two);
    }

    #[test]
    fn 복제_가능성_테스트() {
        let original_config = SystemConfiguration::default();
        let cloned_config = original_config.clone();
        
        assert_eq!(original_config.layer_sizes, cloned_config.layer_sizes);
        assert_eq!(original_config.learning_params.learning_rate, cloned_config.learning_params.learning_rate);
        assert_eq!(original_config.hardware_config.num_cpu_threads, cloned_config.hardware_config.num_cpu_threads);
        
        println!("✅ 복제 가능성 테스트 통과");
        println!("   구성이 성공적으로 복제됨");
    }

    #[test]
    fn 직렬화_호환성_테스트() {
        let config = SystemConfiguration::default();
        
        // Debug trait 구현 확인 (간접적 직렬화 테스트)
        let debug_output = format!("{:?}", config);
        assert!(!debug_output.is_empty());
        
        println!("✅ 직렬화 호환성 테스트 통과");
        println!("   Debug 출력 길이: {} 문자", debug_output.len());
    }

    #[test]
    fn 극단적_구성값_처리_테스트() {
        let mut config = SystemConfiguration::default();
        
        // 극단적인 값들 설정
        config.learning_params.learning_rate = 1e-10; // 매우 작은 학습률
        config.learning_params.batch_size = 1; // 최소 배치 크기
        config.hardware_config.num_cpu_threads = 1; // 단일 스레드
        config.memory_config.cache_size_mb = 1; // 최소 캐시
        
        // 값이 올바르게 설정되었는지 확인
        assert_eq!(config.learning_params.learning_rate, 1e-10);
        assert_eq!(config.learning_params.batch_size, 1);
        assert_eq!(config.hardware_config.num_cpu_threads, 1);
        assert_eq!(config.memory_config.cache_size_mb, 1);
        
        println!("✅ 극단적 구성값 처리 테스트 통과");
        println!("   극단값들이 올바르게 처리됨");
    }
} 