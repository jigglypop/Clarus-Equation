use crate::core::optimizers::{
    OptimizerConfig, AdamConfig, RiemannianAdamConfig, OptimizerType
};
use crate::core::optimizers::config::LearningRateSchedule;
use crate::packed_params::TransformType;

#[test]
fn 옵티마이저구성_기본값_테스트() {
    let config = OptimizerConfig::default();
    
    // Adam 구성 기본값 확인
    assert_eq!(config.adam.beta1, 0.9, "Adam beta1 기본값");
    assert_eq!(config.adam.beta2, 0.999, "Adam beta2 기본값");
    assert_eq!(config.adam.epsilon, 1e-8, "Adam epsilon 기본값");
    
    // Riemannian Adam 구성 기본값 확인
    assert_eq!(config.riemannian_adam.beta1, 0.9, "Riemannian Adam beta1 기본값");
    assert_eq!(config.riemannian_adam.beta2, 0.999, "Riemannian Adam beta2 기본값");
    assert_eq!(config.riemannian_adam.epsilon, 1e-8, "Riemannian Adam epsilon 기본값");
    assert_eq!(config.riemannian_adam.metric_regularization, 1e-4, "메트릭 정규화 기본값");
    
    // 기타 설정 기본값 확인
    assert_eq!(config.learning_rate, 0.001, "학습률 기본값");
    assert!(matches!(config.lr_schedule, LearningRateSchedule::Constant), "학습률 스케줄 기본값");
    assert_eq!(config.gradient_clipping, Some(1.0), "그래디언트 클리핑 기본값");
    assert_eq!(config.weight_decay, 0.0, "가중치 감소 기본값");
    
    println!("✅ 옵티마이저 구성 기본값 테스트 통과");
}

#[test]
fn Adam구성_기본값_테스트() {
    let adam_config = AdamConfig::default();
    
    assert_eq!(adam_config.beta1, 0.9, "Adam beta1 기본값");
    assert_eq!(adam_config.beta2, 0.999, "Adam beta2 기본값");
    assert_eq!(adam_config.epsilon, 1e-8, "Adam epsilon 기본값");
    
    println!("✅ Adam 구성 기본값 테스트 통과");
}

#[test]
fn Riemannian_Adam구성_기본값_테스트() {
    let riemannian_config = RiemannianAdamConfig::default();
    
    assert_eq!(riemannian_config.beta1, 0.9, "Riemannian Adam beta1 기본값");
    assert_eq!(riemannian_config.beta2, 0.999, "Riemannian Adam beta2 기본값");
    assert_eq!(riemannian_config.epsilon, 1e-8, "Riemannian Adam epsilon 기본값");
    assert_eq!(riemannian_config.metric_regularization, 1e-4, "메트릭 정규화 기본값");
    
    println!("✅ Riemannian Adam 구성 기본값 테스트 통과");
}

#[test]
fn 옵티마이저구성_new_테스트() {
    let config1 = OptimizerConfig::new();
    let config2 = OptimizerConfig::default();
    
    // new()와 default()가 동일한 결과를 반환하는지 확인
    assert_eq!(config1.learning_rate, config2.learning_rate, "new()와 default() 동일성");
    assert_eq!(config1.adam.beta1, config2.adam.beta1, "Adam 구성 동일성");
    assert_eq!(config1.riemannian_adam.beta1, config2.riemannian_adam.beta1, "Riemannian Adam 구성 동일성");
    
    println!("✅ 옵티마이저 구성 new 테스트 통과");
}

#[test]
fn 옵티마이저구성_체이닝_테스트() {
    let adam_config = AdamConfig {
        beta1: 0.95,
        beta2: 0.9999,
        epsilon: 1e-7,
    };
    
    let riemannian_config = RiemannianAdamConfig {
        beta1: 0.92,
        beta2: 0.9998,
        epsilon: 1e-6,
        metric_regularization: 1e-3,
    };
    
    let config = OptimizerConfig::new()
        .with_adam_config(adam_config.clone())
        .with_riemannian_adam_config(riemannian_config.clone())
        .with_learning_rate(0.01)
        .with_lr_schedule(LearningRateSchedule::ExponentialDecay { 
            decay_rate: 0.95, 
            decay_steps: 1000 
        })
        .with_gradient_clipping(Some(0.5))
        .with_weight_decay(1e-4);
    
    // 설정된 값들 확인
    assert_eq!(config.adam.beta1, 0.95, "Adam beta1 커스텀 설정");
    assert_eq!(config.adam.beta2, 0.9999, "Adam beta2 커스텀 설정");
    assert_eq!(config.adam.epsilon, 1e-7, "Adam epsilon 커스텀 설정");
    
    assert_eq!(config.riemannian_adam.beta1, 0.92, "Riemannian Adam beta1 커스텀 설정");
    assert_eq!(config.riemannian_adam.metric_regularization, 1e-3, "메트릭 정규화 커스텀 설정");
    
    assert_eq!(config.learning_rate, 0.01, "학습률 커스텀 설정");
    assert!(matches!(config.lr_schedule, LearningRateSchedule::ExponentialDecay { .. }), "학습률 스케줄 커스텀 설정");
    assert_eq!(config.gradient_clipping, Some(0.5), "그래디언트 클리핑 커스텀 설정");
    assert_eq!(config.weight_decay, 1e-4, "가중치 감소 커스텀 설정");
    
    println!("✅ 옵티마이저 구성 체이닝 테스트 통과");
}

#[test]
fn 학습률스케줄_열거형_테스트() {
    let schedules = vec![
        LearningRateSchedule::Constant,
        LearningRateSchedule::ExponentialDecay { decay_rate: 0.9, decay_steps: 100 },
        LearningRateSchedule::CosineAnnealing { min_lr: 1e-6, max_lr: 1e-2, period: 1000 },
        LearningRateSchedule::StepDecay { step_size: 500, gamma: 0.5 },
        LearningRateSchedule::Adaptive { patience: 10, factor: 0.8 },
    ];
    
    for schedule in &schedules {
        // Debug 출력 확인
        let debug_str = format!("{:?}", schedule);
        assert!(!debug_str.is_empty(), "Debug 출력 가능");
        
        // Clone 확인
        let cloned = schedule.clone();
        assert!(matches!(cloned, LearningRateSchedule::Constant) || 
                 matches!(cloned, LearningRateSchedule::ExponentialDecay { .. }) ||
                 matches!(cloned, LearningRateSchedule::CosineAnnealing { .. }) ||
                 matches!(cloned, LearningRateSchedule::StepDecay { .. }) ||
                 matches!(cloned, LearningRateSchedule::Adaptive { .. }), "Clone 동작 확인");
    }
    
    println!("✅ 학습률 스케줄 열거형 테스트 통과");
    println!("   총 {} 개의 스케줄 타입 확인됨", schedules.len());
}

#[test]
fn 구성_복제_테스트() {
    let original_config = OptimizerConfig::new()
        .with_learning_rate(0.005)
        .with_weight_decay(1e-5);
    
    let cloned_config = original_config.clone();
    
    // 원본과 복제본이 같은 값을 가지는지 확인
    assert_eq!(original_config.learning_rate, cloned_config.learning_rate, "학습률 복제 확인");
    assert_eq!(original_config.weight_decay, cloned_config.weight_decay, "가중치 감소 복제 확인");
    assert_eq!(original_config.adam.beta1, cloned_config.adam.beta1, "Adam 구성 복제 확인");
    assert_eq!(original_config.riemannian_adam.metric_regularization, 
               cloned_config.riemannian_adam.metric_regularization, "Riemannian Adam 구성 복제 확인");
    
    println!("✅ 구성 복제 테스트 통과");
}

#[test]
fn 그래디언트클리핑_설정_테스트() {
    // 클리핑 비활성화
    let no_clipping_config = OptimizerConfig::new()
        .with_gradient_clipping(None);
    assert_eq!(no_clipping_config.gradient_clipping, None, "그래디언트 클리핑 비활성화");
    
    // 다양한 클리핑 값
    let clipping_values = [0.1, 0.5, 1.0, 2.0, 5.0];
    for &clip_value in &clipping_values {
        let config = OptimizerConfig::new()
            .with_gradient_clipping(Some(clip_value));
        assert_eq!(config.gradient_clipping, Some(clip_value), "그래디언트 클리핑 값 설정");
    }
    
    println!("✅ 그래디언트 클리핑 설정 테스트 통과");
    println!("   테스트된 클리핑 값: {:?}", clipping_values);
}

#[test]
fn 학습률스케줄_패턴매칭_테스트() {
    let exponential = LearningRateSchedule::ExponentialDecay { decay_rate: 0.95, decay_steps: 100 };
    let cosine = LearningRateSchedule::CosineAnnealing { min_lr: 1e-6, max_lr: 1e-2, period: 1000 };
    let step = LearningRateSchedule::StepDecay { step_size: 500, gamma: 0.5 };
    let adaptive = LearningRateSchedule::Adaptive { patience: 10, factor: 0.8 };
    let constant = LearningRateSchedule::Constant;
    
    // 패턴 매칭 테스트
    match exponential {
        LearningRateSchedule::ExponentialDecay { decay_rate, decay_steps } => {
            assert_eq!(decay_rate, 0.95, "ExponentialDecay decay_rate 추출");
            assert_eq!(decay_steps, 100, "ExponentialDecay decay_steps 추출");
        },
        _ => panic!("잘못된 패턴 매칭"),
    }
    
    match cosine {
        LearningRateSchedule::CosineAnnealing { min_lr, max_lr, period } => {
            assert_eq!(min_lr, 1e-6, "CosineAnnealing min_lr 추출");
            assert_eq!(max_lr, 1e-2, "CosineAnnealing max_lr 추출");
            assert_eq!(period, 1000, "CosineAnnealing period 추출");
        },
        _ => panic!("잘못된 패턴 매칭"),
    }
    
    match step {
        LearningRateSchedule::StepDecay { step_size, gamma } => {
            assert_eq!(step_size, 500, "StepDecay step_size 추출");
            assert_eq!(gamma, 0.5, "StepDecay gamma 추출");
        },
        _ => panic!("잘못된 패턴 매칭"),
    }
    
    match adaptive {
        LearningRateSchedule::Adaptive { patience, factor } => {
            assert_eq!(patience, 10, "Adaptive patience 추출");
            assert_eq!(factor, 0.8, "Adaptive factor 추출");
        },
        _ => panic!("잘못된 패턴 매칭"),
    }
    
    match constant {
        LearningRateSchedule::Constant => {
            // 성공
        },
        _ => panic!("잘못된 패턴 매칭"),
    }
    
    println!("✅ 학습률 스케줄 패턴 매칭 테스트 통과");
}

#[test]
fn 극단적_매개변수_테스트() {
    // 매우 작은 값들
    let small_config = OptimizerConfig::new()
        .with_learning_rate(1e-10)
        .with_weight_decay(1e-15)
        .with_adam_config(AdamConfig {
            beta1: 0.001,
            beta2: 0.001,
            epsilon: 1e-20,
        });
    
    assert_eq!(small_config.learning_rate, 1e-10, "매우 작은 학습률");
    assert_eq!(small_config.weight_decay, 1e-15, "매우 작은 가중치 감소");
    assert_eq!(small_config.adam.beta1, 0.001, "매우 작은 beta1");
    
    // 큰 값들 (합리적인 범위 내에서)
    let large_config = OptimizerConfig::new()
        .with_learning_rate(1.0)
        .with_weight_decay(0.1)
        .with_adam_config(AdamConfig {
            beta1: 0.999,
            beta2: 0.9999,
            epsilon: 1e-4,
        });
    
    assert_eq!(large_config.learning_rate, 1.0, "큰 학습률");
    assert_eq!(large_config.weight_decay, 0.1, "큰 가중치 감소");
    assert_eq!(large_config.adam.beta1, 0.999, "큰 beta1");
    
    println!("✅ 극단적 매개변수 테스트 통과");
}

#[test]
fn 구성_디버그_출력_테스트() {
    let config = OptimizerConfig::new()
        .with_learning_rate(0.01)
        .with_lr_schedule(LearningRateSchedule::CosineAnnealing { 
            min_lr: 1e-6, 
            max_lr: 1e-2, 
            period: 1000 
        });
    
    let debug_output = format!("{:?}", config);
    
    // Debug 출력에 주요 정보가 포함되어 있는지 확인
    assert!(debug_output.contains("learning_rate"), "학습률 정보 포함");
    assert!(debug_output.contains("adam"), "Adam 구성 정보 포함");
    assert!(debug_output.contains("riemannian_adam"), "Riemannian Adam 구성 정보 포함");
    assert!(debug_output.contains("CosineAnnealing"), "학습률 스케줄 정보 포함");
    
    println!("✅ 구성 디버그 출력 테스트 통과");
    println!("   Debug 출력 길이: {} 문자", debug_output.len());
}

// === ENUM 관련 테스트들 ===

#[test]
fn 최적화기타입_열거형_테스트() {
    // OptimizerType enum의 기본 동작 확인
    let types = [
        OptimizerType::Adam,
        OptimizerType::RiemannianAdam,
        OptimizerType::SGD,
        OptimizerType::RMSprop,
    ];
    
    for optimizer_type in &types {
        // Debug 출력 확인
        let debug_str = format!("{:?}", optimizer_type);
        assert!(!debug_str.is_empty(), "Debug 출력 가능");
        
        // Copy 확인 (실제로 Copy trait 구현됨)
        let copied = *optimizer_type;
        assert_eq!(*optimizer_type, copied, "Copy 동작 확인");
        
        // Clone 확인
        let cloned = optimizer_type.clone();
        assert_eq!(*optimizer_type, cloned, "Clone 동작 확인");
        
        // PartialEq 확인
        assert_eq!(*optimizer_type, *optimizer_type, "자기 자신과 같음");
    }
    
    // 서로 다른 타입은 다름을 확인
    assert_ne!(OptimizerType::Adam, OptimizerType::RiemannianAdam);
    assert_ne!(OptimizerType::Adam, OptimizerType::SGD);
    assert_ne!(OptimizerType::RiemannianAdam, OptimizerType::RMSprop);
    
    println!("✅ 최적화기 타입 열거형 테스트 통과");
    println!("   총 {} 개의 최적화기 타입", types.len());
}

#[test]
fn 변환타입_열거형_테스트() {
    // TransformType enum 사용 (analyzer에서 활용)
    let types = [TransformType::Dct, TransformType::Dwt];
    
    for transform_type in &types {
        let debug_str = format!("{:?}", transform_type);
        assert!(!debug_str.is_empty(), "Debug 출력 가능");
        
        // Clone 동작 확인 (Copy trait도 구현되어 있음)
        let cloned = transform_type.clone();
        let copied = *transform_type;
        assert_eq!(*transform_type, cloned, "Clone 동작 확인");
        assert_eq!(*transform_type, copied, "Copy 동작 확인");
    }
    
    // 서로 다른 타입은 다름을 확인
    assert_ne!(TransformType::Dct, TransformType::Dwt);
    
    println!("✅ 변환 타입 열거형 테스트 통과");
} 