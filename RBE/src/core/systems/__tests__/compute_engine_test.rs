//! # 계산 엔진 단위테스트
//!
//! ComputeEngine과 관련 계산 구조체들의 기능 검증

use crate::core::systems::compute_engine::{
    ResidualCompressor, CORDICEngine, BasisFunctionLUT, ParallelGEMMEngine,
    RiemannianGradientComputer, StateTransitionDifferentiator, AdaptiveScheduler,
    InternalTransformType, LearningRateStrategy
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 잔차_압축기_테스트() {
        let mut compressor = ResidualCompressor::new();
        
        assert_eq!(compressor.transform_type, InternalTransformType::DCT);
        assert_eq!(compressor.compression_ratio, 0.1); // 10% 유지
        assert_eq!(compressor.sparsity_threshold, 1e-3); // 0.001
        
        // 압축 설정 변경
        compressor.compression_ratio = 10.0;
        compressor.sparsity_threshold = 0.05;
        compressor.transform_type = InternalTransformType::Hybrid;
        
        assert_eq!(compressor.compression_ratio, 10.0);
        assert_eq!(compressor.sparsity_threshold, 0.05);
        assert_eq!(compressor.transform_type, InternalTransformType::Hybrid);
        
        println!("✅ 잔차 압축기 테스트 통과");
        println!("   압축 비율: {}", compressor.compression_ratio);
    }

    #[test]
    fn CORDIC_엔진_테스트() {
        let mut cordic_engine = CORDICEngine::new(16, 1e-6, 4);
        
        assert_eq!(cordic_engine.iterations, 16);
        assert_eq!(cordic_engine.precision_threshold, 1e-6);
        assert_eq!(cordic_engine.parallel_units, 4);
        
        // CORDIC 설정 변경
        cordic_engine.iterations = 20;
        cordic_engine.precision_threshold = 1e-8;
        cordic_engine.parallel_units = 8;
        
        assert_eq!(cordic_engine.iterations, 20);
        assert_eq!(cordic_engine.precision_threshold, 1e-8);
        assert_eq!(cordic_engine.parallel_units, 8);
        
        println!("✅ CORDIC 엔진 테스트 통과");
        println!("   반복 횟수: {}, 병렬 단위: {}", 
                cordic_engine.iterations, cordic_engine.parallel_units);
    }

    #[test]
    fn 기저함수_룩업테이블_테스트() {
        let lut = BasisFunctionLUT::new(1024);
        
        assert_eq!(lut.sin_lut.len(), 1024);
        assert_eq!(lut.cos_lut.len(), 1024);
        assert_eq!(lut.tanh_lut.len(), 1024);
        assert_eq!(lut.sech2_lut.len(), 1024);
        assert_eq!(lut.exp_lut.len(), 1024);
        assert_eq!(lut.log_lut.len(), 1024);
        assert_eq!(lut.inv_lut.len(), 1024);
        assert_eq!(lut.poly_lut.len(), 1024);
        assert_eq!(lut.resolution, 1024);
        
        // 기본 수학 함수 값 범위 확인
        for &sin_val in &lut.sin_lut {
            assert!(sin_val >= -1.0 && sin_val <= 1.0);
        }
        
        for &cos_val in &lut.cos_lut {
            assert!(cos_val >= -1.0 && cos_val <= 1.0);
        }
        
        println!("✅ 기저함수 룩업테이블 테스트 통과");
        println!("   테이블 해상도: {}", lut.resolution);
    }

    #[test]
    fn 병렬_GEMM_엔진_테스트() {
        let mut gemm_engine = ParallelGEMMEngine::new(4, 64, true);
        
        assert_eq!(gemm_engine.thread_pool_size, 4);
        assert_eq!(gemm_engine.block_size, 64);
        assert_eq!(gemm_engine.cache_optimization, true);
        
        // GEMM 설정 변경
        gemm_engine.thread_pool_size = 8;
        gemm_engine.block_size = 128;
        gemm_engine.cache_optimization = false;
        
        assert_eq!(gemm_engine.thread_pool_size, 8);
        assert_eq!(gemm_engine.block_size, 128);
        assert_eq!(gemm_engine.cache_optimization, false);
        
        println!("✅ 병렬 GEMM 엔진 테스트 통과");
        println!("   스레드 풀 크기: {}, 블록 크기: {}", 
                gemm_engine.thread_pool_size, gemm_engine.block_size);
    }

    #[test]
    fn 리만_그래디언트_계산기_테스트() {
        let mut gradient_computer = RiemannianGradientComputer::new();
        
        assert_eq!(gradient_computer.clipping_threshold, 1.0);
        assert_eq!(gradient_computer.numerical_stability_eps, 1e-8);
        
        // 그래디언트 계산기 설정 변경
        gradient_computer.clipping_threshold = 0.5;
        gradient_computer.numerical_stability_eps = 1e-10;
        
        assert_eq!(gradient_computer.clipping_threshold, 0.5);
        assert_eq!(gradient_computer.numerical_stability_eps, 1e-10);
        
        println!("✅ 리만 그래디언트 계산기 테스트 통과");
        println!("   클리핑 임계값: {}", gradient_computer.clipping_threshold);
    }

    #[test]
    fn 상태전이_미분계산기_테스트() {
        let mut differentiator = StateTransitionDifferentiator::new();
        
        assert_eq!(differentiator.transition_threshold, 0.1);
        assert_eq!(differentiator.state_change_history.len(), 0);
        
        // 상태 변화 히스토리 추가
        differentiator.state_change_history.push(vec![0, 1, 2]);
        differentiator.state_change_history.push(vec![1, 3, 4]);
        
        assert_eq!(differentiator.state_change_history.len(), 2);
        assert_eq!(differentiator.state_change_history[0], vec![0, 1, 2]);
        
        println!("✅ 상태전이 미분계산기 테스트 통과");
        println!("   히스토리 길이: {}", differentiator.state_change_history.len());
    }

    #[test]
    fn 적응적_스케줄러_테스트() {
        let mut scheduler = AdaptiveScheduler::new(0.001);
        
        assert_eq!(scheduler.current_learning_rate, 0.001);
        // LearningRateStrategy는 PartialEq가 구현되지 않아서 직접 비교 불가
        
        // 스케줄러 설정 변경
        scheduler.current_learning_rate = 0.01;
        scheduler.adjustment_strategy = LearningRateStrategy::ExponentialDecay;
        
        assert_eq!(scheduler.current_learning_rate, 0.01);
        
        println!("✅ 적응적 스케줄러 테스트 통과");
        println!("   현재 학습률: {}", scheduler.current_learning_rate);
    }

    #[test]
    fn 변환_타입_테스트() {
        let mut compressor = ResidualCompressor::new();
        
        // 다양한 변환 타입 테스트
        let transform_types = vec![
            InternalTransformType::DCT,
            InternalTransformType::DWT,
            InternalTransformType::Hybrid,
        ];
        
        for transform_type in transform_types {
            compressor.transform_type = transform_type.clone();
            assert_eq!(compressor.transform_type, transform_type);
            
            println!("   변환 타입: {:?}", compressor.transform_type);
        }
        
        // 타입 비교 테스트
        assert_ne!(InternalTransformType::DCT, InternalTransformType::DWT);
        assert_ne!(InternalTransformType::DWT, InternalTransformType::Hybrid);
        
        println!("✅ 변환 타입 테스트 통과");
    }

    #[test]
    fn 학습률_전략_테스트() {
        let mut scheduler = AdaptiveScheduler::new(0.001);
        
        // 다양한 학습률 전략 테스트 (PartialEq 없어서 직접 비교 불가)
        scheduler.adjustment_strategy = LearningRateStrategy::Fixed;
        println!("   전략: {:?}", scheduler.adjustment_strategy);
        
        scheduler.adjustment_strategy = LearningRateStrategy::ExponentialDecay;
        println!("   전략: {:?}", scheduler.adjustment_strategy);
        
        scheduler.adjustment_strategy = LearningRateStrategy::CosineAnnealing;
        println!("   전략: {:?}", scheduler.adjustment_strategy);
        
        scheduler.adjustment_strategy = LearningRateStrategy::Adaptive;
        println!("   전략: {:?}", scheduler.adjustment_strategy);
        
        scheduler.adjustment_strategy = LearningRateStrategy::Cyclic;
        println!("   전략: {:?}", scheduler.adjustment_strategy);
        
        println!("✅ 학습률 전략 테스트 통과");
    }

    #[test]
    fn 구조체_복제_테스트() {
        let compressor = ResidualCompressor::new();
        let cloned_compressor = compressor.clone();
        
        assert_eq!(compressor.transform_type, cloned_compressor.transform_type);
        assert_eq!(compressor.compression_ratio, cloned_compressor.compression_ratio);
        assert_eq!(compressor.sparsity_threshold, cloned_compressor.sparsity_threshold);
        
        let cordic_engine = CORDICEngine::new(16, 1e-6, 4);
        let cloned_engine = cordic_engine.clone();
        
        assert_eq!(cordic_engine.iterations, cloned_engine.iterations);
        assert_eq!(cordic_engine.precision_threshold, cloned_engine.precision_threshold);
        assert_eq!(cordic_engine.parallel_units, cloned_engine.parallel_units);
        
        println!("✅ 구조체 복제 테스트 통과");
    }

    #[test]
    fn 성능_매개변수_테스트() {
        let mut compressor = ResidualCompressor::new();
        
        // 다양한 압축률 테스트
        let compression_ratios = vec![1.0, 2.0, 5.0, 10.0, 50.0];
        
        for &ratio in &compression_ratios {
            compressor.compression_ratio = ratio;
            assert_eq!(compressor.compression_ratio, ratio);
            assert!(compressor.compression_ratio >= 1.0);
        }
        
        // 희소성 임계값 테스트
        let sparsity_thresholds = vec![0.01, 0.05, 0.1, 0.2, 0.5];
        
        for &threshold in &sparsity_thresholds {
            compressor.sparsity_threshold = threshold;
            assert_eq!(compressor.sparsity_threshold, threshold);
            assert!(compressor.sparsity_threshold >= 0.0);
            assert!(compressor.sparsity_threshold <= 1.0);
        }
        
        println!("✅ 성능 매개변수 테스트 통과");
        println!("   최종 압축률: {}", compressor.compression_ratio);
    }

    #[test]
    fn 다양한_해상도_룩업테이블_테스트() {
        let resolutions = vec![256, 512, 1024, 2048];
        
        for &resolution in &resolutions {
            let lut = BasisFunctionLUT::new(resolution);
            
            assert_eq!(lut.resolution, resolution);
            assert_eq!(lut.sin_lut.len(), resolution);
            assert_eq!(lut.cos_lut.len(), resolution);
            assert_eq!(lut.tanh_lut.len(), resolution);
            
            // 모든 룩업테이블이 동일한 크기인지 확인
            assert_eq!(lut.sech2_lut.len(), resolution);
            assert_eq!(lut.exp_lut.len(), resolution);
            assert_eq!(lut.log_lut.len(), resolution);
            assert_eq!(lut.inv_lut.len(), resolution);
            assert_eq!(lut.poly_lut.len(), resolution);
            
            println!("   해상도 {} 테이블 확인됨", resolution);
        }
        
        println!("✅ 다양한 해상도 룩업테이블 테스트 통과");
    }

    #[test]
    fn 메모리_효율성_테스트() {
        let lut = BasisFunctionLUT::new(1024);
        
        // 룩업 테이블 메모리 사용량 추정
        let single_table_memory = lut.resolution * std::mem::size_of::<f32>();
        let total_tables = 8; // 8개의 기저함수 테이블
        let total_memory = single_table_memory * total_tables;
        
        // 메모리 사용량이 합리적인 범위인지 확인
        assert!(total_memory > 0);
        assert!(total_memory < 1024 * 1024); // 1MB 미만
        
        println!("✅ 메모리 효율성 테스트 통과");
        println!("   총 메모리 사용량: {} 바이트", total_memory);
        println!("   단일 테이블 크기: {} 바이트", single_table_memory);
    }

    #[test]
    fn 병렬처리_설정_테스트() {
        let mut gemm_engine = ParallelGEMMEngine::new(4, 64, true);
        let mut cordic_engine = CORDICEngine::new(16, 1e-6, 4);
        
        // 다양한 병렬 설정 테스트
        let thread_counts = vec![1, 2, 4, 8, 16];
        
        for &threads in &thread_counts {
            gemm_engine.thread_pool_size = threads;
            cordic_engine.parallel_units = threads;
            
            assert_eq!(gemm_engine.thread_pool_size, threads);
            assert_eq!(cordic_engine.parallel_units, threads);
        }
        
        // 블록 크기 테스트
        let block_sizes = vec![32, 64, 128, 256];
        
        for &size in &block_sizes {
            gemm_engine.block_size = size;
            assert_eq!(gemm_engine.block_size, size);
        }
        
        println!("✅ 병렬처리 설정 테스트 통과");
        println!("   최종 스레드 수: {}", gemm_engine.thread_pool_size);
    }
} 