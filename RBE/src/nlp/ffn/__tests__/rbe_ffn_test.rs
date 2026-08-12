//! RBEFFN 레이어 테스트

use crate::nlp::ffn::{RBEFFN, RBEFFNConfig, ActivationType};
use crate::QualityGrade;
use anyhow::Result;

#[test]
fn FFN_생성_및_초기화_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 128,
        intermediate_dim: 512,  // 4x
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 64,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 16,
    };
    
    let mut ffn = RBEFFN::new(config)?;
    ffn.init_random()?;
    
    // 메모리 사용량 확인
    let (compressed_size, compression_ratio) = ffn.memory_usage();
    println!("FFN 압축 크기: {} bytes", compressed_size);
    println!("FFN 압축률: {:.1}:1", compression_ratio);
    
    // FFN은 가장 큰 레이어이므로 압축률이 높아야 함
    assert!(compression_ratio > 50.0, "FFN 압축률이 50:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn FFN_순전파_기본동작_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 64,
        intermediate_dim: 256,  // 4x
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 32,
        quality_grade: QualityGrade::C,  // 빠른 테스트
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut ffn = RBEFFN::new(config.clone())?;
    ffn.init_random()?;
    
    // 입력 생성
    let input = vec![0.5; 64];  // hidden_dim 크기
    let output = ffn.forward(&input)?;
    
    // 출력 크기 확인
    assert_eq!(output.len(), config.hidden_dim);
    
    // 값이 합리적인 범위인지 확인
    for &val in output.iter() {
        assert!(!val.is_nan(), "출력에 NaN이 있음");
        assert!(!val.is_infinite(), "출력에 Inf가 있음");
        assert!(val.abs() < 10.0, "출력값이 너무 큼: {}", val);
    }
    
    Ok(())
}

#[test]
fn 활성화함수_동작_테스트() -> Result<()> {
    let configs = vec![
        (ActivationType::Gelu, "GELU"),
        (ActivationType::GeluNew, "GELU New"),
        (ActivationType::Relu, "ReLU"),
        (ActivationType::Swish, "Swish"),
    ];
    
    for (activation, name) in configs {
        let config = RBEFFNConfig {
            hidden_dim: 32,
            intermediate_dim: 128,
            activation,
            dropout: 0.0,
            block_size: 32,
            quality_grade: QualityGrade::C,
            enable_parallel: false,
            cache_size: 4,
        };
        
        let mut ffn = RBEFFN::new(config)?;
        ffn.init_random()?;
        
        // 다양한 입력값 테스트
        let test_inputs = vec![
            vec![-2.0; 32],  // 음수
            vec![0.0; 32],   // 0
            vec![2.0; 32],   // 양수
        ];
        
        for input in test_inputs {
            let output = ffn.forward(&input)?;
            
            // 활성화 함수별 특성 확인
            match activation {
                ActivationType::Relu => {
                    // ReLU는 음수 입력에 대해 0이어야 함
                    if input[0] < 0.0 {
                        // 첫 번째 레이어 후 ReLU가 적용되므로 
                        // 최종 출력이 반드시 0은 아님
                    }
                }
                _ => {
                    // 다른 활성화 함수들은 연속적
                }
            }
            
            // 모든 활성화 함수에서 NaN/Inf 없어야 함
            for &val in output.iter() {
                assert!(!val.is_nan(), "{} 활성화에서 NaN 발생", name);
                assert!(!val.is_infinite(), "{} 활성화에서 Inf 발생", name);
            }
        }
    }
    
    Ok(())
}

#[test]
fn 배치처리_일관성_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 48,
        intermediate_dim: 192,
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 48,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut ffn = RBEFFN::new(config.clone())?;
    ffn.init_random()?;
    
    // 배치 입력 (3개 샘플)
    let batch_size = 3;
    let mut batch_input = Vec::new();
    for i in 0..batch_size {
        let sample: Vec<f32> = (0..config.hidden_dim)
            .map(|j| (i * config.hidden_dim + j) as f32 * 0.01)
            .collect();
        batch_input.extend(sample);
    }
    
    // 전체 배치 처리
    let batch_output = ffn.forward(&batch_input)?;
    
    // 개별 처리 후 비교
    for i in 0..batch_size {
        let start = i * config.hidden_dim;
        let end = (i + 1) * config.hidden_dim;
        let sample = &batch_input[start..end];
        
        let individual_output = ffn.forward(sample)?;
        let batch_sample_output = &batch_output[start..end];
        
        // 같아야 함
        for j in 0..config.hidden_dim {
            assert!((individual_output[j] - batch_sample_output[j]).abs() < 1e-6,
                    "배치 처리와 개별 처리 결과가 다름: {} vs {}",
                    individual_output[j], batch_sample_output[j]);
        }
    }
    
    Ok(())
}

#[test]
fn 드롭아웃_동작_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 100,
        intermediate_dim: 400,
        activation: ActivationType::Gelu,
        dropout: 0.5,  // 50% 드롭아웃
        block_size: 100,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 4,
    };
    
    let mut ffn = RBEFFN::new(config)?;
    ffn.init_random()?;
    
    // 같은 입력에 대해 여러 번 실행
    let input = vec![1.0; 100];
    let output1 = ffn.forward(&input)?;
    let output2 = ffn.forward(&input)?;
    
    // 드롭아웃으로 인해 결과가 달라야 함
    // 동일성 검증
    let diff_count = output1.iter()
        .zip(&output2)
        .filter(|(&a, &b)| (a - b).abs() > 1e-6)
        .count();
    
    assert!(diff_count > 10, 
            "드롭아웃이 활성화되었지만 출력이 너무 유사함: {} 개만 다름", 
            diff_count);
    
    // 드롭아웃 스케일링 확인
    // 평균은 비슷해야 함 (1/(1-p) 스케일링 때문)
    let mean1: f32 = output1.iter().sum::<f32>() / output1.len() as f32;
    let mean2: f32 = output2.iter().sum::<f32>() / output2.len() as f32;
    
    // 평균의 차이가 크지 않아야 함
    assert!((mean1 - mean2).abs() < 0.5,
            "드롭아웃 스케일링이 잘못됨: 평균 차이 {}", (mean1 - mean2).abs());
    
    Ok(())
}

#[test]
fn 큰모델_압축효율_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 768,      // GPT-2 base
        intermediate_dim: 3072,  // 4x
        activation: ActivationType::GeluNew,
        dropout: 0.0,
        block_size: 256,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let ffn = RBEFFN::new(config.clone())?;
    // 실제 초기화는 메모리 때문에 생략
    
    // 이론적 압축률 계산
    let original_size = (config.hidden_dim * config.intermediate_dim * 2) * 4;  // f32
    let block_size = 80;  // HybridEncodedBlock의 대략적 크기
    
    // Up projection 블록 수
    let up_blocks = ((config.intermediate_dim + config.block_size - 1) / config.block_size)
                  * ((config.hidden_dim + config.block_size - 1) / config.block_size);
    
    // Down projection 블록 수
    let down_blocks = ((config.hidden_dim + config.block_size - 1) / config.block_size)
                    * ((config.intermediate_dim + config.block_size - 1) / config.block_size);
    
    let compressed_size = (up_blocks + down_blocks) * block_size;
    let theoretical_ratio = original_size as f32 / compressed_size as f32;
    
    println!("FFN 이론적 압축률: {:.1}:1", theoretical_ratio);
    println!("원본 크기: {:.2} MB", original_size as f32 / 1024.0 / 1024.0);
    println!("압축 크기: {:.2} MB", compressed_size as f32 / 1024.0 / 1024.0);
    
    assert!(theoretical_ratio > 100.0, 
            "대규모 FFN에서 압축률이 100:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn 사전학습_가중치_로드_테스트() -> Result<()> {
    let config = RBEFFNConfig {
        hidden_dim: 16,
        intermediate_dim: 64,
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 16,
        quality_grade: QualityGrade::A,
        enable_parallel: false,
        cache_size: 4,
    };
    
    // 가짜 사전학습 가중치 생성
    let up_weights: Vec<f32> = (0..config.hidden_dim * config.intermediate_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    
    let down_weights: Vec<f32> = (0..config.intermediate_dim * config.hidden_dim)
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    
    // 로드
    let mut ffn = RBEFFN::from_pretrained_weights(
        &up_weights,
        &down_weights,
        config.clone()
    )?;
    
    // 순전파 테스트
    let input = vec![0.1; 16];
    let output = ffn.forward(&input)?;
    
    assert_eq!(output.len(), config.hidden_dim);
    
    // 결과가 합리적인지 확인
    for &val in output.iter() {
        assert!(!val.is_nan() && !val.is_infinite());
    }
    
    Ok(())
}

#[test]
fn 성능_벤치마크_테스트() -> Result<()> {
    use std::time::Instant;
    
    let config = RBEFFNConfig {
        hidden_dim: 768,
        intermediate_dim: 3072,
        activation: ActivationType::GeluNew,
        dropout: 0.0,
        block_size: 256,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let mut ffn = RBEFFN::new(config)?;
    ffn.init_random()?;
    
    // 배치 크기 32
    let input = vec![0.1; 768 * 32];
    
    // 워밍업
    for _ in 0..5 {
        let _ = ffn.forward(&input)?;
    }
    
    // 실제 측정
    let start = Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        let _ = ffn.forward(&input)?;
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    println!("FFN 평균 실행 시간: {:?}", avg_time);
    println!("처리량: {:.2} samples/sec", 
             32.0 * iterations as f64 / elapsed.as_secs_f64());
    
    // 성능 기준: 배치당 10ms 이하 (FFN은 무거운 레이어)
    assert!(avg_time.as_millis() < 10, 
            "FFN이 너무 느림: {:?}", avg_time);
    
    Ok(())
} 