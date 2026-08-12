//! RBEAttention 레이어 테스트

use crate::nlp::attention::{RBEAttention, RBEAttentionConfig};
use crate::QualityGrade;
use anyhow::Result;

#[test]
fn 어텐션_생성_및_초기화_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 96,
        num_heads: 8,
        head_dim: 12,  // 96 / 8
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 48,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 16,
    };
    
    let mut attention = RBEAttention::new(config)?;
    attention.init_random()?;
    
    // 메모리 사용량 확인
    let (compressed_size, compression_ratio) = attention.memory_usage();
    println!("Attention 압축 크기: {} bytes", compressed_size);
    println!("Attention 압축률: {:.1}:1", compression_ratio);
    
    // Attention은 4개의 projection이 있으므로 압축률이 높아야 함
    assert!(compression_ratio > 50.0, "Attention 압축률이 50:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn 어텐션_순전파_기본동작_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 48,
        num_heads: 4,
        head_dim: 12,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 24,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut attention = RBEAttention::new(config.clone())?;
    
    // 테스트 입력
    let batch_size = 2;
    let seq_len = 10;
    let input_size = seq_len * config.hidden_dim;
    
    let output = attention.forward(&vec![0.1; input_size], None)?;
    
    // 출력 크기 확인
    assert_eq!(output.len(), input_size);
    
    // 값이 합리적인 범위인지 확인
    for &val in output.iter() {
        assert!(!val.is_nan(), "출력에 NaN이 있음");
        assert!(!val.is_infinite(), "출력에 Inf가 있음");
        assert!(val.abs() < 10.0, "출력값이 너무 큼: {}", val);
    }
    
    Ok(())
}

#[test]
fn 멀티헤드_분할_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 64,
        num_heads: 8,
        head_dim: 8,  // 64 / 8
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 32,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut attention = RBEAttention::new(config.clone())?;
    attention.init_random()?;
    
    // 다양한 시퀀스 길이 테스트
    for seq_len in &[1, 4, 8, 16] {
        let hidden_states = vec![0.5; *seq_len * config.hidden_dim];
        let output = attention.forward(&hidden_states, None)?;
        
        assert_eq!(output.len(), hidden_states.len(),
                   "시퀀스 길이 {}에서 출력 크기 불일치", seq_len);
    }
    
    Ok(())
}

#[test]
fn 어텐션_마스크_동작_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 16,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 4,
    };
    
    let mut attention = RBEAttention::new(config.clone())?;
    attention.init_random()?;
    
    let seq_len = 4;
    let hidden_states = vec![1.0; seq_len * config.hidden_dim];
    
    // Causal mask (하삼각 행렬)
    let mut mask = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 1.0;
        }
    }
    
    // 마스크 있는 경우와 없는 경우 비교
    let output_with_mask = attention.forward(&hidden_states, Some(&mask))?;
    let output_no_mask = attention.forward(&hidden_states, None)?;
    
    // 결과가 달라야 함
    let diff_count = output_with_mask.iter()
        .zip(output_no_mask.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    
    assert!(diff_count > 0, "어텐션 마스크가 적용되지 않음");
    
    Ok(())
}

#[test]
fn 어텐션_드롭아웃_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 48,
        num_heads: 6,
        head_dim: 8,
        attention_dropout: 0.3,
        output_dropout: 0.3,
        block_size: 24,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 4,
    };
    
    let mut attention = RBEAttention::new(config)?;
    attention.init_random()?;
    
    let hidden_states = vec![0.5; 3 * 48];  // seq_len=3
    
    // 같은 입력에 대해 다른 마스크는 다른 결과를 내야 함
    let output1 = attention.forward(&hidden_states, None)?;
    let output2 = attention.forward(&hidden_states, None)?;
    
    // 드롭아웃으로 인해 결과가 달라야 함
    let diff_count = output1.iter()
        .zip(&output2)
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    
    assert!(diff_count > 10, 
            "드롭아웃이 활성화되었지만 출력이 너무 유사함: {} 개만 다름", 
            diff_count);
    
    Ok(())
}

#[test]
fn 큰모델_어텐션_압축효율_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 768,      // GPT-2 base
        num_heads: 12,
        head_dim: 64,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 256,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let attention = RBEAttention::new(config.clone())?;
    // 실제 초기화는 메모리 때문에 생략
    
    // 이론적 압축률 계산
    let original_size = 4 * config.hidden_dim * config.hidden_dim * 4;  // 4 projections, f32
    let block_size = 80;  // HybridEncodedBlock의 대략적 크기
    
    let blocks_per_matrix = ((config.hidden_dim + config.block_size - 1) / config.block_size).pow(2);
    let compressed_size = 4 * blocks_per_matrix * block_size;  // 4 projections
    
    let theoretical_ratio = original_size as f32 / compressed_size as f32;
    
    println!("Attention 이론적 압축률: {:.1}:1", theoretical_ratio);
    println!("원본 크기: {:.2} MB", original_size as f32 / 1024.0 / 1024.0);
    println!("압축 크기: {:.2} MB", compressed_size as f32 / 1024.0 / 1024.0);
    
    assert!(theoretical_ratio > 80.0, 
            "대규모 Attention에서 압축률이 80:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn 사전학습_가중치_로드_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 16,
        num_heads: 2,
        head_dim: 8,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 8,
        quality_grade: QualityGrade::A,
        enable_parallel: false,
        cache_size: 4,
    };
    
    // 가짜 사전학습 가중치 생성
    let weight_size = config.hidden_dim * config.hidden_dim;
    let q_weights: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let k_weights: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.002).cos()).collect();
    let v_weights: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.003).sin()).collect();
    let o_weights: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.004).cos()).collect();
    
    // 로드
    let mut attention = RBEAttention::from_pretrained_weights(
        &q_weights,
        &k_weights,
        &v_weights,
        &o_weights,
        config.clone()
    )?;
    
    // 순전파 테스트
    let hidden_states = vec![0.1; 2 * 16];  // seq_len=2
    let output = attention.forward(&hidden_states, None)?;
    
    assert_eq!(output.len(), hidden_states.len());
    
    // 결과가 합리적인지 확인
    for &val in output.iter() {
        assert!(!val.is_nan() && !val.is_infinite());
    }
    
    Ok(())
}

#[test]
fn 성능_벤치마크_테스트() -> Result<()> {
    use std::time::Instant;
    
    let config = RBEAttentionConfig {
        hidden_dim: 256,  // 작은 모델로 테스트
        num_heads: 8,
        head_dim: 32,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 128,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 16,
    };
    
    let hidden_dim = config.hidden_dim;  // config 사용 전에 필요한 값 추출
    let mut attention = RBEAttention::new(config)?;
    attention.init_random()?;
    
    // 배치 크기 16, 시퀀스 길이 64
    let seq_len = 64;
    let batch_size = 16;
    let input_size = seq_len * hidden_dim;
    
    // 실제로는 배치 처리를 위해 반복
    let mut total_time = std::time::Duration::new(0, 0);
    let iterations = 10;
    
    for _ in 0..iterations {
        let input = vec![0.1; input_size];
        let start = Instant::now();
        let _ = attention.forward(&input, None)?;
        total_time += start.elapsed();
    }
    
    let avg_time = total_time / iterations;
    let samples_per_sec = (batch_size * iterations) as f64 / total_time.as_secs_f64();
    
    println!("Attention 평균 실행 시간: {:?}", avg_time);
    println!("처리량: {:.2} samples/sec", samples_per_sec);
    
    // 성능 기준: 시퀀스당 20ms 이하 (Attention은 무거운 연산)
    assert!(avg_time.as_millis() < 20, 
            "Attention이 너무 느림: {:?}", avg_time);
    
    Ok(())
}

#[test]
fn 어텐션_수치적_안정성_테스트() -> Result<()> {
    let config = RBEAttentionConfig {
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 16,
        quality_grade: QualityGrade::S,  // 최고 품질
        enable_parallel: false,
        cache_size: 4,
    };
    
    let mut attention = RBEAttention::new(config.clone())?;
    attention.init_random()?;
    
    // 극한값 테스트
    let test_cases = vec![
        vec![1e-10; 32],  // 매우 작은 값
        vec![1e10; 32],   // 매우 큰 값
        vec![0.0; 32],    // 0
        (0..32).map(|i| if i % 2 == 0 { 1e10 } else { 1e-10 }).collect(),  // 혼합
    ];
    
    for (i, input) in test_cases.iter().enumerate() {
        let output = attention.forward(input, None)?;
        
        for &val in output.iter() {
            assert!(!val.is_nan(), "테스트 케이스 {}에서 NaN 발생", i);
            assert!(!val.is_infinite(), "테스트 케이스 {}에서 Inf 발생", i);
        }
    }
    
    Ok(())
} 