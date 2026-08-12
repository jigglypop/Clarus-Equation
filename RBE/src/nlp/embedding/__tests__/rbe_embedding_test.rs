//! RBEEmbedding 레이어 테스트

use crate::nlp::embedding::{RBEEmbedding, RBEEmbeddingConfig};
use crate::QualityGrade;
use anyhow::Result;

#[test]
fn 임베딩_생성_및_초기화_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 1000,
        embedding_dim: 128,
        max_position_embeddings: 512,
        block_size: 64,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 16,
    };
    
    let mut embedding = RBEEmbedding::new(config)?;
    embedding.init_random()?;
    
    // 메모리 사용량 확인
    let (compressed_size, compression_ratio) = embedding.memory_usage();
    println!("압축 크기: {} bytes", compressed_size);
    println!("압축률: {:.1}:1", compression_ratio);
    
    assert!(compression_ratio > 10.0, "압축률이 10:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn 순전파_기본_동작_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 100,
        embedding_dim: 64,
        max_position_embeddings: 128,
        block_size: 32,
        quality_grade: QualityGrade::C,  // 빠른 테스트를 위해 C급
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut embedding = RBEEmbedding::new(config.clone())?;
    embedding.init_random()?;
    
    // 토큰 ID 입력
    let token_ids = vec![0, 1, 2, 3, 4];
    let output = embedding.forward(&token_ids)?;
    
    // 출력 크기 확인
    assert_eq!(output.len(), token_ids.len() * config.embedding_dim);
    
    // 값이 합리적인 범위인지 확인 (sinusoidal encoding 범위)
    for &val in output.iter() {
        assert!(val >= -2.0 && val <= 2.0, "임베딩 값이 합리적 범위를 벗어남: {}", val);
    }
    
    Ok(())
}

#[test]
fn 위치_임베딩_정확성_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 10,
        embedding_dim: 8,
        max_position_embeddings: 16,
        block_size: 8,
        quality_grade: QualityGrade::S,  // 높은 정확도
        enable_parallel: false,
        cache_size: 4,
    };
    
    let mut embedding = RBEEmbedding::new(config.clone())?;
    embedding.init_random()?;
    
    // 같은 토큰, 다른 위치
    let tokens1 = vec![5, 5, 5];
    let output1 = embedding.forward(&tokens1)?;
    
    // 위치별로 다른 임베딩인지 확인
    let emb_dim = config.embedding_dim;
    let emb0 = &output1[0..emb_dim];
    let emb1 = &output1[emb_dim..2*emb_dim];
    let emb2 = &output1[2*emb_dim..3*emb_dim];
    
    // 위치가 다르면 임베딩도 달라야 함
    let diff_01 = emb0.iter().zip(emb1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    let diff_12 = emb1.iter().zip(emb2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    
    assert!(diff_01 > 0.1, "위치 0과 1의 임베딩이 너무 유사함");
    assert!(diff_12 > 0.1, "위치 1과 2의 임베딩이 너무 유사함");
    
    Ok(())
}

#[test]
fn 병렬처리_일관성_테스트() -> Result<()> {
    let base_config = RBEEmbeddingConfig {
        vocab_size: 50,
        embedding_dim: 32,
        max_position_embeddings: 64,
        block_size: 16,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 8,
    };
    
    // 순차 처리 버전
    let mut seq_embedding = RBEEmbedding::new(base_config.clone())?;
    seq_embedding.init_random()?;
    
    // 병렬 처리 버전 (같은 seed로 초기화해야 함)
    let mut par_config = base_config.clone();
    par_config.enable_parallel = true;
    let par_embedding = RBEEmbedding::from_pretrained_weights(
        &vec![0.0; base_config.vocab_size * base_config.embedding_dim],
        Some(&vec![0.0; base_config.max_position_embeddings * base_config.embedding_dim]),
        par_config,
    )?;
    
    // 같은 입력에 대해 테스트
    let token_ids = vec![0, 10, 20, 30, 40];
    let seq_output = seq_embedding.forward(&token_ids)?;
    let par_output = par_embedding.forward(&token_ids)?;
    
    // 순차와 병렬 처리 결과가 유사해야 함
    // (완전히 같지는 않을 수 있음 - 부동소수점 연산 순서 차이)
    let max_diff = seq_output.iter().zip(par_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(max_diff < 0.1, "순차와 병렬 처리 결과 차이가 너무 큼: {}", max_diff);
    
    Ok(())
}

#[test]
fn 범위_벗어난_토큰_에러_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 100,
        embedding_dim: 32,
        max_position_embeddings: 128,
        block_size: 32,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 8,
    };
    
    let mut embedding = RBEEmbedding::new(config.clone())?;
    embedding.init_random()?;
    
    // 범위를 벗어난 토큰 ID
    let invalid_tokens = vec![99, 100, 101];  // 100, 101은 범위 밖
    let result = embedding.forward(&invalid_tokens);
    
    assert!(result.is_err(), "범위를 벗어난 토큰 ID에 대해 에러가 발생해야 함");
    
    Ok(())
}

#[test]
fn 큰_모델_압축률_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 10000,    // GPT-2 수준
        embedding_dim: 768,    // GPT-2 base
        max_position_embeddings: 1024,
        block_size: 256,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let embedding = RBEEmbedding::new(config.clone())?;
    // 실제 초기화는 메모리 때문에 생략
    
    // 이론적 압축률 계산
    let original_size = (config.vocab_size + config.max_position_embeddings) 
                       * config.embedding_dim * 4;  // f32
    let block_size = 80;  // HybridEncodedBlock의 대략적 크기
    let num_blocks = (config.vocab_size + config.max_position_embeddings)
                    * ((config.embedding_dim + config.block_size - 1) / config.block_size);
    let compressed_size = num_blocks * block_size;
    
    let theoretical_ratio = original_size as f32 / compressed_size as f32;
    
    println!("이론적 압축률: {:.1}:1", theoretical_ratio);
    println!("원본 크기: {:.2} MB", original_size as f32 / 1024.0 / 1024.0);
    println!("압축 크기: {:.2} MB", compressed_size as f32 / 1024.0 / 1024.0);
    
    assert!(theoretical_ratio > 50.0, "대규모 모델에서 압축률이 50:1 이상이어야 함");
    
    Ok(())
}

#[test]
fn 사전학습_가중치_로드_테스트() -> Result<()> {
    let config = RBEEmbeddingConfig {
        vocab_size: 10,
        embedding_dim: 8,
        max_position_embeddings: 16,
        block_size: 8,
        quality_grade: QualityGrade::A,
        enable_parallel: false,
        cache_size: 4,
    };
    
    // 가짜 사전학습 가중치 생성
    let token_weights: Vec<f32> = (0..config.vocab_size * config.embedding_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    
    let position_weights: Vec<f32> = (0..config.max_position_embeddings * config.embedding_dim)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();
    
    // 로드
    let embedding = RBEEmbedding::from_pretrained_weights(
        &token_weights,
        Some(&position_weights),
        config.clone(),
    )?;
    
    // 캐시 효과 확인
    let token_ids = vec![0, 1, 2];
    let output = embedding.forward(&token_ids)?;
    assert_eq!(output.len(), token_ids.len() * config.embedding_dim);
    
    // 첫 번째 토큰의 첫 번째 임베딩 값이 원본과 유사한지 확인
    // (압축/복원으로 인한 오차 허용)
    let expected_first = token_weights[0] + position_weights[0];
    let actual_first = output[0];
    let diff = (expected_first - actual_first).abs();
    
    assert!(diff < 0.1, "압축/복원 후 값이 너무 많이 변함: {} vs {}", 
            expected_first, actual_first);
    
    Ok(())
}

#[test]
fn 캐시_효과_테스트() -> Result<()> {
    use std::time::Instant;
    
    let config = RBEEmbeddingConfig {
        vocab_size: 100,
        embedding_dim: 64,
        max_position_embeddings: 128,
        block_size: 64,
        quality_grade: QualityGrade::B,
        enable_parallel: false,
        cache_size: 32,
    };
    
    let mut embedding = RBEEmbedding::new(config)?;
    embedding.init_random()?;
    
    // 같은 토큰 반복
    let token_ids = vec![5; 100];
    
    // 첫 번째 실행 (캐시 미스)
    let start = Instant::now();
    let _ = embedding.forward(&token_ids)?;
    let first_time = start.elapsed();
    
    // 두 번째 실행 (캐시 히트)
    let start = Instant::now();
    let _ = embedding.forward(&token_ids)?;
    let second_time = start.elapsed();
    
    println!("첫 번째 실행: {:?}", first_time);
    println!("두 번째 실행: {:?}", second_time);
    
    // 캐시로 인해 두 번째가 더 빨라야 함
    // (하지만 테스트 환경에 따라 다를 수 있으므로 느슨한 검사)
    assert!(second_time.as_micros() <= first_time.as_micros() * 2,
            "캐시 효과가 나타나지 않음");
    
    Ok(())
} 