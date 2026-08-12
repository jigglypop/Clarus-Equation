//! 성능 개선 검증 예제

use rbe_llm::{
    nlp::{
        linear::RBELinear,
        dropout::RBEDropout,
        rmsnorm::{RBERMSNorm, RBERMSNormConfig},
    },
    QualityGrade,
    core::packed_params::HybridEncodedBlock,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== RBE 성능 개선 검증 ===\n");
    
    // 1. RBELinear forward_into 성능 테스트
    test_linear_performance()?;
    
    // 2. RBEDropout RNG 캐싱 성능 테스트
    test_dropout_performance()?;
    
    // 3. RBERMSNorm 성능 테스트
    test_rmsnorm_performance()?;
    
    Ok(())
}

fn test_linear_performance() -> anyhow::Result<()> {
    println!("1. RBELinear 메모리 재사용 성능 테스트");
    
    let in_features = 768;
    let out_features = 3072;
    let batch_size = 32;
    
    // 더미 블록 생성
    let blocks: Vec<HybridEncodedBlock> = vec![];
    let mut linear = RBELinear::new(blocks, in_features, out_features, None);
    
    // 입력과 출력 버퍼 준비
    let input = vec![0.1f32; in_features * batch_size];
    let mut output_buffer = vec![0.0f32; out_features * batch_size];
    
    // 기존 forward (새 할당)
    let start = Instant::now();
    for i in 0..batch_size {
        let batch_input = &input[i * in_features..(i + 1) * in_features];
        let _output = linear.forward(batch_input);
    }
    let forward_time = start.elapsed();
    
    // forward_into (버퍼 재사용)
    let start = Instant::now();
    for i in 0..batch_size {
        let batch_input = &input[i * in_features..(i + 1) * in_features];
        let batch_output = &mut output_buffer[i * out_features..(i + 1) * out_features];
        linear.forward_into(batch_input, batch_output)?;
    }
    let forward_into_time = start.elapsed();
    
    println!("  - forward (새 할당): {:?}", forward_time);
    println!("  - forward_into (재사용): {:?}", forward_into_time);
    println!("  - 개선률: {:.1}%\n", 
        (1.0 - forward_into_time.as_secs_f32() / forward_time.as_secs_f32()) * 100.0);
    
    Ok(())
}

fn test_dropout_performance() -> anyhow::Result<()> {
    println!("2. RBEDropout RNG 캐싱 성능 테스트");
    
    let size = 768 * 512; // 큰 텐서
    let iterations = 100;
    
    let mut dropout = RBEDropout::new(0.1)?;
    let input = vec![1.0f32; size];
    
    // 워밍업
    for _ in 0..10 {
        let _ = dropout.forward(&input);
    }
    
    // 실제 측정
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = dropout.forward(&input);
    }
    let elapsed = start.elapsed();
    
    println!("  - {} iterations 소요 시간: {:?}", iterations, elapsed);
    println!("  - 평균 시간: {:?}", elapsed / iterations);
    println!("  - 처리량: {:.2} MB/s\n", 
        (size as f64 * 4.0 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0);
    
    Ok(())
}

fn test_rmsnorm_performance() -> anyhow::Result<()> {
    println!("3. RBERMSNorm 성능 테스트");
    
    let config = RBERMSNormConfig {
        normalized_shape: 768,
        epsilon: 1e-5,
        quality_grade: QualityGrade::A,
        enable_parallel: true,
    };
    
    let mut rmsnorm = RBERMSNorm::new(config);
    rmsnorm.init_weights()?;
    
    let batch_size = 64;
    let seq_len = 512;
    let input = vec![1.0f32; batch_size * seq_len * 768];
    
    // 병렬 처리 성능 측정
    let start = Instant::now();
    let _ = rmsnorm.forward(&input)?;
    let parallel_time = start.elapsed();
    
    // 순차 처리와 비교
    rmsnorm.config.enable_parallel = false;
    let start = Instant::now();
    let _ = rmsnorm.forward(&input)?;
    let sequential_time = start.elapsed();
    
    println!("  - 병렬 처리: {:?}", parallel_time);
    println!("  - 순차 처리: {:?}", sequential_time);
    println!("  - 병렬화 speedup: {:.2}x", 
        sequential_time.as_secs_f32() / parallel_time.as_secs_f32());
    
    // 메모리 사용량
    let (compressed_size, compression_ratio) = rmsnorm.memory_usage();
    println!("  - 압축 크기: {} bytes", compressed_size);
    println!("  - 압축률: {:.1}:1\n", compression_ratio);
    
    Ok(())
} 