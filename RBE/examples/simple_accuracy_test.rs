//! 간단한 RBE 정확도 테스트

use rbe_llm::{
    nlp::{
        linear::RBELinear,
        layernorm::{RBELayerNorm, RBELayerNormConfig},
        rmsnorm::{RBERMSNorm, RBERMSNormConfig},
        accuracy_utils::AccuracyMetrics,
    },
    core::{
        encoder::RBEEncoder,
        decoder::WeightGenerator,
    },
    QualityGrade,
};
use rand::{thread_rng, Rng};

fn main() -> anyhow::Result<()> {
    println!("=== RBE 레이어 정확도 검증 ===\n");
    
    // 1. 압축/복원 정확도 테스트
    test_compression_accuracy()?;
    
    // 2. LayerNorm 정확도 테스트
    test_layernorm_accuracy()?;
    
    // 3. RMSNorm 정확도 테스트
    test_rmsnorm_accuracy()?;
    
    Ok(())
}

fn test_compression_accuracy() -> anyhow::Result<()> {
    println!("1. RBE 압축/복원 정확도 테스트");
    
    for quality in [QualityGrade::S, QualityGrade::A, QualityGrade::B] {
        println!("\n품질 등급: {:?}", quality);
        
        // 테스트 데이터
        let mut rng = thread_rng();
        let original: Vec<f32> = (0..1024)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        
        // 압축
        let mut encoder = match quality {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            _ => RBEEncoder::new_b_grade(),
        };
        
        // 1차원 벡터로 인코딩
        let block = encoder.encode_vector(&original);
        
        // 복원
        let generator = WeightGenerator::new();
        let restored = generator.decode_block(&block);
        
        // 크기 맞추기
        let restored_slice = &restored[..original.len().min(restored.len())];
        
        // 정확도 계산
        let metrics = AccuracyMetrics::calculate(&original, restored_slice);
        metrics.report(&format!("압축/복원 ({:?})", quality));
    }
    
    Ok(())
}

fn test_layernorm_accuracy() -> anyhow::Result<()> {
    println!("\n2. LayerNorm 정확도 테스트");
    
    let config = RBELayerNormConfig {
        normalized_shape: vec![768],
        eps: 1e-5,
        elementwise_affine: false, // gamma/beta 없이 순수 정규화만
        use_fused_ops: false,
    };
    
    let layernorm = RBELayerNorm::new(config.clone())?;
    
    // 테스트 입력
    let mut rng = thread_rng();
    let input: Vec<f32> = (0..768)
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();
    
    // 수동 계산
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / input.len() as f32;
    
    let std_dev = (variance + config.eps as f32).sqrt();
    let expected: Vec<f32> = input.iter()
        .map(|x| (x - mean) / std_dev)
        .collect();
    
    // RBE 계산
    let output = layernorm.forward(&input)?;
    
    // 정확도 측정
    let metrics = AccuracyMetrics::calculate(&expected, &output);
    metrics.report("LayerNorm");
    
    Ok(())
}

fn test_rmsnorm_accuracy() -> anyhow::Result<()> {
    println!("\n3. RMSNorm 정확도 테스트");
    
    let config = RBERMSNormConfig {
        normalized_shape: 768,
        epsilon: 1e-5,
        quality_grade: QualityGrade::A,
        enable_parallel: false,
    };
    
    let mut rmsnorm = RBERMSNorm::new(config.clone());
    rmsnorm.init_weights()?;
    
    // 테스트 입력
    let mut rng = thread_rng();
    let input: Vec<f32> = (0..768)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // RMS 수동 계산 (gamma = 1로 가정)
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + config.epsilon).sqrt();
    let expected: Vec<f32> = input.iter()
        .map(|x| x / rms)
        .collect();
    
    // RBE 계산
    let output = rmsnorm.forward(&input)?;
    
    // gamma가 1 근처이므로 근사적으로 비교
    println!("\nRMSNorm 검증:");
    println!("- 입력 RMS: {:.6}", rms);
    println!("- 출력 평균: {:.6}", output.iter().sum::<f32>() / output.len() as f32);
    println!("- 출력 분산: {:.6}", 
        output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32);
    
    // 대략적인 정확도 (gamma 때문에 정확한 비교는 어려움)
    let approx_metrics = AccuracyMetrics::calculate(&expected, &output);
    println!("- 근사 RMSE: {:.6}", approx_metrics.rmse);
    println!("- 상대 오차: {:.2}%", approx_metrics.relative_error * 100.0);
    
    Ok(())
} 