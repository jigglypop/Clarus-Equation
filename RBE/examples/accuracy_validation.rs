//! RBE 레이어 정확도 검증

use rbe_llm::{
    nlp::{
        linear::RBELinear,
        layernorm::{RBELayerNorm, RBELayerNormConfig},
        ffn::{RBEFFN, RBEFFNConfig, ActivationType},
        rmsnorm::{RBERMSNorm, RBERMSNormConfig},
        accuracy_utils::AccuracyMetrics,
    },
    core::{
        encoder::{RBEEncoder, MetricEncoder},
        decoder::WeightGenerator,
        packed_params::HybridEncodedBlock,
    },
    QualityGrade,
};
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Uniform};

fn main() -> anyhow::Result<()> {
    println!("=== RBE 레이어 정확도 검증 ===\n");
    
    // 1. RBELinear 정확도 검증
    test_linear_accuracy()?;
    
    // 2. RBELayerNorm 정확도 검증
    test_layernorm_accuracy()?;
    
    // 3. RBEFFN 정확도 검증
    test_ffn_accuracy()?;
    
    // 4. RBERMSNorm 정확도 검증
    test_rmsnorm_accuracy()?;
    
    Ok(())
}

fn test_linear_accuracy() -> anyhow::Result<()> {
    println!("\n1. RBELinear 정확도 검증");
    
    let mut rng = thread_rng();
    let uniform = Uniform::new(-0.1, 0.1);
    
    for quality in [QualityGrade::S, QualityGrade::A, QualityGrade::B] {
        let in_features = 768;
        let out_features = 3072;
        
        // 원본 가중치 생성
        let original_weights: Vec<f32> = (0..in_features * out_features)
            .map(|_| uniform.sample(&mut rng))
            .collect();
        
        // MetricEncoder로 압축
        let mut encoder = MetricEncoder::default();
        let blocks = encoder.encode_from_weights(&original_weights, out_features, in_features)?;
        
        // RBELinear 생성
        let linear = RBELinear::new(blocks.clone(), in_features, out_features, None);
        
        // 복원된 가중치 추출
        let mut weight_generator = WeightGenerator::new();
        let mut restored_weights = Vec::new();
        
        for block in &blocks {
            let decoded = weight_generator.decode_block(block);
            restored_weights.extend_from_slice(&decoded);
        }
        
        // 크기 조정
        restored_weights.truncate(in_features * out_features);
        
        // 정확도 계산
        let metrics = AccuracyMetrics::calculate(&original_weights, &restored_weights);
        metrics.report(&format!("RBELinear ({:?})", quality));
    }
    
    Ok(())
}

fn test_layernorm_accuracy() -> anyhow::Result<()> {
    println!("\n2. RBELayerNorm 정확도 검증");
    
    let config = RBELayerNormConfig {
        normalized_shape: vec![768],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: false,
    };
    
    let layernorm = RBELayerNorm::new(config.clone())?;
    
    // 테스트 입력
    let mut rng = thread_rng();
    let input: Vec<f32> = (0..768)
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();
    
    // LayerNorm 수동 계산
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / input.len() as f32;
    
    let std_dev = (variance + config.eps as f32).sqrt();
    let manual_norm: Vec<f32> = input.iter()
        .map(|x| (x - mean) / std_dev)
        .collect();
    
    // RBE 계산
    let rbe_output = layernorm.forward(&input)?;
    
    let metrics = AccuracyMetrics::calculate(&manual_norm, &rbe_output);
    metrics.report("RBELayerNorm");
    
    Ok(())
}

fn test_ffn_accuracy() -> anyhow::Result<()> {
    println!("\n3. RBEFFN 정확도 검증");
    
    let config = RBEFFNConfig {
        hidden_dim: 768,
        intermediate_dim: 3072,
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 64,
        enable_parallel: false,
        cache_size: 16,
        quality_grade: QualityGrade::A,
    };
    
    let mut ffn = RBEFFN::new(config.clone())?;
    ffn.init_random()?;
    
    // 테스트 입력
    let input = vec![0.5; 768];
    
    // FFN forward
    let output = ffn.forward(&input)?;
    
    // 기본 검증: 출력 크기
    assert_eq!(output.len(), 768);
    
    // GELU 검증을 위한 간단한 테스트
    println!("\nFFN 출력 샘플:");
    println!("입력: 0.5 (모든 원소)");
    println!("출력 평균: {:.6}", output.iter().sum::<f32>() / output.len() as f32);
    println!("출력 범위: [{:.6}, {:.6}]", 
        output.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    
    Ok(())
}

fn test_rmsnorm_accuracy() -> anyhow::Result<()> {
    println!("\n4. RBERMSNorm 정확도 검증");
    
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
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();
    
    // RMS 수동 계산
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + config.epsilon).sqrt();
    let manual_norm: Vec<f32> = input.iter()
        .map(|x| x / rms)
        .collect();
    
    // RBE 계산
    let rbe_output = rmsnorm.forward(&input)?;
    
    // gamma가 1 근처이므로 대략적인 비교
    let metrics = AccuracyMetrics::calculate(&manual_norm, &rbe_output);
    println!("\nRBERMSNorm 정확도 (gamma 적용 전):");
    println!("RMSE: {:.6}", metrics.rmse);
    println!("상대 오차: {:.2}%", metrics.relative_error * 100.0);
    
    Ok(())
} 