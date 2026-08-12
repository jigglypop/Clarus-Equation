//! NLP 레이어별 사용 예제

use anyhow::Result;
use layer::{
    core::encoder::QualityGrade,
    nlp::{
        embedding::{RBEEmbedding, RBEEmbeddingConfig},
        layernorm::{RBELayerNorm, RBELayerNormConfig},
        ffn::{RBEFFN, RBEFFNConfig, ActivationType},
        attention::{RBEAttention, RBEAttentionConfig},
    },
};

/// RBEEmbedding 사용 예제
fn embedding_example() -> Result<()> {
    println!("\n=== RBEEmbedding 예제 ===");
    
    let config = RBEEmbeddingConfig {
        vocab_size: 1000,
        embedding_dim: 128,
        max_position_embeddings: 512,
        block_size: 64,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 16,
    };
    
    let mut embedding = RBEEmbedding::new(config)?;
    embedding.init_random()?;
    
    // 토큰 ID 입력
    let token_ids = vec![101, 234, 567, 102];  // [CLS] ... [SEP]
    let output = embedding.forward(&token_ids)?;
    
    println!("입력 토큰 수: {}", token_ids.len());
    println!("출력 크기: {} ({}x{})", output.len(), token_ids.len(), config.embedding_dim);
    
    let (size, ratio) = embedding.memory_usage();
    println!("압축 크기: {:.2} MB, 압축률: {:.1}:1", 
             size as f32 / 1024.0 / 1024.0, ratio);
    
    Ok(())
}

/// RBELayerNorm 사용 예제
fn layernorm_example() -> Result<()> {
    println!("\n=== RBELayerNorm 예제 ===");
    
    let config = RBELayerNormConfig {
        normalized_shape: vec![768],  // GPT-2 hidden size
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 배치 입력 (batch_size=2, hidden_dim=768)
    let mut input = vec![1.0, 2.0, 3.0, 4.0; 192];  // 768 elements
    input.extend(vec![5.0, 6.0, 7.0, 8.0; 192]);
    
    let output = layer_norm.forward(&input)?;
    
    println!("입력 크기: {}", input.len());
    println!("정규화 후 첫 번째 샘플 평균: {:.6}", 
             output[..768].iter().sum::<f32>() / 768.0);
    println!("정규화 후 첫 번째 샘플 표준편차: {:.6}", 
             calculate_std(&output[..768]));
    
    Ok(())
}

/// RBEFFN 사용 예제
fn ffn_example() -> Result<()> {
    println!("\n=== RBEFFN 예제 ===");
    
    let config = RBEFFNConfig {
        hidden_dim: 768,
        intermediate_dim: 3072,  // 4x hidden
        activation: ActivationType::GeluNew,
        dropout: 0.0,
        block_size: 256,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let mut ffn = RBEFFN::new(config)?;
    ffn.init_random()?;
    
    // 입력 (batch_size=2, hidden_dim=768)
    let input = vec![0.5; 2 * 768];
    let output = ffn.forward(&input)?;
    
    println!("입력 크기: {} -> 중간층: {} -> 출력 크기: {}", 
             input.len(), config.intermediate_dim, output.len());
    
    let (size, ratio) = ffn.memory_usage();
    println!("FFN 압축 크기: {:.2} MB, 압축률: {:.1}:1", 
             size as f32 / 1024.0 / 1024.0, ratio);
    
    Ok(())
}

/// RBEAttention 사용 예제
fn attention_example() -> Result<()> {
    println!("\n=== RBEAttention 예제 ===");
    
    let config = RBEAttentionConfig {
        hidden_dim: 768,
        num_heads: 12,
        head_dim: 64,  // 768 / 12
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 128,
        quality_grade: QualityGrade::B,
        enable_parallel: true,
        cache_size: 32,
    };
    
    let mut attention = RBEAttention::new(config)?;
    attention.init_random()?;
    
    // 입력 (seq_len=16, hidden_dim=768)
    let seq_len = 16;
    let hidden_states = vec![0.1; seq_len * config.hidden_dim];
    
    // Causal mask 생성 (하삼각 행렬)
    let mut causal_mask = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            causal_mask[i * seq_len + j] = 1.0;
        }
    }
    
    let output = attention.forward(&hidden_states, Some(&causal_mask))?;
    
    println!("시퀀스 길이: {}, 헤드 수: {}", seq_len, config.num_heads);
    println!("입력 크기: {} -> 출력 크기: {}", hidden_states.len(), output.len());
    
    let (size, ratio) = attention.memory_usage();
    println!("Attention 압축 크기: {:.2} MB, 압축률: {:.1}:1", 
             size as f32 / 1024.0 / 1024.0, ratio);
    
    Ok(())
}

/// 간단한 Transformer 블록 예제
fn transformer_block_example() -> Result<()> {
    println!("\n=== 간단한 Transformer 블록 예제 ===");
    
    let hidden_dim = 256;
    let seq_len = 8;
    
    // 각 레이어 초기화
    let mut attention = RBEAttention::new(RBEAttentionConfig {
        hidden_dim,
        num_heads: 8,
        head_dim: 32,
        attention_dropout: 0.0,
        output_dropout: 0.0,
        block_size: 64,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 8,
    })?;
    attention.init_random()?;
    
    let ln1 = RBELayerNorm::new(RBELayerNormConfig {
        normalized_shape: vec![hidden_dim],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    })?;
    
    let mut ffn = RBEFFN::new(RBEFFNConfig {
        hidden_dim,
        intermediate_dim: hidden_dim * 4,
        activation: ActivationType::Gelu,
        dropout: 0.0,
        block_size: 64,
        quality_grade: QualityGrade::C,
        enable_parallel: false,
        cache_size: 8,
    })?;
    ffn.init_random()?;
    
    let ln2 = RBELayerNorm::new(RBELayerNormConfig {
        normalized_shape: vec![hidden_dim],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    })?;
    
    // 입력
    let mut hidden_states = vec![0.5; seq_len * hidden_dim];
    
    // Transformer 블록 순전파
    // 1. Self-Attention + Residual
    let attn_out = attention.forward(&hidden_states, None)?;
    for i in 0..hidden_states.len() {
        hidden_states[i] += attn_out[i];  // Residual connection
    }
    
    // 2. LayerNorm
    hidden_states = ln1.forward(&hidden_states)?;
    
    // 3. FFN + Residual
    let ffn_out = ffn.forward(&hidden_states)?;
    for i in 0..hidden_states.len() {
        hidden_states[i] += ffn_out[i];  // Residual connection
    }
    
    // 4. LayerNorm
    let output = ln2.forward(&hidden_states)?;
    
    println!("Transformer 블록 입출력 크기: {}", output.len());
    println!("처리된 시퀀스 길이: {}", seq_len);
    
    Ok(())
}

/// 표준편차 계산 헬퍼
fn calculate_std(data: &[f32]) -> f32 {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / data.len() as f32;
    variance.sqrt()
}

fn main() -> Result<()> {
    println!("RBE NLP 레이어 데모\n");
    println!("이 예제는 RBE로 압축된 NLP 레이어들의 사용법을 보여줍니다.");
    println!("각 레이어는 극한 압축을 유지하면서도 직접 연산이 가능합니다.");
    
    // 각 레이어 예제 실행
    embedding_example()?;
    layernorm_example()?;
    ffn_example()?;
    attention_example()?;
    transformer_block_example()?;
    
    println!("\n모든 예제 실행 완료!");
    
    Ok(())
} 