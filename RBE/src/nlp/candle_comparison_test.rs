use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{Module, VarBuilder, VarMap};
use crate::nlp::{
    embedding::RBEEmbedding,
    layernorm::RBELayerNorm,
    ffn::RBEFFN,
    attention::RBEAttention,
    dropout::RBEDropout,
};

/// Candle과 RBE 구현 비교 테스트
#[cfg(test)]
mod candle_comparison_tests {
    use super::*;

    #[test]
    fn 임베딩_레이어_candle_비교_테스트() -> Result<()> {
        // 설정
        let vocab_size = 1000;
        let embedding_dim = 128;
        let seq_len = 10;
        let device = Device::Cpu;
        
        // Candle 임베딩 생성
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let candle_embedding = candle_nn::embedding(vocab_size, embedding_dim, vs)?;
        
        // RBE 임베딩 생성 (같은 가중치로 초기화)
        let mut rbe_config = crate::nlp::embedding::RBEEmbeddingConfig {
            vocab_size,
            embedding_dim,
            max_position_embeddings: 512,
            block_size: 64,
            quality_grade: crate::QualityGrade::A,
            enable_parallel: true,
            cache_size: 100,
        };
        
        // 같은 가중치로 초기화
        let weights = candle_embedding.embeddings().to_vec2::<f32>()?;
        let flat_weights: Vec<f32> = weights.into_iter().flatten().collect();
        let rbe_embedding = RBEEmbedding::from_pretrained_weights(&flat_weights, None, rbe_config)?;
        
        // 테스트 입력
        let token_ids = vec![1u32, 5, 10, 100, 500];
        let input_tensor = Tensor::new(token_ids.clone(), &device)?;
        
        // Candle forward
        let candle_output = candle_embedding.forward(&input_tensor)?;
        let candle_result = candle_output.to_vec2::<f32>()?;
        
        // RBE forward
        let rbe_output = rbe_embedding.forward(&token_ids)?;
        let rbe_result: Vec<Vec<f32>> = token_ids.iter()
            .map(|&id| {
                rbe_output[id as usize * embedding_dim..(id as usize + 1) * embedding_dim].to_vec()
            })
            .collect();
        
        // 비교 (압축으로 인한 오차 허용)
        for (i, (candle_row, rbe_row)) in candle_result.iter().zip(&rbe_result).enumerate() {
            let mse: f32 = candle_row.iter()
                .zip(rbe_row)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>() / embedding_dim as f32;
            
            println!("Token {}: MSE = {}", token_ids[i], mse);
            assert!(mse < 0.001, "MSE too high: {}", mse);
        }
        
        // 압축률 확인
        let (compressed_size, compression_ratio) = rbe_embedding.memory_usage();
        println!("압축률: {:.2}:1, 압축 크기: {} bytes", compression_ratio, compressed_size);
        assert!(compression_ratio > 10.0, "압축률이 너무 낮음");
        
        Ok(())
    }

    #[test]
    fn 레이어놈_candle_비교_테스트() -> Result<()> {
        let hidden_size = 768;
        let device = Device::Cpu;
        
        // Candle LayerNorm 생성
        let eps = 1e-5;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let candle_ln = candle_nn::layer_norm(hidden_size, eps, vs)?;
        
        // RBE LayerNorm 생성
        let rbe_ln = RBELayerNorm::new(crate::nlp::layernorm::RBELayerNormConfig {
            normalized_shape: vec![hidden_size],
            eps,
            elementwise_affine: true,
            use_fused_ops: true,
        })?;
        
        // 테스트 입력 (배치 크기 4)
        let input_data: Vec<f32> = (0..4 * hidden_size)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let input_tensor = Tensor::from_vec(input_data.clone(), (4, hidden_size), &device)?;
        
        // Candle forward
        let candle_output = candle_ln.forward(&input_tensor)?;
        let candle_result = candle_output.to_vec2::<f32>()?;
        
        // RBE forward (배치 처리)
        let mut rbe_result = Vec::new();
        for i in 0..4 {
            let start = i * hidden_size;
            let end = (i + 1) * hidden_size;
            let output = rbe_ln.forward(&input_data[start..end])?;
            rbe_result.push(output);
        }
        
        // 비교
        for (i, (candle_row, rbe_row)) in candle_result.iter().zip(&rbe_result).enumerate() {
            let max_diff: f32 = candle_row.iter()
                .zip(rbe_row)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            
            println!("Batch {}: Max diff = {}", i, max_diff);
            assert!(max_diff < 1e-4, "Difference too large: {}", max_diff);
        }
        
        Ok(())
    }

    #[test]
    fn FFN_레이어_candle_비교_테스트() -> Result<()> {
        let hidden_dim = 768;
        let intermediate_dim = 3072;
        let device = Device::Cpu;
        
        // Candle FFN (2개의 Linear + GELU)
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let linear1 = candle_nn::linear(hidden_dim, intermediate_dim, vs.pp("fc1"))?;
        let linear2 = candle_nn::linear(intermediate_dim, hidden_dim, vs.pp("fc2"))?;
        
        // RBE FFN 생성
        let ffn_config = crate::nlp::ffn::RBEFFNConfig {
            hidden_dim,
            intermediate_dim,
            activation: crate::nlp::ffn::ActivationType::Gelu,
            dropout: 0.0,
            block_size: 128,
            quality_grade: crate::QualityGrade::A,
            enable_parallel: true,
            cache_size: 100,
        };
        
        // Candle 가중치 추출 및 RBE 초기화
        let w1 = linear1.weight().to_vec2::<f32>()?;
        let w2 = linear2.weight().to_vec2::<f32>()?;
        let w1_flat: Vec<f32> = w1.into_iter().flatten().collect();
        let w2_flat: Vec<f32> = w2.into_iter().flatten().collect();
        
        let mut rbe_ffn = RBEFFN::from_pretrained_weights(&w1_flat, &w2_flat, ffn_config)?;
        
        // 테스트 입력
        let input_data: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i as f32) * 0.1).cos())
            .collect();
        let input_tensor = Tensor::from_vec(input_data.clone(), hidden_dim, &device)?;
        
        // Candle forward
        let x1 = linear1.forward(&input_tensor)?;
        let x2 = x1.gelu()?; // gelu는 Tensor의 메서드
        let candle_output = linear2.forward(&x2)?;
        let candle_result = candle_output.to_vec1::<f32>()?;
        
        // RBE forward
        let rbe_result = rbe_ffn.forward(&input_data)?;
        
        // 비교
        let mse: f32 = candle_result.iter()
            .zip(&rbe_result)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / hidden_dim as f32;
        
        println!("FFN MSE: {}", mse);
        assert!(mse < 0.01, "MSE too high: {}", mse);
        
        // 압축률 확인
        let (compressed_size, compression_ratio) = rbe_ffn.memory_usage();
        println!("FFN 압축률: {:.2}:1", compression_ratio);
        assert!(compression_ratio > 50.0, "압축률이 너무 낮음");
        
        Ok(())
    }

    #[test]
    fn 드롭아웃_candle_비교_테스트() -> Result<()> {
        // Dropout은 확률적이므로 통계적 비교
        let dropout_prob = 0.5;
        let size = 10000;
        
        // RBE Dropout
        let mut rbe_dropout = RBEDropout::new(dropout_prob)?;
        rbe_dropout.set_training(true);
        
        let input: Vec<f32> = vec![1.0; size];
        
        // 여러 번 실행하여 통계 수집
        let mut rbe_zeros = 0;
        let num_trials = 100;
        
        for _ in 0..num_trials {
            let output = rbe_dropout.forward(&input);
            rbe_zeros += output.iter().filter(|&&x| x == 0.0).count();
        }
        
        let rbe_drop_rate = rbe_zeros as f32 / (size * num_trials) as f32;
        
        println!("RBE Dropout rate: {:.3}", rbe_drop_rate);
        
        // 드롭률이 목표값의 ±5% 이내
        assert!((rbe_drop_rate - dropout_prob).abs() < 0.05);
        
        // 푸앵카레 마스크 테스트
        let poincare_mask = rbe_dropout.generate_poincare_mask(1000);
        let center_rate = poincare_mask[0..100].iter().filter(|&&x| x).count() as f32 / 100.0;
        let boundary_rate = poincare_mask[900..1000].iter().filter(|&&x| x).count() as f32 / 100.0;
        
        println!("Center drop rate: {:.3}, Boundary drop rate: {:.3}", center_rate, boundary_rate);
        
        // 경계에서 더 많이 드롭되어야 함
        assert!(boundary_rate > center_rate * 1.2);
        
        Ok(())
    }
} 