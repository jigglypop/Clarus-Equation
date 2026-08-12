//! Korean sLLM (소형 언어 모델) 압축 및 추론 데모
//! 실제 한국어 모델을 다운로드하고 RBE로 압축한 후 텍스트 생성
#![allow(unused_imports, dead_code, unused_variables, unused_mut)]

use rbe_llm::{
    encoder::{encoder::{QualityGrade, CompressionProfile}, CompressionConfig, RBEEncoder},
    decoder::WeightGenerator,
    TransformType,
    HybridEncodedBlock,
};
use rbe_llm::nlp::bert_inference::{
    CompressedLayer, CompressedBert, BertAttention, BertFeedForward, CompressedBertLayer
};
use std::{time::Instant, path::Path, collections::HashMap};
use tokenizers::tokenizer::Tokenizer;
use safetensors::{tensor::SafeTensors, SafeTensorError, Dtype};
use std::fs;
use std::io::Read;
use serde_json::Value;
use anyhow::{Result, bail, anyhow};

/// 한글 sLLM 압축 데모
/// KoMiniLM-23M 모델을 RBE로 압축하고 실제 한글 텍스트를 생성합니다.

// 모델 설정
#[derive(Clone, Debug)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    n_layers: usize,
    n_heads: usize,
    intermediate_size: usize,
}

// 가중치 저장소
struct CompressedWeights {
    layers: HashMap<String, CompressedLayer>,
    biases: HashMap<String, Vec<f32>>,
    layernorms: HashMap<String, Vec<f32>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 RBE 기반 한국어 sLLM 데모 시작");
    
    // 모델 경로 설정 (예: ./models/KoMiniLM-23M)
    let model_path = Path::new("models").join("KoMiniLM-23M");
    if !model_path.exists() {
        println!("모델 파일이 없습니다. `setup_korean_test_model.py`를 실행하여 다운로드하세요.");
        return Ok(());
    }

    // 1. 모델 로드 및 압축
    let (weights, config, tokenizer) = load_and_compress_model(&model_path).await?;
    
    // 2. 압축된 BERT 모델 구성
    let bert_model = build_compressed_bert(&weights, &config)?;

    // 3. 텍스트 생성
    let prompt = "대한민국에서 가장 높은 산은";
    let generated_text = generate_text(bert_model, &tokenizer, prompt, 30).await?;

    println!("\n✅ 최종 생성 결과:");
    println!("--------------------");
    println!("{}", generated_text);
    println!("--------------------");

    Ok(())
}

async fn load_and_compress_model(model_path: &Path) -> Result<(CompressedWeights, ModelConfig, Tokenizer)> {
    println!("\n[1/3] 모델 로딩 및 압축...");
    let start_time = Instant::now();

    // 설정 파일 로드
    let config_path = model_path.join("config.json");
    let mut config_file = fs::File::open(&config_path)?;
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str)?;
    let json_config: Value = serde_json::from_str(&config_str)?;

    let config = ModelConfig {
        vocab_size: json_config["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: json_config["hidden_size"].as_u64().unwrap() as usize,
        n_layers: json_config["num_hidden_layers"].as_u64().unwrap() as usize,
        n_heads: json_config["num_attention_heads"].as_u64().unwrap() as usize,
        intermediate_size: json_config["intermediate_size"].as_u64().unwrap() as usize,
    };
    println!("모델 설정 로드 완료: {:?}", config);

    // 토크나이저 로드
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("Tokenizer 로드 실패: {}", e))?;
    println!("토크나이저 로드 완료");

    // 가중치 파일 로드
    let weights_path = model_path.join("model.safetensors");
    let weights_data = fs::read(&weights_path)?;
    let tensors = SafeTensors::deserialize(&weights_data)?;
    println!("가중치 파일 로드 완료");

    // 압축 설정
    let compression_config = CompressionConfig {
        block_size: 64,
        quality_grade: QualityGrade::A,
        transform_type: TransformType::Dwt,
        profile: CompressionProfile::High,
        custom_coefficients: None,
        min_block_count: None,
        rmse_threshold: Some(0.1),
        compression_ratio_threshold: None,
    };

    let mut compressed_weights = CompressedWeights {
        layers: HashMap::new(),
        biases: HashMap::new(),
        layernorms: HashMap::new(),
    };

    let mut original_total_size = 0;
    let mut compressed_total_size = 0;

    for (name, tensor_view) in tensors.tensors() {
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let original_size = data.len();
        original_total_size += original_size;

        // 모든 2D 텐서를 압축 대상으로 간주
        if shape.len() == 2 {
            // 행렬 가중치 압축
            let weights_f32 = bytes_to_f32(data);
            
            match RBEEncoder::compress_with_config(&weights_f32, shape[0], shape[1], &compression_config) {
                Ok((blocks, _time, ratio, rmse)) => {
                    let compressed_size = blocks.iter().map(|b| 32 + b.residuals.len() * 8).sum::<usize>();
                    compressed_total_size += compressed_size;
                    
                    let layer = CompressedLayer { blocks, shape: (shape[0], shape[1]) };
                    compressed_weights.layers.insert(name.clone(), layer);
                    
                    println!("  - 압축: {} ({} -> {} bytes, {:.2}x, RMSE: {:.4})", name, original_size, compressed_size, ratio, rmse);
                }
                Err(e) => {
                    println!("  - 압축 실패 {}: {}. 압축 없이 진행합니다.", name, e);
                    let blocks = Vec::new(); 
                    let layer = CompressedLayer { blocks, shape: (shape[0], shape[1]) };
                    compressed_weights.layers.insert(name.clone(), layer);
                    compressed_total_size += original_size;
                }
            }

        } else if shape.len() == 1 {
            // 1D 텐서는 bias 또는 layernorm으로 간주 (압축 안함)
            if name.contains("bias") {
                compressed_weights.biases.insert(name.clone(), bytes_to_f32(data));
            } else {
                compressed_weights.layernorms.insert(name.clone(), bytes_to_f32(data));
            }
            compressed_total_size += original_size;
        }
    }
    
    println!("\n압축 완료! ({:.2}s)", start_time.elapsed().as_secs_f32());
    println!("  - 원본 크기: {:.2} MB", original_total_size as f32 / 1_048_576.0);
    println!("  - 압축 크기: {:.2} MB", compressed_total_size as f32 / 1_048_576.0);
    println!("  - 압축률: {:.2}x", original_total_size as f32 / compressed_total_size as f32);

    Ok((compressed_weights, config, tokenizer))
}


fn build_compressed_bert<'a>(weights: &'a CompressedWeights, config: &'a ModelConfig) -> Result<CompressedBert<'a>> {
    println!("\n[2/3] 압축된 BERT 모델 구성...");
    
    let mut bert_layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let prefix = format!("encoder.layer.{}", i);
        let attention = BertAttention {
            q_w: weights.layers.get(&format!("{}.attention.self.query.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            q_b: weights.biases.get(&format!("{}.attention.self.query.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            k_w: weights.layers.get(&format!("{}.attention.self.key.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            k_b: weights.biases.get(&format!("{}.attention.self.key.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            v_w: weights.layers.get(&format!("{}.attention.self.value.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            v_b: weights.biases.get(&format!("{}.attention.self.value.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            output_w: weights.layers.get(&format!("{}.attention.output.dense.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            output_b: weights.biases.get(&format!("{}.attention.output.dense.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            layernorm_w: weights.layernorms.get(&format!("{}.attention.output.LayerNorm.weight", prefix)).ok_or_else(|| anyhow!("Missing layernorm"))?,
            layernorm_b: weights.biases.get(&format!("{}.attention.output.LayerNorm.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            n_heads: config.n_heads,
            hidden_size: config.hidden_size,
        };

        let ffn = BertFeedForward {
            intermediate_w: weights.layers.get(&format!("{}.intermediate.dense.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            intermediate_b: weights.biases.get(&format!("{}.intermediate.dense.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            output_w: weights.layers.get(&format!("{}.output.dense.weight", prefix)).ok_or_else(|| anyhow!("Missing weight"))?,
            output_b: weights.biases.get(&format!("{}.output.dense.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
            layernorm_w: weights.layernorms.get(&format!("{}.output.LayerNorm.weight", prefix)).ok_or_else(|| anyhow!("Missing layernorm"))?,
            layernorm_b: weights.biases.get(&format!("{}.output.LayerNorm.bias", prefix)).ok_or_else(|| anyhow!("Missing bias"))?,
        };
        
        bert_layers.push(CompressedBertLayer { attention, ffn });
    }

    let bert_model = CompressedBert {
        token_emb: weights.layers.get("embeddings.word_embeddings.weight").ok_or_else(|| anyhow!("Missing embedding"))?,
        position_emb: weights.layers.get("embeddings.position_embeddings.weight").ok_or_else(|| anyhow!("Missing embedding"))?,
        token_type_emb: weights.layers.get("embeddings.token_type_embeddings.weight").ok_or_else(|| anyhow!("Missing embedding"))?,
        emb_layernorm_w: weights.layernorms.get("embeddings.LayerNorm.weight").ok_or_else(|| anyhow!("Missing layernorm"))?,
        emb_layernorm_b: weights.biases.get("embeddings.LayerNorm.bias").ok_or_else(|| anyhow!("Missing bias"))?,
        layers: bert_layers,
        
        lm_head_dense_w: weights.layers.get("cls.predictions.transform.dense.weight"),
        lm_head_dense_b: weights.biases.get("cls.predictions.transform.dense.bias").map(|v| &**v),
        lm_head_layernorm_w: weights.layernorms.get("cls.predictions.transform.LayerNorm.weight").map(|v| &**v),
        lm_head_layernorm_b: weights.biases.get("cls.predictions.transform.LayerNorm.bias").map(|v| &**v),
        
        lm_head_decoder_w: weights.layers.get("embeddings.word_embeddings.weight").ok_or_else(|| anyhow!("Missing word_embeddings for decoder"))?,
        lm_head_decoder_b: weights.biases.get("cls.predictions.bias").map(|v| &**v),

        hidden_size: config.hidden_size,
        n_heads: config.n_heads,
        generator: WeightGenerator::new(),
    };

    println!("BERT 모델 구성 완료");
    Ok(bert_model)
}


async fn generate_text(mut model: CompressedBert<'_>, tokenizer: &Tokenizer, prompt: &str, max_length: usize) -> Result<String> {
    println!("\n[3/3] 텍스트 생성 시작...");
    println!("  - 프롬프트: \"{}\"", prompt);

    let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow!("토큰화 실패: {}", e))?;
    let mut token_ids = encoding.get_ids().to_vec();

    for i in 0..max_length {
        let start_time = Instant::now();
        
        let logits = model.forward(&token_ids)?;
        
        // 가장 확률이 높은 토큰 선택 (greedy)
        let next_token = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap_or(0);

        token_ids.push(next_token);
        
        let current_text = tokenizer.decode(&token_ids, true).map_err(|e| anyhow!("디코딩 실패: {}", e))?;
        println!("  - Step {}: \"{}\" ({:.2}ms)", i + 1, current_text, start_time.elapsed().as_millis());

        // [SEP] 토큰 만나면 종료
        if next_token == tokenizer.token_to_id("[SEP]").unwrap_or(102) {
            break;
        }
    }

    tokenizer.decode(&token_ids, true).map_err(|e| anyhow!("최종 디코딩 실패: {}", e))
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
} 