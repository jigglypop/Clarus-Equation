use std::collections::HashMap;
use std::path::PathBuf;
use anyhow::Result;
use serde_json::Value;

/// 품질 등급 (core와 호환성을 위해 추가)
#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
}

/// 모델 분석 결과
#[derive(Debug, Clone)]
pub struct ModelAnalysis {
    pub model_info: ModelInfo,
    pub layer_analysis: LayerAnalysis,
    pub parameter_analysis: ParameterAnalysis,
    pub compression_suitability: CompressionSuitability,
    pub performance_estimate: PerformanceEstimate,
}

/// 기본 모델 정보
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_name: String,
    pub model_type: String,
    pub total_parameters: u64,
    pub model_size_mb: f64,
    pub architecture: String,
    pub vocab_size: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_layers: Option<u32>,
    pub num_attention_heads: Option<u32>,
}

/// 레이어별 분석
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_types: HashMap<String, u32>,
    pub layer_parameters: HashMap<String, u64>,
    pub largest_layers: Vec<LayerInfo>,
    pub compression_candidates: Vec<LayerInfo>,
}

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub parameters: u64,
    pub shape: Vec<u32>,
    pub compression_ratio_estimate: f32,
}

/// 파라미터 분석
#[derive(Debug, Clone)]
pub struct ParameterAnalysis {
    pub total_parameters: u64,
    pub trainable_parameters: u64,
    pub embedding_parameters: u64,
    pub linear_parameters: u64,
    pub attention_parameters: u64,
    pub parameter_distribution: ParameterDistribution,
}

#[derive(Debug, Clone)]
pub struct ParameterDistribution {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity_ratio: f32,
}

/// 압축 적합성 분석
#[derive(Debug, Clone)]
pub struct CompressionSuitability {
    pub overall_score: f32,
    pub rbe_suitability: f32,
    pub recommended_block_size: u32,
    pub estimated_compression_ratio: f32,
    pub bottleneck_layers: Vec<String>,
    pub memory_reduction_estimate: f32,
}

/// 성능 추정
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    pub inference_speed_ms: f32,
    pub memory_usage_mb: f32,
    pub gpu_memory_mb: f32,
    pub throughput_tokens_per_sec: f32,
}

/// 모델 분석기
pub struct ModelAnalyzer {
    pub analysis_cache: HashMap<String, ModelAnalysis>,
}

impl ModelAnalyzer {
    /// 새로운 분석기 생성
    pub fn new() -> Self {
        Self {
            analysis_cache: HashMap::new(),
        }
    }
    
    /// 모델 경로로부터 분석 수행
    pub async fn analyze_model(&mut self, model_path: &PathBuf) -> Result<ModelAnalysis> {
        let model_name = model_path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
            
        // 캐시 확인
        if let Some(cached) = self.analysis_cache.get(&model_name) {
            return Ok(cached.clone());
        }
        
        println!("🔍 모델 분석 시작: {}", model_name);
        
        // 1. 기본 모델 정보 추출
        let model_info = self.extract_model_info(model_path).await?;
        println!("✓ 기본 정보 분석 완료");
        
        // 2. 레이어 분석
        let layer_analysis = self.analyze_layers(model_path).await?;
        println!("✓ 레이어 구조 분석 완료");
        
        // 3. 파라미터 분석
        let parameter_analysis = self.analyze_parameters(model_path, &model_info).await?;
        println!("✓ 파라미터 분석 완료");
        
        // 4. 압축 적합성 분석
        let compression_suitability = self.analyze_compression_suitability(&model_info, &layer_analysis)?;
        println!("✓ 압축 적합성 분석 완료");
        
        // 5. 성능 추정
        let performance_estimate = self.estimate_performance(&model_info)?;
        println!("✓ 성능 추정 완료");
        
        let analysis = ModelAnalysis {
            model_info,
            layer_analysis,
            parameter_analysis,
            compression_suitability,
            performance_estimate,
        };
        
        // 캐시에 저장
        self.analysis_cache.insert(model_name, analysis.clone());
        
        Ok(analysis)
    }
    
    /// HuggingFace 모델 ID로 분석 (다운로드 후)
    pub async fn analyze_huggingface_model(&mut self, model_id: &str) -> Result<ModelAnalysis> {
        use crate::nlp::model_tools::ModelDownloader;
        
        println!("📥 모델 다운로드 중: {}", model_id);
        let downloader = ModelDownloader::new(model_id);
        let model_path = downloader.download().await?;
        
        self.analyze_model(&model_path).await
    }
    
    /// KoMiniLM-23M 분석 (기본 추천 모델)
    pub async fn analyze_kominilm_23m(&mut self) -> Result<ModelAnalysis> {
        println!("🎯 KoMiniLM-23M 분석 시작 (기본 추천 모델)");
        self.analyze_huggingface_model("BM-K/KoMiniLM").await
    }
    
    /// config.json에서 모델 정보 추출
    pub async fn extract_model_info(&self, model_path: &PathBuf) -> Result<ModelInfo> {
        let config_path = model_path.join("config.json");
        
        if !config_path.exists() {
            return Ok(ModelInfo {
                model_name: model_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                model_type: "unknown".to_string(),
                total_parameters: 0,
                model_size_mb: 0.0,
                architecture: "unknown".to_string(),
                vocab_size: None,
                hidden_size: None,
                num_layers: None,
                num_attention_heads: None,
            });
        }
        
        let config_content = tokio::fs::read_to_string(&config_path).await?;
        let config: Value = serde_json::from_str(&config_content)?;
        
        // 모델 파일 크기 계산
        let model_size_mb = self.calculate_model_size(model_path).await?;
        
        // 파라미터 수 추정
        let total_parameters = self.estimate_parameters_from_config(&config)?;
        
        Ok(ModelInfo {
            model_name: model_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
            model_type: config.get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            total_parameters,
            model_size_mb,
            architecture: config.get("architectures")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.get(0))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            vocab_size: config.get("vocab_size").and_then(|v| v.as_u64()).map(|v| v as u32),
            hidden_size: config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as u32),
            num_layers: config.get("num_hidden_layers").and_then(|v| v.as_u64()).map(|v| v as u32),
            num_attention_heads: config.get("num_attention_heads").and_then(|v| v.as_u64()).map(|v| v as u32),
        })
    }
    
    /// 레이어 구조 분석
    async fn analyze_layers(&self, model_path: &PathBuf) -> Result<LayerAnalysis> {
        let mut layer_types = HashMap::new();
        let mut layer_parameters = HashMap::new();
        let mut largest_layers = Vec::new();
        
        // config.json에서 레이어 정보 추출
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            let config_content = tokio::fs::read_to_string(&config_path).await?;
            let config: Value = serde_json::from_str(&config_content)?;
            
            // 기본 레이어들 추정
            if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                layer_types.insert("attention".to_string(), num_layers as u32);
                layer_types.insert("feed_forward".to_string(), num_layers as u32);
                layer_types.insert("layer_norm".to_string(), num_layers as u32 * 2);
                
                // 파라미터 수 추정
                if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                    let attention_params = num_layers * hidden_size * hidden_size * 4; // Q, K, V, O
                    let ffn_params = num_layers * hidden_size * hidden_size * 8; // 대략적인 FFN 크기
                    
                    layer_parameters.insert("attention".to_string(), attention_params);
                    layer_parameters.insert("feed_forward".to_string(), ffn_params);
                    
                    // 큰 레이어들 식별
                    largest_layers.push(LayerInfo {
                        name: "transformer_layers".to_string(),
                        layer_type: "transformer_block".to_string(),
                        parameters: attention_params + ffn_params,
                        shape: vec![num_layers as u32, hidden_size as u32],
                        compression_ratio_estimate: 3.0, // 예상 압축률
                    });
                }
            }
        }
        
        // 압축 후보 레이어들 식별
        let compression_candidates = largest_layers.iter()
            .filter(|layer| layer.parameters > 1_000_000) // 1M 파라미터 이상
            .cloned()
            .collect();
        
        Ok(LayerAnalysis {
            layer_types,
            layer_parameters,
            largest_layers,
            compression_candidates,
        })
    }
    
    /// 파라미터 분석
    async fn analyze_parameters(&self, _model_path: &PathBuf, model_info: &ModelInfo) -> Result<ParameterAnalysis> {
        // 실제 가중치를 로드하지 않고 config 기반으로 추정
        let total_parameters = model_info.total_parameters;
        let trainable_parameters = total_parameters; // 대부분 훈련 가능
        
        // 레이어별 파라미터 분배 추정
        let vocab_size = model_info.vocab_size.unwrap_or(32000) as u64;
        let hidden_size = model_info.hidden_size.unwrap_or(768) as u64;
        
        let embedding_parameters = vocab_size * hidden_size * 2; // input + output embeddings
        let attention_parameters = total_parameters / 2; // 대략 절반이 attention
        let linear_parameters = total_parameters - embedding_parameters - attention_parameters;
        
        // 가상의 분포 통계 (실제로는 가중치를 로드해야 함)
        let parameter_distribution = ParameterDistribution {
            mean: 0.0,
            std: 0.1,
            min: -1.0,
            max: 1.0,
            sparsity_ratio: 0.05, // 5% 희소성 추정
        };
        
        Ok(ParameterAnalysis {
            total_parameters,
            trainable_parameters,
            embedding_parameters,
            linear_parameters: linear_parameters.max(0),
            attention_parameters,
            parameter_distribution,
        })
    }
    
    /// 압축 적합성 분석
    pub fn analyze_compression_suitability(&self, model_info: &ModelInfo, layer_analysis: &LayerAnalysis) -> Result<CompressionSuitability> {
        // 모델 크기에 따른 적합성 점수
        let size_score = match model_info.total_parameters {
            0..=50_000_000 => 0.9,      // 50M 이하: 매우 적합
            50_000_001..=200_000_000 => 0.8,   // 50M-200M: 적합
            200_000_001..=1_000_000_000 => 0.7, // 200M-1B: 보통
            _ => 0.6,                    // 1B 이상: 제한적
        };
        
        // 레이어 구조에 따른 RBE 적합성
        let has_attention = layer_analysis.layer_types.contains_key("attention");
        let has_ffn = layer_analysis.layer_types.contains_key("feed_forward");
        let rbe_suitability = if has_attention && has_ffn { 0.9 } else { 0.7 };
        
        // 권장 블록 크기 (모델 크기에 따라)
        let recommended_block_size = match model_info.total_parameters {
            0..=25_000_000 => 32,       // 25M 이하: 작은 블록
            25_000_001..=100_000_000 => 64,     // 25M-100M: 중간 블록
            _ => 128,                    // 100M 이상: 큰 블록
        };
        
        // 압축률 추정
        let estimated_compression_ratio = match model_info.total_parameters {
            0..=25_000_000 => 5.0,      // 작은 모델: 높은 압축률
            25_000_001..=100_000_000 => 4.0,    // 중간 모델: 중간 압축률
            _ => 3.0,                    // 큰 모델: 낮은 압축률
        };
        
        Ok(CompressionSuitability {
            overall_score: (size_score + rbe_suitability) / 2.0,
            rbe_suitability,
            recommended_block_size,
            estimated_compression_ratio,
            bottleneck_layers: vec!["attention".to_string(), "feed_forward".to_string()],
            memory_reduction_estimate: 1.0 - (1.0 / estimated_compression_ratio),
        })
    }
    
    /// 성능 추정
    pub fn estimate_performance(&self, model_info: &ModelInfo) -> Result<PerformanceEstimate> {
        // 파라미터 수 기반 성능 추정
        let params_millions = model_info.total_parameters as f32 / 1_000_000.0;
        
        // 추론 속도 (ms) - 파라미터 수에 비례
        let inference_speed_ms = params_millions * 0.1; // 대략적인 추정
        
        // 메모리 사용량 (MB)
        let memory_usage_mb = model_info.model_size_mb as f32;
        let gpu_memory_mb = memory_usage_mb * 1.5; // GPU는 추가 메모리 필요
        
        // 처리량 (tokens/sec)
        let throughput_tokens_per_sec = 1000.0 / inference_speed_ms;
        
        Ok(PerformanceEstimate {
            inference_speed_ms,
            memory_usage_mb,
            gpu_memory_mb,
            throughput_tokens_per_sec,
        })
    }
    
    /// 모델 파일 크기 계산
    async fn calculate_model_size(&self, model_path: &PathBuf) -> Result<f64> {
        let mut total_size = 0u64;
        
        // 일반적인 모델 파일들 확인
        let model_files = vec![
            "pytorch_model.bin",
            "model.safetensors", 
            "tf_model.h5",
            "model.onnx",
        ];
        
        for file_name in model_files {
            let file_path = model_path.join(file_name);
            if file_path.exists() {
                if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                    total_size += metadata.len();
                }
            }
        }
        
        Ok(total_size as f64 / (1024.0 * 1024.0)) // MB로 변환
    }
    
    /// config에서 파라미터 수 추정
    pub fn estimate_parameters_from_config(&self, config: &Value) -> Result<u64> {
        let vocab_size = config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000);
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768);
        let num_layers = config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(12);
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072);
        
        // 임베딩 파라미터
        let embedding_params = vocab_size * hidden_size * 2; // input + output
        
        // 트랜스포머 레이어 파라미터
        let attention_params = num_layers * hidden_size * hidden_size * 4; // Q, K, V, O
        let ffn_params = num_layers * (hidden_size * intermediate_size + intermediate_size * hidden_size);
        let norm_params = num_layers * hidden_size * 4; // layer norms
        
        Ok(embedding_params + attention_params + ffn_params + norm_params)
    }
    
    /// 분석 결과 출력
    pub fn print_analysis(&self, analysis: &ModelAnalysis) {
        println!("\n=== 📊 모델 분석 결과 ===");
        
        // 기본 정보
        println!("\n🏷️  기본 정보:");
        println!("   모델명: {}", analysis.model_info.model_name);
        println!("   아키텍처: {}", analysis.model_info.architecture);
        println!("   파라미터 수: {:.1}M", analysis.model_info.total_parameters as f64 / 1_000_000.0);
        println!("   모델 크기: {:.1} MB", analysis.model_info.model_size_mb);
        
        // 압축 적합성
        println!("\n🗜️  압축 적합성:");
        println!("   전체 점수: {:.1}/1.0", analysis.compression_suitability.overall_score);
        println!("   RBE 적합성: {:.1}/1.0", analysis.compression_suitability.rbe_suitability);
        println!("   권장 블록 크기: {}", analysis.compression_suitability.recommended_block_size);
        println!("   예상 압축률: {:.1}:1", analysis.compression_suitability.estimated_compression_ratio);
        println!("   메모리 절약률: {:.1}%", analysis.compression_suitability.memory_reduction_estimate * 100.0);
        
        // 성능 추정
        println!("\n⚡ 성능 추정:");
        println!("   추론 속도: {:.1} ms", analysis.performance_estimate.inference_speed_ms);
        println!("   메모리 사용량: {:.1} MB", analysis.performance_estimate.memory_usage_mb);
        println!("   GPU 메모리: {:.1} MB", analysis.performance_estimate.gpu_memory_mb);
        println!("   처리량: {:.0} tokens/sec", analysis.performance_estimate.throughput_tokens_per_sec);
        
        // 레이어 분석
        println!("\n🏗️  레이어 구조:");
        for (layer_type, count) in &analysis.layer_analysis.layer_types {
            println!("   {}: {} 개", layer_type, count);
        }
        
        println!("\n✅ 분석 완료!");
    }
} 