use crate::nlp::model_tools::{ModelAnalyzer, ModelInfo, CompressionSuitability, QualityGrade};
use std::path::PathBuf;
use tokio;

#[tokio::test]
async fn 분석기_생성_테스트() {
    let analyzer = ModelAnalyzer::new();
    
    // 기본 상태 확인
    assert!(analyzer.analysis_cache.is_empty());
}

#[tokio::test] 
async fn 모델_정보_추출_테스트() {
    let mut analyzer = ModelAnalyzer::new();
    
    // 테스트용 임시 디렉토리 생성
    let temp_dir = std::env::temp_dir().join("test_model");
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();
    
    // 가짜 config.json 생성
    let config_content = r#"
    {
        "model_type": "bert",
        "architectures": ["BertModel"],
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072
    }
    "#;
    
    let config_path = temp_dir.join("config.json");
    tokio::fs::write(&config_path, config_content).await.unwrap();
    
    // 가짜 모델 파일 생성 (크기 계산용)
    let model_file = temp_dir.join("pytorch_model.bin");
    tokio::fs::write(&model_file, vec![0u8; 1024 * 1024]).await.unwrap(); // 1MB
    
    // 모델 정보 추출 테스트
    let model_info = analyzer.extract_model_info(&temp_dir).await.unwrap();
    
    assert_eq!(model_info.model_type, "bert");
    assert_eq!(model_info.architecture, "BertModel");
    assert_eq!(model_info.vocab_size, Some(32000));
    assert_eq!(model_info.hidden_size, Some(768));
    assert_eq!(model_info.num_layers, Some(12));
    assert!(model_info.total_parameters > 0);
    assert!(model_info.model_size_mb > 0.0);
    
    // 정리
    tokio::fs::remove_dir_all(&temp_dir).await.unwrap();
}

#[tokio::test]
async fn 파라미터_추정_테스트() {
    let analyzer = ModelAnalyzer::new();
    
    // 테스트 config
    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 3072
    });
    
    let total_params = analyzer.estimate_parameters_from_config(&config).unwrap();
    
    // BERT-base 수준의 파라미터 수 (대략 110M)
    assert!(total_params > 100_000_000);
    assert!(total_params < 150_000_000);
}

#[tokio::test]
async fn 압축_적합성_분석_테스트() {
    let analyzer = ModelAnalyzer::new();
    
    // 작은 모델 (23M 파라미터)
    let small_model_info = ModelInfo {
        model_name: "KoMiniLM".to_string(),
        model_type: "bert".to_string(),
        total_parameters: 23_000_000,
        model_size_mb: 92.0,
        architecture: "BertModel".to_string(),
        vocab_size: Some(32000),
        hidden_size: Some(384),
        num_layers: Some(6),
        num_attention_heads: Some(12),
    };
    
    // 가짜 레이어 분석
    let mut layer_types = std::collections::HashMap::new();
    layer_types.insert("attention".to_string(), 6);
    layer_types.insert("feed_forward".to_string(), 6);
    
    let layer_analysis = crate::nlp::model_tools::LayerAnalysis {
        layer_types,
        layer_parameters: std::collections::HashMap::new(),
        largest_layers: vec![],
        compression_candidates: vec![],
    };
    
    let compression_result = analyzer.analyze_compression_suitability(&small_model_info, &layer_analysis).unwrap();
    
    // 작은 모델은 높은 적합성을 가져야 함
    assert!(compression_result.overall_score > 0.8);
    assert!(compression_result.rbe_suitability > 0.8);
    assert_eq!(compression_result.recommended_block_size, 32); // 작은 모델 = 작은 블록
    assert!(compression_result.estimated_compression_ratio > 4.0);
}

#[tokio::test]
async fn 성능_추정_테스트() {
    let analyzer = ModelAnalyzer::new();
    
    let model_info = ModelInfo {
        model_name: "test_model".to_string(),
        model_type: "bert".to_string(),
        total_parameters: 23_000_000, // 23M
        model_size_mb: 92.0,
        architecture: "BertModel".to_string(),
        vocab_size: Some(32000),
        hidden_size: Some(384),
        num_layers: Some(6),
        num_attention_heads: Some(12),
    };
    
    let performance = analyzer.estimate_performance(&model_info).unwrap();
    
    // 23M 파라미터 모델의 성능 추정 검증
    assert!(performance.inference_speed_ms > 0.0);
    assert!(performance.inference_speed_ms < 10.0); // 작은 모델은 빨라야 함
    assert!(performance.memory_usage_mb > 0.0);
    assert!(performance.gpu_memory_mb > performance.memory_usage_mb); // GPU 메모리가 더 많아야 함
    assert!(performance.throughput_tokens_per_sec > 100.0); // 최소 100 토큰/초
}

#[test]
fn 분석결과_출력_테스트() {
    let analyzer = ModelAnalyzer::new();
    
    // 테스트용 분석 결과 생성
    let model_info = ModelInfo {
        model_name: "KoMiniLM-23M".to_string(),
        model_type: "bert".to_string(),
        total_parameters: 23_000_000,
        model_size_mb: 92.0,
        architecture: "BertModel".to_string(),
        vocab_size: Some(32000),
        hidden_size: Some(384),
        num_layers: Some(6),
        num_attention_heads: Some(12),
    };
    
    let compression_suitability = CompressionSuitability {
        overall_score: 0.9,
        rbe_suitability: 0.9,
        recommended_block_size: 32,
        estimated_compression_ratio: 5.0,
        bottleneck_layers: vec!["attention".to_string()],
        memory_reduction_estimate: 0.8,
    };
    
    let performance_estimate = crate::nlp::model_tools::PerformanceEstimate {
        inference_speed_ms: 2.3,
        memory_usage_mb: 92.0,
        gpu_memory_mb: 138.0,
        throughput_tokens_per_sec: 434.8,
    };
    
    let layer_analysis = crate::nlp::model_tools::LayerAnalysis {
        layer_types: {
            let mut map = std::collections::HashMap::new();
            map.insert("attention".to_string(), 6);
            map.insert("feed_forward".to_string(), 6);
            map
        },
        layer_parameters: std::collections::HashMap::new(),
        largest_layers: vec![],
        compression_candidates: vec![],
    };
    
    let parameter_analysis = crate::nlp::model_tools::ParameterAnalysis {
        total_parameters: 23_000_000,
        trainable_parameters: 23_000_000,
        embedding_parameters: 12_288_000,
        linear_parameters: 8_000_000,
        attention_parameters: 11_500_000,
        parameter_distribution: crate::nlp::model_tools::ParameterDistribution {
            mean: 0.0,
            std: 0.1,
            min: -1.0,
            max: 1.0,
            sparsity_ratio: 0.05,
        },
    };
    
    let analysis = crate::nlp::model_tools::ModelAnalysis {
        model_info,
        layer_analysis,
        parameter_analysis,
        compression_suitability,
        performance_estimate,
    };
    
    // 출력 테스트 (패닉 없이 실행되는지 확인)
    analyzer.print_analysis(&analysis);
    
    // 기본 검증
    assert_eq!(analysis.model_info.model_name, "KoMiniLM-23M");
    assert!(analysis.compression_suitability.overall_score > 0.8);
} 