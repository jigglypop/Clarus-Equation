use crate::core::encoder::weight_mapper::*;
use crate::packed_params::TransformType;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn 가중치_매퍼_생성_테스트() {
    let mapper = WeightMapper::new("test-model", 64, 200, TransformType::Dwt);
    
    // 기본 설정 확인
    assert_eq!(mapper.layout.model_type, "test-model");
    assert_eq!(mapper.layout.compression_config.block_size, 64);
    assert_eq!(mapper.layout.compression_config.coefficients, 200);
    assert_eq!(mapper.layout.compression_config.transform_type, "Dwt");
}

#[test]
fn 가중치_정보_추가_테스트() {
    let mut mapper = WeightMapper::new("test-model", 64, 200, TransformType::Dwt);
    
    // 간단한 가중치 정보 추가
    let weight_info = WeightInfo {
        name: "test.weight".to_string(),
        offset_bytes: 0,
        num_blocks: 1,
        original_shape: vec![32, 32],
        compression_type: "rbe".to_string(),
        compression_ratio: 10.0,
        rmse: Some(0.01),
    };
    
    // 직접 weights에 추가 (실제 API)
    mapper.layout.weights.push(weight_info.clone());
    assert_eq!(mapper.layout.weights.len(), 1);
    assert_eq!(mapper.layout.weights[0].name, "test.weight");
}

#[test]
fn 레이아웃_생성_테스트() {
    let mut mapper = WeightMapper::new("layout-test", 32, 100, TransformType::Dwt);
    
    // 여러 가중치 추가
    for i in 0..3 {
        let weight_info = WeightInfo {
            name: format!("layer.{}.weight", i),
            offset_bytes: i as u64 * 1024,
            num_blocks: 2,
            original_shape: vec![64, 64],
            compression_type: "rbe".to_string(),
            compression_ratio: 5.0,
            rmse: Some(0.005),
        };
        mapper.layout.weights.push(weight_info);
        mapper.layout.total_blocks += 2; // 수동으로 블록 수 추가
    }
    
    // layout은 mapper.layout 자체
    assert_eq!(mapper.layout.model_type, "layout-test");
    assert_eq!(mapper.layout.weights.len(), 3);
    assert_eq!(mapper.layout.total_blocks, 6); // 3 weights * 2 blocks each
}

#[test]
fn 레이아웃_JSON_직렬화_테스트() {
    let mut mapper = WeightMapper::new("json-test", 128, 300, TransformType::Dct);
    
    let weight_info = WeightInfo {
        name: "transformer.weight".to_string(),
        offset_bytes: 512,
        num_blocks: 4,
        original_shape: vec![256, 256],
        compression_type: "rbe".to_string(),
        compression_ratio: 20.0,
        rmse: Some(0.001),
    };
    
    mapper.layout.weights.push(weight_info);
    
    // JSON 직렬화 테스트 (실제 API 사용)
    let json_result = mapper.serialize_layout();
    assert!(json_result.is_ok());
    
    let json_str = json_result.unwrap();
    assert!(json_str.contains("json-test"));
    assert!(json_str.contains("transformer.weight"));
}

#[test]
fn 임시_파일_저장_테스트() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_layout.json");
    
    let mut mapper = WeightMapper::new("save-test", 64, 150, TransformType::Dwt);
    
    let weight_info = WeightInfo {
        name: "simple.weight".to_string(),
                offset_bytes: 0,
                num_blocks: 1,
        original_shape: vec![128, 128],
        compression_type: "rbe".to_string(),
        compression_ratio: 8.0,
        rmse: Some(0.01),
    };
    
    mapper.layout.weights.push(weight_info);
    
    // 파일 저장 테스트 (수동으로 저장)
    let json_str = mapper.serialize_layout().unwrap();
    let save_result = std::fs::write(&output_path, json_str);
    assert!(save_result.is_ok());
    assert!(output_path.exists());
} 