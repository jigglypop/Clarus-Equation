use crate::nlp::model_tools::*;
use anyhow::Result;
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compression_config_defaults() {
        println!("=== CompressionConfig 기본값 테스트 ===");
        
        let config = CompressionConfig::default();
        assert_eq!(config.block_size, 256);
        assert_eq!(config.coefficients, 500);
        assert_eq!(config.matrix_size, 768);
        assert_eq!(config.model_name, "kogpt2");
        
        println!("✓ 기본 설정 검증 완료");
    }
    
    #[test]
    fn test_compression_config_presets() {
        println!("=== CompressionConfig 프리셋 테스트 ===");
        
        let high_quality = CompressionConfig::high_quality();
        assert_eq!(high_quality.block_size, 128);
        assert_eq!(high_quality.coefficients, 800);
        
        let fast = CompressionConfig::fast();
        assert_eq!(fast.block_size, 512);
        assert_eq!(fast.coefficients, 200);
        
        let extreme = CompressionConfig::extreme();
        assert_eq!(extreme.block_size, 256);
        assert_eq!(extreme.coefficients, 100);
        
        println!("✓ 프리셋 설정 검증 완료");
    }
    
    #[test]
    fn test_model_compressor_creation() -> Result<()> {
        println!("=== ModelCompressor 생성 테스트 ===");
        
        let config = CompressionConfig::default();
        let compressor = ModelCompressor::new(config);
        
        assert_eq!(compressor.config.block_size, 256);
        assert_eq!(compressor.config.coefficients, 500);
        
        println!("✓ ModelCompressor 생성 완료");
        Ok(())
    }
    
    #[test]
    fn test_test_matrix_generation() -> Result<()> {
        println!("=== 테스트 행렬 생성 테스트 ===");
        
        let mut config = CompressionConfig::default();
        config.matrix_size = 64; // 작은 크기로 테스트
        
        let compressor = ModelCompressor::new(config);
        let matrix = compressor.generate_test_matrix();
        
        assert_eq!(matrix.len(), 64 * 64);
        
        // 행렬이 0이 아닌 값들을 포함하는지 확인
        let non_zero_count = matrix.iter().filter(|&&x| x.abs() > 1e-6).count();
        assert!(non_zero_count > 100, "생성된 행렬이 너무 많은 0값을 포함합니다");
        
        println!("✓ 테스트 행렬 생성 완료: {} 요소", matrix.len());
        println!("  - 0이 아닌 요소: {}/{}", non_zero_count, matrix.len());
        
        Ok(())
    }
    
    #[test]
    fn test_matrix_compression() -> Result<()> {
        println!("=== 행렬 압축 테스트 ===");
        
        let mut config = CompressionConfig::default();
        config.matrix_size = 128; // 작은 크기로 빠른 테스트
        config.block_size = 32;
        config.coefficients = 100;
        
        let mut compressor = ModelCompressor::new(config);
        let matrix = compressor.generate_test_matrix();
        
        let result = compressor.compress_matrix(&matrix)?;
        
        // 압축 결과 검증
        assert!(result.compression_ratio > 1.0, "압축률이 1보다 커야 합니다");
        assert!(result.compressed_size < result.original_size, "압축 크기가 원본보다 작아야 합니다");
        assert!(!result.encoded_blocks.is_empty(), "압축된 블록이 있어야 합니다");
        
        println!("✓ 행렬 압축 완료");
        println!("  - 압축률: {:.2}:1", result.compression_ratio);
        println!("  - 압축 시간: {:.3}초", result.compression_time);
        println!("  - 블록 수: {}", result.total_blocks);
        
        Ok(())
    }
    
    #[test]
    fn test_compression_quality_estimation() -> Result<()> {
        println!("=== 압축 품질 예측 테스트 ===");
        
        let configs = vec![
            CompressionConfig::high_quality(),
            CompressionConfig::fast(),
            CompressionConfig::extreme(),
        ];
        
        for (i, config) in configs.iter().enumerate() {
            let compressor = ModelCompressor::new(config.clone());
            let estimate = compressor.estimate_compression_quality();
            
            println!("설정 {}: {:?}", i + 1, estimate.quality_level);
            println!("  - 예상 압축률: {:.2}:1", estimate.estimated_ratio);
            println!("  - 예상 RMSE: {:.6}", estimate.estimated_rmse);
            println!("  - 블록 수: {}", estimate.blocks_count);
            
            assert!(estimate.estimated_ratio > 1.0);
            assert!(estimate.estimated_rmse > 0.0);
            assert!(estimate.blocks_count > 0);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_different_block_sizes() -> Result<()> {
        println!("=== 다양한 블록 크기 압축 테스트 ===");
        
        let matrix_size = 64;
        let block_sizes = vec![16, 32];
        
        for &block_size in &block_sizes {
            let mut config = CompressionConfig::default();
            config.matrix_size = matrix_size;
            config.block_size = block_size;
            config.coefficients = 50; // 빠른 테스트를 위해 감소
            
            let mut compressor = ModelCompressor::new(config);
            let matrix = compressor.generate_test_matrix();
            
            let result = compressor.compress_matrix(&matrix)?;
            
            println!("블록 크기 {}: 압축률 {:.2}:1, 블록 수 {}", 
                    block_size, result.compression_ratio, result.total_blocks);
            
            assert!(result.compression_ratio > 1.0);
            
            // 블록 크기가 작을수록 블록 수가 더 많아야 함
            let expected_blocks = (matrix_size + block_size - 1) / block_size;
            let expected_total = expected_blocks * expected_blocks;
            assert_eq!(result.total_blocks, expected_total);
        }
        
        Ok(())
    }
    
    #[test] 
    fn test_compress_and_save() -> Result<()> {
        println!("=== 압축 및 저장 테스트 ===");
        
        let mut config = CompressionConfig::default();
        config.matrix_size = 64; // 작은 크기
        config.block_size = 32;
        config.coefficients = 50;
        config.output_dir = PathBuf::from("./test_output");
        config.model_name = "test_model".to_string();
        
        let mut compressor = ModelCompressor::new(config);
        
        // 임시 디렉토리가 없어도 생성되는지 테스트
        let output_path = compressor.compress_and_save("test_compression")?;
        
        // 파일이 실제로 생성되었는지 확인
        assert!(output_path.exists(), "압축 파일이 생성되지 않았습니다");
        
        println!("✓ 압축 파일 저장 완료: {}", output_path.display());
        
        // 테스트 후 정리
        if output_path.exists() {
            std::fs::remove_file(&output_path)?;
        }
        if let Some(parent) = output_path.parent() {
            if parent.exists() && parent.read_dir()?.next().is_none() {
                std::fs::remove_dir(parent)?;
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_config_update() -> Result<()> {
        println!("=== 설정 업데이트 테스트 ===");
        
        let initial_config = CompressionConfig::fast();
        let mut compressor = ModelCompressor::new(initial_config);
        
        assert_eq!(compressor.config.coefficients, 200);
        
        let new_config = CompressionConfig::high_quality();
        compressor.update_config(new_config);
        
        assert_eq!(compressor.config.coefficients, 800);
        assert_eq!(compressor.config.block_size, 128);
        
        println!("✓ 설정 업데이트 완료");
        
        Ok(())
    }
} 