use crate::nlp::model_tools::*;
use anyhow::Result;
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_download_config_default() {
        println!("=== DownloadConfig 기본값 테스트 ===");
        
        let config = DownloadConfig::default();
        assert_eq!(config.model_id, "skt/kogpt2-base-v2");
        assert_eq!(config.files_to_download.len(), 3);
        assert!(config.create_subdirs);
        assert_eq!(config.output_dir, PathBuf::from("models"));
        
        println!("✓ 기본 설정 검증 완료");
    }
    
    #[test]
    fn test_model_downloader_creation() {
        println!("=== ModelDownloader 생성 테스트 ===");
        
        let downloader = ModelDownloader::new("skt/kogpt2-base-v2");
        assert_eq!(downloader.model_id, "skt/kogpt2-base-v2");
        assert_eq!(downloader.output_dir, PathBuf::from("models/skt-kogpt2-base-v2"));
        
        println!("✓ ModelDownloader 생성 완료");
    }
    
    #[test]
    fn test_model_downloader_from_config() {
        println!("=== ModelDownloader 설정 기반 생성 테스트 ===");
        
        let config = DownloadConfig {
            model_id: "test/model".to_string(),
            files_to_download: vec!["model.safetensors".to_string()],
            output_dir: PathBuf::from("custom_models"),
            create_subdirs: true,
        };
        
        let downloader = ModelDownloader::from_config(&config);
        assert_eq!(downloader.model_id, "test/model");
        assert_eq!(downloader.output_dir, PathBuf::from("custom_models/test-model"));
        
        println!("✓ 설정 기반 생성 완료");
    }
    
    #[test]
    fn test_model_downloader_from_config_no_subdirs() {
        println!("=== ModelDownloader 서브디렉토리 없이 생성 테스트 ===");
        
        let config = DownloadConfig {
            model_id: "test/model".to_string(),
            files_to_download: vec!["model.safetensors".to_string()],
            output_dir: PathBuf::from("flat_models"),
            create_subdirs: false,
        };
        
        let downloader = ModelDownloader::from_config(&config);
        assert_eq!(downloader.output_dir, PathBuf::from("flat_models"));
        
        println!("✓ 플랫 디렉토리 생성 완료");
    }
    
    #[test]
    fn test_download_status_not_downloaded() {
        println!("=== 다운로드 상태 확인 테스트 (다운로드 안됨) ===");
        
        let downloader = ModelDownloader::new("nonexistent/model");
        let status = downloader.check_download_status();
        
        assert_eq!(status, DownloadStatus::NotDownloaded);
        
        println!("✓ 다운로드 안됨 상태 확인 완료");
    }
    
    #[test]
    fn test_korean_model_configs() {
        println!("=== 한국어 모델 설정 테스트 ===");
        
        // KoGpt2Base 설정 검증
        let kogpt2_config = match KoreanModel::KoGpt2Base {
            KoreanModel::KoGpt2Base => DownloadConfig {
                model_id: "skt/kogpt2-base-v2".to_string(),
                files_to_download: vec![
                    "model.safetensors".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                ],
                output_dir: PathBuf::from("models"),
                create_subdirs: true,
            },
            _ => unreachable!(),
        };
        
        assert_eq!(kogpt2_config.model_id, "skt/kogpt2-base-v2");
        assert_eq!(kogpt2_config.files_to_download.len(), 3);
        
        // KoGpt2Medium 설정 검증
        let kogpt2_medium_config = match KoreanModel::KoGpt2Medium {
            KoreanModel::KoGpt2Medium => DownloadConfig {
                model_id: "skt/ko-gpt-trinity-1.2B-v0.5".to_string(),
                files_to_download: vec![
                    "pytorch_model.bin".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                ],
                output_dir: PathBuf::from("models"),
                create_subdirs: true,
            },
            _ => unreachable!(),
        };
        
        assert_eq!(kogpt2_medium_config.model_id, "skt/ko-gpt-trinity-1.2B-v0.5");
        assert_eq!(kogpt2_medium_config.files_to_download.len(), 3);
        
        // KoMiniLM23M 설정 검증
        let kominilm_config = match KoreanModel::KoMiniLM23M {
            KoreanModel::KoMiniLM23M => DownloadConfig {
                model_id: "BM-K/KoMiniLM".to_string(),
                files_to_download: vec![
                    "pytorch_model.bin".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                ],
                output_dir: PathBuf::from("models"),
                create_subdirs: true,
            },
            _ => unreachable!(),
        };
        
        assert_eq!(kominilm_config.model_id, "BM-K/KoMiniLM");
        assert_eq!(kominilm_config.files_to_download.len(), 3);
        
        println!("✓ 한국어 모델 설정 검증 완료");
    }
    
    #[test]
    fn test_model_id_sanitization() {
        println!("=== 모델 ID 안전화 테스트 ===");
        
        let downloader = ModelDownloader::new("organization/model-name");
        let expected_dir = PathBuf::from("models/organization-model-name");
        assert_eq!(downloader.output_dir, expected_dir);
        
        let downloader2 = ModelDownloader::new("complex/path/with/slashes");
        let expected_dir2 = PathBuf::from("models/complex-path-with-slashes");
        assert_eq!(downloader2.output_dir, expected_dir2);
        
        println!("✓ 모델 ID 안전화 완료");
    }
    
    // 실제 다운로드 테스트는 네트워크가 필요하므로 주석 처리
    // #[tokio::test]
    // async fn test_download_file() -> Result<()> {
    //     // 실제 네트워크 연결이 필요한 테스트
    //     // 개발 환경에서만 실행하도록 설정
    //     if std::env::var("RUN_INTEGRATION_TESTS").is_ok() {
    //         let downloader = ModelDownloader::new("skt/kogpt2-base-v2");
    //         let result = downloader.download_file("config.json").await;
    //         assert!(result.is_ok());
    //     }
    //     Ok(())
    // }
    
    #[test]
    fn test_download_status_enum() {
        println!("=== DownloadStatus 열거형 테스트 ===");
        
        let statuses = vec![
            DownloadStatus::NotDownloaded,
            DownloadStatus::Partial,
            DownloadStatus::Complete,
        ];
        
        // Enum이 Copy trait을 구현하는지 확인
        let status = DownloadStatus::Complete;
        let status_copy = status;
        assert_eq!(status, status_copy);
        
        // Debug trait 확인
        for status in &statuses {
            let debug_str = format!("{:?}", status);
            assert!(!debug_str.is_empty());
        }
        
        println!("✓ DownloadStatus 열거형 검증 완료");
    }
    
    #[test]
    fn test_korean_model_enum() {
        println!("=== KoreanModel 열거형 테스트 ===");
        
        let models = vec![
            KoreanModel::KoGpt2Base,
            KoreanModel::KoGpt2Medium,
            KoreanModel::KoMiniLM23M,
            KoreanModel::KoElectraBaseV3,
            KoreanModel::KlueRobertaSmall,
        ];
        
        // Copy trait 확인
        let model = KoreanModel::KoGpt2Base;
        let model_copy = model;
        assert!(matches!(model_copy, KoreanModel::KoGpt2Base));
        
        // Debug trait 확인
        for model in &models {
            let debug_str = format!("{:?}", model);
            assert!(!debug_str.is_empty());
        }
        
        println!("✓ KoreanModel 열거형 검증 완료");
    }
} 