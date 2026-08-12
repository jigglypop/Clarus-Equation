use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::path::PathBuf;
use anyhow::Result;
use tokio::fs;

/// HuggingFace Hub에서 모델을 다운로드하는 구조체
#[derive(Debug, Clone)]
pub struct ModelDownloader {
    pub model_id: String,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub model_id: String,
    pub files_to_download: Vec<String>,
    pub output_dir: PathBuf,
    pub create_subdirs: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            model_id: "skt/kogpt2-base-v2".to_string(),
            files_to_download: vec![
                "model.safetensors".to_string(),
                "config.json".to_string(),
                "tokenizer.json".to_string(),
            ],
            output_dir: PathBuf::from("models"),
            create_subdirs: true,
        }
    }
}

impl ModelDownloader {
    /// 새로운 ModelDownloader 생성
    pub fn new(model_id: &str) -> Self {
        let safe_model_name = model_id.replace("/", "-");
        let output_dir = PathBuf::from("models").join(&safe_model_name);
        
        Self {
            model_id: model_id.to_string(),
            output_dir,
        }
    }
    
    /// 설정으로부터 ModelDownloader 생성
    pub fn from_config(config: &DownloadConfig) -> Self {
        let safe_model_name = config.model_id.replace("/", "-");
        let output_dir = if config.create_subdirs {
            config.output_dir.join(&safe_model_name)
        } else {
            config.output_dir.clone()
        };
        
        Self {
            model_id: config.model_id.clone(),
            output_dir,
        }
    }
    
    /// 단일 파일 다운로드
    pub async fn download_file(&self, filename: &str) -> Result<PathBuf> {
        println!("Downloading '{}' from '{}'...", filename, self.model_id);
        
        let api = Api::new()?;
        let repo = api.repo(Repo::new(self.model_id.clone(), RepoType::Model));
        
        let download_path = repo.get(filename).await?;
        
        // 출력 디렉토리 생성
        fs::create_dir_all(&self.output_dir).await?;
        let final_path = self.output_dir.join(filename);
        
        // 파일 복사
        fs::copy(&download_path, &final_path).await?;
        
        println!("✅ Successfully downloaded: '{}'", final_path.display());
        
        Ok(final_path)
    }
    
    /// 기본 모델 파일들 다운로드
    pub async fn download(&self) -> Result<PathBuf> {
        let files = vec!["model.safetensors", "config.json"];
        
        for file in &files {
            match self.download_file(file).await {
                Ok(_) => {},
                Err(e) => {
                    println!("⚠️  Failed to download '{}': {}", file, e);
                    // 일부 파일 실패는 무시하고 계속 진행
                }
            }
        }
        
        println!("📁 Model directory: {}", self.output_dir.display());
        Ok(self.output_dir.clone())
    }
    
    /// 설정 기반 다운로드
    pub async fn download_with_config(config: &DownloadConfig) -> Result<PathBuf> {
        let downloader = Self::from_config(config);
        
        println!("=== Model Download Started ===");
        println!("Model ID: {}", config.model_id);
        println!("Output Dir: {}", downloader.output_dir.display());
        println!("Files: {:?}", config.files_to_download);
        
        for file in &config.files_to_download {
            match downloader.download_file(file).await {
                Ok(_) => {},
                Err(e) => {
                    println!("⚠️  Failed to download '{}': {}", file, e);
                }
            }
        }
        
        println!("=== Download Complete ===");
        Ok(downloader.output_dir)
    }
    
    /// 토크나이저만 다운로드
    pub async fn download_tokenizer(&self) -> Result<PathBuf> {
        let tokenizer_files = vec![
            "tokenizer.json",
            "tokenizer_config.json", 
            "vocab.json",
            "merges.txt",
        ];
        
        println!("=== Tokenizer Download ===");
        let mut downloaded_count = 0;
        
        for file in &tokenizer_files {
            match self.download_file(file).await {
                Ok(_) => downloaded_count += 1,
                Err(e) => {
                    println!("⚠️  토크나이저 파일 '{}' 다운로드 실패: {}", file, e);
                }
            }
        }
        
        if downloaded_count == 0 {
            return Err(anyhow::anyhow!("토크나이저 파일을 하나도 다운로드하지 못했습니다"));
        }
        
        println!("✅ 토크나이저 다운로드 완료: {}/{} 파일", downloaded_count, tokenizer_files.len());
        Ok(self.output_dir.clone())
    }
    
    /// 한국어 모델들 다운로드 (프리셋)
    pub async fn download_korean_model(model_type: KoreanModel) -> Result<PathBuf> {
        let config = match model_type {
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
            KoreanModel::KoElectraBaseV3 => DownloadConfig {
                model_id: "monologg/koelectra-base-v3-generator".to_string(),
                files_to_download: vec![
                    "model.safetensors".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                ],
                output_dir: PathBuf::from("models"),
                create_subdirs: true,
            },
            KoreanModel::KlueRobertaSmall => DownloadConfig {
                model_id: "klue/roberta-small".to_string(),
                files_to_download: vec![
                    "pytorch_model.bin".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                ],
                output_dir: PathBuf::from("models"),
                create_subdirs: true,
            },
        };
        
        Self::download_with_config(&config).await
    }
    
    /// 다운로드 상태 확인
    pub fn check_download_status(&self) -> DownloadStatus {
        if !self.output_dir.exists() {
            return DownloadStatus::NotDownloaded;
        }
        
        let essential_files = vec!["model.safetensors", "config.json"];
        let mut found_files = 0;
        
        for file in &essential_files {
            if self.output_dir.join(file).exists() {
                found_files += 1;
            }
        }
        
        match found_files {
            0 => DownloadStatus::NotDownloaded,
            n if n == essential_files.len() => DownloadStatus::Complete,
            _ => DownloadStatus::Partial,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum KoreanModel {
    KoGpt2Base,
    KoGpt2Medium,
    KoMiniLM23M,         // 23M - 가장 작은 모델
    KoElectraBaseV3,     // 37.2M - 작은 ELECTRA 모델
    KlueRobertaSmall,    // 100M - KLUE 소형 모델
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DownloadStatus {
    NotDownloaded,
    Partial,
    Complete,
} 