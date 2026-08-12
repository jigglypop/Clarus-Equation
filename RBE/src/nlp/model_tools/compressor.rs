use crate::core::encoder::encoder::RBEEncoder;
use crate::core::packed_params::{HybridEncodedBlock, TransformType};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json;

/// 모델 압축을 담당하는 구조체
pub struct ModelCompressor {
    pub encoder: RBEEncoder,
    pub config: CompressionConfig,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub block_size: usize,
    pub coefficients: usize,
    pub transform_type: TransformType,
    pub output_dir: PathBuf,
    pub matrix_size: usize,
    pub model_name: String,
}

#[derive(Debug)]
pub struct CompressionResult {
    pub encoded_blocks: Vec<HybridEncodedBlock>,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f32,
    pub compression_time: f64,
    pub total_blocks: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            coefficients: 500,
            transform_type: TransformType::Dwt,
            output_dir: PathBuf::from("./compressed_models"),
            matrix_size: 768, // GPT-2 hidden size
            model_name: "kogpt2".to_string(),
        }
    }
}

impl CompressionConfig {
    /// 고품질 압축 설정
    pub fn high_quality() -> Self {
        Self {
            block_size: 128,
            coefficients: 800,
            transform_type: TransformType::Dwt,
            ..Default::default()
        }
    }
    
    /// 빠른 압축 설정
    pub fn fast() -> Self {
        Self {
            block_size: 512,
            coefficients: 200,
            transform_type: TransformType::Dwt,
            ..Default::default()
        }
    }
    
    /// 극한 압축 설정
    pub fn extreme() -> Self {
        Self {
            block_size: 256,
            coefficients: 100,
            transform_type: TransformType::Dwt,
            ..Default::default()
        }
    }
}

impl ModelCompressor {
    /// 새로운 ModelCompressor 생성
    pub fn new(config: CompressionConfig) -> Self {
        let encoder = RBEEncoder::new(config.coefficients, config.transform_type);
        
        Self {
            encoder,
            config,
        }
    }
    
    /// 기본 설정으로 ModelCompressor 생성
    pub fn default() -> Self {
        Self::new(CompressionConfig::default())
    }
    
    /// 테스트용 행렬 데이터 생성
    pub fn generate_test_matrix(&self) -> Vec<f32> {
        let size = self.config.matrix_size;
        let mut matrix_data = vec![0.0f32; size * size];
        
        // 압축 가능한 패턴 생성
        for i in 0..size {
            for j in 0..size {
                let x = i as f32 / size as f32;
                let y = j as f32 / size as f32;
                matrix_data[i * size + j] = 
                    (2.0 * std::f32::consts::PI * x).sin() * 
                    (2.0 * std::f32::consts::PI * y).cos() * 0.5;
            }
        }
        
        matrix_data
    }
    
    /// 행렬 데이터 압축
    pub fn compress_matrix(&mut self, matrix_data: &[f32]) -> Result<CompressionResult> {
        let matrix_size = self.config.matrix_size;
        let block_size = self.config.block_size;
        
        println!("\n=== RBE 모델 압축 시작 ===");
        println!("압축 설정:");
        println!("   - 블록 크기: {}×{}", block_size, block_size);
        println!("   - 계수 개수: {} ({:?})", self.config.coefficients, self.config.transform_type);
        println!("   - 행렬 크기: {}×{}", matrix_size, matrix_size);
        
        let start = Instant::now();
        
        // 블록 단위로 압축
        let blocks_per_dim = (matrix_size + block_size - 1) / block_size;
        let total_blocks = blocks_per_dim * blocks_per_dim;
        
        let pb = ProgressBar::new(total_blocks as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40}] {percent}% 블록 {pos}/{len} 압축 중")
                .unwrap()
        );
        
        let mut encoded_blocks = Vec::new();
        
        for block_i in 0..blocks_per_dim {
            for block_j in 0..blocks_per_dim {
                let start_i = block_i * block_size;
                let start_j = block_j * block_size;
                
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; block_size * block_size];
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = start_i + i;
                        let global_j = start_j + j;
                        if global_i < matrix_size && global_j < matrix_size {
                            block_data[i * block_size + j] = 
                                matrix_data[global_i * matrix_size + global_j];
                        }
                    }
                }
                
                // 블록 압축
                let encoded_block = self.encoder.encode_block(&block_data, block_size, block_size);
                encoded_blocks.push(encoded_block);
                
                pb.inc(1);
            }
        }
        pb.finish();
        
        let compression_time = start.elapsed().as_secs_f64();
        
        // 압축 결과 계산
        let original_size = matrix_size * matrix_size * 4; // f32 bytes
        let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("\n✅ 압축 완료!");
        println!("   - 원본 크기: {} bytes ({:.2} MB)", 
            original_size, original_size as f32 / 1_048_576.0);
        println!("   - 압축 크기: {} bytes ({:.2} MB)", 
            compressed_size, compressed_size as f32 / 1_048_576.0);
        println!("   - 압축률: {:.1}:1", compression_ratio);
        println!("   - 압축 시간: {:.2}초", compression_time);
        println!("   - 블록 수: {}", encoded_blocks.len());
        
        Ok(CompressionResult {
            encoded_blocks,
            original_size,
            compressed_size,
            compression_ratio,
            compression_time,
            total_blocks: total_blocks,
        })
    }
    
    /// 압축 결과를 파일로 저장
    pub fn save_compressed_model(&self, result: &CompressionResult, model_id: &str) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join(format!(
            "{}_{}x{}_w{}.rbe", 
            self.config.model_name,
            self.config.block_size,
            self.config.block_size,
            self.config.coefficients
        ));
        
        println!("   - 저장 경로: {}", output_path.display());

        // 압축된 데이터를 포함한 JSON 생성
        let compressed_data = serde_json::json!({
            "metadata": {
                "model_name": model_id,
                "matrix_size": self.config.matrix_size,
                "block_size": self.config.block_size,
                "coefficients": self.config.coefficients,
                "transform_type": format!("{:?}", self.config.transform_type),
                "compression_ratio": result.compression_ratio,
                "original_size_bytes": result.original_size,
                "compressed_size_bytes": result.compressed_size,
                "total_blocks": result.total_blocks,
                "compression_time_seconds": result.compression_time,
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
            },
            "blocks": result.encoded_blocks
        });
        
        // 디렉토리 생성
        fs::create_dir_all(&self.config.output_dir)?;
        
        // 파일로 저장
        let json_string = serde_json::to_string_pretty(&compressed_data)?;
        fs::write(&output_path, json_string)?;
        
        println!("✅ 압축 모델 저장 완료: {}", output_path.display());
        
        Ok(output_path)
    }
    
    /// 전체 압축 프로세스 (생성 + 압축 + 저장)
    pub fn compress_and_save(&mut self, model_id: &str) -> Result<PathBuf> {
        // 1. 테스트 데이터 생성
        let matrix_data = self.generate_test_matrix();
        
        // 2. 압축 수행
        let result = self.compress_matrix(&matrix_data)?;
        
        // 3. 저장
        let output_path = self.save_compressed_model(&result, model_id)?;
        
        Ok(output_path)
    }
    
    /// 실제 모델 파일에서 가중치 추출 및 압축
    pub async fn compress_model_file(&mut self, model_path: &PathBuf) -> Result<PathBuf> {
        println!("=== 실제 모델 파일 압축 ===");
        println!("모델 경로: {:?}", model_path);
        
        // TODO: 실제 모델 파일에서 가중치 추출하는 로직 구현
        // 현재는 테스트 데이터로 대체
        println!("⚠️  실제 모델 로딩은 아직 구현되지 않음. 테스트 데이터 사용.");
        
        let model_id = model_path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
            
        self.compress_and_save(&model_id)
    }
    
    /// 압축 설정 업데이트
    pub fn update_config(&mut self, new_config: CompressionConfig) {
        self.config = new_config;
        self.encoder = RBEEncoder::new(self.config.coefficients, self.config.transform_type);
    }
    
    /// 압축 품질 예측
    pub fn estimate_compression_quality(&self) -> CompressionQualityEstimate {
        let total_elements = self.config.matrix_size * self.config.matrix_size;
        let elements_per_block = self.config.block_size * self.config.block_size;
        let blocks_count = (total_elements + elements_per_block - 1) / elements_per_block;
        
        // 압축률 예측
        let original_size = total_elements * 4; // f32
        let rbe_params_size = blocks_count * 8 * 4; // 8 params * f32
        let residual_size = blocks_count * self.config.coefficients * 8; // ResidualCoefficient 구조체 크기 추정
        let estimated_compressed_size = rbe_params_size + residual_size;
        let estimated_ratio = original_size as f32 / estimated_compressed_size as f32;
        
        // 품질 등급 예측
        let coeffs_ratio = self.config.coefficients as f32 / elements_per_block as f32;
        let quality = if coeffs_ratio > 0.5 {
            QualityLevel::High
        } else if coeffs_ratio > 0.2 {
            QualityLevel::Medium
        } else {
            QualityLevel::Low
        };
        
        CompressionQualityEstimate {
            estimated_ratio,
            quality_level: quality,
            estimated_rmse: match quality {
                QualityLevel::High => 0.001,
                QualityLevel::Medium => 0.01,
                QualityLevel::Low => 0.1,
            },
            blocks_count,
        }
    }
}

#[derive(Debug)]
pub struct CompressionQualityEstimate {
    pub estimated_ratio: f32,
    pub quality_level: QualityLevel,
    pub estimated_rmse: f32,
    pub blocks_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum QualityLevel {
    High,   // > 50% 계수 유지
    Medium, // 20-50% 계수 유지
    Low,    // < 20% 계수 유지
} 