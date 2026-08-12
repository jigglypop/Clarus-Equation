#[derive(Debug, Clone, Copy)]
pub enum QualityLevel {
    Ultra,   // PSNR > 50 dB, 32×32 블록
    High,    // PSNR > 40 dB, 64×64 블록
    Medium,  // PSNR > 30 dB, 128×128 블록
    Low,     // PSNR > 20 dB, 256×256 블록
}

impl QualityLevel {
    /// 품질 등급에 따른 최적 블록 크기 반환
    pub fn optimal_block_size(&self) -> usize {
        match self {
            QualityLevel::Ultra => 32,
            QualityLevel::High => 64,
            QualityLevel::Medium => 128,
            QualityLevel::Low => 256,
        }
    }
    
    /// 목표 PSNR 값
    pub fn target_psnr(&self) -> f32 {
        match self {
            QualityLevel::Ultra => 50.0,
            QualityLevel::High => 40.0,
            QualityLevel::Medium => 30.0,
            QualityLevel::Low => 20.0,
        }
    }
    
    /// 압축률
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QualityLevel::Ultra => 200.0,
            QualityLevel::High => 500.0,
            QualityLevel::Medium => 1000.0,
            QualityLevel::Low => 2000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityStats {
    pub total_error: f32,
    pub psnr: f32,
    pub compression_ratio: f32,
    pub memory_usage_bytes: usize,
    pub total_blocks: usize,
}

impl QualityStats {
    /// 품질 보고서 출력
    pub fn print_report(&self) {
        println!("=== 품질 통계 보고서 ===");
        println!("총 오차: {:.6}", self.total_error);
        println!("PSNR: {:.2} dB", self.psnr);
        println!("압축률: {:.1}:1", self.compression_ratio);
        println!("메모리 사용량: {:.2} KB", self.memory_usage_bytes as f32 / 1024.0);
        println!("총 블록 수: {}", self.total_blocks);
        
        // 압축 효율성 등급
        let efficiency_grade = if self.compression_ratio > 1000.0 {
            "A+"
        } else if self.compression_ratio > 500.0 {
            "A"
        } else if self.compression_ratio > 200.0 {
            "B"
        } else {
            "C"
        };
        
        println!("압축 효율성 등급: {}", efficiency_grade);
    }
} 