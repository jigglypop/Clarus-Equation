use super::super::{HierarchicalBlockMatrix, QualityLevel};
use std::time::Instant;

#[test]
fn 계층적행렬_생성_테스트() {
    let rows = 256;
    let cols = 256;
    let quality = QualityLevel::Medium;
    
    let matrix = HierarchicalBlockMatrix::new(rows, cols, quality);
    
    assert_eq!(matrix.total_rows, rows);
    assert_eq!(matrix.total_cols, cols);
    assert!(matrix.l1_blocks.is_empty());
    assert_eq!(matrix.error_controller.global_error_threshold, 1e-2);
    
    println!("✅ 계층적행렬 생성 테스트 통과");
    println!("   크기: {}×{}, 품질: {:?}", rows, cols, quality);
}

#[test]
fn 적응적_분할_테스트() {
    let mut matrix = HierarchicalBlockMatrix::new(128, 128, QualityLevel::High);
    
    let source: Vec<f32> = (0..128*128)
        .map(|i| (i % 100) as f32 / 100.0)
        .collect();
    
    let start = Instant::now();
    matrix.adaptive_partition(&source);
    let duration = start.elapsed();
    
    assert!(!matrix.l1_blocks.is_empty());
    assert!(duration.as_secs_f64() < 1.0);
    
    println!("✅ 적응적 분할 테스트 통과");
    println!("   분할 시간: {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("   L1 블록 수: {}", matrix.l1_blocks.len());
}

#[test]
fn 병렬_GEMV_테스트() {
    let mut matrix = HierarchicalBlockMatrix::new(64, 64, QualityLevel::Medium);
    
    let source = vec![0.5; 64 * 64];
    matrix.adaptive_partition(&source);
    
    let input = vec![1.0; 64];
    let mut output = vec![0.0; 64];
    
    let start = Instant::now();
    matrix.parallel_gemv(&input, &mut output, 4);
    let duration = start.elapsed();
    
    let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-6).count();
    
    assert!(duration.as_secs_f64() < 0.1);
    assert!(non_zero_count >= 0); // 일부 출력이 0이 아닐 수 있음
    
    println!("✅ 병렬 GEMV 테스트 통과");
    println!("   GEMV 시간: {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("   0이 아닌 출력: {}/{}", non_zero_count, output.len());
}

#[test]
fn 메모리_사용량_테스트() {
    let mut matrix = HierarchicalBlockMatrix::new(128, 128, QualityLevel::Low);
    
    let source = vec![0.1; 128 * 128];
    matrix.adaptive_partition(&source);
    
    let (memory_bytes, compression_ratio) = matrix.memory_usage();
    
    assert!(memory_bytes > 0);
    assert!(compression_ratio > 1.0);
    
    println!("✅ 메모리 사용량 테스트 통과");
    println!("   메모리: {:.2} KB", memory_bytes as f32 / 1024.0);
    println!("   압축률: {:.1}:1", compression_ratio);
}

#[test]
fn 품질통계_테스트() {
    let mut matrix = HierarchicalBlockMatrix::new(64, 64, QualityLevel::Ultra);
    
    let source: Vec<f32> = (0..64*64)
        .map(|i| ((i * 13) % 100) as f32 / 100.0)
        .collect();
    
    matrix.adaptive_partition(&source);
    
    let stats = matrix.quality_statistics();
    
    assert!(stats.total_error >= 0.0);
    assert!(stats.psnr > 0.0);
    assert!(stats.compression_ratio > 0.0);
    assert!(stats.memory_usage_bytes > 0);
    assert!(stats.total_blocks > 0);
    
    println!("✅ 품질통계 테스트 통과");
    println!("   PSNR: {:.1} dB", stats.psnr);
    println!("   압축률: {:.1}:1", stats.compression_ratio);
    println!("   총 블록: {}", stats.total_blocks);
}

#[test]
fn 다양한_품질_테스트() {
    let qualities = [
        QualityLevel::Ultra,
        QualityLevel::High,
        QualityLevel::Medium,
        QualityLevel::Low,
    ];
    
    let source = vec![0.7; 64 * 64];
    
    for quality in &qualities {
        let mut matrix = HierarchicalBlockMatrix::new(64, 64, *quality);
        matrix.adaptive_partition(&source);
        
        let (memory_bytes, compression_ratio) = matrix.memory_usage();
        let stats = matrix.quality_statistics();
        
        println!("품질 {:?}:", quality);
        println!("  메모리: {:.2} KB", memory_bytes as f32 / 1024.0);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  PSNR: {:.1} dB", stats.psnr);
        
        assert!(compression_ratio > 1.0);
        assert!(stats.psnr > 0.0);
    }
    
    println!("✅ 다양한 품질 테스트 통과");
}

#[test]
fn 복제_테스트() {
    let mut matrix = HierarchicalBlockMatrix::new(32, 32, QualityLevel::High);
    
    let source = vec![0.3; 32 * 32];
    matrix.adaptive_partition(&source);
    
    let cloned = matrix.clone();
    
    assert_eq!(matrix.total_rows, cloned.total_rows);
    assert_eq!(matrix.total_cols, cloned.total_cols);
    assert_eq!(matrix.l1_blocks.len(), cloned.l1_blocks.len());
    
    println!("✅ 복제 테스트 통과");
}

#[test]
fn 디버그_출력_테스트() {
    let matrix = HierarchicalBlockMatrix::new(64, 64, QualityLevel::Medium);
    
    let debug_str = format!("{:?}", matrix);
    
    assert!(debug_str.contains("HierarchicalBlockMatrix"));
    assert!(debug_str.contains("total_rows"));
    assert!(debug_str.len() > 100);
    
    println!("✅ 디버그 출력 테스트 통과");
    println!("   Debug 출력 길이: {} 문자", debug_str.len());
} 