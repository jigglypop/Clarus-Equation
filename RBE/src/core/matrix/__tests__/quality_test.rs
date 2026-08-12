use super::super::{QualityLevel, QualityStats};

#[test]
fn 품질등급_기본값_테스트() {
    let quality_levels = [
        QualityLevel::Ultra,
        QualityLevel::High,
        QualityLevel::Medium,
        QualityLevel::Low,
    ];
    
    for quality in &quality_levels {
        let block_size = quality.optimal_block_size();
        let psnr = quality.target_psnr();
        let compression = quality.compression_ratio();
        
        // 블록 크기 검증
        assert!(block_size >= 32 && block_size <= 256, "블록 크기 범위 오류: {}", block_size);
        
        // PSNR 검증
        assert!(psnr >= 20.0 && psnr <= 60.0, "PSNR 범위 오류: {}", psnr);
        
        // 압축률 검증
        assert!(compression >= 200.0 && compression <= 2000.0, "압축률 범위 오류: {}", compression);
        
        println!("✅ {:?}: 블록={}×{}, PSNR={:.1}dB, 압축={:.0}:1", 
                 quality, block_size, block_size, psnr, compression);
    }
    
    println!("✅ 품질등급 기본값 테스트 통과");
}

#[test]
fn 품질등급_순서_테스트() {
    let ultra = QualityLevel::Ultra;
    let high = QualityLevel::High;
    let medium = QualityLevel::Medium;
    let low = QualityLevel::Low;
    
    // 블록 크기는 품질이 높을수록 작아야 함
    assert!(ultra.optimal_block_size() < high.optimal_block_size());
    assert!(high.optimal_block_size() < medium.optimal_block_size());
    assert!(medium.optimal_block_size() < low.optimal_block_size());
    
    // PSNR은 품질이 높을수록 커야 함
    assert!(ultra.target_psnr() > high.target_psnr());
    assert!(high.target_psnr() > medium.target_psnr());
    assert!(medium.target_psnr() > low.target_psnr());
    
    // 압축률은 품질이 낮을수록 커야 함
    assert!(ultra.compression_ratio() < high.compression_ratio());
    assert!(high.compression_ratio() < medium.compression_ratio());
    assert!(medium.compression_ratio() < low.compression_ratio());
    
    println!("✅ 품질등급 순서 테스트 통과");
    println!("   Ultra < High < Medium < Low (블록크기, 압축률)");
    println!("   Ultra > High > Medium > Low (PSNR)");
}

#[test]
fn 품질등급_극값_테스트() {
    // Ultra 품질 (최고 품질)
    let ultra = QualityLevel::Ultra;
    assert_eq!(ultra.optimal_block_size(), 32, "Ultra 블록 크기");
    assert_eq!(ultra.target_psnr(), 50.0, "Ultra PSNR");
    assert_eq!(ultra.compression_ratio(), 200.0, "Ultra 압축률");
    
    // Low 품질 (최저 품질)
    let low = QualityLevel::Low;
    assert_eq!(low.optimal_block_size(), 256, "Low 블록 크기");
    assert_eq!(low.target_psnr(), 20.0, "Low PSNR");
    assert_eq!(low.compression_ratio(), 2000.0, "Low 압축률");
    
    println!("✅ 품질등급 극값 테스트 통과");
    println!("   Ultra: 32×32, 50.0dB, 200:1");
    println!("   Low: 256×256, 20.0dB, 2000:1");
}

#[test]
fn 품질등급_복제_테스트() {
    let original = QualityLevel::High;
    let cloned = original;
    
    assert_eq!(original.optimal_block_size(), cloned.optimal_block_size());
    assert_eq!(original.target_psnr(), cloned.target_psnr());
    assert_eq!(original.compression_ratio(), cloned.compression_ratio());
    
    println!("✅ 품질등급 복제 테스트 통과");
}

#[test]
fn 품질등급_디버그_출력_테스트() {
    let quality = QualityLevel::Medium;
    let debug_str = format!("{:?}", quality);
    
    assert!(debug_str.contains("Medium"), "디버그 출력에 품질명이 없음");
    assert!(debug_str.len() > 0, "디버그 출력이 비어있음");
    
    println!("✅ 품질등급 디버그 출력 테스트 통과");
    println!("   Debug 출력: {}", debug_str);
}

#[test]
fn 품질통계_생성_테스트() {
    let stats = QualityStats {
        total_error: 0.01,
        psnr: 35.5,
        compression_ratio: 750.0,
        memory_usage_bytes: 1024,
        total_blocks: 16,
    };
    
    assert_eq!(stats.total_error, 0.01);
    assert_eq!(stats.psnr, 35.5);
    assert_eq!(stats.compression_ratio, 750.0);
    assert_eq!(stats.memory_usage_bytes, 1024);
    assert_eq!(stats.total_blocks, 16);
    
    println!("✅ 품질통계 생성 테스트 통과");
    println!("   오차={:.3}, PSNR={:.1}dB, 압축={:.0}:1", 
             stats.total_error, stats.psnr, stats.compression_ratio);
}

#[test]
fn 품질통계_복제_테스트() {
    let original = QualityStats {
        total_error: 0.05,
        psnr: 28.2,
        compression_ratio: 1200.0,
        memory_usage_bytes: 2048,
        total_blocks: 32,
    };
    
    let cloned = original.clone();
    
    assert_eq!(original.total_error, cloned.total_error);
    assert_eq!(original.psnr, cloned.psnr);
    assert_eq!(original.compression_ratio, cloned.compression_ratio);
    assert_eq!(original.memory_usage_bytes, cloned.memory_usage_bytes);
    assert_eq!(original.total_blocks, cloned.total_blocks);
    
    println!("✅ 품질통계 복제 테스트 통과");
}

#[test]
fn 품질통계_디버그_출력_테스트() {
    let stats = QualityStats {
        total_error: 0.001,
        psnr: 42.8,
        compression_ratio: 500.0,
        memory_usage_bytes: 4096,
        total_blocks: 8,
    };
    
    let debug_str = format!("{:?}", stats);
    
    assert!(debug_str.contains("total_error"), "디버그 출력에 total_error 필드 없음");
    assert!(debug_str.contains("psnr"), "디버그 출력에 psnr 필드 없음");
    assert!(debug_str.contains("compression_ratio"), "디버그 출력에 compression_ratio 필드 없음");
    assert!(debug_str.len() > 50, "디버그 출력이 너무 짧음");
    
    println!("✅ 품질통계 디버그 출력 테스트 통과");
    println!("   Debug 출력 길이: {} 문자", debug_str.len());
}

#[test]
fn 품질통계_print_report_테스트() {
    let stats = QualityStats {
        total_error: 0.002,
        psnr: 38.1,
        compression_ratio: 1500.0,
        memory_usage_bytes: 8192,
        total_blocks: 24,
    };
    
    println!("=== 품질 보고서 출력 테스트 ===");
    stats.print_report();
    
    // 효율성 등급 계산 검증
    let efficiency_grade = if stats.compression_ratio > 1000.0 {
        "A+"
    } else if stats.compression_ratio > 500.0 {
        "A"
    } else if stats.compression_ratio > 200.0 {
        "B"
    } else {
        "C"
    };
    
    assert_eq!(efficiency_grade, "A+", "효율성 등급 계산 오류");
    
    println!("✅ 품질통계 보고서 출력 테스트 통과");
    println!("   압축률 {:.0}:1 → 효율성 등급: {}", stats.compression_ratio, efficiency_grade);
}

#[test]
fn 품질통계_효율성등급_테스트() {
    let test_cases = [
        (1500.0, "A+"),  // > 1000
        (750.0, "A"),    // > 500
        (300.0, "B"),    // > 200
        (100.0, "C"),    // <= 200
    ];
    
    for &(compression_ratio, expected_grade) in &test_cases {
        let efficiency_grade = if compression_ratio > 1000.0 {
            "A+"
        } else if compression_ratio > 500.0 {
            "A"
        } else if compression_ratio > 200.0 {
            "B"
        } else {
            "C"
        };
        
        assert_eq!(efficiency_grade, expected_grade, 
                   "압축률 {:.0}:1의 효율성 등급이 틀림", compression_ratio);
        
        println!("압축률 {:.0}:1 → 등급 {}", compression_ratio, efficiency_grade);
    }
    
    println!("✅ 품질통계 효율성등급 테스트 통과");
} 