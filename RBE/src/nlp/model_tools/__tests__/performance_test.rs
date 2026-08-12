use crate::nlp::model_tools::*;
use std::time::Instant;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn 다운로더_성능_벤치마크() {
    println!("\n🚀 다운로더 성능 벤치마크 시작");
    
    let start = Instant::now();
    
    // 작은 모델로 테스트
    let downloader = ModelDownloader::new("BM-K/KoMiniLM");
    
    println!("📊 다운로더 설정:");
    println!("  - 모델 ID: {}", downloader.model_id);
    println!("  - 출력 경로: {:?}", downloader.output_dir);
    
    let setup_time = start.elapsed();
    println!("⏱️  다운로더 설정 시간: {:?}", setup_time);
    
    // 설정 체크 (실제 다운로드는 너무 오래 걸림)
    let check_start = Instant::now();
    let status = downloader.check_download_status();
    let check_time = check_start.elapsed();
    
    println!("📋 다운로드 상태: {:?}", status);
    println!("⏱️  상태 확인 시간: {:?}", check_time);
    println!("✅ 다운로더 성능 벤치마크 완료\n");
}

#[test]
fn 압축기_성능_벤치마크() {
    println!("\n🚀 압축기 성능 벤치마크 시작");
    
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("compressed_model");
    
    let start = Instant::now();
    
    // 테스트용 작은 데이터 생성
    let test_data = generate_test_matrix(128, 256); // 128x256 행렬
    println!("📊 테스트 데이터:");
    println!("  - 크기: 128x256 = {} 요소", test_data.len());
    println!("  - 메모리: {} KB", test_data.len() * 4 / 1024);
    
    let data_gen_time = start.elapsed();
    println!("⏱️  데이터 생성 시간: {:?}", data_gen_time);
    
    // 압축기 생성
    let compress_start = Instant::now();
    let config = crate::nlp::model_tools::compressor::CompressionConfig::default();
    let mut compressor = ModelCompressor::new(config);
    
    println!("📋 압축 설정:");
    println!("  - 프로파일: Default");
    println!("  - 블록 크기: {}", compressor.config.block_size);
    
    let setup_time = compress_start.elapsed();
    println!("⏱️  압축기 설정 시간: {:?}", setup_time);
    
    // 실제 압축 수행
    let actual_compress_start = Instant::now();
    let result = compressor.compress_matrix(&test_data);
    
    let compress_time = actual_compress_start.elapsed();
    
    match result {
        Ok(result) => {
            println!("✅ 압축 성공!");
            println!("📊 압축 결과:");
            println!("  - 압축률: {:.2}x", result.compression_ratio);
            println!("  - 총 블록: {}", result.total_blocks);
            println!("  - 압축 시간: {:.3}s", result.compression_time);
            println!("⏱️  압축 시간: {:?}", compress_time);
            println!("🔥 처리량: {:.1} MB/s", 
                     (test_data.len() * 4) as f64 / 1024.0 / 1024.0 / compress_time.as_secs_f64());
        }
        Err(e) => {
            println!("❌ 압축 실패: {}", e);
        }
    }
    
    println!("✅ 압축기 성능 벤치마크 완료\n");
}

#[tokio::test]
async fn 분석기_성능_벤치마크() {
    println!("\n🚀 분석기 성능 벤치마크 시작");
    
    let start = Instant::now();
    
    // 분석기 생성
    let mut analyzer = ModelAnalyzer::new();
    println!("📊 분석기 초기 상태:");
    println!("  - 캐시된 분석: {} 개", analyzer.analysis_cache.len());
    
    let setup_time = start.elapsed();
    println!("⏱️  분석기 설정 시간: {:?}", setup_time);
    
    // 테스트용 임시 모델 디렉토리 생성
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();
    
    // 가짜 config.json 생성
    let config_content = r#"{
        "model_type": "bert",
        "architectures": ["BertModel"],
        "vocab_size": 30000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512
    }"#;
    
    std::fs::write(model_path.join("config.json"), config_content).unwrap();
    
    // 모델 정보 추출 성능 테스트
    let extract_start = Instant::now();
    let model_info_result = analyzer.extract_model_info(&model_path.to_path_buf()).await;
    let extract_time = extract_start.elapsed();
    
    println!("⏱️  모델 정보 추출 시간: {:?}", extract_time);
    
    if let Ok(model_info) = model_info_result {
        println!("📊 추출된 모델 정보:");
        println!("  - 모델 타입: {}", model_info.model_type);
        println!("  - 아키텍처: {}", model_info.architecture);
        println!("  - 총 파라미터: {}", model_info.total_parameters);
        println!("  - 모델 크기: {:.1} MB", model_info.model_size_mb);
        
        if let Some(hidden_size) = model_info.hidden_size {
            println!("  - Hidden Size: {}", hidden_size);
        }
        if let Some(num_layers) = model_info.num_layers {
            println!("  - 레이어 수: {}", num_layers);
        }
        
        // 성능 추정 테스트
        let perf_start = Instant::now();
        let perf_result = analyzer.estimate_performance(&model_info);
        let perf_time = perf_start.elapsed();
        
        println!("⏱️  성능 추정 시간: {:?}", perf_time);
        
        if let Ok(performance) = perf_result {
            println!("📊 성능 추정 결과:");
            println!("  - 추론 시간: {:.1} ms", performance.inference_speed_ms);
            println!("  - 메모리 사용량: {:.1} MB", performance.memory_usage_mb);
            println!("  - GPU 메모리: {:.1} MB", performance.gpu_memory_mb);
        }
    }
    
    let total_time = start.elapsed();
    println!("⏱️  전체 분석 시간: {:?}", total_time);
    println!("✅ 분석기 성능 벤치마크 완료\n");
}

#[test]
fn 통합_성능_벤치마크() {
    println!("\n🚀 model_tools 통합 성능 벤치마크 시작");
    
    let overall_start = Instant::now();
    
    // 1. 다운로더
    println!("1️⃣ 다운로더 테스트...");
    let dl_start = Instant::now();
    let downloader = ModelDownloader::new("BM-K/KoMiniLM");
    let dl_time = dl_start.elapsed();
    println!("   ✅ 다운로더: {:?}", dl_time);
    
    // 2. 분석기
    println!("2️⃣ 분석기 테스트...");
    let analyze_start = Instant::now();
    let analyzer = ModelAnalyzer::new();
    let analyze_time = analyze_start.elapsed();
    println!("   ✅ 분석기: {:?}", analyze_time);
    
    // 3. 압축기
    println!("3️⃣ 압축기 테스트...");
    let compress_start = Instant::now();
    let temp_dir = TempDir::new().unwrap();
    let compressor = ModelCompressor::default();
    let compress_time = compress_start.elapsed();
    println!("   ✅ 압축기: {:?}", compress_time);
    
    let total_time = overall_start.elapsed();
    
    println!("📊 통합 성능 요약:");
    println!("  - 다운로더 생성: {:?}", dl_time);
    println!("  - 분석기 생성: {:?}", analyze_time);
    println!("  - 압축기 생성: {:?}", compress_time);
    println!("  - 전체 시간: {:?}", total_time);
    
    // 성능 기준 확인
    assert!(dl_time.as_millis() < 10, "다운로더가 너무 느림");
    assert!(analyze_time.as_millis() < 5, "분석기가 너무 느림");
    assert!(compress_time.as_millis() < 10, "압축기가 너무 느림");
    
    println!("✅ 모든 성능 기준 통과!");
    println!("✅ 통합 성능 벤치마크 완료\n");
}

/// 테스트용 행렬 데이터 생성
fn generate_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let x = (i % cols) as f32 / cols as f32;
            let y = (i / cols) as f32 / rows as f32;
            (2.0 * std::f32::consts::PI * x).sin() * (2.0 * std::f32::consts::PI * y).cos()
        })
        .collect()
}

/// 메모리 사용량 측정 (근사치)
fn get_memory_usage() -> usize {
    // 간단한 메모리 사용량 추정
    std::mem::size_of::<ModelDownloader>() + 
    std::mem::size_of::<ModelAnalyzer>() + 
    std::mem::size_of::<ModelCompressor>()
} 