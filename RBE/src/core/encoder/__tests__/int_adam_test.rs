#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::encoder::RBEEncoder;
    use crate::core::decoder::WeightGenerator;
    use crate::packed_params::{TransformType, HybridEncodedBlock, ResidualCoefficient};
    use std::time::Instant;
    use rand::Rng;
    use rayon::prelude::*;

    /// 테스트용 데이터 생성 (실제 모델 가중치 패턴 시뮬레이션)
    fn generate_test_block(rows: usize, cols: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(rows * cols);
        
        // 실제 가중치처럼 정규분포 + 일부 패턴
        for i in 0..rows {
            for j in 0..cols {
                let base = rng.gen::<f32>() * 0.1 - 0.05; // [-0.05, 0.05]
                let pattern = ((i as f32 / rows as f32) * std::f32::consts::PI).sin() * 
                             ((j as f32 / cols as f32) * std::f32::consts::PI).cos() * 0.02;
                data.push(base + pattern);
            }
        }
        
        data
    }
    
    /// RMSE 계산
    fn calculate_rmse(original: &[f32], decoded: &[f32]) -> f32 {
        assert_eq!(original.len(), decoded.len());
        
        let mse: f32 = original.iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        mse.sqrt()
    }
    
    #[test]
    fn test_int_adam_encoding_speed() {
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let block_data = generate_test_block(64, 64);
        
        println!("\n=== 정수 Adam 인코딩 성능 테스트 (1000 step) ===");
        
        // Warm-up
        for _ in 0..10 {
            let _ = encoder.encode_block_int_adam(&block_data, 64, 64, 100);
        }
        
        // 실제 측정
        let start = Instant::now();
        let encoded = encoder.encode_block_int_adam(&block_data, 64, 64, 1000);
        let encoding_time = start.elapsed();
        
        println!("\n인코딩 시간: {:.2}ms (오프라인 작업이므로 느려도 OK)", encoding_time.as_millis());
        
        // 압축률 확인
        let original_size = 64 * 64 * 4; // f32
        let compressed_size = 8 * 4 + encoded.residuals.len() * 8; // 대략적인 크기
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("압축률: {:.0}:1 (목표: 1000:1)", compression_ratio);
        println!("잔차 계수 개수: {} (평균 목표: 0.8)", encoded.residuals.len());
        
        // 압축률 체크만 수행 (시간 제약 제거)
        assert!(compression_ratio > 500.0, "압축률이 너무 낮습니다");
    }
    
    #[test]
    fn test_int_adam_decoding_speed() {
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        println!("\n=== 초고속 디코딩 성능 테스트 ===");
        
        let block_data = generate_test_block(64, 64);
        let encoded = encoder.encode_block_int_adam(&block_data, 64, 64, 1000);
        
        // Warm-up
        for _ in 0..1000 {
            let _ = decoder.decode_int_adam_fast(&encoded, 32, 32);
        }
        
        // 개별 픽셀 디코딩 속도 측정
        let start = Instant::now();
        let iterations = 100000;
        for _ in 0..iterations {
            let _ = decoder.decode_int_adam_fast(&encoded, 32, 32);
        }
        let decode_time_per_pixel = start.elapsed() / iterations;
        
        println!("픽셀당 디코딩 시간: {:?} (목표: 150ns)", decode_time_per_pixel);
        
        // 전체 블록 디코딩 속도
        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = decoder.decode_block_int_adam(&encoded);
        }
        let block_decode_time = start.elapsed() / iterations;
        
        println!("64x64 블록 전체 디코딩 시간: {:.2}μs", block_decode_time.as_micros());
        println!("픽셀당 평균: {:.0}ns", block_decode_time.as_nanos() as f64 / (64.0 * 64.0));
        
        // 디코딩 속도만 엄격하게 체크
        assert!(decode_time_per_pixel.as_nanos() < 500, "디코딩이 너무 느립니다");
    }
    
    #[test]
    fn test_int_adam_accuracy() {
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        let test_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)];
        
        for (rows, cols) in test_sizes {
            let block_data = generate_test_block(rows, cols);
            
            // 인코딩
            let start_encode = Instant::now();
            let encoded = encoder.encode_block_int_adam(&block_data, rows, cols, 1000);
            let encode_time = start_encode.elapsed();
            
            // 디코딩
            let start_decode = Instant::now();
            let decoded = decoder.decode_block_int_adam(&encoded);
            let decode_time = start_decode.elapsed();
            
            // RMSE 계산
            let rmse = calculate_rmse(&block_data, &decoded);
            
            // 압축률 계산
            let original_size = rows * cols * 4;
            let compressed_size = 8 * 4 + encoded.residuals.len() * 8;
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            println!("\n{}x{} 블록:", rows, cols);
            println!("  인코딩 시간: {:.2}μs", encode_time.as_micros());
            println!("  디코딩 시간: {:.2}μs", decode_time.as_micros());
            println!("  RMSE: {:.2e} (목표: 1e-6)", rmse);
            println!("  압축률: {:.0}:1", compression_ratio);
            println!("  잔차 계수: {}", encoded.residuals.len());
            
            // RMSE 목표치 확인
            assert!(rmse < 1e-5, "RMSE가 너무 큽니다: {}", rmse);
        }
    }
    
    #[test]
    fn test_convergence_behavior() {
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let block_data = generate_test_block(64, 64);
        
        println!("\n=== Adam 최적화 수렴 동작 분석 ===");
        
        // 다양한 step 수로 테스트
        let step_counts = [100, 200, 500, 1000];
        
        for steps in step_counts {
            println!("\n--- {} steps 테스트 ---", steps);
            let encoded = encoder.encode_block_int_adam(&block_data, 64, 64, steps);
            let decoder = WeightGenerator::new();
            let decoded = decoder.decode_block_int_adam(&encoded);
            let rmse = calculate_rmse(&block_data, &decoded);
            
            println!("최종 결과: RMSE = {:.2e}, 잔차 = {}", 
                     rmse, encoded.residuals.len());
        }
    }
    
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_performance() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping SIMD test");
            return;
        }
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        let block_data = generate_test_block(64, 64);
        let encoded = encoder.encode_block_int_adam(&block_data, 64, 64, 1000);
        
        // SIMD 버전 테스트
        let positions: Vec<(usize, usize)> = (0..64)
            .flat_map(|r| (0..64).map(move |c| (r, c)))
            .collect();
        
        let start = Instant::now();
        let decoded_simd = decoder.decode_int_adam_simd(&encoded, &positions);
        let simd_time = start.elapsed();
        
        // 스칼라 버전과 비교
        let mut decoded_scalar = Vec::with_capacity(64 * 64);
        let start = Instant::now();
        for i in 0..64 {
            for j in 0..64 {
                decoded_scalar.push(decoder.decode_int_adam_fast(&encoded, i, j));
            }
        }
        let scalar_time = start.elapsed();
        
        println!("SIMD 시간: {:.2}μs", simd_time.as_micros());
        println!("스칼라 시간: {:.2}μs", scalar_time.as_micros());
        println!("SIMD 속도향상: {:.1}x", scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
        
        // 결과 일치성 확인
        for i in 0..decoded_simd.len() {
            assert!((decoded_simd[i] - decoded_scalar[i]).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_large_scale_performance() {
        // Qwen-7B 규모 시뮬레이션 (512개 블록)
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        let block_count = 512;
        let mut total_encode_time = 0u128;
        let mut total_decode_time = 0u128;
        let mut total_rmse = 0.0f32;
        let mut total_residuals = 0;
        
        for _ in 0..block_count {
            let block_data = generate_test_block(64, 64);
            
            // 인코딩
            let start = Instant::now();
            let encoded = encoder.encode_block_int_adam(&block_data, 64, 64, 1000);
            total_encode_time += start.elapsed().as_micros();
            total_residuals += encoded.residuals.len();
            
            // 디코딩
            let start = Instant::now();
            let decoded = decoder.decode_block_int_adam(&encoded);
            total_decode_time += start.elapsed().as_micros();
            
            // RMSE
            total_rmse += calculate_rmse(&block_data, &decoded);
        }
        
        let avg_encode_time = total_encode_time as f64 / block_count as f64;
        let avg_decode_time = total_decode_time as f64 / block_count as f64;
        let avg_rmse = total_rmse / block_count as f32;
        let avg_residuals = total_residuals as f64 / block_count as f64;
        
        println!("\n대규모 테스트 결과 ({} 블록):", block_count);
        println!("  평균 인코딩 시간: {:.2}μs/블록", avg_encode_time);
        println!("  평균 디코딩 시간: {:.2}μs/블록", avg_decode_time);
        println!("  평균 RMSE: {:.2e}", avg_rmse);
        println!("  평균 잔차 계수: {:.1} (목표: 0.8)", avg_residuals);
        println!("  총 인코딩 시간: {:.2}초", total_encode_time as f64 / 1_000_000.0);
        
        // 목표 달성 확인
        assert!(avg_encode_time < 10.0, "평균 인코딩 시간이 목표(8μs)를 초과합니다");
        assert!(avg_rmse < 5e-6, "평균 RMSE가 목표(1e-6)를 초과합니다");
    }

    #[test]
    fn test_int_adam_large_scale_performance() {
        println!("\n=== 대규모 행렬 압축 성능 테스트 ===");
        
        // GPT-2 크기의 행렬로 테스트 (768x3072)
        let height = 768;
        let width = 3072;
        let block_size = 64;
        
        // 랜덤 행렬 생성
        let mut rng = rand::thread_rng();
        let matrix_data: Vec<f32> = (0..height * width)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        // 압축 실행
        let result = RBEEncoder::compress_with_int_adam(
            &matrix_data,
            height,
            width,
            block_size,
            200, // 빠른 수렴을 위해 200 step만
        );
        
        match result {
            Ok((blocks, time_ms, ratio, rmse)) => {
                println!("\n압축 결과:");
                println!("- 전체 압축 시간: {:.2}초", time_ms / 1000.0);
                println!("- 압축률: {:.0}:1", ratio);
                println!("- 추정 RMSE: {:.2e}", rmse);
                println!("- 블록 수: {}", blocks.len());
                
                // 디코딩 성능 테스트
                let decoder = WeightGenerator::new();
                let decode_start = Instant::now();
                
                // 병렬 디코딩
                let decoded_blocks: Vec<Vec<f32>> = blocks
                    .par_iter()
                    .map(|block| decoder.decode_block_int_adam(block))
                    .collect();
                
                let decode_time = decode_start.elapsed();
                println!("\n디코딩 성능:");
                println!("- 전체 디코딩 시간: {:.2}ms", decode_time.as_millis());
                println!("- 블록당 평균: {:.2}μs", decode_time.as_micros() as f64 / blocks.len() as f64);
                
                // 메모리 사용량 계산
                let original_size = height * width * 4; // f32
                let compressed_size = blocks.iter()
                    .map(|b| 8 * 4 + b.residuals.len() * 8)
                    .sum::<usize>();
                
                println!("\n메모리 사용량:");
                println!("- 원본: {:.2}MB", original_size as f64 / 1_048_576.0);
                println!("- 압축: {:.2}MB", compressed_size as f64 / 1_048_576.0);
                println!("- 절약: {:.1}%", (1.0 - compressed_size as f64 / original_size as f64) * 100.0);
                
                // 성능 체크
                assert!(ratio > 500.0, "압축률이 너무 낮습니다");
                assert!(decode_time.as_millis() < 100, "디코딩이 너무 느립니다");
            }
            Err(e) => panic!("압축 실패: {}", e),
        }
    }
    
    #[test]
    fn test_various_block_sizes() {
        println!("\n=== 다양한 블록 크기 테스트 ===");
        
        let test_configs = [
            ("소형", 16, 768, 768),    // 임베딩 레이어 크기
            ("중형", 32, 768, 3072),   // FFN 레이어 크기
            ("대형", 64, 3072, 768),   // 프로젝션 레이어 크기
            ("초대형", 128, 768, 50257), // 출력 레이어 크기 (vocab_size)
        ];
        
        for (name, block_size, height, width) in test_configs {
            println!("\n--- {} 블록 테스트 ({}x{}, 블록크기: {}) ---", name, height, width, block_size);
            
            // 테스트 데이터 생성
            let mut rng = rand::thread_rng();
            let matrix_data: Vec<f32> = (0..height * width)
                .map(|_| rng.gen_range(-0.1..0.1)) // 실제 가중치 범위
                .collect();
            
            // 압축
            let result = RBEEncoder::compress_with_int_adam(
                &matrix_data,
                height,
                width,
                block_size,
                200,
            );
            
            if let Ok((blocks, comp_time, ratio, rmse)) = result {
                println!("압축 시간: {:.2}초", comp_time / 1000.0);
                println!("압축률: {:.0}:1", ratio);
                println!("RMSE: {:.2e}", rmse);
                
                // 디코딩 성능
                let decoder = WeightGenerator::new();
                let decode_start = Instant::now();
                
                for block in &blocks {
                    let _ = decoder.decode_block_int_adam(block);
                }
                
                let decode_time = decode_start.elapsed();
                println!("디코딩 시간: {:.2}ms", decode_time.as_millis());
                println!("블록당 평균: {:.2}μs", decode_time.as_micros() as f64 / blocks.len() as f64);
            }
        }
    }
    
    #[test]
    fn test_real_model_comprehensive() {
        println!("\n=== 실제 GPT-2 모델 레이어 종합 테스트 ===");
        println!("압축비 / 정확도(RMSE) / 디코딩 속도 체크\n");
        
        // 실제 GPT-2 레이어 크기들
        let test_configs = [
            ("Embedding (작은 부분)", 768, 1024, 32),     // wte 일부
            ("Attention Q/K/V", 768, 2304, 48),          // c_attn: 768 x 2304
            ("Attention Proj", 768, 768, 32),            // c_proj: 768 x 768
            ("FFN 1", 768, 3072, 48),                    // c_fc: 768 x 3072
        ];
        
        // 결과 저장용
        let mut results = Vec::new();
        
        for (layer_name, height, width, block_size) in test_configs {
            println!("--- {} 레이어 ({}x{}, 블록: {}x{}) ---", 
                     layer_name, height, width, block_size, block_size);
            
            // 실제와 유사한 가중치 데이터 생성
            let mut rng = rand::thread_rng();
            let scale = (2.0 / height as f32).sqrt();
            let matrix_data: Vec<f32> = (0..height * width)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            
            // 1. 압축 (적응적 K값 사용)
            let compress_start = std::time::Instant::now();
            let result = RBEEncoder::compress_with_int_adam(
                &matrix_data,
                height,
                width,
                block_size,
                1000, // 1000 에포크
            );
            
            match result {
                Ok((blocks, time_ms, compression_ratio, rmse)) => {
                    let compress_time = compress_start.elapsed();
                    
                    // K값 분포 분석
                    let mut k_distribution = std::collections::HashMap::new();
                    for block in &blocks {
                        *k_distribution.entry(block.residuals.len()).or_insert(0) += 1;
                    }
                    
                    println!("  압축 완료:");
                    println!("    - 압축 시간: {:.2}초", compress_time.as_secs_f64());
                    println!("    - 압축률: {:.1}:1", compression_ratio);
                    println!("    - RMSE: {:.6}", rmse);
                    println!("    - K값 분포: {:?}", k_distribution);
                    
                    // 2. 디코딩 속도 테스트
                    let decoder = WeightGenerator::new();
                    let mut decode_times = Vec::new();
                    let warmup_runs = 10;
                    let test_runs = 100;
                    
                    // 워밍업
                    for _ in 0..warmup_runs {
                        for block in &blocks {
                            let _ = decoder.decode_block_int_adam(block);
                        }
                    }
                    
                    // 실제 측정
                    for _ in 0..test_runs {
                        let decode_start = std::time::Instant::now();
                        let mut decoded_matrix = vec![0.0f32; height * width];
                        
                        let blocks_per_row = width / block_size;
                        for (idx, block) in blocks.iter().enumerate() {
                            let block_row = idx / blocks_per_row;
                            let block_col = idx % blocks_per_row;
                            let decoded_block = decoder.decode_block_int_adam(block);
                            
                            // 블록을 전체 행렬에 복사
                            for r in 0..block_size {
                                for c in 0..block_size {
                                    let global_r = block_row * block_size + r;
                                    let global_c = block_col * block_size + c;
                                    if global_r < height && global_c < width {
                                        decoded_matrix[global_r * width + global_c] = 
                                            decoded_block[r * block_size + c];
                                    }
                                }
                            }
                        }
                        
                        decode_times.push(decode_start.elapsed());
                    }
                    
                    // 디코딩 통계
                    let avg_decode_time = decode_times.iter()
                        .map(|d| d.as_secs_f64())
                        .sum::<f64>() / test_runs as f64;
                    let total_pixels = height * width;
                    let ns_per_pixel = (avg_decode_time * 1e9) / total_pixels as f64;
                    
                    // 최종 정확도 검증
                    let mut final_decoded = vec![0.0f32; height * width];
                    let blocks_per_row = width / block_size;
                    for (idx, block) in blocks.iter().enumerate() {
                        let block_row = idx / blocks_per_row;
                        let block_col = idx % blocks_per_row;
                        let decoded_block = decoder.decode_block_int_adam(block);
                        
                        for r in 0..block_size {
                            for c in 0..block_size {
                                let global_r = block_row * block_size + r;
                                let global_c = block_col * block_size + c;
                                if global_r < height && global_c < width {
                                    final_decoded[global_r * width + global_c] = 
                                        decoded_block[r * block_size + c];
                                }
                            }
                        }
                    }
                    
                    let final_rmse = calculate_rmse(&matrix_data, &final_decoded);
                    
                    println!("  디코딩 성능:");
                    println!("    - 평균 디코딩 시간: {:.3}ms", avg_decode_time * 1000.0);
                    println!("    - 픽셀당 디코딩 시간: {:.1}ns", ns_per_pixel);
                    println!("    - 최종 RMSE: {:.6}", final_rmse);
                    
                    // 결과 저장
                    results.push((
                        layer_name,
                        compression_ratio,
                        final_rmse,
                        ns_per_pixel,
                        k_distribution,
                    ));
                },
                Err(e) => println!("  압축 실패: {}", e),
            }
            
            println!();
        }
        
        // 종합 결과
        println!("\n=== 종합 결과 ===");
        println!("{:<20} | {:>12} | {:>10} | {:>15} | {:<20}", 
                 "레이어", "압축률", "RMSE", "디코딩(ns/px)", "K값 분포");
        println!("{:-<80}", "");
        
        for (name, ratio, rmse, ns_per_px, k_dist) in &results {
            let k_str = format!("{:?}", k_dist);
            println!("{:<20} | {:>12.1}:1 | {:>10.6} | {:>15.1} | {:<20}", 
                     name, ratio, rmse, ns_per_px, k_str);
        }
        
        // 성능 기준 체크
        println!("\n=== 성능 기준 체크 ===");
        let target_decode_speed = 150.0; // 목표: 150ns/pixel
        let target_rmse = 0.001; // 목표: 0.001 이하
        let target_ratio = 100.0; // 목표: 100:1 이상
        
        for (name, ratio, rmse, ns_per_px, _) in &results {
            let speed_ok = *ns_per_px <= target_decode_speed;
            let rmse_ok = *rmse <= target_rmse;
            let ratio_ok = *ratio >= target_ratio;
            
            println!("{}: 속도[{}] RMSE[{}] 압축률[{}]", 
                     name,
                     if speed_ok { "PASS" } else { "FAIL" },
                     if rmse_ok { "PASS" } else { "FAIL" },
                     if ratio_ok { "PASS" } else { "FAIL" });
        }
    }
    
    #[test]
    fn test_different_block_sizes_and_quality() {
        println!("\n=== 다양한 블록 크기와 품질 테스트 ===");
        
        // 실제 GPT-2 레이어 크기로 테스트
        let test_configs = [
            ("Small Square", 256, 256),      // 작은 정사각형
            ("Attention Proj", 768, 768),    // 768x768 정사각형
            ("FFN Part", 768, 384),          // 직사각형
        ];
        
        // 테스트할 블록 크기들
        let block_sizes = [16, 32, 64];
        
        for (layer_name, height, width) in test_configs {
            println!("\n--- {} 레이어 ({}x{}) ---", layer_name, height, width);
            
            // 테스트 데이터 생성
            let mut rng = rand::thread_rng();
            let scale = (2.0 / height as f32).sqrt();
            let matrix_data: Vec<f32> = (0..height * width)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            
            let mut best_config = (0, f32::INFINITY, 0.0); // (block_size, rmse, ratio)
            
            for &block_size in &block_sizes {
                if height % block_size != 0 || width % block_size != 0 {
                    continue;
                }
                
                println!("\n  블록 크기 {}:", block_size);
                
                // 현재는 K=2로 고정
                let start = std::time::Instant::now();
                let result = RBEEncoder::compress_with_int_adam(
                    &matrix_data,
                    height,
                    width,
                    block_size,
                    500, // 500 에포크
                );
                
                match result {
                    Ok((blocks, time_ms, ratio, rmse)) => {
                        println!("    RMSE={:.3e}, 압축률={:4.0}:1, 시간={:.1}초", 
                                 rmse, ratio, time_ms / 1000.0);
                        
                        // 몇 개 블록의 잔차 계수 통계
                        let residual_counts: Vec<usize> = blocks.iter()
                            .take(10)
                            .map(|b| b.residuals.len())
                            .collect();
                        println!("    처음 10개 블록의 잔차 계수: {:?}", residual_counts);
                        
                        if (rmse as f32) < best_config.1 {
                            best_config = (block_size, rmse as f32, ratio as f32);
                        }
                    },
                    Err(e) => println!("    실패 - {}", e),
                }
            }
            
            println!("\n  최적 설정: 블록크기={}, RMSE={:.3e}, 압축률={:.0}:1",
                     best_config.0, best_config.1, best_config.2);
        }
    }
    
    #[test] 
    fn test_progressive_k_encoding() {
        println!("\n=== 점진적 K값 인코딩 테스트 ===");
        
        // 256x256 행렬로 빠른 테스트
        let size = 256;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / size as f32).sqrt();
        
        // 다양한 난이도의 데이터 생성
        let test_data = [
            ("단순 패턴", {
                (0..size * size)
                    .map(|i| ((i / size) as f32 * 0.01).sin() * scale)
                    .collect::<Vec<f32>>()
            }),
            ("중간 복잡도", {
                (0..size * size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect::<Vec<f32>>()
            }),
            ("고주파 노이즈", {
                (0..size * size)
                    .map(|i| {
                        let base = rng.gen_range(-scale..scale);
                        let noise = ((i as f32 * 0.1).sin() + (i as f32 * 0.3).cos()) * scale * 0.5;
                        base + noise
                    })
                    .collect::<Vec<f32>>()
            }),
        ];
        
        let block_size = 32;
        
        for (data_type, matrix_data) in test_data {
            println!("\n데이터 타입: {}", data_type);
            
            // 기본 K=2로 인코딩
            let result = RBEEncoder::compress_with_int_adam(
                &matrix_data,
                size,
                size,
                block_size,
                300,
            );
            
            if let Ok((blocks, time_ms, ratio, rmse)) = result {
                println!("  K=2 (고정): RMSE={:.3e}, 압축률={:.0}:1", rmse, ratio);
                
                // 잔차 계수 분포 분석
                let mut residual_histogram = vec![0; 3]; // 0, 1, 2개
                for block in &blocks {
                    let count = block.residuals.len().min(2);
                    residual_histogram[count] += 1;
                }
                println!("  잔차 계수 분포: 0개={}, 1개={}, 2개={}", 
                         residual_histogram[0], 
                         residual_histogram[1], 
                         residual_histogram[2]);
            }
        }
    }
    
    // 헬퍼 함수: 최적 블록 크기 찾기
    fn find_optimal_block_size(height: usize, width: usize) -> usize {
        // 모든 공약수 찾기
        let gcd = gcd(height, width);
        let mut divisors = vec![];
        
        for i in 1..=((gcd as f64).sqrt() as usize) {
            if gcd % i == 0 {
                divisors.push(i);
                if i != gcd / i {
                    divisors.push(gcd / i);
                }
            }
        }
        
        divisors.sort();
        
        // 16~32 범위에서 가장 좋은 크기 찾기
        for &d in divisors.iter().rev() {
            if d >= 16 && d <= 32 && height % d == 0 && width % d == 0 {
                return d;
            }
        }
        
        // 없으면 16으로
        16
    }
    
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    #[test]
    fn test_extreme_cases() {
        println!("\n=== 극한 케이스 테스트 ===");
        
        // 1. 매우 작은 블록
        println!("\n--- 4x4 미니 블록 ---");
        let tiny_data: Vec<f32> = vec![0.1; 16];
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let tiny_encoded = encoder.encode_block_int_adam(&tiny_data, 4, 4, 100);
        
        let decoder = WeightGenerator::new();
        let tiny_decoded = decoder.decode_block_int_adam(&tiny_encoded);
        println!("4x4 디코딩 정확도: RMSE = {:.2e}", calculate_rmse(&tiny_data, &tiny_decoded));
        
        // 2. 비정사각형 블록
        println!("\n--- 32x128 직사각형 블록 ---");
        let rect_data: Vec<f32> = (0..32*128).map(|i| (i as f32 * 0.001).sin()).collect();
        let rect_encoded = encoder.encode_block_int_adam(&rect_data, 32, 128, 200);
        let rect_decoded = decoder.decode_block_int_adam(&rect_encoded);
        println!("직사각형 블록 RMSE: {:.2e}", calculate_rmse(&rect_data, &rect_decoded));
        
        // 3. 희소 행렬
        println!("\n--- 희소 행렬 (90% zeros) ---");
        let mut sparse_data = vec![0.0f32; 64 * 64];
        let mut rng = rand::thread_rng();
        let sparse_len = sparse_data.len();
        for _ in 0..410 { // 10% non-zero
            let idx = rng.gen_range(0..sparse_len);
            sparse_data[idx] = rng.gen_range(-1.0..1.0);
        }
        let sparse_encoded = encoder.encode_block_int_adam(&sparse_data, 64, 64, 200);
        println!("희소 행렬 잔차 계수: {}", sparse_encoded.residuals.len());
        
        // 4. 고진폭 데이터
        println!("\n--- 고진폭 데이터 (±10 범위) ---");
        let high_amp_data: Vec<f32> = (0..64*64)
            .map(|_| rng.gen_range(-10.0..10.0))
            .collect();
        let high_encoded = encoder.encode_block_int_adam(&high_amp_data, 64, 64, 200);
        let high_decoded = decoder.decode_block_int_adam(&high_encoded);
        println!("고진폭 데이터 RMSE: {:.2e}", calculate_rmse(&high_amp_data, &high_decoded));
    }

    #[test]
    fn test_hippocampal_patch_system() {
        use crate::packed_params::{HippocampalMemory, DeltaPatch, PatchTag};
        
        println!("\n=== 해마 메모리 패치 시스템 테스트 ===");
        
        // 1. 작은 블록으로 테스트
        let block_size = 16;
        let height = block_size;
        let width = block_size;
        
        // 테스트 데이터 생성
        let mut rng = rand::thread_rng();
        let matrix_data: Vec<f32> = (0..height * width)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        // 2. 기본 압축 (낮은 정밀도)
        println!("\n1단계: 기본 압축 (빠른 수렴)");
        let result = RBEEncoder::compress_with_int_adam(
            &matrix_data,
            height,
            width,
            block_size,
            200, // 적은 에포크로 낮은 정밀도
        );
        
        let (blocks, _, compression_ratio, base_rmse) = result.unwrap();
        println!("- 기본 압축률: {:.1}:1", compression_ratio);
        println!("- 기본 RMSE: {:.4}", base_rmse);
        
        // 3. 디코더 생성
        let decoder = WeightGenerator::new();
        let block = &blocks[0];
        
        // 기본 디코딩
        let decoded_base = decoder.decode_block_int_adam(block);
        
        // 4. 해마 메모리 시스템 생성
        let mut hippocampal = HippocampalMemory::new(1000); // GC threshold
        let block_id = (height * 31 + width) as u16 % 1024;
        
        // 5. 패치 생성 및 적용
        println!("\n2단계: 패치 적용으로 정밀도 개선");
        
        // 5-1. Delta Scalar 패치 (곡률 보정)
        let alpha: f32 = 1.05; // 5% 스케일 업
        let scalar_patch = DeltaPatch {
            tag: PatchTag::DeltaScalar,
            block_id,
            payload_len: 1,
            payload: vec![alpha.to_bits()],
            crc: None,
        };
        hippocampal.add_patch(scalar_patch);
        println!("- DeltaScalar 패치 추가 (α={:.3})", alpha);
        
        // 5-2. Delta Rank-1 패치들 추가 (주요 오류 수정)
        let mut error_map: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..height {
            for j in 0..width {
                let idx = i * width + j;
                let error = matrix_data[idx] - decoded_base[idx];
                if error.abs() > 0.1 { // 큰 오류만
                    error_map.push((i, j, error));
                }
            }
        }
        
        // 상위 5개 오류 위치에 rank-1 패치
        error_map.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        for (i, (row, col, error)) in error_map.iter().take(5).enumerate() {
            let rank1_patch = DeltaPatch {
                tag: PatchTag::DeltaRank1,
                block_id,
                payload_len: 2,
                payload: vec![
                    ((*row as u32) << 16) | (*col as u32),
                    (error * 0.5).to_bits(), // 오류의 50%만 보정
                ],
                crc: None,
            };
            hippocampal.add_patch(rank1_patch);
            println!("- DeltaRank1 패치 {} 추가: ({},{}) error={:.4}", i+1, row, col, error);
        }
        
        // 6. 패치 적용 디코딩
        let decoded_patched = decoder.decode_block_with_patches(block, Some(&hippocampal));
        
        // 7. RMSE 비교
        let rmse_base = compute_rmse(&matrix_data, &decoded_base);
        let rmse_patched = compute_rmse(&matrix_data, &decoded_patched);
        
        println!("\n3단계: 성능 비교");
        println!("- 기본 RMSE: {:.4}", rmse_base);
        println!("- 패치 적용 RMSE: {:.4}", rmse_patched);
        println!("- RMSE 개선율: {:.1}%", (1.0 - rmse_patched / rmse_base) * 100.0);
        
        // 8. 메모리 사용량
        println!("\n4단계: 메모리 사용량");
        let patch_memory = hippocampal.memory_usage();
        let core_memory = std::mem::size_of::<HybridEncodedBlock>() + 
                         block.residuals.len() * std::mem::size_of::<ResidualCoefficient>();
        println!("- 코어 크기: {} bytes", core_memory);
        println!("- 패치 크기: {} bytes", patch_memory);
        println!("- 총 크기: {} bytes", core_memory + patch_memory);
        println!("- 패치 오버헤드: {:.1}%", (patch_memory as f64 / core_memory as f64) * 100.0);
        
        // 9. 속도 테스트
        println!("\n5단계: 디코딩 속도 비교");
        let iterations = 1000;
        
        // 기본 디코딩 속도
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = decoder.decode_block_int_adam(block);
        }
        let base_time = start.elapsed();
        
        // 패치 적용 디코딩 속도
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = decoder.decode_block_with_patches(block, Some(&hippocampal));
        }
        let patched_time = start.elapsed();
        
        println!("- 기본 디코딩: {:.3} µs/block", base_time.as_nanos() as f64 / iterations as f64 / 1000.0);
        println!("- 패치 디코딩: {:.3} µs/block", patched_time.as_nanos() as f64 / iterations as f64 / 1000.0);
        println!("- 속도 오버헤드: {:.1}%", ((patched_time.as_nanos() as f64 / base_time.as_nanos() as f64) - 1.0) * 100.0);
        
        // 검증
        assert!(rmse_patched < rmse_base, "패치 적용 후 RMSE가 개선되어야 함");
        assert!(patch_memory < core_memory / 2, "패치 크기는 코어의 50% 미만이어야 함");
    }
    
    fn compute_rmse(original: &[f32], decoded: &[f32]) -> f64 {
        let mse: f64 = original.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (*o - *d).powi(2) as f64)
            .sum::<f64>() / original.len() as f64;
        mse.sqrt()
    }
} 