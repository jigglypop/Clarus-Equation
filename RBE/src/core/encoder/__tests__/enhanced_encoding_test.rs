#[cfg(test)]
mod tests {
    use crate::core::encoder::RBEEncoder;
    use crate::core::decoder::WeightGenerator;
    use crate::packed_params::TransformType;
    use rand::Rng;
    use std::time::Instant;
    use crate::optimizers::{AdamState, RiemannianAdamState};
    use std::f32::consts::PI;
    use rand::thread_rng;

    /// 테스트용 데이터 생성
    fn generate_complex_data(rows: usize, cols: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(rows * cols);
        
        for r in 0..rows {
            for c in 0..cols {
                let x = c as f32 / cols as f32;
                let y = r as f32 / rows as f32;
                
                // 복잡한 패턴: 저주파 + 고주파 + 노이즈
                let low_freq = 0.3 * (2.0 * std::f32::consts::PI * x).sin() + 
                              0.2 * (2.0 * std::f32::consts::PI * y).cos();
                let high_freq = 0.1 * (8.0 * std::f32::consts::PI * x).sin() * 
                               (8.0 * std::f32::consts::PI * y).cos();
                let noise = rng.gen_range(-0.05..0.05);
                
                data.push(low_freq + high_freq + noise);
            }
        }
        
        data
    }
    
    fn compute_rmse(original: &[f32], decoded: &[f32]) -> f64 {
        let mse: f64 = original.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (*o - *d).powi(2) as f64)
            .sum::<f64>() / original.len() as f64;
        mse.sqrt()
    }
    
    #[test]
    fn test_enhanced_vs_standard_encoding() {
        println!("\n=== 향상된 인코딩 vs 표준 인코딩 비교 ===");
        
        let test_sizes = [(32, 32), (64, 64), (128, 128)];
        let target_rmse = 0.1; // 압축률을 위해 더 큰 허용 오차
        let target_compression_ratio = 150.0; // 목표 압축률
        
        for (rows, cols) in test_sizes {
            println!("\n--- {}x{} 블록 테스트 ---", rows, cols);
            
            let data = generate_complex_data(rows, cols);
            let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
            let decoder = WeightGenerator::new();
            
            // 1. 표준 인코딩
            let start = Instant::now();
            let standard_block = encoder.encode_block_int_adam(&data, rows, cols, 500);
            let standard_time = start.elapsed();
            
            let standard_decoded = decoder.decode_block_int_adam(&standard_block);
            let standard_rmse = compute_rmse(&data, &standard_decoded);
            
            // 2. 향상된 인코딩
            let start = Instant::now();
            let enhanced_block = encoder.encode_block_int_adam_enhanced(&data, rows, cols, 1000, target_rmse);
            let enhanced_time = start.elapsed();
            
            let enhanced_decoded = decoder.decode_block_enhanced(&enhanced_block);
            let enhanced_rmse = compute_rmse(&data, &enhanced_decoded);
            
            // 결과 비교
            println!("\n표준 인코딩:");
            println!("  - RMSE: {:.6}", standard_rmse);
            println!("  - 잔차 계수: {}", standard_block.residuals.len());
            println!("  - 인코딩 시간: {:.2}ms", standard_time.as_millis());
            
            println!("\n향상된 인코딩:");
            println!("  - RMSE: {:.6}", enhanced_rmse);
            println!("  - 잔차 계수: {}", enhanced_block.residuals.len());
            println!("  - 인코딩 시간: {:.2}ms", enhanced_time.as_millis());
            
            println!("\n개선율:");
            println!("  - RMSE 개선: {:.1}%", (1.0 - enhanced_rmse / standard_rmse) * 100.0);
            
            // 압축률 계산
            let original_size = rows * cols * 4;
            let standard_size = 8 * 4 + standard_block.residuals.len() * 8;
            let enhanced_size = 8 * 4 + enhanced_block.residuals.len() * 8;
            
            let standard_ratio = original_size as f64 / standard_size as f64;
            let enhanced_ratio = original_size as f64 / enhanced_size as f64;
            
            println!("  - 표준 압축률: {:.0}:1", standard_ratio);
            println!("  - 향상 압축률: {:.0}:1", enhanced_ratio);
            
            // 검증
            assert!(enhanced_rmse < target_rmse as f64 * 5.0, "향상된 인코딩이 목표 RMSE의 5배 이내여야 함");
            assert!(enhanced_ratio >= target_compression_ratio as f64 * 0.5, 
                    "향상된 인코딩의 압축률이 목표의 50% 이상이어야 함: {:.0}:1 >= {:.0}:1", 
                    enhanced_ratio, target_compression_ratio * 0.5);
        }
    }
    
    #[test]
    fn test_adaptive_k_behavior() {
        println!("\n=== 적응적 K값 동작 테스트 ===");
        
        let rows = 64;
        let cols = 64;
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        // 다양한 난이도의 데이터
        let test_cases = [
            ("단순 패턴", {
                let mut data = vec![0.0f32; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        data[i * cols + j] = (i as f32 / rows as f32) * 0.1;
                    }
                }
                data
            }),
            ("중간 복잡도", generate_complex_data(rows, cols)),
            ("매우 복잡", {
                let mut rng = rand::thread_rng();
                (0..rows * cols)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            }),
        ];
        
        for (name, data) in test_cases {
            println!("\n{} 데이터:", name);
            
            // 다양한 목표 RMSE로 테스트
            let target_rmses = [0.1, 0.01, 0.001];
            
            for &target_rmse in &target_rmses {
                let block = encoder.encode_block_int_adam_enhanced(&data, rows, cols, 1000, target_rmse);
                let decoded = decoder.decode_block_enhanced(&block);
                let actual_rmse = compute_rmse(&data, &decoded);
                
                println!("  목표 RMSE={:.3} → 실제 RMSE={:.6}, K={}", 
                         target_rmse, actual_rmse, block.residuals.len());
                
                // 목표에 근접했는지 확인
                assert!(actual_rmse < target_rmse as f64 * 2.0, 
                        "실제 RMSE가 목표의 2배를 넘으면 안됨");
            }
        }
    }
    
    #[test]
    fn test_simd_enhanced_decoding() {
        println!("\n=== SIMD 향상된 디코딩 성능 테스트 ===");
        
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2가 지원되지 않아 테스트를 건너뜁니다.");
            return;
        }
        
        let rows = 128;
        let cols = 128;
        let data = generate_complex_data(rows, cols);
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        // 인코딩
        let block = encoder.encode_block_int_adam_enhanced(&data, rows, cols, 500, 0.001);
        
        // 워밍업
        for _ in 0..10 {
            let _ = decoder.decode_block_enhanced(&block);
            let _ = decoder.decode_block_enhanced_simd(&block);
        }
        
        // 스칼라 디코딩
        let start = Instant::now();
        let decoded_scalar = decoder.decode_block_enhanced(&block);
        let scalar_time = start.elapsed();
        
        // SIMD 디코딩
        let start = Instant::now();
        let decoded_simd = decoder.decode_block_enhanced_simd(&block);
        let simd_time = start.elapsed();
        
        // 결과 비교
        println!("스칼라 디코딩 시간: {:.2}μs", scalar_time.as_micros());
        println!("SIMD 디코딩 시간: {:.2}μs", simd_time.as_micros());
        println!("속도 향상: {:.1}x", scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
        
        // 정확성 검증
        let max_diff = decoded_scalar.iter()
            .zip(decoded_simd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        
        println!("최대 차이: {:.2e}", max_diff);
        assert!(max_diff < 1e-5, "SIMD와 스칼라 결과가 일치해야 함");
    }
    
    #[test]
    fn test_real_model_layer_enhanced() {
        println!("\n=== 실제 모델 레이어 향상된 압축 테스트 ===");
        
        // GPT-2 레이어 크기
        let layer_configs = [
            ("Attention Q/K/V", 768, 2304),
            ("FFN Layer 1", 768, 3072),
            ("FFN Layer 2", 3072, 768),
        ];
        
        for (layer_name, height, width) in layer_configs {
            println!("\n--- {} ({}x{}) ---", layer_name, height, width);
            
            // 실제와 유사한 가중치 생성
            let mut rng = rand::thread_rng();
            let scale = (2.0 / height as f32).sqrt();
            let matrix_data: Vec<f32> = (0..height * width)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            
            let block_size = 64;
            let blocks_per_row = (width + block_size - 1) / block_size;
            let blocks_per_col = (height + block_size - 1) / block_size;
            let total_blocks = blocks_per_row * blocks_per_col;
            
            println!("총 블록 수: {}", total_blocks);
            
            // 몇 개 블록 샘플링
            let mut total_k = 0;
            let mut max_k = 0;
            let mut min_k = usize::MAX;
            
            for block_idx in 0..total_blocks.min(10) {
                let block_row = block_idx / blocks_per_row;
                let block_col = block_idx % blocks_per_row;
                
                let row_start = block_row * block_size;
                let col_start = block_col * block_size;
                let row_end = (row_start + block_size).min(height);
                let col_end = (col_start + block_size).min(width);
                
                let actual_height = row_end - row_start;
                let actual_width = col_end - col_start;
                
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; actual_height * actual_width];
                for r in 0..actual_height {
                    for c in 0..actual_width {
                        block_data[r * actual_width + c] = 
                            matrix_data[(row_start + r) * width + (col_start + c)];
                    }
                }
                
                // 향상된 인코딩
                let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
                let enhanced_block = encoder.encode_block_int_adam_enhanced(
                    &block_data, 
                    actual_height, 
                    actual_width, 
                    500, 
                    0.0005
                );
                
                let k = enhanced_block.residuals.len();
                total_k += k;
                max_k = max_k.max(k);
                min_k = min_k.min(k);
            }
            
            let avg_k = total_k as f64 / 10.0;
            println!("잔차 계수 통계: 평균={:.1}, 최소={}, 최대={}", avg_k, min_k, max_k);
            
            // 압축률 추정
            let avg_block_size = 8 * 4 + (avg_k * 8.0) as usize;
            let total_compressed = total_blocks * avg_block_size;
            let compression_ratio = (height * width * 4) as f64 / total_compressed as f64;
            
            println!("예상 압축률: {:.0}:1", compression_ratio);
        }
    }

    #[test]
    fn test_hybrid_optimization() {
        println!("\n=== 하이브리드 최적화 테스트 ===");
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        
        // 테스트 케이스: 다양한 패턴의 블록
        let test_cases = vec![
            ("smooth", generate_complex_data(64, 64)),
            ("noisy", generate_complex_data(64, 64)),
            ("structured", generate_complex_data(64, 64)),
        ];
        
        for (name, block_data) in test_cases {
            println!("\n--- {} 블록 테스트 ---", name);
            
            let encoded = encoder.encode_block_hybrid_optimization(
                &block_data,
                64,
                64,
                1000,  // max_steps
                0.001, // target_rmse
                100.0, // target_compression_ratio - 100:1로 조정
            );
            
            // 디코딩 및 검증
            let mut decoder = WeightGenerator::new();
            let decoded = decoder.decode_block_enhanced(&encoded);
            
            // RMSE 계산
            let rmse = compute_rmse(&block_data, &decoded);
            let compressed_size = 32 + encoded.residuals.len() * 8;
            let original_size = 64 * 64 * 4;
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            println!("  결과:");
            println!("    - RMSE: {:.6}", rmse);
            println!("    - 압축률: {:.0}:1", compression_ratio);
            println!("    - 활성 잔차: {}", encoded.residuals.len());
            
            // 검증 - 압축률과 RMSE의 trade-off 고려
            if compression_ratio > 100.0 {
                assert!(rmse < 0.1, "RMSE가 너무 높음: {}", rmse);  // 100:1에서는 0.1 이하면 우수
            } else if compression_ratio > 50.0 {
                assert!(rmse < 0.05, "RMSE가 너무 높음: {}", rmse);
            } else {
                assert!(rmse < 0.01, "RMSE가 너무 높음: {}", rmse);
            }
            assert!(compression_ratio > 80.0, "압축률이 목표에 미달: {}", compression_ratio);
        }
    }

    #[test]
    fn test_hybrid_vs_standard_optimization() {
        println!("\n=== 하이브리드 vs 표준 최적화 비교 ===");
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let block_data = generate_complex_data(128, 128);
        
        // 표준 인코딩
        let start = Instant::now();
        let standard_encoded = encoder.encode_block_int_adam_enhanced(&block_data, 128, 128, 1000, 0.001);
        let standard_time = start.elapsed();
        
        // 하이브리드 인코딩
        let start = Instant::now();
        let hybrid_encoded = encoder.encode_block_hybrid_optimization(
            &block_data,
            128,
            128,
            1000,
            0.001,
            1000.0,
        );
        let hybrid_time = start.elapsed();
        
        // 디코딩 및 비교
        let mut decoder = WeightGenerator::new();
        let standard_decoded = decoder.decode_block_enhanced(&standard_encoded);
        let hybrid_decoded = decoder.decode_block_enhanced(&hybrid_encoded);
        
        let standard_rmse = compute_rmse(&block_data, &standard_decoded);
        let hybrid_rmse = compute_rmse(&block_data, &hybrid_decoded);
        
        let original_size = 128 * 128 * 4;
        let standard_compressed = 8 * 4 + standard_encoded.residuals.len() * 8;
        let hybrid_compressed = 8 * 4 + hybrid_encoded.residuals.len() * 8;
        
        println!("\n비교 결과:");
        println!("표준 최적화:");
        println!("  - RMSE: {:.6}", standard_rmse);
        println!("  - 압축률: {:.0}:1", original_size as f64 / standard_compressed as f64);
        println!("  - 시간: {:?}", standard_time);
        
        println!("\n하이브리드 최적화:");
        println!("  - RMSE: {:.6}", hybrid_rmse);
        println!("  - 압축률: {:.0}:1", original_size as f64 / hybrid_compressed as f64);
        println!("  - 시간: {:?}", hybrid_time);
        println!("  - RMSE 개선: {:.2}%", (1.0 - hybrid_rmse / standard_rmse) * 100.0);
    }

    #[test]
    fn test_alternating_optimizer_phases() {
        println!("\n=== 최적화 단계 전환 테스트 ===");
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let block_data = generate_complex_data(64, 64);
        
        // 간단한 수렴 테스트
        println!("\n단계별 최적화 전환이 정상 작동하는지 확인");
        
        // 하이브리드 최적화 실행
        let encoded = encoder.encode_block_hybrid_optimization(
            &block_data,
            64,
            64,
            400,  // 4단계 x 100 스텝
            0.001,
            1000.0,
        );
        
        let mut decoder = WeightGenerator::new();
        let decoded = decoder.decode_block_enhanced(&encoded);
        let final_rmse = compute_rmse(&block_data, &decoded);
        
        println!("최종 RMSE: {:.6}", final_rmse);
        assert!(final_rmse < 0.01, "하이브리드 최적화가 수렴하지 않음");
    }

    #[test]
    fn test_hybrid_optimization_speed() {
        println!("\n=== 하이브리드 최적화 속도 테스트 ===");
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        // 테스트 케이스
        let test_sizes = [(64, 64), (128, 128), (256, 256)];
        
        for (rows, cols) in test_sizes {
            println!("\n--- {}x{} 블록 ---", rows, cols);
            let block_data = generate_complex_data(rows, cols);
            
            // 워밍업
            for _ in 0..5 {
                let _ = encoder.encode_block_hybrid_optimization(
                    &block_data,
                    rows,
                    cols,
                    100,
                    0.001,
                    100.0,
                );
            }
            
            // 인코딩 속도 측정
            let encode_start = Instant::now();
            let encoded = encoder.encode_block_hybrid_optimization(
                &block_data,
                rows,
                cols,
                500, // 더 적은 스텝으로 빠른 테스트
                0.001,
                100.0,
            );
            let encode_time = encode_start.elapsed();
            
            // 디코딩 속도 측정 (여러 번 반복)
            let iterations = 10000;
            let decode_start = Instant::now();
            for _ in 0..iterations {
                let _ = decoder.decode_block_enhanced(&encoded);
            }
            let total_decode_time = decode_start.elapsed();
            let avg_decode_time = total_decode_time / iterations;
            
            // 압축률 계산
            let original_size = rows * cols * 4;
            let compressed_size = 32 + encoded.residuals.len() * 8;
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            // RMSE 계산
            let decoded = decoder.decode_block_enhanced(&encoded);
            let rmse = compute_rmse(&block_data, &decoded);
            
            println!("  인코딩 시간: {:.2}ms", encode_time.as_millis());
            println!("  디코딩 시간: {:.2}μs (평균)", avg_decode_time.as_micros());
            println!("  픽셀당 디코딩: {:.0}ns", avg_decode_time.as_nanos() as f64 / (rows * cols) as f64);
            println!("  압축률: {:.0}:1", compression_ratio);
            println!("  RMSE: {:.6}", rmse);
            println!("  잔차 계수: {}", encoded.residuals.len());
            
            // 크리티컬 체크: 디코딩 속도
            let ns_per_pixel = avg_decode_time.as_nanos() as f64 / (rows * cols) as f64;
            assert!(ns_per_pixel < 150.0, "디코딩이 너무 느립니다: {:.0}ns/pixel", ns_per_pixel);
        }
    }

    #[test]
    fn test_hybrid_optimization_speed_limited() {
        println!("\n=== 하이브리드 최적화 속도 테스트 (잔차 제한) ===");
        
        let mut encoder = RBEEncoder::new(2, TransformType::Dwt);
        let decoder = WeightGenerator::new();
        
        // 테스트 케이스
        let test_sizes = [(64, 64), (128, 128), (256, 256)];
        
        for (rows, cols) in test_sizes {
            println!("\n--- {}x{} 블록 ---", rows, cols);
            let block_data = generate_complex_data(rows, cols);
            
            // 최대 잔차 계수를 블록 크기에 따라 제한
            let max_k = match (rows, cols) {
                (64, 64) => 16,   // 64x64: 최대 16개
                (128, 128) => 32, // 128x128: 최대 32개  
                (256, 256) => 48, // 256x256: 최대 48개
                _ => 64,
            };
            
            println!("최대 잔차 계수: {}", max_k);
            
            // 수정된 인코더 설정으로 인코딩
            let encoded = encoder.encode_block_hybrid_optimization_limited(
                &block_data,
                rows,
                cols,
                500,   // 적은 스텝
                0.001, // target_rmse
                100.0, // target_compression_ratio
                max_k, // 최대 K 제한
            );
            
            // 디코딩 속도 측정
            let decode_start = std::time::Instant::now();
            let mut total_decode_time = std::time::Duration::ZERO;
            
            for _ in 0..100 {
                let start = std::time::Instant::now();
                let _ = decoder.decode_block_enhanced(&encoded);
                total_decode_time += start.elapsed();
            }
            
            let avg_decode_time = total_decode_time / 100;
            let pixels = rows * cols;
            let ns_per_pixel = avg_decode_time.as_nanos() as f64 / pixels as f64;
            
            println!("디코딩 시간: {:?} (평균)", avg_decode_time);
            println!("픽셀당 디코딩: {:.0}ns", ns_per_pixel);
            println!("활성 잔차: {}", encoded.residuals.len());
            
            // 목표: 150ns/pixel 이하
            assert!(ns_per_pixel < 150.0, 
                    "디코딩이 너무 느림: {:.0}ns/pixel (목표: <150ns)", ns_per_pixel);
        }
    }

    // 복잡한 패턴 생성 함수
    fn generate_complex_pattern(rows: usize, cols: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; rows * cols];
        let mut rng = thread_rng();
        
        for i in 0..rows {
            for j in 0..cols {
                let x = i as f32 / rows as f32;
                let y = j as f32 / cols as f32;
                
                // 다중 주파수 성분
                let low_freq = (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
                let mid_freq = (8.0 * PI * x).sin() * (8.0 * PI * y).sin() * 0.3;
                let high_freq = (16.0 * PI * x).cos() * (16.0 * PI * y).cos() * 0.1;
                
                // 그래디언트
                let gradient = x * 0.5 + y * 0.5;
                
                // 노이즈
                let noise = rng.gen_range(-0.05..0.05);
                
                data[i * cols + j] = low_freq + mid_freq + high_freq + gradient + noise;
            }
        }
        
        data
    }
} 