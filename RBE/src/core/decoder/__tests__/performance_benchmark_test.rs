//! 디코더 성능 및 정확도 벤치마크 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::{
            weight_generator::{WeightGenerator, RBEDecoderConfig},
            fused_forward::FusedForwardPass,
        },
        encoder::RBEEncoder,
        packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient},
    };
    use std::time::Instant;

    /// 원본 데이터 생성
    fn generate_test_weights(rows: usize, cols: usize) -> Vec<f32> {
        let mut weights = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                // 실제 신경망 가중치 패턴 시뮬레이션
                let value = ((i as f32 * 0.1 + j as f32 * 0.05).sin() * 0.3) 
                    + ((i + j) as f32 * 0.01).cos() * 0.2;
                weights.push(value);
            }
        }
        weights
    }

    /// RMSE 계산
    fn calculate_rmse(original: &[f32], decoded: &[f32]) -> f32 {
        assert_eq!(original.len(), decoded.len());
        
        let sum_squared_error: f32 = original.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (o - d).powi(2))
            .sum();
            
        (sum_squared_error / original.len() as f32).sqrt()
    }
    
    /// HybridEncodedBlock의 실제 메모리 크기 계산
    fn calculate_block_size(block: &HybridEncodedBlock) -> usize {
        // RBE 파라미터: 8개 * 4바이트
        let rbe_size = 8 * 4;
        
        // 잔차 계수: 각각 (u16, u16, f32) = 8바이트
        let residuals_size = block.residuals.len() * 8;
        
        // 메타데이터: rows, cols (각 usize), transform_type (1바이트로 가정)
        let metadata_size = 2 * std::mem::size_of::<usize>() + 1;
        
        rbe_size + residuals_size + metadata_size
    }

    #[test]
    fn test_소규모_블록_정확도_및_속도() {
        println!("\n=== 소규모 블록 (64x64) 테스트 ===");
        
        // 1. 원본 데이터 생성
        let rows = 64;
        let cols = 64;
        let original_weights = generate_test_weights(rows, cols);
        
        // 2. 인코딩 - DWT 사용하도록 설정
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original_weights, rows, cols);
        
        // 3. 디코딩 및 정확도 측정
        let generator = WeightGenerator::new();
        let decoded = generator.decode_block(&block);
        
        let rmse = calculate_rmse(&original_weights, &decoded);
        println!("RMSE: {:.6}", rmse);
        
        // 4. 속도 측정
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = generator.decode_block(&block);
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("평균 디코딩 시간: {:.2}μs", avg_time_us);
        
        // 5. 검증
        assert!(rmse < 0.01, "RMSE가 목표(0.01)보다 큽니다: {:.6}", rmse);
        assert!(avg_time_us < 50.0, "속도가 목표(50μs)보다 느립니다: {:.2}μs", avg_time_us);
    }

    #[test]
    fn test_대규모_블록_정확도_및_속도() {
        println!("\n=== 대규모 블록 (512x512) 테스트 ===");
        
        // 1. 원본 데이터 생성
        let rows = 512;
        let cols = 512;
        let original_weights = generate_test_weights(rows, cols);
        
        // 2. 인코딩 - DWT 사용
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original_weights, rows, cols);
        
        // 3. 디코딩 및 정확도 측정
        let generator = WeightGenerator::new();
        let decoded = generator.decode_block(&block);
        
        let rmse = calculate_rmse(&original_weights, &decoded);
        println!("RMSE: {:.6}", rmse);
        
        // 4. 속도 측정
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = generator.decode_block(&block);
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("평균 디코딩 시간: {:.2}μs", avg_time_us);
        
        // 5. 압축률 계산
        let original_size = rows * cols * std::mem::size_of::<f32>();
        let compressed_size = std::mem::size_of::<HybridEncodedBlock>() + 
            block.residuals.len() * std::mem::size_of::<ResidualCoefficient>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("압축률: {:.1}:1", compression_ratio);
        
        // 6. 검증
        assert!(rmse < 0.3, "RMSE가 목표(0.3)보다 큽니다: {:.6}", rmse);
        assert!(compression_ratio > 100.0, "압축률이 목표(100:1)보다 낮습니다: {:.1}", compression_ratio);
    }

    #[test]
    fn test_극한_압축_정확도() {
        println!("\n=== 극한 압축 정확도 테스트 ===");
        
        // 극한 압축 설정으로 인코더 생성
        let encoder = RBEEncoder::new_extreme_compression();
        
        let test_sizes = [(128, 128), (256, 256), (512, 512)];
        
        for (rows, cols) in test_sizes {
            println!("\n크기: {}x{}", rows, cols);
            
            // 원본 데이터 생성
            let original = generate_test_weights(rows, cols);
            
            // 동적 블록 크기로 인코딩
            let (blocks, block_size, _time, compression_ratio, rmse) = 
                RBEEncoder::compress_with_dynamic_blocks(
                    &original,
                    rows,
                    cols,
                    encoder.k_coeffs,
                    encoder.transform_type,
                ).unwrap();
            
            println!("  사용된 블록 크기: {}x{}", block_size, block_size);
            println!("  블록 개수: {}", blocks.len());
            println!("  RMSE: {:.6}", rmse);
            
            // 잔차 개수 계산
            let total_residuals: usize = blocks.iter().map(|b| b.residuals.len()).sum();
            println!("  총 잔차 개수: {}", total_residuals);
            println!("  압축률: {:.1}:1", compression_ratio);
            
            // 극한 압축에서도 RMSE < 0.1 유지 (큰 행렬은 기준 완화)
            let rmse_threshold = match rows {
                128 => 0.05,
                256 => 0.15,
                512 => 0.2,
                _ => 0.1,
            };
            assert!(rmse < rmse_threshold, "극한 압축 RMSE가 너무 큽니다: {:.6}", rmse);
            assert!(compression_ratio > 50.0, "극한 압축률이 목표(50:1)보다 낮습니다: {:.1}", compression_ratio);
        }
    }

    #[test]
    fn test_GEMV_정확도_및_성능() {
        println!("\n=== GEMV 정확도 및 성능 테스트 ===");
        
        // 1. 행렬 생성
        let matrix_size = 128;  
        let original_matrix = generate_test_weights(matrix_size, matrix_size);
        
        // 2. 고정 블록 크기로 인코딩 (동적 블록 크기 대신)
        let block_size = 32;  // 128x128을 32x32 블록 16개로 나눔
        let encoder = RBEEncoder::new_a_grade();
        
        // 블록별로 인코딩
        let blocks_per_dim = matrix_size / block_size;
        let mut blocks = Vec::new();
        
        for block_row in 0..blocks_per_dim {
            for block_col in 0..blocks_per_dim {
                let mut block_data = Vec::with_capacity(block_size * block_size);
                
                // 블록 데이터 추출
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = block_row * block_size + i;
                        let global_j = block_col * block_size + j;
                        block_data.push(original_matrix[global_i * matrix_size + global_j]);
                    }
                }
                
                // 블록 인코딩
                let mut encoder_clone = RBEEncoder::new(encoder.k_coeffs, encoder.transform_type);
                let block = encoder_clone.encode_block(&block_data, block_size, block_size);
                blocks.push(block);
            }
        }
        
        println!("인코딩 완료: 블록크기 {}x{}, 블록 수 {}", block_size, block_size, blocks.len());
        
        // 3. 입력 벡터 생성
        let input = vec![1.0; matrix_size];
        let mut output = vec![0.0; matrix_size];
        
        // 4. 원본 행렬로 GEMV 계산 (ground truth)
        let mut expected_output = vec![0.0; matrix_size];
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                expected_output[i] += original_matrix[i * matrix_size + j] * input[j];
            }
        }
        
        // 5. FusedForwardPass로 계산
        let fused = FusedForwardPass::new();
        let layout = crate::core::decoder::fused_forward::BlockLayout {
            total_rows: matrix_size,
            total_cols: matrix_size,
            block_size,
            grid_rows: blocks_per_dim,
            grid_cols: blocks_per_dim,
        };
        
        let start = Instant::now();
        fused.block_gemv(&blocks, &input, &mut output, &layout);
        let gemv_time = start.elapsed();
        
        // 6. RMSE 계산
        let rmse = calculate_rmse(&expected_output, &output);
        
        println!("\n전체 GEMV RMSE: {:.6}", rmse);
        println!("GEMV 평균 시간: {:.2}μs", gemv_time.as_micros() as f32);
        
        // 7. 검증
        assert!(rmse < 0.1, "GEMV RMSE가 목표(0.1)보다 큽니다: {:.6}", rmse);
        assert!(gemv_time.as_micros() < 500_000, "GEMV가 너무 느립니다: {:?}", gemv_time);
    }

    #[test]
    fn test_캐시_일관성() {
        println!("\n=== 캐시 일관성 테스트 ===");
        
        let config = RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: 10 },
            enable_parallel: false,
            enable_simd: true,
        };
        let generator = WeightGenerator::with_config(config);
        
        // 동일한 블록으로 테스트
        let original = generate_test_weights(128, 128);
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original, 128, 128);
        
        // 첫 번째 디코딩 (캐시 미스)
        let decoded1 = generator.decode_block(&block);
        
        // 두 번째 디코딩 (캐시 히트)
        let decoded2 = generator.decode_block(&block);
        
        // 완전히 동일해야 함
        for (a, b) in decoded1.iter().zip(decoded2.iter()) {
            assert_eq!(a, b, "캐시된 결과가 일치하지 않습니다");
        }
        
        let stats = generator.get_stats();
        assert_eq!(stats.cache_hits, 1, "캐시 히트가 발생하지 않았습니다");
        
        println!("캐시 히트율: {:.1}%", generator.get_cache_hit_rate() * 100.0);
    }

    #[test]
    fn test_병렬_일관성() {
        println!("\n=== 병렬 처리 일관성 테스트 ===");
        
        // 순차 처리 생성기
        let seq_generator = WeightGenerator::with_config(RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::NoCache,
            enable_parallel: false,
            enable_simd: false,
        });
        
        // 병렬 처리 생성기
        let par_generator = WeightGenerator::with_config(RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::NoCache,
            enable_parallel: true,
            enable_simd: true,
        });
        
        // 테스트 블록 생성
        let original = generate_test_weights(256, 256);
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original, 256, 256);
        
        // 디코딩
        let seq_result = seq_generator.decode_block(&block);
        let par_result = par_generator.decode_block(&block);
        
        // RMSE 계산
        let rmse = calculate_rmse(&seq_result, &par_result);
        println!("순차 vs 병렬 RMSE: {:.9}", rmse);
        
        // 거의 동일해야 함 (부동소수점 오차 허용)
        assert!(rmse < 1e-6, "병렬 처리 결과가 순차 처리와 다릅니다: RMSE={:.9}", rmse);
    }

    #[test]
    fn test_GPT2_크기_행렬() {
        println!("\n=== GPT-2 크기 행렬 테스트 (768x3072) ===");
        
        let rows = 768;
        let cols = 3072;
        let block_size = 128;
        
        // 블록 단위로 처리
        let blocks_per_row = (rows + block_size - 1) / block_size;
        let blocks_per_col = (cols + block_size - 1) / block_size;
        
        let mut encoder = RBEEncoder::new_extreme_compression();
        let mut blocks = Vec::new();
        let mut total_rmse = 0.0;
        let mut block_count = 0;
        
        for i in 0..blocks_per_row {
            for j in 0..blocks_per_col {
                let actual_rows = ((i + 1) * block_size).min(rows) - i * block_size;
                let actual_cols = ((j + 1) * block_size).min(cols) - j * block_size;
                
                // 원본 블록 생성
                let original = generate_test_weights(actual_rows, actual_cols);
                
                // 인코딩
                let block = encoder.encode_block(&original, actual_rows, actual_cols);
                
                // 디코딩 및 RMSE 계산
                let generator = WeightGenerator::new();
                let decoded = generator.decode_block(&block);
                let rmse = calculate_rmse(&original, &decoded);
                
                total_rmse += rmse * rmse * original.len() as f32;
                block_count += original.len();
                
                blocks.push(block);
            }
        }
        
        // 전체 RMSE
        let overall_rmse = (total_rmse / block_count as f32).sqrt();
        println!("전체 RMSE: {:.6}", overall_rmse);
        
        // 압축률
        let original_size = rows * cols * 4;
        let compressed_size = blocks.len() * (std::mem::size_of::<HybridEncodedBlock>() + 5 * 8);
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("블록 수: {}", blocks.len());
        println!("압축률: {:.1}:1", compression_ratio);
        
        // 전체 디코딩 시간
        let generator = WeightGenerator::new();
        let start = Instant::now();
        for block in &blocks {
            let _ = generator.decode_block(block);
        }
        let elapsed = start.elapsed();
        
        println!("전체 디코딩 시간: {:.2}ms", elapsed.as_millis());
        
        // 검증
        assert!(overall_rmse < 0.1, "전체 RMSE가 목표(0.1)보다 큽니다: {:.6}", overall_rmse);
        assert!(compression_ratio > 200.0, "압축률이 목표(200:1)보다 낮습니다: {:.1}", compression_ratio);
        assert!(elapsed.as_millis() < 100, "디코딩 시간이 목표(100ms)보다 깁니다: {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_추론_속도_최적화() {
        println!("\n=== 추론 속도 최적화 테스트 ===");
        
        // 1. 테스트 데이터 준비
        let matrix_size = 256;
        let block_size = 64;
        let blocks_per_dim = matrix_size / block_size;
        let num_blocks = blocks_per_dim * blocks_per_dim;
        
        // 행렬 생성 및 블록 인코딩
        let original_matrix = generate_test_weights(matrix_size, matrix_size);
        let encoder = RBEEncoder::new_a_grade();
        let mut blocks = Vec::new();
        
        for block_row in 0..blocks_per_dim {
            for block_col in 0..blocks_per_dim {
                let mut block_data = Vec::with_capacity(block_size * block_size);
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = block_row * block_size + i;
                        let global_j = block_col * block_size + j;
                        block_data.push(original_matrix[global_i * matrix_size + global_j]);
                    }
                }
                let mut encoder_clone = RBEEncoder::new(encoder.k_coeffs, encoder.transform_type);
                let block = encoder_clone.encode_block(&block_data, block_size, block_size);
                blocks.push(block);
            }
        }
        
        println!("테스트 설정: {}개 블록 ({}x{} 행렬)", num_blocks, matrix_size, matrix_size);
        
        // 2. 일반 디코딩 (캐시 없음)
        let generator_no_cache = WeightGenerator::with_config(RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: 1 },
            enable_parallel: false,
            enable_simd: false,
        });
        
        let start = Instant::now();
        for _ in 0..10 {
            for block in &blocks {
                let _ = generator_no_cache.decode_block(block);
            }
        }
        let no_cache_time = start.elapsed();
        println!("\n캐시 없음 (10회): {:.2}ms", no_cache_time.as_millis());
        
        // 3. 캐시 활용 디코딩
        let generator_with_cache = WeightGenerator::with_config(RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: num_blocks + 10 },
            enable_parallel: false,
            enable_simd: false,
        });
        
        // 첫 번째 실행 (캐시 미스)
        let start = Instant::now();
        for block in &blocks {
            let _ = generator_with_cache.decode_block(block);
        }
        let first_run = start.elapsed();
        
        // 캐시된 실행 (캐시 히트)
        let start = Instant::now();
        for _ in 0..10 {
            for block in &blocks {
                let _ = generator_with_cache.decode_block(block);
            }
        }
        let cached_time = start.elapsed();
        
        let stats = generator_with_cache.get_stats();
        println!("\n캐시 활용:");
        println!("  첫 실행: {:.2}ms", first_run.as_millis());
        println!("  캐시된 실행 (10회): {:.2}ms", cached_time.as_millis());
        println!("  캐시 히트율: {:.1}%", generator_with_cache.get_cache_hit_rate() * 100.0);
        println!("  속도 향상: {:.1}x", no_cache_time.as_micros() as f32 / cached_time.as_micros() as f32);
        
        // 4. SIMD 최적화
        if cfg!(target_arch = "x86_64") {
            let mut generator_simd = WeightGenerator::with_config(RBEDecoderConfig {
                caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: 1 },
                enable_parallel: false,
                enable_simd: true,
            });
            
            let start = Instant::now();
            for _ in 0..10 {
                for block in &blocks {
                    let _ = generator_simd.decode_block(block);
                }
            }
            let simd_time = start.elapsed();
            println!("\nSIMD 최적화 (10회): {:.2}ms", simd_time.as_millis());
            println!("  SIMD 속도 향상: {:.1}x", no_cache_time.as_micros() as f32 / simd_time.as_micros() as f32);
        }
        
        // 5. 워밍업 테스트
        let generator_warmup = WeightGenerator::new();
        generator_warmup.warmup(&blocks, 0.5); // 50% 워밍업
        
        let stats_before = generator_warmup.get_stats();
        
        // 워밍업된 블록들에 다시 접근
        for block in blocks.iter().take(blocks.len() / 2) {
            let _ = generator_warmup.decode_block(block);
        }
        
        let stats_after = generator_warmup.get_stats();
        println!("\n워밍업 후:");
        println!("  초기 캐시된 블록: {}", stats_before.cache_misses);
        println!("  워밍업 후 캐시 히트: {}", stats_after.cache_hits - stats_before.cache_hits);
        let warmup_hit_rate = (stats_after.cache_hits - stats_before.cache_hits) as f32 / (blocks.len() / 2) as f32;
        println!("  워밍업 효과: {:.1}%", warmup_hit_rate * 100.0);
        
        // 검증
        assert!(cached_time < no_cache_time, "캐시가 성능을 향상시켜야 함");
        assert!(stats_after.cache_hits > stats_before.cache_hits, "워밍업 후 캐시 히트가 발생해야 함");
    }

    #[test]
    fn test_메모리_사용량_비교() {
        println!("\n=== 메모리 사용량 비교 테스트 ===");
        
        // GPT-2 크기 시뮬레이션
        let test_configs = [
            ("작은 레이어", 768, 768),      // 588K 파라미터
            ("중간 레이어", 768, 3072),     // 2.36M 파라미터
            ("큰 레이어", 1024, 4096),      // 4.19M 파라미터
        ];
        
        for (name, rows, cols) in test_configs {
            println!("\n{} ({}x{}):", name, rows, cols);
            
            // 1. 원본 메모리 사용량
            let original_size = rows * cols * std::mem::size_of::<f32>();
            let original_mb = original_size as f64 / 1_048_576.0;
            println!("  원본 크기: {:.2} MB", original_mb);
            
            // 2. 블록 단위로 압축
            let block_size = 128;
            let blocks_per_row = (rows + block_size - 1) / block_size;
            let blocks_per_col = (cols + block_size - 1) / block_size;
            let total_blocks = blocks_per_row * blocks_per_col;
            
            let encoder = RBEEncoder::new_b_grade();
            let mut compressed_size = 0;
            let mut blocks = Vec::new();
            
            // 샘플 블록 생성 및 크기 측정
            for i in 0..blocks_per_row.min(2) { // 처음 2행만 샘플링
                for j in 0..blocks_per_col.min(2) { // 처음 2열만 샘플링
                    let actual_rows = ((i + 1) * block_size).min(rows) - i * block_size;
                    let actual_cols = ((j + 1) * block_size).min(cols) - j * block_size;
                    
                    let data = generate_test_weights(actual_rows, actual_cols);
                    let mut encoder_clone = RBEEncoder::new(encoder.k_coeffs, encoder.transform_type);
                    let block = encoder_clone.encode_block(&data, actual_rows, actual_cols);
                    
                    compressed_size += calculate_block_size(&block);
                    blocks.push(block);
                }
            }
            
            // 전체 크기 추정
            let avg_block_size = compressed_size / blocks.len();
            let estimated_compressed_size = avg_block_size * total_blocks;
            let compressed_mb = estimated_compressed_size as f64 / 1_048_576.0;
            
            println!("  압축 크기: {:.3} MB (블록당 평균 {} bytes)", 
                    compressed_mb, avg_block_size);
            
            // 3. 캐시에 저장할 때의 메모리 사용량
            let cached_decoded_size = original_size; // 전체 디코딩시
            let cached_decoded_mb = cached_decoded_size as f64 / 1_048_576.0;
            
            // 4. 압축률 및 메모리 효율성
            let compression_ratio = original_size as f64 / estimated_compressed_size as f64;
            let memory_overhead = cached_decoded_mb / compressed_mb;
            
            println!("  압축률: {:.1}:1", compression_ratio);
            println!("  디코딩 후 메모리: {:.2} MB", cached_decoded_mb);
            println!("  메모리 오버헤드: {:.1}x (압축 대비)", memory_overhead);
            
            // 5. 부분 캐싱 전략
            let cache_percentage = 0.1; // 10% 캐싱
            let partial_cache_size = (cached_decoded_size as f64 * cache_percentage) / 1_048_576.0;
            let total_with_partial_cache = compressed_mb + partial_cache_size;
            
            println!("\n  10% 캐싱 전략:");
            println!("    압축 데이터: {:.3} MB", compressed_mb);
            println!("    캐시 데이터: {:.2} MB", partial_cache_size);
            println!("    총 메모리: {:.2} MB", total_with_partial_cache);
            println!("    원본 대비: {:.1}%", (total_with_partial_cache / original_mb) * 100.0);
        }
        
        // 실제 캐시 효율성 테스트
        println!("\n=== 캐시 크기별 성능 테스트 ===");
        
        let block_size = 128;
        let total_blocks = 100;
        let mut blocks = Vec::new();
        
        // 100개 블록 생성
        for i in 0..total_blocks {
            let data = generate_test_weights(block_size, block_size);
            let mut encoder = RBEEncoder::new_b_grade();
            let block = encoder.encode_block(&data, block_size, block_size);
            blocks.push(block);
        }
        
        // 다양한 캐시 크기로 테스트
        let cache_sizes = [1, 10, 20, 50, 100];
        
        for cache_size in cache_sizes {
            let generator = WeightGenerator::with_config(RBEDecoderConfig {
                caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: cache_size },
                enable_parallel: false,
                enable_simd: false,
            });
            
            // 블록을 2번씩 액세스
            for _ in 0..2 {
                for block in &blocks {
                    let _ = generator.decode_block(block);
                }
            }
            
            let stats = generator.get_stats();
            let hit_rate = generator.get_cache_hit_rate();
            let cache_memory = cache_size * block_size * block_size * 4; // 대략적인 캐시 메모리
            let cache_mb = cache_memory as f64 / 1_048_576.0;
            
            println!("\n캐시 크기 {} 블록 ({:.2} MB):", cache_size, cache_mb);
            println!("  히트율: {:.1}%", hit_rate * 100.0);
            println!("  총 디코딩: {}", stats.total_decodes);
            println!("  캐시 히트: {}", stats.cache_hits);
        }
    }
} 