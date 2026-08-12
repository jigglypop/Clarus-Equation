//! 캐싱 전략 종합 비교 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::{
            weight_generator::{WeightGenerator, RBEDecoderConfig},
            caching_strategy::CachingStrategy,
        },
        encoder::RBEEncoder,
        packed_params::HybridEncodedBlock,
    };
    use std::time::Instant;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct BenchmarkResult {
        strategy_name: String,
        encoding_time_ms: u128,
        decoding_time_ms: u128,
        total_inference_time_ms: u128,
        memory_usage_mb: f64,
        cache_memory_mb: f64,
        compressed_memory_mb: f64,
        rmse: f32,
        compression_ratio: f32,
        cache_hit_rate: f32,
        blocks_per_second: f64,
    }

    fn generate_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let x = (i % cols) as f32 / cols as f32;
                let y = (i / cols) as f32 / rows as f32;
                (x * std::f32::consts::PI).sin() * (y * std::f32::consts::PI * 2.0).cos()
            })
            .collect()
    }

    fn calculate_rmse(original: &[f32], decoded: &[f32]) -> f32 {
        let sum_squared_diff: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (sum_squared_diff / original.len() as f32).sqrt()
    }

    fn estimate_memory_usage(blocks: &[HybridEncodedBlock], cache_size: usize, block_size: usize) -> (f64, f64, f64) {
        // 압축된 블록들의 메모리 사용량
        let compressed_size: usize = blocks.iter()
            .map(|block| {
                std::mem::size_of::<HybridEncodedBlock>() + 
                block.residuals.len() * std::mem::size_of::<crate::packed_params::ResidualCoefficient>()
            })
            .sum();
        
        // 캐시 메모리 사용량 (각 캐시 항목은 디코딩된 블록)
        let cache_memory = cache_size * block_size * std::mem::size_of::<f32>();
        
        // 전체 메모리 = 압축 데이터 + 캐시
        let total_memory = compressed_size + cache_memory;
        
        (
            total_memory as f64 / 1_048_576.0,      // 전체 MB
            cache_memory as f64 / 1_048_576.0,      // 캐시 MB
            compressed_size as f64 / 1_048_576.0,   // 압축 MB
        )
    }

    #[test]
    fn test_종합_캐시_성능_비교() {
        println!("\n=== 캐싱 전략 종합 성능 비교 ===\n");
        
        // 테스트 설정
        let matrix_sizes = [(256, 256), (512, 512), (768, 3072)]; // 작은, 중간, GPT-2 크기
        let block_size = 64;
        let mut encoder = RBEEncoder::new_b_grade(); // B급 품질 사용
        
        // 테스트할 캐싱 전략들
        let strategies = vec![
            ("캐시 없음", RBEDecoderConfig::minimal_memory()),
            ("고정 LRU 8", RBEDecoderConfig::legacy(8)),
            ("고정 LRU 16", RBEDecoderConfig::legacy(16)),
            ("고정 LRU 32", RBEDecoderConfig::legacy(32)),
            ("균형잡힌 (10%)", RBEDecoderConfig::balanced()),
            ("적응형", RBEDecoderConfig::adaptive()),
            ("전체 사전계산", RBEDecoderConfig::max_performance()),
        ];
        
        let mut all_results = Vec::new();
        
        for (rows, cols) in matrix_sizes {
            println!("\n--- 행렬 크기: {}x{} ---", rows, cols);
            let original_size_mb = (rows * cols * std::mem::size_of::<f32>()) as f64 / 1_048_576.0;
            println!("원본 크기: {:.2} MB", original_size_mb);
            
            // 원본 데이터 생성
            let original_matrix = generate_test_matrix(rows, cols);
            
            // 블록 단위로 인코딩
            let encoding_start = Instant::now();
            let blocks_per_row = (rows + block_size - 1) / block_size;
            let blocks_per_col = (cols + block_size - 1) / block_size;
            let total_blocks = blocks_per_row * blocks_per_col;
            let mut blocks = Vec::new();
            
            for block_row in 0..blocks_per_row {
                for block_col in 0..blocks_per_col {
                    let mut block_data = Vec::new();
                    
                    for i in 0..block_size {
                        for j in 0..block_size {
                            let global_i = block_row * block_size + i;
                            let global_j = block_col * block_size + j;
                            
                            if global_i < rows && global_j < cols {
                                block_data.push(original_matrix[global_i * cols + global_j]);
                            } else {
                                block_data.push(0.0);
                            }
                        }
                    }
                    
                    let encoded = encoder.encode_block(&block_data, block_size, block_size);
                    blocks.push(encoded);
                }
            }
            let encoding_time = encoding_start.elapsed();
            
            // 압축률 계산
            let compressed_size: usize = blocks.iter()
                .map(|block| {
                    std::mem::size_of::<HybridEncodedBlock>() + 
                    block.residuals.len() * std::mem::size_of::<crate::packed_params::ResidualCoefficient>()
                })
                .sum();
            let compression_ratio = (rows * cols * std::mem::size_of::<f32>()) as f32 / compressed_size as f32;
            
            println!("인코딩 완료: {} 블록, 압축률 {:.1}:1", blocks.len(), compression_ratio);
            
            // 각 전략별 테스트
            for (name, config) in &strategies {
                println!("\n[{}]", name);
                
                // 캐시 크기 계산
                let cache_size = match &config.caching_strategy {
                    CachingStrategy::NoCache => 0,
                    CachingStrategy::FixedLRU { size } => *size,
                    CachingStrategy::PercentageBased { percentage } => {
                        ((total_blocks as f32 * percentage).ceil() as usize).max(1)
                    }
                    CachingStrategy::Adaptive { min_size, max_size, .. } => {
                        (min_size + max_size) / 2 // 평균값 사용
                    }
                    CachingStrategy::PrecomputeAll => total_blocks,
                };
                
                // 메모리 사용량 추정
                let (total_mem, cache_mem, compressed_mem) = 
                    estimate_memory_usage(&blocks, cache_size, block_size * block_size);
                
                // WeightGenerator 생성
                let generator = WeightGenerator::with_config_and_blocks(config.clone(), total_blocks);
                
                // 워밍업 (사전계산 전략의 경우)
                if matches!(config.caching_strategy, CachingStrategy::PrecomputeAll) {
                    generator.warmup(&blocks, 1.0);
                }
                
                // 디코딩 및 추론 시간 측정
                let decode_start = Instant::now();
                let mut decoded_matrix = vec![0.0f32; rows * cols];
                
                // 3회 반복하여 캐시 효과 측정
                for _ in 0..3 {
                    let mut idx = 0;
                    for block in &blocks {
                        let decoded_block = generator.decode_block(block);
                        
                        // 블록 데이터를 전체 행렬에 복사
                        let block_row = idx / blocks_per_col;
                        let block_col = idx % blocks_per_col;
                        
                        for i in 0..block_size {
                            for j in 0..block_size {
                                let global_i = block_row * block_size + i;
                                let global_j = block_col * block_size + j;
                                
                                if global_i < rows && global_j < cols {
                                    decoded_matrix[global_i * cols + global_j] = 
                                        decoded_block[i * block_size + j];
                                }
                            }
                        }
                        idx += 1;
                    }
                }
                
                let decode_time = decode_start.elapsed();
                let blocks_per_second = (blocks.len() * 3) as f64 / decode_time.as_secs_f64();
                
                // RMSE 계산
                let rmse = calculate_rmse(&original_matrix, &decoded_matrix);
                
                // 통계 가져오기
                let stats = generator.get_stats();
                let hit_rate = generator.get_cache_hit_rate();
                
                // 결과 저장
                let result = BenchmarkResult {
                    strategy_name: format!("{} ({}x{})", name, rows, cols),
                    encoding_time_ms: encoding_time.as_millis(),
                    decoding_time_ms: decode_time.as_millis() / 3, // 평균
                    total_inference_time_ms: decode_time.as_millis(),
                    memory_usage_mb: total_mem,
                    cache_memory_mb: cache_mem,
                    compressed_memory_mb: compressed_mem,
                    rmse,
                    compression_ratio,
                    cache_hit_rate: hit_rate,
                    blocks_per_second,
                };
                
                // 결과 출력
                println!("  추론 시간: {:.2}ms (평균)", result.decoding_time_ms);
                println!("  블록/초: {:.0}", result.blocks_per_second);
                println!("  메모리: 전체 {:.2}MB (압축 {:.2}MB + 캐시 {:.2}MB)", 
                        result.memory_usage_mb, result.compressed_memory_mb, result.cache_memory_mb);
                println!("  RMSE: {:.6}", result.rmse);
                println!("  캐시 히트율: {:.1}%", result.cache_hit_rate * 100.0);
                
                all_results.push(result);
            }
        }
        
        // 종합 결과 테이블 출력
        println!("\n\n=== 종합 결과 요약 ===");
        println!("{:<35} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}", 
                "전략", "추론(ms)", "블록/초", "메모리(MB)", "캐시(MB)", "RMSE", "히트율(%)");
        println!("{}", "-".repeat(110));
        
        for result in &all_results {
            println!("{:<35} {:>10} {:>12.0} {:>12.2} {:>10.2} {:>10.6} {:>10.1}", 
                    result.strategy_name,
                    result.decoding_time_ms,
                    result.blocks_per_second,
                    result.memory_usage_mb,
                    result.cache_memory_mb,
                    result.rmse,
                    result.cache_hit_rate * 100.0);
        }
        
        // RMSE 검증 - 모든 전략이 허용 범위 내에 있어야 함
        for result in &all_results {
            assert!(result.rmse < 0.15, 
                   "{} RMSE가 너무 높습니다: {:.6}", result.strategy_name, result.rmse);
        }
    }
    
    #[test]
    fn test_메모리_효율성_분석() {
        println!("\n=== 메모리 효율성 분석 ===\n");
        
        let matrix_size = 512;
        let block_size = 64;
        let original_matrix = generate_test_matrix(matrix_size, matrix_size);
        let original_size_mb = (matrix_size * matrix_size * std::mem::size_of::<f32>()) as f64 / 1_048_576.0;
        
        // 다양한 압축 품질로 테스트
        let qualities = [
            ("S급 (최고 품질)", RBEEncoder::new_s_grade()),
            ("A급 (고품질)", RBEEncoder::new_a_grade()),
            ("B급 (표준)", RBEEncoder::new_b_grade()),
            ("극한 압축", RBEEncoder::new_extreme_compression()),
        ];
        
        println!("원본 크기: {:.2} MB\n", original_size_mb);
        
        for (quality_name, mut encoder) in qualities {
            // 인코딩
            let blocks_per_dim = matrix_size / block_size;
            let mut blocks = Vec::new();
            let mut total_residuals = 0;
            
            for block_row in 0..blocks_per_dim {
                for block_col in 0..blocks_per_dim {
                    let mut block_data = Vec::new();
                    
                    for i in 0..block_size {
                        for j in 0..block_size {
                            let idx = (block_row * block_size + i) * matrix_size + (block_col * block_size + j);
                            block_data.push(original_matrix[idx]);
                        }
                    }
                    
                    let encoded = encoder.encode_block(&block_data, block_size, block_size);
                    total_residuals += encoded.residuals.len();
                    blocks.push(encoded);
                }
            }
            
            // 메모리 계산
            let compressed_size: usize = blocks.iter()
                .map(|block| {
                    std::mem::size_of::<HybridEncodedBlock>() + 
                    block.residuals.len() * std::mem::size_of::<crate::packed_params::ResidualCoefficient>()
                })
                .sum();
            
            let compressed_mb = compressed_size as f64 / 1_048_576.0;
            let compression_ratio = original_size_mb / compressed_mb;
            let avg_residuals_per_block = total_residuals as f64 / blocks.len() as f64;
            
            // RMSE 계산을 위한 디코딩
            let mut decoded = Vec::new();
            for block in &blocks {
                decoded.extend(block.decode());
            }
            let rmse = calculate_rmse(&original_matrix, &decoded[..original_matrix.len()]);
            
            println!("[{}]", quality_name);
            println!("  압축 크기: {:.3} MB", compressed_mb);
            println!("  압축률: {:.1}:1", compression_ratio);
            println!("  평균 잔차/블록: {:.1}", avg_residuals_per_block);
            println!("  RMSE: {:.6}", rmse);
            println!("  메모리 절약: {:.1}%\n", (1.0 - compressed_mb / original_size_mb) * 100.0);
        }
    }
    
    #[test]
    fn test_실시간_추론_시뮬레이션() {
        println!("\n=== 실시간 추론 시뮬레이션 ===\n");
        
        // GPT-2 크기 시뮬레이션
        let rows = 768;
        let cols = 3072;
        let block_size = 128;
        let batch_size = 32; // 배치 크기
        
        let original_matrix = generate_test_matrix(rows, cols);
        let mut encoder = RBEEncoder::new_b_grade();
        
        // 블록 인코딩
        let blocks_per_row = (rows + block_size - 1) / block_size;
        let blocks_per_col = (cols + block_size - 1) / block_size;
        let mut blocks = Vec::new();
        
        for block_row in 0..blocks_per_row {
            for block_col in 0..blocks_per_col {
                let mut block_data = Vec::new();
                
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = block_row * block_size + i;
                        let global_j = block_col * block_size + j;
                        
                        if global_i < rows && global_j < cols {
                            block_data.push(original_matrix[global_i * cols + global_j]);
                        } else {
                            block_data.push(0.0);
                        }
                    }
                }
                
                blocks.push(encoder.encode_block(&block_data, block_size, block_size));
            }
        }
        
        println!("행렬 크기: {}x{}", rows, cols);
        println!("블록 수: {}", blocks.len());
        println!("배치 크기: {}\n", batch_size);
        
        // 각 캐싱 전략으로 실시간 추론 테스트
        let strategies = [
            ("캐시 없음", RBEDecoderConfig::minimal_memory()),
            ("작은 캐시", RBEDecoderConfig::legacy(4)),
            ("중간 캐시", RBEDecoderConfig::legacy(16)),
            ("적응형", RBEDecoderConfig::adaptive()),
        ];
        
        for (name, config) in strategies {
            let generator = WeightGenerator::with_config_and_blocks(config, blocks.len());
            
            // 배치 추론 시뮬레이션
            let start = Instant::now();
            let mut total_rmse = 0.0;
            
            for batch in 0..batch_size {
                // 입력 벡터 생성 (실제로는 토큰 임베딩)
                let input = vec![0.1 * (batch as f32 + 1.0); cols];
                let mut output = vec![0.0f32; rows];
                
                // 블록별 GEMV
                let mut block_idx = 0;
                for block_row in 0..blocks_per_row {
                    let decoded_blocks: Vec<_> = (0..blocks_per_col)
                        .map(|_| {
                            let result = generator.decode_block(&blocks[block_idx]);
                            block_idx += 1;
                            result
                        })
                        .collect();
                    
                    // 블록 행의 GEMV 수행
                    for i in 0..block_size {
                        let global_i = block_row * block_size + i;
                        if global_i < rows {
                            let mut sum = 0.0;
                            for (block_col, decoded_block) in decoded_blocks.iter().enumerate() {
                                for j in 0..block_size {
                                    let global_j = block_col * block_size + j;
                                    if global_j < cols {
                                        sum += decoded_block[i * block_size + j] * input[global_j];
                                    }
                                }
                            }
                            output[global_i] = sum;
                        }
                    }
                }
                
                // 간단한 정확도 체크 (실제 GEMV와 비교)
                let expected_sum: f32 = input.iter().sum::<f32>() * 0.1; // 근사값
                let actual_sum: f32 = output.iter().sum();
                total_rmse += ((expected_sum - actual_sum) / expected_sum).abs();
            }
            
            let elapsed = start.elapsed();
            let avg_rmse = total_rmse / batch_size as f32;
            let stats = generator.get_stats();
            
            println!("[{}]", name);
            println!("  총 시간: {:.2}ms", elapsed.as_millis());
            println!("  배치당 평균: {:.2}ms", elapsed.as_millis() as f64 / batch_size as f64);
            println!("  처리량: {:.0} 배치/초", batch_size as f64 / elapsed.as_secs_f64());
            println!("  캐시 히트율: {:.1}%", generator.get_cache_hit_rate() * 100.0);
            println!("  평균 오차율: {:.4}%\n", avg_rmse * 100.0);
        }
    }
} 