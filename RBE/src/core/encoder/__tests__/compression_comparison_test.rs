//! 압축 방법 종합 비교 테스트

#[cfg(test)]
mod tests {
    use crate::encoder::encoder::{
        RBEEncoder, CompressionConfig, QualityGrade, CompressionProfile,
    };
    use crate::core::encoder::{
        MetricTensorEncoder, MetricTensorDecoder,
        SvdEncoder, SvdDecoder,
    };
    use crate::packed_params::TransformType;
    use std::time::Instant;

    fn generate_test_weights(size: usize) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        (0..size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect()
    }

    fn calculate_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
        let mse: f32 = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        mse.sqrt()
    }

    #[test]
    fn test_종합_압축_방법_비교() {
        println!("\n🔬 압축 방법 종합 비교\n");
        println!("{:-^80}", " Compression Methods Comparison ");
        
        let sizes = [(64, 64), (128, 128), (256, 256)];
        let ranks = [4, 8, 12, 15];
        
        for (rows, cols) in sizes {
            println!("\n📊 Matrix Size: {}x{}", rows, cols);
            println!("{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}", 
                "Method", "Rank", "RMSE", "Compression", "Encode(ms)", "Decode(ms)");
            println!("{}", "-".repeat(85));
            
            let weights = generate_test_weights(rows * cols);
            let original_size = weights.len() * 4;
            
            // 1. RBE 방식
            for block_size in [8, 16] {
                let config = CompressionConfig {
                    block_size,
                    quality_grade: QualityGrade::B,
                    transform_type: TransformType::Dct,
                    profile: CompressionProfile::Balanced,
                    custom_coefficients: None,
                    min_block_count: None,
                    rmse_threshold: None,
                    compression_ratio_threshold: None,
                    enable_residual: false,
                    residual_threshold: 0.0,
                };
                
                let mut encoder = RBEEncoder::new(10, TransformType::Dct); // k=10 계수
                
                let encode_start = Instant::now();
                let block = encoder.encode_block(&weights, rows, cols);
                let encode_time = encode_start.elapsed();
                
                // 블록은 단일 HybridEncodedBlock
                let compressed_size = 8 * 4 + 10 * 6; // 대략적인 크기
                
                // 디코딩
                let decode_start = Instant::now();
                let decoded = block.decode();
                let decode_time = decode_start.elapsed();
                
                // RMSE
                let block_size_actual = block_size.min(rows).min(cols);
                let block_original = &weights[..(block_size_actual * block_size_actual)];
                let rmse = calculate_rmse(block_original, &decoded);
                
                println!("{:<15} {:<10} {:<15.6} {:<15.1} {:<15.3} {:<15.3}",
                    format!("RBE({}x{})", block_size, block_size),
                    "-",
                    rmse,
                    format!("{:.1}:1", original_size as f32 / compressed_size as f32),
                    encode_time.as_secs_f64() * 1000.0,
                    decode_time.as_secs_f64() * 1000.0
                );
            }
            
            // 2. SVD 방식
            for &rank in &ranks {
                let encoder = SvdEncoder::new(rank);
                let decoder = SvdDecoder::new();
                
                let encode_start = Instant::now();
                let block = match encoder.encode(&weights, rows, cols) {
                    Ok(b) => b,
                    Err(e) => {
                        println!("SVD encoding error: {}", e);
                        continue;
                    }
                };
                let serialized = match encoder.serialize(&block) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("SVD serialization error: {}", e);
                        continue;
                    }
                };
                let encode_time = encode_start.elapsed();
                
                let decode_start = Instant::now();
                let deserialized = match decoder.deserialize(&serialized) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("SVD deserialization error: {}", e);
                        continue;
                    }
                };
                let decoded = match decoder.decode(&deserialized) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("SVD decoding error: {}", e);
                        continue;
                    }
                };
                let decode_time = decode_start.elapsed();
                
                let rmse = calculate_rmse(&weights, &decoded);
                let compression_ratio = original_size as f32 / serialized.len() as f32;
                
                println!("{:<15} {:<10} {:<15.6} {:<15.1} {:<15.3} {:<15.3}",
                    "SVD",
                    rank,
                    rmse,
                    format!("{:.1}:1", compression_ratio),
                    encode_time.as_secs_f64() * 1000.0,
                    decode_time.as_secs_f64() * 1000.0
                );
            }
            
            // 3. 메트릭 텐서 (참고용)
            for &rank in &[4, 8] {
                let metric_encoder = MetricTensorEncoder::new(rank);
                let metric_decoder = MetricTensorDecoder::new();
                
                let encode_start = Instant::now();
                let metric_block = match metric_encoder.encode_from_weights(&weights, rows, cols) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let serialized = match metric_encoder.serialize(&metric_block) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let encode_time = encode_start.elapsed();
                
                // 메트릭 텐서는 W 복원 불가, 압축률만 표시
                let compression_ratio = original_size as f32 / serialized.len() as f32;
                
                println!("{:<15} {:<10} {:<15} {:<15.1} {:<15.3} {:<15}",
                    "Metric Tensor",
                    rank,
                    "N/A",
                    format!("{:.1}:1", compression_ratio),
                    encode_time.as_secs_f64() * 1000.0,
                    "N/A"
                );
            }
        }
        
        println!("\n{:-^80}", " Summary ");
        println!("✅ SVD: 낮은 RMSE, 직접 복원 가능");
        println!("✅ RBE: 빠른 속도, 블록 단위 처리");
        println!("✅ Metric Tensor: 최적화 전용, 복원 불가");
    }
    
    #[test]
    fn test_svd_rank_vs_rmse() {
        println!("\n📈 SVD Rank vs RMSE Analysis\n");
        
        let sizes = [(64, 64), (128, 128)];
        let ranks = vec![1, 2, 4, 8, 12, 16, 24, 32, 48, 64];
        
        for (rows, cols) in sizes {
            println!("\nMatrix Size: {}x{}", rows, cols);
            println!("{:<10} {:<15} {:<15} {:<20}", "Rank", "RMSE", "Compression", "Bits per Weight");
            println!("{}", "-".repeat(60));
            
            let weights = generate_test_weights(rows * cols);
            let original_bits = weights.len() * 32;
            
            for &rank in &ranks {
                if rank > rows.min(cols) {
                    continue;
                }
                
                let encoder = SvdEncoder::new(rank);
                let decoder = SvdDecoder::new();
                
                match encoder.encode(&weights, rows, cols) {
                    Ok(block) => {
                        match encoder.serialize(&block) {
                            Ok(serialized) => {
                                match decoder.deserialize(&serialized) {
                                    Ok(deserialized) => {
                                        match decoder.decode(&deserialized) {
                                            Ok(decoded) => {
                                                let rmse = calculate_rmse(&weights, &decoded);
                                                let compressed_bits = serialized.len() * 8;
                                                let bits_per_weight = compressed_bits as f32 / weights.len() as f32;
                                                let compression_ratio = original_bits as f32 / compressed_bits as f32;
                                                
                                                println!("{:<10} {:<15.6} {:<15.1} {:<20.2}",
                                                    rank,
                                                    rmse,
                                                    format!("{:.1}:1", compression_ratio),
                                                    bits_per_weight
                                                );
                                                
                                                // RMSE < 0.001 달성 체크
                                                if rmse < 0.001 {
                                                    println!("  ⭐ RMSE < 0.001 achieved!");
                                                }
                                            }
                                            Err(e) => println!("  Decode error: {}", e),
                                        }
                                    }
                                    Err(e) => println!("  Deserialize error: {}", e),
                                }
                            }
                            Err(e) => println!("  Serialize error: {}", e),
                        }
                    }
                    Err(e) => println!("  Encode error: {}", e),
                }
            }
        }
    }
} 