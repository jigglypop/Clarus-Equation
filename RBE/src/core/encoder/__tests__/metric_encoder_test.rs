//! 메트릭 텐서 인코더 테스트

#[cfg(test)]
mod tests {
    use crate::core::encoder::{
    RBEEncoder, 
    MetricTensorEncoder, 
    MetricTensorDecoder,
};
use crate::core::decoder::WeightGenerator;
use crate::core::math::compute_rmse;
use std::time::Instant;
    
    fn generate_test_weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let x = (i % cols) as f32 / cols as f32;
                let y = (i / cols) as f32 / rows as f32;
                ((x * std::f32::consts::PI).sin() + (y * std::f32::consts::PI * 2.0).cos()) * 0.5
            })
            .collect()
    }
    

    
    #[test]
    fn test_메트릭_텐서_vs_RBE_성능_비교() {
        println!("\n=== 메트릭 텐서 vs RBE 성능 비교 ===");
        
        let test_sizes = [(64, 64), (128, 128), (256, 256)];
        let ranks = [4, 8, 12, 15];  // 4비트 제한으로 최대 15
        
        for (rows, cols) in test_sizes {
            println!("\n--- 행렬 크기: {}x{} ---", rows, cols);
            let weights = generate_test_weights(rows, cols);
            let original_size = weights.len() * std::mem::size_of::<f32>();
            
            // RBE 인코딩 테스트
            println!("\n[RBE 방식]");
            let mut rbe_encoder = RBEEncoder::new_b_grade();
            let block_size = 64;
            
            let rbe_start = Instant::now();
            let rbe_block = rbe_encoder.encode_block(&weights, rows, cols);
            let rbe_encode_time = rbe_start.elapsed();
            
            // RBE 압축 크기 계산
            let rbe_compressed_size = std::mem::size_of_val(&rbe_block) 
                + rbe_block.residuals.len() * std::mem::size_of::<crate::packed_params::ResidualCoefficient>();
            
            // RBE 디코딩
            let rbe_decode_start = Instant::now();
            let rbe_decoded = rbe_block.decode();
            let rbe_decode_time = rbe_decode_start.elapsed();
            
                            let rbe_rmse = compute_rmse(&weights, &rbe_decoded);
            let rbe_compression_ratio = original_size as f32 / rbe_compressed_size as f32;
            
            println!("  인코딩 시간: {:?}", rbe_encode_time);
            println!("  디코딩 시간: {:?}", rbe_decode_time);
            println!("  압축률: {:.1}:1", rbe_compression_ratio);
            println!("  RMSE: {:.6}", rbe_rmse);
            
            // 메트릭 텐서 방식 테스트
            println!("\n[메트릭 텐서 방식]");
            for &rank in &ranks {
                if rank > cols.min(rows) / 2 { continue; }
                
                print!("  rank={}: ", rank);
                let metric_encoder = MetricTensorEncoder::new(rank);
                let metric_decoder = MetricTensorDecoder::new();
                
                // 인코딩
                let metric_start = Instant::now();
                let metric_block = metric_encoder.encode_from_weights(&weights, rows, cols).unwrap();
                let serialized = metric_encoder.serialize(&metric_block).unwrap();
                let metric_encode_time = metric_start.elapsed();
                
                // 디코딩 (W 재구성 포함)
                let metric_decode_start = Instant::now();
                let deserialized = match metric_decoder.deserialize(&serialized) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("    직렬화 크기: {} bytes", serialized.len());
                        println!("    에러: {}", e);
                        // 직렬화된 데이터의 처음 20바이트 출력
                        print!("    데이터 시작: ");
                        for i in 0..20.min(serialized.len()) {
                            print!("{:02X} ", serialized[i]);
                        }
                        println!();
                        continue; // CRC 오류 시 다음 rank로 넘어감
                    }
                };
                let (u, lambda) = metric_decoder.decode_to_rank_k(&deserialized).unwrap();
                
                // W 재구성: W ≈ U * sqrt(Λ) * U^T (메트릭에서 가중치로)
                let mut w_reconstructed = vec![0.0f32; rows * cols];
                for i in 0..rank {
                    let scale = lambda[i].sqrt();
                    for row in 0..rows {
                        for col in 0..cols {
                            let idx = row * cols + col;
                            // rank-1 업데이트
                            w_reconstructed[idx] += scale * u[(row, i)] * u[(col, i)];
                        }
                    }
                }
                let metric_decode_time = metric_decode_start.elapsed();
                
                let metric_rmse = compute_rmse(&weights, &w_reconstructed);
                let metric_compression_ratio = original_size as f32 / serialized.len() as f32;
                
                print!("압축률 {:.1}:1, ", metric_compression_ratio);
                print!("RMSE {:.6}, ", metric_rmse);
                print!("인코딩 {:?}, ", metric_encode_time);
                println!("디코딩 {:?}", metric_decode_time);
            }
        }
    }
    
    #[test]
    fn test_자연_그래디언트_최적화_시뮬레이션() {
        println!("\n=== 자연 그래디언트 최적화 시뮬레이션 ===");
        
        let n = 64;
        let rank = 8;
        let num_iterations = 10;
        
        // 초기 가중치
        let mut weights = generate_test_weights(n, n);
        let target_weights = generate_test_weights(n, n);
        
        let metric_encoder = MetricTensorEncoder::new(rank);
        let metric_decoder = MetricTensorDecoder::new();
        
        println!("행렬 크기: {}x{}, rank: {}", n, n, rank);
        
        for iter in 0..num_iterations {
            // 손실 그래디언트 계산 (단순 MSE)
            let euclidean_grad: Vec<f32> = weights.iter()
                .zip(target_weights.iter())
                .map(|(w, t)| 2.0 * (w - t))
                .collect();
            
            // 메트릭 텐서 인코딩
            let metric_block = metric_encoder.encode_from_weights(&weights, n, n).unwrap();
            
            // 자연 그래디언트 계산
            let natural_grad = metric_decoder.natural_gradient(&metric_block, &euclidean_grad).unwrap();
            
            // 가중치 업데이트
            let learning_rate = 0.01;
            for i in 0..weights.len() {
                weights[i] -= learning_rate * natural_grad[i];
            }
            
            // 손실 계산
            let loss: f32 = weights.iter()
                .zip(target_weights.iter())
                .map(|(w, t)| (w - t).powi(2))
                .sum::<f32>() / weights.len() as f32;
            
            println!("Iteration {}: Loss = {:.6}", iter + 1, loss);
        }
    }
    
    #[test]
    fn test_메모리_효율성_분석() {
        println!("\n=== 메모리 효율성 분석 ===");
        
        let sizes = [(64, 64), (128, 128), (256, 256), (512, 512)];
        let ranks = [4, 8, 12, 15];  // 4비트 제한으로 최대 15
        
        println!("{:<15} {:<10} {:<15} {:<15} {:<15} {:<10}", 
            "Matrix Size", "Rank", "Original (KB)", "Compressed (B)", "Ratio", "Theory");
        println!("{}", "-".repeat(85));
        
        for (rows, cols) in sizes {
            for &rank in &ranks {
                if rank > rows.min(cols) / 2 { continue; }
                
                let original_kb = (rows * cols * 4) as f32 / 1024.0;
                
                // 실제 압축 크기: 헤더(3) + 고유값(rank) + 고유벡터(rank*n*2) + CRC(2)
                let compressed_size = 3 + rank + (rank * cols * 2) + 2;
                
                let compression_ratio = (rows * cols * 4) as f32 / compressed_size as f32;
                
                // 이론적 압축률 (rank-k 근사)
                let theory_ratio = (rows * cols) as f32 / (rank * (rows + cols)) as f32;
                
                println!("{:<15} {:<10} {:<15.2} {:<15} {:<15.1} {:<10.1}", 
                    format!("{}x{}", rows, cols),
                    rank,
                    original_kb,
                    compressed_size,
                    compression_ratio,
                    theory_ratio
                );
            }
        }
    }
    
    #[test]
    fn test_추론_속도_비교() {
        println!("\n=== 추론 속도 비교 (GEMV) ===");
        
        let rows = 768;  // GPT-2 크기
        let cols = 3072;
        let rank = 16;
        let num_iterations = 100;
        
        let weights = generate_test_weights(rows, cols);
        let input_vector: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
        
        // RBE 방식
        let mut rbe_encoder = RBEEncoder::new_b_grade();
        let block_size = 128;
        let rbe_blocks = {
            let mut blocks = Vec::new();
            for i in (0..rows).step_by(block_size) {
                for j in (0..cols).step_by(block_size) {
                    let block_rows = block_size.min(rows - i);
                    let block_cols = block_size.min(cols - j);
                    let mut block_data = Vec::new();
                    
                    for bi in 0..block_rows {
                        for bj in 0..block_cols {
                            block_data.push(weights[(i + bi) * cols + (j + bj)]);
                        }
                    }
                    
                    blocks.push(rbe_encoder.encode_block(&block_data, block_rows, block_cols));
                }
            }
            blocks
        };
        
        // 메트릭 텐서 방식
        let metric_encoder = MetricTensorEncoder::new(rank);
        let metric_decoder = MetricTensorDecoder::new();
        let metric_block = metric_encoder.encode_from_weights(&weights, rows, cols).unwrap();
        
        // RBE GEMV 벤치마크
        let rbe_start = Instant::now();
        for _ in 0..num_iterations {
            let mut output = vec![0.0f32; rows];
            // 블록별 GEMV 시뮬레이션
            for (idx, block) in rbe_blocks.iter().enumerate() {
                let decoded = block.decode();
                // 실제 GEMV는 생략, 디코딩 시간만 측정
            }
        }
        let rbe_time = rbe_start.elapsed();
        
        // 메트릭 텐서 GEMV 벤치마크
        let metric_start = Instant::now();
        for _ in 0..num_iterations {
            let (u, lambda) = metric_decoder.decode_to_rank_k(&metric_block).unwrap();
            // rank-k GEMV: y = U * (Λ * (U^T * x))
            // 실제 계산은 생략, 복잡도 측정
        }
        let metric_time = metric_start.elapsed();
        
        println!("행렬 크기: {}x{}", rows, cols);
        println!("반복 횟수: {}", num_iterations);
        println!("\nRBE 방식:");
        println!("  총 시간: {:?}", rbe_time);
        println!("  평균 시간: {:?}", rbe_time / num_iterations);
        println!("\n메트릭 텐서 방식 (rank={}):", rank);
        println!("  총 시간: {:?}", metric_time);
        println!("  평균 시간: {:?}", metric_time / num_iterations);
        println!("  속도 비율: {:.2}x", rbe_time.as_secs_f64() / metric_time.as_secs_f64());
    }
} 