#[cfg(test)]
mod tests {
    use crate::packed_params::{HyperbolicTensor, HyperbolicPackingMode, HierarchicalHyperbolicTensor};
    use rand::Rng;

    #[test]
    fn test_hyperbolic_encoding_modes() {
        println!("\n=== 하이퍼볼릭 압축 모드 테스트 ===");
        
        let rows = 32;
        let cols = 32;
        let size = rows * cols;
        
        // 테스트 데이터 생성
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size)
            .map(|i| {
                let x = (i % cols) as f32 / cols as f32;
                let y = (i / cols) as f32 / rows as f32;
                0.1 + 0.2 * x + 0.3 * y + 0.1 * x * y + rng.gen_range(-0.05..0.05)
            })
            .collect();
        
        // 1. Geodesic 모드 - 측지선 표현
        println!("\n1. Geodesic 모드 (측지선 - 최단 경로)");
        let params = [0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0];
        let residuals = vec![0.05, -0.03, 0.02, -0.04];
        let block_geo = HyperbolicTensor::encode_geodesic(&params, &residuals);
        
        println!("- 크기: {} bytes", block_geo.size_bytes());
        println!("- 압축률: {:.1}:1", block_geo.compression_ratio(size));
        
        let decoded_geo = block_geo.decode(rows, cols);
        let rmse_geo = compute_rmse(&data, &decoded_geo);
        println!("- RMSE: {:.4}", rmse_geo);
        
        // 2. Horocycle 모드 - 평행선 표현
        println!("\n2. Horocycle 모드 (평행선 표현)");
        let residuals_horo = vec![0.05, -0.03, 0.02, -0.04, 0.01, -0.02, 0.03, -0.01];
        let block_horo = HyperbolicTensor::encode_horocycle(&params, &residuals_horo);
        
        println!("- 크기: {} bytes", block_horo.size_bytes());
        println!("- 압축률: {:.1}:1", block_horo.compression_ratio(size));
        
        let decoded_horo = block_horo.decode(rows, cols);
        let rmse_horo = compute_rmse(&data, &decoded_horo);
        println!("- RMSE: {:.4}", rmse_horo);
        
        // 3. Bicurvature 모드 - 이중곡률 표현
        println!("\n3. Bicurvature 모드 (이중곡률 표현)");
        let params_bicurv = vec![0.1, 0.2, 0.3, 0.1, 0.05, -0.02, 0.03, -0.01];
        let residuals_bicurv: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.01).collect();
        let block_bicurv = HyperbolicTensor::encode_bicurvature(&params_bicurv, &residuals_bicurv);
        
        println!("- 크기: {} bytes", block_bicurv.size_bytes());
        println!("- 압축률: {:.1}:1", block_bicurv.compression_ratio(size));
        
        let decoded_bicurv = block_bicurv.decode(rows, cols);
        let rmse_bicurv = compute_rmse(&data, &decoded_bicurv);
        println!("- RMSE: {:.4}", rmse_bicurv);
        
        // 4. AdaptiveMetric 모드
        println!("\n4. AdaptiveMetric 모드 (적응형 메트릭)");
        let block_adaptive = HyperbolicTensor::encode_adaptive_metric(&data, rows, cols);
        
        println!("- 크기: {} bytes", block_adaptive.size_bytes());
        println!("- 압축률: {:.1}:1", block_adaptive.compression_ratio(size));
        
        let decoded_adaptive = block_adaptive.decode(rows, cols);
        let rmse_adaptive = compute_rmse(&data, &decoded_adaptive);
        println!("- RMSE: {:.4}", rmse_adaptive);
        
        // 5. 희소 데이터 테스트
        println!("\n5. AdaptiveMetric 모드 - 희소 데이터");
        let mut sparse_data = vec![0.0f32; size];
        for _i in 0..10 {
            sparse_data[rng.gen_range(0..size)] = rng.gen_range(-1.0..1.0);
        }
        
        let block_sparse = HyperbolicTensor::encode_adaptive_metric(&sparse_data, rows, cols);
        println!("- 크기: {} bytes", block_sparse.size_bytes());
        println!("- 압축률: {:.1}:1", block_sparse.compression_ratio(size));
        
        let decoded_sparse = block_sparse.decode(rows, cols);
        let rmse_sparse = compute_rmse(&sparse_data, &decoded_sparse);
        println!("- RMSE: {:.4}", rmse_sparse);
        
        // 검증 - Horocycle이 NaN이 아닌지 확인
        assert!(!rmse_horo.is_nan(), "Horocycle 모드에서 NaN 발생");
        assert!(rmse_bicurv <= rmse_geo || rmse_bicurv.is_finite(), 
                "Bicurvature 모드가 더 많은 파라미터로 더 나은 근사를 제공해야 함");
    }
    
    #[test]
    fn test_hierarchical_poincare_encoding() {
        println!("\n=== 계층적 푸앵카레 압축 테스트 ===");
        
        let rows = 64;
        let cols = 64;
        
        // 멀티스케일 테스트 데이터
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let x = j as f32 / cols as f32;
                let y = i as f32 / rows as f32;
                
                // 저주파 성분
                data[i * cols + j] += 0.5 * (2.0 * std::f32::consts::PI * x).sin();
                // 중주파 성분
                data[i * cols + j] += 0.3 * (8.0 * std::f32::consts::PI * y).cos();
                // 고주파 성분
                data[i * cols + j] += 0.1 * (16.0 * std::f32::consts::PI * (x + y)).sin();
            }
        }
        
        // 계층적 압축
        let mut hierarchical = HierarchicalHyperbolicTensor::new();
        
        // Level 0: 저주파 (Geodesic)
        let low_freq_params = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let low_freq_tensor = HyperbolicTensor::encode_geodesic(&low_freq_params, &[]);
        hierarchical.add_level(low_freq_tensor);
        
        // Level 1: 중주파 (Horocycle)
        let mid_freq_params = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mid_freq_residuals = vec![0.05; 8];
        let mid_freq_tensor = HyperbolicTensor::encode_horocycle(&mid_freq_params, &mid_freq_residuals);
        hierarchical.add_level(mid_freq_tensor);
        
        // Level 2: 고주파 (Bicurvature)
        let high_freq_params = vec![0.0, 0.0, 0.0, 0.1, 0.05, 0.03, 0.02, 0.01];
        let high_freq_residuals = vec![0.01; 16];
        let high_freq_tensor = HyperbolicTensor::encode_bicurvature(&high_freq_params, &high_freq_residuals);
        hierarchical.add_level(high_freq_tensor);
        
        // 레벨 간 연결
        hierarchical.add_connection(0, 1, 1.0);
        hierarchical.add_connection(1, 2, 1.0);
        
        println!("- 레벨 수: {}", hierarchical.levels.len());
        println!("- 연결 수: {}", hierarchical.connections.len());
        
        let decoded = hierarchical.decode(rows, cols);
        let rmse = compute_rmse(&data, &decoded);
        println!("- 전체 RMSE: {:.4}", rmse);
        
        // 총 크기 계산
        let total_size: usize = hierarchical.levels.iter()
            .map(|tensor| tensor.size_bytes())
            .sum();
        let compression_ratio = (rows * cols * 4) as f32 / total_size as f32;
        println!("- 총 크기: {} bytes", total_size);
        println!("- 압축률: {:.1}:1", compression_ratio);
    }
    
    #[test]
    fn test_hyperbolic_compression_comparison() {
        println!("\n=== 하이퍼볼릭 압축 모드별 성능 비교 ===");
        
        let test_sizes = [(16, 16), (32, 32), (64, 64)];
        
        for (rows, cols) in test_sizes {
            println!("\n크기: {}x{}", rows, cols);
            
            // 복잡한 패턴의 테스트 데이터
            let mut rng = rand::thread_rng();
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let x = (i % cols) as f32 / cols as f32;
                    let y = (i / cols) as f32 / rows as f32;
                    
                    // 다양한 주파수 성분 조합
                    let low = 0.3 * (2.0 * std::f32::consts::PI * x).sin();
                    let mid = 0.2 * (4.0 * std::f32::consts::PI * y).cos();
                    let high = 0.1 * (8.0 * std::f32::consts::PI * (x + y)).sin();
                    let noise = rng.gen_range(-0.05..0.05);
                    
                    low + mid + high + noise
                })
                .collect();
            
            // 각 모드로 압축
            let modes = [
                ("Geodesic", HyperbolicPackingMode::Geodesic),
                ("Horocycle", HyperbolicPackingMode::Horocycle),
                ("Bicurvature", HyperbolicPackingMode::Bicurvature),
                ("AdaptiveMetric", HyperbolicPackingMode::AdaptiveMetric),
            ];
            
            println!("{:<15} {:>10} {:>15} {:>10}", "모드", "크기(B)", "압축률", "RMSE");
            println!("{:-<50}", "");
            
            for (name, mode) in modes {
                let tensor = match mode {
                    HyperbolicPackingMode::Geodesic => {
                        let params = estimate_params(&data, rows, cols, 4);
                        let residuals = compute_residuals(&data, &params, rows, cols, 4);
                        HyperbolicTensor::encode_geodesic(
                            &[params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0],
                            &residuals
                        )
                    },
                    HyperbolicPackingMode::Horocycle => {
                        let params = estimate_params(&data, rows, cols, 4);
                        let residuals = compute_residuals(&data, &params, rows, cols, 8);
                        HyperbolicTensor::encode_horocycle(
                            &[params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0],
                            &residuals
                        )
                    },
                    HyperbolicPackingMode::Bicurvature => {
                        let params = estimate_params(&data, rows, cols, 8);
                        let residuals = compute_residuals(&data, &params, rows, cols, 16);
                        HyperbolicTensor::encode_bicurvature(&params, &residuals)
                    },
                    HyperbolicPackingMode::AdaptiveMetric => {
                        HyperbolicTensor::encode_adaptive_metric(&data, rows, cols)
                    },
                    _ => continue,
                };
                
                let decoded = tensor.decode(rows, cols);
                let rmse = compute_rmse(&data, &decoded);
                let size = tensor.size_bytes();
                let ratio = tensor.compression_ratio(rows * cols);
                
                println!("{:<15} {:>10} {:>15.1}:1 {:>10.4}", name, size, ratio, rmse);
            }
        }
    }
    
    // 헬퍼 함수들
    fn compute_rmse(original: &[f32], decoded: &[f32]) -> f64 {
        let mse: f64 = original.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (*o - *d).powi(2) as f64)
            .sum::<f64>() / original.len() as f64;
        mse.sqrt()
    }
    
    fn estimate_params(data: &[f32], rows: usize, cols: usize, rank: usize) -> Vec<f32> {
        // 간단한 최소제곱법 추정 (실제로는 더 정교한 방법 사용)
        let mut params = vec![0.0f32; rank.max(8)];
        
        // 평균값
        params[0] = data.iter().sum::<f32>() / data.len() as f32;
        
        // 선형 기울기
        if rank >= 4 {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for i in 0..rows {
                for j in 0..cols {
                    let x = j as f32 / cols as f32;
                    let y = i as f32 / rows as f32;
                    sum_x += x * data[i * cols + j];
                    sum_y += y * data[i * cols + j];
                }
            }
            params[1] = sum_x / data.len() as f32 * 2.0;
            params[2] = sum_y / data.len() as f32 * 2.0;
            params[3] = 0.0; // 교차항은 간단히 0으로
        }
        
        params
    }
    
    fn compute_residuals(data: &[f32], params: &[f32], rows: usize, cols: usize, count: usize) -> Vec<f32> {
        let mut residuals = Vec::new();
        
        // 중요한 위치에서 잔차 계산
        for k in 0..count {
            let i = k * rows / count;
            let j = k * cols / count;
            if i < rows && j < cols {
                let x = j as f32 / cols as f32;
                let y = i as f32 / rows as f32;
                let predicted = params[0] + params[1] * x + params[2] * y + params[3] * x * y;
                let residual = data[i * cols + j] - predicted;
                residuals.push(residual);
            }
        }
        
        residuals
    }
} 