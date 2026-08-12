//! RBE 기반 가중치 생성기 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::weight_generator::{WeightGenerator, RBEDecoderConfig},
        packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient, RbeParameters},
    };
    
    #[test]
    fn test_블록_디코딩_정확성() {
        let mut generator = WeightGenerator::new();
        
        // 테스트용 블록 생성
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params: [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            residuals: vec![
                ResidualCoefficient { index: (0, 0), value: 0.1 },
                ResidualCoefficient { index: (1, 1), value: 0.05 },
            ],
            transform_type: TransformType::Dct,
        };
        
        // 블록 디코딩
        let decoded = generator.decode_block(&block);
        
        // 크기 확인
        assert_eq!(decoded.len(), 16); // 4x4
        
        // 값 범위 확인
        for &val in decoded.iter() {
            assert!(val.is_finite());
            assert!(val.abs() < 10.0); // 합리적인 범위
        }
    }
    
    #[test]
    fn test_캐시_효율성() {
        let generator = WeightGenerator::new();
        
        let block = HybridEncodedBlock {
            rows: 8,
            cols: 8,
            rbe_params: [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            residuals: vec![],
            transform_type: TransformType::Dct,
        };
        
        // 첫 번째 호출 (캐시 미스)
        let _decoded1 = generator.decode_block(&block);
        let stats1 = generator.get_stats();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);
        
        // 두 번째 호출 (캐시 히트)
        let _decoded2 = generator.decode_block(&block);
        let stats2 = generator.get_stats();
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);
        
        // 캐시 히트율 확인
        let hit_rate = generator.get_cache_hit_rate();
        assert_eq!(hit_rate, 0.5); // 1 hit / 2 total
    }
    
    #[test]
    fn test_다양한_블록_크기() {
        let generator = WeightGenerator::new();
        
        let sizes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)];
        
        for (rows, cols) in sizes {
            let block = HybridEncodedBlock {
                rows,
                cols,
                rbe_params: [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
                residuals: vec![],
                transform_type: TransformType::Dct,
            };
            
            let decoded = generator.decode_block(&block);
            assert_eq!(decoded.len(), rows * cols);
        }
    }
    
    #[test]
    fn test_병렬_처리_일관성() {
        let config = RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: 16 },
            enable_parallel: true,
            enable_simd: false,
        };
        
        let generator_parallel = WeightGenerator::with_config(config.clone());
        
        let mut config_seq = config;
        config_seq.enable_parallel = false;
        let generator_sequential = WeightGenerator::with_config(config_seq);
        
        let block = HybridEncodedBlock {
            rows: 64,
            cols: 64,
            rbe_params: [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            residuals: vec![
                ResidualCoefficient { index: (10, 20), value: 0.5 },
                ResidualCoefficient { index: (30, 40), value: -0.3 },
            ],
            transform_type: TransformType::Dct,
        };
        
        let decoded_parallel = generator_parallel.decode_block(&block);
        let decoded_sequential = generator_sequential.decode_block(&block);
        
        // 병렬과 순차 처리 결과가 동일해야 함
        assert_eq!(decoded_parallel.len(), decoded_sequential.len());
        
        for (i, (&p, &s)) in decoded_parallel.iter().zip(decoded_sequential.iter()).enumerate() {
            assert!((p - s).abs() < 1e-6, "Mismatch at index {}: {} vs {}", i, p, s);
        }
    }
    
    #[test]
    fn test_잔차_복원() {
        let generator = WeightGenerator::new();
        
        // 잔차가 있는 블록
        let block_with_residuals = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params: [0.0; 8], // RBE 부분은 0
            residuals: vec![
                ResidualCoefficient { index: (0, 0), value: 1.0 },
                ResidualCoefficient { index: (1, 1), value: 0.5 },
                ResidualCoefficient { index: (2, 2), value: 0.25 },
            ],
            transform_type: TransformType::Dwt,
        };
        
        let decoded = generator.decode_block(&block_with_residuals);
        
        // 잔차가 제대로 복원되었는지 확인
        assert!(decoded[0] != 0.0); // (0,0)에 잔차 있음
        assert!(decoded[5] != 0.0); // (1,1)에 잔차 있음
        assert!(decoded[10] != 0.0); // (2,2)에 잔차 있음
    }
    
    #[test]
    fn test_캐시_크기_제한() {
        let config = RBEDecoderConfig {
            caching_strategy: crate::core::decoder::CachingStrategy::FixedLRU { size: 2 },
            enable_parallel: false,
            enable_simd: false,
        };
        
        let generator = WeightGenerator::with_config(config);
        
        // 3개의 서로 다른 블록 생성
        let blocks: Vec<HybridEncodedBlock> = (0..3).map(|i| {
            HybridEncodedBlock {
                rows: 4,
                cols: 4,
                rbe_params: [i as f32; 8], // 각각 다른 파라미터
                residuals: vec![],
                transform_type: TransformType::Dct,
            }
        }).collect();
        
        // 모든 블록 디코딩
        for block in &blocks {
            let _ = generator.decode_block(block);
        }
        
        // 첫 번째 블록 다시 디코딩 (캐시에서 제거되었을 것)
        let _ = generator.decode_block(&blocks[0]);
        
        let stats = generator.get_stats();
        // 캐시 크기가 2이므로 첫 번째 블록은 캐시에서 제거됨
        assert_eq!(stats.cache_misses, 4); // 3개 + 재디코딩 1개
    }
    
    #[test]
    fn test_성능_통계() {
        let mut generator = WeightGenerator::new();
        
        let block = HybridEncodedBlock {
            rows: 16,
            cols: 16,
            rbe_params: [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            residuals: vec![],
            transform_type: TransformType::Dct,
        };
        
        // 여러 번 디코딩
        for _ in 0..10 {
            let _ = generator.decode_block(&block);
        }
        
        let stats = generator.get_stats();
        assert_eq!(stats.total_decodes, 10);
        assert_eq!(stats.cache_hits, 9); // 첫 번째만 미스
        assert_eq!(stats.cache_misses, 1);
        assert!(stats.total_decode_time_ns > 0);
        
        // 캐시 히트율
        let hit_rate = generator.get_cache_hit_rate();
        assert_eq!(hit_rate, 0.9); // 90%
    }
} 