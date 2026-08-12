//! 캐싱 전략 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::{
            weight_generator::{WeightGenerator, RBEDecoderConfig},
            caching_strategy::{CachingStrategy, create_cache},
        },
        encoder::RBEEncoder,
        packed_params::HybridEncodedBlock,
    };
    use std::time::Instant;

    fn generate_test_weights(size: usize) -> Vec<f32> {
        (0..size).map(|i| (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_캐싱_전략_비교() {
        println!("\n=== 캐싱 전략 비교 테스트 ===");
        
        // 테스트 데이터 준비
        let block_size = 64;
        let num_blocks = 50;
        let mut blocks = Vec::new();
        
        for i in 0..num_blocks {
            let data = generate_test_weights(block_size * block_size);
            let mut encoder = RBEEncoder::new_b_grade();
            let mut block = encoder.encode_block(&data, block_size, block_size);
            // 각 블록을 약간씩 다르게 만들기
            block.rbe_params[0] += i as f32 * 0.01;
            blocks.push(block);
        }
        
        // 액세스 패턴 (일부 블록은 자주 액세스)
        let mut access_pattern = Vec::new();
        for _ in 0..5 {
            // 처음 10개 블록은 자주 액세스
            for i in 0..10 {
                access_pattern.push(i);
            }
        }
        // 나머지는 한번씩만
        for i in 10..num_blocks {
            access_pattern.push(i);
        }
        
        // 각 전략 테스트
        let strategies = [
            ("캐시 없음", RBEDecoderConfig::minimal_memory()),
            ("균형잡힌 (10%)", RBEDecoderConfig::balanced()),
            ("고정 LRU 16", RBEDecoderConfig::legacy(16)),
            ("적응형", RBEDecoderConfig::adaptive()),
        ];
        
        for (name, config) in strategies {
            let generator = WeightGenerator::with_config_and_blocks(config, num_blocks);
            
            let start = Instant::now();
            for &idx in &access_pattern {
                let _ = generator.decode_block(&blocks[idx]);
            }
            let elapsed = start.elapsed();
            
            let stats = generator.get_stats();
            let hit_rate = generator.get_cache_hit_rate();
            
            println!("\n{} 전략:", name);
            println!("  실행 시간: {:.2}ms", elapsed.as_millis());
            println!("  히트율: {:.1}%", hit_rate * 100.0);
            println!("  캐시 히트: {}", stats.cache_hits);
            println!("  캐시 미스: {}", stats.cache_misses);
        }
    }
    
    #[test]
    fn test_적응형_캐시_동작() {
        println!("\n=== 적응형 캐시 동작 테스트 ===");
        
        let cache = create_cache(
            CachingStrategy::Adaptive { 
                min_size: 5,     // 최소 크기를 5로 설정
                max_size: 30, 
                target_hit_rate: 0.85  // 목표 히트율을 85%로 설정
            },
            None
        );
        
        // 키 생성
        let keys: Vec<u64> = (0..30).collect();
        let values: Vec<_> = (0..30)
            .map(|i| std::sync::Arc::new(vec![i as f32; 100]))
            .collect();
        
        // 첫 단계: 낮은 히트율 (모든 키 균등 액세스)
        println!("\n첫 단계: 다양한 키 액세스");
        for i in 0..3000 {
            let idx = i % 30;
            if cache.get(keys[idx]).is_none() {
                cache.put(keys[idx], values[idx].clone());
            }
        }
        
        let stats1 = cache.stats();
        println!("  히트율: {:.1}%", stats1.hit_rate() * 100.0);
        println!("  캐시 크기: {}", stats1.current_size);
        
        // 두 번째 단계: 높은 히트율 유도 (소수 키만 집중 액세스)
        println!("\n두 번째 단계: 일부 키 집중 액세스");
        cache.clear(); // 캐시 초기화
        
        // 처음 5개 키만 반복 액세스
        for i in 0..5000 {
            let idx = i % 5; // 처음 5개만 액세스
            if cache.get(keys[idx]).is_none() {
                cache.put(keys[idx], values[idx].clone());
            }
        }
        
        let stats2 = cache.stats();
        println!("  히트율: {:.1}%", stats2.hit_rate() * 100.0);
        println!("  캐시 크기: {}", stats2.current_size);
        
        // 적응형 캐시는 소수 키 집중 액세스 시 높은 히트율을 달성해야 함
        assert!(stats2.hit_rate() > 0.85, 
                "적응형 캐시가 집중 액세스 패턴에서 높은 히트율을 달성해야 함: {:.1}%", 
                stats2.hit_rate() * 100.0);
    }
    
    #[test]
    fn test_사전계산_캐시() {
        println!("\n=== 사전계산 캐시 테스트 ===");
        
        let config = RBEDecoderConfig::max_performance();
        let generator = WeightGenerator::with_config(config);
        
        // 블록 생성
        let mut blocks = Vec::new();
        for i in 0..10 {
            let data = generate_test_weights(64 * 64);
            let mut encoder = RBEEncoder::new_b_grade();
            let mut block = encoder.encode_block(&data, 64, 64);
            block.rbe_params[0] += i as f32 * 0.1;
            blocks.push(block);
        }
        
        // 워밍업 (사전계산)
        generator.warmup(&blocks, 1.0); // 100% 워밍업
        
        // 모든 블록이 캐시되어 있어야 함
        let start = Instant::now();
        for _ in 0..100 {
            for block in &blocks {
                let _ = generator.decode_block(block);
            }
        }
        let elapsed = start.elapsed();
        
        let stats = generator.get_stats();
        println!("100회 반복 시간: {:.2}ms", elapsed.as_millis());
        println!("히트율: {:.1}%", generator.get_cache_hit_rate() * 100.0);
        
        // 사전계산 캐시는 100% 히트율을 가져야 함
        assert!(generator.get_cache_hit_rate() > 0.99, "사전계산 캐시는 거의 100% 히트율을 가져야 함");
    }
    
    #[test]
    fn test_메모리_비율_기반_캐싱() {
        println!("\n=== 메모리 비율 기반 캐싱 테스트 ===");
        
        let total_blocks = 100;
        let percentages = [0.05, 0.1, 0.2, 0.5];
        
        for percentage in percentages {
            let cache = create_cache(
                CachingStrategy::PercentageBased { percentage },
                Some(total_blocks)
            );
            
            let expected_size = ((total_blocks as f32 * percentage).ceil() as usize).max(1);
            
            // 캐시 채우기
            for i in 0..total_blocks {
                cache.put(i as u64, std::sync::Arc::new(vec![i as f32; 64]));
            }
            
            let stats = cache.stats();
            println!("\n{}% 캐싱:", (percentage * 100.0) as u32);
            println!("  예상 크기: {}", expected_size);
            println!("  실제 크기: {}", stats.current_size);
            println!("  evictions: {}", stats.evictions);
            
            // LRU 캐시의 경우 크기가 제한됨
            assert!(stats.current_size <= expected_size, 
                   "캐시 크기가 예상보다 큼: {} > {}", stats.current_size, expected_size);
        }
    }
} 