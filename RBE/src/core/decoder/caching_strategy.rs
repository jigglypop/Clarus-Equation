//! 디코더 캐싱 전략 모듈

use std::sync::Arc;
use parking_lot::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::collections::HashMap;
use crate::packed_params::HybridEncodedBlock;

/// 캐싱 전략
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CachingStrategy {
    /// 캐싱 없음 - 매번 디코딩
    NoCache,
    /// 고정 크기 LRU 캐시
    FixedLRU { size: usize },
    /// 메모리 비율 기반 캐싱 (전체 모델 크기의 N%)
    PercentageBased { percentage: f32 },
    /// 적응형 캐싱 - 접근 패턴에 따라 동적 조정
    Adaptive { 
        min_size: usize,
        max_size: usize,
        target_hit_rate: f32,
    },
    /// 전체 사전 디코딩 (최대 성능, 최대 메모리)
    PrecomputeAll,
}

impl Default for CachingStrategy {
    fn default() -> Self {
        CachingStrategy::FixedLRU { size: 16 }
    }
}

/// 캐시 통계
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_size: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

/// 캐시 구현 트레이트
pub trait DecoderCache: Send + Sync {
    /// 캐시에서 블록 가져오기
    fn get(&self, key: u64) -> Option<Arc<Vec<f32>>>;
    
    /// 캐시에 블록 저장
    fn put(&self, key: u64, value: Arc<Vec<f32>>);
    
    /// 캐시 비우기
    fn clear(&self);
    
    /// 통계 가져오기
    fn stats(&self) -> CacheStats;
    
    /// 캐시 크기 조정 (지원하는 경우)
    fn resize(&self, new_size: usize) -> bool {
        false // 기본적으로 지원하지 않음
    }
}

/// 캐싱 없음 구현
pub struct NoCache;

impl DecoderCache for NoCache {
    fn get(&self, _key: u64) -> Option<Arc<Vec<f32>>> {
        None
    }
    
    fn put(&self, _key: u64, _value: Arc<Vec<f32>>) {
        // 아무것도 하지 않음
    }
    
    fn clear(&self) {
        // 아무것도 하지 않음
    }
    
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

/// LRU 캐시 구현
pub struct LRUCache {
    cache: Arc<RwLock<LruCache<u64, Arc<Vec<f32>>>>>,
    stats: Arc<RwLock<CacheStats>>,
}

impl LRUCache {
    pub fn new(size: usize) -> Self {
        let cache_size = NonZeroUsize::new(size.max(1)).unwrap();
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
}

impl DecoderCache for LRUCache {
    fn get(&self, key: u64) -> Option<Arc<Vec<f32>>> {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        if let Some(value) = cache.get(&key) {
            stats.hits += 1;
            Some(value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }
    
    fn put(&self, key: u64, value: Arc<Vec<f32>>) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        if cache.put(key, value).is_some() {
            stats.evictions += 1;
        }
        stats.current_size = cache.len();
    }
    
    fn clear(&self) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        cache.clear();
        stats.current_size = 0;
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }
    
    fn resize(&self, new_size: usize) -> bool {
        let cache_size = match NonZeroUsize::new(new_size) {
            Some(size) => size,
            None => return false,
        };
        
        let mut cache = self.cache.write();
        let old_cache = std::mem::replace(&mut *cache, LruCache::new(cache_size));
        
        // 기존 항목 복사 (최대 new_size개)
        for (k, v) in old_cache {
            cache.put(k, v);
        }
        
        true
    }
}

/// 전체 사전 계산 캐시
pub struct PrecomputedCache {
    cache: Arc<RwLock<HashMap<u64, Arc<Vec<f32>>>>>,
    stats: Arc<RwLock<CacheStats>>,
}

impl PrecomputedCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
}

impl DecoderCache for PrecomputedCache {
    fn get(&self, key: u64) -> Option<Arc<Vec<f32>>> {
        let cache = self.cache.read();
        let mut stats = self.stats.write();
        
        if let Some(value) = cache.get(&key) {
            stats.hits += 1;
            Some(value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }
    
    fn put(&self, key: u64, value: Arc<Vec<f32>>) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        cache.insert(key, value);
        stats.current_size = cache.len();
    }
    
    fn clear(&self) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        cache.clear();
        stats.current_size = 0;
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }
}

/// 적응형 캐시
pub struct AdaptiveCache {
    cache: Arc<RwLock<LruCache<u64, Arc<Vec<f32>>>>>,
    stats: Arc<RwLock<CacheStats>>,
    config: AdaptiveConfig,
    adjustment_counter: Arc<RwLock<u64>>,
}

#[derive(Debug, Clone)]
struct AdaptiveConfig {
    min_size: usize,
    max_size: usize,
    target_hit_rate: f32,
    adjustment_interval: u64,
}

impl AdaptiveCache {
    pub fn new(min_size: usize, max_size: usize, target_hit_rate: f32) -> Self {
        let initial_size = (min_size + max_size) / 2;
        let cache_size = NonZeroUsize::new(initial_size).unwrap();
        
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            config: AdaptiveConfig {
                min_size,
                max_size,
                target_hit_rate,
                adjustment_interval: 100, // 100번 접근마다 조정
            },
            adjustment_counter: Arc::new(RwLock::new(0)),
        }
    }
    
    fn adjust_size(&self) {
        let stats = self.stats.read();
        let current_hit_rate = stats.hit_rate();
        let current_size = self.cache.read().cap().get();
        drop(stats);
        
        let new_size = if current_hit_rate < self.config.target_hit_rate - 0.05 {
            // 히트율이 목표보다 5% 이상 낮으면 캐시 크기 증가
            (current_size * 3 / 2).min(self.config.max_size)
        } else if current_hit_rate > self.config.target_hit_rate + 0.1 {
            // 히트율이 목표보다 10% 이상 높으면 캐시 크기 감소
            (current_size * 2 / 3).max(self.config.min_size)
        } else {
            return; // 적절한 범위
        };
        
        if new_size != current_size {
            println!("적응형 캐시 크기 조정: {} → {} (현재 히트율: {:.1}%)", 
                    current_size, new_size, current_hit_rate * 100.0);
            self.resize(new_size);
        }
    }
}

impl DecoderCache for AdaptiveCache {
    fn get(&self, key: u64) -> Option<Arc<Vec<f32>>> {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        let mut counter = self.adjustment_counter.write();
        
        let result = if let Some(value) = cache.get(&key) {
            stats.hits += 1;
            Some(value.clone())
        } else {
            stats.misses += 1;
            None
        };
        
        *counter += 1;
        if *counter >= self.config.adjustment_interval {
            *counter = 0;
            drop(cache);
            drop(stats);
            drop(counter);
            self.adjust_size();
        }
        
        result
    }
    
    fn put(&self, key: u64, value: Arc<Vec<f32>>) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        if cache.put(key, value).is_some() {
            stats.evictions += 1;
        }
        stats.current_size = cache.len();
    }
    
    fn clear(&self) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        
        cache.clear();
        stats.current_size = 0;
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }
    
    fn resize(&self, new_size: usize) -> bool {
        let cache_size = match NonZeroUsize::new(new_size) {
            Some(size) => size,
            None => return false,
        };
        
        let mut cache = self.cache.write();
        let old_cache = std::mem::replace(&mut *cache, LruCache::new(cache_size));
        
        // 기존 항목 복사
        for (k, v) in old_cache {
            cache.put(k, v);
        }
        
        true
    }
}

/// 캐싱 전략에 따른 캐시 생성
pub fn create_cache(strategy: CachingStrategy, total_blocks: Option<usize>) -> Box<dyn DecoderCache> {
    match strategy {
        CachingStrategy::NoCache => Box::new(NoCache),
        CachingStrategy::FixedLRU { size } => Box::new(LRUCache::new(size)),
        CachingStrategy::PercentageBased { percentage } => {
            let size = if let Some(total) = total_blocks {
                ((total as f32 * percentage).ceil() as usize).max(1)
            } else {
                16 // 기본값
            };
            Box::new(LRUCache::new(size))
        }
        CachingStrategy::Adaptive { min_size, max_size, target_hit_rate } => {
            Box::new(AdaptiveCache::new(min_size, max_size, target_hit_rate))
        }
        CachingStrategy::PrecomputeAll => Box::new(PrecomputedCache::new()),
    }
} 