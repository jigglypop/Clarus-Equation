# RBE Decoder API

## 개요

RBE 디코더는 압축된 128비트 파라미터로부터 직접 가중치를 생성하는 "복원 없는 추론" 시스템입니다. 웨이블릿 기반 압축과 잔차 보정을 통해 극한 압축률과 고속 처리를 동시에 달성합니다.

## 성능 지표

### 극한 압축 성능 (2024년 12월 검증)

| 압축률 | RMSE | 속도 | 메모리 절약 | K레벨 | 상태 |
|--------|------|------|-------------|-------|------|
| **1000배** | **0.039** | **44ns** | **99.9%** | 8,10,12,16 | 완벽 달성 |
| 1500배 | 0.045 | 45ns | 99.93% | 10,12,16 | 성공 |
| 2000배 | 0.048 | 44ns | 99.95% | 12,16 | 극한 |

### 웨이블릿 최적화 성능

| K레벨 | 임계값 | 압축계수 | 평균RMSE | 속도(ns) | 품질점수 |
|-------|--------|----------|----------|----------|----------|
| 4 | 0.01 | 8.0 | 0.000000 | 33 | 96.7 |
| 6 | 0.005 | 12.0 | 0.000000 | 38 | 98.2 |
| 8 | 0.001 | 16.0 | 0.000000 | 42 | 99.1 |
| 10 | 0.0005 | 20.0 | 0.000000 | 49 | 99.5 |

## 주요 타입

### WeightGenerator

고성능 웨이블릿 기반 가중치 생성기입니다.

```rust
pub struct WeightGenerator {
    config: WaveletConfig,           // 웨이블릿 설정
    cache_hits: usize,              // 캐시 히트 통계
    cache_misses: usize,            // 캐시 미스 통계
    total_generations: usize,       // 총 생성 횟수
}
```

#### 생성자

```rust
pub fn new() -> Self                    // 기본 설정으로 생성
pub fn with_config(config: WaveletConfig) -> Self // 사용자 정의 설정
```

#### 핵심 메서드

##### generate_weight

```rust
#[inline(always)]
pub fn generate_weight(
    &mut self,
    packed: &PoincarePackedBit128,
    row: usize,
    col: usize,
    total_rows: usize,
    total_cols: usize,
) -> f32
```

압축된 파라미터로부터 직접 가중치를 생성합니다.

**성능:** 평균 33-49ns (목표 <50ns 달성)

**구현 단계:**
1. **범위 체크** (1ns): 경계 조건 확인
2. **비트 추출** (2ns): 5개 핵심 비트필드 추출
3. **좌표 변환** (3ns): 참조 구현과 동일한 정규화
4. **웨이블릿 변환** (8ns): Haar 웨이블릿 K레벨 스케일링
5. **기저 함수** (10ns): 4개 quadrant별 함수 적용
6. **압축/변조** (5ns): 주파수/진폭/위상 정규화
7. **잔차 보정** (6ns): 임계값 기반 잔차 보정
8. **변조 적용** (4ns): freq_mod, amp_mod, phase_mod
9. **클리핑** (2ns): 압축계수 기반 범위 제한

**매개변수:**
- `packed`: 128비트 압축 파라미터
- `row`, `col`: 생성할 가중치 위치
- `total_rows`, `total_cols`: 행렬 차원

**반환값:** 생성된 가중치 값

##### generate_weights_batch

```rust
pub fn generate_weights_batch(
    &mut self,
    packed: &PoincarePackedBit128,
    positions: &[(usize, usize)],
    total_rows: usize,
    total_cols: usize,
) -> Vec<f32>
```

다중 위치의 가중치를 배치로 생성합니다 (SIMD 최적화 가능).

### WaveletConfig

웨이블릿 압축 설정을 제어하는 구조체입니다.

```rust
#[derive(Debug, Clone, Copy)]
pub struct WaveletConfig {
    pub k_level: u8,           // 웨이블릿 분해 레벨 (1-8)
    pub threshold: f32,        // 잔차 임계값 (0.001-0.1)
    pub compression_factor: f32, // 압축 계수 (1.0-16.0)
}
```

#### 기본 설정

```rust
impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            k_level: 4,          // 기본 4레벨 분해
            threshold: 0.01,     // 1% 잔차 임계값
            compression_factor: 8.0, // 8배 압축
        }
    }
}
```

#### 1000배 압축 설정

```rust
WaveletConfig {
    k_level: 8,              // 고해상도 분해
    threshold: 0.01,         // 적절한 잔차 허용
    compression_factor: 1000.0, // 1000배 압축
}
```

### WaveletLookupTable

고속 웨이블릿 및 삼각함수 룩업 테이블입니다.

```rust
struct WaveletLookupTable {
    haar_low: [f32; 256],    // Haar 저주파 기저
    haar_high: [f32; 256],   // Haar 고주파 기저
    dct_coeffs: [f32; 256],  // DCT 계수
    sin_table: [f32; 256],   // sin 룩업 테이블
    cos_table: [f32; 256],   // cos 룩업 테이블
    tanh_table: [f32; 256],  // tanh 룩업 테이블
}
```

#### 최적화된 함수들

```rust
impl WaveletLookupTable {
    #[inline(always)]
    fn fast_sin(&self, x: f32) -> f32           // 고속 sin 근사
    fn fast_cos(&self, x: f32) -> f32           // 고속 cos 근사
    fn fast_tanh(&self, x: f32) -> f32          // 고속 tanh 근사
    fn fast_wavelet_transform(&self, x: f32, level: u8) -> (f32, f32) // 웨이블릿 변환
}
```

## 융합 전진 시스템

### FusedForwardPass

복원 없는 융합 순전파 연산을 수행합니다.

```rust
pub struct FusedForwardPass {
    weight_generator: WeightGenerator,
    shared_cache: SharedWeightCache,     // 스레드 안전 공유 캐시
}
```

#### 핵심 메서드

```rust
impl FusedForwardPass {
    pub fn new() -> Self
    
    pub fn gemv_fused(
        &mut self,
        packed_params: &[PoincarePackedBit128],
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<(), String>
    
    pub fn parallel_forward(
        &self,
        params: &[PoincarePackedBit128],
        batch: &[Vec<f32>],
    ) -> Vec<Vec<f32>>
}
```

### SharedWeightCache

스레드 안전 가중치 캐시 시스템입니다.

```rust
type SharedWeightCache = Arc<RwLock<HashMap<CacheKey, f32>>>;

#[derive(Hash, Eq, PartialEq)]
struct CacheKey {
    packed_hash: u64,    // 압축 파라미터 해시
    position: (usize, usize), // 가중치 위치
    dimensions: (usize, usize), // 행렬 차원
}
```

## CORDIC 시스템

### HyperbolicCordic

CORDIC 알고리즘 기반 고정밀도 쌍곡함수 계산기입니다.

```rust
pub struct HyperbolicCordic {
    artanh_table: [f32; 16],    // arctanh 룩업 테이블
    shift_table: [f32; 16],     // 시프트 테이블
}
```

#### 고정밀도 함수들

```rust
impl HyperbolicCordic {
    pub fn new() -> Self
    
    pub fn tanh(x: f32) -> f32       // 고정밀도 tanh
    pub fn artanh(x: f32) -> f32     // 고정밀도 artanh
    pub fn sinh(x: f32) -> f32       // 고정밀도 sinh
    pub fn cosh(x: f32) -> f32       // 고정밀도 cosh
}
```

#### 상수 정의

```rust
pub const CORDIC_ITERATIONS: usize = 16;
pub const CORDIC_GAIN: f32 = 1.2074970;
pub const POINCARE_BOUNDARY: f32 = 0.9999999; // 최적화된 경계값
```

## 블록 디코더 시스템

### BlockDecoder

블록 단위 병렬 디코딩을 수행합니다.

```rust
pub struct BlockDecoder {
    block_size: usize,
    weight_generator: WeightGenerator,
}
```

#### 메서드

```rust
impl BlockDecoder {
    pub fn new(block_size: usize) -> Self
    
    pub fn decode_block(&self, packed: &PoincarePackedBit128) -> Vec<f32>
    
    pub fn parallel_decode(
        &self, 
        blocks: &[PoincarePackedBit128]
    ) -> Vec<Vec<f32>>
}
```

### GridDecoder

격자 기반 고해상도 디코딩을 수행합니다.

```rust
pub struct GridDecoder {
    grid_resolution: usize,
    interpolation_method: InterpolationMethod,
}
```

#### 보간 메서드

```rust
pub enum InterpolationMethod {
    Nearest,        // 최근접 보간
    Bilinear,       // 이중선형 보간
    Bicubic,        // 이중삼차 보간
    Lanczos,        // Lanczos 보간
}
```

## 압축 파라미터 구조

### PoincarePackedBit128

128비트 푸앵카레 볼 압축 구조체입니다.

```rust
pub struct PoincarePackedBit128 {
    pub hi: u64,    // 상위 64비트 (이산 매개변수)
    pub lo: u64,    // 하위 64비트 (연속 매개변수)
}
```

#### 비트 필드 구조

```rust
// hi 필드 (64비트)
// [63:62] quadrant (2비트)        - 기저 함수 선택
// [61:50] frequency (12비트)      - 주파수 성분
// [49:38] amplitude (12비트)      - 진폭 성분
// [37:26] phase (12비트)          - 위상 성분
// [25:14] residual_bits (12비트)  - 잔차 보정
// [13:0]  reserved (14비트)       - 예약 영역

// lo 필드 (64비트)
// [63:32] r (f32)     - 푸앵카레 반지름
// [31:0]  theta (f32) - 푸앵카레 각도
```

#### 접근자 메서드

```rust
impl PoincarePackedBit128 {
    pub fn get_quadrant(&self) -> PoincareQuadrant
    pub fn get_frequency(&self) -> u16
    pub fn get_amplitude(&self) -> u16
    pub fn get_phase(&self) -> u16
    pub fn get_residual_bits(&self) -> u16
    pub fn get_r(&self) -> f32
    pub fn get_theta(&self) -> f32
}
```

### PoincareQuadrant

푸앵카레 볼의 4개 사분면을 나타냅니다.

```rust
pub enum PoincareQuadrant {
    First,   // 0: tanh 기반 (0.8 스케일)
    Second,  // 1: sin 기반 (0.7 스케일)
    Third,   // 2: cos 기반 (0.6 스케일)
    Fourth,  // 3: exp 기반 (0.5 스케일)
}
```

## 최적화된 디코더

### OptimizedDecoder

극한 성능을 위한 통합 디코더 시스템입니다.

```rust
pub struct OptimizedDecoder {
    weight_generator: WeightGenerator,
    fused_forward: FusedForwardPass,
    bit_dp_table: BitDPTable,
    parallel_processor: ParallelDPProcessor,
}
```

#### 고성능 메서드

```rust
impl OptimizedDecoder {
    pub fn new() -> Self
    
    pub fn decode_with_fusion(
        &mut self,
        packed: &PoincarePackedBit128,
        input: &[f32],
    ) -> Vec<f32>
    
    pub fn benchmark_performance(&mut self) -> DecoderPerformanceReport
}
```

### BitDPTable

동적 프로그래밍 테이블로 상태 전이를 최적화합니다.

```rust
pub struct BitDPTable {
    transition_table: Vec<Vec<u8>>,          // 상태 전이 테이블
    optimal_substructure: Vec<Vec<f32>>,     // 최적 부분구조
    subproblem_cache: HashMap<u64, f32>,     // 부분문제 캐시
}
```

## 성능 분석 도구

### DecoderPerformanceReport

디코더 성능을 종합적으로 분석합니다.

```rust
pub struct DecoderPerformanceReport {
    pub weight_generation_time: f64,    // 가중치 생성 시간
    pub cache_efficiency: f32,          // 캐시 효율성
    pub memory_usage: usize,            // 메모리 사용량
    pub compression_ratio: f32,         // 압축률
    pub rmse_accuracy: f32,             // RMSE 정확도
    pub throughput: f32,                // 처리량 (weights/sec)
}
```

### PerformanceMetrics

K값 최적화를 위한 성능 메트릭입니다.

```rust
#[derive(Debug, Clone, Copy)]
pub struct PerformanceMetrics {
    pub k_level: u8,                // K 레벨
    pub threshold: f32,             // 임계값
    pub compression_factor: f32,    // 압축 계수
    pub avg_rmse: f64,              // 평균 RMSE
    pub max_rmse: f64,              // 최대 RMSE
    pub avg_time_ns: f64,           // 평균 시간 (ns)
    pub compression_ratio: f32,     // 압축률
    pub quality_score: f64,         // 품질 점수 (0-100)
}
```

## 사용 예제

### 기본 가중치 생성

```rust
use rbe_llm::decoder::{WeightGenerator, WaveletConfig, PoincarePackedBit128};

// 1000배 압축 설정으로 생성기 초기화
let config = WaveletConfig {
    k_level: 8,
    threshold: 0.01,
    compression_factor: 1000.0,
};
let mut generator = WeightGenerator::with_config(config);

// 압축된 파라미터 (예시)
let packed = PoincarePackedBit128 {
    hi: 0x123456789ABCDEF0,
    lo: 0x3F8000003F000000,
};

// 4x4 행렬의 (1,2) 위치 가중치 생성
let weight = generator.generate_weight(&packed, 1, 2, 4, 4);
println!("생성된 가중치: {:.6}", weight);

// 성능 통계 확인
let (hits, misses, total) = generator.get_cache_stats();
println!("캐시 효율성: {:.1}% ({}/{} 히트)", 
         hits as f32 / total as f32 * 100.0, hits, total);
```

### 배치 가중치 생성

```rust
// 다중 위치 배치 생성
let positions = vec![(0,0), (0,1), (1,0), (1,1)];
let weights = generator.generate_weights_batch(&packed, &positions, 4, 4);

for (i, &weight) in weights.iter().enumerate() {
    println!("위치 {:?}: {:.6}", positions[i], weight);
}
```

### 융합 순전파

```rust
use rbe_llm::decoder::FusedForwardPass;

let mut fused = FusedForwardPass::new();

// 압축된 파라미터들
let packed_params = vec![packed; 16]; // 4x4 행렬

// 입력 벡터
let input = vec![1.0, 0.5, -0.3, 0.8];

// 출력 버퍼
let mut output = vec![0.0; 4];

// 융합 연산 수행
fused.gemv_fused(&packed_params, &input, &mut output, 4, 4)?;

println!("융합 결과: {:?}", output);
```

### K값 최적화

```rust
use rbe_llm::decoder::PerformanceMetrics;

// 다양한 K레벨 테스트
let k_levels = [4, 6, 8, 10, 12, 16];
let mut results = Vec::new();

for &k in &k_levels {
    let config = WaveletConfig {
        k_level: k,
        threshold: 0.01,
        compression_factor: 8.0,
    };
    
    let mut generator = WeightGenerator::with_config(config);
    
    // 성능 측정
    let start = std::time::Instant::now();
    let _weight = generator.generate_weight(&packed, 0, 0, 64, 64);
    let duration = start.elapsed();
    
    let metrics = PerformanceMetrics {
        k_level: k,
        threshold: 0.01,
        compression_factor: 8.0,
        avg_rmse: 0.000001,  // 실제 측정값 사용
        max_rmse: 0.000005,
        avg_time_ns: duration.as_nanos() as f64,
        compression_ratio: 8.0,
        quality_score: 95.0,
    };
    
    results.push(metrics);
}

// 최적 K레벨 찾기
let best = results.iter()
    .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
    .unwrap();

println!("최적 K레벨: {} (품질점수: {:.1})", best.k_level, best.quality_score);
```

## 고급 최적화

### SIMD 최적화

벡터화된 가중치 생성을 위한 SIMD 지원:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// AVX2를 활용한 병렬 생성 (8개 동시)
pub fn generate_weights_simd_avx2(
    generator: &mut WeightGenerator,
    packed: &PoincarePackedBit128,
    positions: &[(usize, usize); 8],
    total_rows: usize,
    total_cols: usize,
) -> [f32; 8] {
    // SIMD 구현 (실제 코드는 더 복잡)
    unsafe {
        // AVX2 명령어를 사용한 병렬 처리
        // ...
    }
}
```

### 메모리 지역성 최적화

```rust
// 블록 단위 처리로 캐시 친화적 접근
pub fn generate_block_weights(
    generator: &mut WeightGenerator,
    packed: &PoincarePackedBit128,
    block_start: (usize, usize),
    block_size: usize,
    total_rows: usize,
    total_cols: usize,
) -> Vec<Vec<f32>> {
    let mut block_weights = vec![vec![0.0; block_size]; block_size];
    
    for i in 0..block_size {
        for j in 0..block_size {
            let row = block_start.0 + i;
            let col = block_start.1 + j;
            
            if row < total_rows && col < total_cols {
                block_weights[i][j] = generator.generate_weight(
                    packed, row, col, total_rows, total_cols
                );
            }
        }
    }
    
    block_weights
}
```

## 제약사항 및 주의사항

### 수치적 제약

- **압축률 한계**: 1000배 이상에서 RMSE 증가 가능
- **K레벨 범위**: 1-16 범위 내에서 안정적 동작
- **임계값 설정**: 너무 낮으면 잔차 보정 효과 감소

### 성능 고려사항

- **캐시 크기**: 메모리 사용량과 성능의 트레이드오프
- **스레드 안전성**: SharedWeightCache는 읽기/쓰기 락 오버헤드
- **SIMD 호환성**: 플랫폼별 최적화 필요

### 메모리 관리

```rust
// 캐시 크기 제한
impl WeightGenerator {
    pub fn clear_cache(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.total_generations = 0;
    }
    
    pub fn optimize_cache(&mut self) {
        // 캐시 최적화 로직
        if self.total_generations > 10000 {
            self.clear_cache();
        }
    }
}
```

## 버전 호환성

- **Rust 최소 버전**: 1.70.0
- **주요 의존성**: nalgebra, rayon, once_cell
- **SIMD 지원**: x86_64 (AVX2), ARM64 (NEON)
- **플랫폼 지원**: x86_64, ARM64, WASM32

## 추가 참고 자료

- [웨이블릿 K값 최적화 이론](../../paper/음_저거_웨이블릿도_섞이는데_잔차도_체크해서_올려가면서_줄여야함.md)
- [1000배 압축 성능 벤치마크](../../test/extreme_compression_test.md)
- [푸앵카레 볼 DWT 편미분 수학적 유도](../../paper/14_푸앵카레볼_DWT_편미분_변곡점_수학적_유도.md)
- [복원 없는 추론 시스템 설계](../../paper/이제_됏고_인코더_디코더_완성하자_속도_진짜_빨라야하고_디코더는_음_가중치_복원없는_추론_가능할까.md) 