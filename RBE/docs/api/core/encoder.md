# RBE Encoder API

## 개요

RBE (Riemannian Basis Encoding) 인코더는 푸앵카레 볼 공간에서 고효율 행렬 압축을 수행하는 핵심 컴포넌트입니다. 수학적으로 검증된 계수 예측 공식과 품질등급 시스템을 통해 정밀한 압축률 제어를 제공합니다.

## 성능 지표

### 검증된 압축 성능 (2024년 12월)

| 품질등급 | 계수(K) | RMSE | 압축률 | 실행시간 | 메모리 절약 |
|----------|---------|------|--------|----------|-------------|
| **S급**  | 1024    | 0.000033 | 204.8x | 3.60초 | 99.51% |
| **A급**  | 148     | 0.000988 | 204.8x | 0.52초 | 99.51% |
| **B급**  | 40      | 0.007768 | 204.8x | 0.14초 | 99.51% |
| **C급**  | 8       | 0.035870 | 204.8x | 0.03초 | 99.51% |

### 수학적 정확성

푸앵카레 볼 계수 예측 공식: **100% 정확도** 달성

```
K = ⌈(블록크기²) / R⌉
R = 32 - log₂(블록크기/16)
```

## 주요 타입

### RBEEncoder

핵심 압축 엔진입니다.

```rust
pub struct RBEEncoder {
    pub k_coeffs: usize,           // 유지할 잔차 계수 개수
    pub transform_type: TransformType, // 변환 타입 (DCT/DWT)
    dct_planner_f32: DctPlanner<f32>, // DCT 플래너
}
```

#### 생성자

```rust
pub fn new(k_coeffs: usize, transform_type: TransformType) -> Self
```

압축 계수와 변환 타입을 지정하여 새로운 RBE 인코더를 생성합니다.

**매개변수:**
- `k_coeffs`: 유지할 잔차 계수의 개수
- `transform_type`: 변환 타입 (`TransformType::Dct` 또는 `TransformType::Dwt`)

**반환값:** 초기화된 `RBEEncoder` 인스턴스

#### 품질등급별 생성자

```rust
pub fn new_s_grade() -> Self     // S급: K=500, DWT
pub fn new_a_grade() -> Self     // A급: K=300, DWT  
pub fn new_b_grade() -> Self     // B급: K=200, DWT
pub fn new_extreme_compression() -> Self // 극한: K=50, DWT
```

### 핵심 메서드

#### encode_block

```rust
pub fn encode_block(
    &mut self, 
    block_data: &[f32], 
    rows: usize, 
    cols: usize
) -> HybridEncodedBlock
```

단일 블록을 RBE로 압축합니다.

**성능:** 블록 크기와 계수에 따라 0.03초~3.6초

**매개변수:**
- `block_data`: 압축할 행렬 데이터 (row-major 순서)
- `rows`: 행 수
- `cols`: 열 수

**반환값:** 압축된 `HybridEncodedBlock`

**구현 세부사항:**
1. **입력 검증**: 데이터 길이와 차원 일치성 확인
2. **A Matrix 캐싱**: 동일 크기 블록에 대한 재사용
3. **SVD 분해**: 수치적 안정성을 위한 tolerance 적용
4. **RBE 파라미터 계산**: 8차원 기저 함수 기반
5. **잔차 변환**: DCT 또는 DWT 적용
6. **계수 선택**: 상위 K개 계수 유지

#### compress_with_profile

```rust
pub fn compress_with_profile(
    matrix_data: &[f32],
    height: usize,
    width: usize,
    block_size: usize,
    coefficients: usize,
    transform_type: TransformType,
) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String>
```

대용량 행렬의 블록 단위 병렬 압축을 수행합니다.

**반환값:** `(압축블록들, 압축시간, 압축률, RMSE)`

**최적화 기법:**
- **병렬 처리**: Rayon을 활용한 블록별 병렬 압축
- **비대칭 매트릭스 지원**: height ≠ width 처리
- **메모리 효율적**: 블록 단위 처리로 메모리 지역성 향상

#### compress_with_config

```rust
pub fn compress_with_config(
    matrix_data: &[f32],
    height: usize,
    width: usize,
    config: &CompressionConfig,
) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String>
```

설정 기반 지능형 압축을 수행합니다.

**검증 기능:**
- RMSE 임계값 초과 검사
- 압축률 최소 요구사항 확인
- 최소 블록 개수 요구사항 검증

### 지능형 계수 결정

#### find_critical_coefficients_single_block

```rust
pub fn find_critical_coefficients_single_block(
    data: &[f32],
    rows: usize,
    cols: usize,
    rmse_threshold: f32,
    transform_type: TransformType,
) -> Result<usize, String>
```

이분 탐색을 통한 최적 계수 자동 결정입니다.

**알고리즘:**
1. **예측 공식 적용**: 수학적 예측으로 초기값 설정
2. **품질 검증**: 예측값으로 빠른 품질 확인
3. **이분 탐색**: 필요시 정밀 탐색 (2배 범위 내)
4. **최적값 반환**: RMSE 임계값을 만족하는 최소 계수

#### create_quality_encoder

```rust
pub fn create_quality_encoder(
    data: &[f32],
    rows: usize,
    cols: usize,
    grade: QualityGrade,
    transform_type: TransformType,
) -> Result<RBEEncoder, String>
```

품질등급에 따른 자동 인코더 생성입니다.

## 설정 시스템

### CompressionConfig

압축 설정을 세밀하게 제어하는 구조체입니다.

```rust
pub struct CompressionConfig {
    pub block_size: usize,                    // 블록 크기
    pub quality_grade: QualityGrade,          // 품질 등급
    pub transform_type: TransformType,        // 변환 타입
    pub profile: CompressionProfile,          // 성능 프로파일
    pub custom_coefficients: Option<usize>,   // 사용자 정의 계수
    pub min_block_count: Option<usize>,       // 최소 블록 개수
    pub rmse_threshold: Option<f32>,          // RMSE 임계값
    pub compression_ratio_threshold: Option<f32>, // 압축률 임계값
}
```

#### 프리셋 구성

```rust
impl CompressionConfig {
    pub fn default() -> Self              // 균형잡힌 기본 설정
    pub fn ultra_high() -> Self           // 초고품질 설정
    pub fn fast() -> Self                 // 고속 압축 설정
    pub fn custom(                        // 사용자 정의 설정
        block_size: usize,
        rmse_threshold: f32,
        compression_ratio: f32,
        min_blocks: Option<usize>
    ) -> Self
}
```

### QualityGrade

품질 등급 열거형입니다.

```rust
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001 (초고품질) - 의료영상, 과학계산
    A,  // RMSE ≤ 0.001 (고품질) - 그래픽, 오디오 처리
    B,  // RMSE ≤ 0.01 (표준품질) - 일반 신경망 학습
    C,  // RMSE ≤ 0.1 (실용품질) - 실시간, 모바일 환경
}
```

### CompressionProfile

성능 프로파일 열거형입니다.

```rust
pub enum CompressionProfile {
    UltraHigh,   // 최고 품질, 느린 속도 (계수 4배)
    High,        // 고품질, 중간 속도 (계수 2배)
    Balanced,    // 균형 (예측 계수 그대로)
    Fast,        // 빠른 속도, 낮은 품질 (계수 1/2)
    UltraFast,   // 최고 속도, 최저 품질 (계수 1/4)
}
```

### TransformType

변환 타입 열거형입니다.

```rust
pub enum TransformType {
    Dct,      // Discrete Cosine Transform
    Dwt,      // Discrete Wavelet Transform (권장)
    Adaptive, // 데이터 특성에 따른 자동 선택
}
```

## 압축 데이터 구조

### HybridEncodedBlock

압축된 블록을 나타내는 구조체입니다.

```rust
pub struct HybridEncodedBlock {
    pub rbe_params: [f32; 8],              // RBE 기저 함수 파라미터
    pub residual_coefficients: Vec<ResidualCoefficient>, // 잔차 계수들
    pub transform_type: TransformType,      // 사용된 변환 타입
    pub original_rows: usize,              // 원본 행 수
    pub original_cols: usize,              // 원본 열 수
}
```

#### 메서드

```rust
impl HybridEncodedBlock {
    pub fn decode(&self) -> Vec<f32>       // 블록 복원
    pub fn get_compression_ratio(&self) -> f32 // 압축률 계산
    pub fn get_memory_usage(&self) -> usize    // 메모리 사용량
}
```

### ResidualCoefficient

잔차 계수를 나타내는 구조체입니다.

```rust
pub struct ResidualCoefficient {
    pub position: (usize, usize),         // 계수 위치
    pub value: f32,                       // 계수 값
    pub significance: f32,                // 중요도 점수
}
```

## 가중치 매핑 시스템

### WeightMapper

신경망 가중치의 압축과 메타데이터 관리를 담당합니다.

```rust
pub struct WeightMapper {
    pub encoder: RBEEncoder,              // 인코더 인스턴스
    pub layout: ModelLayout,              // 모델 레이아웃
    current_offset: u64,                  // 현재 오프셋
}
```

#### 핵심 메서드

```rust
impl WeightMapper {
    pub fn new(
        model_type: &str,
        block_size: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Self
    
    pub fn compress_weight(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Vec<HybridEncodedBlock>, String>
    
    pub fn get_layout(&self) -> &ModelLayout
    pub fn save_layout(&self, path: &str) -> Result<(), String>
}
```

### ModelLayout

모델 전체의 압축 레이아웃을 관리합니다.

```rust
pub struct ModelLayout {
    pub model_type: String,               // 모델 타입 (예: "gpt2")
    pub total_params: usize,              // 전체 파라미터 수
    pub total_blocks: usize,              // 전체 압축 블록 수
    pub weights: Vec<WeightInfo>,         // 가중치 정보 목록
    pub metadata: HashMap<String, String>, // 추가 메타데이터
    pub compression_config: CompressionMetadata, // 압축 설정
}
```

### WeightInfo

개별 가중치의 상세 정보입니다.

```rust
pub struct WeightInfo {
    pub name: String,                     // 가중치 이름
    pub offset_bytes: u64,                // 바이너리 오프셋
    pub num_blocks: usize,                // 블록 개수
    pub original_shape: Vec<usize>,       // 원본 텐서 모양
    pub compression_type: String,         // 압축 방식
    pub compression_ratio: f32,           // 압축률
    pub rmse: Option<f32>,               // 복원 품질
}
```

## 분석 도구

### FrequencyAnalysisResult

주파수 분석 결과를 나타냅니다.

```rust
pub struct FrequencyAnalysisResult {
    pub dominant_frequency: (usize, usize), // 주요 주파수
    pub max_energy: f32,                    // 최대 에너지
    pub total_energy: f32,                  // 전체 에너지
    pub frequency_type: FrequencyType,      // 주파수 타입
    pub normalized_frequencies: (f32, f32), // 정규화된 주파수
}
```

### FrequencyType

주파수 특성 분류입니다.

```rust
pub enum FrequencyType {
    LowFreqMonotonic,    // 저주파, 단조증가
    LowFreqSymmetric,    // 저주파, 대칭패턴
    HighFreqSaturated,   // 고주파, 포화패턴
    Localized,           // 국소화된 특징
}
```

## 성능 최적화

### A Matrix 캐싱

동일한 크기의 블록에 대해 A matrix를 재사용합니다.

```rust
static A_MATRIX_CACHE: Lazy<Arc<RwLock<HashMap<(usize, usize), Arc<DMatrix<f32>>>>>>
```

**기저 함수:**
```
A[i] = [1.0, d, d², cos(πx), cos(πy), cos(2πx), cos(2πy), cos(πx)cos(πy)]
```

### 병렬 처리

#### 블록별 병렬 압축

```rust
let encoded_blocks: Vec<HybridEncodedBlock> = (0..total_blocks)
    .into_par_iter()
    .map(|block_idx| {
        let mut local_encoder = RBEEncoder::new(coefficients, transform_type);
        local_encoder.encode_block(&block_data, block_size, block_size)
    })
    .collect();
```

#### 변환 병렬화

- **DCT**: 행별/열별 병렬 처리
- **DWT**: 웨이블릿 분해 병렬화

## 사용 예제

### 기본 사용법

```rust
use rbe_llm::encoder::{RBEEncoder, TransformType, QualityGrade};

// S급 품질 인코더 생성
let mut encoder = RBEEncoder::new_s_grade();

// 64x64 테스트 데이터
let test_data: Vec<f32> = (0..4096)
    .map(|i| ((i as f32) / 4096.0 * 2.0 * std::f32::consts::PI).sin())
    .collect();

// 블록 압축
let encoded = encoder.encode_block(&test_data, 64, 64);

// 압축 결과 확인
println!("압축률: {:.1}x", encoded.get_compression_ratio());
println!("메모리 사용량: {} bytes", encoded.get_memory_usage());

// 복원
let decoded = encoded.decode();
```

### 설정 기반 압축

```rust
use rbe_llm::encoder::{CompressionConfig, QualityGrade};

// 대용량 행렬 (512x1024)
let matrix_data = generate_test_matrix(512, 1024);

// UltraHigh 품질 설정
let config = CompressionConfig::ultra_high();

// 압축 실행
let result = RBEEncoder::compress_with_config(
    &matrix_data, 512, 1024, &config
);

match result {
    Ok((blocks, time, ratio, rmse)) => {
        println!("압축 완료: {}개 블록, {:.1}x 압축률", blocks.len(), ratio);
        println!("품질: RMSE {:.6}, 시간: {:.3}초", rmse, time);
    }
    Err(e) => println!("압축 실패: {}", e),
}
```

### 품질등급별 성능 비교

```rust
use rbe_llm::encoder::{RBEEncoder, QualityGrade, TransformType};

let test_data = generate_sine_pattern(64);
let grades = [QualityGrade::S, QualityGrade::A, QualityGrade::B, QualityGrade::C];

for grade in grades {
    let mut encoder = RBEEncoder::create_quality_encoder(
        &test_data, 64, 64, grade, TransformType::Dwt
    ).unwrap();
    
    let start = std::time::Instant::now();
    let encoded = encoder.encode_block(&test_data, 64, 64);
    let duration = start.elapsed();
    
    let decoded = encoded.decode();
    let rmse = calculate_rmse(&test_data, &decoded);
    
    println!("{:?}급: K={}, RMSE={:.6}, 시간={:?}", 
             grade, encoder.k_coeffs, rmse, duration);
}
```

## 오류 처리

### 일반적인 오류 상황

```rust
// RMSE 임계값 초과
"RMSE 임계값 초과: 0.010000 > 0.001000 (계수를 148에서 더 늘리거나 블록을 더 작게 하세요)"

// 압축률 미달
"압축률 임계값 미달: 150.0x < 200.0x (계수를 256에서 더 줄이거나 블록을 더 크게 하세요)"

// 블록 개수 부족
"블록 개수 부족: 실제 32개 < 최소 100개 (블록 크기를 64에서 더 작게 조정하세요)"
```

### 권장 해결책

1. **RMSE가 높을 때**: 계수 증가 또는 블록 크기 감소
2. **압축률이 낮을 때**: 계수 감소 또는 블록 크기 증가
3. **성능이 느릴 때**: Fast 프로파일 사용 또는 계수 감소

## 제약사항

### 수치적 제약

- **블록 크기**: 최소 16x16, 최대 1024x1024 권장
- **계수 개수**: 최소 8개, 최대 블록크기² 미만
- **RMSE 정확도**: f32 정밀도 범위 내

### 메모리 제약

- **A Matrix 캐시**: 큰 블록 크기에서 메모리 사용량 증가
- **병렬 처리**: 블록별 local_encoder로 메모리 사용량 배수 증가

## 버전 호환성

- **Rust 최소 버전**: 1.70.0
- **주요 의존성**: nalgebra, ndarray, rayon, rustdct, omni-wave
- **플랫폼 지원**: x86_64, ARM64

## 추가 참고 자료

- [RBE 수학적 기초](../math.md)
- [성능 벤치마크 보고서](../../test/encoder_report.md)
- [푸앵카레 볼 블록 크기 계수 최적화](../../paper/13_RBE_블록크기_계수_최적화_수학적_일반식.md) 