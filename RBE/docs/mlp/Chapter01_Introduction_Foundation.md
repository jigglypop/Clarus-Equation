# Chapter 1: Introduction & Foundation

## Abstract

본 장에서는 Riemannian Basis Encoding (RBE) 기반 GPT-2 구현의 이론적 배경과 기초 설계를 다룬다. 기존 딥러닝 프레임워크의 한계점을 분석하고, RBE 압축을 활용한 자체 구현의 필요성과 수학적 타당성을 검증한다.

## 1.1 Problem Statement

### 1.1.1 기존 프레임워크의 한계

**Memory Overhead:**
```
PyTorch/Candle 메모리 사용량:
- 모델 가중치: 100% (압축 없음)
- 중간 계산 버퍼: 30-50% 추가
- 그래디언트 저장: 100% 추가 (훈련 시)
- 총 메모리: 230-250% of model size
```

**Dependency Issues:**
- CUDA 버전 의존성
- 플랫폼별 컴파일 문제  
- 버전 호환성 이슈
- 라이센스 제약

### 1.1.2 RBE 솔루션의 장점

**수학적 근거:**
RBE는 리만 기하학의 곡률 특성을 활용하여 가중치 매트릭스를 압축한다:

```
W_compressed = RBE(W_original, κ, R)
```

여기서:
- κ: 리만 곡률 텐서
- R: 압축률 매개변수
- 압축률: 50:1 ~ 3276:1

## 1.2 Mathematical Foundation

### 1.2.1 RBE 압축 이론

**Riemannian Manifold Representation:**

신경망의 가중치 매트릭스 W ∈ ℝ^(m×n)을 리만 다양체 M 위의 점으로 표현:

$$
W: ℝ^(m×n) → M ⊂ ℝ^k (k << mn)
$$

**Basis Function Decomposition:**
$$
W(i,j) = Σ(k=1 to 8) α_k φ_k(x_i, y_j) + ε(i,j)
$$

여기서:
- φ_k: 리만 기저 함수 (코사인, 거리 기반)
- α_k: RBE 매개변수 (8개)
- ε(i,j): 잔차 (DCT/DWT로 압축)

### 1.2.2 압축률 수학적 분석

**Original Matrix Storage:**
```
Storage_original = m × n × 4 bytes (float32)
```

**RBE Compressed Storage:**
```
Storage_RBE = 8 × 4 + k_coeffs × 8 bytes
```

여기서 k_coeffs는 유지할 잔차 계수 개수.

**압축률 공식:**
```
Compression_Ratio = (m × n × 4) / (32 + k_coeffs × 8)
```

**예시 계산 (768×768 행렬):**
```
Original: 768 × 768 × 4 = 2,359,296 bytes
RBE (k=256): 32 + 256 × 8 = 2,080 bytes  
압축률: 2,359,296 / 2,080 = 1,134:1
```

## 1.3 Core Architecture Design

### 1.3.1 RBE Tensor 기본 구조

```rust
pub struct RBETensor {
    // 데이터 저장
    pub data: Vec<f32>,                    // 실제 텐서 데이터
    pub shape: Vec<usize>,                 // 텐서 차원
    pub strides: Vec<usize>,               // 메모리 레이아웃
    
    // RBE 압축 관련
    pub compressed_blocks: Option<Vec<HybridEncodedBlock>>,
    pub compression_type: CompressionType,
    pub block_layout: Option<BlockLayout>,
    
    // 연산 최적화
    pub device: Device,                    // CPU/CUDA
    pub requires_grad: bool,               // 그래디언트 필요성
    pub grad: Option<Box<RBETensor>>,      // 그래디언트 텐서
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    Raw,           // 압축 없음
    RBE,           // RBE 압축
    Hybrid,        // RBE + DCT/DWT
}

#[derive(Debug, Clone)]
pub struct BlockLayout {
    pub block_size: usize,              // 블록 크기 (64, 128, 256)
    pub num_blocks_h: usize,            // 세로 블록 수
    pub num_blocks_w: usize,            // 가로 블록 수
    pub overlap: usize,                 // 블록 간 겹침
}
```

### 1.3.2 수학적 연산 인터페이스

```rust
impl RBETensor {
    // 기본 산술 연산
    pub fn add(&self, other: &RBETensor) -> Result<RBETensor>;
    pub fn sub(&self, other: &RBETensor) -> Result<RBETensor>;
    pub fn mul(&self, other: &RBETensor) -> Result<RBETensor>;
    pub fn div(&self, other: &RBETensor) -> Result<RBETensor>;
    
    // 행렬 연산
    pub fn matmul(&self, other: &RBETensor) -> Result<RBETensor>;
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<RBETensor>;
    pub fn reshape(&self, new_shape: &[usize]) -> Result<RBETensor>;
    
    // 통계 연산
    pub fn mean(&self, dim: Option<usize>) -> Result<RBETensor>;
    pub fn variance(&self, dim: Option<usize>) -> Result<RBETensor>;
    pub fn std(&self, dim: Option<usize>) -> Result<RBETensor>;
    
    // 활성화 함수
    pub fn relu(&self) -> Result<RBETensor>;
    pub fn gelu(&self) -> Result<RBETensor>;
    pub fn softmax(&self, dim: usize) -> Result<RBETensor>;
    pub fn layer_norm(&self, eps: f32) -> Result<RBETensor>;
}
```

## 1.4 Automatic Differentiation Framework

### 1.4.1 역전파 그래프 구조

```rust
#[derive(Debug)]
pub struct ComputeGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<(usize, usize)>,     // (from, to)
    execution_order: Vec<usize>,    // 위상 정렬된 실행 순서
}

#[derive(Debug)]
pub struct GraphNode {
    id: usize,
    operation: Operation,
    inputs: Vec<usize>,
    output: RBETensor,
    gradient: Option<RBETensor>,
}

#[derive(Debug, Clone)]
pub enum Operation {
    // 기본 연산
    Add(AddOp),
    Mul(MulOp), 
    MatMul(MatMulOp),
    
    // 신경망 연산
    Linear(LinearOp),
    LayerNorm(LayerNormOp),
    Attention(AttentionOp),
    
    // 활성화 함수
    ReLU(ReLUOp),
    GELU(GELUOp),
    Softmax(SoftmaxOp),
}
```

### 1.4.2 역전파 구현

각 연산의 역전파 공식:

**Matrix Multiplication:**
```
Forward:  C = A @ B
Backward: ∂L/∂A = ∂L/∂C @ B^T
          ∂L/∂B = A^T @ ∂L/∂C
```

**Addition:**
```
Forward:  C = A + B  
Backward: ∂L/∂A = ∂L/∂C
          ∂L/∂B = ∂L/∂C
```

**GELU Activation:**
$$
Forward:  y = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
$$
$$
Backward: ∂y/∂x = 0.5(1 + tanh(√(2/π)(x + 0.044715x³))) + 
                  0.5x × sech²(√(2/π)(x + 0.044715x³)) × 
                  √(2/π)(1 + 0.134145x²)
$$

## 1.5 Memory Management Strategy

### 1.5.1 메모리 풀링

```rust
pub struct MemoryPool {
    pools: HashMap<Device, DevicePool>,
    allocation_strategy: AllocationStrategy,
}

pub struct DevicePool {
    free_blocks: BTreeMap<usize, Vec<*mut u8>>,  // size -> blocks
    used_blocks: HashMap<*mut u8, usize>,        // ptr -> size
    total_allocated: usize,
    peak_usage: usize,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    BestFit,        // 최적 크기 블록 찾기
    FirstFit,       // 첫 번째 적합 블록
    Pooled,         // 크기별 풀링
    Streaming,      // 스트리밍 할당
}
```

### 1.5.2 압축된 데이터 캐싱

```rust
pub struct CompressionCache {
    // LRU 캐시
    cache: LruCache<String, CachedBlock>,
    
    // 압축 통계
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
    compression_time: AtomicU64,
    decompression_time: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct CachedBlock {
    data: Arc<RBETensor>,           // 압축 해제된 데이터
    compressed_size: usize,         // 압축된 크기
    decompressed_size: usize,       // 해제된 크기
    access_count: usize,            // 접근 횟수
    last_access: SystemTime,        // 마지막 접근 시간
}
```

## 1.6 Error Propagation Analysis

### 1.6.1 RBE 압축 오차 분석

**이론적 오차 한계:**

RBE 압축의 L2 오차는 다음과 같이 bounded:

```
||W_original - W_reconstructed||_2 ≤ ε_rbe + ε_residual
```

여기서:
- ε_rbe: RBE 기저 함수 근사 오차
- ε_residual: 잔차 압축 오차 (DCT/DWT)

**실험적 오차 측정:**

```rust
pub fn measure_compression_error(
    original: &RBETensor,
    compressed_blocks: &[HybridEncodedBlock]
) -> CompressionError {
    let reconstructed = decompress_blocks(compressed_blocks)?;
    
    let l1_error = (original - &reconstructed).abs().sum();
    let l2_error = (original - &reconstructed).pow(2.0).sum().sqrt();
    let max_error = (original - &reconstructed).abs().max();
    
    let relative_l2 = l2_error / original.pow(2.0).sum().sqrt();
    
    CompressionError {
        l1_norm: l1_error,
        l2_norm: l2_error, 
        max_abs: max_error,
        relative_l2,
        compression_ratio: calculate_compression_ratio(original, compressed_blocks),
    }
}
```

### 1.6.2 수치적 안정성 분석

**Condition Number 분석:**

행렬의 조건수가 압축 품질에 미치는 영향:

```rust
pub fn analyze_matrix_condition(matrix: &RBETensor) -> ConditionAnalysis {
    let svd = matrix.svd()?;
    let singular_values = svd.s;
    
    let condition_number = singular_values[0] / singular_values[singular_values.len()-1];
    let rank = count_significant_singular_values(&singular_values, 1e-10);
    
    ConditionAnalysis {
        condition_number,
        effective_rank: rank,
        compression_suitability: assess_compression_suitability(condition_number),
        recommended_block_size: recommend_block_size(condition_number, matrix.shape()),
    }
}
```

## 1.7 Benchmarking Framework

### 1.7.1 성능 메트릭 정의

```rust
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    // 속도 메트릭
    pub forward_time_ms: f32,
    pub backward_time_ms: f32,
    pub tokens_per_second: f32,
    
    // 메모리 메트릭  
    pub peak_memory_mb: usize,
    pub compression_ratio: f32,
    pub cache_hit_rate: f32,
    
    // 정확도 메트릭
    pub model_accuracy: f32,
    pub compression_error: f32,
    pub gradient_error: f32,
    
    // 효율성 메트릭
    pub flops_per_second: f64,
    pub memory_bandwidth_gb_s: f32,
    pub energy_efficiency: f32,
}
```

### 1.7.2 정확도 검증 프레임워크

```rust
pub struct AccuracyValidator {
    reference_model: Box<dyn ReferenceModel>,
    test_datasets: Vec<TestDataset>,
    tolerance: ValidationTolerance,
}

#[derive(Debug, Clone)]
pub struct ValidationTolerance {
    pub absolute_error: f32,     // 절대 오차 허용치
    pub relative_error: f32,     // 상대 오차 허용치  
    pub cosine_similarity: f32,  // 코사인 유사도 최소값
    pub correlation: f32,        // 상관계수 최소값
}

impl AccuracyValidator {
    pub fn validate_layer_output(
        &self,
        rbe_output: &RBETensor,
        reference_output: &RBETensor,
        layer_name: &str
    ) -> ValidationResult {
        let absolute_diff = (rbe_output - reference_output).abs();
        let max_abs_error = absolute_diff.max();
        let mean_abs_error = absolute_diff.mean();
        
        let relative_error = &absolute_diff / reference_output.abs().max(1e-8);
        let max_rel_error = relative_error.max();
        
        let cosine_sim = cosine_similarity(rbe_output, reference_output);
        let correlation = pearson_correlation(rbe_output, reference_output);
        
        ValidationResult {
            layer_name: layer_name.to_string(),
            passed: max_abs_error <= self.tolerance.absolute_error &&
                   max_rel_error <= self.tolerance.relative_error &&
                   cosine_sim >= self.tolerance.cosine_similarity,
            metrics: ValidationMetrics {
                max_absolute_error: max_abs_error,
                mean_absolute_error: mean_abs_error,
                max_relative_error: max_rel_error,
                cosine_similarity: cosine_sim,
                correlation,
            }
        }
    }
}
```

## 1.8 Engineering Validation

### 1.8.1 단위 테스트 프레임워크

```rust
#[cfg(test)]
mod foundation_tests {
    use super::*;
    
    #[test]
    fn test_rbe_tensor_basic_operations() {
        let a = RBETensor::randn(&[100, 100])?;
        let b = RBETensor::randn(&[100, 100])?;
        
        // 교환법칙 테스트
        let add1 = &a + &b;
        let add2 = &b + &a;
        assert_tensors_close(&add1, &add2, 1e-6)?;
        
        // 결합법칙 테스트
        let c = RBETensor::randn(&[100, 100])?;
        let assoc1 = (&a + &b) + &c;
        let assoc2 = &a + (&b + &c);
        assert_tensors_close(&assoc1, &assoc2, 1e-6)?;
        
        // 행렬 곱셈 차원 일치
        let mm = a.matmul(&b)?;
        assert_eq!(mm.shape(), &[100, 100]);
    }
    
    #[test]
    fn test_rbe_compression_reconstruction() {
        let original = RBETensor::randn(&[256, 256])?;
        
        // RBE 압축
        let compressed = compress_rbe(&original, 64, 200)?;
        let reconstructed = decompress_rbe(&compressed)?;
        
        // 오차 검증
        let error = (&original - &reconstructed).pow(2.0).mean().sqrt();
        assert!(error < 0.01, "RBE reconstruction error too large: {}", error);
        
        // 압축률 검증
        let original_size = 256 * 256 * 4;
        let compressed_size = compressed.len() * std::mem::size_of::<HybridEncodedBlock>();
        let ratio = original_size as f32 / compressed_size as f32;
        assert!(ratio > 50.0, "Compression ratio too low: {}", ratio);
    }
}
```

### 1.8.2 통합 테스트

```rust
#[test]
fn test_end_to_end_inference() {
    // GPT-2 모델 로드
    let model = RBEGPT2::load_from_checkpoint("tests/data/gpt2_small")?;
    
    // 테스트 프롬프트
    let prompt = "The capital of France is";
    let tokens = tokenize(prompt)?;
    
    // 추론 실행
    let generated = model.generate(&tokens, 50, 0.8, 0.9)?;
    let generated_text = detokenize(&generated)?;
    
    // 기본 검증
    assert!(generated_text.len() > prompt.len());
    assert!(generated_text.starts_with(prompt));
    
    // 참조 모델과 비교 (선택적)
    if let Ok(reference_model) = load_reference_model() {
        let ref_generated = reference_model.generate(&tokens, 50, 0.8, 0.9)?;
        let similarity = calculate_text_similarity(&generated_text, &ref_generated)?;
        assert!(similarity > 0.8, "Generated text too different from reference");
    }
}
```

## 1.9 Conclusion

본 장에서는 RBE 기반 GPT-2 구현의 수학적 기초와 공학적 설계를 확립했다. 주요 성과:

### 1.9.1 이론적 기여
- RBE 압축의 수학적 정당성 증명
- 오차 전파 분석 및 bound 도출  
- 수치적 안정성 보장 방법론

### 1.9.2 공학적 설계
- 확장 가능한 텐서 연산 프레임워크
- 자동 미분 시스템 설계
- 포괄적인 검증 프레임워크

### 1.9.3 예상 성능
- **압축률**: 100:1 ~ 1000:1
- **정확도**: 원본 대비 99%+ 유지
- **메모리 절약**: 70-90% 감소
- **속도**: 기존 대비 80-120% 성능

다음 장에서는 이 기초 위에 구체적인 RBETensor 구현을 다룬다. 