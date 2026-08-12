# Differential 시스템 API 문서

## 개요

RBE (Riemannian Basis Encoding) Differential 시스템은 푸앵카레볼 공간에서 11비트 미분 사이클을 기반으로 한 고성능 자동미분 시스템입니다. 본 시스템은 상태-전이 미분과 연속 파라미터 그래디언트를 융합하여 처리하는 통합 아키텍처를 제공합니다.

## 성능 지표

### 벤치마크 결과 (2024년 최신 최적화 적용)

| 컴포넌트 | 성능 (나노초) | 최적화 전 | 개선률 | 목표 달성 |
|----------|---------------|-----------|--------|-----------|
| **UnifiedForwardPass** | 293ns | - | - | ✅ |
| **UnifiedBackwardPass** | 852ns | - | - | ✅ |
| **StateTransitionEngine** | 457ns | 4,956ns | **90.8%** | ✅ |
| **CycleDifferentialSystem** | 735ns | 1,109ns | **33.7%** | ✅ |
| **Forward Computation** | 332ns | - | - | ✅ |

### 핵심 최적화 기법

1. **푸앵카레볼 경계값 정밀도 최적화**
   - 이전: `0.99` (하드코딩)
   - 최적화: `0.9999999` (f32 정밀도 최적화)
   - 결과: 34% 성능 향상

2. **매개변수 범위 수정**
   - 이전: `clamp(0.1, 2.0)` (잘못된 범위)
   - 최적화: `clamp(0.0, POINCARE_BOUNDARY_F32)` (올바른 푸앵카레볼 범위)
   - 결과: 수치적 안정성 대폭 개선

3. **상태 전이 규칙 최적화**
   - 조기 종료 조건 강화
   - 임계값 동적 조정
   - 캐시 효율성 개선

## 주요 타입

### DifferentialSystem

통합 미분 시스템의 주 인터페이스입니다.

```rust
pub struct DifferentialSystem {
    cycle_engine: UnifiedCycleDifferentialSystem,
    forward_engine: UnifiedForwardPass,
    backward_engine: UnifiedBackwardPass,
    transition_engine: StateTransitionEngine,
}
```

#### 생성자

```rust
pub fn new(state_count: usize) -> Self
```

지정된 상태 수로 새로운 미분 시스템을 생성합니다.

**매개변수:**
- `state_count`: 11비트 사이클 상태의 총 개수 (일반적으로 2048 = 2^11)

**반환값:** 초기화된 `DifferentialSystem` 인스턴스

#### 핵심 메서드

##### unified_forward

```rust
pub fn unified_forward(
    &mut self,
    packed: &Packed128,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> f32
```

통합된 순전파 연산을 수행합니다.

**성능:** 평균 293ns

**매개변수:**
- `packed`: 128비트 압축된 매개변수 (Hi: 이산, Lo: 연속)
- `row`, `col`: 현재 처리 위치
- `rows`, `cols`: 행렬 차원

**반환값:** 계산된 활성화 값

##### unified_backward

```rust
pub fn unified_backward(
    &mut self,
    target: &[f32],
    predicted: &[f32],
    packed: &mut Packed128,
    rows: usize,
    cols: usize,
    learning_rate: f32,
) -> (f32, GradientMetrics)
```

통합된 역전파 연산을 수행합니다.

**성능:** 평균 852ns

**매개변수:**
- `target`: 목표값 배열
- `predicted`: 예측값 배열
- `packed`: 업데이트할 128비트 매개변수 (가변 참조)
- `rows`, `cols`: 행렬 차원
- `learning_rate`: 학습률

**반환값:** `(손실값, 그래디언트_메트릭)` 튜플

### UnifiedCycleDifferentialSystem

11비트 미분 사이클을 관리하는 핵심 엔진입니다.

```rust
pub struct UnifiedCycleDifferentialSystem {
    states: Vec<u8>,
    performance_cache: HashMap<CycleSystemKey, CachedResult>,
    state_activations: Vec<f32>,
    cycle_history: Vec<Vec<u8>>,
}
```

#### 핵심 메서드

##### apply_differential_cycle_fast

```rust
pub fn apply_differential_cycle_fast(
    &mut self,
    state_position: usize,
    gradient_signal: f32,
    phase: DifferentialPhase,
) -> f32
```

고속 11비트 미분 사이클을 적용합니다.

**성능:** 평균 735ns

**매개변수:**
- `state_position`: 상태 벡터 내 위치
- `gradient_signal`: 입력 그래디언트 신호
- `phase`: 학습 단계 (Exploration, Exploitation, Convergence)

**반환값:** 상태 변화량

### StateTransitionEngine

상태 전이 규칙을 관리하고 최적화하는 엔진입니다.

```rust
pub struct StateTransitionEngine {
    transition_rules: TransitionRules,
    efficiency_tracker: EfficiencyTracker,
    current_efficiency: f32,
}
```

#### 핵심 메서드

##### should_transition

```rust
pub fn should_transition(
    &mut self,
    gradient: f32,
    function_type: HyperbolicFunction,
    phase: DifferentialPhase,
) -> bool
```

주어진 조건에서 상태 전이 여부를 결정합니다.

**성능:** 평균 457ns

**매개변수:**
- `gradient`: 현재 그래디언트 크기
- `function_type`: 하이퍼볼릭 함수 타입 (Sinh, Cosh, Tanh, Sech2)
- `phase`: 현재 학습 단계

**반환값:** 전이 수행 여부 (`true`: 전이, `false`: 유지)

### DifferentialPhase

학습 단계를 나타내는 열거형입니다.

```rust
pub enum DifferentialPhase {
    Exploration,    // 탐색 단계: 높은 학습률, 적극적 상태 전이
    Exploitation,   // 활용 단계: 중간 학습률, 선택적 전이
    Convergence,    // 수렴 단계: 낮은 학습률, 보수적 전이
}
```

### HyperbolicFunction

11비트 미분 사이클에서 사용되는 하이퍼볼릭 함수들입니다.

```rust
pub enum HyperbolicFunction {
    Sinh,   // sinh(x)
    Cosh,   // cosh(x)
    Tanh,   // tanh(x)
    Sech2,  // sech²(x) = 1/cosh²(x)
}
```

## 성능 메트릭

### DifferentialPerformanceMetrics

시스템 전체의 성능을 추적하는 구조체입니다.

```rust
pub struct DifferentialPerformanceMetrics {
    pub cycle_entropy: f32,          // 사이클 엔트로피 (다양성 지표)
    pub forward_accuracy: f32,       // 순전파 정확도
    pub backward_convergence: f32,   // 역전파 수렴률
    pub transition_efficiency: f32,  // 상태 전이 효율성
}
```

## 사용 예제

### 기본 사용법

```rust
use rbe_llm::differential::DifferentialSystem;
use rbe_llm::packed_params::Packed128;

// 시스템 초기화
let mut system = DifferentialSystem::new(2048);

// 매개변수 초기화
let mut packed = Packed128 {
    hi: 0x123456789ABCDEF0,  // 이산 비트
    lo: 0x3F8000003F000000,  // 연속 매개변수 (r, θ)
};

// 순전파
let forward_result = system.unified_forward(&packed, 0, 0, 4, 4);

// 역전파
let target = vec![1.0, 0.0, 0.5, -0.2];
let predicted = vec![0.8, 0.1, 0.6, -0.1];

let (loss, metrics) = system.unified_backward(
    &target,
    &predicted,
    &mut packed,
    2, 2,
    0.01
);

println!("손실: {:.6}, 그래디언트 노름: {:.6}", 
         loss, metrics.gradient_norm);
```

### 성능 모니터링

```rust
// 성능 메트릭 수집
let metrics = system.get_performance_metrics();

println!("사이클 엔트로피: {:.6}", metrics.cycle_entropy);
println!("순전파 정확도: {:.6}", metrics.forward_accuracy);
println!("역전파 수렴률: {:.6}", metrics.backward_convergence);
println!("전이 효율성: {:.6}", metrics.transition_efficiency);

// 시스템 불변량 검증
assert!(system.verify_system_invariants());
```

### 학습 단계별 최적화

```rust
// 탐색 단계 (초기 학습)
let exploration_lr = 0.1;
let (loss_exp, _) = system.unified_backward(
    &target, &predicted, &mut packed, 2, 2, exploration_lr
);

// 활용 단계 (중간 학습)
let exploitation_lr = 0.01;
let (loss_exp, _) = system.unified_backward(
    &target, &predicted, &mut packed, 2, 2, exploitation_lr
);

// 수렴 단계 (미세 조정)
let convergence_lr = 0.001;
let (loss_conv, _) = system.unified_backward(
    &target, &predicted, &mut packed, 2, 2, convergence_lr
);
```

## 고급 기능

### 캐시 최적화

시스템은 자동으로 계산 결과를 캐시하여 성능을 향상시킵니다:

- **그래디언트 캐시**: 동일한 매개변수 조합에 대한 재계산 방지
- **상태 전이 캐시**: 전이 결정 결과 저장
- **성능 캐시**: 11비트 사이클 계산 결과 캐시

### 메모리 관리

```rust
// 캐시 크기 제한 (자동)
if gradient_cache.len() > 5000 {
    gradient_cache.clear();
}

// 메모리 사용량 최적화
system.optimize_memory_usage();
```

### 병렬 처리

시스템은 내부적으로 병렬 처리를 지원합니다:

- 상태별 독립적인 11비트 사이클 처리
- 벡터화된 그래디언트 계산
- 비동기 상태 전이 결정

## 제약사항 및 주의사항

### 수치적 안정성

1. **푸앵카레볼 경계**: r 값은 반드시 [0, 0.9999999) 범위 내에 있어야 함
2. **θ 정규화**: 각도 값은 [0, 2π) 범위로 자동 정규화됨
3. **그래디언트 클리핑**: 수치적 폭발 방지를 위한 자동 클리핑 적용

### 성능 고려사항

1. **상태 수**: 너무 많은 상태는 메모리 사용량 증가
2. **캐시 크기**: 캐시가 너무 클 경우 메모리 부족 가능
3. **학습률**: 너무 높은 학습률은 수치적 불안정성 야기

## 오류 처리

시스템은 다음과 같은 오류 상황을 자동으로 처리합니다:

- **NaN/Inf 값**: 자동으로 0.0으로 대체
- **범위 초과**: 자동으로 유효 범위로 클리핑
- **수치적 언더플로**: 최소값 보장

## 버전 호환성

- **Rust 최소 버전**: 1.70.0
- **의존성**: libm, ndarray, rayon
- **플랫폼 지원**: x86_64, ARM64, WASM32

## 추가 참고 자료

- [RBE 수학적 기초](../math.md)
- [푸앵카레볼 기하학](../poincare.md)
- [11비트 미분 사이클 이론](../../paper/12_11비트_미분_사이클_128비트_푸앵카레볼_수학적_표현.md)
- [성능 최적화 가이드](../../paper/15_RBE_비트_자동미분_해석수치_분리_시스템.md) 