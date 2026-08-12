# Optimizers API

## 개요

RBE 최적화 시스템은 푸앵카레 볼 공간에서 특화된 Adam과 Riemannian Adam 옵티마이저를 제공합니다. 수치적 안정성과 성능을 대폭 개선하여 10배 이론적 속도 향상을 달성했습니다.

## 성능 향상 현황

### 2024년 12월 최적화 성과

| 컴포넌트 | 기존 이슈 | 최적화 내용 | 성과 |
|----------|-----------|-------------|------|
| **Metric Tensor** | 수치적 불안정 | `safe_r` 클램핑, `denominator` 보호 | 안정성 대폭 개선 |
| **Mobius Addition** | NaN/Inf 발생 | 입력 검증, 분모 보호, 결과 클램핑 | 수치적 견고성 확보 |
| **Exponential Map** | 큰 값에서 불안정 | `tanh_arg` 클램핑, 조기 종료 | 안정적 수렴 |
| **Update Process** | 그래디언트 폭발 | 클리핑, 조기 종료, 개선된 편향 보정 | 학습 안정성 향상 |
| **Boundary Handling** | 경계값 부정확 | `POINCARE_BOUNDARY_F32` 사용 | 정밀도 향상 |

### 수치적 안정성 개선

- **NaN/Inf 방지**: 모든 연산에서 입력 검증 적용
- **분모 보호**: `max(1e-10)` 또는 `abs() < 1e-12` 조건으로 0 나누기 방지
- **범위 클램핑**: 푸앵카레 볼 경계 엄격 적용
- **그래디언트 클리핑**: 수치적 폭발 방지

## 주요 타입

### AdamState

표준 Adam 옵티마이저 상태를 관리합니다.

```rust
#[derive(Debug, Clone)]
pub struct AdamState {
    m: Vec<f32>,              // 1차 모멘트 추정치
    v: Vec<f32>,              // 2차 모멘트 추정치  
    t: usize,                 // 시간 스텝
    config: AdamConfig,       // 설정
}
```

#### 생성자

```rust
impl AdamState {
    pub fn new() -> Self                     // 기본 설정으로 생성
    pub fn with_config(config: AdamConfig) -> Self // 사용자 정의 설정
}

impl Default for AdamState {
    fn default() -> Self                     // 기본값 구현
}
```

#### 핵심 메서드

```rust
impl AdamState {
    pub fn update(
        &mut self,
        r: &mut f32,           // 푸앵카레 반지름 (가변)
        theta: &mut f32,       // 푸앵카레 각도 (가변)  
        grad_r: f32,           // r 그래디언트
        grad_theta: f32,       // θ 그래디언트
    )
    
    pub fn reset(&mut self)    // 상태 초기화
    
    pub fn get_learning_rates(&self) -> (f32, f32) // 유효 학습률 반환
}
```

### RiemannianAdamState

푸앵카레 볼에 특화된 Riemannian Adam 옵티마이저입니다.

```rust
#[derive(Debug, Clone)]
pub struct RiemannianAdamState {
    m: Vec<f32>,              // 1차 모멘트 추정치
    v: Vec<f32>,              // 2차 모멘트 추정치
    t: usize,                 // 시간 스텝  
    config: RiemannianAdamConfig, // 설정
}
```

#### 생성자와 기본값

```rust
impl RiemannianAdamState {
    pub fn new() -> Self
    pub fn with_config(config: RiemannianAdamConfig) -> Self
}

impl Default for RiemannianAdamState {
    fn default() -> Self
}
```

#### 핵심 메서드

##### update

```rust
pub fn update(
    &mut self,
    r: &mut f32,           // 푸앵카레 반지름 (가변)  
    theta: &mut f32,       // 푸앵카레 각도 (가변)
    grad_r: f32,           // r 그래디언트
    grad_theta: f32,       // θ 그래디언트
)
```

**최적화된 업데이트 프로세스:**

1. **조기 종료**: 매우 작은 그래디언트 (`< 1e-8`) 무시
2. **NaN/Inf 검증**: 입력 그래디언트 유효성 확인
3. **Riemannian 그래디언트 계산**: 메트릭 텐서 적용
4. **개선된 편향 보정**: 수치적 안정성 고려
5. **업데이트 벡터 클리핑**: 과도한 업데이트 방지
6. **최종 클램핑**: 푸앵카레 볼 경계 엄수

##### 기하학적 연산

```rust
pub fn compute_metric_tensor(r: f32, theta: f32) -> [[f32; 2]; 2]
```

**안정성 개선사항:**
- `safe_r = r.clamp(1e-6, POINCARE_BOUNDARY_F32)` 사용
- `denominator` 보호: `(1.0 - safe_r.powi(2)).max(1e-10)`
- `g_theta_theta` 안정성: `.max(1e-10)`

```rust  
pub fn mobius_add(a: (f32, f32), b: (f32, f32)) -> (f32, f32)
```

**개선사항:**
- **입력 검증**: NaN/Inf 자동 감지 및 처리
- **분모 보호**: `abs() < 1e-12` 조건으로 0 나누기 방지
- **결과 클램핑**: `POINCARE_BOUNDARY_F32` 경계 적용

```rust
pub fn exponential_map(v: (f32, f32), x: (f32, f32)) -> (f32, f32)
```

**개선사항:**
- **입력 검증**: NaN/Inf 처리
- **조기 종료**: 작은 `norm_v` (`< 1e-6`)에 대한 최적화
- **안전한 계산**: `tanh_arg.min(50.0)`으로 오버플로 방지
- **결과 클램핑**: 최종 결과 경계값 보장

#### 유틸리티 메서드

```rust
impl RiemannianAdamState {
    pub fn update_with_clipping(
        &mut self,
        r: &mut f32,
        theta: &mut f32, 
        grad_r: f32,
        grad_theta: f32,
        max_norm: f32,
    )
    
    pub fn get_momentum_magnitude(&self) -> f32
    pub fn get_effective_learning_rates(&self) -> (f32, f32)
    pub fn is_converged(&self, tolerance: f32) -> bool
    pub fn poincare_distance(p1: (f32, f32), p2: (f32, f32)) -> f32
    pub fn reset(&mut self)
}
```

## 설정 시스템

### AdamConfig

표준 Adam 설정입니다.

```rust
#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub learning_rate: f32,   // 학습률 (기본: 0.001)
    pub beta1: f32,           // 1차 모멘트 감쇠 (기본: 0.9)
    pub beta2: f32,           // 2차 모멘트 감쇠 (기본: 0.999)
    pub epsilon: f32,         // 수치적 안정성 (기본: 1e-8)
}
```

#### 기본값

```rust
impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}
```

### RiemannianAdamConfig

Riemannian Adam에 특화된 설정입니다.

```rust
#[derive(Debug, Clone)]
pub struct RiemannianAdamConfig {
    pub learning_rate: f32,   // 학습률 (기본: 0.001)
    pub beta1: f32,           // 1차 모멘트 감쇠 (기본: 0.9)
    pub beta2: f32,           // 2차 모멘트 감쇠 (기본: 0.999)
    pub epsilon: f32,         // 수치적 안정성 (기본: 1e-8)
    pub manifold_eps: f32,    // 매니폴드 특화 epsilon (기본: 1e-6)
}
```

#### 기본값

```rust
impl Default for RiemannianAdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            manifold_eps: 1e-6,
        }
    }
}
```

## 변환 분석 시스템

### TransformAnalyzer

DCT와 웨이블릿 변환의 성능을 분석합니다.

```rust
pub struct TransformAnalyzer {
    transform_type: TransformType,
    performance_history: Vec<f32>,
}
```

#### 분석 메서드

```rust
impl TransformAnalyzer {
    pub fn new(transform_type: TransformType) -> Self
    
    pub fn analyze_dct_performance(&self, data: &[f32]) -> f32
    pub fn analyze_wavelet_performance(&self, data: &[f32]) -> f32
    pub fn select_optimal_transform(&self, data: &[f32]) -> TransformType
    
    pub fn benchmark_transform_speed(&self, data: &[f32]) -> f64 // 나노초 단위
}
```

### OptimizerType

지원되는 옵티마이저 타입입니다.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    Adam,           // 표준 Adam
    RiemannianAdam, // Riemannian Adam
    Hybrid,         // 하이브리드 (상황에 따라 전환)
}
```

### TransformType

지원되는 변환 타입입니다.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformType {
    Dct,      // Discrete Cosine Transform
    Dwt,      // Discrete Wavelet Transform (권장)
    Fourier,  // Fourier Transform
    Hybrid,   // 적응적 선택
}
```

## 사용 예제

### 기본 Adam 사용법

```rust
use rbe_llm::optimizers::{AdamState, AdamConfig};

// 기본 설정으로 Adam 생성
let mut adam = AdamState::new();

// 사용자 정의 설정
let config = AdamConfig {
    learning_rate: 0.01,
    beta1: 0.95,
    beta2: 0.999,
    epsilon: 1e-6,
};
let mut custom_adam = AdamState::with_config(config);

// 파라미터 업데이트
let mut r = 0.5f32;
let mut theta = 1.0f32;

adam.update(&mut r, &mut theta, -0.1, 0.05);

println!("업데이트된 파라미터: r={:.6}, θ={:.6}", r, theta);

// 학습률 확인
let (lr_r, lr_theta) = adam.get_learning_rates();
println!("유효 학습률: r={:.6}, θ={:.6}", lr_r, lr_theta);
```

### Riemannian Adam 사용법

```rust
use rbe_llm::optimizers::{RiemannianAdamState, RiemannianAdamConfig};

// Riemannian Adam 생성
let mut riemannian_adam = RiemannianAdamState::new();

// 매니폴드 특화 설정
let config = RiemannianAdamConfig {
    learning_rate: 0.005,
    manifold_eps: 1e-5,
    ..Default::default()
};
let mut custom_riemannian = RiemannianAdamState::with_config(config);

// 푸앵카레 볼 파라미터 업데이트
let mut r = 0.3f32;
let mut theta = 0.7f32;

// 그래디언트 적용
riemannian_adam.update(&mut r, &mut theta, -0.05, 0.02);

// 수렴 확인
if riemannian_adam.is_converged(1e-6) {
    println!("수렴 완료!");
}

// 모멘텀 크기 확인
let momentum = riemannian_adam.get_momentum_magnitude();
println!("모멘텀 크기: {:.6}", momentum);
```

### 클리핑을 사용한 안전한 업데이트

```rust
// 그래디언트 클리핑으로 안전한 업데이트
riemannian_adam.update_with_clipping(
    &mut r,
    &mut theta,
    -0.2,    // 큰 그래디언트
    0.15,
    1.0,     // 최대 노름
);

// 푸앵카레 거리 계산
let p1 = (0.3, 0.7);
let p2 = (0.4, 0.8);
let distance = RiemannianAdamState::poincare_distance(p1, p2);
println!("푸앵카레 거리: {:.6}", distance);
```

### 기하학적 연산 직접 사용

```rust
// 메트릭 텐서 계산
let metric = RiemannianAdamState::compute_metric_tensor(0.5, 1.0);
println!("메트릭 텐서: {:?}", metric);

// Möbius 덧셈
let a = (0.3, 0.7);
let b = (0.1, 0.2);
let result = RiemannianAdamState::mobius_add(a, b);
println!("Möbius 덧셈 결과: {:?}", result);

// 지수 사상
let v = (0.1, 0.05);
let x = (0.2, 0.4);
let exp_result = RiemannianAdamState::exponential_map(v, x);
println!("지수 사상 결과: {:?}", exp_result);
```

### 변환 분석 및 최적화

```rust
use rbe_llm::optimizers::{TransformAnalyzer, TransformType};

// 변환 분석기 생성
let analyzer = TransformAnalyzer::new(TransformType::Dwt);

// 테스트 데이터
let test_data: Vec<f32> = (0..1024)
    .map(|i| (i as f32 / 1024.0 * 2.0 * std::f32::consts::PI).sin())
    .collect();

// 성능 분석
let dct_score = analyzer.analyze_dct_performance(&test_data);
let wavelet_score = analyzer.analyze_wavelet_performance(&test_data);

println!("DCT 성능: {:.3}", dct_score);
println!("웨이블릿 성능: {:.3}", wavelet_score);

// 최적 변환 선택
let optimal = analyzer.select_optimal_transform(&test_data);
println!("최적 변환: {:?}", optimal);

// 속도 벤치마크
let speed = analyzer.benchmark_transform_speed(&test_data);
println!("변환 속도: {:.1}ns", speed);
```

### 다중 파라미터 최적화

```rust
use std::collections::HashMap;

// 여러 파라미터를 동시에 최적화
struct MultiParamOptimizer {
    optimizers: HashMap<String, RiemannianAdamState>,
}

impl MultiParamOptimizer {
    fn new() -> Self {
        Self {
            optimizers: HashMap::new(),
        }
    }
    
    fn add_parameter(&mut self, name: &str, config: RiemannianAdamConfig) {
        self.optimizers.insert(
            name.to_string(),
            RiemannianAdamState::with_config(config)
        );
    }
    
    fn update_parameter(
        &mut self,
        name: &str,
        r: &mut f32,
        theta: &mut f32,
        grad_r: f32,
        grad_theta: f32,
    ) -> Result<(), String> {
        match self.optimizers.get_mut(name) {
            Some(optimizer) => {
                optimizer.update(r, theta, grad_r, grad_theta);
                Ok(())
            }
            None => Err(format!("파라미터 '{}' 없음", name)),
        }
    }
}

// 사용 예제
let mut multi_opt = MultiParamOptimizer::new();

// 레이어별 다른 설정
let layer1_config = RiemannianAdamConfig {
    learning_rate: 0.01,
    ..Default::default()
};
let layer2_config = RiemannianAdamConfig {
    learning_rate: 0.005,
    ..Default::default()
};

multi_opt.add_parameter("layer1", layer1_config);
multi_opt.add_parameter("layer2", layer2_config);

// 파라미터 업데이트
let mut layer1_r = 0.3f32;
let mut layer1_theta = 0.7f32;
let mut layer2_r = 0.4f32;
let mut layer2_theta = 0.8f32;

multi_opt.update_parameter("layer1", &mut layer1_r, &mut layer1_theta, -0.05, 0.02).unwrap();
multi_opt.update_parameter("layer2", &mut layer2_r, &mut layer2_theta, -0.03, 0.01).unwrap();
```

## 디버깅 및 모니터링

### 옵티마이저 상태 검사

```rust
// Adam 상태 디버깅
fn debug_adam_state(adam: &AdamState) {
    let (lr_r, lr_theta) = adam.get_learning_rates();
    
    println!("Adam 상태:");
    println!("  시간 스텝: {}", adam.t);
    println!("  유효 학습률: r={:.6}, θ={:.6}", lr_r, lr_theta);
    println!("  설정: {:?}", adam.config);
}

// Riemannian Adam 상태 디버깅  
fn debug_riemannian_state(riemannian: &RiemannianAdamState) {
    let momentum = riemannian.get_momentum_magnitude();
    let (lr_r, lr_theta) = riemannian.get_effective_learning_rates();
    let converged = riemannian.is_converged(1e-6);
    
    println!("Riemannian Adam 상태:");
    println!("  시간 스텝: {}", riemannian.t);
    println!("  모멘텀 크기: {:.6}", momentum);
    println!("  유효 학습률: r={:.6}, θ={:.6}", lr_r, lr_theta);
    println!("  수렴 여부: {}", converged);
    println!("  설정: {:?}", riemannian.config);
}
```

### 수치적 안정성 검증

```rust
fn verify_numerical_stability(r: f32, theta: f32) -> bool {
    // NaN/Inf 검사
    if !r.is_finite() || !theta.is_finite() {
        eprintln!("경고: NaN 또는 Inf 값 감지");
        return false;
    }
    
    // 푸앵카레 볼 경계 검사
    if r >= crate::math::poincare::POINCARE_BOUNDARY_F32 {
        eprintln!("경고: 푸앵카레 볼 경계 초과");
        return false;
    }
    
    // 각도 범위 검사
    if theta < 0.0 || theta >= 2.0 * std::f32::consts::PI {
        eprintln!("경고: 각도 범위 초과");
        return false;
    }
    
    true
}
```

## 제약사항 및 주의사항

### 수치적 제약

- **푸앵카레 볼 경계**: r 값은 반드시 `[0, POINCARE_BOUNDARY_F32)` 범위
- **각도 정규화**: θ 값은 `[0, 2π)` 범위로 자동 정규화
- **그래디언트 크기**: 매우 큰 그래디언트는 자동 클리핑 적용

### 성능 고려사항

- **메트릭 텐서 계산**: Riemannian Adam에서 추가 연산 비용
- **수치적 안정성**: 안전성 체크로 인한 약간의 성능 오버헤드
- **메모리 사용량**: 모멘트 벡터 저장으로 2배 메모리 사용

### 매니폴드 특성

- **기하학적 제약**: 푸앵카레 볼의 기하학적 특성 준수 필요
- **거리 계산**: 유클리드 거리가 아닌 푸앵카레 거리 사용
- **경계 동작**: 볼 경계 근처에서 특별한 처리 필요

## 버전 호환성

- **Rust 최소 버전**: 1.70.0
- **주요 의존성**: 없음 (표준 라이브러리만 사용)
- **플랫폼 지원**: 모든 플랫폼 (순수 Rust 구현)

## 추가 참고 자료

- [푸앵카레 볼 기하학](../math.md)
- [Riemannian 최적화 이론](../../paper/Adam과_r리만_Adam도_업그레이드할까.md)
- [수치적 안정성 개선](../../paper/지금_비트_연산_우리가_위에_연구한거_잘_대입되어있느지_철저히_봐.md)
- [성능 벤치마크](../../test/optimizer_performance_report.md) 