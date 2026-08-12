# Generator API

## PoincareLearning

푸앵카레 학습 시스템

```rust
pub struct PoincareLearning {
    learning_rate: f32,
    momentum: f32,
    regularization: f32,
}
```

### 함수

- `new(learning_rate: f32, momentum: f32, regularization: f32) -> Self`
- `train_step(&mut self, params: &mut Packed128, target: &[Vec<f32>]) -> f32`
- `compute_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> f32`
- `update_parameters(&mut self, params: &mut Packed128, gradients: &[f32])`

## ConstraintProjection

제약 조건 투영

```rust
pub struct ConstraintProjection;
```

### 함수

- `project_to_poincare_ball(point: &mut (f32, f32))`
- `project_to_hyperbolic_space(point: &mut (f32, f32))`
- `enforce_bounds(params: &mut Packed128, min_val: f32, max_val: f32)`

## Convergence

수렴 분석

```rust
pub struct Convergence {
    tolerance: f32,
    max_iterations: usize,
    history: Vec<f32>,
}
```

### 함수

- `new(tolerance: f32, max_iterations: usize) -> Self`
- `check_convergence(&mut self, current_loss: f32) -> bool`
- `get_convergence_rate(&self) -> f32`
- `reset(&mut self)`

## HybridOptimizer

하이브리드 최적화기

```rust
pub struct HybridOptimizer {
    adam_state: AdamState,
    riemannian_state: RiemannianAdamState,
}
```

### 함수

- `new(config: OptimizerConfig) -> Self`
- `step(&mut self, params: &mut Packed128, gradients: &[f32]) -> f32`
- `switch_optimizer(&mut self, optimizer_type: OptimizerType)`

## Regularization

정규화 기법

```rust
pub struct Regularization {
    l1_weight: f32,
    l2_weight: f32,
    dropout_rate: f32,
}
```

### 함수

- `new(l1: f32, l2: f32, dropout: f32) -> Self`
- `apply_l1_regularization(&self, params: &mut Packed128)`
- `apply_l2_regularization(&self, params: &mut Packed128)`
- `apply_dropout(&self, values: &mut [f32], rng: &mut impl Rng)`

## StateTransition

상태 전이 관리

```rust
pub struct StateTransition {
    transition_rate: f32,
    smoothing_factor: f32,
}
```

### 함수

- `new(rate: f32, smoothing: f32) -> Self`
- `smooth_transition(&self, from: &Packed128, to: &Packed128, alpha: f32) -> Packed128`
- `adaptive_transition(&self, current: &Packed128, targets: &[Packed128]) -> Packed128` 