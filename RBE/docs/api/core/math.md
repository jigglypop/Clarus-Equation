# Math API

## 기본 수학 함수

### 기저 함수

- `legendre_basis(x: f32, n: usize) -> f32` - 르장드르 다항식
- `chebyshev_basis(x: f32, n: usize) -> f32` - 체비셰프 다항식
- `hermite_basis(x: f32, n: usize) -> f32` - 에르미트 다항식

### 특수 함수

- `gamma_function(x: f32) -> f32` - 감마 함수
- `beta_function(a: f32, b: f32) -> f32` - 베타 함수
- `erf(x: f32) -> f32` - 오차 함수

## 푸앵카레 수학

### HyperbolicMath

```rust
pub struct HyperbolicMath;
```

#### 함수

- `poincare_distance(z1: (f32, f32), z2: (f32, f32)) -> f32`
- `mobius_add(z1: (f32, f32), z2: (f32, f32)) -> (f32, f32)`
- `exp_map(v: (f32, f32), x: (f32, f32)) -> (f32, f32)`
- `log_map(y: (f32, f32), x: (f32, f32)) -> (f32, f32)`

### 메트릭 텐서

- `metric_tensor(point: (f32, f32)) -> [[f32; 2]; 2]`
- `christoffel_symbols(point: (f32, f32)) -> [[[f32; 2]; 2]; 2]`

## 베셀 함수

### BesselFunction

```rust
pub struct BesselFunction {
    pub order: f32,
}
```

#### 함수

- `new(order: f32) -> Self`
- `j_bessel(x: f32) -> f32` - 1종 베셀 함수
- `y_bessel(x: f32) -> f32` - 2종 베셀 함수
- `modified_bessel_i(x: f32) -> f32` - 수정 베셀 함수 I
- `modified_bessel_k(x: f32) -> f32` - 수정 베셀 함수 K

## 융합 연산

### FusedOps

```rust
pub struct FusedOps;
```

#### 함수

- `fused_mul_add(a: f32, b: f32, c: f32) -> f32`
- `fused_exp_normalize(x: &[f32]) -> Vec<f32>`
- `fused_sigmoid_tanh(x: f32) -> (f32, f32)`
- `batch_norm_fused(x: &[f32], mean: f32, var: f32) -> Vec<f32>`

## 그래디언트 계산

### Gradient

```rust
pub struct Gradient;
```

#### 함수

- `numerical_gradient(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32`
- `central_difference(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32`
- `richardson_extrapolation(f: impl Fn(f32) -> f32, x: f32) -> f32` 