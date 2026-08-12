# Types API

## Packed128

128비트 압축 타입의 핵심 구조체

```rust
pub struct Packed128 {
    pub hi: u64,
    pub lo: u64,
}
```

### 생성자

- `new(r: f32, theta: f32, phi: f32) -> Self`
- `random(rng: &mut impl Rng) -> Self`
- `from_bits(hi: u64, lo: u64) -> Self`

### 핵심 함수

- `fused_forward(i: usize, j: usize, rows: usize, cols: usize) -> f32`
- `backward_pass(&mut self, grad: f32, lr: f32, clip_norm: Option<f32>)`
- `poincare_distance(&self, other: &Self) -> f32`
- `transition_to(&mut self, target: &Self, alpha: f32)`

## Packed64

64비트 압축 타입

```rust
pub struct Packed64 {
    pub data: u64,
}
```

### 함수

- `new(r: f32, theta: f32) -> Self`
- `decode(&self) -> (f32, f32)`

## DecodedParams

디코딩된 파라미터

```rust
pub struct DecodedParams {
    pub r_fp32: f32,
    pub theta_fp32: f32,
}
```

## PoincarePackedBit128

푸앵카레 볼 특화 128비트 타입

### 생성자

- `new(quadrant, frequency, amplitude, basis_func, cordic_rotation, r, theta) -> Self`

### 접근자

- `get_quadrant() -> PoincareQuadrant`
- `get_hyperbolic_frequency() -> u16`
- `get_geodesic_amplitude() -> u16`
- `get_cordic_rotation_sequence() -> u32`

## PoincareQuadrant

```rust
pub enum PoincareQuadrant {
    First, Second, Third, Fourth,
}
``` 