# Matrix API

## HierarchicalMatrix

계층적 행렬

```rust
pub struct HierarchicalMatrix {
    l1_blocks: Vec<Vec<L1Block>>,
    l2_blocks: Vec<Vec<L2Block>>,
    l3_blocks: Vec<Vec<L3Block>>,
    l4_blocks: Vec<Vec<L4Block>>,
}
```

### 함수

- `new(rows: usize, cols: usize) -> Self`
- `parallel_gemv(&self, input: &[f32], output: &mut [f32])`
- `get_quality_stats(&self) -> QualityStats`
- `memory_usage(&self) -> (usize, f32)`

## Block 타입들

### L1Block

```rust
pub struct L1Block {
    pub data: Packed128,
    pub row_start: usize,
    pub col_start: usize,
    pub size: usize,
}
```

### L2Block, L3Block, L4Block

유사한 구조로 각각 다른 블록 크기를 가짐

## ErrorController

오차 제어기

```rust
pub struct ErrorController {
    max_error: f32,
    error_history: Vec<f32>,
}
```

### 함수

- `new(max_error: f32) -> Self`
- `update_error(&mut self, current_error: f32)`
- `should_split_block(&self, block_error: f32) -> bool`
- `compute_total_error(&self) -> f32`

## QualityStats

품질 통계

```rust
pub struct QualityStats {
    pub total_blocks: usize,
    pub avg_quality: f32,
    pub min_quality: f32,
    pub max_quality: f32,
    pub quality_distribution: Vec<(QualityGrade, usize)>,
}
```

### 함수

- `new() -> Self`
- `efficiency_grade(&self) -> EfficiencyGrade`
- `print_report(&self)`

## QualityGrade

```rust
pub enum QualityGrade {
    Excellent,
    Good,
    Average,
    Poor,
}
```

## EfficiencyGrade

```rust
pub enum EfficiencyGrade {
    Optimal,
    Good,
    Acceptable,
    NeedsImprovement,
}
``` 