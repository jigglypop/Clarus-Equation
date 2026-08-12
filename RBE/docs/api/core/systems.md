# Systems API

## HybridPoincareRBESystem

하이브리드 푸앵카레 RBE 시스템

```rust
pub struct HybridPoincareRBESystem {
    config: SystemConfiguration,
    performance: PerformanceMonitor,
}
```

### 함수

- `new(config: SystemConfiguration) -> Self`
- `forward(&mut self, input: &[f32]) -> Vec<f32>`
- `backward(&mut self, grad_output: &[f32]) -> Vec<f32>`
- `get_performance_report(&self) -> String`

## ComputeEngine

계산 엔진

```rust
pub struct ComputeEngine {
    cordic: CORDICEngine,
    gemm: ParallelGEMMEngine,
    lookup_table: BasisFunctionLookupTable,
}
```

### 함수

- `new() -> Self`
- `process_batch(&mut self, batch: &[Vec<f32>]) -> Vec<Vec<f32>>`
- `parallel_computation(&self, tasks: &[ComputeTask]) -> Vec<f32>`

## CoreLayer

코어 레이어

```rust
pub struct CoreLayer {
    layer_type: LayerType,
    params: Vec<Packed128>,
}
```

### 함수

- `new(layer_type: LayerType, size: usize) -> Self`
- `fused_forward(&self, input: &[f32]) -> Vec<f32>`
- `fused_backward(&mut self, grad: &[f32], lr: f32) -> Vec<f32>`

## PerformanceMonitor

성능 모니터

```rust
pub struct PerformanceMonitor {
    layer_metrics: HashMap<String, LayerPerformanceData>,
    system_metrics: SystemMetrics,
}
```

### 함수

- `new() -> Self`
- `record_layer_performance(&mut self, layer_name: &str, data: LayerPerformanceData)`
- `get_compression_metrics(&self) -> CompressionMetrics`

## StateManager

상태 관리자

```rust
pub struct StateManager {
    learning_state: LearningState,
    parameter_manager: ParameterManager,
}
```

### 함수

- `new() -> Self`
- `update_learning_state(&mut self, loss: f32, epoch: usize)`
- `get_convergence_status(&self) -> ConvergenceStatus`

## Configuration

### SystemConfiguration

```rust
pub struct SystemConfiguration {
    pub hardware: HardwareConfiguration,
    pub learning: LearningParameters,
    pub optimization: OptimizationConfiguration,
    pub memory: MemoryConfiguration,
}
```

### HardwareConfiguration

```rust
pub struct HardwareConfiguration {
    pub use_gpu: bool,
    pub num_threads: usize,
    pub cache_size: usize,
}
```

### LearningParameters

```rust
pub struct LearningParameters {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
} 