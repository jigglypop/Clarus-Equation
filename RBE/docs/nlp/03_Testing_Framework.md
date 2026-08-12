# 테스트 프레임워크 가이드

## 개요

이 문서는 RBE NLP 구현의 정확성과 성능을 검증하기 위한 포괄적인 테스트 프레임워크를 다룹니다. 단위 테스트부터 통합 테스트까지 체계적인 검증 방법론을 제시합니다.

## 테스트 철학

### 1. 수치적 정확성 우선
모든 수학적 연산은 기준 구현 대비 허용 오차 내에서 동작해야 합니다.

### 2. 성능 회귀 방지  
기존 성능 수준을 유지하면서 기능을 확장해야 합니다.

### 3. 포괄적 커버리지
엣지 케이스까지 포함한 완전한 테스트 커버리지를 확보합니다.

## 테스트 디렉토리 구조

```
src/nlp/
├── tensor/
│   └── __tests__/
│       ├── tensor_test.rs
│       ├── operations_test.rs
│       └── autodiff_test.rs
├── layers/
│   └── __tests__/
│       ├── layer_norm_test.rs
│       ├── embedding_test.rs
│       └── activation_test.rs
├── models/
│   └── __tests__/
│       ├── mini_gpt2_test.rs
│       └── integration_test.rs
└── __tests__/
    ├── benchmark_test.rs
    └── e2e_test.rs
```

## 단위 테스트 프레임워크

### 기본 테스트 헬퍼

```rust
use anyhow::Result;
use approx::{assert_relative_eq, assert_abs_diff_eq};

/// 테스트 유틸리티 모듈
pub mod test_utils {
    use super::*;
    use crate::nlp::tensor::RBETensor;
    use rand::prelude::*;
    
    /// 허용 오차 상수
    pub const FLOAT_TOLERANCE: f32 = 1e-5;
    pub const STRICT_TOLERANCE: f32 = 1e-6;
    pub const LOOSE_TOLERANCE: f32 = 1e-3;
    
    /// 테스트용 랜덤 텐서 생성
    pub fn random_tensor(shape: &[usize], min: f32, max: f32) -> Result<RBETensor> {
        let total_elements: usize = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.gen_range(min..max))
            .collect();
        
        RBETensor::new(data, shape.to_vec())
    }
    
    /// 두 텐서의 근사 동등성 검사
    pub fn assert_tensors_close(
        actual: &RBETensor, 
        expected: &RBETensor, 
        tolerance: f32
    ) -> Result<()> {
        assert_eq!(actual.shape(), expected.shape(), 
                  "Shape mismatch: {:?} vs {:?}", actual.shape(), expected.shape());
        
        for (i, (&a, &e)) in actual.data.iter().zip(expected.data.iter()).enumerate() {
            assert_abs_diff_eq!(a, e, epsilon = tolerance,
                "Values differ at index {}: {} vs {} (tolerance: {})", 
                i, a, e, tolerance);
        }
        
        Ok(())
    }
    
    /// 상대 오차 계산
    pub fn compute_relative_error(actual: &[f32], expected: &[f32]) -> f32 {
        if actual.len() != expected.len() {
            return f32::INFINITY;
        }
        
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            let diff = (a - e).abs();
            let exp_abs = e.abs();
            
            num += diff * diff;
            den += exp_abs * exp_abs;
        }
        
        if den < 1e-10 {
            return if num < 1e-10 { 0.0 } else { f32::INFINITY };
        }
        
        (num / den).sqrt()
    }
    
    /// 성능 측정 헬퍼
    pub fn benchmark_function<F, R>(f: F, iterations: usize) -> (R, std::time::Duration)
    where
        F: Fn() -> R + Clone,
    {
        let start = std::time::Instant::now();
        let mut result = None;
        
        for _ in 0..iterations {
            result = Some(f());
        }
        
        let duration = start.elapsed();
        (result.unwrap(), duration)
    }
    
    /// 메모리 사용량 측정 (근사값)
    pub fn measure_memory_usage<F, R>(f: F) -> (R, usize) 
    where
        F: FnOnce() -> R,
    {
        // 실제 구현에서는 더 정확한 메모리 측정 도구 사용
        let before = get_memory_usage();
        let result = f();
        let after = get_memory_usage();
        
        (result, after.saturating_sub(before))
    }
    
    fn get_memory_usage() -> usize {
        // Placeholder - 실제 구현에서는 OS별 메모리 측정 함수 사용
        0
    }
}
```

### RBETensor 단위 테스트

```rust
#[cfg(test)]
mod tensor_tests {
    use super::test_utils::*;
    use crate::nlp::tensor::RBETensor;
    
    #[test]
    fn 텐서_생성_테스트() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = RBETensor::new(data.clone(), vec![2, 3])?;
        
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.data, data);
        
        Ok(())
    }
    
    #[test]
    fn 텐서_생성_오류_테스트() -> Result<()> {
        // 데이터 길이와 shape 불일치
        let result = RBETensor::new(vec![1.0, 2.0], vec![2, 3]);
        assert!(result.is_err());
        
        Ok(())
    }
    
    #[test]
    fn 덧셈_연산_정확성_테스트() -> Result<()> {
        let a = RBETensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = RBETensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        
        let c = a.add(&b)?;
        let expected = RBETensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2])?;
        
        assert_tensors_close(&c, &expected, FLOAT_TOLERANCE)?;
        
        Ok(())
    }
    
    #[test]
    fn 브로드캐스팅_테스트() -> Result<()> {
        let a = random_tensor(&[3, 1], -1.0, 1.0)?;
        let b = random_tensor(&[1, 4], -1.0, 1.0)?;
        
        let c = a.add(&b)?;
        assert_eq!(c.shape(), &[3, 4]);
        
        // 브로드캐스팅 결과 수동 검증
        for i in 0..3 {
            for j in 0..4 {
                let expected = a.data[i] + b.data[j];
                let actual = c.data[i * 4 + j];
                assert_abs_diff_eq!(actual, expected, epsilon = FLOAT_TOLERANCE);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn 행렬곱셈_정확성_테스트() -> Result<()> {
        // 간단한 2x2 행렬 테스트
        let a = RBETensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = RBETensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        
        let c = a.matmul(&b)?;
        
        // 수동 계산: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let expected = RBETensor::new(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2])?;
        
        assert_tensors_close(&c, &expected, FLOAT_TOLERANCE)?;
        
        Ok(())
    }
    
    #[test]
    fn 대형_행렬곱셈_성능_테스트() -> Result<()> {
        let size = 512;
        let a = random_tensor(&[size, size], -1.0, 1.0)?;
        let b = random_tensor(&[size, size], -1.0, 1.0)?;
        
        let (result, duration) = benchmark_function(|| {
            a.matmul(&b).unwrap()
        }, 5);
        
        assert_eq!(result.shape(), &[size, size]);
        
        // 성능 기준: 512x512 행렬곱셈이 1초 이내
        assert!(duration.as_secs() < 1, 
               "Matrix multiplication too slow: {:?}", duration);
        
        println!("512x512 matmul: {:?}", duration);
        
        Ok(())
    }
    
    #[test]
    fn 메모리_효율성_테스트() -> Result<()> {
        let size = 1000;
        
        let (tensor, memory_used) = measure_memory_usage(|| {
            random_tensor(&[size, size], -1.0, 1.0).unwrap()
        });
        
        let expected_memory = size * size * 4; // f32 = 4 bytes
        
        // 메모리 사용량이 예상 범위 내인지 확인 (오버헤드 고려)
        assert!(memory_used <= expected_memory * 2,
               "Memory usage too high: {} bytes (expected ~{})", 
               memory_used, expected_memory);
        
        println!("Memory usage for {}x{} tensor: {} bytes", size, size, memory_used);
        
        Ok(())
    }
}
```

### LayerNorm 테스트

```rust
#[cfg(test)]
mod layer_norm_tests {
    use super::test_utils::*;
    use crate::nlp::layers::RBELayerNorm;
    
    #[test]
    fn 레이어노름_수학적_정확성_테스트() -> Result<()> {
        let layer_norm = RBELayerNorm::new(vec![4], 1e-5)?;
        let input = RBETensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;
        
        let output = layer_norm.forward(&input)?;
        
        // 수동 계산
        let mean = 2.5; // (1+2+3+4)/4
        let variance = 1.25; // var([1,2,3,4])
        let std_dev = (variance + 1e-5).sqrt();
        
        let expected = vec![
            (1.0 - mean) / std_dev,
            (2.0 - mean) / std_dev,
            (3.0 - mean) / std_dev,
            (4.0 - mean) / std_dev,
        ];
        
        let expected_tensor = RBETensor::new(expected, vec![1, 4])?;
        assert_tensors_close(&output, &expected_tensor, FLOAT_TOLERANCE)?;
        
        Ok(())
    }
    
    #[test]
    fn 배치_레이어노름_테스트() -> Result<()> {
        let layer_norm = RBELayerNorm::new(vec![3], 1e-5)?;
        let batch_size = 4;
        let input = random_tensor(&[batch_size, 3], -2.0, 2.0)?;
        
        let output = layer_norm.forward(&input)?;
        
        assert_eq!(output.shape(), &[batch_size, 3]);
        
        // 각 배치의 평균이 0에 가까운지 확인
        for i in 0..batch_size {
            let batch_start = i * 3;
            let batch_end = batch_start + 3;
            let batch_mean: f32 = output.data[batch_start..batch_end].iter().sum::<f32>() / 3.0;
            
            assert_abs_diff_eq!(batch_mean, 0.0, epsilon = LOOSE_TOLERANCE,
                "Batch {} mean should be close to 0, got {}", i, batch_mean);
        }
        
        Ok(())
    }
    
    #[test]
    fn 레이어노름_수치_안정성_테스트() -> Result<()> {
        let layer_norm = RBELayerNorm::new(vec![2], 1e-5)?;
        
        // 매우 큰 값으로 테스트
        let large_input = RBETensor::new(vec![1e6, 1e6], vec![1, 2])?;
        let output1 = layer_norm.forward(&large_input)?;
        
        // 매우 작은 값으로 테스트
        let small_input = RBETensor::new(vec![1e-6, 1e-6], vec![1, 2])?;
        let output2 = layer_norm.forward(&small_input)?;
        
        // 출력이 NaN이나 Inf가 아닌지 확인
        for &val in &output1.data {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
        for &val in &output2.data {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
        
        Ok(())
    }
}
```

### 임베딩 테스트

```rust
#[cfg(test)]
mod embedding_tests {
    use super::test_utils::*;
    use crate::nlp::layers::{TokenEmbedding, PositionalEmbedding};
    
    #[test]
    fn 토큰_임베딩_기본_테스트() -> Result<()> {
        let vocab_size = 1000;
        let embed_dim = 256;
        let embedding = TokenEmbedding::new(vocab_size, embed_dim, Some(0))?;
        
        let tokens = vec![0, 1, 2, 999]; // padding 포함
        let output = embedding.forward(&tokens)?;
        
        assert_eq!(output.shape(), &[4, embed_dim]);
        
        // Padding 토큰 (0번)의 임베딩은 모두 0이어야 함
        let padding_slice = &output.data[0..embed_dim];
        assert!(padding_slice.iter().all(|&x| x == 0.0),
               "Padding embedding should be zero");
        
        Ok(())
    }
    
    #[test]
    fn 토큰_임베딩_범위_검사_테스트() -> Result<()> {
        let embedding = TokenEmbedding::new(100, 64, None)?;
        
        // 유효하지 않은 토큰 ID
        let invalid_tokens = vec![100, 200];
        let result = embedding.forward(&invalid_tokens);
        
        assert!(result.is_err(), "Should fail with out-of-range token ID");
        
        Ok(())
    }
    
    #[test]
    fn 위치_임베딩_테스트() -> Result<()> {
        let max_len = 512;
        let embed_dim = 128;
        let pos_embedding = PositionalEmbedding::new(max_len, embed_dim)?;
        
        let seq_len = 10;
        let output = pos_embedding.forward(seq_len, 0)?;
        
        assert_eq!(output.shape(), &[seq_len, embed_dim]);
        
        Ok(())
    }
    
    #[test]
    fn sinusoidal_위치_임베딩_테스트() -> Result<()> {
        let max_len = 100;
        let embed_dim = 64; // 짝수여야 함
        let pos_embedding = PositionalEmbedding::new_sinusoidal(max_len, embed_dim)?;
        
        let output = pos_embedding.forward(10, 0)?;
        
        // Sinusoidal 패턴 검증
        for pos in 0..10 {
            for i in (0..embed_dim).step_by(2) {
                let sin_val = output.data[pos * embed_dim + i];
                let cos_val = output.data[pos * embed_dim + i + 1];
                
                // sin²+cos² = 1 (부동소수점 오차 고려)
                let sum_squares = sin_val * sin_val + cos_val * cos_val;
                assert_abs_diff_eq!(sum_squares, 1.0, epsilon = LOOSE_TOLERANCE,
                    "Sinusoidal property violated at pos={}, i={}", pos, i);
            }
        }
        
        Ok(())
    }
}
```

## 통합 테스트

### End-to-End 테스트

```rust
#[cfg(test)]
mod integration_tests {
    use super::test_utils::*;
    use crate::nlp::{layers::*, models::*};
    
    #[test]
    fn 미니_gpt2_순전파_테스트() -> Result<()> {
        let config = MiniGPT2Config {
            vocab_size: 1000,
            hidden_size: 256,
            num_layers: 2,
            num_heads: 4,
            seq_len: 64,
        };
        
        let model = MiniGPT2::new(config)?;
        let tokens = vec![1, 2, 3, 4, 5];
        
        let (output, _) = measure_memory_usage(|| {
            model.forward(&tokens).unwrap()
        });
        
        assert_eq!(output.shape(), &[5, 1000]); // [seq_len, vocab_size]
        
        // 출력이 유효한 확률 분포인지 확인 (softmax 적용 후)
        for i in 0..5 {
            let start_idx = i * 1000;
            let end_idx = start_idx + 1000;
            let logits = &output.data[start_idx..end_idx];
            
            // 모든 값이 유한해야 함
            assert!(logits.iter().all(|&x| x.is_finite()),
                   "Non-finite values in output");
        }
        
        Ok(())
    }
    
    #[test]
    fn 텍스트_생성_테스트() -> Result<()> {
        let config = MiniGPT2Config::default();
        let mut model = MiniGPT2::new(config)?;
        
        let prompt = vec![1, 2, 3]; // 간단한 프롬프트
        let generated = model.generate(&prompt, 10, 0.8, 0.9)?;
        
        assert_eq!(generated.len(), 10);
        assert!(generated.iter().all(|&token| token < 1000),
               "Generated tokens out of vocabulary range");
        
        Ok(())
    }
}
```

### 성능 벤치마크

```rust
#[cfg(test)]
mod benchmark_tests {
    use super::test_utils::*;
    use criterion::{black_box, Criterion};
    
    #[test]
    fn 전체_파이프라인_성능_벤치마크() -> Result<()> {
        let config = MiniGPT2Config::default();
        let model = MiniGPT2::new(config)?;
        
        let batch_sizes = vec![1, 4, 8, 16];
        let seq_lens = vec![32, 64, 128];
        
        for &batch_size in &batch_sizes {
            for &seq_len in &seq_lens {
                let input_data = vec![vec![1u32; seq_len]; batch_size];
                
                let (_, duration) = benchmark_function(|| {
                    model.forward_batch(&input_data).unwrap()
                }, 10);
                
                let tokens_per_second = (batch_size * seq_len) as f32 / duration.as_secs_f32();
                
                println!("Batch={}, SeqLen={}: {:.1} tokens/sec", 
                        batch_size, seq_len, tokens_per_second);
                
                // 최소 성능 기준 (예시)
                assert!(tokens_per_second > 100.0,
                       "Performance too low: {:.1} tokens/sec", tokens_per_second);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn 메모리_효율성_벤치마크() -> Result<()> {
        let config = MiniGPT2Config::default();
        
        let (model, model_memory) = measure_memory_usage(|| {
            MiniGPT2::new(config).unwrap()
        });
        
        let (_, inference_memory) = measure_memory_usage(|| {
            let tokens = vec![1, 2, 3, 4, 5];
            model.forward(&tokens).unwrap()
        });
        
        println!("Model memory: {} MB", model_memory / 1024 / 1024);
        println!("Inference memory: {} MB", inference_memory / 1024 / 1024);
        
        // 메모리 효율성 기준
        let expected_model_memory = 100 * 1024 * 1024; // 100MB
        assert!(model_memory < expected_model_memory,
               "Model memory usage too high: {} bytes", model_memory);
        
        Ok(())
    }
}
```

## 회귀 테스트

### 정확도 회귀 방지

```rust
#[cfg(test)]
mod regression_tests {
    use super::test_utils::*;
    
    /// 알려진 출력값과 비교하는 골든 테스트
    #[test]
    fn 골든_출력_회귀_테스트() -> Result<()> {
        // 고정된 시드로 재현 가능한 결과 생성
        let fixed_input = RBETensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
            vec![2, 3]
        )?;
        
        let layer_norm = RBELayerNorm::new(vec![3], 1e-5)?;
        let output = layer_norm.forward(&fixed_input)?;
        
        // 미리 계산된 예상 출력값 (골든 값)
        let expected_golden = vec![
            -1.2247356, 0.0, 1.2247356,
            -1.2247356, 0.0, 1.2247356
        ];
        
        let golden_tensor = RBETensor::new(expected_golden, vec![2, 3])?;
        assert_tensors_close(&output, &golden_tensor, STRICT_TOLERANCE)?;
        
        Ok(())
    }
    
    /// 성능 회귀 방지
    #[test]
    fn 성능_회귀_테스트() -> Result<()> {
        let baseline_times = load_baseline_performance()?;
        let current_times = measure_current_performance()?;
        
        for (operation, &baseline) in &baseline_times {
            let current = current_times.get(operation).unwrap();
            let regression_threshold = 1.1; // 10% 이하 성능 저하만 허용
            
            assert!(current <= &(baseline * regression_threshold),
                   "Performance regression in {}: {:?} -> {:?}",
                   operation, baseline, current);
        }
        
        Ok(())
    }
    
    fn load_baseline_performance() -> Result<std::collections::HashMap<String, std::time::Duration>> {
        // 실제로는 파일에서 로드하거나 환경 변수에서 가져옴
        let mut baselines = std::collections::HashMap::new();
        baselines.insert("matmul_512x512".to_string(), std::time::Duration::from_millis(100));
        baselines.insert("layer_norm_1024".to_string(), std::time::Duration::from_millis(10));
        Ok(baselines)
    }
    
    fn measure_current_performance() -> Result<std::collections::HashMap<String, std::time::Duration>> {
        let mut current = std::collections::HashMap::new();
        
        // matmul 성능 측정
        let (_, duration) = benchmark_function(|| {
            let a = random_tensor(&[512, 512], -1.0, 1.0).unwrap();
            let b = random_tensor(&[512, 512], -1.0, 1.0).unwrap();
            a.matmul(&b).unwrap()
        }, 5);
        current.insert("matmul_512x512".to_string(), duration);
        
        // layer_norm 성능 측정  
        let (_, duration) = benchmark_function(|| {
            let layer_norm = RBELayerNorm::new(vec![1024], 1e-5).unwrap();
            let input = random_tensor(&[32, 1024], -1.0, 1.0).unwrap();
            layer_norm.forward(&input).unwrap()
        }, 10);
        current.insert("layer_norm_1024".to_string(), duration);
        
        Ok(current)
    }
}
```

## CI/CD 통합

### GitHub Actions 설정

```yaml
# .github/workflows/nlp_tests.yml
name: NLP Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run unit tests
      run: cargo test --lib nlp -- --nocapture
      
    - name: Run integration tests
      run: cargo test --test integration -- --nocapture
      
    - name: Run benchmarks
      run: cargo test --release benchmark -- --nocapture
      
    - name: Check memory leaks
      run: cargo test --features=valgrind
```

## 테스트 실행 가이드

### 로컬 테스트 실행

```bash
# 전체 테스트 실행
cargo test

# NLP 모듈만 테스트
cargo test --lib nlp

# 특정 테스트 실행
cargo test 텐서_생성_테스트

# 성능 테스트 (릴리즈 모드)
cargo test --release benchmark

# 상세 출력과 함께 실행
cargo test -- --nocapture

# 병렬 테스트 비활성화 (디버깅용)
cargo test -- --test-threads=1
```

### 커버리지 측정

```bash
# tarpaulin 설치
cargo install cargo-tarpaulin

# 커버리지 측정
cargo tarpaulin --out Html

# 특정 모듈 커버리지
cargo tarpaulin --packages rbe_llm --out Html
```

## 테스트 작성 가이드라인

### 1. 명명 규칙
- 테스트 함수명은 한글로 작성
- 형식: `기능_조건_테스트()`
- 예: `텐서_덧셈_브로드캐스팅_테스트()`

### 2. 테스트 구조
```rust
#[test]
fn 기능_테스트() -> Result<()> {
    // 1. 준비 (Arrange)
    let input = create_test_input();
    
    // 2. 실행 (Act)  
    let result = function_under_test(input);
    
    // 3. 검증 (Assert)
    assert_expected_result(result);
    
    Ok(())
}
```

### 3. 오차 허용치
- 수학적 연산: `FLOAT_TOLERANCE` (1e-5)
- 수치적 안정성 테스트: `STRICT_TOLERANCE` (1e-6)  
- 근사 알고리즘: `LOOSE_TOLERANCE` (1e-3)

이 테스트 프레임워크를 통해 RBE NLP 구현의 정확성과 성능을 보장할 수 있습니다. 