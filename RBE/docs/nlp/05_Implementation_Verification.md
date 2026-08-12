# 구현 검증 방법론

## 개요

본 문서는 RBE NLP 구현의 정확성, 성능, 안정성을 체계적으로 검증하기 위한 포괄적인 방법론을 제시합니다. 각 구성요소별 검증 기준과 구체적인 테스트 방법을 상세히 설명합니다.

## 1. 전체 검증 프레임워크

### 1.1 검증 레벨

#### Level 0: 단위 검증 (Unit Verification)
- **범위**: 개별 함수/메서드 수준
- **목표**: 기본 기능의 정확성 확인
- **허용 오차**: 1e-6 (strict)

#### Level 1: 모듈 검증 (Module Verification)  
- **범위**: 클래스/구조체 수준
- **목표**: 구성요소 간 상호작용 검증
- **허용 오차**: 1e-5 (standard)

#### Level 2: 시스템 검증 (System Verification)
- **범위**: 전체 파이프라인
- **목표**: End-to-end 기능성 검증
- **허용 오차**: 1e-3 (relaxed)

#### Level 3: 성능 검증 (Performance Verification)
- **범위**: 실제 사용 시나리오
- **목표**: 성능 목표 달성 확인
- **허용 범위**: 기준 대비 ±10%

### 1.2 검증 체크리스트

```rust
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub component: String,
    pub level: VerificationLevel,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub coverage_percentage: f32,
    pub performance_score: f32,
    pub issues: Vec<VerificationIssue>,
}

#[derive(Debug, Clone)]
pub enum VerificationLevel {
    Unit,
    Module, 
    System,
    Performance,
}

#[derive(Debug, Clone)]
pub struct VerificationIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub location: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone)]
pub enum IssueSeverity {
    Critical,    // 즉시 수정 필요
    Major,       // 다음 릴리즈 전 수정
    Minor,       // 최적화 대상
    Info,        // 참고 사항
}
```

## 2. RBETensor 검증

### 2.1 기본 연산 정확성 검증

#### 텐서 생성 검증
```rust
#[cfg(test)]
mod tensor_creation_verification {
    use super::*;
    
    #[test]
    fn 텐서_생성_정확성_검증() -> Result<()> {
        // 1. 다양한 크기의 텐서 생성
        let test_cases = vec![
            (vec![1], "1D tensor"),
            (vec![3, 4], "2D tensor"),
            (vec![2, 3, 4], "3D tensor"),
            (vec![2, 3, 4, 5], "4D tensor"),
        ];
        
        for (shape, description) in test_cases {
            let total_elements: usize = shape.iter().product();
            let data: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
            
            let tensor = RBETensor::new(data.clone(), shape.clone())?;
            
            // 기본 속성 검증
            assert_eq!(tensor.shape(), shape.as_slice(), 
                      "Shape mismatch for {}", description);
            assert_eq!(tensor.numel(), total_elements,
                      "Element count mismatch for {}", description);
            assert_eq!(tensor.data, data,
                      "Data mismatch for {}", description);
                      
            // 메모리 레이아웃 검증
            verify_memory_layout(&tensor)?;
        }
        
        Ok(())
    }
    
    fn verify_memory_layout(tensor: &RBETensor) -> Result<()> {
        // Stride 계산 검증
        let expected_strides = compute_expected_strides(tensor.shape());
        assert_eq!(tensor.strides, expected_strides,
                  "Stride calculation incorrect");
        
        // 인덱싱 검증
        for i in 0..tensor.numel().min(100) { // 샘플 검증
            let multi_idx = flat_to_multi_index(i, tensor.shape());
            let flat_idx = tensor.flat_index(&multi_idx)?;
            assert_eq!(flat_idx, i, "Index conversion mismatch");
        }
        
        Ok(())
    }
}
```

#### 산술 연산 검증
```rust
#[cfg(test)]
mod arithmetic_verification {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn 덧셈_연산_정확성_검증() -> Result<()> {
        let test_configurations = vec![
            TestConfig::same_shape(vec![3, 4]),
            TestConfig::broadcastable(vec![3, 1], vec![1, 4]),
            TestConfig::scalar_broadcast(vec![3, 4], vec![1]),
        ];
        
        for config in test_configurations {
            let (a, b) = config.generate_test_tensors()?;
            let result = a.add(&b)?;
            
            // 참조 구현과 비교
            let expected = reference_add(&a, &b)?;
            verify_tensor_close(&result, &expected, 1e-6)?;
            
            // 형태 검증
            let expected_shape = broadcast_shape(a.shape(), b.shape())?;
            assert_eq!(result.shape(), expected_shape);
            
            // 자동미분 설정 검증
            if a.requires_grad || b.requires_grad {
                assert!(result.requires_grad, "Gradient tracking lost");
                assert!(!result.is_leaf, "Result should not be leaf");
            }
        }
        
        Ok(())
    }
    
    fn reference_add(a: &RBETensor, b: &RBETensor) -> Result<RBETensor> {
        // NumPy/PyTorch와 동일한 결과를 생성하는 참조 구현
        // 이를 통해 브로드캐스팅 로직의 정확성 검증
        unimplemented!("Reference implementation using ndarray")
    }
}
```

#### 행렬 곱셈 검증
```rust
#[cfg(test)]
mod matmul_verification {
    use super::*;
    
    #[test]
    fn 행렬곱셈_수학적_정확성_검증() -> Result<()> {
        // 다양한 크기의 행렬 곱셈 테스트
        let test_cases = vec![
            ((2, 3), (3, 4)),        // 기본 경우
            ((1, 5), (5, 1)),        // 벡터 외적
            ((10, 10), (10, 10)),    // 정사각 행렬
            ((100, 50), (50, 200)),  // 큰 행렬
        ];
        
        for ((m, k), (k2, n)) in test_cases {
            assert_eq!(k, k2, "Matrix dimension mismatch");
            
            let a = create_test_matrix(m, k, TestPattern::Random)?;
            let b = create_test_matrix(k, n, TestPattern::Random)?;
            
            // RBE 행렬곱셈
            let result = a.matmul(&b)?;
            
            // 참조 구현 (BLAS 사용)
            let expected = reference_matmul(&a, &b)?;
            
            // 수치적 정확성 검증
            verify_tensor_close(&result, &expected, 1e-5)?;
            
            // 성능 검증 (큰 행렬의 경우)
            if m * n * k > 1000000 {
                benchmark_matmul_performance(&a, &b)?;
            }
        }
        
        Ok(())
    }
    
    fn benchmark_matmul_performance(a: &RBETensor, b: &RBETensor) -> Result<()> {
        let iterations = 10;
        let start = std::time::Instant::now();
        
        for _ in 0..iterations {
            let _ = a.matmul(b)?;
        }
        
        let duration = start.elapsed();
        let avg_time = duration / iterations;
        
        // 성능 기준 (예: 1 GFLOPS 이상)
        let ops = 2 * a.shape()[0] * a.shape()[1] * b.shape()[1];
        let gflops = ops as f64 / (avg_time.as_secs_f64() * 1e9);
        
        assert!(gflops > 1.0, 
               "Matrix multiplication too slow: {:.2} GFLOPS", gflops);
        
        println!("MatMul performance: {:.2} GFLOPS", gflops);
        Ok(())
    }
}
```

### 2.2 브로드캐스팅 검증

```rust
#[cfg(test)]
mod broadcasting_verification {
    use super::*;
    
    #[test]
    fn 브로드캐스팅_규칙_검증() -> Result<()> {
        // NumPy 브로드캐스팅 규칙과 동일한지 검증
        let test_cases = vec![
            // (shape1, shape2, expected_result_shape, should_succeed)
            (vec![3, 4], vec![4], vec![3, 4], true),
            (vec![3, 1], vec![4], vec![3, 4], true),
            (vec![3, 1], vec![1, 4], vec![3, 4], true),
            (vec![3, 4], vec![2, 4], vec![], false), // 불가능한 브로드캐스팅
            (vec![2, 1, 4], vec![3, 1], vec![2, 3, 4], true),
        ];
        
        for (shape1, shape2, expected, should_succeed) in test_cases {
            let result = broadcast_shape_calculation(&shape1, &shape2);
            
            if should_succeed {
                assert!(result.is_ok(), 
                       "Broadcasting should succeed for {:?} and {:?}", shape1, shape2);
                assert_eq!(result.unwrap(), expected,
                          "Broadcasting result mismatch");
            } else {
                assert!(result.is_err(),
                       "Broadcasting should fail for {:?} and {:?}", shape1, shape2);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn 브로드캐스팅_연산_결과_검증() -> Result<()> {
        // 실제 브로드캐스팅 연산 결과가 예상과 일치하는지 검증
        let a = RBETensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let b = RBETensor::new(vec![10.0, 20.0], vec![1, 2])?;
        
        let result = a.add(&b)?;
        
        // 수동으로 계산한 예상 결과
        let expected_data = vec![
            11.0, 21.0,  // [1] + [10, 20]
            12.0, 22.0,  // [2] + [10, 20] 
            13.0, 23.0,  // [3] + [10, 20]
        ];
        let expected = RBETensor::new(expected_data, vec![3, 2])?;
        
        verify_tensor_close(&result, &expected, 1e-6)?;
        
        Ok(())
    }
}
```

## 3. 압축/해제 정확성 검증

### 3.1 RBE 인코딩/디코딩 검증

```rust
#[cfg(test)]
mod compression_verification {
    use super::*;
    
    #[test]
    fn rbe_압축_복원_정확성_검증() -> Result<()> {
        let quality_grades = vec![
            QualityGrade::S,  // 최고 품질
            QualityGrade::A,  // 고품질
            QualityGrade::B,  // 중품질
            QualityGrade::C,  // 저품질
        ];
        
        for grade in quality_grades {
            let original_weights = generate_realistic_weights(1000)?;
            
            // 압축
            let compressed = compress_weights(&original_weights, grade.clone())?;
            
            // 해제
            let restored_weights = decompress_weights(&compressed)?;
            
            // 정확성 검증
            let rmse = compute_rmse(&original_weights, &restored_weights);
            let max_error = grade.max_allowable_error();
            
            assert!(rmse <= max_error,
                   "RMSE {:.2e} exceeds limit {:.2e} for grade {:?}",
                   rmse, max_error, grade);
            
            // 압축률 검증
            let compression_ratio = original_weights.len() as f32 * 4.0 / compressed.serialized_size() as f32;
            let min_ratio = grade.min_compression_ratio();
            
            assert!(compression_ratio >= min_ratio,
                   "Compression ratio {:.1}:1 below minimum {:.1}:1 for grade {:?}",
                   compression_ratio, min_ratio, grade);
                   
            println!("Grade {:?}: RMSE={:.2e}, Ratio={:.1}:1", 
                    grade, rmse, compression_ratio);
        }
        
        Ok(())
    }
    
    fn generate_realistic_weights(size: usize) -> Result<Vec<f32>> {
        // 실제 신경망 가중치와 유사한 분포 생성
        use rand_distr::{Normal, Distribution};
        let normal = Normal::new(0.0, 0.1)?;
        let mut rng = rand::thread_rng();
        
        Ok((0..size).map(|_| normal.sample(&mut rng)).collect())
    }
    
    fn compute_rmse(original: &[f32], restored: &[f32]) -> f32 {
        assert_eq!(original.len(), restored.len());
        
        let mse: f32 = original.iter()
            .zip(restored.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
            
        mse.sqrt()
    }
}
```

### 3.2 디코딩리스 연산 검증

```rust
#[cfg(test)]
mod decoding_less_verification {
    use super::*;
    
    #[test]
    fn 디코딩리스_행렬곱셈_정확성_검증() -> Result<()> {
        let sizes = vec![(64, 128), (256, 512), (512, 1024)];
        
        for (input_size, output_size) in sizes {
            // 원본 가중치 생성
            let original_weights = generate_test_matrix(output_size, input_size)?;
            let input_vector = generate_test_vector(input_size)?;
            
            // 전통적 방법: 압축 해제 후 연산
            let traditional_result = traditional_matmul(&original_weights, &input_vector)?;
            
            // RBE 방법: 압축 상태에서 직접 연산
            let compressed_weights = compress_matrix(&original_weights, QualityGrade::A)?;
            let rbe_result = decoding_less_matmul(&compressed_weights, &input_vector)?;
            
            // 결과 비교
            let relative_error = compute_relative_error(&traditional_result, &rbe_result);
            assert!(relative_error < 1e-3,
                   "Decoding-less operation error {:.2e} too high", relative_error);
                   
            // 성능 비교
            let speedup = benchmark_speedup(&original_weights, &compressed_weights, &input_vector)?;
            assert!(speedup > 1.0, "No performance improvement detected");
            
            println!("Size {}x{}: Error={:.2e}, Speedup={:.2}x", 
                    output_size, input_size, relative_error, speedup);
        }
        
        Ok(())
    }
    
    fn benchmark_speedup(
        original: &[f32], 
        compressed: &CompressedMatrix, 
        input: &[f32]
    ) -> Result<f32> {
        let iterations = 100;
        
        // 전통적 방법 벤치마크
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = traditional_matmul(original, input)?;
        }
        let traditional_time = start.elapsed();
        
        // RBE 방법 벤치마크
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = decoding_less_matmul(compressed, input)?;
        }
        let rbe_time = start.elapsed();
        
        Ok(traditional_time.as_secs_f32() / rbe_time.as_secs_f32())
    }
}
```

## 4. 자동미분 시스템 검증

### 4.1 그래디언트 정확성 검증

```rust
#[cfg(test)]
mod autodiff_verification {
    use super::*;
    
    #[test]
    fn 그래디언트_계산_정확성_검증() -> Result<()> {
        let test_functions = vec![
            TestFunction::Simple(|x| x.pow(2.0)),           // f(x) = x²
            TestFunction::Composite(|x| x.exp().sin()),     // f(x) = sin(exp(x))
            TestFunction::MultiVar(|x, y| x.mul(y).sum()),  // f(x,y) = sum(x*y)
        ];
        
        for test_fn in test_functions {
            // RBE 자동미분 결과
            let (output, rbe_grad) = compute_rbe_gradient(&test_fn)?;
            
            // 수치적 미분 결과 (참조)
            let numerical_grad = compute_numerical_gradient(&test_fn)?;
            
            // 정확성 검증
            let gradient_error = compute_relative_error(&rbe_grad, &numerical_grad);
            assert!(gradient_error < 1e-4,
                   "Gradient error {:.2e} too high for {:?}", 
                   gradient_error, test_fn);
                   
            println!("Function {:?}: Gradient error = {:.2e}", 
                    test_fn, gradient_error);
        }
        
        Ok(())
    }
    
    fn compute_numerical_gradient(func: &TestFunction) -> Result<Vec<f32>> {
        const H: f32 = 1e-5;  // 작은 perturbation
        
        // 중앙 차분법을 사용한 수치적 미분
        // grad_i = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
        unimplemented!("Numerical gradient computation")
    }
}
```

### 4.2 비트 자동미분 검증

```rust
#[cfg(test)]
mod bit_autodiff_verification {
    use super::*;
    
    #[test]
    fn 비트_레벨_그래디언트_검증() -> Result<()> {
        // 11비트 미분 사이클 검증
        let mut cycle_state = DifferentialCycle::new();
        let test_input = generate_test_bit_pattern(128)?;
        
        // 전체 사이클 실행
        let mut accumulated_gradient = 0.0f64;
        
        for cycle_idx in 0..2048 {
            cycle_state.cycle_index = cycle_idx as u16;
            
            // 순전파
            let forward_result = cycle_state.forward_step(&test_input)?;
            
            // 역전파
            let bit_gradient = cycle_state.backward_step(forward_result)?;
            accumulated_gradient += bit_gradient;
            
            // 상태 전이
            cycle_state.transition(bit_gradient)?;
            
            // 중간 검증
            if cycle_idx % 256 == 0 {
                verify_cycle_state_consistency(&cycle_state)?;
            }
        }
        
        // 최종 그래디언트 검증
        let final_gradient = cycle_state.get_accumulated_gradient();
        let expected_range = (-1e-3, 1e-3);  // 예상 범위
        
        assert!(final_gradient >= expected_range.0 && 
                final_gradient <= expected_range.1,
               "Final gradient {:.2e} outside expected range", final_gradient);
        
        Ok(())
    }
    
    fn verify_cycle_state_consistency(state: &DifferentialCycle) -> Result<()> {
        // 상태 일관성 검사
        assert!(state.cycle_index < 2048, "Cycle index out of range");
        assert!(state.bit_state & 0x7FF == state.bit_state, "Invalid bit state");
        assert!(state.accumulator.is_finite(), "Accumulator not finite");
        
        // 오차 누적 제한 검사
        assert!(state.accumulator.abs() < 1e3, "Accumulator overflow");
        
        Ok(())
    }
}
```

## 5. NLP 레이어 검증

### 5.1 LayerNorm 검증

```rust
#[cfg(test)]
mod layernorm_verification {
    use super::*;
    
    #[test]
    fn 레이어노름_수학적_정확성_검증() -> Result<()> {
        let test_configs = vec![
            LayerNormConfig { shape: vec![4], eps: 1e-5 },
            LayerNormConfig { shape: vec![64], eps: 1e-5 },
            LayerNormConfig { shape: vec![768], eps: 1e-5 },
        ];
        
        for config in test_configs {
            let layer_norm = RBELayerNorm::new(config.shape.clone(), config.eps)?;
            
            // 다양한 입력으로 테스트
            let test_inputs = vec![
                generate_normal_input(&config.shape)?,      // 정규분포
                generate_extreme_input(&config.shape)?,     // 극값
                generate_zero_input(&config.shape)?,        // 0 입력
                generate_constant_input(&config.shape)?,    // 상수 입력
            ];
            
            for (i, input) in test_inputs.iter().enumerate() {
                let output = layer_norm.forward(input)?;
                
                // 기본 속성 검증
                assert_eq!(output.shape(), input.shape(), 
                          "Shape changed in LayerNorm");
                
                // 통계적 속성 검증
                verify_layernorm_properties(&output, config.eps, i)?;
                
                // 참조 구현과 비교
                let reference_output = reference_layernorm(input, &config)?;
                verify_tensor_close(&output, &reference_output, 1e-5)?;
            }
        }
        
        Ok(())
    }
    
    fn verify_layernorm_properties(
        output: &RBETensor, 
        eps: f64, 
        test_case: usize
    ) -> Result<()> {
        let data = &output.data;
        let n = data.len() as f64;
        
        // 평균이 0에 가까운지 검증
        let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / n;
        assert!(mean.abs() < 1e-5, 
               "Mean {:.2e} not close to 0 in test case {}", mean, test_case);
        
        // 분산이 1에 가까운지 검증 (eps 고려)
        let variance: f64 = data.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / n;
        assert!((variance - 1.0).abs() < 1e-4,
               "Variance {:.6} not close to 1 in test case {}", variance, test_case);
        
        // 수치적 안정성 검증 (NaN, Inf 없음)
        assert!(data.iter().all(|&x| x.is_finite()),
               "Non-finite values detected in test case {}", test_case);
        
        Ok(())
    }
}
```

### 5.2 Attention 메커니즘 검증

```rust
#[cfg(test)]
mod attention_verification {
    use super::*;
    
    #[test]
    fn 어텐션_수학적_정확성_검증() -> Result<()> {
        let configs = vec![
            AttentionConfig { d_model: 64, num_heads: 4, seq_len: 16 },
            AttentionConfig { d_model: 256, num_heads: 8, seq_len: 64 },
            AttentionConfig { d_model: 512, num_heads: 16, seq_len: 128 },
        ];
        
        for config in configs {
            let attention = MultiHeadAttention::new(config.clone())?;
            
            // 테스트 입력 생성
            let input = generate_test_sequence(
                config.seq_len, 
                config.d_model
            )?;
            
            // Attention 연산
            let (output, attention_weights) = attention.forward(&input)?;
            
            // 기본 속성 검증
            verify_attention_output_properties(&output, &input, &config)?;
            verify_attention_weights_properties(&attention_weights, &config)?;
            
            // 수학적 정확성 검증
            verify_scaled_dot_product_attention(&input, &attention_weights, &output, &config)?;
            
            // Causal masking 검증 (GPT 스타일)
            verify_causal_masking(&attention_weights, config.seq_len)?;
        }
        
        Ok(())
    }
    
    fn verify_attention_weights_properties(
        weights: &RBETensor, 
        config: &AttentionConfig
    ) -> Result<()> {
        let expected_shape = vec![
            config.num_heads, 
            config.seq_len, 
            config.seq_len
        ];
        assert_eq!(weights.shape(), expected_shape.as_slice());
        
        // Softmax 속성 검증: 각 행의 합이 1
        let tolerance = 1e-5;
        for head in 0..config.num_heads {
            for i in 0..config.seq_len {
                let row_start = head * config.seq_len * config.seq_len + i * config.seq_len;
                let row_end = row_start + config.seq_len;
                let row_sum: f32 = weights.data[row_start..row_end].iter().sum();
                
                assert!((row_sum - 1.0).abs() < tolerance,
                       "Attention weights row sum {:.6} != 1.0", row_sum);
            }
        }
        
        // 모든 값이 0~1 범위
        assert!(weights.data.iter().all(|&x| x >= 0.0 && x <= 1.0),
               "Attention weights outside [0,1] range");
        
        Ok(())
    }
    
    fn verify_causal_masking(
        weights: &RBETensor, 
        seq_len: usize
    ) -> Result<()> {
        // 상삼각 부분이 0인지 확인 (causal masking)
        let num_heads = weights.shape()[0];
        
        for head in 0..num_heads {
            for i in 0..seq_len {
                for j in (i+1)..seq_len {
                    let idx = head * seq_len * seq_len + i * seq_len + j;
                    let weight_val = weights.data[idx];
                    
                    assert!(weight_val.abs() < 1e-7,
                           "Causal masking violated: weight[{},{},{}] = {:.2e}", 
                           head, i, j, weight_val);
                }
            }
        }
        
        Ok(())
    }
}
```

## 6. 성능 검증

### 6.1 메모리 효율성 검증

```rust
#[cfg(test)]
mod memory_verification {
    use super::*;
    
    #[test]
    fn 메모리_사용량_검증() -> Result<()> {
        let model_sizes = vec![
            ModelSize::Tiny   { params: 1_000_000 },      // 1M parameters
            ModelSize::Small  { params: 10_000_000 },     // 10M parameters  
            ModelSize::Medium { params: 100_000_000 },    // 100M parameters
            ModelSize::Large  { params: 1_000_000_000 },  // 1B parameters
        ];
        
        for model_size in model_sizes {
            println!("Testing memory efficiency for {} parameters", model_size.params());
            
            // 메모리 사용량 측정
            let memory_before = get_memory_usage()?;
            
            // 모델 생성 (압축됨)
            let compressed_model = create_compressed_model(model_size.clone())?;
            
            let memory_after = get_memory_usage()?;
            let memory_used = memory_after - memory_before;
            
            // 이론적 메모리 사용량 계산
            let uncompressed_size = model_size.params() * 4; // f32 = 4 bytes
            let compression_ratio = compressed_model.compression_ratio();
            let expected_size = uncompressed_size as f32 / compression_ratio;
            
            // 검증
            let actual_ratio = uncompressed_size as f32 / memory_used as f32;
            let efficiency = actual_ratio / compression_ratio;
            
            assert!(efficiency > 0.8, 
                   "Memory efficiency {:.2} below 80% for {} params", 
                   efficiency, model_size.params());
            
            println!("  Compression: {:.1}:1, Efficiency: {:.1}%", 
                    actual_ratio, efficiency * 100.0);
        }
        
        Ok(())
    }
    
    fn get_memory_usage() -> Result<usize> {
        // Platform-specific memory usage measurement
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")?;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: usize = line.split_whitespace()
                        .nth(1)
                        .unwrap()
                        .parse()?;
                    return Ok(kb * 1024); // Convert to bytes
                }
            }
        }
        
        Ok(0) // Fallback
    }
}
```

### 6.2 추론 성능 검증

```rust
#[cfg(test)]
mod inference_performance_verification {
    use super::*;
    
    #[test]
    fn 추론_속도_검증() -> Result<()> {
        let test_configurations = vec![
            InferenceConfig {
                batch_size: 1,
                seq_len: 128,
                model_size: ModelSize::Small,
                target_tokens_per_sec: 100.0,
            },
            InferenceConfig {
                batch_size: 8,
                seq_len: 256,
                model_size: ModelSize::Medium,
                target_tokens_per_sec: 500.0,
            },
        ];
        
        for config in test_configurations {
            let model = load_test_model(config.model_size.clone())?;
            let input_batch = generate_test_batch(config.batch_size, config.seq_len)?;
            
            // 워밍업
            for _ in 0..3 {
                let _ = model.forward(&input_batch)?;
            }
            
            // 실제 성능 측정
            let iterations = 20;
            let start = std::time::Instant::now();
            
            for _ in 0..iterations {
                let _output = model.forward(&input_batch)?;
            }
            
            let total_time = start.elapsed();
            let avg_time = total_time / iterations;
            
            // 처리량 계산
            let total_tokens = config.batch_size * config.seq_len;
            let tokens_per_sec = total_tokens as f32 / avg_time.as_secs_f32();
            
            // 성능 기준 검증
            assert!(tokens_per_sec >= config.target_tokens_per_sec,
                   "Performance {:.1} tokens/sec below target {:.1} tokens/sec",
                   tokens_per_sec, config.target_tokens_per_sec);
            
            println!("Config {:?}: {:.1} tokens/sec (target: {:.1})", 
                    config, tokens_per_sec, config.target_tokens_per_sec);
        }
        
        Ok(())
    }
}
```

## 7. 통합 검증 프레임워크

### 7.1 전체 시스템 검증

```rust
#[cfg(test)]
mod integration_verification {
    use super::*;
    
    #[test]
    fn 전체_파이프라인_검증() -> Result<()> {
        // End-to-end 파이프라인 검증
        let test_scenarios = vec![
            Scenario::TextGeneration {
                prompt: "Hello, world!",
                max_length: 50,
                expected_quality: TextQuality::Coherent,
            },
            Scenario::QuestionAnswering {
                context: "The capital of France is Paris.",
                question: "What is the capital of France?",
                expected_answer: "Paris",
            },
        ];
        
        for scenario in test_scenarios {
            let result = run_end_to_end_test(&scenario)?;
            verify_scenario_result(&scenario, &result)?;
        }
        
        Ok(())
    }
    
    fn run_end_to_end_test(scenario: &Scenario) -> Result<TestResult> {
        match scenario {
            Scenario::TextGeneration { prompt, max_length, .. } => {
                let tokenizer = load_tokenizer()?;
                let model = load_rbe_model()?;
                
                // 토큰화
                let input_tokens = tokenizer.encode(prompt)?;
                
                // 추론
                let output_tokens = model.generate(&input_tokens, *max_length)?;
                
                // 디코딩
                let generated_text = tokenizer.decode(&output_tokens)?;
                
                Ok(TestResult::Text(generated_text))
            },
            // ... 다른 시나리오들
        }
    }
}
```

### 7.2 회귀 테스트

```rust
#[cfg(test)]
mod regression_verification {
    use super::*;
    
    #[test]
    fn 성능_회귀_검증() -> Result<()> {
        // 저장된 기준 성능과 비교
        let baseline_results = load_baseline_performance()?;
        let current_results = measure_current_performance()?;
        
        for (metric_name, baseline_value) in baseline_results {
            let current_value = current_results.get(&metric_name)
                .ok_or_else(|| anyhow::anyhow!("Missing metric: {}", metric_name))?;
            
            let regression_threshold = 0.1; // 10% 이하 성능 저하만 허용
            let relative_change = (baseline_value - current_value) / baseline_value;
            
            assert!(relative_change <= regression_threshold,
                   "Performance regression in {}: {:.1}% degradation",
                   metric_name, relative_change * 100.0);
            
            if relative_change < -0.05 { // 5% 이상 개선
                println!("Performance improvement in {}: {:.1}%", 
                        metric_name, -relative_change * 100.0);
            }
        }
        
        Ok(())
    }
}
```

## 8. 검증 자동화

### 8.1 CI/CD 통합

```yaml
# .github/workflows/verification.yml
name: RBE Verification

on: [push, pull_request]

jobs:
  verification:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run unit verification
      run: |
        cargo test --lib -- --nocapture \
          --test-threads=1 \
          verification::level_0
        
    - name: Run module verification
      run: |
        cargo test --lib -- --nocapture \
          verification::level_1
        
    - name: Run system verification
      run: |
        cargo test --integration -- --nocapture \
          verification::level_2
        
    - name: Run performance verification
      run: |
        cargo test --release -- --nocapture \
          verification::level_3
        
    - name: Generate verification report
      run: |
        cargo run --bin verification-report \
          --output-format html \
          --output-file verification-report.html
        
    - name: Upload verification report
      uses: actions/upload-artifact@v3
      with:
        name: verification-report
        path: verification-report.html
```

이 검증 방법론을 통해 RBE NLP 구현의 정확성과 성능을 체계적으로 보장할 수 있습니다. 