# 기본 레이어 구현 가이드

## 개요

이 문서는 NLP 모델의 핵심 구성요소인 Layer Normalization, Embedding, Activation 함수들의 구현 방법을 다룹니다.

## LayerNorm 구현

### 수학적 정의

Layer Normalization은 다음과 같이 정의됩니다:

```
LN(x) = γ ⊙ (x - μ) / σ + β
```

여기서:
- μ = mean(x, dim=-1): 마지막 차원에서의 평균
- σ = std(x, dim=-1): 마지막 차원에서의 표준편차
- γ: learnable scale parameter
- β: learnable shift parameter

### RBE LayerNorm 구현

```rust
use crate::nlp::tensor::RBETensor;
use anyhow::Result;

#[derive(Debug)]
pub struct RBELayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,  // 수치 안정성을 위해 f64 사용
    
    // 학습 가능한 매개변수
    gamma: RBETensor,  // scale parameter (initialized to 1.0)
    beta: RBETensor,   // shift parameter (initialized to 0.0)
    
    // 최적화 설정
    use_fused_ops: bool,
    cache_statistics: bool,
}

impl RBELayerNorm {
    /// 새로운 LayerNorm 레이어 생성
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Result<Self> {
        let total_elements: usize = normalized_shape.iter().product();
        
        // γ = 1, β = 0으로 초기화
        let gamma = RBETensor::ones(&normalized_shape)?;
        let beta = RBETensor::zeros(&normalized_shape)?;
        
        Ok(Self {
            normalized_shape,
            eps,
            gamma,
            beta,
            use_fused_ops: true,
            cache_statistics: false,
        })
    }
    
    /// 순전파
    pub fn forward(&self, input: &RBETensor) -> Result<RBETensor> {
        self.validate_input(input)?;
        
        if self.use_fused_ops {
            self.forward_fused(input)
        } else {
            self.forward_standard(input)
        }
    }
    
    /// 최적화된 융합 연산 순전파
    fn forward_fused(&self, input: &RBETensor) -> Result<RBETensor> {
        let input_shape = input.shape();
        let batch_dims = &input_shape[..input_shape.len() - self.normalized_shape.len()];
        let feature_dims = &input_shape[input_shape.len() - self.normalized_shape.len()..];
        
        let batch_size: usize = batch_dims.iter().product();
        let feature_size: usize = feature_dims.iter().product();
        
        let mut output_data = vec![0.0f32; input.data.len()];
        
        // 배치별 병렬 정규화
        use rayon::prelude::*;
        output_data.par_chunks_mut(feature_size)
            .zip(input.data.par_chunks(feature_size))
            .try_for_each(|(output_chunk, input_chunk)| -> Result<()> {
                self.normalize_chunk(input_chunk, output_chunk)?;
                Ok(())
            })?;
        
        let mut output = RBETensor::new(output_data, input_shape.to_vec())?;
        
        // 자동미분 설정
        if input.requires_grad {
            output.requires_grad = true;
            output.is_leaf = false;
            // grad_fn 설정은 추후 구현
        }
        
        Ok(output)
    }
    
    /// 단일 청크 정규화 (SIMD 최적화)
    fn normalize_chunk(&self, input_chunk: &[f32], output_chunk: &mut [f32]) -> Result<()> {
        let n = input_chunk.len() as f64;
        
        // 1. 평균 계산 (Kahan summation for numerical stability)
        let mut sum = 0.0f64;
        let mut c = 0.0f64;  // compensation
        
        for &x in input_chunk {
            let y = x as f64 - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        let mean = sum / n;
        
        // 2. 분산 계산
        let mut var_sum = 0.0f64;
        let mut var_c = 0.0f64;
        
        for &x in input_chunk {
            let diff = x as f64 - mean;
            let y = diff * diff - var_c;
            let t = var_sum + y;
            var_c = (t - var_sum) - y;
            var_sum = t;
        }
        let variance = var_sum / n;
        let std_dev = (variance + self.eps).sqrt();
        
        // 3. 정규화 + scale/shift 적용
        for (i, (&x, output)) in input_chunk.iter().zip(output_chunk.iter_mut()).enumerate() {
            let normalized = ((x as f64 - mean) / std_dev) as f32;
            
            // gamma와 beta 적용 (broadcasting 고려)
            let gamma_idx = i % self.gamma.data.len();
            let beta_idx = i % self.beta.data.len();
            
            *output = normalized * self.gamma.data[gamma_idx] + self.beta.data[beta_idx];
        }
        
        Ok(())
    }
    
    /// 표준 순전파 (디버깅용)
    fn forward_standard(&self, input: &RBETensor) -> Result<RBETensor> {
        // 1. 평균 계산
        let mean = self.compute_mean(input)?;
        
        // 2. 분산 계산
        let variance = self.compute_variance(input, &mean)?;
        
        // 3. 정규화
        let centered = input.sub(&mean)?;
        let std_dev = variance.add_scalar(self.eps as f32)?.sqrt()?;
        let normalized = centered.div(&std_dev)?;
        
        // 4. Scale and shift
        let scaled = normalized.mul(&self.gamma)?;
        let output = scaled.add(&self.beta)?;
        
        Ok(output)
    }
    
    /// 평균 계산 (마지막 차원들에서)
    fn compute_mean(&self, input: &RBETensor) -> Result<RBETensor> {
        let mut result = input.clone();
        
        // 마지막 normalized_shape.len() 개 차원에서 평균 계산
        for _ in 0..self.normalized_shape.len() {
            let last_dim = result.shape().len() - 1;
            result = result.mean_dim(last_dim)?;
        }
        
        // Broadcasting을 위해 차원 확장
        for _ in 0..self.normalized_shape.len() {
            result = result.unsqueeze(-1)?;
        }
        
        Ok(result)
    }
    
    /// 분산 계산
    fn compute_variance(&self, input: &RBETensor, mean: &RBETensor) -> Result<RBETensor> {
        let centered = input.sub(mean)?;
        let squared = centered.pow(2.0)?;
        
        let mut result = squared;
        for _ in 0..self.normalized_shape.len() {
            let last_dim = result.shape().len() - 1;
            result = result.mean_dim(last_dim)?;
        }
        
        // Broadcasting을 위해 차원 확장
        for _ in 0..self.normalized_shape.len() {
            result = result.unsqueeze(-1)?;
        }
        
        Ok(result)
    }
    
    /// 입력 검증
    fn validate_input(&self, input: &RBETensor) -> Result<()> {
        let input_shape = input.shape();
        let expected_suffix = &self.normalized_shape;
        
        if input_shape.len() < expected_suffix.len() {
            return Err(anyhow::anyhow!(
                "Input has {} dimensions, but LayerNorm expects at least {}",
                input_shape.len(), expected_suffix.len()
            ));
        }
        
        let actual_suffix = &input_shape[input_shape.len() - expected_suffix.len()..];
        if actual_suffix != expected_suffix {
            return Err(anyhow::anyhow!(
                "Input shape suffix {:?} doesn't match normalized_shape {:?}",
                actual_suffix, expected_suffix
            ));
        }
        
        Ok(())
    }
}
```

## Embedding 구현

### Token Embedding

```rust
#[derive(Debug)]
pub struct TokenEmbedding {
    vocab_size: usize,
    embed_dim: usize,
    weight: RBETensor,  // [vocab_size, embed_dim]
    padding_idx: Option<usize>,
}

impl TokenEmbedding {
    /// 새로운 토큰 임베딩 생성
    pub fn new(vocab_size: usize, embed_dim: usize, padding_idx: Option<usize>) -> Result<Self> {
        // Xavier 초기화
        let std = (2.0 / (vocab_size + embed_dim) as f32).sqrt();
        let mut weight = RBETensor::randn(&[vocab_size, embed_dim])?;
        weight = weight.mul_scalar(std)?;
        
        // padding_idx가 있으면 해당 임베딩을 0으로 설정
        if let Some(pad_idx) = padding_idx {
            for i in 0..embed_dim {
                weight.data[pad_idx * embed_dim + i] = 0.0;
            }
        }
        
        Ok(Self {
            vocab_size,
            embed_dim,
            weight,
            padding_idx,
        })
    }
    
    /// 순전파
    pub fn forward(&self, input: &[u32]) -> Result<RBETensor> {
        let seq_len = input.len();
        let mut output_data = Vec::with_capacity(seq_len * self.embed_dim);
        
        for &token_id in input {
            if token_id as usize >= self.vocab_size {
                return Err(anyhow::anyhow!(
                    "Token ID {} out of vocabulary range [0, {})",
                    token_id, self.vocab_size
                ));
            }
            
            let start_idx = token_id as usize * self.embed_dim;
            let end_idx = start_idx + self.embed_dim;
            output_data.extend_from_slice(&self.weight.data[start_idx..end_idx]);
        }
        
        let output = RBETensor::new(output_data, vec![seq_len, self.embed_dim])?;
        Ok(output)
    }
    
    /// 배치 순전파
    pub fn forward_batch(&self, input: &[Vec<u32>]) -> Result<RBETensor> {
        let batch_size = input.len();
        if batch_size == 0 {
            return Err(anyhow::anyhow!("Empty batch"));
        }
        
        let seq_len = input[0].len();
        
        // 모든 시퀀스가 같은 길이인지 확인
        for (i, seq) in input.iter().enumerate() {
            if seq.len() != seq_len {
                return Err(anyhow::anyhow!(
                    "Sequence {} has length {}, expected {}",
                    i, seq.len(), seq_len
                ));
            }
        }
        
        let mut output_data = Vec::with_capacity(batch_size * seq_len * self.embed_dim);
        
        for seq in input {
            for &token_id in seq {
                if token_id as usize >= self.vocab_size {
                    return Err(anyhow::anyhow!(
                        "Token ID {} out of vocabulary range",
                        token_id
                    ));
                }
                
                let start_idx = token_id as usize * self.embed_dim;
                let end_idx = start_idx + self.embed_dim;
                output_data.extend_from_slice(&self.weight.data[start_idx..end_idx]);
            }
        }
        
        let output = RBETensor::new(output_data, vec![batch_size, seq_len, self.embed_dim])?;
        Ok(output)
    }
}
```

### Position Embedding

```rust
#[derive(Debug)]
pub struct PositionalEmbedding {
    max_len: usize,
    embed_dim: usize,
    weight: RBETensor,  // [max_len, embed_dim]
}

impl PositionalEmbedding {
    /// 새로운 위치 임베딩 생성
    pub fn new(max_len: usize, embed_dim: usize) -> Result<Self> {
        let weight = RBETensor::randn(&[max_len, embed_dim])?;
        
        Ok(Self {
            max_len,
            embed_dim,
            weight,
        })
    }
    
    /// Sinusoidal 위치 임베딩 생성 (학습되지 않음)
    pub fn new_sinusoidal(max_len: usize, embed_dim: usize) -> Result<Self> {
        let mut pos_encoding = vec![0.0f32; max_len * embed_dim];
        
        for pos in 0..max_len {
            for i in 0..embed_dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
                
                if i % 2 == 0 {
                    pos_encoding[pos * embed_dim + i] = angle.sin();
                } else {
                    pos_encoding[pos * embed_dim + i] = angle.cos();
                }
            }
        }
        
        let weight = RBETensor::new(pos_encoding, vec![max_len, embed_dim])?;
        
        Ok(Self {
            max_len,
            embed_dim,
            weight,
        })
    }
    
    /// 순전파
    pub fn forward(&self, seq_len: usize, offset: usize) -> Result<RBETensor> {
        if offset + seq_len > self.max_len {
            return Err(anyhow::anyhow!(
                "Position {} + {} exceeds max_len {}",
                offset, seq_len, self.max_len
            ));
        }
        
        let start_idx = offset * self.embed_dim;
        let end_idx = (offset + seq_len) * self.embed_dim;
        
        let output_data = self.weight.data[start_idx..end_idx].to_vec();
        let output = RBETensor::new(output_data, vec![seq_len, self.embed_dim])?;
        
        Ok(output)
    }
}
```

## Activation 함수들

### GELU Activation

```rust
pub struct GELU;

impl GELU {
    /// GELU 활성화 함수
    /// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn forward(input: &RBETensor) -> Result<RBETensor> {
        let output_data: Vec<f32> = input.data.iter()
            .map(|&x| Self::gelu_scalar(x))
            .collect();
        
        let mut output = RBETensor::new(output_data, input.shape().to_vec())?;
        
        if input.requires_grad {
            output.requires_grad = true;
            output.is_leaf = false;
            // GELU backward 구현 필요
        }
        
        Ok(output)
    }
    
    /// 스칼라 GELU 계산
    #[inline]
    fn gelu_scalar(x: f32) -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.7978845608; // √(2/π)
        const COEFF: f32 = 0.044715;
        
        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        0.5 * x * (1.0 + inner.tanh())
    }
    
    /// 빠른 근사 GELU (더 빠른 성능)
    pub fn forward_fast(input: &RBETensor) -> Result<RBETensor> {
        let output_data: Vec<f32> = input.data.iter()
            .map(|&x| Self::gelu_fast_scalar(x))
            .collect();
        
        RBETensor::new(output_data, input.shape().to_vec())
    }
    
    #[inline]
    fn gelu_fast_scalar(x: f32) -> f32 {
        0.5 * x * (1.0 + (1.702 * x).tanh())
    }
}
```

### Softmax Activation

```rust
pub struct Softmax {
    dim: isize,  // 어느 차원에서 softmax 적용할지
}

impl Softmax {
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
    
    /// Softmax 순전파 (수치적으로 안정한 구현)
    pub fn forward(&self, input: &RBETensor) -> Result<RBETensor> {
        let actual_dim = if self.dim < 0 {
            (input.shape().len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };
        
        if actual_dim >= input.shape().len() {
            return Err(anyhow::anyhow!(
                "Softmax dimension {} out of range for tensor with {} dimensions",
                actual_dim, input.shape().len()
            ));
        }
        
        // 수치적 안정성을 위해 max 값 빼기
        let max_vals = self.compute_max_along_dim(input, actual_dim)?;
        let shifted = input.sub(&max_vals)?;
        
        // exp 계산
        let exp_vals = shifted.exp()?;
        
        // sum 계산
        let sum_vals = self.compute_sum_along_dim(&exp_vals, actual_dim)?;
        
        // normalize
        let output = exp_vals.div(&sum_vals)?;
        
        Ok(output)
    }
    
    fn compute_max_along_dim(&self, input: &RBETensor, dim: usize) -> Result<RBETensor> {
        // 구현 필요: 특정 차원에서의 max 계산
        // 지금은 placeholder
        input.clone()
    }
    
    fn compute_sum_along_dim(&self, input: &RBETensor, dim: usize) -> Result<RBETensor> {
        // 구현 필요: 특정 차원에서의 sum 계산
        // 지금은 placeholder
        input.clone()
    }
}
```

## 레이어 조합 예제

### 간단한 MLP 블록

```rust
#[derive(Debug)]
pub struct MLPBlock {
    layer_norm: RBELayerNorm,
    linear1: RBELinear,
    activation: GELU,
    linear2: RBELinear,
    dropout_prob: f32,
}

impl MLPBlock {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let layer_norm = RBELayerNorm::new(vec![hidden_size], 1e-5)?;
        
        // Linear layers는 이미 구현된 RBELinear 사용
        let linear1 = RBELinear::from_dense_weights(
            &random_weights(hidden_size, intermediate_size),
            hidden_size, intermediate_size,
            None, 64, 256
        )?;
        
        let linear2 = RBELinear::from_dense_weights(
            &random_weights(intermediate_size, hidden_size),
            intermediate_size, hidden_size,
            None, 64, 256
        )?;
        
        Ok(Self {
            layer_norm,
            linear1,
            activation: GELU,
            linear2,
            dropout_prob: 0.1,
        })
    }
    
    pub fn forward(&self, input: &RBETensor) -> Result<RBETensor> {
        // 1. Layer normalization
        let normed = self.layer_norm.forward(input)?;
        
        // 2. First linear transformation
        let hidden = self.linear1.forward_tensor(&normed)?;
        
        // 3. Activation
        let activated = GELU::forward(&hidden)?;
        
        // 4. Second linear transformation
        let output = self.linear2.forward_tensor(&activated)?;
        
        // 5. Residual connection
        let result = input.add(&output)?;
        
        Ok(result)
    }
}

// 헬퍼 함수
fn random_weights(rows: usize, cols: usize) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..rows * cols).map(|_| rng.gen_range(-0.1..0.1)).collect()
}
```

## 테스트 프레임워크

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn 레이어노름_기본_테스트() -> Result<()> {
        let layer_norm = RBELayerNorm::new(vec![4], 1e-5)?;
        let input = RBETensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;
        
        let output = layer_norm.forward(&input)?;
        
        // 출력의 평균은 0에 가깝고 분산은 1에 가까워야 함
        let mean = output.data.iter().sum::<f32>() / output.data.len() as f32;
        assert!((mean.abs() < 1e-6), "Mean should be close to 0, got {}", mean);
        
        Ok(())
    }
    
    #[test]
    fn 토큰_임베딩_테스트() -> Result<()> {
        let embedding = TokenEmbedding::new(1000, 256, Some(0))?;
        let tokens = vec![1, 2, 3, 0];  // 0은 padding
        
        let output = embedding.forward(&tokens)?;
        
        assert_eq!(output.shape(), &[4, 256]);
        
        // padding 토큰의 임베딩은 모두 0이어야 함
        let padding_embedding = &output.data[3 * 256..4 * 256];
        assert!(padding_embedding.iter().all(|&x| x == 0.0));
        
        Ok(())
    }
    
    #[test]
    fn gelu_테스트() -> Result<()> {
        let input = RBETensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5])?;
        let output = GELU::forward(&input)?;
        
        // GELU(0) ≈ 0
        assert!((output.data[2].abs() < 1e-6), "GELU(0) should be close to 0");
        
        // GELU는 단조증가함수
        for i in 0..4 {
            assert!(output.data[i] < output.data[i + 1]);
        }
        
        Ok(())
    }
}
```

## 성능 최적화 팁

### 1. LayerNorm 최적화
- 융합 연산으로 메모리 접근 최소화
- Kahan summation으로 수치 안정성 확보
- 병렬 처리로 배치 처리 가속화

### 2. Embedding 최적화
- 임베딩 테이블 캐싱
- Sparse gradient 업데이트
- 메모리 정렬 최적화

### 3. Activation 최적화
- 룩업 테이블 활용
- SIMD 명령어 사용
- 함수 근사로 계산 가속화

이 가이드를 기반으로 `src/nlp/layers/` 디렉토리에 각 레이어를 구현하면 됩니다. 