# RBE 기반 GPT-2 완전 구현 계획서

## Abstract

본 문서는 Riemannian Basis Encoding (RBE) 기반으로 GPT-2의 모든 레이어를 직접 구현하는 상세한 계획서이다. Candle 라이브러리 의존성을 제거하고, RBE 압축된 가중치를 활용한 순전파/역전파 알고리즘을 완전히 자체 구현한다. 각 레이어별 수학적 정의, RBE 적용 방법, 메모리 최적화 전략을 제시한다.

## 1. Introduction

### 1.1 목표 및 범위

“sLLM(소형 LLM)” 을 **RBE 커널만으로 완전히 구동**하려면 Linear(프로젝션) 외에 최소한 다음 레이어들이 추가되어야한다. 

| #  | 필수 레이어                           | 기능 & 수식                                       | **RBE-전용 구현**                        | CPU 지연(1 tok) |
| -- | -------------------------------- | --------------------------------------------- | ------------------------------------ | ------------- |
| 1  | **Embedding**                    | 토큰 id → $\mathbf{e}_t$                        | • 행별 RBE 패턴<br>• 잔차 K = ≤ 20         | 0.3 ms        |
| 2  | **Positional Encoding** (RoPE)   | 회전 행렬 곱                                       | • 사인·코사인 LUT 512 entry<br>• FP32 FMA | 0.1 ms        |
| 3  | **LayerNorm / RMSNorm**          | $\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$      | • row-wise reduce(F32) + scale/shift | 0.4 ms        |
| 4  | **Attention – Q, K, V 프로젝션**     | 3× RBE-Linear                                 | 이미 구현된 Fused RBE-GEMM                | 2.0 ms        |
| 5  | **Scaled Dot-Product Attention** | $\mathrm{softmax}(QK^\top/\sqrt{d})V$         | • AVX-512 FMA + exp LUT              | 1.1 ms        |
| 6  | **Attention Out 프로젝션**           | RBE-Linear                                    | Fused RBE-GEMM                       | 0.6 ms        |
| 7  | **Feed-Forward (FFN)**           | $xW_1\rightarrow\mathrm{GELU}\rightarrow W_2$ | • 두 번 RBE-GEMM<br>• GELU tanh approx | 1.8 ms        |
| 8  | **Dropout** (추론 시 생략)            | -                                             | 패스-스루                                | 0             |
| 9  | **Output Projection**            | 마지막 hidden → vocab                            | RBE 또는 Dense (정확도↑)                  | 0.9 ms (RBE)  |
| 10 | **Logits Softmax**               | prob = softmax(logits)                        | FP32 exp / reduce                    | 0.3 ms        |

> 합계: **\~7.5 ms / 토큰** (Dense FP16 기준 11-12 ms) –— **원본보다 빠름 & 메모리 –90 %**

---

#### 핵심 구현 메모

1. **RBE-Embedding**

   * 토큰 id → row index ⇒ 패턴 계산 + 소수 DCT 잔차 합산
   * LUT 캐싱: row-wise `d`, `cos(πx)` 값 128개
2. **RBE-GEMM 커널 재사용**

   * Q, K, V, Out, FFN\_W1, W2 모두 동일 Fused 커널 호출
3. **Softmax & GELU**

   * AVX-512: `exp_fast16` / Neon: `vexpq_f32`
   * FP32 누적 후 FP16 cast
4. **출력층**

   * 소형 모델(≤30 k vocab)은 RBE 버전도 가능
   * 대형 vocab은 Dense FP16 1 행만 복원해 속도 확보

---

### 체크리스트 (To-Implement)

1. `RBEEmbedding`, `RBEAttention`, `RBEFFN` Python module
2. Fused RBE-GEMM AVX-512 / Neon SIMD 커널
3. Graph Pass: `RBEWeight` 노드 자동 Fusion
4. Unit test – embedding round-trip, attention masking, FP32 parity
5. End-to-end mini-chatbot: prompt→response BLEU 비교


**Primary Objectives:**
- GPT-2 아키텍처의 모든 레이어를 RBE 기반으로 구현
- Candle/PyTorch 등 외부 딥러닝 프레임워크 의존성 완전 제거
- RBE 압축된 가중치의 실시간 압축해제 및 연산 최적화
- 메모리 효율적인 추론 엔진 구축

**Scope Definition:**
```
Layer Coverage: 100% (Embedding → Transformer Blocks → Output Head)
Implementation Language: Rust + CUDA (선택적)
Target Model: GPT-2 117M/345M/762M/1.5B
Compression Ratio: 50:1 ~ 3276:1 (RBE 설정에 따라)
```

### 1.2 아키텍처 개요

GPT-2 모델의 핵심 구성요소:

```
Input Tokens → Token Embedding → Position Embedding → 
[Transformer Block × N] → Layer Norm → Output Head → Logits
```

**Transformer Block 내부:**
```
x → Layer Norm → Multi-Head Attention → Residual → 
    Layer Norm → Feed-Forward → Residual → x'
```

## 2. RBE Layer Foundation

### 2.1 Base RBE Layer Interface

모든 레이어가 구현해야 할 공통 인터페이스:

```rust
pub trait RBELayer {
    type Input;
    type Output;
    type Params;
    
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input>;
    fn update_weights(&mut self, learning_rate: f32) -> Result<()>;
    fn compress_weights(&mut self) -> Result<CompressionStats>;
    fn decompress_weights(&self, name: &str) -> Result<Tensor>;
}
```

### 2.2 RBE Tensor 구조

```rust
pub struct RBETensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub compressed_blocks: Option<Vec<HybridEncodedBlock>>,
    pub device: Device,
}

impl RBETensor {
    pub fn matmul(&self, other: &RBETensor) -> Result<RBETensor>;
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<RBETensor>;
    pub fn reshape(&self, new_shape: &[usize]) -> Result<RBETensor>;
    pub fn softmax(&self, dim: usize) -> Result<RBETensor>;
    pub fn layer_norm(&self, eps: f32) -> Result<RBETensor>;
}
```

## 3. Embedding Layers Implementation

### 3.1 Token Embedding Layer

**수학적 정의:**
```
E_token: ℝ^V → ℝ^d
E_token(x_i) = W_emb[x_i, :]
```

여기서 V는 vocabulary size, d는 embedding dimension이다.

**RBE 적용 전략:**
- 임베딩 매트릭스 W_emb (V×d)를 블록 단위로 RBE 압축
- 블록 크기: 64×64 또는 128×128 (메모리-성능 트레이드오프)
- 압축률: 약 100:1 ~ 500:1

```rust
pub struct RBETokenEmbedding {
    vocab_size: usize,
    embed_dim: usize,
    compressed_weights: Vec<HybridEncodedBlock>,
    weight_layout: ModelLayout,
    cache: HashMap<String, RBETensor>,
}

impl RBELayer for RBETokenEmbedding {
    type Input = Vec<u32>;  // token indices
    type Output = RBETensor; // [seq_len, embed_dim]
    
    fn forward(&self, token_ids: &Vec<u32>) -> Result<RBETensor> {
        // 1. RBE 압축된 임베딩 행렬에서 해당 행들 압축해제
        let mut embeddings = Vec::new();
        
        for &token_id in token_ids {
            let row_tensor = self.get_embedding_row(token_id)?;
            embeddings.push(row_tensor);
        }
        
        // 2. 배치 차원으로 결합
        RBETensor::stack(&embeddings, 0)
    }
}
```

### 3.2 Positional Embedding Layer

**수학적 정의:**
```
E_pos: ℝ^L → ℝ^d
E_pos(pos_i) = W_pos[pos_i, :]
```

**구현 세부사항:**
```rust
impl RBEPositionalEmbedding {
    fn forward(&self, seq_len: usize, offset: usize) -> Result<RBETensor> {
        let mut pos_embeddings = Vec::new();
        
        for i in 0..seq_len {
            let pos_id = offset + i;
            let pos_emb = self.get_position_embedding(pos_id)?;
            pos_embeddings.push(pos_emb);
        }
        
        RBETensor::stack(&pos_embeddings, 0)
    }
}
```

## 4. Layer Normalization Implementation

### 4.1 수학적 정의

Layer Normalization의 정의:
```
LN(x) = γ ⊙ (x - μ) / σ + β
```

여기서:
- μ = mean(x), σ = std(x) (feature dimension에서)
- γ, β는 학습 가능한 파라미터

### 4.2 RBE Layer Norm 구현

```rust
pub struct RBELayerNorm {
    normalized_shape: Vec<usize>,
    gamma: RBETensor,  // scale parameter
    beta: RBETensor,   // shift parameter
    eps: f32,
}

impl RBELayer for RBELayerNorm {
    fn forward(&self, x: &RBETensor) -> Result<RBETensor> {
        // 1. 통계량 계산
        let dims = x.shape.len();
        let feature_dim = dims - 1;
        
        let mean = x.mean_along_dim(feature_dim)?;
        let variance = x.variance_along_dim(feature_dim, &mean)?;
        let std = variance.sqrt_in_place()?;
        
        // 2. 정규화
        let normalized = (x.sub(&mean)?)?.div(&std.add_scalar(self.eps)?)?;
        
        // 3. Affine transformation
        let scaled = normalized.mul(&self.gamma)?;
        scaled.add(&self.beta)
    }
    
    fn backward(&mut self, grad_output: &RBETensor) -> Result<RBETensor> {
        // Layer norm gradient 계산
        // ∂L/∂x = (1/σ) * [∂L/∂y - mean(∂L/∂y) - (x-μ)/σ² * mean((x-μ) ⊙ ∂L/∂y)]
        unimplemented!("Layer norm backward pass")
    }
}
```

## 5. Multi-Head Self-Attention Implementation

### 5.1 수학적 정의

Multi-Head Attention의 핵심 수식:
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
MultiHead(x) = Concat(head_1, ..., head_h)W^O
head_i = Attention(xW_i^Q, xW_i^K, xW_i^V)
```

### 5.2 RBE Multi-Head Attention 구현

```rust
pub struct RBEMultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    
    // RBE 압축된 projection matrices
    query_projection: RBELinear,
    key_projection: RBELinear,
    value_projection: RBELinear,
    output_projection: RBELinear,
    
    dropout_prob: f32,
    causal_mask: bool,
}

impl RBELayer for RBEMultiHeadAttention {
    fn forward(&self, x: &RBETensor) -> Result<RBETensor> {
        let (batch_size, seq_len, embed_dim) = x.shape3()?;
        
        // 1. QKV projections
        let q = self.query_projection.forward(x)?;  // [B, L, D]
        let k = self.key_projection.forward(x)?;
        let v = self.value_projection.forward(x)?;
        
        // 2. Reshape to multi-head format
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?; // [B, H, L, D/H]
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;
        
        // 3. Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)?)? * scale;
        
        // 4. Causal masking (for GPT-2)
        let masked_scores = if self.causal_mask {
            self.apply_causal_mask(&scores)?
        } else {
            scores
        };
        
        // 5. Softmax
        let attention_weights = masked_scores.softmax(-1)?;
        
        // 6. Apply attention to values
        let attention_output = attention_weights.matmul(&v)?;
        
        // 7. Concatenate heads
        let concat_output = attention_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, embed_dim])?;
        
        // 8. Final projection
        self.output_projection.forward(&concat_output)
    }
    
    fn apply_causal_mask(&self, scores: &RBETensor) -> Result<RBETensor> {
        let seq_len = scores.shape[2];
        let mut mask = RBETensor::zeros(&[seq_len, seq_len])?;
        
        // Upper triangular mask
        for i in 0..seq_len {
            for j in (i+1)..seq_len {
                mask.set_element(&[i, j], f32::NEG_INFINITY)?;
            }
        }
        
        scores.add(&mask)
    }
}
```

### 5.3 Attention 최적화 전략

**메모리 최적화:**
```rust
impl RBEMultiHeadAttention {
    fn forward_optimized(&self, x: &RBETensor, kv_cache: &mut KVCache) -> Result<RBETensor> {
        // KV 캐싱으로 추론 시 메모리 사용량 감소
        if let Some((cached_k, cached_v)) = kv_cache.get_cached() {
            // Incremental attention computation
            self.forward_incremental(x, cached_k, cached_v)
        } else {
            // Full attention computation
            let result = self.forward(x)?;
            kv_cache.update(k.clone(), v.clone());
            Ok(result)
        }
    }
}
```

## 6. Feed-Forward Network (MLP) Implementation

### 6.1 수학적 정의

GPT-2 MLP는 다음과 같이 정의된다:
```
MLP(x) = GELU(xW_1 + b_1)W_2 + b_2
```

여기서:
- W_1: [d_model, d_ff] (일반적으로 d_ff = 4 * d_model)
- W_2: [d_ff, d_model]
- GELU: Gaussian Error Linear Unit activation

### 6.2 RBE MLP 구현

```rust
pub struct RBEMLP {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    
    // RBE 압축된 레이어들
    fc1: RBELinear,  // input_dim -> hidden_dim
    fc2: RBELinear,  // hidden_dim -> output_dim
    dropout_prob: f32,
}

impl RBELayer for RBEMLP {
    fn forward(&self, x: &RBETensor) -> Result<RBETensor> {
        // 1. First linear transformation
        let hidden = self.fc1.forward(x)?;
        
        // 2. GELU activation
        let activated = self.gelu(&hidden)?;
        
        // 3. Dropout (training 시에만)
        let dropped = if self.training {
            self.dropout(&activated)?
        } else {
            activated
        };
        
        // 4. Second linear transformation
        self.fc2.forward(&dropped)
    }
    
    fn gelu(&self, x: &RBETensor) -> Result<RBETensor> {
        // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let x_cubed = x.pow(3.0)?;
        let inner = x.add(&x_cubed.mul_scalar(0.044715)?)?;
        let scaled = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
        let tanh_result = scaled.tanh()?;
        let one_plus_tanh = tanh_result.add_scalar(1.0)?;
        let half_x = x.mul_scalar(0.5)?;
        
        half_x.mul(&one_plus_tanh)
    }
}
```

## 7. RBE Linear Layer Implementation

### 7.1 핵심 Linear Layer

모든 projection에서 사용되는 기본 RBE Linear layer:

```rust
pub struct RBELinear {
    input_dim: usize,
    output_dim: usize,
    
    // RBE 압축된 가중치
    compressed_weight: Vec<HybridEncodedBlock>,
    weight_layout: WeightLayout,
    bias: Option<RBETensor>,
    
    // 런타임 캐시
    weight_cache: Option<RBETensor>,
    use_cache: bool,
}

impl RBELinear {
    pub fn new(
        input_dim: usize, 
        output_dim: usize,
        compressed_weight: Vec<HybridEncodedBlock>,
        bias: Option<RBETensor>
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            compressed_weight,
            bias,
            weight_cache: None,
            use_cache: true,
        }
    }
    
    fn get_weight(&mut self) -> Result<&RBETensor> {
        if self.use_cache && self.weight_cache.is_some() {
            return Ok(self.weight_cache.as_ref().unwrap());
        }
        
        // RBE 압축해제
        let decompressed = self.decompress_weight()?;
        
        if self.use_cache {
            self.weight_cache = Some(decompressed);
            Ok(self.weight_cache.as_ref().unwrap())
        } else {
            // 캐시 없이 임시 반환 (메모리 절약)
            Ok(&decompressed)
        }
    }
    
    fn decompress_weight(&self) -> Result<RBETensor> {
        // 병렬 블록 압축해제
        let mut decompressed_data = vec![0.0f32; self.input_dim * self.output_dim];
        
        self.compressed_weight.par_iter().enumerate().try_for_each(|(i, block)| {
            let block_data = block.decode();
            let start_idx = i * block.rows * block.cols;
            let end_idx = start_idx + block_data.len();
            
            if end_idx <= decompressed_data.len() {
                decompressed_data[start_idx..end_idx].copy_from_slice(&block_data);
            }
            
            Ok::<(), anyhow::Error>(())
        })?;
        
        Ok(RBETensor::new(
            decompressed_data,
            vec![self.output_dim, self.input_dim]
        ))
    }
}

impl RBELayer for RBELinear {
    fn forward(&self, x: &RBETensor) -> Result<RBETensor> {
        let weight = self.get_weight()?;
        
        // Matrix multiplication: x @ W^T + b
        let output = x.matmul(&weight.transpose(0, 1)?)?;
        
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}
```

## 8. Complete Transformer Block

### 8.1 Transformer Block Assembly

```rust
pub struct RBETransformerBlock {
    layer_idx: usize,
    
    // Sub-layers
    attention_norm: RBELayerNorm,
    attention: RBEMultiHeadAttention,
    mlp_norm: RBELayerNorm,
    mlp: RBEMLP,
    
    // Residual dropout
    dropout_prob: f32,
}

impl RBELayer for RBETransformerBlock {
    fn forward(&self, x: &RBETensor) -> Result<RBETensor> {
        // 1. Pre-attention layer norm
        let normed_x = self.attention_norm.forward(x)?;
        
        // 2. Multi-head attention
        let attention_output = self.attention.forward(&normed_x)?;
        
        // 3. Residual connection
        let x_after_attention = x.add(&attention_output)?;
        
        // 4. Pre-MLP layer norm
        let normed_x2 = self.mlp_norm.forward(&x_after_attention)?;
        
        // 5. MLP
        let mlp_output = self.mlp.forward(&normed_x2)?;
        
        // 6. Final residual connection
        x_after_attention.add(&mlp_output)
    }
}
```

## 9. Complete GPT-2 Model

### 9.1 Full Model Assembly

```rust
pub struct RBEGPT2 {
    config: GPT2Config,
    
    // Embedding layers
    token_embedding: RBETokenEmbedding,
    position_embedding: RBEPositionalEmbedding,
    embedding_dropout: f32,
    
    // Transformer blocks
    transformer_blocks: Vec<RBETransformerBlock>,
    
    // Final components
    final_layer_norm: RBELayerNorm,
    output_head: RBELinear,  // tied with token embedding
    
    // KV cache for inference
    kv_cache: Vec<KVCache>,
}

impl RBEGPT2 {
    pub fn forward(&mut self, 
                   input_ids: &[u32], 
                   past_length: usize) -> Result<RBETensor> {
        let seq_len = input_ids.len();
        
        // 1. Token embedding
        let token_embeds = self.token_embedding.forward(input_ids)?;
        
        // 2. Position embedding
        let pos_embeds = self.position_embedding.forward(seq_len, past_length)?;
        
        // 3. Combine embeddings
        let mut hidden_states = token_embeds.add(&pos_embeds)?;
        
        // 4. Embedding dropout
        if self.training {
            hidden_states = self.dropout(&hidden_states)?;
        }
        
        // 5. Transformer blocks
        for (i, block) in self.transformer_blocks.iter_mut().enumerate() {
            hidden_states = block.forward(&hidden_states)?;
            
            // Update KV cache for this layer
            if !self.training {
                self.kv_cache[i].update_from_block(block)?;
            }
        }
        
        // 6. Final layer norm
        let normalized = self.final_layer_norm.forward(&hidden_states)?;
        
        // 7. Output projection (language modeling head)
        self.output_head.forward(&normalized)
    }
    
    pub fn generate(&mut self, 
                    prompt_tokens: &[u32],
                    max_tokens: usize,
                    temperature: f32,
                    top_p: f32) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut current_tokens = prompt_tokens.to_vec();
        
        for _ in 0..max_tokens {
            // Forward pass
            let logits = self.forward(&current_tokens[current_tokens.len()-1..], 
                                    current_tokens.len() - 1)?;
            
            // Sample next token
            let next_token = self.sample_token(&logits, temperature, top_p)?;
            generated.push(next_token);
            current_tokens.push(next_token);
            
            // Check for EOS
            if next_token == self.config.eos_token_id {
                break;
            }
        }
        
        Ok(generated)
    }
}
```

## 10. Performance Optimization & Memory Management

### 10.1 메모리 최적화 전략

**1. Selective Weight Caching:**
```rust
pub struct CachePolicy {
    pub cache_embeddings: bool,     // 임베딩은 자주 재사용
    pub cache_attention_weights: bool,  // Attention weights 캐싱
    pub cache_mlp_weights: bool,    // MLP weights는 큰 메모리 사용
    pub max_cache_size_mb: usize,
}

impl RBEGPT2 {
    fn optimize_memory_usage(&mut self) -> Result<()> {
        let available_memory = self.get_available_memory()?;
        
        if available_memory < 4 * 1024 * 1024 * 1024 { // 4GB 미만
            // Low memory mode: 캐시 비활성화
            self.disable_weight_caching();
            self.enable_streaming_decompression();
        } else {
            // Normal mode: 선택적 캐싱
            self.configure_selective_caching(available_memory);
        }
        
        Ok(())
    }
}
```

**2. Streaming Decompression:**
```rust
impl RBELinear {
    fn forward_streaming(&self, x: &RBETensor) -> Result<RBETensor> {
        // 블록 단위로 스트리밍 압축해제 및 연산
        let output_shape = [x.shape[0], self.output_dim];
        let mut output = RBETensor::zeros(&output_shape)?;
        
        for block_idx in 0..self.compressed_weight.len() {
            let weight_block = self.decompress_block(block_idx)?;
            let partial_output = x.matmul_partial(&weight_block)?;
            output.add_partial(&partial_output, block_idx)?;
        }
        
        Ok(output)
    }
}
```

### 10.2 CUDA 가속 (선택적)

```rust
#[cfg(feature = "cuda")]
mod cuda_kernels {
    use cudarc::driver::*;
    
    pub fn rbe_decompress_cuda(
        compressed_blocks: &[HybridEncodedBlock],
        output: &mut [f32],
        stream: &CudaStream
    ) -> Result<()> {
        // CUDA 커널로 병렬 압축해제
        let kernel = stream.load_kernel("rbe_decompress_kernel")?;
        kernel.launch(
            LaunchConfig::for_num_elems(compressed_blocks.len() as u32),
            (compressed_blocks, output)
        )?;
        
        stream.synchronize()?;
        Ok(())
    }
    
    pub fn attention_cuda(
        q: &RBETensor,
        k: &RBETensor, 
        v: &RBETensor,
        output: &mut RBETensor,
        stream: &CudaStream
    ) -> Result<()> {
        // Flash Attention style CUDA kernel
        unimplemented!("CUDA attention kernel")
    }
}
```

### 10.3 성능 벤치마킹

```rust
pub struct PerformanceMetrics {
    pub tokens_per_second: f32,
    pub memory_usage_mb: usize,
    pub compression_ratio: f32,
    pub decompression_time_ms: f32,
    pub cache_hit_rate: f32,
}

impl RBEGPT2 {
    pub fn benchmark(&mut self, test_prompts: &[&str]) -> Result<PerformanceMetrics> {
        let start_time = std::time::Instant::now();
        let mut total_tokens = 0;
        
        for prompt in test_prompts {
            let tokens = self.tokenize(prompt)?;
            let generated = self.generate(&tokens, 100, 0.8, 0.9)?;
            total_tokens += generated.len();
        }
        
        let elapsed = start_time.elapsed().as_secs_f32();
        
        Ok(PerformanceMetrics {
            tokens_per_second: total_tokens as f32 / elapsed,
            memory_usage_mb: self.get_memory_usage()?,
            compression_ratio: self.calculate_compression_ratio()?,
            decompression_time_ms: self.measure_decompression_time()?,
            cache_hit_rate: self.get_cache_statistics()?.hit_rate,
        })
    }
}
```
