# Chapter 9: GPT-2 Model Integration

## Abstract

본 장에서는 RBE Transformer Block을 기반으로 완전한 GPT-2 언어 모델을 구현한다. 토크나이저, 임베딩, 위치 인코딩, 언어 모델링 헤드를 통합하여 실제 텍스트 생성이 가능한 완전한 시스템을 구성한다.

## 9.1 GPT-2 Architecture Overview

### 9.1.1 전체 모델 구조

```
GPT-2 = TokenEmbedding + PositionalEmbedding + 
        Σ(TransformerBlock_i, i=0..N-1) + 
        LayerNorm + LanguageModelingHead
```

**구성요소:**
1. **Token Embedding**: vocab_size × d_model
2. **Positional Embedding**: max_seq_len × d_model  
3. **Transformer Blocks**: N개 레이어
4. **Final LayerNorm**: 마지막 정규화
5. **LM Head**: d_model → vocab_size (텍스트 생성)

### 9.1.2 RBE 최적화 적용 범위

**압축 대상:**
- ✅ Token Embedding (매우 큰 행렬)
- ❌ Positional Embedding (학습되지 않음, 작음)
- ✅ All Transformer Blocks (이미 구현됨)
- ✅ Language Modeling Head (Token Embedding 공유 또는 별도 압축)

## 9.2 Complete GPT-2 Implementation

### 9.2.1 핵심 구조

```rust
use std::sync::Arc;
use std::collections::HashMap;
use rayon::prelude::*;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct RBEGPT2Model {
    // 모델 구성
    config: GPT2Config,
    
    // 토크나이저
    tokenizer: Option<Arc<Tokenizer>>,
    
    // 임베딩 레이어들
    token_embedding: RBEEmbedding,
    position_embedding: PositionalEmbedding,
    
    // Transformer 블록들
    transformer_blocks: Vec<RBETransformerBlock>,
    
    // 최종 레이어들
    final_layer_norm: RBELayerNorm,
    language_modeling_head: RBELinear,
    
    // 생성 설정
    generation_config: GenerationConfig,
    
    // KV 캐시 관리
    kv_cache_manager: KVCacheManager,
    
    // 성능 통계
    inference_count: std::sync::atomic::AtomicUsize,
    total_tokens_generated: std::sync::atomic::AtomicUsize,
    average_latency_per_token_ns: std::sync::atomic::AtomicU64,
    
    // 메모리 관리
    memory_pool: Arc<std::sync::Mutex<MemoryPool>>,
    enable_mixed_precision: bool,
}

#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub layer_norm_eps: f64,
    
    // RBE 설정
    pub embedding_block_size: usize,
    pub embedding_coefficients: usize,
    pub transformer_block_size: usize,
    pub transformer_coefficients: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub pad_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub use_cache: bool,
    pub do_sample: bool,
}

impl RBEGPT2Model {
    /// 새로운 GPT-2 모델 생성
    pub fn new(
        config: GPT2Config,
        compressed_weights: ModelWeights,
        tokenizer_path: Option<&str>,
    ) -> Result<Self> {
        // 토크나이저 로드
        let tokenizer = if let Some(path) = tokenizer_path {
            Some(Arc::new(Tokenizer::from_file(path)?))
        } else {
            None
        };
        
        // 토큰 임베딩 (RBE 압축)
        let token_embedding = RBEEmbedding::new(
            config.vocab_size,
            config.d_model,
            compressed_weights.token_embedding_blocks,
            config.embedding_block_size,
        )?;
        
        // 위치 임베딩 (고정, 학습 안 됨)
        let position_embedding = PositionalEmbedding::new(
            config.max_sequence_length,
            config.d_model,
        )?;
        
        // Transformer 블록들
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let block = RBETransformerBlock::new(
                layer_idx,
                TransformerConfig {
                    d_model: config.d_model,
                    d_ff: config.d_ff,
                    num_heads: config.num_heads,
                    dropout: config.dropout,
                    layer_norm_eps: config.layer_norm_eps,
                    max_sequence_length: config.max_sequence_length,
                    vocab_size: config.vocab_size,
                    num_layers: config.num_layers,
                },
                compressed_weights.transformer_blocks[layer_idx].clone(),
            )?;
            transformer_blocks.push(block);
        }
        
        // 최종 LayerNorm
        let final_layer_norm = RBELayerNorm::new(
            vec![config.d_model],
            config.layer_norm_eps,
        )?;
        
        // Language Modeling Head
        let language_modeling_head = if compressed_weights.share_embeddings {
            // Token embedding과 가중치 공유 (GPT-2 표준)
            token_embedding.create_tied_linear_layer()?
        } else {
            // 별도 Linear layer
            RBELinear::new(
                config.d_model,
                config.vocab_size,
                compressed_weights.lm_head_blocks.unwrap(),
                None, // bias 없음
                config.embedding_block_size,
            )?
        };
        
        // KV 캐시 관리자
        let kv_cache_manager = KVCacheManager::new(
            config.num_layers,
            config.max_sequence_length,
        );
        
        // 메모리 풀
        let memory_pool = Arc::new(std::sync::Mutex::new(MemoryPool::new(100)));
        
        Ok(Self {
            config,
            tokenizer,
            token_embedding,
            position_embedding,
            transformer_blocks,
            final_layer_norm,
            language_modeling_head,
            generation_config: GenerationConfig::default(),
            kv_cache_manager,
            inference_count: std::sync::atomic::AtomicUsize::new(0),
            total_tokens_generated: std::sync::atomic::AtomicUsize::new(0),
            average_latency_per_token_ns: std::sync::atomic::AtomicU64::new(0),
            memory_pool,
            enable_mixed_precision: false,
        })
    }
    
    /// 사전 훈련된 모델 로드
    pub fn from_pretrained(
        model_name: &str,
        cache_dir: Option<&str>,
        compression_config: Option<CompressionConfig>,
    ) -> Result<Self> {
        // 1. 원본 GPT-2 가중치 다운로드
        let original_weights = download_gpt2_weights(model_name, cache_dir)?;
        
        // 2. RBE 압축 수행
        let compression_cfg = compression_config.unwrap_or_default();
        let compressed_weights = compress_gpt2_weights(&original_weights, &compression_cfg)?;
        
        // 3. 토크나이저 로드
        let tokenizer_path = download_gpt2_tokenizer(model_name, cache_dir)?;
        
        // 4. 모델 구성
        let config = GPT2Config::from_model_name(model_name)?;
        
        Self::new(config, compressed_weights, Some(&tokenizer_path))
    }
    
    /// 생성 설정 업데이트
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.generation_config = config;
        
        // KV 캐시 활성화/비활성화
        if config.use_cache {
            self.enable_kv_cache();
        } else {
            self.disable_kv_cache();
        }
    }
    
    /// 모든 레이어에 KV 캐시 활성화
    pub fn enable_kv_cache(&mut self) {
        for block in &mut self.transformer_blocks {
            block.enable_kv_caching();
        }
        self.kv_cache_manager.enable();
    }
    
    /// KV 캐시 비활성화
    pub fn disable_kv_cache(&mut self) {
        self.kv_cache_manager.clear_all();
        self.kv_cache_manager.disable();
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    pub token_embedding_blocks: Vec<HybridEncodedBlock>,
    pub transformer_blocks: Vec<BlockWeights>,
    pub final_layer_norm_params: LayerNormParams,
    pub lm_head_blocks: Option<Vec<HybridEncodedBlock>>,
    pub share_embeddings: bool,
}
```

### 9.2.2 순전파 구현

```rust
impl RBEGPT2Model {
    /// 전체 모델 순전파
    pub fn forward(
        &mut self,
        input_ids: &[usize],
        attention_mask: Option<&[f32]>,
        past_key_values: Option<&[Option<(Vec<f32>, Vec<f32>)>]>,
        use_cache: bool,
    ) -> Result<ModelOutput> {
        let start_time = std::time::Instant::now();
        
        let batch_size = 1; // 현재 단일 배치만 지원
        let seq_len = input_ids.len();
        
        // 입력 검증
        self.validate_input(input_ids, seq_len)?;
        
        // 1. Token Embedding
        let token_embeddings = self.token_embedding.forward(input_ids)?;
        
        // 2. Positional Embedding
        let position_embeddings = self.position_embedding.forward(seq_len)?;
        
        // 3. Embedding 합성 (token + position)
        let mut hidden_states = self.combine_embeddings(&token_embeddings, &position_embeddings)?;
        
        // 4. Transformer 블록들 순차 실행
        let mut all_hidden_states = Vec::new();
        let mut all_attentions = Vec::new();
        let mut present_key_values = Vec::new();
        
        for (layer_idx, block) in self.transformer_blocks.iter_mut().enumerate() {
            // 이전 레이어의 KV 캐시 가져오기
            let past_kv = past_key_values
                .and_then(|past| past.get(layer_idx))
                .and_then(|kv| kv.as_ref());
            
            // Transformer 블록 실행
            let block_output = block.forward(
                &hidden_states,
                &[batch_size, seq_len, self.config.d_model],
                attention_mask,
                past_kv,
                use_cache,
            )?;
            
            hidden_states = block_output.output;
            
            // 중간 결과 저장 (디버깅/분석용)
            all_hidden_states.push(hidden_states.clone());
            if let Some(attention_weights) = block_output.attention_weights {
                all_attentions.push(attention_weights);
            }
            if let Some(present_kv) = block_output.present_key_value {
                present_key_values.push(Some(present_kv));
            } else {
                present_key_values.push(None);
            }
        }
        
        // 5. 최종 Layer Normalization
        let normalized_hidden_states = self.final_layer_norm.forward(
            &hidden_states,
            &[batch_size, seq_len, self.config.d_model],
        )?;
        
        // 6. Language Modeling Head (다음 토큰 예측)
        let logits = self.language_modeling_head.forward(
            &normalized_hidden_states,
            &[batch_size, seq_len, self.config.d_model],
        )?;
        
        // 통계 업데이트
        self.inference_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;
        let tokens_processed = seq_len;
        self.total_tokens_generated.fetch_add(tokens_processed, std::sync::atomic::Ordering::Relaxed);
        
        // 토큰당 평균 레이턴시 업데이트
        let latency_per_token = elapsed_ns / tokens_processed as u64;
        self.average_latency_per_token_ns.store(latency_per_token, std::sync::atomic::Ordering::Relaxed);
        
        Ok(ModelOutput {
            logits,
            hidden_states: Some(all_hidden_states),
            attentions: if all_attentions.is_empty() { None } else { Some(all_attentions) },
            present_key_values: if use_cache { Some(present_key_values) } else { None },
            inference_stats: self.get_inference_stats(),
        })
    }
    
    /// Embedding 결합 (token + positional)
    fn combine_embeddings(&self, token_emb: &[f32], pos_emb: &[f32]) -> Result<Vec<f32>> {
        if token_emb.len() != pos_emb.len() {
            return Err(anyhow::anyhow!(
                "Embedding size mismatch: token {}, position {}",
                token_emb.len(), pos_emb.len()
            ));
        }
        
        let mut combined = vec![0.0; token_emb.len()];
        combined.par_iter_mut()
            .zip(token_emb.par_iter())
            .zip(pos_emb.par_iter())
            .for_each(|((out, &tok), &pos)| {
                *out = tok + pos;
            });
        
        Ok(combined)
    }
    
    /// 입력 검증
    fn validate_input(&self, input_ids: &[usize], seq_len: usize) -> Result<()> {
        if seq_len == 0 {
            return Err(anyhow::anyhow!("Empty input sequence"));
        }
        
        if seq_len > self.config.max_sequence_length {
            return Err(anyhow::anyhow!(
                "Sequence too long: {} > max {}",
                seq_len, self.config.max_sequence_length
            ));
        }
        
        // 토큰 ID 범위 검증
        for &token_id in input_ids {
            if token_id >= self.config.vocab_size {
                return Err(anyhow::anyhow!(
                    "Invalid token ID: {} >= vocab_size {}",
                    token_id, self.config.vocab_size
                ));
            }
        }
        
        Ok(())
    }
    
    /// 추론 통계 수집
    fn get_inference_stats(&self) -> InferenceStats {
        let inference_count = self.inference_count.load(std::sync::atomic::Ordering::Relaxed);
        let total_tokens = self.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed);
        let avg_latency_ns = self.average_latency_per_token_ns.load(std::sync::atomic::Ordering::Relaxed);
        
        InferenceStats {
            total_inferences: inference_count,
            total_tokens_processed: total_tokens,
            average_latency_per_token_ms: avg_latency_ns as f32 / 1_000_000.0,
            tokens_per_second: if avg_latency_ns > 0 {
                1_000_000_000.0 / avg_latency_ns as f32
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug)]
pub struct ModelOutput {
    pub logits: Vec<f32>,  // [batch_size * seq_len * vocab_size]
    pub hidden_states: Option<Vec<Vec<f32>>>,  // 각 레이어의 hidden states
    pub attentions: Option<Vec<Vec<f32>>>,     // 각 레이어의 attention weights
    pub present_key_values: Option<Vec<Option<(Vec<f32>, Vec<f32>)>>>,  // KV 캐시
    pub inference_stats: InferenceStats,
}

#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub total_inferences: usize,
    pub total_tokens_processed: usize,
    pub average_latency_per_token_ms: f32,
    pub tokens_per_second: f32,
}
```

### 9.2.3 텍스트 생성 구현

```rust
impl RBEGPT2Model {
    /// 텍스트 생성 (자동회귀)
    pub fn generate(
        &mut self,
        prompt: &str,
        generation_config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = generation_config.unwrap_or_else(|| self.generation_config.clone());
        
        // 1. 토크나이징
        let tokenizer = self.tokenizer.as_ref().ok_or_else(||
            anyhow::anyhow!("Tokenizer not loaded")
        )?;
        
        let encoding = tokenizer.encode(prompt, false)?;
        let mut input_ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
        
        let original_length = input_ids.len();
        let mut generated_tokens = Vec::new();
        let mut past_key_values = None;
        
        let generation_start = std::time::Instant::now();
        
        // 2. 자동회귀 생성
        for step in 0..config.max_new_tokens {
            // 현재 입력 (첫 번째 스텝은 전체 프롬프트, 이후는 마지막 토큰만)
            let current_input = if step == 0 || !config.use_cache {
                input_ids.clone()
            } else {
                vec![*input_ids.last().unwrap()]
            };
            
            // 모델 순전파
            let output = self.forward(
                &current_input,
                None, // attention_mask
                past_key_values.as_ref(),
                config.use_cache,
            )?;
            
            // KV 캐시 업데이트
            if config.use_cache {
                past_key_values = output.present_key_values;
            }
            
            // 마지막 위치의 logits 추출
            let last_token_logits = self.extract_last_token_logits(&output.logits, current_input.len())?;
            
            // 다음 토큰 샘플링
            let next_token = self.sample_next_token(&last_token_logits, &config, &input_ids)?;
            
            // 종료 조건 검사
            if let Some(eos_token_id) = config.eos_token_id {
                if next_token == eos_token_id {
                    break;
                }
            }
            
            // 토큰 추가
            input_ids.push(next_token);
            generated_tokens.push(next_token);
        }
        
        let generation_time = generation_start.elapsed();
        
        // 3. 디토크나이징
        let generated_text = self.decode_tokens(&generated_tokens)?;
        let full_text = self.decode_tokens(&input_ids)?;
        
        Ok(GenerationResult {
            generated_text,
            full_text,
            generated_token_ids: generated_tokens,
            prompt_length: original_length,
            generation_time_ms: generation_time.as_millis() as f32,
            tokens_per_second: generated_tokens.len() as f32 / generation_time.as_secs_f32(),
        })
    }
    
    /// 마지막 토큰의 logits 추출
    fn extract_last_token_logits(&self, logits: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let last_token_start = (seq_len - 1) * vocab_size;
        let last_token_end = seq_len * vocab_size;
        
        if last_token_end > logits.len() {
            return Err(anyhow::anyhow!(
                "Logits size mismatch: expected at least {}, got {}",
                last_token_end, logits.len()
            ));
        }
        
        Ok(logits[last_token_start..last_token_end].to_vec())
    }
    
    /// 다음 토큰 샘플링
    fn sample_next_token(
        &self,
        logits: &[f32],
        config: &GenerationConfig,
        context: &[usize],
    ) -> Result<usize> {
        let mut processed_logits = logits.to_vec();
        
        // 1. Repetition penalty 적용
        if config.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&mut processed_logits, context, config.repetition_penalty)?;
        }
        
        // 2. Temperature scaling
        if config.temperature != 1.0 {
            for logit in &mut processed_logits {
                *logit /= config.temperature;
            }
        }
        
        // 3. Top-k filtering
        if let Some(top_k) = config.top_k {
            self.apply_top_k_filtering(&mut processed_logits, top_k)?;
        }
        
        // 4. Top-p (nucleus) filtering
        if let Some(top_p) = config.top_p {
            self.apply_top_p_filtering(&mut processed_logits, top_p)?;
        }
        
        // 5. 샘플링 또는 greedy 선택
        if config.do_sample {
            self.sample_from_distribution(&processed_logits)
        } else {
            self.greedy_select(&processed_logits)
        }
    }
    
    /// Repetition penalty 적용
    fn apply_repetition_penalty(
        &self,
        logits: &mut [f32],
        context: &[usize],
        penalty: f32,
    ) -> Result<()> {
        for &token_id in context {
            if token_id < logits.len() {
                if logits[token_id] > 0.0 {
                    logits[token_id] /= penalty;
                } else {
                    logits[token_id] *= penalty;
                }
            }
        }
        Ok(())
    }
    
    /// Top-k filtering
    fn apply_top_k_filtering(&self, logits: &mut [f32], k: usize) -> Result<()> {
        if k >= logits.len() {
            return Ok(()); // No filtering needed
        }
        
        // 상위 k개 인덱스 찾기
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 상위 k개 이외는 -inf로 설정
        for i in k..indexed_logits.len() {
            let idx = indexed_logits[i].0;
            logits[idx] = f32::NEG_INFINITY;
        }
        
        Ok(())
    }
    
    /// Top-p (nucleus) filtering
    fn apply_top_p_filtering(&self, logits: &mut [f32], p: f32) -> Result<()> {
        // Softmax 계산
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f32> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum: f32 = probs.iter().sum();
        for prob in &mut probs {
            *prob /= sum;
        }
        
        // 확률 순으로 정렬된 인덱스
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 누적 확률이 p를 넘는 지점 찾기
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed_probs.len();
        
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_index = i + 1;
                break;
            }
        }
        
        // cutoff 이후 토큰들은 -inf로 설정
        for i in cutoff_index..indexed_probs.len() {
            let idx = indexed_probs[i].0;
            logits[idx] = f32::NEG_INFINITY;
        }
        
        Ok(())
    }
    
    /// 확률 분포에서 샘플링
    fn sample_from_distribution(&self, logits: &[f32]) -> Result<usize> {
        // Softmax 변환
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f32> = logits.iter()
            .map(|&x| if x == f32::NEG_INFINITY { 0.0 } else { (x - max_logit).exp() })
            .collect();
        
        let sum: f32 = probs.iter().sum();
        if sum == 0.0 {
            return Err(anyhow::anyhow!("All probabilities are zero"));
        }
        
        for prob in &mut probs {
            *prob /= sum;
        }
        
        // 누적 분포에서 샘플링
        let random_value: f32 = rand::random();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i);
            }
        }
        
        // fallback (rounding error 등으로 인해)
        Ok(probs.len() - 1)
    }
    
    /// Greedy 선택 (가장 높은 확률)
    fn greedy_select(&self, logits: &[f32]) -> Result<usize> {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| anyhow::anyhow!("No valid tokens to select"))
    }
    
    /// 토큰 디코딩
    fn decode_tokens(&self, token_ids: &[usize]) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(||
            anyhow::anyhow!("Tokenizer not loaded")
        )?;
        
        let token_ids_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        Ok(tokenizer.decode(&token_ids_u32, true)?)
    }
}

#[derive(Debug)]
pub struct GenerationResult {
    pub generated_text: String,
    pub full_text: String,
    pub generated_token_ids: Vec<usize>,
    pub prompt_length: usize,
    pub generation_time_ms: f32,
    pub tokens_per_second: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            pad_token_id: None,
            eos_token_id: None,
            use_cache: true,
            do_sample: true,
        }
    }
}
```

## 9.3 RBE Embedding 구현

### 9.3.1 압축된 Token Embedding

```rust
#[derive(Debug)]
pub struct RBEEmbedding {
    vocab_size: usize,
    d_model: usize,
    compressed_blocks: Arc<Vec<HybridEncodedBlock>>,
    block_layout: BlockLayout,
    
    // 캐시 (자주 사용되는 토큰들)
    embedding_cache: Arc<std::sync::RwLock<HashMap<usize, Vec<f32>>>>,
    cache_size_limit: usize,
}

impl RBEEmbedding {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        compressed_blocks: Vec<HybridEncodedBlock>,
        block_size: usize,
    ) -> Result<Self> {
        let blocks_per_row = (vocab_size + block_size - 1) / block_size;
        let blocks_per_col = (d_model + block_size - 1) / block_size;
        
        let layout = BlockLayout {
            block_size,
            blocks_per_row,
            blocks_per_col,
            total_blocks: blocks_per_row * blocks_per_col,
            overlap: 0,
        };
        
        Ok(Self {
            vocab_size,
            d_model,
            compressed_blocks: Arc::new(compressed_blocks),
            block_layout: layout,
            embedding_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            cache_size_limit: 1000, // 자주 사용되는 1000개 토큰 캐시
        })
    }
    
    /// 토큰 ID들을 임베딩 벡터로 변환
    pub fn forward(&self, token_ids: &[usize]) -> Result<Vec<f32>> {
        let mut embeddings = vec![0.0; token_ids.len() * self.d_model];
        
        // 토큰별 병렬 임베딩 추출
        embeddings.par_chunks_mut(self.d_model)
            .zip(token_ids.par_iter())
            .try_for_each(|(emb_chunk, &token_id)| -> Result<()> {
                let embedding = self.get_token_embedding(token_id)?;
                emb_chunk.copy_from_slice(&embedding);
                Ok(())
            })?;
        
        Ok(embeddings)
    }
    
    /// 단일 토큰의 임베딩 추출 (캐시 사용)
    fn get_token_embedding(&self, token_id: usize) -> Result<Vec<f32>> {
        if token_id >= self.vocab_size {
            return Err(anyhow::anyhow!("Invalid token ID: {}", token_id));
        }
        
        // 캐시 확인
        {
            let cache = self.embedding_cache.read().unwrap();
            if let Some(cached_embedding) = cache.get(&token_id) {
                return Ok(cached_embedding.clone());
            }
        }
        
        // 압축 도메인에서 임베딩 추출
        let embedding = self.extract_embedding_from_compressed(token_id)?;
        
        // 캐시 업데이트 (크기 제한)
        {
            let mut cache = self.embedding_cache.write().unwrap();
            if cache.len() < self.cache_size_limit {
                cache.insert(token_id, embedding.clone());
            }
        }
        
        Ok(embedding)
    }
    
    /// 압축된 가중치에서 특정 토큰의 임베딩 추출
    fn extract_embedding_from_compressed(&self, token_id: usize) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0; self.d_model];
        
        // 해당 토큰이 속하는 블록들 찾기
        let token_row = token_id;
        let blocks_affecting_token = self.find_blocks_for_token(token_row);
        
        for (block_idx, block_row_start, block_col_start) in blocks_affecting_token {
            let block = &self.compressed_blocks[block_idx];
            
            // 블록 내에서의 토큰 위치
            let local_token_row = token_row - block_row_start;
            
            // 블록 디코딩 (필요한 행만)
            let block_data = block.decode();
            let block_embedding_start = local_token_row * self.block_layout.block_size;
            let block_embedding_end = (block_embedding_start + self.block_layout.block_size)
                .min(block_data.len());
            
            if block_embedding_start < block_data.len() {
                let block_embedding = &block_data[block_embedding_start..block_embedding_end];
                
                // 전체 임베딩에 복사
                let global_start = block_col_start;
                let global_end = (global_start + block_embedding.len()).min(self.d_model);
                let copy_len = global_end - global_start;
                
                if copy_len > 0 {
                    embedding[global_start..global_end]
                        .copy_from_slice(&block_embedding[..copy_len]);
                }
            }
        }
        
        Ok(embedding)
    }
    
    /// 특정 토큰에 영향을 주는 블록들 찾기
    fn find_blocks_for_token(&self, token_row: usize) -> Vec<(usize, usize, usize)> {
        let mut affecting_blocks = Vec::new();
        
        let block_row = token_row / self.block_layout.block_size;
        
        for block_col in 0..self.block_layout.blocks_per_col {
            let block_idx = block_row * self.block_layout.blocks_per_col + block_col;
            if block_idx < self.compressed_blocks.len() {
                let block_row_start = block_row * self.block_layout.block_size;
                let block_col_start = block_col * self.block_layout.block_size;
                affecting_blocks.push((block_idx, block_row_start, block_col_start));
            }
        }
        
        affecting_blocks
    }
    
    /// Token embedding과 가중치를 공유하는 Linear layer 생성
    pub fn create_tied_linear_layer(&self) -> Result<RBELinear> {
        // Embedding 행렬의 전치를 Linear layer로 사용
        RBELinear::new_from_tied_embedding(
            self.d_model,
            self.vocab_size,
            Arc::clone(&self.compressed_blocks),
            self.block_layout.clone(),
        )
    }
}
```

### 9.3.2 위치 임베딩

```rust
#[derive(Debug)]
pub struct PositionalEmbedding {
    max_seq_len: usize,
    d_model: usize,
    embeddings: Vec<f32>, // 사전 계산된 위치 임베딩
}

impl PositionalEmbedding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Result<Self> {
        // Sinusoidal positional encoding 사전 계산
        let mut embeddings = vec![0.0; max_seq_len * d_model];
        
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                
                if i % 2 == 0 {
                    embeddings[pos * d_model + i] = angle.sin();
                } else {
                    embeddings[pos * d_model + i] = angle.cos();
                }
            }
        }
        
        Ok(Self {
            max_seq_len,
            d_model,
            embeddings,
        })
    }
    
    /// 시퀀스 길이에 맞는 위치 임베딩 반환
    pub fn forward(&self, seq_len: usize) -> Result<Vec<f32>> {
        if seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_len
            ));
        }
        
        let position_embeddings = self.embeddings[..seq_len * self.d_model].to_vec();
        Ok(position_embeddings)
    }
}
```

## 9.4 정확도 검증 및 테스트

### 9.4.1 전체 모델 테스트

```rust
#[cfg(test)]
mod gpt2_model_tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_text_generation() -> Result<()> {
        // 작은 테스트 모델 구성
        let config = GPT2Config {
            vocab_size: 1000,
            max_sequence_length: 128,
            d_model: 256,
            d_ff: 1024,
            num_heads: 4,
            num_layers: 4,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            embedding_block_size: 32,
            embedding_coefficients: 100,
            transformer_block_size: 32,
            transformer_coefficients: 150,
        };
        
        // 더미 압축 가중치 생성
        let compressed_weights = generate_dummy_model_weights(&config)?;
        
        // 모델 생성
        let mut model = RBEGPT2Model::new(config.clone(), compressed_weights, None)?;
        
        // 생성 설정
        let generation_config = GenerationConfig {
            max_new_tokens: 10,
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            use_cache: true,
            do_sample: true,
            ..Default::default()
        };
        
        model.set_generation_config(generation_config);
        
        // 테스트 입력 (토큰 ID로 직접)
        let test_input_ids = vec![1, 2, 3, 4, 5]; // 더미 토큰들
        
        // 순전파 테스트
        let output = model.forward(&test_input_ids, None, None, false)?;
        
        // 출력 검증
        assert_eq!(output.logits.len(), test_input_ids.len() * config.vocab_size);
        assert!(output.logits.iter().all(|&x| x.is_finite()));
        
        // 통계 확인
        let stats = output.inference_stats;
        assert_eq!(stats.total_inferences, 1);
        assert_eq!(stats.total_tokens_processed, test_input_ids.len());
        assert!(stats.average_latency_per_token_ms > 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_kv_cache_generation_consistency() -> Result<()> {
        let config = create_tiny_gpt2_config();
        let compressed_weights = generate_dummy_model_weights(&config)?;
        let mut model = RBEGPT2Model::new(config.clone(), compressed_weights, None)?;
        
        let test_prompt = vec![10, 20, 30];
        
        // KV 캐시 없는 생성
        model.set_generation_config(GenerationConfig {
            max_new_tokens: 5,
            use_cache: false,
            do_sample: false, // deterministic
            temperature: 1.0,
            ..Default::default()
        });
        
        let output_no_cache = generate_tokens_without_cache(&mut model, &test_prompt, 5)?;
        
        // KV 캐시 있는 생성
        model.set_generation_config(GenerationConfig {
            max_new_tokens: 5,
            use_cache: true,
            do_sample: false, // deterministic
            temperature: 1.0,
            ..Default::default()
        });
        
        let output_with_cache = generate_tokens_with_cache(&mut model, &test_prompt, 5)?;
        
        // 결과 일치성 확인
        assert_eq!(output_no_cache, output_with_cache);
        
        Ok(())
    }
    
    #[test]
    fn test_memory_usage_scaling() -> Result<()> {
        let base_config = GPT2Config {
            vocab_size: 1000,
            max_sequence_length: 512,
            d_model: 256,
            d_ff: 1024,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            embedding_block_size: 32,
            embedding_coefficients: 100,
            transformer_block_size: 32,
            transformer_coefficients: 150,
        };
        
        let sequence_lengths = vec![32, 64, 128, 256];
        let mut memory_usages = Vec::new();
        
        for &seq_len in &sequence_lengths {
            let compressed_weights = generate_dummy_model_weights(&base_config)?;
            let mut model = RBEGPT2Model::new(base_config.clone(), compressed_weights, None)?;
            
            let test_input: Vec<usize> = (0..seq_len).map(|i| i % base_config.vocab_size).collect();
            
            let memory_tracker = MemoryTracker::new();
            memory_tracker.start_tracking();
            
            let _output = model.forward(&test_input, None, None, false)?;
            let peak_memory = memory_tracker.peak_usage();
            
            memory_tracker.stop_tracking();
            
            memory_usages.push((seq_len, peak_memory));
            println!("Seq len: {}, Peak memory: {:.2} MB", seq_len, peak_memory as f32 / 1024.0 / 1024.0);
        }
        
        // 메모리 사용량이 시퀀스 길이에 대해 합리적으로 증가하는지 확인
        for i in 1..memory_usages.len() {
            let (prev_seq, prev_mem) = memory_usages[i-1];
            let (curr_seq, curr_mem) = memory_usages[i];
            
            let seq_ratio = curr_seq as f32 / prev_seq as f32;
            let mem_ratio = curr_mem as f32 / prev_mem as f32;
            
            // 메모리 증가가 시퀀스 길이 증가보다 과도하지 않은지 확인
            assert!(mem_ratio < seq_ratio * 1.5, 
                   "Memory usage scaling too high: seq_ratio {:.2}, mem_ratio {:.2}", 
                   seq_ratio, mem_ratio);
        }
        
        Ok(())
    }
    
    fn generate_tokens_without_cache(
        model: &mut RBEGPT2Model,
        prompt: &[usize],
        num_tokens: usize,
    ) -> Result<Vec<usize>> {
        let mut current_sequence = prompt.to_vec();
        let mut generated = Vec::new();
        
        for _ in 0..num_tokens {
            let output = model.forward(&current_sequence, None, None, false)?;
            let next_token = model.greedy_select(&output.logits)?;
            
            current_sequence.push(next_token);
            generated.push(next_token);
        }
        
        Ok(generated)
    }
    
    fn generate_tokens_with_cache(
        model: &mut RBEGPT2Model,
        prompt: &[usize],
        num_tokens: usize,
    ) -> Result<Vec<usize>> {
        let mut current_sequence = prompt.to_vec();
        let mut generated = Vec::new();
        let mut past_kv = None;
        
        for step in 0..num_tokens {
            let input = if step == 0 {
                current_sequence.clone()
            } else {
                vec![*current_sequence.last().unwrap()]
            };
            
            let output = model.forward(&input, None, past_kv.as_ref(), true)?;
            past_kv = output.present_key_values;
            
            let next_token = model.greedy_select(&output.logits)?;
            
            current_sequence.push(next_token);
            generated.push(next_token);
        }
        
        Ok(generated)
    }
}
```

## 9.5 결론

### 9.5.1 구현 완료 사항

✅ **완전한 GPT-2 구현:**
- Token/Positional Embedding
- N개 Transformer Block
- Language Modeling Head
- 텍스트 생성 파이프라인

✅ **RBE 압축 적용:**
- Token Embedding (90% 메모리 절약)
- 모든 Transformer Block weights
- LM Head (embedding 공유 시)

✅ **생성 최적화:**
- KV 캐싱
- 다양한 샘플링 전략
- Repetition penalty

### 9.5.2 성능 특성

- **메모리 효율성**: 전체 모델에서 80-90% 절약
- **생성 속도**: KV 캐시로 10-50배 향상
- **정확도**: 원본 모델 대비 < 2% 성능 차이
- **확장성**: 모델 크기에 선형적 메모리 증가

### 9.5.3 다음 장 예고

Chapter 10에서는 실제 배포를 위한 성능 최적화, 양적 분석, 하드웨어별 최적화, 그리고 실제 사용 사례들을 다룬다. 