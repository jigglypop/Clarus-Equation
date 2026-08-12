//! RBE 압축 가중치를 사용한 BERT 모델 추론
use anyhow::{bail, Result};
use crate::{
    decoder::WeightGenerator,
    HybridEncodedBlock,
};
use std::ops::{AddAssign, MulAssign};

// --- Helper Functions ---

fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let hidden_size = weight.len();
    for chunk in x.chunks_mut(hidden_size) {
        let n = chunk.len();
        if n == 0 { continue; }
        let mean = chunk.iter().sum::<f32>() / n as f32;
        let var = chunk.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev_inv = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            chunk[i] = (chunk[i] - mean) * std_dev_inv * weight[i] + bias[i];
        }
    }
}

fn gelu(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = 0.5 * *val * (1.0 + (*val * 0.7978845608 * (1.0 + 0.044715 * *val * *val)).tanh());
    }
}

fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    if sum > 0.0 {
        for val in x.iter_mut() {
            *val /= sum;
        }
    }
}

// --- Core Data Structures ---

#[derive(Clone, Debug)]
pub struct CompressedLayer {
    pub blocks: Vec<HybridEncodedBlock>,
    pub shape: (usize, usize), // (rows, cols)
}

impl CompressedLayer {
    /// 선형 변환 (y = Wx + b)
    pub fn linear_forward(&self, generator: &WeightGenerator, input: &[f32], bias: &[f32]) -> Result<Vec<f32>> {
        let (output_size, input_size) = self.shape;
        if input.len() != input_size {
            bail!("Input size mismatch for linear_forward. Expected {}, got {}", input_size, input.len());
        }
        let mut output = if bias.is_empty() {
            vec![0.0f32; output_size]
        } else {
            if bias.len() != output_size {
                bail!("Bias size mismatch for linear_forward. Expected {} or 0, got {}", output_size, bias.len());
            }
            bias.to_vec()
        };
        
        let blocks_per_row = (self.shape.1 + 63) / 64;
        
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let decoded_arc = generator.decode_block(block);
            let decoded_flat = &*decoded_arc;
            
            let block_row = (block_idx / blocks_per_row) * 64;
            let block_col = (block_idx % blocks_per_row) * 64;
            
            let block_rows = 64;
            let block_cols = 64;

            for i in 0..block_rows {
                let out_idx = block_row + i;
                if out_idx >= output_size { continue; }
                
                for j in 0..block_cols {
                    let in_idx = block_col + j;
                    if in_idx >= input_size { continue; }

                    let flat_idx = i * block_cols + j;
                    if flat_idx < decoded_flat.len() {
                         output[out_idx] += decoded_flat[flat_idx] * input[in_idx];
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn get_row(&self, generator: &WeightGenerator, row_idx: usize) -> Result<Vec<f32>> {
        let (_, cols) = self.shape;
        let mut row = vec![0.0f32; cols];
        
        let blocks_per_row = (cols + 63) / 64;
        let start_block_idx = (row_idx / 64) * blocks_per_row;
        let row_in_block = row_idx % 64;

        for col_block_idx in 0..blocks_per_row {
            let block_idx = start_block_idx + col_block_idx;
            if block_idx >= self.blocks.len() { continue; }

            let decoded_arc = generator.decode_block(&self.blocks[block_idx]);
            let decoded_flat = &*decoded_arc;

            let col_start = col_block_idx * 64;
            let col_end = (col_start + 64).min(cols);
            
            for (i, col) in (col_start..col_end).enumerate() {
                let flat_idx = row_in_block * 64 + i;
                if flat_idx < decoded_flat.len() {
                    row[col] = decoded_flat[flat_idx];
                }
            }
        }
        Ok(row)
    }
}


// --- BERT Component Structs ---

#[derive(Debug)]
pub struct BertAttention<'a> {
    pub q_w: &'a CompressedLayer, pub q_b: &'a [f32],
    pub k_w: &'a CompressedLayer, pub k_b: &'a [f32],
    pub v_w: &'a CompressedLayer, pub v_b: &'a [f32],
    pub output_w: &'a CompressedLayer, pub output_b: &'a [f32],
    pub layernorm_w: &'a [f32], pub layernorm_b: &'a [f32],
    pub n_heads: usize,
    pub hidden_size: usize,
}

#[derive(Debug)]
pub struct BertFeedForward<'a> {
    pub intermediate_w: &'a CompressedLayer, pub intermediate_b: &'a [f32],
    pub output_w: &'a CompressedLayer, pub output_b: &'a [f32],
    pub layernorm_w: &'a [f32], pub layernorm_b: &'a [f32],
}

#[derive(Debug)]
pub struct CompressedBertLayer<'a> {
    pub attention: BertAttention<'a>,
    pub ffn: BertFeedForward<'a>,
}

#[derive(Debug)]
pub struct CompressedBert<'a> {
    // Embeddings
    pub token_emb: &'a CompressedLayer,
    pub position_emb: &'a CompressedLayer,
    pub token_type_emb: &'a CompressedLayer,
    pub emb_layernorm_w: &'a [f32],
    pub emb_layernorm_b: &'a [f32],
    
    // Encoder Layers
    pub layers: Vec<CompressedBertLayer<'a>>,
    
    // Prediction Head
    pub lm_head_dense_w: Option<&'a CompressedLayer>,
    pub lm_head_dense_b: Option<&'a [f32]>,
    pub lm_head_layernorm_w: Option<&'a [f32]>,
    pub lm_head_layernorm_b: Option<&'a [f32]>,
    pub lm_head_decoder_w: &'a CompressedLayer,
    pub lm_head_decoder_b: Option<&'a [f32]>,

    // Config
    pub hidden_size: usize,
    pub n_heads: usize,
    
    // RBE Core
    pub generator: WeightGenerator,
}


// --- Forward Pass Implementations ---

impl<'a> BertAttention<'a> {
    fn forward(&self, hidden_states: &mut [f32], generator: &WeightGenerator) -> Result<()> {
        let seq_len = hidden_states.len() / self.hidden_size;
        let head_size = self.hidden_size / self.n_heads;
        let original_input = hidden_states.to_vec(); // For residual connection

        let mut all_q = Vec::with_capacity(seq_len * self.hidden_size);
        let mut all_k = Vec::with_capacity(seq_len * self.hidden_size);
        let mut all_v = Vec::with_capacity(seq_len * self.hidden_size);

        for i in 0..seq_len {
            let chunk = &hidden_states[i * self.hidden_size..(i + 1) * self.hidden_size];
            all_q.extend(self.q_w.linear_forward(generator, chunk, self.q_b)?);
            all_k.extend(self.k_w.linear_forward(generator, chunk, self.k_b)?);
            all_v.extend(self.v_w.linear_forward(generator, chunk, self.v_b)?);
        }

        let scale = 1.0 / (head_size as f32).sqrt();
        let mut attention_output = vec![0.0f32; seq_len * self.hidden_size];

        for i in 0..seq_len {
            let mut context_layer = vec![0.0f32; self.hidden_size];
            for h in 0..self.n_heads {
                let q_offset = i * self.hidden_size + h * head_size;
                let q_head = &all_q[q_offset..q_offset + head_size];
                
                let mut scores = vec![0.0f32; seq_len];
                for j in 0..seq_len {
                    let k_offset = j * self.hidden_size + h * head_size;
                    let k_head = &all_k[k_offset..k_offset + head_size];
                    scores[j] = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                softmax(&mut scores);

                for j in 0..seq_len {
                    let v_offset = j * self.hidden_size + h * head_size;
                    let v_head = &all_v[v_offset..v_offset + head_size];
                    for d in 0..head_size {
                        context_layer[h * head_size + d] += scores[j] * v_head[d];
                    }
                }
            }
            let output_chunk = &mut attention_output[i * self.hidden_size..(i + 1) * self.hidden_size];
            output_chunk.copy_from_slice(&self.output_w.linear_forward(generator, &context_layer, self.output_b)?);
        }

        for i in 0..hidden_states.len() {
            hidden_states[i] = original_input[i] + attention_output[i];
        }
        layer_norm(hidden_states, self.layernorm_w, self.layernorm_b, 1e-12);
        
        Ok(())
    }
}

impl<'a> BertFeedForward<'a> {
    fn forward(&self, hidden_states: &mut [f32], generator: &WeightGenerator) -> Result<()> {
        let input_size = self.intermediate_w.shape.1;
        let seq_len = hidden_states.len() / input_size;
        let original_input = hidden_states.to_vec(); // For residual connection
        let mut final_output = vec![0.0; hidden_states.len()];

        for i in 0..seq_len {
            let chunk = &hidden_states[i*input_size..(i+1)*input_size];
            
            // Intermediate layer
            let mut intermediate = self.intermediate_w.linear_forward(generator, chunk, self.intermediate_b)?;
            gelu(&mut intermediate);

            // Output layer
            let output_chunk = &mut final_output[i*input_size..(i+1)*input_size];
            let processed_output = self.output_w.linear_forward(generator, &intermediate, self.output_b)?;
            output_chunk.copy_from_slice(&processed_output);
        }
        
        // Residual and LayerNorm
        for i in 0..hidden_states.len() {
            hidden_states[i] = original_input[i] + final_output[i];
        }
        layer_norm(hidden_states, self.layernorm_w, self.layernorm_b, 1e-12);
        
        Ok(())
    }
}

impl<'a> CompressedBert<'a> {
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();
        let mut hidden_states = vec![0.0f32; seq_len * self.hidden_size];

        // 1. Embeddings
        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_emb = self.token_emb.get_row(&self.generator, token_id as usize)?;
            let pos_emb = self.position_emb.get_row(&self.generator, i)?;
            let type_emb = self.token_type_emb.get_row(&self.generator, 0)?; // Assuming token_type_id 0

            for j in 0..self.hidden_size {
                hidden_states[i * self.hidden_size + j] = token_emb[j] + pos_emb[j] + type_emb[j];
            }
        }
        layer_norm(&mut hidden_states, self.emb_layernorm_w, self.emb_layernorm_b, 1e-12);

        // 2. BERT Encoder Layers
        for layer in &self.layers {
            layer.attention.forward(&mut hidden_states, &self.generator)?;
            layer.ffn.forward(&mut hidden_states, &self.generator)?;
        }

        // 3. Prediction Head
        let last_token_hidden = &hidden_states[(seq_len - 1) * self.hidden_size..];
        let mut lm_head_hidden = last_token_hidden.to_vec();

        // Apply transform if it exists
        if let (Some(dense_w), Some(dense_b), Some(ln_w), Some(ln_b)) =
            (self.lm_head_dense_w, self.lm_head_dense_b, self.lm_head_layernorm_w, self.lm_head_layernorm_b) {
            lm_head_hidden = dense_w.linear_forward(&self.generator, &lm_head_hidden, dense_b)?;
            gelu(&mut lm_head_hidden);
            layer_norm(&mut lm_head_hidden, ln_w, ln_b, 1e-12);
        }

        let decoder_bias = self.lm_head_decoder_b.unwrap_or(&[]);
        let logits = self.lm_head_decoder_w.linear_forward(&self.generator, &lm_head_hidden, decoder_bias)?;

        Ok(logits)
    }
} 