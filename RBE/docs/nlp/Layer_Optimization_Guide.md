# 🚀 RBE 레이어별 최적화 가이드

## 📊 **성능 테스트 결과 요약**

### **Core 모듈 벤치마크 결과**

| **모듈** | **달성 성능** | **목표 성능** | **상태** |
|----------|---------------|---------------|----------|
| WeightGenerator | 25ns/element | 10μs/op | ✅ 달성 |
| FusedForwardPass | 150ns/element | 200ns/element | ✅ 달성 |
| WeightMapper | 압축률 41:1 | 1000:1 | ⚠️ 개선 필요 |
| 극한 압축 (K=16) | 1000:1, RMSE 0.4 | 1000:1, RMSE 0.0001 | ⚠️ 정확도 개선 필요 |

### **최적 설정 (실전 검증)**
- **K레벨**: 5 (RMSE < 0.05, 속도 42ns)
- **임계값**: 0.01
- **압축률**: 8배 (현실적 목표)
- **블록 크기**: 128x128 (메모리/성능 균형)

---

## 1. **RBELinear 레이어**

### **현재 구현 분석**
```rust
// 현재: 미리 계산된 RBE 가중치 사용
OptimizedBlock {
    rbe_weights: Vec<f32>,  // 미리 계산됨
    residual_contributions: Vec<(usize, usize, f32)>,
}
```

### **개선된 구현 (WeightGenerator 직접 사용)**

#### **핵심 변경사항**
1. **미리 계산 제거** → WeightGenerator 직접 호출
2. **메모리 절약**: 64x64 블록 = 16KB → 100 bytes (160:1)
3. **속도 최적화**: 병렬 처리 + SIMD

#### **목표 성능**
- **압축률**: 100:1 이상 (현재 41:1)
- **속도**: 10μs/operation (현재 25ns/element ✅)
- **RMSE**: < 0.1 (현재 0.05 ✅)
- **메모리**: 블록당 100 bytes

#### **구현 전략**
```rust
pub struct OptimizedRBELinear {
    // HybridEncodedBlock 직접 저장 (미리 계산 X)
    compressed_blocks: Vec<HybridEncodedBlock>,
    weight_generator: WeightGenerator,
    block_layout: BlockLayout,
}

impl OptimizedRBELinear {
    pub fn forward_optimized(&self, input: &[f32]) -> Vec<f32> {
        // 블록별 병렬 처리
        self.compressed_blocks.par_iter()
            .enumerate()
            .map(|(idx, block)| {
                // WeightGenerator로 즉석 생성
                self.weight_generator.generate_weights_batch_from_rbe_params(
                    &block.rbe_params,
                    &positions,
                    block.rows,
                    block.cols
                )
            })
            .collect()
    }
}
```

#### **최적화 기법**
1. **배치 처리**: `generate_weights_batch_from_rbe_params` 사용
2. **병렬화**: Rayon으로 블록별 병렬 처리
3. **캐싱**: 자주 사용되는 블록만 선택적 캐싱
4. **SIMD**: 내부 벡터 연산 최적화

---

## 2. **RBELayerNorm 레이어**

### **구현 전략**
- **파라미터가 작음** (hidden_dim × 2)
- RBE 압축 효과 제한적 → **표준 구현 권장**

#### **선택적 압축**
```rust
pub struct RBELayerNorm {
    // 파라미터 크기에 따라 선택
    gamma: LayerNormParam,
    beta: LayerNormParam,
}

enum LayerNormParam {
    Uncompressed(Vec<f32>),      // < 1024 elements
    Compressed(HybridEncodedBlock), // >= 1024 elements
}
```

#### **목표 성능**
- **속도**: < 1μs/token (압축 없이)
- **정확도**: 수치적 안정성 유지

---

## 3. **RBEEmbedding 레이어**

### **구현 전략**
- **대용량** (vocab_size × hidden_dim)
- **행별 압축** 효과적

#### **최적화된 구현**
```rust
pub struct RBEEmbedding {
    // 각 토큰별로 압축된 임베딩
    compressed_embeddings: Vec<HybridEncodedBlock>,
    weight_mapper: WeightMapper,
}

impl RBEEmbedding {
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        token_ids.par_iter()
            .flat_map(|&token_id| {
                let block = &self.compressed_embeddings[token_id as usize];
                self.weight_mapper.decode_row(block)
            })
            .collect()
    }
}
```

#### **목표 성능**
- **압축률**: 200:1 (vocab embedding)
- **속도**: < 100ns/token
- **메모리**: 90% 절약

---

## 4. **RBEAttention 레이어**

### **구현 전략**
- **QKV 행렬**: RBELinear 3개 사용
- **병렬 처리** 필수

#### **최적화된 구현**
```rust
pub struct RBEAttention {
    q_proj: OptimizedRBELinear,
    k_proj: OptimizedRBELinear,
    v_proj: OptimizedRBELinear,
    o_proj: OptimizedRBELinear,
    fused_forward: FusedForwardPass,
}

impl RBEAttention {
    pub fn forward(&self, hidden_states: &[f32]) -> Vec<f32> {
        // QKV 병렬 계산
        let (q, k, v) = rayon::join(
            || self.q_proj.forward_optimized(hidden_states),
            || rayon::join(
                || self.k_proj.forward_optimized(hidden_states),
                || self.v_proj.forward_optimized(hidden_states)
            )
        );
        
        // Attention 계산 (표준 구현)
        let attention_output = self.compute_attention(&q, &k, &v);
        
        // Output projection
        self.o_proj.forward_optimized(&attention_output)
    }
}
```

#### **목표 성능**
- **압축률**: 100:1 (각 projection)
- **속도**: < 50μs/token (전체)
- **병렬 효율**: 90% 이상

---

## 5. **RBEFFN 레이어**

### **구현 전략**
- **가장 큰 레이어** (hidden_dim × 4 × hidden_dim)
- **압축 효과 극대화**

#### **2단계 압축**
```rust
pub struct RBEFFN {
    // 2단계로 분리
    up_proj: OptimizedRBELinear,    // hidden -> 4*hidden
    down_proj: OptimizedRBELinear,  // 4*hidden -> hidden
}

impl RBEFFN {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // 1단계: 확장
        let expanded = self.up_proj.forward_optimized(x);
        
        // 활성화 함수 (GELU)
        let activated = gelu(&expanded);
        
        // 2단계: 축소
        self.down_proj.forward_optimized(&activated)
    }
}
```

#### **목표 성능**
- **압축률**: 200:1 이상
- **속도**: < 100μs/layer
- **메모리**: 95% 절약

---

## 📈 **성능 벤치마크 계획**

### **각 레이어별 테스트**
```rust
#[test]
fn benchmark_layer_performance() {
    let configs = vec![
        ("RBELinear", 768, 768, 128),      // hidden x hidden
        ("RBEEmbedding", 32000, 768, 256), // vocab x hidden
        ("RBEFFN_up", 768, 3072, 256),     // hidden x 4*hidden
        ("RBEFFN_down", 3072, 768, 256),   // 4*hidden x hidden
    ];
    
    for (name, in_dim, out_dim, block_size) in configs {
        // 압축률 측정
        let compression_ratio = measure_compression(in_dim, out_dim, block_size);
        
        // 속도 측정
        let speed_us = measure_speed(in_dim, out_dim, block_size);
        
        // RMSE 측정
        let rmse = measure_accuracy(in_dim, out_dim, block_size);
        
        println!("{}: 압축률 {:.0}:1, 속도 {:.1}μs, RMSE {:.4}", 
                 name, compression_ratio, speed_us, rmse);
    }
}
```

### **목표 vs 현재 성능**

| **레이어** | **목표 압축률** | **목표 속도** | **목표 RMSE** | **현재 상태** |
|------------|-----------------|---------------|---------------|---------------|
| RBELinear | 100:1 | 10μs | 0.1 | 41:1, 25ns, 0.05 |
| RBEEmbedding | 200:1 | 100ns/token | 0.05 | 미구현 |
| RBEAttention | 100:1 | 50μs | 0.1 | 미구현 |
| RBEFFN | 200:1 | 100μs | 0.1 | 미구현 |

---

## 🔧 **구현 우선순위**

1. **RBELinear 개선** (핵심)
   - WeightGenerator 직접 사용
   - 블록 크기 최적화 (128x128)
   - 병렬 처리 강화

2. **RBEFFN 구현** (최대 효과)
   - 가장 큰 메모리 사용
   - 압축 효과 극대화

3. **RBEEmbedding 구현** (중요)
   - Vocab 크기에 따라 큰 효과
   - 행별 압축 최적화

4. **RBEAttention 구현** (복잡)
   - QKV 병렬 처리
   - FusedForwardPass 활용

5. **RBELayerNorm** (선택적)
   - 작은 파라미터는 압축 안함
   - 큰 모델에서만 선택적 압축

---

## 🚀 **다음 단계**

1. **RBELinear 리팩토링**
   - 현재 구현을 WeightGenerator 직접 사용으로 교체
   - 성능 테스트 및 검증

2. **블록 크기 실험**
   - 64x64, 128x128, 256x256 비교
   - 메모리/성능 균형점 찾기

3. **극한 압축 모드**
   - K=16 설정으로 1000:1 달성
   - RMSE 개선 방안 연구

4. **통합 테스트**
   - 미니 GPT-2 모델로 전체 성능 검증
   - 메모리 사용량 및 추론 속도 측정 