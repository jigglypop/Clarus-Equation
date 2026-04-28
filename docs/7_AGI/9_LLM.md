# CE-LLM 실전 구축 가이드

> 관련: 2-6장(이론), `examples/ai/clarus_lm.py`(처음부터 학습), `examples/ai/ce_gpt2.py`(기존 모델 이식), `examples/ai/train_clarus.py`(학습 스크립트)
>
> 이 장은 CE-AGI 원리를 적용한 LLM을 실제로 만드는 세 가지 경로를 다룬다. 이론이 아니라 코드와 명령어 중심.

---

## 1. 세 가지 구축 경로

| 경로 | 설명 | 난이도 | 소요 | 결과물 |
|---|---|---|---|---|
| **A. 처음부터 학습** | ClarusLM을 스크래치로 학습 | 낮음 | GPU 수시간 | 소형 CE-LLM |
| **B. 기존 모델 이식** | GPT-2/Llama 등에 CE 모듈 이식 | 중간 | GPU 수시간 | CE-강화 LLM |
| **C. 대규모 사전학습** | CE 아키텍처로 대규모 학습 | 높음 | 클러스터 수일 | 실용급 CE-LLM |

---

## 2. 경로 A: 처음부터 학습 (ClarusLM)

### 2.1 기존 코드 구조

`examples/ai/clarus_lm.py`가 CE-LLM의 완전한 모델 정의를 포함한다.

```
ClarusLM
  ├── tok_emb (Embedding)
  ├── pos_emb (Embedding)
  ├── blocks[] (ClarusBlock x N)
  │     ├── norm1 (LBONorm)          // LayerNorm + LBO 확산
  │     ├── attn (ClarusAttention)    // MHA + spectral norm
  │     ├── norm2 (LBONorm)
  │     └── ffn (GaugeLattice)        // 3x3+1 게이지 격자
  │           ├── su3 (SU(3) binding, 74.1%)
  │           ├── su2 (SU(2) decision, 21.1%)
  │           ├── u1 (U(1) attention, 4.9%)
  │           └── phi (LBONorm, smoothing)
  ├── norm (LBONorm)
  └── head (Linear, weight tied)
```

CE 수정 4가지가 모두 내장되어 있다:

1. **LBONorm**: `F.layer_norm` + 저랭크 LBO 확산 (`V^T V`)
2. **GaugeLattice**: 채널 비율 `alpha_s : alpha_w : alpha_em`으로 자동 분할
3. **Spectral Norm**: `nn.utils.spectral_norm(proj)` -- 유니타리 제약
4. **곡률 손실**: `loss = ce + lambda_curv * curv`

### 2.2 학습 실행

**데이터 준비:**

텍스트 파일 하나면 된다. 한국어, 영어, 코드, 수학 -- 무엇이든 가능.

```bash
# 예: 위키피디아 덤프, 논문 텍스트, 코드 파일 등
cat *.txt > train_data.txt
```

**학습 명령:**

```bash
cd examples/ai
python train_clarus.py \
    --data train_data.txt \
    --dim 256 \
    --n_layers 6 \
    --n_heads 8 \
    --seq_len 256 \
    --batch_size 32 \
    --lr 3e-4 \
    --steps 5000 \
    --lambda_curv 0.01 \
    --device cuda
```

**출력 예시:**

```
ClarusLM  4.23M params
  vocab=95  dim=256  layers=6  heads=8
  train=1234567  val=65000 chars
  device=cuda  lambda_curv=0.01

3x3+1 lattice:
  SU(3) binding:   189 dims (74.1%)
  SU(2) decision:   54 dims (21.1%)
  U(1)  attention:  13 dims (4.9%)
  Phi   smoothing: LBO (rank=32)

step     1 | loss 4.5432 | val 4.5123 | curv 0.012345 | ...
step   200 | loss 2.3456 | val 2.4567 | curv 0.003456 | ...
...
```

### 2.3 규모별 설정

| 규모 | dim | layers | heads | 파라미터 | GPU 메모리 | 학습 시간 |
|---|---|---|---|---|---|---|
| Micro | 128 | 4 | 4 | ~1M | < 1GB | 수분 |
| Small | 256 | 6 | 8 | ~4M | < 2GB | 수십분 |
| Medium | 512 | 12 | 8 | ~30M | ~4GB | 수시간 |
| Large | 768 | 12 | 12 | ~85M | ~8GB | 반일 |
| XL | 1024 | 24 | 16 | ~350M | ~24GB | 수일 |

### 2.4 캐릭터 레벨 → 서브워드 토크나이저

기존 `train_clarus.py`는 캐릭터 레벨 토크나이저를 사용한다. 실용급으로 올리려면 서브워드 토크나이저가 필요하다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # BPE, 50257 vocab

model = ClarusLM(
    vocab_size=tokenizer.vocab_size,  # 50257
    dim=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=1024,
    lambda_curv=0.01,
)
```

학습 루프는 `train_clarus.py`의 구조를 그대로 사용하되, `CharTokenizer`를 `AutoTokenizer`로 교체한다.

---

## 3. 경로 B: 기존 모델 이식 (CE-GPT2)

### 3.1 2단계 이식 전략

`examples/ai/ce_gpt2.py`가 GPT-2에 CE를 이식하는 완전한 코드다.

**Phase 1 -- 비파괴 이식 (성능 보존):**

- `LayerNorm` $\to$ `LBONorm` (h=0 초기화, scale/bias 복사 $\to$ 원본과 동일 출발)
- `c_proj` $\to$ `spectral_norm` (가중치 보존 + 유니타리 제약)

이 시점에서 모델 출력은 원본 GPT-2와 **완전히 동일**하다. CE 모듈이 추가되었지만 h=0이므로 LBO 확산이 꺼져 있다.

**Phase 2 -- MLP 압축 (선택적):**

- `MLP` $\to$ `GaugeLatticeV2` (cross-channel mixing 포함)
- 증류(distillation)로 초기화: 원본 MLP의 입출력을 모방하도록 학습

### 3.2 실행

```bash
cd examples/ai

# Phase 1만 (안전, 빠름)
python ce_gpt2.py --data train_data.txt --phase 1 --steps 200

# Phase 2 포함 (MLP 교체, 37% 파라미터 절감)
python ce_gpt2.py --data train_data.txt --phase 2 --steps 500
```

### 3.3 다른 모델에 이식

GPT-2 외의 모델(Llama, Mistral, Phi 등)에도 동일한 원리로 이식 가능하다. 핵심은 3가지:

**1) LayerNorm $\to$ LBONorm:**

```python
def transplant_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            dim = module.normalized_shape[0]
            lbo = LBONorm(dim)
            lbo.scale.data = module.weight.data.clone()
            lbo.bias.data = module.bias.data.clone()
            lbo.h.data.fill_(0.0)  # h=0: 원본과 동일 출발
            parent = get_parent(model, name)
            setattr(parent, name.split('.')[-1], lbo)
```

**2) Attention 출력 사영에 Spectral Norm:**

```python
def transplant_spectral(model):
    for block in model.layers:
        block.self_attn.o_proj = nn.utils.spectral_norm(block.self_attn.o_proj)
```

**3) (선택) MLP $\to$ GaugeLatticeV2:**

```python
def transplant_mlp(model, distill_steps=500):
    for block in model.layers:
        old_mlp = block.mlp
        dim = old_mlp.gate_proj.in_features
        new_lattice = GaugeLatticeV2(dim, mult=4, mix_rank=dim//8)
        distill(old_mlp, new_lattice, steps=distill_steps)
        block.mlp = new_lattice
```

### 3.4 미세조정

이식 후 CE 파라미터(LBO의 h, V, 곡률 정규화)를 미세조정한다.

```python
# CE 파라미터만 학습 (나머지 동결)
for name, param in model.named_parameters():
    param.requires_grad = False
    if any(k in name for k in ['LBONorm', 'lbo', 'phi', 'spectral']):
        param.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5, weight_decay=0.01
)
```

---

## 4. 경로 C: 대규모 CE 사전학습

### 4.1 아키텍처 설정

1B 규모 CE-LLM:

```python
model = ClarusLM(
    vocab_size=32000,       # SentencePiece
    dim=2048,
    n_layers=24,
    n_heads=16,
    max_seq_len=2048,
    ffn_mult=4,
    lambda_curv=0.005,
)
# ~1.3B params (표준 Transformer 대비 37% 적음 -> ~800M 유효)
```

### 4.2 수면 학습 순환 (3장 적용)

대규모 학습에서 수면 순환을 적용하는 방법:

```python
def sleep_train(model, dataloader, n_cycles=10, device='cuda'):
    for cycle in range(n_cycles):
        # === 각성(Wake): 표준 학습 ===
        model.train()
        accumulated_grads = {}
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            # 그래디언트 누적만, 업데이트 보류
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = torch.zeros_like(p.grad)
                    accumulated_grads[name] += p.grad.clone()
            model.zero_grad()

        # === NREM: 곡률 기반 선택적 업데이트 ===
        model.eval()
        EPSILON_SQ = 0.0487
        for name, p in model.named_parameters():
            if name in accumulated_grads:
                g = accumulated_grads[name]
                # 상위 4.87%만 통과
                threshold = torch.quantile(g.abs().flatten(), 1.0 - EPSILON_SQ)
                mask = (g.abs() >= threshold).float()
                p.data -= lr * g * mask

        # === REM: 비선택 그래디언트 재탐색 ===
        for name, p in model.named_parameters():
            if name in accumulated_grads:
                g = accumulated_grads[name]
                threshold = torch.quantile(g.abs().flatten(), 1.0 - EPSILON_SQ)
                pruned = g * (g.abs() < threshold).float()
                # 노이즈 주입 + 재평가
                noise = torch.randn_like(pruned) * pruned.std() * 0.1
                candidate = pruned + noise
                # 개선되면 채택 (간소화된 버전)
                p.data -= lr * 0.01 * candidate

        # 수면 압력 리셋
        print(f"Cycle {cycle+1}/{n_cycles} complete")
```

### 4.3 희소 추론 (5장 적용)

학습 후 추론 시 Top-k 활성화 적용:

```python
class SparseGaugeLattice(GaugeLattice):
    """추론 시 4.87% 활성화만 사용."""

    EPSILON_SQ = 0.0487

    def forward(self, x):
        y = super().forward(x)
        if not self.training:
            # 추론 시 Top-k 활성화
            k = max(1, int(self.EPSILON_SQ * y.shape[-1]))
            topk_vals, topk_idx = torch.topk(y.abs(), k, dim=-1)
            mask = torch.zeros_like(y)
            mask.scatter_(-1, topk_idx, 1.0)
            y = y * mask * (y.shape[-1] / k)  # 스케일 보정
        return y
```

### 4.4 환각 억제 추론 (6장 적용)

생성 시 곡률 모니터링 + 개입:

```python
@torch.no_grad()
def generate_with_curvature_check(model, idx, n_tokens,
                                   temperature=0.8, top_k=40,
                                   curv_threshold=0.1, max_retry=3):
    for _ in range(n_tokens):
        x = idx[:, -model.max_seq_len:]
        logits, _ = model(x)
        logits = logits[:, -1] / temperature

        # 곡률 측정
        avg_curv = sum(b.curvature for b in model.blocks) / len(model.blocks)

        retry = 0
        while avg_curv > curv_threshold and retry < max_retry:
            # 곡률 평탄화: 마지막 블록의 hidden state에 LBO 확산 추가 적용
            for block in model.blocks:
                block.norm1.h.data *= 1.5  # 일시적으로 확산 강도 증가
            logits, _ = model(x)
            logits = logits[:, -1] / temperature
            avg_curv = sum(b.curvature for b in model.blocks) / len(model.blocks)
            retry += 1
            for block in model.blocks:
                block.norm1.h.data /= 1.5  # 복원

        # Top-k 샘플링
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        next_token = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, next_token], 1)

    return idx
```

### 4.5 Grounded CE-LLM: 감각 발화 집합 추가

텍스트-only LLM보다 AGI 쪽으로 가려면, 앞단에 모달리티별 sparse encoder를 두는 편이 더 자연스럽다(`7_AGI/12_Equation.md` 6.8-6.9절).

```python
class GroundedCELLM(nn.Module):
    def __init__(self, text_model, vision_encoder, audio_encoder, touch_encoder):
        super().__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.touch_encoder = touch_encoder
        self.epsilon_sq = 0.0487

    def topk_act(self, h):
        k = max(1, int(self.epsilon_sq * h.shape[-1]))
        vals, idx = torch.topk(h.abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, idx, 1.0)
        return h * mask

    def forward(self, text_ids, image=None, audio=None, touch=None):
        h_t = self.topk_act(self.text_model.embed(text_ids))
        h_v = self.topk_act(self.vision_encoder(image)) if image is not None else None
        h_a = self.topk_act(self.audio_encoder(audio)) if audio is not None else None
        h_h = self.topk_act(self.touch_encoder(touch)) if touch is not None else None
        h_joint = bind_xi([x for x in [h_t, h_v, h_a, h_h] if x is not None])
        return self.text_model.decode_from_joint(h_joint)
```

이 경로의 핵심 예측은 두 가지다.

1. 모달별로 먼저 `4-5%` 발화 집합을 만든 뒤 결합하는 편이 early fusion보다 효율적이다.
2. 시각/청각/촉각 grounding이 추가되면, 텍스트-only 모델보다 멀티모달 환각이 줄어들어야 한다.

---

## 5. CE 모듈별 구현 상세

### 5.1 LBONorm 내부 동작

```python
def forward(self, x):
    # 1단계: 표준 LayerNorm (안정성 보장)
    x = F.layer_norm(x, (self.dim,))

    # 2단계: 저랭크 LBO 확산
    #   xW = x V^T V  (x를 V의 열공간으로 사영)
    #   Lx = x - xW   (사영 잔차 = 고곡률 성분)
    xW = F.linear(F.linear(x, self.V), self.V.T)
    Lx = x - xW

    # 3단계: 확산 적용
    #   h > 0이면 고곡률 성분 Lx가 감쇠됨
    #   h = 0이면 LayerNorm과 동일
    h = self.h.abs().clamp(max=0.5)
    self._curvature = (Lx * Lx).mean()  # 곡률 에너지 저장

    return (x - h * Lx) * self.scale + self.bias
```

핵심: `V`가 "평탄한 부분공간"을 학습한다. $x$를 이 부분공간으로 사영한 것이 $xW$이고, 사영 잔차 $Lx$가 "고곡률 성분"이다. $h > 0$이면 이 고곡률 성분이 감쇠된다.

### 5.2 GaugeLattice 채널 분할

```python
# d=768 예시
total = 0.11789 + 0.03352 + 0.00775  # = 0.15916
d3 = round(768 * 0.11789 / 0.15916)  # = 568 (SU(3) binding)
d2 = round(768 * 0.03352 / 0.15916)  # = 162 (SU(2) decision)
d1 = 768 - 568 - 162                  # =  38 (U(1) attention)

# 입력 x를 [x_3 | x_2 | x_1]으로 분할
# 각각 독립적인 MLP를 통과
# 결과를 concat
```

### 5.3 Spectral Norm 적용

```python
# 적용 전: sigma_1(W) 제약 없음 (정보 증폭 가능)
proj = nn.Linear(dim, dim, bias=False)

# 적용 후: sigma_1(W) <= 1 (유니타리 제약)
proj = nn.utils.spectral_norm(nn.Linear(dim, dim, bias=False))
```

PyTorch의 `spectral_norm`은 power iteration으로 최대 특이값을 추정하고, forward 시 자동으로 `W / sigma_1(W)`를 적용한다.

---

## 6. 학습 모니터링: 무엇을 봐야 하는가

### 6.1 핵심 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| `loss` | Cross-entropy 손실 | 단조 감소 |
| `curv` | 평균 곡률 에너지 $\|\Delta_g h\|^2$ | 학습 초반 증가 후 안정화 |
| `val_loss` | 검증 손실 | train과 괴리 없어야 함 |
| `lr` | 학습률 | warmup + cosine decay |
| `active_ratio` | 실제 활성 비율 | `4-5%` 중심, `3-7%` 실용 대역 |
| `bootstrap_resid` | $\|p_n - p^*\|$ 또는 proxy | 수면 루프에서 감소 |
| `hall_corr` | 곡률-오류 상관 | 양의 상관 기대 |
| `ground_align` | 모달 정합도 | grounded 모델에서 증가 기대 |

### 6.2 곡률 에너지의 해석

- **curv 단조 증가**: 모델이 복잡한 패턴을 학습 중 (정상)
- **curv 급등**: 불안정한 영역 진입 (lambda_curv 증가 고려)
- **curv 수렴**: 모델이 안정적 표현 공간을 찾음 (이상적)
- **curv가 0에 수렴**: LBO가 과도하게 평탄화 (lambda_curv 감소 고려)

### 6.3 3x3+1 격자 균형

`model.lattice_summary()`로 격자 구조를 확인:

```
SU(3) binding:   189 dims (74.1%)   # 결합: 지각 요소 통합
SU(2) decision:   54 dims (21.1%)   # 결정: 분기/선택
U(1)  attention:  13 dims (4.9%)    # 주의: 억제적 게이팅
Phi   smoothing: LBO (rank=32)      # 안정화: 전역 평탄화
```

이 비율은 CE 결합 상수에서 고정이므로 조정 불필요.

### 6.4 예측 점검 루프

실전에서는 아래 순서로만 해석해야 한다.

1. **예측 고정**
   - 활성 비율 중심 `4.87%`
   - 수면 루프 잔차 `1회 15.5%`, `2회 2.4%`, `3회 0.37%`
   - 곡률 제약은 hard bound가 아니라 안정화 편향
2. **A/B 측정**
   - Dense와 Sparse
   - Wake-only와 Sleep
   - Text-only와 Grounded
3. **게이트 판정**
   - 최적점이 `4-5%` 근방인가
   - sleep이 wake-only보다 drift를 줄이는가
   - grounded가 text-only보다 모달 불일치를 줄이는가
4. **실패 시 하향**
   - 맞지 않으면 CE 전체를 선언하지 말고, 해당 예측만 `bridge` 또는 `hypothesis`로 내린다

### 6.5 최소 체크리스트

| 항목 | 기대값 | 실패 시 해석 |
|---|---|---|
| `active_ratio` 스위프 | 최적점 `4-5%`, 실용 대역 `3-7%` | 과제 의존성이 더 큼 |
| sleep residual | `2-3`회 순환에서 급감 | 현재 구현의 동역학이 CE 최소 반복식과 다름 |
| curvature vs error | 양의 상관 | P5는 일반 안정화 regularizer에 가까움 |
| grounded vs text-only | grounding 오류 감소 | 결합 순서 또는 encoder 설계 재검토 |

---

## 7. GaugeLatticeV2: 채널 혼합 구현

`ce_gpt2.py`에 이미 구현된 V2 격자:

```python
class GaugeLatticeV2(nn.Module):
    def __init__(self, dim, mult=4, mix_rank=64):
        super().__init__()
        # ... (채널 분할은 V1과 동일)

        # 채널 간 저랭크 혼합 (섭동적)
        self.mix_down = nn.Linear(dim, mix_rank, bias=False)
        self.mix_up = nn.Linear(mix_rank, dim, bias=False)
        nn.init.zeros_(self.mix_up.weight)  # 0 초기화: 시작 시 혼합 없음

    def forward(self, x):
        # 블록 대각 전이
        y = concat(su3(x_3), su2(x_2), u1(x_1))
        # 섭동적 혼합 추가
        y = y + self.mix_up(self.mix_down(y))
        return self.phi(y)
```

`mix_up`을 0으로 초기화하므로, 시작 시 V1과 동일하다. 학습이 진행되면서 필요한 만큼 채널 간 혼합이 자동으로 학습된다.

---

## 8. 실전 팁

### 8.1 lambda_curv 선택

| 모델 규모 | 권장 lambda_curv | 이유 |
|---|---|---|
| Micro (~1M) | 0.01-0.05 | 작은 모델은 곡률 제약 강하게 |
| Small (~10M) | 0.005-0.01 | |
| Medium (~100M) | 0.001-0.005 | |
| Large (~1B+) | 0.0005-0.001 | 큰 모델은 자연 평탄화 경향 |

### 8.2 LBO rank 선택

$$r = \max(4,\; d / 8)$$

이 경험적 규칙이 대부분의 경우 작동한다. $r$이 너무 작으면 확산이 불충분하고, 너무 크면 파라미터 낭비.

### 8.3 Spectral Norm 주의사항

- 학습 초반에 spectral norm이 그래디언트를 불안정하게 만들 수 있다
- 해결: warmup 동안 spectral norm의 power iteration을 1회만 수행 (기본값)
- `nn.utils.spectral_norm(module, n_power_iterations=1)` (기본값이므로 변경 불필요)

### 8.4 메모리 최적화

CE 모듈의 추가 메모리 비용:
- LBONorm: $r \times d$ (V 행렬) -- 표준 LayerNorm 대비 미미
- GaugeLattice: 표준 FFN 대비 37% 감소
- Spectral Norm: 원래 크기와 동일 + $u, v$ 벡터 (미미)

총합: 표준 Transformer 대비 메모리 **감소**.
