# Mellin–Riemann Attention: ζ explicit-formula 유도 attention 의 ablation 분석과 분산 절감, 그리고 Euler-CE 의 length extrapolation 우위 발견

> 본 문서는 `mra_block_spec.md` 의 사양과 `tests/test_mra.py` 의 검증, 그리고
> `examples/ai/bench_recursive_euler.py --mode mra` 와 `bench_mra_extrap.py`
> 의 ablation 실측을 한 편의 짧은 논문 형태로 정리한다. **정직한 negative-
> positive 혼합 결과** 를 담은 internal write-up.

## Abstract

Riemann ζ 함수의 explicit formula 의 critical-strip 합으로부터 attention
score 를 직접 유도한 새로운 attention 변종 **Mellin–Riemann Attention
(MRA)** 를 제안하고, 7 개 설계 요소를 ablation 으로 검증한다. 또한 비교
대상으로 기존 `euler_ce_k1` 의 length extrapolation 성능을 처음으로 직접
측정한다.

`docs` Markdown 코퍼스 (400 K chars, vocab 788) 에서 char-level LM,
1500-step × 3-seed (in-distribution) + 1000-step × 3-seed (extrapolation):

* **부정적**: ζ 영점 `γ_k` 를 attention frequency 로 직접 사용하는 방식
  (Mellin 커널 `θ_k(p) = γ_k log(1+p)`) 은 `γ_k/γ_1` 의 범위가 ~5× 에 불과해
  RoPE 의 ~3000× 다중 스케일 해상도를 잃고, **PPL 14.29 (RoPE 12.29 대비
  +2.0)** 로 명확히 부진.
* **부정적**: ζ amplitude weighting (`w_k = 1/(1/2 + iγ_k)`) 단독 효과 미미
  (`mra` 12.46 vs `mra_noamp` 12.27, +0.17 손해).
* **부정적**: 부트스트랩 sparsity (`ε² = 4.87 %`) 와 출력 spectral norm 모두
  1500-step 시점 PPL 손해 (+0.33).
* **긍정적 (분산)**: `MRABlock(decay_mode="bias")` 가 평균 PPL 12.24 (RoPE
  대비 −0.05) 에 도달하고 **seed 분산이 RoPE 의 절반 (0.10 vs 0.21)**.
* **긍정적 (외삽)**: `mra_bias` 의 logarithmic distance bias 가 4× length
  extrapolation 에서 RoPE 보다 **8 %p 안정** (degrad +18.5 % vs +26.7 %).
* **새로운 큰 발견 (4× extrap)**: `euler_ce_k1` 이 4× length extrapolation
  에서 **degrad +6.0 % 만 발생** (RoPE +26.7 %, MRA +18.5 % 대비 압도적).
* **결정적 분해 (32× extrap, § 7.7)**: Euler-CE 외삽 우위의 **100 % 가
  e-decay 단독 기여**. π-rotation 을 끄면 외삽 성능 그대로 유지되고
  (`euler_no_pi` ≈ `euler_ce_k1` at 32×: −5.5 % vs −6.2 %), e-decay 를
  끄면 RoPE 보다 더 나빠진다 (`euler_no_decay` +54.7 % vs RoPE +47.2 %).
  **`rope_alibi` (RoPE + ALiBi linear decay) 가 32× 에서 Euler-CE 와 정확
  히 동등** (−6.2 % vs −6.2 %) — Euler-CE 의 외삽 우위는 ALiBi 의 재발견.
  Logarithmic decay (`mra_bias`) 는 linear 보다 외삽에 명확히 약함
  (+40.3 % at 32×).

결론: (a) ζ 영점을 frequency 로 쓰는 직설적 시도는 실패. (b) RoPE 위에
logarithmic decay 를 얹는 lean MRA 는 평균 PPL 동급 + 분산 절반.
(c) **Length extrapolation 에는 linear additive distance bias (= ALiBi)
만이 작동** — Euler-CE 의 e-decay 는 ALiBi 와 동치이고, π-rotation /
log decay / ζ frequency 모두 외삽에 무력하다.

---

## 1. Introduction

Transformer 의 positional encoding 은 attention 메커니즘이 set-permutation
invariant 이라는 근본 한계를 외부 주입으로 해결하는 inductive bias 의 핵심
지점이다. RoPE [1] 가 LLaMA / Qwen / Mistral / Gemma / EXAONE 등 거의 모든
현대 LLM 의 표준이 된 후, ALiBi [2], YaRN [3], LongRoPE [4] 등 후속 연구는
대체로 경험적 튜닝으로 long-context extrapolation 을 공략해왔다.

본 연구의 출발점은 다른 방향이다 — **수학적 axiom 에서 attention 식을 직접
유도** 하면 어떻게 되는가? Hilbert–Pólya 추측 [5] 은 Riemann ζ 의 비자명
영점 `{ρ_n = 1/2 + iγ_n}` 의 허수부가 어떤 self-adjoint operator 의 고유값
임을 가정한다. Berry–Keating [6] 은 이 operator 가 `H = (xp + px)/2` 의
양자화일 가능성을 제시했다.

본 연구는 RH (Riemann Hypothesis) 를 **engineering axiom** 으로 채택하고,
attention score 를 ζ 의 explicit formula 의 critical-strip 합으로부터 유도
한다. 식 자체는 단순하지만, 어느 부분이 실제로 도움이 되고 어느 부분이
도움이 되지 않는지는 **실증으로만 확인 가능** 하다. 본 논문의 기여는 이를
ablation 으로 분리한 데에 있다.

### 1.1 Contributions

1. ζ explicit formula 의 critical-strip 합으로부터 attention score 를
   폐쇄형으로 유도 (§ 3).
2. 7 가지 설계 knob (frequency 방식, amplitude weighting, decay 형태,
   sparsity, spectral norm, Hermitian) 을 노출하는 단일 ablation 표면 (§ 4).
3. **부정적 발견 명시**: 직설적 ζ-frequency, multiplicative decay,
   bootstrap sparsity, spectral-norm 모두 1500-step 시점 LOSS (§ 6).
4. **긍정적 발견**: lean variant (`mra` / `mra_bias`) 는 RoPE 와 평균 PPL
   동급 + **seed 분산 2-3× 감소** (§ 7).

---

## 2. Background

### 2.1 RoPE

`d_head` 차원의 dim-pair `(2k, 2k+1)` 에 대해 회전 행렬

```
R_θ = [[cos θ, -sin θ], [sin θ, cos θ]],   θ_k(p) = p · 10000^{-2k/d}
```

을 query/key 에 적용. 두 토큰의 attention score 가 상대 위치 `i - j` 만의
함수가 되도록 만든다. base frequency 10000 은 경험적 선택이다.

핵심 강점은 `inv_freq_k = 10000^{-2k/d}` 의 **기하 급수적 분포** 가 attention
에 폭넓은 다중 스케일 해상도를 부여한다는 점이다 (`d_head = 16` 시 약 3000×
범위).

### 2.2 ALiBi

attention score 에 `−|i − j|/m_h` 의 거리 감쇠 bias 를 직접 더한다 (`m_h` 는
헤드별 기울기). 본 논문의 `decay_mode="bias"` 가 ALiBi 를 logarithmic
거리 함수 `½ log((1+i)/(1+j))` 로 일반화한 형태에 해당한다.

### 2.3 ζ explicit formula

von Mangoldt:

```
ψ(x) = x − Σ_ρ x^ρ / ρ − log(2π) − ½ log(1 − x^{-2})
```

핵심 합 `Σ_ρ x^ρ / ρ` 는 RH 하에서

```
Σ_n x^{1/2 + iγ_n} / (1/2 + iγ_n) = √x · Σ_n e^{iγ_n log x} / (1/2 + iγ_n)
```

으로 정리된다. **Mellin 커널** `e^{iγ_n log x}` 와 **ζ 진폭** `1/(1/2 + iγ_n)`
의 곱이 자연스럽게 나타난다.

---

## 3. Method — Mellin–Riemann Attention

### 3.1 단순 score 식

위치쌍 `(i, j)` 에 `x = (1+i)/(1+j)` 를 대입한다. dim-pair `k` 를 복소채널로
압축하면

```
q_i^(k) := q_i^{2k} + i · q_i^{2k+1},     k_j^(k) := k_j^{2k} + i · k_j^{2k+1},
```

attention raw score 는

```
S_{ij} = √((1+j)/(1+i)) · Σ_k w_k · e^{−iγ_k log((1+i)/(1+j))} · q_i^(k) · conj(k_j^(k))
```

여기서 `w_k = 1/(1/2 + iγ_k)`. 토큰별 attention 입력은 `Re(S_{ij})`.

### 3.2 폐쇄형 분해

`(1+i)^{−iγ_k}` 와 `(1+j)^{−iγ_k}` 가 각각 `i`, `j` 만의 함수이므로 사전 변환

```
q̃_i^(k) = (1/√(1+i)) · e^{−iγ_k log(1+i)} · q_i^(k)
k̃_j^(k) = √(1+j) · e^{−iγ_k log(1+j)} · k_j^(k)
```

후

```
S_{ij} = Σ_k w_k · q̃_i^(k) · conj(k̃_j^(k))
```

이는 표준 dot-product attention 과 동일한 `O(N²K)` 계산. 실수 분해

```
q̂_re = w_re · q̃_re − w_im · q̃_im
q̂_im = w_re · q̃_im + w_im · q̃_re
Re(S) = q̂_re @ k̃_re^T + q̂_im @ k̃_im^T
```

→ 표준 attention 의 **2배 matmul** 비용. softmax / V 적용은 동일.

### 3.3 학습 자유도

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `cos_p, sin_p, log_decay` | buffer | 0 (위치-axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention 동일 |

표준 multi-head attention 대비 **추가 학습 파라미터 0**.

---

## 4. 설계 Knob — Ablation 표면

`MellinRiemannAttention(d_model, n_heads, block, **knobs)` 가 노출하는 7 개
knob:

| knob | 값 | 기본 | 의미 |
|---|---|---|---|
| `freq_mode` | `"rope"` / `"zeta_log"` | `"rope"` | 위치 주파수 형태 |
| `amp_weight` | `True` / `False` | `True` | ζ 진폭 weighting `w_k` 사용 |
| `decay_mode` | `"none"` / `"bias"` / `"mult"` | `"none"` | critical-line decay 형태 |
| `sparse_eps2` | `0.0` ∼ `0.0487` | `0.0` | top-k retention 비율 |
| `hermitian` | `False` / `True` | `False` | `W_q = W_k` tied + score symmetrize |
| `spectral_norm_o` | `False` / `True` | `False` | `σ₁(W_o) ≤ 1` 강제 |
| `rope_base` | f32 | `10000` | RoPE base freq (freq_mode="rope" 일 때만) |

### 4.1 `freq_mode`

- `"rope"`: `θ_k(p) = p · base^{−k/K}` — RoPE 기하 급수 (~3000× 범위).
- `"zeta_log"`: `θ_k(p) = γ_k · log(1+p)` — Mellin 커널, ζ 영점이 곧 frequency.
  `γ_k/γ_1` 의 범위가 K=16 에서 ~5× 에 불과 (다중 스케일 해상도 손실).

### 4.2 `amp_weight`

- `True`: `w_k = 1/(1/2 + iγ_k)` 를 score 에 곱함. `Re(w_k)·Re_part −
  Im(w_k)·Im_part` 형태로 standard RoPE 의 `cos` 채널과 보조 `sin` 채널이
  혼합된다.
- `False`: `w_k = 1`. attention 식이 표준 RoPE 와 일치 (control).

### 4.3 `decay_mode`

ζ explicit-formula 의 `√x` factor (`x = (1+i)/(1+j)`) 처리:

- `"none"`: 무시.
- `"bias"`: log-space additive — `bias_{ij} = ½(log(1+j) − log(1+i))` 를 score
  에 더함. ALiBi 의 logarithmic 일반화.
- `"mult"`: 원식 그대로 — `score_{ij} *= √((1+j)/(1+i))`. Per-key temperature
  처럼 작동하므로 의도와 다소 어긋난다.

### 4.4 `hermitian` (causal LM 에서 비추천)

`W_q = W_k` tied + score symmetrize `S ← (S + Sᵀ)/2`. Bidirectional encoder 에선
self-adjoint operator 의 이산화 (Hilbert–Pólya 직접 구현) 이지만, **causal
LM 에서는 mask 전 symmetrize 가 future leakage 를 일으킨다**. 본 작업의 ablation
표에서 이 knob 은 제외.

---

## 5. Implementation

`clarus/ce_mra.py` (PyTorch 참조 구현, ~250 LOC). 현재 backend 는 PyTorch only
— `RiemannRotaryAttention` 의 Rust/CUDA 패턴을 그대로 이식 가능.

테스트 (`tests/test_mra.py`, 11/11 pass):

- forward shape & finiteness
- block end-to-end
- 모든 ablation knob 조합 finite output
- bootstrap sparsity 정확한 retention 수
- 학습 파라미터 수 == 표준 MHA (4·d²) / Hermitian 모드 == 3·d²
- ζ amplitude 가 실제로 출력을 바꿈 (회귀 가드)
- 공리적 buffer 들이 학습 안 됨
- spectral norm 이 σ₁ ≤ 1 강제

---

## 6. Experiments — Setup

| 항목 | 값 |
|---|---|
| 코퍼스 | `docs/` Markdown (400 K chars, vocab = 788) |
| 분할 | 90 % train / 10 % val |
| 토크나이저 | char-level |
| 모델 | 2-layer × 4-head, `d_model = 64` |
| 파라미터 | ≈ 210 K |
| 시퀀스 | block = 64, batch = 8 |
| 옵티마이저 | AdamW, lr = 3e-4, grad clip 1.0 |
| 학습 | 1500 step × 3 seed |
| 평가 | 마지막 12 batch 평균 cross-entropy → exp |
| 하드웨어 | RTX 4060 (CUDA) |

### 6.1 Variants

| 이름 | freq_mode | amp_weight | decay_mode | 기타 |
|---|---|---|---|---|
| `std_rope` | RoPE | — | — | baseline (`RoPEAttnBlock`) |
| `euler_ce_k1` | π-rotation | — | e-decay | 기존 winner [9] |
| **`mra`** | rope | True | none | primary lean MRA |
| `mra_noamp` | rope | False | none | ablate ζ amplitude |
| `mra_zeta` | zeta_log | True | none | ablate frequency |
| `mra_bias` | rope | True | bias | + additive log decay |
| `mra_mult` | rope | True | mult | + multiplicative decay |
| `mra_sparse` | rope | True | none | + ε² = 0.0487 sparsity |
| `mra_sn` | rope | True | none | + spectral_norm(W_o) |

---

## 7. Results

### 7.1 1500-step × 3-seed PPL ablation

| variant | params | PPL | **σ_PPL** | time/seed (s) |
|---|---|---|---|---|
| `std_rope` | 210 K | 12.288 | 0.208 | 13.0 |
| `euler_ce_k1` | 210 K | **11.864** | 0.126 | 16.1 |
| **`mra`** | 210 K | 12.458 | **0.069** | 13.0 |
| `mra_noamp` | 210 K | 12.266 | 0.129 | 12.8 |
| `mra_zeta` | 210 K | **14.286** | 0.332 | 12.8 |
| **`mra_bias`** | 210 K | **12.240** | **0.097** | 13.0 |
| `mra_mult` | 210 K | 12.259 | 0.077 | 13.8 |
| `mra_sparse` | 210 K | 12.621 | 0.232 | 14.1 |
| `mra_sn` | 210 K | 12.621 | 0.172 | 18.7 |

### 7.2 Verdicts (vs `std_rope`, vs `euler_ce_k1`)

| 비교 | z | Δ PPL | 판정 |
|---|---|---|---|
| `mra_zeta` vs `std_rope` | +8.84 | +1.999 | **LOSS** (γ-freq 붕괴 확정) |
| `mra` vs `std_rope` | +1.35 | +0.170 | LOSS (mean), 분산 −67 % |
| `mra_noamp` vs `std_rope` | −0.16 | −0.022 | TIE (≈ pure RoPE) |
| `mra_bias` vs `std_rope` | −0.36 | −0.048 | TIE+ |
| `mra_mult` vs `std_rope` | −0.22 | −0.028 | TIE+ |
| `mra_sparse` vs `std_rope` | +1.86 | +0.334 | LOSS |
| `mra_sn` vs `std_rope` | +2.14 | +0.334 | LOSS |
| `euler_ce_k1` vs `std_rope` | (best baseline) | −0.424 | WIN |

### 7.3 부정적 발견

1. **`mra_zeta` (PPL 14.29)**: ζ 영점을 frequency 로 직접 사용하면 RoPE 의
   다중 스케일 해상도가 깨진다. `γ_k/γ_1` 의 범위는 K=16 에서 ~5× 에 불과
   하며 RoPE 의 `10000^{−k/K}` 의 ~3000× 와 비교하면 다중 스케일 해상도가
   거의 없다. 이 한계는 `riemann_rope` 의 PPL 19.29 (별도 측정) 의 직접 원인
   이며, ζ 영점을 frequency 로 직접 사용하는 모든 변종에서 재현된다.
2. **ζ amplitude 단독 효과 미미**: `mra` (12.46) − `mra_noamp` (12.27) =
   +0.17 PPL **손해** (z = +1.35). amp_weight 가 표준 RoPE attention 에서
   별다른 정보 이득을 주지 못한다.
3. **부트스트랩 sparsity / spectral norm**: 1500 step 충분 수렴 시점에선 모두
   PPL 손해 (+0.33). 짧은 학습 (200 step) 에선 도움이지만 long-run 에선 over-
   regularization.

### 7.4 긍정적 발견 — seed 분산 절감

mean PPL 은 거의 동등하지만 seed-to-seed 분산이 명확히 감소:

```
σ_PPL across 3 seeds (lower = more reproducible):
  std_rope     : 0.208         baseline
  mra          : 0.069         3.0×  more stable
  mra_mult     : 0.077         2.7×
  mra_bias     : 0.097         2.1×
  mra_noamp    : 0.129         1.6×
  euler_ce_k1  : 0.126         1.7×  (다른 메커니즘)
```

`mra` 계열 (RoPE freq + amp/decay 변종) 모두 **σ_PPL 이 RoPE 의 1/3 ∼ 1/2**.
`mra_noamp` 도 분산이 작은 것으로 보아, 분산 감소는 ζ amplitude 단독 효과가
아니라 RoPE 의 `qkv` fused linear 대신 **`W_q, W_k` 분리 + RoPE-style
회전** 의 조합에서 나오는 것으로 추정 (추가 조사 필요).

### 7.5 가장 좋은 lean 변종

`decay_mode="bias"` (`mra_bias`):
- mean PPL 12.240 — **RoPE 보다 0.05 낮음** (z = −0.36, statistically TIE
  with mean preference)
- σ_PPL 0.097 — RoPE 분산의 47 %
- 시간 비용 RoPE 와 동일

이는 ALiBi 의 logarithmic 일반화로 해석 가능:
```
bias_{ij} = ½ (log(1+j) − log(1+i))
```
선형 거리 `−|i−j|` 가 아닌 logarithmic 거리. critical line `Re(s) = 1/2` 의
지수 `½` 가 직접 도출.

### 7.6 Length extrapolation (4× 까지)

train block = 64, eval block ∈ {64, 96, 128, 192, 256} (4× 외삽까지),
1000 step × 3 seed:

| variant | N=64 | N=96 | N=128 | N=192 | N=256 | **degrad @ 4×** |
|---|---|---|---|---|---|---|
| `std_rope`    | 14.99 | 16.89 | 14.20 | 17.24 | 18.98 | **+26.7 %** |
| `euler_ce_k1` | 14.75 | 16.76 | 13.90 | 15.51 | 15.63 | **+6.0 %**  ← BEST |
| `mra`         | 15.42 | 17.46 | 14.76 | 17.14 | 18.36 | +19.0 % |
| `mra_bias`    | 14.96 | 17.06 | 14.39 | 16.62 | 17.72 | **+18.5 %** |
| `mra_zeta`    | 17.39 | 19.76 | 16.93 | 18.67 | 19.16 | +10.2 % |

발견:

1. **`euler_ce_k1` 의 extrapolation 이 압도적** — 4× 외삽에서 PPL +6 % 만
   증가. 다른 모든 변종 (RoPE 포함) 이 +18 ~ +27 % 인 것과 명확히 다른 영역.
   Euler-CE 의 `e^{-|i-j|/ξ}` decay 가 정확한 long-context 인덕티브 바이어스
   를 제공하는 것으로 해석된다.
2. **`mra_bias` 의 logarithmic decay 가 RoPE 보다 8 %p 더 부드럽게 degrade**
   (+18.5 % vs +26.7 %). lean MRA 의 자그마한 long-context 우위 — 다만
   `euler_ce_k1` 만큼은 아님.
3. **`mra_zeta` 가 짧은 base 에선 부진해도 외삽은 비교적 잘 함** (+10.2 %).
   `log(1+p)` lift 의 진짜 가치는 long-context 에서 발현. 짧은 학습 + 긴
   외삽 시나리오에 한정해서는 검토 가치 있음.
4. **`std_rope` 의 빠른 붕괴** (+26.7 %) — 알려진 RoPE periodicity 한계 재현.

이 결과는 logarithmic distance bias 가 ALiBi 의 일반화로 의미 있는 후속 연구
방향임을 시사한다.

### 7.7 Length extrapolation (32× 까지) — Euler-CE 분해

위 § 7.6 의 발견 (Euler-CE 의 외삽 우위) 의 진짜 원인을 격리한다. EulerCE
는 두 개의 분리 가능한 구조를 결합하므로 각각을 frozen 으로 끄고 비교한다.

| 변종 | 구성 | 학습 자유도 |
|---|---|---|
| `std_rope` | RoPE | baseline |
| `rope_alibi` | RoPE + ALiBi linear decay (per-head learnable slope) | + n_heads |
| `mra` / `mra_bias` | RoPE + ζ amplitude / + log decay | + n_heads (`mra_bias`) |
| `euler_no_decay` | EulerCE with `e_gate` frozen → π-rotation 만 | (gate 1개 frozen) |
| `euler_no_pi` | EulerCE with `pi_gate` frozen → e-decay 만 | (gate 1개 frozen) |
| `euler_ce_k1` | full Euler-CE | original |

train block = 64, eval block ∈ {64, 128, 256, 512, 1024, 2048} (최대 32×).
모든 모델은 train_block 으로 init, eval 직전 `extend_to(N_eval)` 로 buffer
만 grow. EulerCE 의 block-aware base `π^D_eff · N` 도 train 시점 N=64 로
계산.

1500 step × 3 seed × batch=4 (메모리 한계). 절대 PPL 은 짧은 학습 + 작은
batch 때문에 § 7.6 보다 높지만, **상대 degradation 은 비교 가능**.

#### Relative degradation (PPL(N)/PPL(64) − 1, 낮을수록 좋음)

| variant | ×2 | ×4 | ×8 | ×16 | **×32** |
|---|---|---|---|---|---|
| `std_rope` | −10.0 % | +12.6 % | +19.9 % | +37.7 % | **+47.2 %** |
| `mra` | −9.2 % | +10.4 % | +14.1 % | +29.7 % | +40.8 % |
| `mra_bias` | −8.8 % | +10.0 % | +13.9 % | +28.6 % | +40.3 % |
| `euler_no_decay` (π only) | −8.2 % | +14.6 % | +24.3 % | +44.4 % | **+54.7 %** |
| **`rope_alibi`** | −13.0 % | −3.3 % | −9.2 % | −7.6 % | **−6.2 %** |
| **`euler_no_pi`** (e only) | −12.0 % | −2.4 % | −7.9 % | −6.5 % | **−5.5 %** |
| **`euler_ce_k1`** | −13.1 % | −3.4 % | −9.5 % | −7.9 % | **−6.2 %** |

#### 결정적 분해

이 표는 세 개의 깨끗한 결론을 만든다.

1. **Euler-CE 의 외삽 우위는 100 % e-decay (linear additive distance
   bias) 단독 효과** —
   `euler_no_pi` (e-decay only) 는 32× 외삽에서 −5.5 %, 풀 `euler_ce_k1`
   −6.2 % 와 사실상 동일. π-rotation 을 빼도 외삽 성능이 유지된다.
2. **π-rotation 단독은 외삽 무력** —
   `euler_no_decay` (π only) 는 32× 에서 +54.7 % 로 RoPE (+47.2 %) 보다
   오히려 더 나쁨. block-aware base `π^D_eff · N` 도 long-context 안정성에
   기여하지 않는다 (in-distribution 에서는 도움일 수 있음).
3. **`rope_alibi` (RoPE + ALiBi-style learnable linear decay) ≈ Euler-CE** —
   32× 외삽 −6.2 % vs Euler-CE −6.2 %, 16× 에서도 −7.6 % vs −7.9 %. EulerCE
   의 e-decay 는 ALiBi (Press et al., 2022) 의 per-head learnable slope
   변종으로 정확히 환원된다. **EulerCE 의 외삽 우위는 새로운 발견이 아니라
   ALiBi 의 재발견**이다.
4. **logarithmic decay (`mra_bias`) 는 linear decay 보다 외삽에 약함** —
   32× 에서 +40.3 % vs ALiBi −6.2 %. logarithmic 감쇠는 long distance 에서
   너무 천천히 줄어들어 학습 분포 안에서 weak signal 만 제공하고, 외삽 시
   새 거리에 대한 강한 인덕티브 바이어스가 없다. **Length extrapolation
   에는 linear additive bias 가 명확히 우수.**

이 분해는 본 작업의 가장 명확한 결론이다 — π/e/log/RoPE 의 4 가지 distance
처리 방식 중 **linear additive (= ALiBi) 만이 외삽한다**.

---

## 8. ML 이론 함의 — 정직한 평가

### 8.1 무엇이 작동했나

* **Logarithmic distance bias** (ζ explicit-formula 의 `√x` 로부터 유도)
  는 ALiBi 의 합리적 일반화이다.
* **Seed 분산 감소** (RoPE 대비 50–70 % 감소) 는 reproducibility 관점에서
  실용적 가치가 있다 — 큰 모델 / 비싼 학습에서 한 자릿수 작은 분산은 평균
  0.05 PPL 우위보다 훨씬 큰 비용 절감.

### 8.2 무엇이 작동하지 않았나

* **ζ 영점을 frequency 로 직접 사용 (`zeta_log`)**: 이론적으로 매력적이지만
  실측에서 다중 스케일 해상도 손실로 인한 명확한 LOSS. 이 결과는 "Riemann
  영점이 LM positional structure 를 직접 부여한다"는 strong 한 가설이 적어도
  소규모에서는 성립하지 않음을 시사한다.
* **ζ amplitude weighting 단독**: 식적으로는 RoPE 의 `cos` 채널에 `sin` 채널
  을 섞는 효과지만, 실측 영향은 거의 없거나 약간 부정적.
* **부트스트랩 sparsity, spectral norm**: 작은 모델 / 짧은 학습에서는 도움
  이지만 충분 수렴 시점에서 over-regularization. CE 부트스트랩 비율이 LM
  attention 에 직접 transfer 되지 않는다는 시사.

### 8.3 어떤 가설이 다음 단계에서 검증 가능한가

* **Length extrapolation**: 짧은 context 학습 후 긴 context 에서의 안정성.
  `mra_bias` 의 logarithmic decay 는 RoPE 의 wrap-around 한계를 넘어설 가능성
  이 있다. 별도 측정 필요.
* **대형 모델 분산**: 분산 감소가 모델 크기와 함께 어떻게 변하는지.
* **Attention 스펙트럼**: 학습된 MRA 의 attention 행렬 고유값 분포가 GUE
  통계 [13] 를 따르는지 직접 측정.

---

## 9. Related Work

- **RoPE** [1]: 회전 기반 상대위치, 본 연구의 직접적 baseline.
- **ALiBi** [2]: 거리 감쇠 bias, `mra_bias` 의 logarithmic 변종이 일반화.
- **xPos** [10]: RoPE + exponential decay. ad-hoc 결합.
- **YaRN / LongRoPE** [3, 4]: 경험적 frequency rescaling.
- **Geometric Deep Learning** [11]: 기하학적 inductive bias 일반.
- **Equivariant Transformers** [12]: 대칭성 내장.

본 연구는 ζ 영점이라는 **수론** 객체에서 식을 직접 유도하려 시도했다는 점
에서 새롭다. 결과는 mixed: ζ amplitude 와 logarithmic decay 는 작은 효과,
ζ frequency 는 negative 효과.

---

## 10. Limitations & Future Work

### 10.1 한계

1. **소규모 실험**: 210 K 파라미터, 400 K char 코퍼스. 본격 결론을 위해서는
   1 M+ 파라미터 × 1 M+ 토큰 × 5+ seed 가 필요.
2. **Char-level only**: BPE / sentencepiece 토큰화에서의 거동 미검증.
3. **Length extrapolation 미정밀 측정**: 별도 실험 필요.
4. **Native 백엔드 부재**: PyTorch-only, wall-clock 비교 손해. Rust/CUDA 포팅
   가능.
5. **Causal LM 가정**: bidirectional encoder 에서는 Hermitian knob 의 본격
   효과 검증 가능하나 본 작업 범위 밖.

### 10.2 후속

1. Length extrapolation: train N=64, eval N={128, 256, 512} 에서
   `mra_bias` 의 logarithmic decay 가 RoPE 의 periodic wrap 보다 실제로 잘
   버티는지.
2. Scale law: `d_model ∈ {128, 256, 512}` × `n_layers ∈ {4, 8, 12}` 에서
   분산 감소 효과의 안정성.
3. Attention spectrum: 학습된 attention 의 spectral statistics 측정.
4. Bidirectional MRA: `hermitian=True` 모드를 BERT-style 인코더에서 검증.
5. GaugeLattice FFN 결합: AGI 5대 원리 중 채널 분할까지 묶은 full ClarusBlock.

---

## 11. Conclusion

본 작업의 핵심 결론 세 가지:

1. **ζ 영점을 attention frequency 로 직접 사용하는 시도는 실패** — γ_k 의
   범위가 RoPE 의 다중 스케일 해상도를 따라가지 못한다. 본격 검증 후 폐기.
2. **분산 감소** — lean MRA 변종들이 RoPE 의 1/3 ~ 1/2 σ_PPL 을 달성. 평균
   PPL 동급 + reproducibility 우위.
3. **Length extrapolation 의 진짜 메커니즘은 linear additive distance bias
   (= ALiBi)** — § 7.7 의 32× 분해가 보여주듯, π-rotation / log decay /
   block-aware base / ζ frequency 모두 외삽에 무력하고 오직 e-decay (=
   ALiBi 변종) 만이 외삽한다. EulerCE 의 외삽 우위는 ALiBi 의 재발견이며,
   이 결론은 향후 **long-context attention 설계는 linear additive bias
   계열에 집중**해야 함을 시사.

본 작업의 가장 큰 가치는 ζ-attention 가설의 negative findings 를 명시하고,
Euler-CE 외삽 우위의 진짜 메커니즘 (e-decay = ALiBi) 을 32× extrapolation
까지 분해한 데에 있다.

---

## References

[1] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.

[2] O. Press et al., "Train Short, Test Long: Attention with Linear Biases (ALiBi)," ICLR 2022.

[3] B. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023.

[4] Y. Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.

[5] D. Hilbert, G. Pólya, conjecture (folklore, ca. 1910s).

[6] M. V. Berry, J. P. Keating, "H = xp and the Riemann zeros," 1999.

[7] (Internal) `docs/6_뇌/sleep.md` — CE 부트스트랩 고정점 `ε² = 4.87 %`.

[8] (Internal) `docs/7_AGI/2_Architecture.md` § 4 — 유니타리 제약.

[9] (Internal) `clarus/ce_euler.py` — Euler-CE attention.

[10] Y. Sun et al., "A Length-Extrapolatable Transformer (xPos)," ACL 2023.

[11] M. M. Bronstein et al., "Geometric Deep Learning," 2021.

[12] V. G. Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML 2021.

[13] H. L. Montgomery, "The pair correlation of zeros of the zeta function," 1973.

---

## Appendix A. 200-step transient (참고용)

소규모 짧은 학습 (200 step × 2 seed) 에서는 분산이 더 dramatic 하게 보였으나
(σ_PPL: RoPE 1.22, mra 0.16, mra_h 0.06; 7-20× 감소), 1500-step 충분 수렴 시점
에는 σ 가 모두 0.07-0.21 범위로 평준화된다. **소규모 실험의 분산 격차는
early-stopping artifact 이며 영구적 안정성 우위가 아니다**. 1500-step 결과
(§ 7) 만이 신뢰할 수 있는 결론.

이 발견 자체가 ML 평가에서 reproducibility 의 함정을 보여주는 사례 — 짧은
학습에서 본 "결정적 차이" 는 충분히 학습하면 사라질 수 있다.
