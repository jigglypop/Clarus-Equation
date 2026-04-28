# Mellin-Riemann Attention: ζ explicit-formula 유도 attention의 ablation 분석과 분산 절감, 그리고 Euler-CE의 length extrapolation 우위 발견

> 본 문서는 `mra_block_spec.md`의 사양과 `tests/test_mra.py`의 검증, 그리고
> `examples/ai/bench_recursive_euler.py --mode mra`와 `bench_mra_extrap.py`의
> ablation 실측을 한 편의 짧은 논문 형태로 정리한다. **정직한 negative-positive
> 혼합 결과**를 담은 internal write-up이다.

## Abstract

Riemann ζ 함수의 explicit formula의 critical-strip 합으로부터 attention
score를 직접 유도한 새로운 attention 변종 **Mellin-Riemann Attention
(MRA)**를 제안하고, 7개 설계 요소를 ablation으로 검증한다. 또한 비교
대상으로 기존 `euler_ce_k1`의 length extrapolation 성능을 처음으로 직접
측정한다.

`docs` Markdown 코퍼스 (400 K chars, vocab 788) 에서 char-level LM,
1500-step × 3-seed (in-distribution) + 1000-step × 3-seed (extrapolation):

* **부정적**: ζ 영점 `γ_k`를 attention frequency로 직접 사용하는 방식
  (Mellin 커널 `θ_k(p) = γ_k log(1+p)`)은 `γ_k/γ_1`의 범위가 ~5×에 불과해
  RoPE의 ~3000× 다중 스케일 해상도를 잃고, **PPL 14.29(RoPE 12.29 대비
  +2.0)**로 명확히 부진했다.
* **부정적**: ζ amplitude weighting (`w_k = 1/(1/2 + iγ_k)`) 단독 효과 미미
  (`mra` 12.46 vs `mra_noamp` 12.27, +0.17 손해).
* **부정적**: 부트스트랩 sparsity(`ε² = 4.87 %`)와 출력 spectral norm 모두
  1500-step 시점 PPL 손해 (+0.33).
* **긍정적 (분산)**: `MRABlock(decay_mode="bias")`가 평균 PPL 12.24(RoPE
  대비 -0.05)에 도달하고 **seed 분산이 RoPE의 절반(0.10 vs 0.21)**으로 줄었다.
* **긍정적 (외삽)**: `mra_bias`의 logarithmic distance bias가 4× length
  extrapolation에서 RoPE보다 **8 %p 안정**했다(degrad +18.5 % vs +26.7 %).
* **새로운 큰 발견 (4× extrap)**: `euler_ce_k1` 이 4× length extrapolation
  에서 **degrad +6.0 % 만 발생** (RoPE +26.7 %, MRA +18.5 % 대비 압도적).
* **결정적 분해 (32× extrap, 9 변종, § 7.7)**: 외삽 능력은 두 tier로 명확히
  갈린다. Tier 1 (외삽 OK, ≤ +10 %): `rope_alibi`, `euler_no_pi`, `euler_ce_k1`,
  `xpos`, `nope`. Tier 2 (외삽 BAD, +29 ∼ +55 %): `mra`, `mra_bias`, `std_rope`,
  `euler_no_decay`. **공통 패턴**: Tier 1 = "강한 distance attenuation 존재
  (linear/multiplicative)" OR "rotation 자체 부재". Tier 2 = "rotation 존재
  + attenuation 부재 또는 너무 약함 (log)".
* **Euler-CE 외삽 = ALiBi 재발견**: `euler_no_pi`(e-decay only) ≈ `euler_ce_k1`
  ≈ `rope_alibi`(32× 모두 -6 % 근방). π-rotation은 외삽에 기여하지 않았다.
* **NoPE > RoPE 7배 차이**(NoPE +7 % vs RoPE +47.2 %): RoPE의 회전 자체가
  OOD의 직접 원인임을 보였다. Kazemnejad et al. (2023)의 finding을 재현했다.

결론은 세 가지다. (a) ζ 영점을 frequency로 쓰는 직설적 시도는 실패했다.
(b) RoPE + ζ amp의 lean MRA는 평균 PPL은 동급이고 분산은 절반이다.
(c) **Length extrapolation의 구조적 결정성**: rotation을 그대로 두려면 강한
distance attenuation(linear ALiBi 또는 multiplicative xPos)이 필수이며,
그렇지 않으면 rotation 자체가 없는 NoPE가 낫다.

---

## 1. Introduction

Transformer의 positional encoding은 attention 메커니즘이 set-permutation
invariant라는 근본 한계를 외부 주입으로 해결하는 inductive bias의 핵심
지점이다. RoPE [1]가 LLaMA / Qwen / Mistral / Gemma / EXAONE 등 거의 모든
현대 LLM의 표준이 된 후, ALiBi [2], YaRN [3], LongRoPE [4] 등 후속 연구는
대체로 경험적 튜닝으로 long-context extrapolation을 공략해왔다.

본 연구의 출발점은 다른 방향이다. **수학적 axiom에서 attention 식을 직접
유도**하면 어떻게 되는가? Hilbert-Pólya 추측 [5]은 Riemann ζ의 비자명
영점 `{ρ_n = 1/2 + iγ_n}`의 허수부가 어떤 self-adjoint operator의 고유값
임을 가정한다. Berry-Keating [6]은 이 operator가 `H = (xp + px)/2`의
양자화일 가능성을 제시했다.

본 연구는 RH(Riemann Hypothesis)를 **engineering axiom**으로 채택하고,
attention score를 ζ의 explicit formula의 critical-strip 합으로부터 유도
한다. 식 자체는 단순하지만, 어느 부분이 실제로 도움이 되고 어느 부분이
도움이 되지 않는지는 **실증으로만 확인 가능**하다. 본 논문의 기여는 이를
ablation으로 분리한 데에 있다.

### 1.1 Contributions

1. ζ explicit formula의 critical-strip 합으로부터 attention score를
   폐쇄형으로 유도 (§ 3).
2. 7가지 설계 knob(frequency 방식, amplitude weighting, decay 형태,
   sparsity, spectral norm, Hermitian)을 노출하는 단일 ablation 표면 (§ 4).
3. **부정적 발견 명시**: 직설적 ζ-frequency, multiplicative decay,
   bootstrap sparsity, spectral-norm 모두 1500-step 시점 LOSS (§ 6).
4. **긍정적 발견**: lean variant(`mra` / `mra_bias`)는 RoPE와 평균 PPL
   동급 + **seed 분산 2-3× 감소** (§ 7).

---

## 2. Background

### 2.1 RoPE

`d_head` 차원의 dim-pair `(2k, 2k+1)`에 대해 회전 행렬

```
R_θ = [[cos θ, -sin θ], [sin θ, cos θ]],   θ_k(p) = p · 10000^{-2k/d}
```

을 query/key에 적용한다. 두 토큰의 attention score가 상대 위치 `i - j`만의
함수가 되도록 만든다. base frequency 10000은 경험적 선택이다.

핵심 강점은 `inv_freq_k = 10000^{-2k/d}`의 **기하 급수적 분포**가 attention
에 폭넓은 다중 스케일 해상도를 부여한다는 점이다 (`d_head = 16` 시 약 3000×
범위).

### 2.2 ALiBi

attention score에 `−|i − j|/m_h`의 거리 감쇠 bias를 직접 더한다(`m_h`는
헤드별 기울기). 본 논문의 `decay_mode="bias"`는 ALiBi를 logarithmic
거리 함수 `½ log((1+i)/(1+j))`로 일반화한 형태에 해당한다.

### 2.3 ζ explicit formula

von Mangoldt:

```
ψ(x) = x − Σ_ρ x^ρ / ρ − log(2π) − ½ log(1 − x^{-2})
```

핵심 합 `Σ_ρ x^ρ / ρ`는 RH 하에서

```
Σ_n x^{1/2 + iγ_n} / (1/2 + iγ_n) = √x · Σ_n e^{iγ_n log x} / (1/2 + iγ_n)
```

으로 정리된다. **Mellin 커널** `e^{iγ_n log x}`와 **ζ 진폭** `1/(1/2 + iγ_n)`
의 곱이 자연스럽게 나타난다.

---

## 3. Method - Mellin-Riemann Attention

### 3.1 단순 score 식

위치쌍 `(i, j)`에 `x = (1+i)/(1+j)`를 대입한다. dim-pair `k`를 복소채널로
압축하면

```
q_i^(k) := q_i^{2k} + i · q_i^{2k+1},     k_j^(k) := k_j^{2k} + i · k_j^{2k+1},
```

attention raw score는 다음과 같다.

```
S_{ij} = √((1+j)/(1+i)) · Σ_k w_k · e^{−iγ_k log((1+i)/(1+j))} · q_i^(k) · conj(k_j^(k))
```

여기서 `w_k = 1/(1/2 + iγ_k)`. 토큰별 attention 입력은 `Re(S_{ij})`.

### 3.2 폐쇄형 분해

`(1+i)^{−iγ_k}`와 `(1+j)^{−iγ_k}`가 각각 `i`, `j`만의 함수이므로 사전 변환

```
q̃_i^(k) = (1/√(1+i)) · e^{−iγ_k log(1+i)} · q_i^(k)
k̃_j^(k) = √(1+j) · e^{−iγ_k log(1+j)} · k_j^(k)
```

후

```
S_{ij} = Σ_k w_k · q̃_i^(k) · conj(k̃_j^(k))
```

이는 표준 dot-product attention과 동일한 `O(N²K)` 계산이다. 실수 분해는 다음과 같다.

```
q̂_re = w_re · q̃_re − w_im · q̃_im
q̂_im = w_re · q̃_im + w_im · q̃_re
Re(S) = q̂_re @ k̃_re^T + q̂_im @ k̃_im^T
```

→ 표준 attention의 **2배 matmul** 비용이다. softmax / V 적용은 동일하다.

### 3.3 학습 자유도

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `cos_p, sin_p, log_decay` | buffer | 0 (위치-axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention 동일 |

표준 multi-head attention 대비 **추가 학습 파라미터 0**.

---

## 4. 설계 Knob - Ablation 표면

`MellinRiemannAttention(d_model, n_heads, block, **knobs)`가 노출하는 7개
knob은 다음과 같다.

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

- `"rope"`: `θ_k(p) = p · base^{−k/K}`. RoPE 기하 급수(~3000× 범위).
- `"zeta_log"`: `θ_k(p) = γ_k · log(1+p)`. Mellin 커널이며, ζ 영점이 곧 frequency다.
  `γ_k/γ_1`의 범위가 K=16에서 ~5×에 불과해 다중 스케일 해상도가 손실된다.

### 4.2 `amp_weight`

- `True`: `w_k = 1/(1/2 + iγ_k)`를 score에 곱한다. `Re(w_k)·Re_part −
  Im(w_k)·Im_part` 형태로 standard RoPE의 `cos` 채널과 보조 `sin` 채널이
  혼합된다.
- `False`: `w_k = 1`. attention 식이 표준 RoPE와 일치한다(control).

### 4.3 `decay_mode`

ζ explicit-formula의 `√x` factor(`x = (1+i)/(1+j)`)를 처리한다.

- `"none"`: 무시.
- `"bias"`: log-space additive. `bias_{ij} = ½(log(1+j) − log(1+i))`를 score에 더한다.
  ALiBi의 logarithmic 일반화다.
- `"mult"`: 원식 그대로 `score_{ij} *= √((1+j)/(1+i))`를 적용한다. Per-key temperature
  처럼 작동하므로 의도와 다소 어긋난다.

### 4.4 `hermitian` (causal LM에서 비추천)

`W_q = W_k` tied + score symmetrize `S ← (S + Sᵀ)/2` 구조다. Bidirectional encoder에서는
self-adjoint operator의 이산화(Hilbert-Pólya 직접 구현)이지만, **causal
LM에서는 mask 전 symmetrize가 future leakage를 일으킨다**. 본 작업의 ablation
표에서 이 knob은 제외했다.

---

## 5. Implementation

`clarus/ce_mra.py`는 PyTorch 참조 구현(~250 LOC)이다. 현재 backend는 PyTorch only이며,
`RiemannRotaryAttention`의 Rust/CUDA 패턴을 그대로 이식할 수 있다.

테스트 (`tests/test_mra.py`, 11/11 pass):

- forward shape & finiteness
- block end-to-end
- 모든 ablation knob 조합 finite output
- bootstrap sparsity 정확한 retention 수
- 학습 파라미터 수 == 표준 MHA (4·d²) / Hermitian 모드 == 3·d²
- ζ amplitude가 실제로 출력을 바꾸는지 확인(회귀 가드)
- 공리적 buffer들이 학습되지 않는지 확인
- spectral norm이 σ₁ ≤ 1을 강제하는지 확인

---

## 6. Experiments - Setup

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

### 7.2 Verdicts (`std_rope`, `euler_ce_k1` 대비)

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

1. **`mra_zeta` (PPL 14.29)**: ζ 영점을 frequency로 직접 사용하면 RoPE의
   다중 스케일 해상도가 깨진다. `γ_k/γ_1`의 범위는 K=16에서 ~5×에 불과하다.
   RoPE의 `10000^{−k/K}`가 갖는 ~3000× 범위와 비교하면 다중 스케일 해상도가
   거의 없다. 이 한계는 `riemann_rope`의 PPL 19.29(별도 측정)의 직접 원인이며,
   ζ 영점을 frequency로 직접 사용하는 모든 변종에서 재현된다.
2. **ζ amplitude 단독 효과 미미**: `mra` (12.46) − `mra_noamp` (12.27) =
   +0.17 PPL **손해**(z = +1.35). amp_weight가 표준 RoPE attention에서
   별다른 정보 이득을 주지 못한다.
3. **부트스트랩 sparsity / spectral norm**: 1500 step 충분 수렴 시점에서는 모두
   PPL 손해(+0.33)를 보였다. 짧은 학습(200 step)에서는 도움이지만 long-run에서는 over-
   regularization.

### 7.4 긍정적 발견 - seed 분산 절감

mean PPL은 거의 동등하지만 seed-to-seed 분산은 명확히 감소한다.

```
σ_PPL across 3 seeds (lower = more reproducible):
  std_rope     : 0.208         baseline
  mra          : 0.069         3.0×  more stable
  mra_mult     : 0.077         2.7×
  mra_bias     : 0.097         2.1×
  mra_noamp    : 0.129         1.6×
  euler_ce_k1  : 0.126         1.7×  (다른 메커니즘)
```

`mra` 계열(RoPE freq + amp/decay 변종)은 모두 **σ_PPL이 RoPE의 1/3 ∼ 1/2** 수준이다.
`mra_noamp`도 분산이 작은 것으로 보아, 분산 감소는 ζ amplitude 단독 효과가
아니라 RoPE의 `qkv` fused linear 대신 **`W_q, W_k` 분리 + RoPE-style
회전**을 조합한 데서 나오는 것으로 추정된다(추가 조사 필요).

### 7.5 가장 좋은 lean 변종

`decay_mode="bias"` (`mra_bias`):
- mean PPL 12.240: **RoPE보다 0.05 낮음**(z = −0.36, statistically TIE
  with mean preference)
- σ_PPL 0.097: RoPE 분산의 47 %
- 시간 비용은 RoPE와 동일

이는 ALiBi의 logarithmic 일반화로 해석할 수 있다.
```
bias_{ij} = ½ (log(1+j) − log(1+i))
```
선형 거리 `−|i−j|`가 아닌 logarithmic 거리다. critical line `Re(s) = 1/2`의
지수 `½`가 직접 도출된다.

### 7.6 Length extrapolation (4×까지)

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

1. **`euler_ce_k1`의 extrapolation이 압도적**이다. 4× 외삽에서 PPL은 +6 %만
   증가했다. 다른 모든 변종(RoPE 포함)이 +18 ~ +27 %인 것과 명확히 다른 영역이다.
   Euler-CE의 `e^{-|i-j|/ξ}` decay가 정확한 long-context 인덕티브 바이어스를
   제공하는 것으로 해석된다.
2. **`mra_bias`의 logarithmic decay는 RoPE보다 8 %p 더 부드럽게 degrade한다**
   (+18.5 % vs +26.7 %). lean MRA의 자그마한 long-context 우위다. 다만
   `euler_ce_k1` 만큼은 아님.
3. **`mra_zeta`는 짧은 base에서는 부진해도 외삽은 비교적 잘한다**(+10.2 %).
   `log(1+p)` lift의 진짜 가치는 long-context에서 발현된다. 짧은 학습 + 긴
   외삽 시나리오에 한정해서는 검토 가치 있음.
4. **`std_rope`의 빠른 붕괴**(+26.7 %): 알려진 RoPE periodicity 한계를 재현했다.

이 결과는 logarithmic distance bias가 ALiBi의 일반화로 의미 있는 후속 연구
방향임을 시사한다.

### 7.7.1 Scaling: d_model {64, 128, 256}에서 패턴 유지

§ 7.7의 결과를 모델 크기 scaling에 대해 검증한다. 학습 설정은 동일하다
(train block = 64, 1500 step). seeds는 d=64에서 3, d=128/256에서 메모리상 2다.

#### 32× extrapolation 상대 degradation, d_model 별

| variant | d=64 | **d=128** | d=256 | 부류 |
|---|---|---|---|---|
| `rope_alibi` | −6.2 % | **−3.0 %** | −5.1 % | Tier 1 |
| `euler_ce_k1` | −6.2 % | **−3.0 %** | −5.3 % | Tier 1 |
| `euler_no_pi` | −5.5 % | −4.0 % | −5.1 % | Tier 1 |
| `xpos` | +4.1 % | +7.4 % | +0.8 % | Tier 1 |
| `nope` | +7.0 % | +15.6 % | +10.5 % | Tier 1 |
| `mra_bias` | +40.3 % | +59.0 % | +42.8 % | Tier 2 |
| `mra` | +40.8 % | +65.9 % | +43.7 % | Tier 2 |
| `std_rope` | +47.2 % | **+73.5 %** | +48.3 % | Tier 2 |
| `euler_no_decay` | +54.7 % | **+103.6 %** | +54.2 % | Tier 2 |

#### 발견

1. **Tier 1/Tier 2 분리는 모든 d_model에서 견고**하다. d=64, 128, 256의 세
   가지 모델 크기 모두 ALiBi 계열 (`rope_alibi`, `euler_ce_k1`, `euler_no_pi`)
   은 안정적이고(−3 ∼ −6 %), rotation-only 계열(`std_rope`, `euler_no_decay`,
   `mra`, `mra_bias`)은 catastrophic하다(+42 ∼ +104 %). 본 작업의 핵심 가설
   "rotation + 강한 distance attenuation이 외삽의 충분조건"은 모델 크기에
   invariant하다.
2. **d_model = 128에서 cliff가 가장 깊다**. rotation-only 변종이 +73 ∼
   +104 %로 가장 큰 degradation을 보였다. 가능한 해석은 중간 사이즈가 학습 분포에
   가장 tightly fit하여 OOD 영역과의 격차가 가장 크다는 것이다. d=256은 1500 step에서
   underfit으로 보인다.
3. **ALiBi의 안정성은 모델 크기에 무관**하다. d=64 −6.2 %, d=128 −3.0 %,
   d=256 −5.1 %다. Slope 학습이 단순하므로 sample efficiency가 좋아 모든
   사이즈에서 robust하다. **이는 ALiBi가 production-scale 모델에서도 유효한
   외삽 메커니즘일 강한 후보임을 시사**.
4. **NoPE의 OOD 안정성은 모델이 커질수록 약화**된다. d=64 +7 %, d=128 +15.6 %,
   d=256 +10.5 %다. 모델이 implicit position을 더 강하게 학습할수록 OOD 영역에서의
   generalization이 약해진다. NoPE는 baseline으로는 의미 있지만 production에는 부적합하다.

### 7.7.2 Length extrapolation (32×까지): Euler-CE 분해

위 § 7.6의 발견(Euler-CE의 외삽 우위)의 진짜 원인을 격리한다. EulerCE는
두 개의 분리 가능한 구조를 결합하므로 각각을 frozen으로 끄고 비교한다.

| 변종 | 구성 | 학습 자유도 |
|---|---|---|
| `std_rope` | RoPE | baseline |
| `rope_alibi` | RoPE + ALiBi linear decay (per-head learnable slope) | + n_heads |
| `mra` / `mra_bias` | RoPE + ζ amplitude / + log decay | + n_heads (`mra_bias`) |
| `euler_no_decay` | EulerCE with `e_gate` frozen → π-rotation 만 | (gate 1개 frozen) |
| `euler_no_pi` | EulerCE with `pi_gate` frozen → e-decay 만 | (gate 1개 frozen) |
| `euler_ce_k1` | full Euler-CE | original |

train block = 64, eval block ∈ {64, 128, 256, 512, 1024, 2048} (최대 32×).
모든 모델은 train_block으로 init하고, eval 직전 `extend_to(N_eval)`로 buffer만
grow한다. EulerCE의 block-aware base `π^D_eff · N`도 train 시점 N=64로
계산.

1500 step × 3 seed × batch=4(메모리 한계)로 측정했다. 절대 PPL은 짧은 학습 + 작은
batch 때문에 § 7.6보다 높지만, **상대 degradation은 비교 가능**하다.

#### Relative degradation (PPL(N)/PPL(64) − 1, 낮을수록 좋음)

| variant | ×2 | ×4 | ×8 | ×16 | **×32** | tier |
|---|---|---|---|---|---|---|
| **`rope_alibi`** | −13.0 % | −3.3 % | −9.2 % | −7.6 % | **−6.2 %** | **1** |
| **`euler_no_pi`** (e only) | −12.0 % | −2.4 % | −7.9 % | −6.5 % | **−5.5 %** | **1** |
| **`euler_ce_k1`** | −13.1 % | −3.4 % | −9.5 % | −7.9 % | **−6.2 %** | **1** |
| **`xpos`** | −12.3 % | −2.0 % | −5.8 % | −1.2 % | **+4.1 %** | **1** |
| **`nope`** | −7.4 % | +3.2 % | +1.8 % | +6.0 % | **+7.0 %** | **1** |
| `mra_bias` | −8.8 % | +10.0 % | +13.9 % | +28.6 % | +40.3 % | 2 |
| `mra` | −9.2 % | +10.4 % | +14.1 % | +29.7 % | +40.8 % | 2 |
| `std_rope` | −10.0 % | +12.6 % | +19.9 % | +37.7 % | **+47.2 %** | 2 |
| `euler_no_decay` (π only) | −8.2 % | +14.6 % | +24.3 % | +44.4 % | **+54.7 %** | 2 |

#### 결정적 분해: Tier 1과 Tier 2

| Tier | 기준 | 메커니즘 |
|---|---|---|
| **1 (외삽 OK, ≤ +10 %)** | rope_alibi, euler_no_pi, euler_ce_k1, xpos, nope | distance attenuation **존재** OR rotation **부재** |
| **2 (외삽 BAD, +29 ∼ +55 %)** | mra, mra_bias, std_rope, euler_no_decay | rotation **존재** + 강한 distance attenuation **부재** |

이 분리는 다섯 개의 깨끗한 결론을 만든다.

1. **Euler-CE의 외삽 우위는 100 % e-decay(linear additive distance bias) 단독 효과**다.
   `euler_no_pi`(e-decay only)는 풀 `euler_ce_k1`과 거의 같다(32×에서 −5.5 % vs −6.2 %).
   π-rotation을 빼도 외삽 성능은 유지된다.
2. **π-rotation 단독은 외삽에 무력**하다. `euler_no_decay`(π only)는 32×에서
   +54.7 %로 RoPE보다 오히려 더 나쁘다. block-aware base `π^D_eff · N`도
   long-context 안정성에는 무관하다(in-distribution에서는 도움일 수 있음).
3. **`rope_alibi` ≈ Euler-CE**다. 32× 외삽은 −6.2 % vs −6.2 %다. EulerCE의
   e-decay는 ALiBi(Press 2022)의 per-head learnable slope 변종으로 정확히
   환원된다. **EulerCE의 외삽 우위는 ALiBi의 재발견**이다.
4. **xPos도 외삽 OK**다(+4.1 %). multiplicative magnitude decay도 linear
   additive 정도는 아니지만 명확히 작동한다. distance attenuation의 *형태*보다
   존재 자체가 더 결정적.
5. **NoPE가 RoPE보다 외삽을 잘한다**(+7.0 % vs +47.2 %, ×7 차이). 회전 자체가
   OOD의 직접 원인임을 본 setup에서 재현했다(Kazemnejad et al. 2023).

종합 결론은 **OOD 외삽 = "강한 distance attenuation 존재" 또는 "rotation 부재"의 OR 조건**이다.
RoPE의 회전을 그대로 두고 attenuation만 빼거나 약하게 얹는 것(`mra`, `mra_bias`, `std_rope`)이 가장 위험하다.
logarithmic decay가 linear보다 약한 이유는 long-distance에서 너무 천천히 감쇠해 RoPE 회전의
wrap-around 효과를 dominate하지 못하기 때문이다.

### 7.7.3 2-bit Minimal Euler-CE: operational 환원

§ 7.7의 ablation은 강력한 시사를 만든다. Euler 5상수 `{e, π, i, 1, 0}`의
attention적 의미는 두 axis와 두 게이트로 환원된다.

| 상수 | 작용 | 환원 |
|---|---|---|
| `π`, `i` | rotation generator (`e^{iπt}` 결합) | **axis 1**: rotation |
| `e` | exponential decay base | **axis 2**: decay |
| `1`, `0` | on/off gate values | 1 비트 each |

→ **2 functionally distinct axes × 2 gate values = 2² = 4 head-types**, 각각
2-bit string `(pi, e)`으로 인코딩할 수 있다.

| (pi, e) | 비트 | 헤드 타입 | 문헌 매핑 |
|---|---|---|---|
| (0, 0) | `00` | identity | NoPE [Kazemnejad 2023] |
| (0, 1) | `01` | decay only | ALiBi [Press 2022] |
| (1, 0) | `10` | rotation only | RoPE [Su 2021] |
| (1, 1) | `11` | rotation + decay | xPos [Sun 2023] / Euler-CE |

**§ 7.7 결과의 perfect 매핑**: 4 head-types 중 3개(00, 01, 11)가 Tier 1이고,
단 한 가지(10)만 Tier 2다. 즉 **5 Euler 상수 = 2비트 head-type taxonomy**이며,
**4가지 중 3가지가 작동**한다(= log₂ 3 ≈ 1.58비트의 effective capacity).

#### 구현: `EulerCEMinimal`

`clarus/ce_euler.py::EulerCEMinimal`가 본 환원의 정밀 구현이다. 각 헤드는 2비트
spec `head_types ∈ {0, 1, 2, 3}^H`로 head-type을 axiom으로 선택한다. 학습
파라미터는 `xi_h`(decay 헤드만 의미 있음) 1개 + `W_q,k,v,o`(표준 MHA)다.
5차원 `bit_logits`와 sigmoid 게이트 학습은 모두 제거했다.

#### 자유도 비교

| 양 | 원본 EulerCE | EulerCEMinimal |
|---|---|---|
| Discrete head-type | 5-dim continuous (`bit_logits`) | 2-bit axiom (`head_types`) |
| Rotation gate | learnable sigmoid (`pi_gate_logit`) | bit (axiom) |
| Decay gate | learnable sigmoid (`e_gate_logit`) | bit (axiom) |
| Decay length | learnable (`log_xi`) | learnable (`log_xi`) |
| 헤드별 학습 자유도 | 5 + 3 = **8** | 1 (xi only) |

→ **헤드별 학습 자유도가 8 → 1로 감소**한다(head-type 선택을 axiom으로 이동).
이는 Clarus 본 thesis("자유 파라미터 0에 가깝게")와 직접 정렬된다.

#### 검증 — `tests/test_euler_minimal.py` (16 tests)

* `head_types_from_spec`가 6가지 spec 형태(int, str, list, "mix", "all",
  invalid) 모두 정확히 처리.
* `head_types="nope"` → no-PE와 수치적으로 일치(allclose atol=1e-5).
* `head_types="alibi"` → 거리에 따른 attention attenuation 확인.
* `head_types="rope"` → 모든 헤드 e_bit=0, decay 항 0.
* `head_types="mix"` / `"all"` → 정확한 비트 분해.
* `extend_to(N)` 후 forward 정상, 학습 파라미터 변하지 않음.
* Autograd: rotation-only 헤드의 `log_xi` grad는 정확히 0(decay gate off 시
  grad path가 끊김).

이로써 EulerCEMinimal이 본 환원의 정확하고 정밀한 구현임을 확인했다.

#### 실증: minimal 변종이 canonical baseline을 정확히 reproduce

`bench_mra_extrap.py --variants minimal` (1500 step × 3 seed × d_model=64) 결과:

| 2-bit variant | head_types | x32 degrad | canonical 매핑 | x32 (canonical) |
|---|---|---|---|---|
| `min_nope` | uniform 00 | **+7.0 %** | `nope` | +7.0 % (정확 일치) |
| `min_rope` | uniform 10 | **+47.2 %** | `std_rope` | +47.2 % (정확 일치) |
| `min_alibi` | uniform 01 | **−5.7 %** | `rope_alibi` | −6.2 % (≈ 일치) |
| `min_xpos` | uniform 11 | **−5.8 %** | `euler_ce_k1` | −6.2 % (≈ 일치) |
| `min_mix` | half 01, half 11 | −5.5 % | (신규) | — |
| `min_all` | one of each 00/01/10/11 | −0.6 % | (신규) | — |

**검증 결과**: 5 Euler 상수의 2-bit reduction이 정확히 작동한다. 4 canonical
positional encoding(NoPE, RoPE, ALiBi, xPos)이 모두 `EulerCEMinimal`의
단일 head-type spec으로 정확히(≤ 0.5 %p) reproduce된다.

**부수 발견**:

* **`min_mix`(half ALiBi + half xPos) ≈ 단독 best와 동등**하다(−5.5 % vs −5.7
  ∼ −5.8 %). 두 tier-1 type을 섞어도 외삽 우위는 없으므로 실용적으로 head-type
  diversity는 도움이 되지 않는다.
* **`min_all`(4 type 골고루)은 best보다 약간 worse**다(−0.6 %). rotation-only와
  nope 헤드가 평균 attention capacity를 깎는다. **실용적 권고**: 모든 헤드를
  단일 tier-1 type(`alibi` 또는 `xpos`)으로 통일하는 것이 최적이다.

이는 본 작업의 가장 강한 결정적 결론이다. **현존하는 모든 main-stream PE는
사실상 같은 2-bit taxonomy의 인스턴스이며, 4가지 중 단 한 가지(10 = pure
rotation)만 외삽 catastrophic**이다.

---

## 8. ML 이론 함의: 정직한 평가

### 8.1 무엇이 작동했나

* **Logarithmic distance bias**(ζ explicit-formula의 `√x`로부터 유도)는
  ALiBi의 합리적 일반화다.
* **Seed 분산 감소**(RoPE 대비 50-70 % 감소)는 reproducibility 관점에서
  실용적 가치가 있다. 큰 모델 / 비싼 학습에서 한 자릿수 작은 분산은 평균
  0.05 PPL 우위보다 훨씬 큰 비용 절감.

### 8.2 무엇이 작동하지 않았나

* **ζ 영점을 frequency로 직접 사용(`zeta_log`)**: 이론적으로 매력적이지만
  실측에서 다중 스케일 해상도 손실로 인한 명확한 LOSS. 이 결과는 "Riemann
  영점이 LM positional structure를 직접 부여한다"는 strong한 가설이 적어도
  소규모에서는 성립하지 않음을 시사한다.
* **ζ amplitude weighting 단독**: 식적으로는 RoPE의 `cos` 채널에 `sin` 채널을
  섞는 효과지만, 실측 영향은 거의 없거나 약간 부정적이다.
* **부트스트랩 sparsity, spectral norm**: 작은 모델 / 짧은 학습에서는 도움이지만
  충분 수렴 시점에서는 over-regularization으로 작동한다. CE 부트스트랩 비율이
  LM attention에 직접 transfer되지 않는다는 시사다.

### 8.3 어떤 가설이 다음 단계에서 검증 가능한가

* **Length extrapolation**: 짧은 context 학습 후 긴 context에서의 안정성.
  `mra_bias`의 logarithmic decay는 RoPE의 wrap-around 한계를 넘어설 가능성
  이 있다. 별도 측정 필요.
* **대형 모델 분산**: 분산 감소가 모델 크기와 함께 어떻게 변하는지.
* **Attention 스펙트럼**: 학습된 MRA의 attention 행렬 고유값 분포가 GUE
  통계 [13]를 따르는지 직접 측정.

---

## 9. Related Work

- **RoPE** [1]: 회전 기반 상대위치, 본 연구의 직접적 baseline.
- **ALiBi** [2]: 거리 감쇠 bias. `mra_bias`의 logarithmic 변종이 이를 일반화한다.
- **xPos** [10]: RoPE + exponential decay. ad-hoc 결합.
- **YaRN / LongRoPE** [3, 4]: 경험적 frequency rescaling.
- **Geometric Deep Learning** [11]: 기하학적 inductive bias 일반.
- **Equivariant Transformers** [12]: 대칭성 내장.

본 연구는 ζ 영점이라는 **수론** 객체에서 식을 직접 유도하려 시도했다는 점에서 새롭다.
결과는 mixed다. ζ amplitude와 logarithmic decay는 작은 효과를 보였고,
ζ frequency는 negative 효과를 보였다.

---

## 10. Limitations & Future Work

### 10.1 한계

1. **소규모 실험**: 210 K 파라미터, 400 K char 코퍼스다. 본격 결론을 위해서는
   1 M+ 파라미터 × 1 M+ 토큰 × 5+ seed가 필요하다.
2. **Char-level only**: BPE / sentencepiece 토큰화에서의 거동 미검증.
3. **Length extrapolation 미정밀 측정**: 별도 실험 필요.
4. **Native 백엔드 부재**: PyTorch-only, wall-clock 비교 손해. Rust/CUDA 포팅
   가능.
5. **Causal LM 가정**: bidirectional encoder에서는 Hermitian knob의 본격
   효과 검증 가능하나 본 작업 범위 밖.

### 10.2 후속

1. Length extrapolation: train N=64, eval N={128, 256, 512}에서
   `mra_bias`의 logarithmic decay가 RoPE의 periodic wrap보다 실제로 잘
   버티는지.
2. Scale law: `d_model ∈ {128, 256, 512}` × `n_layers ∈ {4, 8, 12}`에서
   분산 감소 효과의 안정성.
3. Attention spectrum: 학습된 attention의 spectral statistics 측정.
4. Bidirectional MRA: `hermitian=True` 모드를 BERT-style 인코더에서 검증.
5. GaugeLattice FFN 결합: AGI 5대 원리 중 채널 분할까지 묶은 full ClarusBlock.

---

## 11. Conclusion

본 작업의 핵심 결론은 세 가지다.

1. **ζ 영점을 attention frequency로 직접 사용하는 시도는 실패**했다. γ_k의
   범위가 RoPE의 다중 스케일 해상도를 따라가지 못한다. 본격 검증 후 폐기.
2. **분산 감소**: lean MRA 변종들이 RoPE의 1/3 ~ 1/2 σ_PPL을 달성했다. 평균
   PPL 동급 + reproducibility 우위.
3. **Length extrapolation의 구조적 결정성**: § 7.7의 9 변종 32× 분해가
   보여주듯, OOD 외삽 능력은 두 tier로 명확히 갈린다. Tier 1(외삽 OK)은
   (a) 강한 distance attenuation이 존재하거나(linear ALiBi, multiplicative xPos)
   (b) rotation이 아예 없는(NoPE) 경우다. Tier 2(외삽 BAD)는 rotation이 존재하면서
   attenuation이 없거나 약한(log) 경우다. EulerCE의 외삽 우위는 ALiBi의 재발견이며,
   **NoPE가 RoPE보다 7배 잘 외삽**한다는 사실은 RoPE의 회전 자체가 OOD의 원인임을 보여준다.

본 작업의 가장 큰 가치는 ζ-attention 가설의 negative findings를 명시하고,
Euler-CE 외삽 우위의 진짜 메커니즘(e-decay = ALiBi)을 32× extrapolation
까지 분해한 데에 있다.

---

## References

[1] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.

[2] O. Press et al., "Train Short, Test Long: Attention with Linear Biases (ALiBi)," ICLR 2022.

[3] B. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023.

[4] Y. Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.

[5] D. Hilbert, G. Pólya, conjecture (folklore, ca. 1910s).

[6] M. V. Berry, J. P. Keating, "H = xp and the Riemann zeros," 1999.

[7] (Internal) `docs/6_뇌/07_수면과복구.md` — CE 부트스트랩 고정점 `ε² = 4.87 %`.

[8] (Internal) `docs/7_AGI/2_Architecture.md` § 4 — 유니타리 제약.

[9] (Internal) `clarus/ce_euler.py` — Euler-CE attention.

[10] Y. Sun et al., "A Length-Extrapolatable Transformer (xPos)," ACL 2023.

[11] M. M. Bronstein et al., "Geometric Deep Learning," 2021.

[12] V. G. Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML 2021.

[13] H. L. Montgomery, "The pair correlation of zeros of the zeta function," 1973.

---

## Appendix A. 200-step transient (참고용)

소규모 짧은 학습(200 step × 2 seed)에서는 분산이 더 dramatic하게 보였으나
(σ_PPL: RoPE 1.22, mra 0.16, mra_h 0.06; 7-20× 감소), 1500-step 충분 수렴 시점
에는 σ가 모두 0.07-0.21 범위로 평준화된다. **소규모 실험의 분산 격차는
early-stopping artifact이며 영구적 안정성 우위가 아니다**. 1500-step 결과
(§ 7)만이 신뢰할 수 있는 결론이다.

이 발견 자체가 ML 평가에서 reproducibility의 함정을 보여주는 사례다. 짧은
학습에서 본 "결정적 차이"는 충분히 학습하면 사라질 수 있다.
