# Mellin-Riemann Attention: ζ explicit-formula 영감 feature의 ablation 분석

> 본 문서는 `mra_block_spec.md`의 사양과 `tests/test_mra.py`의 연산 검증을
> 논문 형태로 정리한다. 과거 인용된 `bench_recursive_euler.py`,
> `bench_mra_extrap.py`, 결과 JSON/checkpoint/config는 현재 checkout에 없다.
> 따라서 6--7절의 수치는 `recorded historical result / non-evidence`이며,
> 재현된 실측 결론으로 사용하지 않는다.
>
> **수식 교정 (2026-08-06).** 기록된 실험은 3절의 directed score와 표준
> row-softmax를 사용했다. 이는 self-adjoint attention이 아니다. 교정된
> bidirectional Hermitian construction은 `mra_block_spec.md`의
> \(H=(D+D^\dagger)/2\) 및 symmetric normalization이며 아직 이 표의
> benchmark 대상이 아니다. 따라서 아래 실험을 Hilbert–Pólya, RH 또는 GUE의
> 검증으로 읽지 않는다.

## Abstract

Riemann ζ 함수 explicit formula의 유한 positive-zero 항에서 영감을 받은
attention 변종 **Mellin-Riemann Attention (MRA)**와 7개 설계 knob를 제안한다.
유한 truncation과 attention feature 삽입은 explicit formula 자체도,
Hilbert--Pólya/RH 검증도 아닌 공학 설계다.

현재 재현 가능한 length-OOD toy는 MRA가 아니라 tracked
`../experiments/ood_length_repro.py`의 네 canonical head뿐이다. 약 127 K
parameter, 3 seed, train 64/eval 2048에서 ALiBi와 xPos는 각각
$-9.1\%\pm5.7\%$, $-8.8\%\pm5.4\%$, NoPE와 RoPE는
$+31.7\%\pm10.2\%$, $+505\%\pm41.4\%$ degradation을 기록했다. 이 toy는
강한 distance attenuation의 방향성 근거지만 MRA, scaling, 자연어, AGI-OOD의
결론이 아니다. MRA의 성능은 7.8절 acceptance protocol을 새 artifact로
통과하기 전까지 `open`이다.

---

## 1. Introduction

Transformer의 positional encoding은 attention 메커니즘이 set-permutation
invariant라는 근본 한계를 외부 주입으로 해결하는 inductive bias의 핵심
지점이다. RoPE [1]가 LLaMA / Qwen / Mistral / Gemma / EXAONE 등 거의 모든
현대 LLM의 표준이 된 후, ALiBi [2], YaRN [3], LongRoPE [4] 등 후속 연구는
대체로 경험적 튜닝으로 long-context extrapolation을 공략해왔다.

본 연구의 출발점은 다른 방향이다. **수론적 frequency bank가 attention의
inductive bias로 유용한가?** Hilbert–Pólya는 Riemann ζ의 비자명 영점
ordinate를 어떤 self-adjoint operator의 spectrum으로 실현하려는 추측이다.
Berry–Keating의 \(H=(xp+px)/2\)는 후보적 프로그램이지 완성된 연산자가
아니다.

본 연구는 유한하게 검증된 positive-zero ordinate를 deterministic buffer로
사용한다. 이 유한 사용에는 전체 RH가 필요하지 않으며, 해당 ordinate를
attention frequency로 쓰는 것 역시 Hilbert–Pólya operator의 구성이 아니다.
식은 단순하지만 어느 부분이 실제로 도움이 되고 어느 부분이
도움이 되지 않는지는 **실증으로만 확인 가능**하다. 본 논문의 기여는 이를
ablation으로 분리한 데에 있다.

### 1.1 Contributions

1. ζ explicit formula의 유한 positive-zero 항을 directed attention
   feature로 옮긴 convention을 고정 (§ 3).
2. 7가지 설계 knob(frequency 방식, amplitude weighting, decay 형태,
   sparsity, spectral norm, Hermitian)을 노출하는 단일 ablation 표면 (§ 4).
3. 과거 1500-step/length 표를 재현 artifact 부재의 historical ledger로
   보존하고 경험적 결론과 분리 (§6--7).
4. MRA와 canonical PE를 재검증할 artifact schema와 acceptance protocol을
   명시 (§7.8).

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

explicit formula의 zero 합은 모든 영점과 적절한 대칭 truncation을 포함한다.
RH를 추가로 가정하고 conjugate pair를 묶으면 양의 ordinate에 대해

```
Σ_ρ x^ρ/ρ
= 2 Re Σ_{γ_n>0} [√x · e^{iγ_n log x}/(1/2+iγ_n)]
```

형태다. positive-\(\gamma\) 복소합 하나만은 full real zero contribution과
같지 않다. MRA는 이 유한 positive-zero 항의 **Mellin phase**와 **amplitude**
를 feature로 차용한다. finite truncation, token-pair \(x_{ij}\), query/key
곱은 explicit formula에서 정해지는 것이 아니라 공학적 삽입이다.

---

## 3. Method - Mellin-Riemann Attention

### 3.1 단순 score 식

모든 관련 문서와 같은 convention으로

```
τ_i = log(1+i),  Δ_ij = τ_i-τ_j,
x_ij = exp(-Δ_ij) = (1+j)/(1+i)
```

를 둔다. 기록된 backend는 raw zero가 아니라

```
ν_k = γ_k/γ_0,  a_k = 1/(1/2+iν_k)
```

를 사용한다. 따라서 \(a_k\)는 zeta amplitude가 아니라 normalized design
weight다. dim-pair `k`를 복소채널로 압축하면

```
q_i^(k) := q_i^{2k} + i · q_i^{2k+1},     k_j^(k) := k_j^{2k} + i · k_j^{2k+1},
```

`freq_mode="zeta_log"`, `amp_weight=True`, `decay_mode="mult"`를 모두 켠
**full directed reference score**는 다음과 같다.

```
S_{ij} = √((1+j)/(1+i)) · Σ_k a_k · e^{−iν_k log((1+i)/(1+j))} · q_i^(k) · conj(k_j^(k))
```

토큰별 attention 입력은 `Re(S_{ij})`.
prefactor와 complex \(a_k\) 때문에 tied query/key를 사용해도 일반적으로
\(S_{ji}\ne\overline{S_{ij}}\)다.

기록된 primary `mra`/`mra_bias`는 `freq_mode="rope"`이므로 위
\(e^{-i\nu_k\Delta_{ij}}\) 대신 표준 RoPE phase를 사용한다. 또한
`decay_mode="none"`이면 \(\sqrt{x_{ij}}\)를 생략한다. 따라서 위 한 식을
모든 ablation variant의 실제 forward라고 읽지 않는다. `mra_zeta`가
normalized zero-frequency 분기를 격리한다.

### 3.2 full directed reference의 폐쇄형 분해

`(1+i)^{−iν_k}`와 `(1+j)^{−iν_k}`가 각각 `i`, `j`만의 함수이므로 사전 변환

```
q̃_i^(k) = (1/√(1+i)) · e^{−iν_k log(1+i)} · q_i^(k)
k̃_j^(k) = √(1+j) · e^{−iν_k log(1+j)} · k_j^(k)
```

후

```
S_{ij} = Σ_k a_k · q̃_i^(k) · conj(k̃_j^(k))
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
| `ν_k=γ_k/γ_0` | buffer | 0 (verified table + engineering normalization) |
| `a_k = 1/(1/2 + iν_k)` | buffer | 0 (normalized design weight) |
| `cos_p, sin_p, log_decay` | buffer | 0 (deterministic design buffer) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention 동일 |

표준 multi-head attention 대비 **추가 학습 파라미터 0**.

---

## 4. 설계 Knob - Ablation 표면

`MellinRiemannAttention(d_model, n_heads, block, **knobs)`가 노출하는 7개
knob은 다음과 같다.

| knob | 값 | 기본 | 의미 |
|---|---|---|---|
| `freq_mode` | `"rope"` / `"zeta_log"` | `"rope"` | 위치 주파수 형태 |
| `amp_weight` | `True` / `False` | `True` | normalized complex weight \(a_k\) 사용 |
| `decay_mode` | `"none"` / `"bias"` / `"mult"` | `"none"` | critical-line decay 형태 |
| `sparse_eps2` | `0.0` ∼ `0.0487` | `0.0` | 당시 ablation의 top-k retention 비율; 정준 target은 `0.04863825851598632` |
| `hermitian` | `False` / `True` | `False` | tied projection + pre-mask real-score symmetrize; operator Hermiticity는 별도 |
| `spectral_norm_o` | `False` / `True` | `False` | `σ₁(W_o) ≤ 1` 강제 |
| `rope_base` | f32 | `10000` | RoPE base freq (freq_mode="rope" 일 때만) |

### 4.1 `freq_mode`

- `"rope"`: `θ_k(p) = p · base^{−k/K}`. RoPE 기하 급수(~3000× 범위).
- `"zeta_log"`: `θ_k(p) = ν_k · log(1+p)`. zero ordinate의
  정규화 비가 frequency다. \(ν_k\) 범위가 K=16에서 약 5배라 다중 스케일
  해상도가 손실된다.

### 4.2 `amp_weight`

- `True`: `a_k = 1/(1/2 + iν_k)`를 score에 곱한다.
  `Re(a_k)·Re_part − Im(a_k)·Im_part` 형태로 standard RoPE의
  `cos` 채널과 보조 `sin` 채널이
  혼합된다.
- `False`: `a_k = 1`. attention 식이 표준 RoPE와 일치한다(control).

### 4.3 `decay_mode`

positive-zero feature의 `√x` factor
(`x = (1+j)/(1+i)`)를 처리한다.

- `"none"`: 무시.
- `"bias"`: log-space additive. `bias_{ij} = ½(log(1+j) − log(1+i))`를 score에 더한다.
  ALiBi의 logarithmic 일반화다.
- `"mult"`: 원식 그대로 `score_{ij} *= √((1+j)/(1+i))`를 적용한다. Per-key temperature
  처럼 작동하므로 의도와 다소 어긋난다.

### 4.4 `hermitian` (causal LM에서 비추천)

`W_q = W_k`만으로는 directed score가 Hermitian이 아니다. 실수 logits
\(L=\operatorname{Re}S\)를

```
H_R = (L + L^T)/2
```

로 사영하면 \(H_R^T=H_R\)는 exact다. 그러나 표준 row-softmax
\(P=D^{-1}\exp(H_R)\)는 일반적으로 \(P^T\ne P\)다. ordinary Euclidean
inner product에서 self-adjoint operator가 필요하면

```
K_ij = exp(H_R,ij),  d_i = Σ_j K_ij,
A_sym = diag(d)^{-1/2} K diag(d)^{-1/2}
```

를 써야 한다. 현재 구현은 symmetric normalization까지 구현하지 않으므로
`hermitian=True`는 “pre-mask real score symmetrization”만 뜻한다.
Hilbert–Pólya 직접 구현이라고 부르지 않는다.

strict causal mask는 비자명한 symmetric operator와 양립하지 않는다.
causal decoder에서는 directed score만 쓰고 Hermitian claim을 하지 않는다.
이 knob은 기록된 causal ablation 표에서 제외했다.

---

## 5. Implementation

`reality_stone/python/reality_stone/clarus/ce_mra.py`는 PyTorch 참조 구현(~250 LOC)이다. 현재 backend는 PyTorch only이며,
`RiemannRotaryAttention`의 Rust/CUDA 패턴을 그대로 이식할 수 있다.

테스트 (`tests/test_mra.py`, 11/11 pass):

- forward shape & finiteness
- block end-to-end
- 모든 ablation knob 조합 finite output
- bootstrap sparsity 정확한 retention 수
- 학습 파라미터 수 == 표준 MHA (4·d²) / Hermitian 모드 == 3·d²
- 정규화 complex weight가 실제로 출력을 바꾸는지 확인(회귀 가드)
- 공리적 buffer들이 학습되지 않는지 확인
- spectral norm이 σ₁ ≤ 1을 강제하는지 확인

---

## 6. Experiments - Setup

> **Historical protocol only.** 아래 설정을 실행한 script, corpus snapshot,
> split hash, checkpoint, raw result가 현재 checkout에 없다. 6--7.7.3의 표는
> 과거 기록을 보존한 evidence ledger이며 통계적 판정이나 성능 주장의 근거가
> 아니다. 현재 실행 가능한 canonical-head toy와 새 acceptance protocol은
> 7.8절에 분리한다.

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
| `mra_noamp` | rope | False | none | ablate normalized complex weight |
| `mra_zeta` | zeta_log | True | none | ablate frequency |
| `mra_bias` | rope | True | bias | + additive log decay |
| `mra_mult` | rope | True | mult | + multiplicative decay |
| `mra_sparse` | rope | True | none | + 당시 반올림값 ε² = 0.0487 sparsity |
| `mra_sn` | rope | True | none | + spectral_norm(W_o) |

---

## 7. Historical result ledger (현재 non-evidence)

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

### 7.2 과거 verdict label (`std_rope`, `euler_ce_k1` 대비; 현재 무효)

| 비교 | z | Δ PPL | 판정 |
|---|---|---|---|
| `mra_zeta` vs `std_rope` | +8.84 | +1.999 | archived LOSS label; current unverified |
| `mra` vs `std_rope` | +1.35 | +0.170 | archived LOSS/variance label; current unverified |
| `mra_noamp` vs `std_rope` | −0.16 | −0.022 | archived TIE label; current unverified |
| `mra_bias` vs `std_rope` | −0.36 | −0.048 | archived TIE+ label; current unverified |
| `mra_mult` vs `std_rope` | −0.22 | −0.028 | archived TIE+ label; current unverified |
| `mra_sparse` vs `std_rope` | +1.86 | +0.334 | archived LOSS label; current unverified |
| `mra_sn` vs `std_rope` | +2.14 | +0.334 | archived LOSS label; current unverified |
| `euler_ce_k1` vs `std_rope` | (historical baseline) | −0.424 | archived WIN label; current unverified |

### 7.3 과거 부정적 기록 (재검증 필요)

1. **`mra_zeta` (PPL 14.29)**: 정규화한 zero ordinate를 frequency로 사용하면 RoPE의
   다중 스케일 해상도가 깨진다. `γ_k/γ_1`의 범위는 K=16에서 ~5×에 불과하다.
   RoPE의 `10000^{−k/K}`가 갖는 ~3000× 범위와 비교하면 다중 스케일 해상도가
   거의 없다는 구조적 차이는 계산 가능하다. 다만 이것이 과거 PPL 차이의
   직접 원인인지는 artifact와 개입 ablation이 없어 현재 미판정이다.
2. **정규화 complex weight 단독 효과 미미**: `mra` (12.46) − `mra_noamp` (12.27) =
   +0.17 PPL로 기록됐다. raw seed 결과가 없어 정보 이득의 부재 또는 손해를
   현재 통계적으로 판정하지 않는다.
3. **부트스트랩 sparsity / spectral norm**: 1500 step 충분 수렴 시점에서는 모두
   PPL +0.33으로 기록됐다. long-run over-regularization이라는 인과 해석은
   재현 전에는 사용하지 않는다.

### 7.4 과거 seed 분산 기록 (재검증 필요)

아래 표는 과거 요약값이다. 3 seed의 표본 표준편차만으로 분산 감소를
확정할 수 없고 raw run도 없으므로 재현 전에는 방향성 기록으로만 남긴다.

```
σ_PPL across 3 seeds (lower = more reproducible):
  std_rope     : 0.208         baseline
  mra          : 0.069         3.0×  more stable
  mra_mult     : 0.077         2.7×
  mra_bias     : 0.097         2.1×
  mra_noamp    : 0.129         1.6×
  euler_ce_k1  : 0.126         1.7×  (다른 메커니즘)
```

이 기록에서 보인 비율의 원인을 normalized weight 또는 분리된 $W_q,W_k$에
귀속하지 않는다. 최소 10 seed와 Levene/Brown--Forsythe 또는 bootstrap
variance-ratio CI가 필요하다.

### 7.5 과거 lean 후보

`decay_mode="bias"` (`mra_bias`):
- mean PPL 12.240: **RoPE보다 0.05 낮음**(z = −0.36, statistically TIE
  with mean preference)
- σ_PPL 0.097: RoPE 분산의 47 %
- 시간 비용은 RoPE와 동일

이는 ALiBi와 다른 logarithmic bias 후보로 쓸 수 있다.
```
bias_{ij} = ½ (log(1+j) − log(1+i))
```
선형 거리 `−|i−j|`가 아닌 logarithmic 거리다. critical line `Re(s) = 1/2`의
지수 `½`를 critical-line 실수부에서 가져온 설계 선택이다. 이 선택이 성능을
유도하거나 유일하게 만든다는 뜻은 아니다.

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

과거 표의 단순 산술에서는 `euler_ce_k1`과 `mra_zeta`의 상대 degradation이
작고 `mra_bias`가 RoPE보다 8.2 percentage points 작았다. 그러나 raw artifact가
없으므로 우월성, RoPE 붕괴 재현, logarithmic bias의 효과라는 결론은 모두
철회한다. 동일 설정의 새 paired runs가 7.8절을 통과해야 한다.

### 7.7.1 과거 scaling 표: d_model {64, 128, 256}

과거 기록은 다음 설정을 사용했다고 적고 있으나 artifact가 없어 scaling
검증으로 인정하지 않는다. 학습 설정은
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

#### 현재 판정

이 표에서 보이는 tier와 $d_{\rm model}$ 패턴은 historical hypothesis다.
d=128/256가 2 seed뿐이고 raw run·학습곡선이 없으므로 모델 크기 불변성,
underfit, production-scale ALiBi, NoPE 부적합 중 어느 것도 결론내리지 않는다.
현재 tracked toy는 오히려 NoPE를 안정 tier로 재현하지 못했다.

### 7.7.2 과거 32× Euler-CE 분해 표

과거 실험은 EulerCE의 두 구조를 frozen ablation으로 비교했다고 기록한다.
artifact 부재 때문에 원인 격리로 인정하지 않는다.

| 변종 | 구성 | 학습 자유도 |
|---|---|---|
| `std_rope` | RoPE | baseline |
| `rope_alibi` | RoPE + ALiBi linear decay (per-head learnable slope) | + n_heads |
| `mra` / `mra_bias` | RoPE + normalized complex weight / + log decay | + n_heads (`mra_bias`) |
| `euler_no_decay` | EulerCE with `e_gate` frozen → π-rotation 만 | (gate 1개 frozen) |
| `euler_no_pi` | EulerCE with `pi_gate` frozen → e-decay 만 | (gate 1개 frozen) |
| `euler_ce_k1` | full Euler-CE | original |

train block = 64, eval block ∈ {64, 128, 256, 512, 1024, 2048} (최대 32×).
모든 모델은 train_block으로 init하고, eval 직전 `extend_to(N_eval)`로 buffer만
grow한다. EulerCE의 block-aware base `π^D_eff · N`도 train 시점 N=64로
계산.

1500 step × 3 seed × batch=4로 측정했다고 기록돼 있다. checkpoint와 동일
evaluation sample이 없으므로 상대 degradation의 공정 비교도 현재 확인할 수 없다.

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

#### 과거 tier 분류 (현재 철회)

| Tier | 기준 | 메커니즘 |
|---|---|---|
| **1 (외삽 OK, ≤ +10 %)** | rope_alibi, euler_no_pi, euler_ce_k1, xpos, nope | distance attenuation **존재** OR rotation **부재** |
| **2 (외삽 BAD, +29 ∼ +55 %)** | mra, mra_bias, std_rope, euler_no_decay | rotation **존재** + 강한 distance attenuation **부재** |

이 표로부터 과거에 내렸던 "e-decay 100% 단독 효과", "NoPE 안전",
"rotation이 직접 원인", "ALiBi와 정확한 환원" 결론은 철회한다. 현재 tracked
toy는 ALiBi/xPos의 안정 방향과 RoPE의 큰 악화를 재현했지만 NoPE도
$+31.7\%$ 악화됐다. MRA와 log bias는 그 toy에 포함되지 않았다. 따라서
현재 허용되는 가설은 **강한 distance attenuation이 이 tiny length task에서
도움이 될 수 있다**까지다.

### 7.7.3 2-bit Minimal Euler-CE: 구현 taxonomy

Euler 5상수 `{e, π, i, 1, 0}`에서 가져온 두 구현 axis를 다음처럼 둔다.
이는 성능에서 연역된 정보이론적 환원이 아니라 코드 taxonomy다.

| 상수 | 작용 | 환원 |
|---|---|---|
| `π`, `i` | rotation generator (`e^{iπt}` 결합) | **axis 1**: rotation |
| `e` | exponential decay base | **axis 2**: strong distance attenuation |
| `1`, `0` | on/off gate values | 1 비트 each |

→ **2 functionally distinct axes × 2 gate values = 2² = 4 head-types**, 각각
2-bit string `(pi, e)`으로 인코딩할 수 있다.

| (pi, e) | 비트 | 헤드 타입 | 문헌 매핑 |
|---|---|---|---|
| (0, 0) | `00` | identity | NoPE [Kazemnejad 2023] |
| (0, 1) | `01` | decay only | ALiBi [Press 2022] |
| (1, 0) | `10` | rotation only | RoPE [Su 2021] |
| (1, 1) | `11` | rotation + decay | xPos [Sun 2023] / Euler-CE |

strong attenuation bit는 additive logit $b(d)$에 대해
$\liminf_{d\to\infty}-b(d)/d>0$일 때 1로 정의한다. 그러면 linear ALiBi와
$0<\zeta<1$인 multiplicative $\zeta^d$는 1, $-c\log d$와 distance에 따라
충분히 줄지 않는 MRA amplitude는 0이다. 이 threshold 없이 "attenuation
있음"이라 부르면 MRA/MRA-bias의 과거 Tier-2 표와 모순된다.

4개 상태의 fixed binary register는 2 bit가 필요하다. $\log_2 3\simeq1.58$
bit는 세 상태의 균등 Shannon entropy/평균 code-length 하한이지 fixed code
길이가 아니다. 현재 toy에서 NoPE도 악화됐으므로 3/4 상태가 작동한다는
perfect mapping은 철회한다.

#### 구현: `EulerCEMinimal`

`reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEMinimal`가 본 환원의 정밀 구현이다. 각 헤드는 2비트
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

표의 parameterization에서는 연속 gate를 고정해 학습 scalar를 줄인다. 이는
더 작은 hypothesis class라는 구현 선택이지, 동일 표현력이나 일반화를
보존한다는 증명이 아니다.

#### 연산 검증 — `tests/test_euler_minimal.py`

* `head_types_from_spec`가 6가지 spec 형태(int, str, list, "mix", "all",
  invalid) 모두 정확히 처리.
* `head_types="nope"` → no-PE와 수치적으로 일치(allclose atol=1e-5).
* `head_types="alibi"` → 거리에 따른 attention attenuation 확인.
* `head_types="rope"` → 모든 헤드 e_bit=0, decay 항 0.
* `head_types="mix"` / `"all"` → 정확한 비트 분해.
* `extend_to(N)` 후 forward 정상, 학습 파라미터 변하지 않음.
* Autograd: rotation-only 헤드의 `log_xi` grad는 정확히 0(decay gate off 시
  grad path가 끊김).

이 테스트는 head 연산과 gate 경로의 코드 회귀를 검사한다. OOD 성능이나
5상수의 정보이론적 환원을 검증하지 않는다.

#### 과거 minimal 성능표 (현재 non-evidence)

missing `bench_mra_extrap.py --variants minimal`의 과거 기록:

| 2-bit variant | head_types | x32 degrad | canonical 매핑 | x32 (canonical) |
|---|---|---|---|---|
| `min_nope` | uniform 00 | **+7.0 %** | `nope` | +7.0 % (정확 일치) |
| `min_rope` | uniform 10 | **+47.2 %** | `std_rope` | +47.2 % (정확 일치) |
| `min_alibi` | uniform 01 | **−5.7 %** | `rope_alibi` | −6.2 % (≈ 일치) |
| `min_xpos` | uniform 11 | **−5.8 %** | `euler_ce_k1` | −6.2 % (≈ 일치) |
| `min_mix` | half 01, half 11 | −5.5 % | (신규) | — |
| `min_all` | one of each 00/01/10/11 | −0.6 % | (신규) | — |

raw artifact가 없으므로 $\le0.5$ percentage-point 재현, mix/all 비교,
단일 head-type 권고를 현재 결과로 쓰지 않는다. 네 canonical 연산을 하나의
2-bit API로 표현할 수 있다는 코드 사실만 남는다.

### 7.8 현재 toy와 MRA acceptance protocol

현재 tracked `../experiments/ood_length_repro.py`는 약 127 K parameter,
3 seed, 500 step, repository Python char corpus에서 네 canonical head만
비교한다.

| head | 32× degradation mean ± sample std | 범위 |
|---|---:|---|
| ALiBi | $-9.1\%\pm5.7\%$ | current toy |
| xPos | $-8.8\%\pm5.4\%$ | current toy |
| NoPE | $+31.7\%\pm10.2\%$ | current toy; 과거 Tier-1 반증 |
| RoPE | $+505\%\pm41.4\%$ | current toy |

MRA를 경험적으로 판정하려면 `std_rope`, ALiBi, xPos, NoPE, `mra`,
`mra_noamp`, `mra_zeta`, `mra_bias`를 같은 tokenizer/corpus split SHA256,
parameter count $\pm0.1\%$, train tokens, optimizer schedule로 비교한다. 최소
10 seed와 두 corpus, length $\{64,128,256,512,1024,2048\}$를 사용하고
각 run의 config/commit/environment hash, raw PPL, wall-clock을 JSONL로 남긴다.
사전등록한 paired bootstrap 또는 mixed-effects CI가 unseen corpus에서도
재현될 때만 평균/분산/length claim을 승격한다.

---

## 8. ML 이론 함의: 정직한 평가

### 8.1 무엇이 작동했나

* **코드 수준**: directed MRA와 교정된 Hermitian 후보의 convention 및 unit
  tests가 명시됐다.
* **현재 canonical-head toy**: ALiBi/xPos가 NoPE/RoPE보다 안정 방향을 보였다.
  MRA의 logarithmic bias와 seed 분산은 현재 artifact가 없어 미판정이다.

### 8.2 무엇이 작동하지 않았나

* **정규화 zero ordinate frequency, complex weighting, sparsity, spectral
  norm**: 과거 표에는 negative/tie 값이 기록돼 있지만 현재 재현되지 않았다.
  성능 방향은 open이다.
* **강한 수론 해석**: 유한 zero feature는 RH/Hilbert--Pólya 증거가 아니며,
  성능이 재현되더라도 수론적 동일성을 뜻하지 않는다.

### 8.3 어떤 가설이 다음 단계에서 검증 가능한가

* **Length extrapolation**: 짧은 context 학습 후 긴 context에서의 안정성.
  `mra_bias`의 logarithmic decay는 RoPE의 wrap-around 한계를 넘어설 가능성
  이 있다. 별도 측정 필요.
* **대형 모델 분산**: 분산 감소가 모델 크기와 함께 어떻게 변하는지.
* **Attention 스펙트럼**: bidirectional \(A_{\rm sym}\)의 unfolded local
  spacing을 Poisson/GOE/GUE null과 비교. GUE 적합이 나오더라도 이는 유한
  random-matrix 진단이며 RH나 Hilbert–Pólya의 증거로 해석하지 않는다.

---

## 9. Related Work

- **RoPE** [1]: 회전 기반 상대위치, 본 연구의 직접적 baseline.
- **ALiBi** [2]: 거리 감쇠 bias. `mra_bias`의 logarithmic 변종이 이를 일반화한다.
- **xPos** [10]: RoPE + exponential decay. ad-hoc 결합.
- **YaRN / LongRoPE** [3, 4]: 경험적 frequency rescaling.
- **Geometric Deep Learning** [11]: 기하학적 inductive bias 일반.
- **Equivariant Transformers** [12]: 대칭성 내장.

본 연구는 ζ 영점이라는 **수론** 객체를 attention feature로 옮기는 ablation
설계를 명시한다. MRA의 과거 mixed 결과는 artifact 부재로 현재 미판정이다.

---

## 10. Limitations & Future Work

### 10.1 한계

1. **MRA artifact 부재**: 과거 210 K/400 K-char 표의 script, raw result,
   checkpoint가 없다. 7.8절의 10+ seed protocol부터 재실행해야 한다.
2. **Char-level only**: BPE / sentencepiece 토큰화에서의 거동 미검증.
3. **Length extrapolation 범위 제한**: 현재 tracked toy는 canonical 4-head,
   tiny char-LM, single corpus뿐이며 MRA/scaling을 검증하지 않는다.
4. **Native 백엔드 부재**: PyTorch-only, wall-clock 비교 손해. Rust/CUDA 포팅
   가능.
5. **Self-adjoint backend 미검증**: 기록된 causal 실험은 standard
   row-softmax를 사용했다. bidirectional \(A_{\rm sym}\)의 성능과 spectrum은
   본 작업 범위 밖이다.

### 10.2 후속

1. Length extrapolation: train N=64, eval N={128, 256, 512}에서
   `mra_bias`의 logarithmic decay가 RoPE의 periodic wrap보다 실제로 잘
   버티는지.
2. Scale law: `d_model ∈ {128, 256, 512}` × `n_layers ∈ {4, 8, 12}`에서
   분산 감소 효과의 안정성.
3. Attention spectrum: 학습된 attention의 spectral statistics 측정.
4. Bidirectional MRA: \(H=(D+D^\dagger)/2\),
   \(K_{ij}=\exp(\operatorname{Re}H_{ij})\),
   \(A_{\rm sym}=D_d^{-1/2}KD_d^{-1/2}\)를 구현한 뒤
   BERT-style 인코더에서 검증.
5. GaugeLattice FFN 결합: AGI 5대 원리 중 채널 분할까지 묶은 full ClarusBlock.

---

## 11. Conclusion

현재 닫힌 결론은 세 가지다.

1. MRA는 ζ explicit-formula에서 영감을 받은 **공학 feature**이며 RH,
   Hilbert--Pólya, GUE의 검증이 아니다.
2. MRA의 평균 PPL, seed 분산, 4×/32× extrapolation에 관한 과거 표는 생성
   artifact 부재로 non-evidence다. 성공/실패 방향을 현재 결론내리지 않는다.
3. tracked canonical-head tiny toy에서는 강한 distance attenuation을 가진
   ALiBi/xPos가 안정 방향을 보였지만 NoPE도 악화됐다. 이 결과는 MRA,
   production scaling, OOD 일반의 정리가 아니며 7.8절 protocol이 다음 gate다.

---

## References

[1] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.

[2] O. Press et al., "Train Short, Test Long: Attention with Linear Biases (ALiBi)," ICLR 2022.

[3] B. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023.

[4] Y. Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.

[5] D. Hilbert, G. Pólya, conjecture (folklore, ca. 1910s).

[6] M. V. Berry, J. P. Keating, "H = xp and the Riemann zeros," 1999.

[7] (Internal) `docs/6_뇌/07_수면과복구.md` — CE 부트스트랩 고정점; 현재 정준값 `ε² = 0.04863825851598632`.

[8] (Internal) `docs/7_AGI/2_Architecture.md` § 4 — 유니타리 제약.

[9] (Internal) `reality_stone/python/reality_stone/clarus/ce_euler.py` — Euler-CE attention.

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
