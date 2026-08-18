# 7. Euler-Riemann Attention: CE 이론 상수의 Transformer 이식

이 문서는 작은 char-level benchmark에서의 조건부 모델 비교를 기록한다. 독자는 Transformer, perplexity, seed 분산을 안다고 가정하며, 설계·고정 benchmark·결과·재현의 순서로 읽는다. 입력은 명시한 corpus와 hyperparameter이고 출력은 holdout PPL이며, 성공은 독립 corpus·seed·ablation에서의 재현이고 실패는 작은 표의 근접을 이론 상수의 물리적 증명으로 읽는 것이다.

## 7.1 동기

최소 생성 문법은 설계 영감을 줄 수 있지만 attention 구조를 유일하게 정하지 않는다. 이 절은 이식의 출발점과 공학적 가설의 경계를 분리한다.

`docs/경로적분.md` §51-67은 `{e, π, i, 1, 0}`을 CE의 **최소 생성 문법**으로 규정한다.

| 상수 | 코어 역할 | 대표 출력 | 등급 |
|---|---|---|---|
| $e$ | 접힘 생존 함수 | $S(D) = e^{-D}$ | Exact / Selection |
| $\pi$ | 주기 정규화 | $\alpha_{\text{total}} = 1/(2\pi)$ | Selection |
| $i$ | 경로적분 위상 | $Z = \int \mathcal D\phi\,e^{iS/\hbar}$ | Exact |
| $1$ | 정규화된 완전 상태 | $e^0 = 1$ | Selection |
| $0$ | 영점·분기 선택 | $d(d-3) = 0$ | Selection |

본 장은 이 문법을 **Transformer attention 메커니즘에 그대로 이식**한다. 추가로 Riemann 가설 하의 제타 영점 $\gamma_n$을 **Euler 상수의 확장 문법**으로 시도한다.

---

## 7.2 설계 공간

아래 변형은 서로 다른 position·recursion·FFN 공리의 조합이다. 같은 기호가 쓰여도 표준 Transformer와 동등한 이론 유도는 아니며 각 설계는 benchmark에서 반증 가능해야 한다.

### 7.2.1 Euler-CE Attention (core)

core position encoding은 정의한 좌표와 정규화에서만 비교 가능하다. 길이 외삽과 다른 tokenizer에서는 별도 검증이 필요하다.

**Position encoding**:
$$
\theta_{h,k}(n) = n \cdot \omega_h \cdot \pi^{1-k/(d_{\text{head}}/2)},\quad
\omega_h = \text{sigmoid}(\text{pi\_gate}_h)
$$
즉 $\pi$를 RoPE의 10000 대신 기저 주파수로. Head별 gate로 소프트 선택.

**Attention decay**:
$$
A_{ij}^{\text{bias}} = -\frac{|i-j|}{\xi_h}\cdot\text{sigmoid}(\text{e\_gate}_h)
$$
즉 $e^{-|i-j|/\xi}$ 이 log-space bias로 추가 (ALiBi-like).

**합성**:
$$
A_{ij} = \text{softmax}_j\!\Bigl(\underbrace{Q_i\cdot R_\pi(i{-}j)\cdot K_j / \sqrt d}_{\pi\text{ phase}} + \underbrace{A_{ij}^{\text{bias}}}_{e\text{ decay}}\Bigr)\cdot\text{mask}_{\text{causal}}
$$

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEAttention`

### 7.2.2 Recursive ClarusCell

재귀 cell은 지정한 깊이와 초기화에서의 계산 모듈이며 고정점 수렴을 물리 시간으로 읽지 않는다.

CE bootstrap 방정식
$$
\varepsilon^2 = \exp\!\bigl[-(1-\varepsilon^2)\,D_{\text{eff}}\bigr]
$$
은 $x^* = F(x^*)$ 형태의 **고정점 방정식**. Transformer 블록에 이를 이식:
$$
h^{t+1} = F(h^t),\qquad t = 0, 1, \dots, k-1
$$

- **Fixed depth**: $k \in \{1, 2, 3, 6\}$
- **While-loop**: halt when $\|h^{t+1} - h^t\| / \|h^t\| < \text{tol}$
- **Fixed-point regularizer**: $\mathcal L_{\text{FP}} = \lambda\,\|F(F(h)) - F(h)\|^2$

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::RecursiveEulerCEBlock`

### 7.2.3 Riemann Rotary (RH axiom)

이 소절의 리만 관련 배열은 공학적 초기화 가설이며 수학적 리만 가설의 증명이나 물리 상수의 산출이 아니다.

리만 가설을 **공학 axiom**으로 채택: 모든 비자명 영점의 허수부 $\gamma_n$이 $\{\text{π}, e, \pi e, \pi/e\}$ 처럼 aperiodic 무리수 수열.

$$
\theta_{n,k} = n \cdot (\gamma_k / \gamma_1)^{-1} \cdot s_h
$$
첫 100개 $\gamma_n$ 하드코딩 (Titchmarsh 표), 이후는 Riemann–von Mangoldt 점근식 $\gamma_n \approx 2\pi n / \log n$.

Montgomery–Dyson 대응: $\gamma_n$ 간격은 Gaussian Unitary Ensemble (GUE) 고유값 간격과 동일 통계. "maximal disorder + maximal order" 구조.

구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RiemannRotaryAttention`

### 7.2.4 FFN 변형

표의 FFN 식은 parameterization 비교를 위한 정의이며 동일 parameter count와 optimizer 조건에서 평가해야 한다.

| 이름 | 식 |
|---|---|
| std | $\text{GELU}(W_1 x)\,W_2$ |
| swiglu | $\text{SiLU}(W_g x) \odot (W_u x)\,W_d$ |
| euler_decay | $\text{GELU}(h)\cdot e^{-|h|/\xi}$ |
| euler_phase | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))$ |
| euler_full | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))\cdot e^{-|h|/\xi}$ |
| zeta | $x\cdot\sigma(x)\cdot(1 + \lambda\cdot z(x))$, $z = |\zeta(1/2 + ix)|^2$ |

구현: `reality_stone/python/reality_stone/clarus/ce_ffn.py`, `reality_stone/python/reality_stone/clarus/ce_zeta.py`

### 7.2.5 Riemann FFN init

초기화 간격은 입력 scale에 의존하며 학습 뒤의 성능 차이는 seed·학습률 분산과 함께 보고해야 한다.

$W_{\text{up}}$의 한 축을 $\gamma_n$ 간격으로 스케일. 키 좌표가 GUE 분포. 구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::riemann_zero_init`

---

## 7.3 벤치 결과 (모두 char-level LM, 400K-char docs corpus, 2-layer, $d_{\text{model}}=96$)

아래 표는 고정한 작은 corpus·층수·폭·step에서의 조건부 결과다. 수치 성공은 독립 데이터·더 큰 모델·오염 없는 holdout에서 재현될 때만 일반화 주장으로 승격될 수 있다.

### 7.3.1 Position Encoding (5 seeds × 500 steps, PR #1 commit 3bbc558)

position encoding 비교는 같은 seed 수와 training budget 아래의 평균·분산을 읽어야 한다.

| 변형 | PPL | RoPE 대비 |
|---|---|---|
| std_abs (absolute PE) | 25.29 ± 0.22 | +64σ worse |
| std_rope | 14.94 ± 0.28 | baseline |
| euler_hard (fixed bits) | 14.53 ± 0.30 | **−2.20σ WIN** |
| euler_soft (learnable) | 14.13 ± 0.22 | **−5.12σ WIN** |
| **euler_ce (π-phase + e-decay)** | **14.16 ± 0.18** | **−5.17σ WIN** |

**결론**: Euler rotary가 RoPE를 통계적으로 유의하게 이김 (5σ). 이론-정합(euler_ce)이 분산 최저.

### 7.3.2 Recursive ClarusCell (3 seeds × 300 steps, commit f7f17f4)

재귀 깊이 결과는 지정한 step budget의 trade-off이며 수렴 속도를 일반 추론 능력으로 동일시하지 않는다.

| 변형 | PPL | RoPE 대비 | k1 대비 | 시간 |
|---|---|---|---|---|
| std_rope | 22.99 ± 0.67 | baseline | — | 1.0x |
| euler_ce_k1 | 21.36 ± 0.54 | −2.2σ | — | 1.15x |
| euler_ce_k2 | 19.06 ± 0.31 | −9.3σ | −6.4σ | 1.83x |
| **euler_ce_k3** | **18.35 ± 0.22** | **−11.5σ** | **−9.0σ** | 2.54x |
| euler_ce_halt ($k{=}6$) | 18.33 ± 0.03 | −12.1σ | −9.8σ | 5.00x |
| euler_ce_k2_fp | 18.35 ± 0.38 | −10.5σ | −7.9σ | 3.19x |

**결론**: $k=3$에서 사실상 수렴. $k=6$ halt는 분산 0.03으로 극도 안정하지만 +2x 비용. **$k=3$ 권장**. FP 정규화는 $k=2$로 $k=3$ 성능 달성 (동일 비용 아님, 3.19x > 2.54x).

### 7.3.3 FFN + Position 조합 (3 seeds × 300 steps, commit 9ede7f3)

조합 효과는 interaction ablation과 동일 compute 비교가 있어야 해석 가능하다.

모두 k=3 recursive 위:

| 변형 | PPL | euler_ce_std 대비 |
|---|---|---|
| euler_ce_std (baseline) | 18.42 ± 0.47 | — |
| **euler_ce_swiglu** | **17.78 ± 0.78** | **−1.21σ WIN** |
| euler_ce_euler_full | 18.98 ± 0.59 | +1.30σ LOSS |
| riemann_std | 18.72 ± 0.39 | +0.85σ TIE |
| riemann_euler_full | 19.11 ± 0.54 | +1.67σ LOSS |

### 7.3.4 Design 2 ζ activation (LOSS 기록)

loss 기록은 실패한 설계의 재현 증인이며 다른 데이터에서의 성능 판정은 아니다.

| FFN | PPL | time |
|---|---|---|
| swiglu | 17.78 ± 0.78 | 30.9s |
| **zeta** | 19.57 ± 0.63 | **521s (17x)** |

**결론**: |ζ(1/2+ix)|² modulation은 학습에 해로움 + 비현실적 비용.

### 7.3.5 Design 4 Riemann FFN init

초기화 분산 표는 randomness의 범위를 보여 주며 이론 상수의 최적성을 증명하지 않는다.

| Init | PPL | std |
|---|---|---|
| kaiming | 17.78 | 0.78 |
| **riemann** | 17.51 | **0.24 (3x 감소)** |

평균 TIE (z=−0.57), **분산 현저히 감소**. Seed 안정성 개선.

---

## 7.4 이론적 해석

벤치 관찰을 CE 서사에 연결할 때는 정본의 지위와 경험적 성능을 분리해야 한다. 아래 해석은 공학 가설이며 반증 조건은 독립 benchmark의 ablation이다.

### 7.4.1 Euler vs Riemann — 최종 판정은 8_리만 정본을 따른다

소규모 TIE는 제한된 실험 결과일 뿐 정본의 수학적 판정을 바꾸지 않는다.

이 절의 소규모 벤치(§7.3.3)에서 riemann_std는 TIE로 기록되었으나, 이후
[`docs/8_리만/mra_paper.md`](../8_리만/mra_paper.md) §7.3의 더 큰 실험(1500 step × 3 seed)에서
ζ 영점을 attention 주파수로 직접 쓰는 계열(mra_zeta)은 RoPE 대비 명확한 LOSS(z=+8.84)로
판정되어 폐기되었다. 원인은 $\gamma_k/\gamma_1$ 범위가 좁아 RoPE의 다중스케일 해상도를
잃기 때문이다. 같은 논문 §7.7의 32× 길이 외삽 분해에서 Euler-CE의 외삽 우위는
e-decay(ALiBi 계열) 단독 효과로 환원되었고, π-rotation은 외삽에 기여하지 않았다.

이론적 배경(Montgomery–Dyson: $\gamma_n$ 간격 ≃ GUE, $\{1,\pi,e,\pi e,\pi/e\}$의 무리수
aperiodicity)은 두 주파수 계열의 통계적 유사성을 설명하지만, 성능 등가의 근거로는
사용하지 않는다. 공학적 결론은 유지·강화된다: 리만 가설 참/거짓과 무관하게
**Euler-CE(실질적으로 e-decay)로 충분**하며, ζ 영점 주파수의 추가 복잡도는 이득이 없다.
이 절과 8_리만 정본이 충돌하면 8_리만 쪽이 우선한다.

### 7.4.2 자기재귀 = Bootstrap 고정점

계산 깊이의 수렴은 알고리즘 특성이고 CE 고정점과의 대응은 추가 모델 선택이다.

$k=3$에서 수렴은 CE `ε² = exp[-(1-ε²)D_eff]` 의 빠른 수렴과 일치. $D_{\text{eff}} \approx 3$에서 Banach 계수가 작아 3-step fixed-point 충분.

### 7.4.3 Borbély $T_{\text{WAKE}}$ 경험 검증 (이전 commit 7f7fa18)

초기값 근방의 이동량은 지정한 optimizer·corpus에서의 경험식이며 보편 최적성의 증거는 아니다.

Dual-graph attention의 gate를 학습 자유도로 풀었을 때, Borbély 2-process에서 유도한 $T_{\text{WAKE}} = 0.315$ 초기값에서 평균 +0.002만 이동. **이론 상수의 경험적 준최적성 확인**.

---

## 7.5 최종 Winning Stack

winning stack은 이 benchmark의 선택 결과이며 배포 전에는 비용·안전성·외부 holdout을 함께 통과해야 한다.

| 층 | 채택 | 공헌 |
|---|---|---|
| Position encoding | **EulerCEAttention** (π-phase + e-decay) | RoPE 대비 −5.17σ |
| Depth structure | **Recursive k=3** (bootstrap fixed-point) | single-pass 대비 −9.0σ |
| FFN | **SwiGLU + Riemann init** | −1.21σ + 분산 3x 감소 |

**최종 PPL**: 22.99 (RoPE 기준) → 17.51 (= **−23.8% reduction**)

---

## 7.6 버림 / 실패 목록

실패 목록은 탐색 편향을 기록하는 negative control이며, 이후 설계가 같은 조건을 우회했는지 점검하는 기준이다.

| 설계 | 결과 | 원인 |
|---|---|---|
| Dual-Laplacian attention (ce_dual) | −3.4σ LOSS | Convex softmax 혼합이 sharpness 희석 |
| Parallel std+dual (ce_par) | +52% params로 TIE | 비효율 |
| Euler-FFN full modulation | +1.30σ LOSS | Pointwise MLP에 π-phase modulation 과함 |
| ζ(1/2+ix) activation | +1.8 PPL, 17x 느림 | 진동 과다 + 비현실적 비용 |
| Berry-Keating $xp$ attention (Design 3) | 미구현 | 연구급 난이도 |

---

## 7.7 실험 재현

아래 명령은 지정한 환경과 데이터 snapshot에서만 같은 결과를 목표로 한다. checksum·dependency·seed가 다르면 수치 차이는 실패 원인과 함께 기록해야 한다.

```bash
# Position encoding 벤치
python3 examples/ai/bench_euler_rotary.py --steps 500 --seeds 5

# Recursive depth 벤치
python3 examples/ai/bench_recursive_euler.py --steps 300 --seeds 3

# Position × FFN 조합
python3 examples/ai/bench_riemann_combo.py --steps 300 --seeds 3

# Design 4 FFN init 분산
python3 examples/ai/bench_riemann_ffn_init.py --steps 300 --seeds 3
```

---

## 7.8 이론 상수 출처

상수의 출처는 코드와 문헌으로 추적하되, 공학 initialisation에 썼다는 사실이 물리적 지위를 변경하지 않는다.

- Riemann 영점 첫 100개: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RIEMANN_ZEROS_IM` (Titchmarsh 표, Odlyzko 검증)
- Euler 상수 bitfield: `reality_stone/python/reality_stone/clarus/ce_euler.py::EULER_BASIS`
- Borbély $T_{\text{WAKE}} = 1/(3 + \text{AD}(1-\text{AD}))$: `reality_stone/python/reality_stone/clarus/constants.py`
- 모든 유도는 `docs/경로적분.md` §51-67 최소생성문법과 `docs/1_강의/A_연역적_유도.md` §801 고정점 개념에 근거.
