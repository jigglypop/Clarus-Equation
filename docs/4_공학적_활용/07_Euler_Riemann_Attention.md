# 7. Euler-Riemann Attention: CE 이론 상수의 Transformer 이식

## 7.1 동기

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

### 7.2.1 Euler-CE Attention (core)

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

구현: `clarus/ce_euler.py::EulerCEAttention`

### 7.2.2 Recursive ClarusCell

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

구현: `clarus/ce_euler.py::RecursiveEulerCEBlock`

### 7.2.3 Riemann Rotary (RH axiom)

리만 가설을 **공학 axiom**으로 채택: 모든 비자명 영점의 허수부 $\gamma_n$이 $\{\text{π}, e, \pi e, \pi/e\}$ 처럼 aperiodic 무리수 수열.

$$
\theta_{n,k} = n \cdot (\gamma_k / \gamma_1)^{-1} \cdot s_h
$$
첫 100개 $\gamma_n$ 하드코딩 (Titchmarsh 표), 이후는 Riemann–von Mangoldt 점근식 $\gamma_n \approx 2\pi n / \log n$.

Montgomery–Dyson 대응: $\gamma_n$ 간격은 Gaussian Unitary Ensemble (GUE) 고유값 간격과 동일 통계. "maximal disorder + maximal order" 구조.

구현: `clarus/ce_riemann_attn.py::RiemannRotaryAttention`

### 7.2.4 FFN 변형

| 이름 | 식 |
|---|---|
| std | $\text{GELU}(W_1 x)\,W_2$ |
| swiglu | $\text{SiLU}(W_g x) \odot (W_u x)\,W_d$ |
| euler_decay | $\text{GELU}(h)\cdot e^{-|h|/\xi}$ |
| euler_phase | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))$ |
| euler_full | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))\cdot e^{-|h|/\xi}$ |
| zeta | $x\cdot\sigma(x)\cdot(1 + \lambda\cdot z(x))$, $z = |\zeta(1/2 + ix)|^2$ |

구현: `clarus/ce_ffn.py`, `clarus/ce_zeta.py`

### 7.2.5 Riemann FFN init

$W_{\text{up}}$의 한 축을 $\gamma_n$ 간격으로 스케일. 키 좌표가 GUE 분포. 구현: `clarus/ce_riemann_attn.py::riemann_zero_init`

---

## 7.3 벤치 결과 (모두 char-level LM, 400K-char docs corpus, 2-layer, $d_{\text{model}}=96$)

### 7.3.1 Position Encoding (5 seeds × 500 steps, PR #1 commit 3bbc558)

| 변형 | PPL | RoPE 대비 |
|---|---|---|
| std_abs (absolute PE) | 25.29 ± 0.22 | +64σ worse |
| std_rope | 14.94 ± 0.28 | baseline |
| euler_hard (fixed bits) | 14.53 ± 0.30 | **−2.20σ WIN** |
| euler_soft (learnable) | 14.13 ± 0.22 | **−5.12σ WIN** |
| **euler_ce (π-phase + e-decay)** | **14.16 ± 0.18** | **−5.17σ WIN** |

**결론**: Euler rotary가 RoPE를 통계적으로 유의하게 이김 (5σ). 이론-정합(euler_ce)이 분산 최저.

### 7.3.2 Recursive ClarusCell (3 seeds × 300 steps, commit f7f17f4)

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

모두 k=3 recursive 위:

| 변형 | PPL | euler_ce_std 대비 |
|---|---|---|
| euler_ce_std (baseline) | 18.42 ± 0.47 | — |
| **euler_ce_swiglu** | **17.78 ± 0.78** | **−1.21σ WIN** |
| euler_ce_euler_full | 18.98 ± 0.59 | +1.30σ LOSS |
| riemann_std | 18.72 ± 0.39 | +0.85σ TIE |
| riemann_euler_full | 19.11 ± 0.54 | +1.67σ LOSS |

### 7.3.4 Design 2 ζ activation (LOSS 기록)

| FFN | PPL | time |
|---|---|---|
| swiglu | 17.78 ± 0.78 | 30.9s |
| **zeta** | 19.57 ± 0.63 | **521s (17x)** |

**결론**: |ζ(1/2+ix)|² modulation은 학습에 해로움 + 비현실적 비용.

### 7.3.5 Design 4 Riemann FFN init

| Init | PPL | std |
|---|---|---|
| kaiming | 17.78 | 0.78 |
| **riemann** | 17.51 | **0.24 (3x 감소)** |

평균 TIE (z=−0.57), **분산 현저히 감소**. Seed 안정성 개선.

---

## 7.4 이론적 해석

### 7.4.1 Euler = Riemann (공학적 등가)

Design 1 (Riemann rotary) 이 Euler-CE와 TIE. 이는 **예측된 결과**:
- Montgomery–Dyson: $\gamma_n$ 간격 ≃ GUE
- $\{1, \pi, e, \pi e, \pi/e\}$: 무리수 aperiodic
- 두 수열 모두 **최대 질서 + 최대 혼돈** 통계 공유

공학적 함의: 리만 가설 참/거짓에 관계없이 **Euler-CE가 충분**. 추가 복잡도는 이득 없음.

### 7.4.2 자기재귀 = Bootstrap 고정점

$k=3$에서 수렴은 CE `ε² = exp[-(1-ε²)D_eff]` 의 빠른 수렴과 일치. $D_{\text{eff}} \approx 3$에서 Banach 계수가 작아 3-step fixed-point 충분.

### 7.4.3 Borbély $T_{\text{WAKE}}$ 경험 검증 (이전 commit 7f7fa18)

Dual-graph attention의 gate를 학습 자유도로 풀었을 때, Borbély 2-process에서 유도한 $T_{\text{WAKE}} = 0.315$ 초기값에서 평균 +0.002만 이동. **이론 상수의 경험적 준최적성 확인**.

---

## 7.5 최종 Winning Stack

| 층 | 채택 | 공헌 |
|---|---|---|
| Position encoding | **EulerCEAttention** (π-phase + e-decay) | RoPE 대비 −5.17σ |
| Depth structure | **Recursive k=3** (bootstrap fixed-point) | single-pass 대비 −9.0σ |
| FFN | **SwiGLU + Riemann init** | −1.21σ + 분산 3x 감소 |

**최종 PPL**: 22.99 (RoPE 기준) → 17.51 (= **−23.8% reduction**)

---

## 7.6 버림 / 실패 목록

| 설계 | 결과 | 원인 |
|---|---|---|
| Dual-Laplacian attention (ce_dual) | −3.4σ LOSS | Convex softmax 혼합이 sharpness 희석 |
| Parallel std+dual (ce_par) | +52% params로 TIE | 비효율 |
| Euler-FFN full modulation | +1.30σ LOSS | Pointwise MLP에 π-phase modulation 과함 |
| ζ(1/2+ix) activation | +1.8 PPL, 17x 느림 | 진동 과다 + 비현실적 비용 |
| Berry-Keating $xp$ attention (Design 3) | 미구현 | 연구급 난이도 |

---

## 7.7 실험 재현

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

- Riemann 영점 첫 100개: `clarus/ce_riemann_attn.py::RIEMANN_ZEROS_IM` (Titchmarsh 표, Odlyzko 검증)
- Euler 상수 bitfield: `clarus/ce_euler.py::EULER_BASIS`
- Borbély $T_{\text{WAKE}} = 1/(3 + \text{AD}(1-\text{AD}))$: `clarus/constants.py`
- 모든 유도는 `docs/경로적분.md` §51-67 최소생성문법과 `docs/1_강의/A_연역적_유도.md` §801 고정점 개념에 근거.
