# LLM Context Full Extract

This file concatenates LLM-relevant source, tests, and docs from the current working tree.
---
## File: `docs/4_공학적_활용/07_Euler_Riemann_Attention.md`

```markdown
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

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEAttention`

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

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::RecursiveEulerCEBlock`

### 7.2.3 Riemann Rotary (RH axiom)

리만 가설을 **공학 axiom**으로 채택: 모든 비자명 영점의 허수부 $\gamma_n$이 $\{\text{π}, e, \pi e, \pi/e\}$ 처럼 aperiodic 무리수 수열.

$$
\theta_{n,k} = n \cdot (\gamma_k / \gamma_1)^{-1} \cdot s_h
$$
첫 100개 $\gamma_n$ 하드코딩 (Titchmarsh 표), 이후는 Riemann–von Mangoldt 점근식 $\gamma_n \approx 2\pi n / \log n$.

Montgomery–Dyson 대응: $\gamma_n$ 간격은 Gaussian Unitary Ensemble (GUE) 고유값 간격과 동일 통계. "maximal disorder + maximal order" 구조.

구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RiemannRotaryAttention`

### 7.2.4 FFN 변형

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

$W_{\text{up}}$의 한 축을 $\gamma_n$ 간격으로 스케일. 키 좌표가 GUE 분포. 구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::riemann_zero_init`

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

- Riemann 영점 첫 100개: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RIEMANN_ZEROS_IM` (Titchmarsh 표, Odlyzko 검증)
- Euler 상수 bitfield: `reality_stone/python/reality_stone/clarus/ce_euler.py::EULER_BASIS`
- Borbély $T_{\text{WAKE}} = 1/(3 + \text{AD}(1-\text{AD}))$: `reality_stone/python/reality_stone/clarus/constants.py`
- 모든 유도는 `docs/경로적분.md` §51-67 최소생성문법과 `docs/1_강의/A_연역적_유도.md` §801 고정점 개념에 근거.
```
---
## File: `docs/4_공학적_활용/08_기상_그래프잔차_연구.md`

```markdown
21# 8. 기상 관측망 CE 그래프 잔차 연구
4
## 8.1 목표와 지위

이 문서는 대용량 기상 CSV를 단순한 DB 분석 문제가 아니라, CE의 관측 사상, 그래프 라플라시안, 잔차 이완, 예측 gate로 접는 연구 초안이다.

지위는 `Application Bridge`다. 기존 기상학, 수치예보, 통계 예측을 대체한다고 주장하지 않는다. CE가 제안하는 것은 관측망 위의 잔차장을 안정 functional로 정리하고, 그것이 단순 baseline보다 나은지 검증하는 절차다.

핵심 질문은 다음이다.

$$
\left(Y_{r,t}, G_{\mathrm{weather}}, Q_t\right)
\longrightarrow
Y_{r,t+\Delta}
$$

여기서 \(r\)은 관측소, \(t\)는 시간, \(Y_{r,t}\)는 기온, 강수, 습도, 풍속 같은 관측 벡터다.

---

## 8.2 문제 재정의

원자료를 바로 모델에 넣지 않는다. 먼저 각 관측소와 시간대의 평년값을 빼서 무차원 잔차장으로 만든다.

$$
e^v_{r,t}
=
\frac{
Y^v_{r,t}
-
\mu^v_{r,\mathrm{doy},\mathrm{hour}}
}{
\sigma^v_{r,\mathrm{doy},\mathrm{hour}}+\epsilon
}
$$

- \(v\): 기온, 강수, 습도, 풍속 등 변수
- \(\mu\): 관측소별, 연중일별, 시간대별 기준장
- \(\sigma\): 같은 조건의 표준편차 또는 robust scale

이 정의는 `참조/무차원_감사_수학.md`의 규칙을 따른다. CE 코어의 exp, log, selection score에는 차원 있는 섭씨, mm, m/s를 직접 넣지 않고, 반드시 무차원 잔차 \(e\)를 넣는다.

---

### 8.2.1 무차원 잔차장 원칙과 hallucination 감소

핵심은 예측기가 원자료 \(Y\)를 직접 생성하지 않는다는 점이다. 먼저 기준장 \(B\)와 스케일장 \(\Sigma\)를 고정하고,

$$
e
=
\Sigma^{-1}(Y-B)
$$

만 모델의 입력과 출력으로 허용한다. 모델이 만드는 것은 "새 사실"이 아니라 기준장으로 설명되지 않는 무차원 이탈량이다.

$$
\widehat Y
=
B+\Sigma\widehat e
$$

따라서 \(\widehat e\)가 작고 안정 gate를 통과하면 \(Y\)의 복원이 허용되고, \(\widehat e\)가 크거나 graph/source curvature가 깨지면 생성이 아니라 보류, 검색, 재질문으로 넘어간다. 이것은 LLM hallucination 감소와 같은 구조다.

LLM 문맥에서는 \(Y\)를 답변, \(B\)를 검색된 근거/인용/스키마/계산 결과, \(\Sigma\)를 근거 불확실성 또는 문맥별 허용 오차로 읽는다.

$$
e_{\mathrm{claim}}
=
\Sigma_{\mathrm{evidence}}^{-1}
\left(
Y_{\mathrm{claim}}
-
B_{\mathrm{evidence}}
\right)
$$

답변 생성은 \(Y_{\mathrm{claim}}\)을 자유롭게 쓰는 것이 아니라, 근거장 위의 잔차가 selection gate를 통과할 때만 허용된다.

$$
\phi
=
w_1\|e_{\mathrm{claim}}\|^2
+w_2\|\Delta_G e_{\mathrm{claim}}\|^2
+w_3\|\partial_t e_{\mathrm{claim}}\|^2,
\qquad
S=\exp(-D_{\mathrm{eff}}\phi)
$$

\(S\)가 낮으면 hallucination 위험이 높은 상태다. 즉 CE식 접근은 "AI가 더 많이 상상하게 하기"가 아니라, 기준장 밖으로 나온 무차원 잔차를 측정하고 안정한 잔차만 답변으로 되돌리는 hallucination 억제 규칙이다.

---

## 8.3 관측소 그래프

관측소망을 가중 그래프로 둔다.

$$
G_{\mathrm{weather}}=(V_{\mathrm{station}},E,A)
$$

간선 가중치는 거리, 고도, 지형, 과거 상관을 섞어 만든다.

$$
a_{rs}
=
\exp\left(
-\frac{d_{\mathrm{geo}}(r,s)}{\ell_d}
-\frac{|h_r-h_s|}{\ell_h}
\right)
\cdot
\max(0,\operatorname{corr}(e_r,e_s))
\cdot
\tau_{\mathrm{terrain}}(r,s)
$$

조건은 다음이다.

$$
a_{rs}\ge 0,\qquad a_{rr}=0,\qquad a_{rs}=a_{sr}
$$

그래프 라플라시안은 기존 문서와 같은 방식으로 쓴다.

$$
\Delta_G f(r)=\sum_s a_{rs}(f_s-f_r)
$$

이 항은 이웃 관측소와의 차이를 줄이는 smoothing이지만, 실제 기상에서는 전선, 산악 효과, 해안 효과가 있으므로 완전 평탄화가 아니라 안정성 gate 아래에서만 사용한다.

---

## 8.4 최소 동역학

계절성과 일주기는 이미 baseline에서 제거했다고 두면, 최소 잔차 동역학은 다음이다.

$$
\widehat e_{t+\Delta}
=
\rho e_t
+\gamma\Delta_G e_t
+U_{\mathrm{synoptic},t}
+F_{\mathrm{slow},t}
$$

더 단순한 1차 gate에서는 외부 forcing 없이 시작한다.

$$
\widehat e_{t+\Delta}^{\mathrm{graph}}
=
\rho e_t+\gamma\Delta_G e_t
$$

비교할 flat baseline은 다음이다.

$$
\widehat e_{t+\Delta}^{\mathrm{flat}}
=
\rho e_t
$$

그래프 항이 연구식으로 남으려면 holdout에서 다음을 만족해야 한다.

$$
\mathcal L_{\mathrm{graph}}
<
\min
\left(
\mathcal L_{\mathrm{flat}},
\mathcal L_{\mathrm{climatology}},
\mathcal L_{\mathrm{persistence}},
\mathcal L_{\mathrm{shuffled\;graph}},
\mathcal L_{\mathrm{distance\;only}}
\right)
$$

---

## 8.5 안정성 조건

잡음과 forcing을 빼면 잔차는 다음처럼 움직인다.

$$
e_{t+\Delta}=(\rho I-\gamma L_G)e_t
$$

따라서 선형 안정성은 기존 graph gate와 동일하다.

$$
\max_k|\rho-\gamma\lambda_k(L_G)|<1
$$

이 조건이 깨지면 예측 성능이 좋아 보여도 CE 안정식으로 승격하지 않는다. 특히 관측소가 많아지면 \(\lambda_{\max}\)가 커질 수 있으므로, \(\gamma\)는 station graph마다 다시 상한을 계산해야 한다.

---

## 8.6 CE selection score

관측소 \(r\), 시간 \(t\)에서 잔차 에너지를 다음처럼 둔다.

$$
\phi_{r,t}
=
w_1\|e_{r,t}\|^2
+w_2\|\Delta_G e_{r,t}\|^2
+w_3\|e_{r,t}-e_{r,t-\Delta}\|^2
+w_4\|e_{r,t}-2e_{r,t-\Delta}+e_{r,t-2\Delta}\|^2
$$

선택 확률은 CE의 정보 포텐셜 규칙과 맞춰

$$
S_{r,t}
=
\exp(-D_{\mathrm{eff}}\phi_{r,t})
$$

로 둔다. 여기서 \(D_{\mathrm{eff}}\)는 무차원 계수로만 사용한다. 이상 점수는

$$
\mathrm{stress}_{r,t}
=
\frac{1-S_{r,t}}{1-\varepsilon^2}
$$

로 정규화한다.

해석은 다음이다.

| 상태 | 의미 |
|---|---|
| 낮은 \(\phi\), 높은 \(S\) | 평년값과 이웃 관측소가 함께 설명하는 안정 상태 |
| 시간 잔차만 큼 | 국소 급변, 센서 오류 또는 국소 기상 이벤트 |
| 그래프 잔차가 큼 | 전선, 지형 효과, 관측소 이상치 후보 |
| \(S<\varepsilon^2\) | selection failure 후보, 별도 검토 |

---

## 8.7 세 가지 직접 응용

### 8.7.1 결측 보간

빠진 값은 시간-그래프 action을 최소화해서 복원한다.

$$
\min_{\{e_{r,t}: (r,t)\in M\}}
\sum_t
\left\|
e_{t+\Delta}
-
\rho e_t
-
\gamma\Delta_Ge_t
\right\|^2
+
\alpha_1\|\nabla_Ge_t\|^2
+
\alpha_2\|L_Ge_t\|^2
$$

비교 baseline은 linear interpolation, station climatology, inverse distance weighting, kNN station 평균이다.

### 8.7.2 이상 탐지

이상 탐지는 원자료 값의 크기가 아니라 잔차 곡률로 본다.

$$
\mathrm{anomaly}(r,t)
=
\|\Delta_Ge_{r,t}\|^2
+
\eta
\|e_{r,t}-2e_{r,t-\Delta}+e_{r,t-2\Delta}\|^2
$$

센서 오류는 단일 관측소에서만 크게 튀고, 실제 기상 이벤트는 인접 관측소와 시간축에 구조적으로 퍼진다는 점을 이용한다.

### 8.7.3 단기 예측

1시간, 3시간, 6시간, 24시간 horizon에서 다음을 비교한다.

$$
\widehat Y^v_{r,t+\Delta}
=
\mu^v_{r,\mathrm{doy},\mathrm{hour}}
+
\sigma^v_{r,\mathrm{doy},\mathrm{hour}}
\widehat e^v_{r,t+\Delta}
$$

CE 모델은 절대 기상장을 처음부터 예측하는 것이 아니라, 평년값으로 설명되지 않는 잔차장을 예측한다.

---

## 8.8 대용량 데이터 처리 위치

2억 행 CSV는 CE 알고리즘의 입력 형식이 아니라 전처리 대상이다. 권장 흐름은 다음이다.

1차 공개 데이터 후보는 기상자료개방포털의 `종관기상관측(ASOS)` 자료다.

| 항목 | ASOS 적용 |
|---|---|
| 출처 | `https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36` |
| 자료형태 | 분, 시간(매정시), 일, 월, 연 |
| 제공기간 | 1904년부터. 단, 지점별/요소별 기간은 다름 |
| 제공지점 | 105개 |
| 제공요소 | 기온, 강수, 바람, 기압, 습도, 일사, 일조, 눈, 구름, 시정, 지면상태, 지면/초상온도, 일기현상, 증발량, 현상번호 |
| 품질정보 | 시간/분 자료에 QC FLAG 제공. 기온, 습도, 기압, 지면온도, 풍향, 풍속, 일조에 대해 0 정상, 1 오류, 9 결측 |
| 조회 제한 | 1회 조회 기준 분 1일, 시간 1년, 일 10년, 월/연 제한 없음. 장기간 자료는 파일셋 조회 사용 |

ASOS에 없는 지점이나 더 조밀한 관측망이 필요하면 같은 포털의 `방재기상관측(AWS)`를 보조 graph로 붙인다. 단, 첫 연구 gate는 ASOS만으로 닫는 편이 낫다. ASOS는 지점 수가 적지만 품질 플래그와 장기성이 좋아서 baseline과 station graph를 안정적으로 만들 수 있다.

1. 원본 CSV 보존
2. 관측소 메타데이터 분리: 위도, 경도, 고도, 지형, 관측 시작일
3. CSV를 `year/month/variable` 기준 Parquet으로 변환
4. DuckDB 또는 Polars로 streaming groupby 수행
5. 관측소별 \(\mu,\sigma\) baseline table 생성
6. 잔차 Parquet 생성
7. CE graph residual gate 실행

즉 DuckDB, Polars, Parquet은 저장과 스캔 도구이고, 연구 핵심은 그래프 잔차장과 gate다.

---

## 8.9 최소 실험 프로토콜

처음부터 1907년 전체와 분자료를 섞지 않는다.

1차 실험:

| 항목 | 선택 |
|---|---|
| 데이터 | KMA ASOS 시간 자료 |
| 기간 | 2010-2024 |
| 해상도 | 1시간 |
| 변수 | 기온 먼저, 이후 강수/습도/풍속 |
| 관측소 | 결측률 낮은 지상 관측소 |
| horizon | 1h, 3h, 6h, 24h |
| split | year holdout, station holdout, event holdout |

가설:

| 가설 | 통과 조건 |
|---|---|
| H1 graph residual | \(L_{\mathrm{graph}} < L_{\mathrm{flat}}\) |
| H2 real graph | 실제 graph가 shuffled graph보다 낮은 손실 |
| H3 imputation | CE 보간이 IDW/평년값보다 낮은 MAE/RMSE |
| H4 event stress | 폭염, 한파, 집중강수 시 stress가 상승 |
| H5 stability | 모든 split에서 \(\max_k|\rho-\gamma\lambda_k|<1\) |

---

## 8.10 코드 이식 경로

현재 레포의 가까운 출발점은 다음이다.

| 기존 파일 | 기상 연구에서의 역할 |
|---|---|
| `examples/physics/graph_vs_flat_prediction.py` | graph-vs-flat gate의 최소형 |
| `examples/physics/brain_equation_integrated_gate.py` | 전역 전이, ablation, real JSON schema |
| `reality_stone/python/reality_stone/clarus/ce_laplacian.py` | 그래프 라플라시안/row-stochastic kernel 구현 참고 |
| `quant/ce_market_residual_index.py` | residual stress, selection score, regime report 구조 참고 |
| `reality_stone/python/reality_stone/clarus/dimensionless.py` | 기상 변수 무차원화 gate |

현재 추가된 실행 파일은 다음이다.

```text
examples/weather/kma_asos_fetch.py
examples/weather/weather_graph_residual_gate.py
examples/weather/README.md
```

KMA APIHub 인증키가 있으면 다음처럼 시간자료를 받는다.

```powershell
$env:KMA_AUTH_KEY="..."
.\.venv\Scripts\python.exe examples\weather\kma_asos_fetch.py `
  --asos-hourly `
  --start 202401010000 `
  --end 202401312300 `
  --stations 108:112:133 `
  --out data\weather\asos_202401.raw
```

지점정보도 같은 인증키로 받는다.

```powershell
.\.venv\Scripts\python.exe examples\weather\kma_asos_fetch.py `
  --station-info `
  --stations 0 `
  --out data\weather\stations_sfc.raw
```

받은 자료나 포털 CSV를 gate에 넣는다.

```powershell
.\.venv\Scripts\python.exe examples\weather\weather_graph_residual_gate.py `
  --csv data\weather\asos_202401.raw `
  --station-meta data\weather\stations_sfc.raw `
  --variable temperature `
  --graph-mode geo_corr
```

외부 데이터 없이 코드 gate만 확인하려면 다음을 실행한다.

```powershell
.\.venv\Scripts\python.exe examples\weather\weather_graph_residual_gate.py --demo
```

---

## 8.11 2차 관성-곡률 gate

1차 gate는 다음 선형 잔차식을 검증한다.

$$
\widehat e_{t+\Delta}
=
\rho e_t+\gamma\Delta_G e_t
$$

실제 ASOS 시간 기온에서는 이 항만으로도 flat보다 낫지만, 시간 잔차는 단순 1차 AR보다 2차 관성을 가진다. 따라서 다음 승격 모델을 둔다.

$$
\widehat e_{t+\Delta}
=
a_0 e_t
+a_1 e_{t-\Delta}
+\gamma_0 \Delta_G e_t
+\gamma_1 \Delta_G e_{t-\Delta}.
$$

라플라시안 고유값 \(\lambda_k\)에서 각 mode는

$$
e^{(k)}_{t+\Delta}
=
(a_0-\gamma_0\lambda_k)e^{(k)}_t
+
(a_1-\gamma_1\lambda_k)e^{(k)}_{t-\Delta}
$$

로 움직인다. 안정성 gate는 모든 \(k\)에 대해 companion matrix의 최대 고유반경이 1보다 작아야 한다.

$$
\rho_\mathrm{max}
=
\max_k
\rho\left(
\begin{bmatrix}
a_0-\gamma_0\lambda_k & a_1-\gamma_1\lambda_k\\
1 & 0
\end{bmatrix}
\right)
<1
$$

2024년 1월 1일 00시부터 2024년 6월 30일 23시까지 KMA ASOS 93개 지점 시간 기온으로 돌린 1차 결과는 다음이다.

| model | MSE |
|---|---:|
| persistence | 0.02773961 |
| flat | 0.02802772 |
| graph | 0.02733911 |
| inertial flat | 0.01757276 |
| inertial graph | 0.01248087 |

핵심 비율은 다음이다.

| comparison | ratio | improvement |
|---|---:|---:|
| graph / flat | 0.975431 | 2.46% |
| inertial graph / inertial flat | 0.710240 | 28.98% |
| inertial graph / graph | 0.456521 | 54.35% |
| inertial graph / best shuffled inertial graph | 0.903662 | 9.63% |

계수는

$$
a_0=1.837888,\quad
a_1=-0.854994,\quad
\gamma_0=0.113668,\quad
\gamma_1=-0.093366.
$$

이 자유계수식은 다음 CE 감쇠-곡률식으로 다시 쓸 수 있다.

$$
\widehat e_{t+\Delta}
=
(1-\beta)e_t
+\kappa(e_t-e_{t-\Delta})
+c^2\Delta_Ge_t
+\nu(\Delta_Ge_t-\Delta_Ge_{t-\Delta})
$$

대응은

$$
a_0=1-\beta+\kappa,\quad
a_1=-\kappa,\quad
\gamma_0=c^2+\nu,\quad
\gamma_1=-\nu
$$

이고 실제 fit은

$$
\beta=0.017105,\quad
\kappa=0.854994,\quad
c^2=0.020303,\quad
\nu=0.093366.
$$

즉 계수는 모두 양의 감쇠, 양의 관성, 양의 곡률 stiffness, 양의 곡률 viscosity로 읽힌다. 이 형태가 더 중요한 이유는 성능 향상을 자유계수 트릭이 아니라 무차원 잔차장의 안정한 감쇠-복원 방정식으로 해석하게 해 주기 때문이다.

최대 modal radius는 \(0.924659\)으로 안정 gate 안쪽이다. 따라서 현재 데이터에서는 단순 graph residual보다 2차 관성-곡률 residual이 더 강한 후보로 남는다.

반대로 라플라시안 고유모드 band별로 독립 계수를 주는 `modal_band` 모델은 같은 split에서 실패했다. 이는 "곡률 mode별 persistence를 자유롭게 두는 것"보다 "현재/직전 곡률의 되돌림 구조"가 이 데이터에서는 더 중요한 설명 축이라는 뜻이다.

---

## 8.12 반증 조건

이 연구는 다음 중 하나가 반복되면 약해진다.

- 실제 station graph가 shuffled graph보다 예측력이 낮다.
- \(\gamma\) 안정 조건을 만족하는 구간에서 graph term의 이득이 사라진다.
- 잔차 곡률과 알려진 이상기상/센서 오류가 상관하지 않는다.
- station holdout에서 성능이 무너지고, 같은 관측소 안의 시간 보간에서만 좋다.
- 강수처럼 sparse한 변수에서 selection score가 단순 결측률 또는 관측소별 bias만 따라간다.

이 경우 CE 해석은 기상 예측 원리 후보가 아니라, 단순한 graph regularizer 또는 데이터 품질 진단 도구로 하향한다.

---

## 8.13 한 줄 결론

이 문제의 CE식 해법은 "2억 행을 AI에게 읽히는 것"이 아니라, 관측소망을 그래프로 만들고 평년값에서 벗어난 무차원 잔차장을 먼저 \(\rho e+\gamma\Delta_Ge\)로 gate하고, 통과하면 \(a_0e_t+a_1e_{t-\Delta}+\gamma_0\Delta_Ge_t+\gamma_1\Delta_Ge_{t-\Delta}\) 같은 2차 관성-곡률항으로 승격하는 것이다. 대용량 처리는 Parquet/DuckDB가 맡고, 연구 결론은 graph-vs-flat, inertial-vs-inertial-flat, shuffled graph, station/time/event holdout gate가 결정한다.
```
---
## File: `docs/4_공학적_활용/09_무차원_잔차장_환각억제.md`

```markdown
# 9. 무차원 잔차장 Hallucination Gate

## 9.1 목적

이 문서는 LLM hallucination을 "모델이 거짓말을 한다"가 아니라, **근거장으로 설명되지 않는 무차원 잔차가 gate 없이 답변으로 복원되는 현상**으로 정식화한다.

핵심 원칙은 다음이다.

$$
\text{LLM이 직접 생성할 대상은 }Y\text{가 아니라 }e=\Sigma^{-1}(Y-B)\text{이다.}
$$

여기서

| 기호 | 의미 |
|---|---|
| \(Y\) | claim 또는 답변 단위의 출력 |
| \(B\) | 검색, 문서, 계산, DB, 스키마로 만든 기준장 |
| \(\Sigma\) | 근거 불확실성, source disagreement, 허용 오차 |
| \(e\) | 기준장 밖으로 나온 무차원 잔차 |

답변 복원은 오직 다음 꼴로만 허용한다.

$$
\widehat Y = B+\Sigma\widehat e
$$

\(\widehat e\)가 안정 gate를 통과하지 못하면 \(\widehat Y\)를 만들지 않는다. 이때 동작은 생성이 아니라 검색, 재질문, 보류, 거절이다.

---

## 9.2 Claim-Evidence 기준장

답변을 claim 단위로 나누고 각 claim \(i\)를 스칼라 또는 벡터 값 \(y_i\)로 둔다. 실제 구현에서는 NLI, entailment score, numeric parser, schema validator, SQL result, unit checker가 \(y_i\)를 만든다. 이 문서의 수학은 \(y_i\)가 스칼라라고 두고 쓴다.

claim \(i\)에 대해 근거 source \(m\)이 값 \(b_{im}\), 불확실도 \(\sigma_{im}\), 신뢰 가중치 \(w_{im}\)를 준다고 하자. precision은

$$
p_{im}=\frac{w_{im}}{\sigma_{im}^2+\epsilon_\sigma^2}
$$

기준장은 precision-weighted mean이다.

$$
B_i
=
\frac{\sum_m p_{im}b_{im}}{\sum_m p_{im}}
$$

source disagreement는

$$
d_i^2
=
\frac{\sum_m p_{im}(b_{im}-B_i)^2}{\sum_m p_{im}}
$$

최종 scale은 sampling uncertainty와 disagreement를 합친다.

$$
\Sigma_i
=
\sqrt{
\frac{1}{\sum_m p_{im}}
+d_i^2
+\epsilon_\sigma^2
}
$$

따라서 claim residual은

$$
e_i
=
\frac{y_i-B_i}{\Sigma_i}.
$$

근거가 없으면 \(\sum_m p_{im}=0\)이고, 해당 claim은 자동으로 missing-evidence penalty를 받는다. 이 경우 모델이 자신감을 갖고 말할수록 \(e_i\)가 커진다.

---

## 9.3 Claim Graph Curvature

claim들은 서로 독립이 아니다. 어떤 claim은 다른 claim을 지지하고, 어떤 claim은 다른 claim과 모순된다. claim graph를

$$
G_C=(V_C,E_C,A,R)
$$

로 둔다.

| 기호 | 의미 |
|---|---|
| \(A_{ij}\ge0\) | claim \(i,j\)의 연결 강도 |
| \(R_{ij}\in\{+1,-1\}\) | \(+1\): 같은 방향, \(-1\): 모순 방향 |
| \(q_j\in[\epsilon^2,1]\) | claim \(j\)의 local evidence reliability |

먼저 graph 없이 local reliability를 계산한다.

$$
q_i
=
\exp\left[
-D_{\mathrm{eff}}
\left(
\alpha e_i^2
+(1-\alpha)\tau_i^2
+P_{\mathrm{missing},i}
\right)
\right]
$$

\(\tau_i=d_i/\Sigma_i\)는 source tension이다.

claim curvature는 이웃의 신뢰도로 가중한다.

$$
\Delta_G e_i
=
\sum_j A_{ij}q_j\left(R_{ij}e_j-e_i\right)
$$

이 \(q_j\)가 중요하다. 거짓 claim 하나가 참 claim을 오염시키지 않도록, 근거 신뢰도가 낮은 이웃은 curvature에 약하게만 들어간다. 반대로 거짓 claim은 근거 신뢰도가 높은 참 claim과 모순될 때 큰 curvature penalty를 받는다.

---

## 9.4 Selection Gate

claim \(i\)의 잔차 에너지는

$$
\phi_i
=
w_e e_i^2
+w_G(\Delta_G e_i)^2
+w_\tau\tau_i^2
+P_{\mathrm{missing},i}
$$

로 둔다.

selection score는 CE 규칙에 맞춰

$$
S_i
=
\exp(-D_{\mathrm{eff}}\phi_i)
$$

이며 stress는

$$
\mathrm{stress}_i
=
\frac{1-S_i}{1-\epsilon^2}.
$$

decision rule은 다음처럼 둔다.

| 조건 | 동작 |
|---|---|
| \(S_i\ge S_{\mathrm{accept}}\), \(|e_i|\le e_{\max}\), \(|\Delta_Ge_i|\le g_{\max}\) | claim 허용 |
| 근거 없음 | 거절 또는 검색 |
| \(S_i\le S_{\mathrm{reject}}\) 또는 \(|e_i|\) 과대 | 거절 |
| 중간 영역 | 추가 검색, 재질문, human review |

즉 답변은 accepted claim들의 조합이다. rejected/review claim은 문장 생성에 들어가지 않는다.

---

## 9.5 RAG와의 차이

RAG는 종종 "문서를 검색해서 prompt에 붙인다"로 쓰인다. CE residual gate에서는 역할이 다르다.

| 단계 | RAG식 읽기 | CE residual gate식 읽기 |
|---|---|---|
| 검색 | 컨텍스트 추가 | \(B\) 기준장 생성 |
| source 품질 | ranking score | \(\Sigma\), \(\tau\) 생성 |
| 생성 | 문맥 조건부 답변 | \(\widehat e\)가 gate를 통과할 때만 \(B+\Sigma\widehat e\) 복원 |
| 불확실성 | 후처리 confidence | selection stress로 claim별 차단 |

따라서 이 접근의 목적은 더 그럴듯한 답변이 아니라, **근거장 밖의 무차원 잔차를 보이는 claim을 출력하지 않는 것**이다.

---

## 9.6 최소 실행 예제

구현 파일:

```text
examples/ai/claim_residual_gate.py
```

데모 실행:

```powershell
.\.venv\Scripts\python.exe examples\ai\claim_residual_gate.py
```

현재 데모 결과:

| claim | decision | 의미 |
|---|---|---|
| `asos_93_period` | accept | 근거와 claim이 일치 |
| `ce_damped_passed` | accept | 근거가 있는 연구 결과 |
| `modal_band_passed` | reject | 보고서와 반대되는 claim |
| `unbacked_generalization` | reject_no_evidence | 근거 없는 과잉 일반화 |

이 예제는 자연어 이해기를 구현하지 않는다. 대신 자연어 이해기, 검색기, 계산기, DB 질의기가 만든 구조화된 score를 CE residual gate가 어떻게 걸러야 하는지 보여 준다.

---

## 9.7 기상 잔차장과의 동일 구조

기상 문제에서

$$
e^v_{r,t}
=
\frac{
Y^v_{r,t}
-
\mu^v_{r,\mathrm{doy},\mathrm{hour}}
}{
\sigma^v_{r,\mathrm{doy},\mathrm{hour}}+\epsilon
}
$$

였다. hallucination gate에서는

$$
e_{\mathrm{claim},i}
=
\frac{
Y_{\mathrm{claim},i}
-
B_{\mathrm{evidence},i}
}{
\Sigma_{\mathrm{evidence},i}
}
$$

이다. 두 경우 모두 절대값 \(Y\)를 직접 다루지 않는다. 기준장으로 설명되지 않는 무차원 잔차만 다룬다.

따라서 "기상 데이터 해법"과 "hallucination 억제"는 같은 수학 구조다.

1. 기준장 \(B\)를 만든다.
2. 불확실성 \(\Sigma\)를 만든다.
3. 무차원 잔차 \(e=\Sigma^{-1}(Y-B)\)를 만든다.
4. graph/source curvature를 계산한다.
5. selection score가 낮으면 출력하지 않는다.

이것이 CE 응용에서 가장 중요한 안전장치다.

---

## 9.8 강화 방향

현재 최소 구현은 claim 하나를 스칼라 값으로 두고

$$
e_i=\frac{Y_i-B_i}{\Sigma_i}
$$

를 gate한다. 이것만으로도 근거 없는 claim과 보고서에 반대되는 claim을 막을 수 있지만, 실제 hallucination 억제기로 쓰려면 잔차장을 더 풍부하게 만들어야 한다.

### 9.8.1 스칼라 잔차에서 공분산 잔차로

claim은 하나의 숫자가 아니라 여러 검증 축을 가진다.

$$
z_i=
\begin{bmatrix}
z_{\mathrm{entail}}\\
z_{\mathrm{numeric}}\\
z_{\mathrm{temporal}}\\
z_{\mathrm{unit}}\\
z_{\mathrm{citation}}\\
z_{\mathrm{causal}}
\end{bmatrix},
\qquad
B_i=
\begin{bmatrix}
B_{\mathrm{entail}}\\
B_{\mathrm{numeric}}\\
B_{\mathrm{temporal}}\\
B_{\mathrm{unit}}\\
B_{\mathrm{citation}}\\
B_{\mathrm{causal}}
\end{bmatrix}.
$$

따라서 강화된 무차원 잔차는

$$
e_i=C_i^{-1/2}(z_i-B_i)
$$

이고 claim 에너지는 Mahalanobis 형태로

$$
E_i=e_i^\top e_i
=(z_i-B_i)^\top C_i^{-1}(z_i-B_i)
$$

가 된다. 여기서 \(C_i\)는 source disagreement, extractor uncertainty, 단위 변환 오차, 시간 범위 오차를 모두 포함한 공분산이다.

이 승격이 중요한 이유는 hallucination이 한 축에서만 생기지 않기 때문이다. 숫자는 맞지만 기간이 틀린 claim, citation은 있지만 entailment가 안 되는 claim, 단위가 섞인 claim은 모두 서로 다른 방향의 잔차다. 스칼라 gate는 이 차이를 잃어버린다.

### 9.8.2 source 신뢰도와 signed claim graph의 결합

claim graph만 두면 나쁜 claim이 이웃 claim을 오염시킬 수 있다. 따라서 source reliability를 별도 동역학으로 둔다.

source \(m\)의 신뢰도는 최근 검증된 claim 잔차 평균으로

$$
q_m
=
\exp\left[
-D_{\mathrm{eff}}
\left(
\overline{E}_m
+\lambda_\tau\overline{\tau}_m^2
\right)
\right]
$$

처럼 둔다. source \(m\)이 제공한 claim들의 잔차 에너지가 계속 크면 \(q_m\)이 낮아진다.

하지만 source id가 둘이라고 해서 독립 근거가 둘이라는 뜻은 아니다. 같은 원문을 복사한 mirror source, 같은 vendor feed, 또는 한 source가 precision mass를 거의 다 차지하는 경우는 하나의 근거장으로 봐야 한다. 따라서 독립 source family \(g\)별 precision mass를

$$
M_{ig}
=
\sum_{m\in g}\sum_a
\frac{w_{ima}q_m}{\sigma_{ima}^2+\epsilon_\sigma^2}
$$

로 모으고, Kish effective sample size 형태의 독립 근거 수를 둔다.

$$
N_{\mathrm{eff},i}
=
\frac{\left(\sum_g M_{ig}\right)^2}
{\sum_g M_{ig}^2}.
$$

이에 따른 독립성 penalty는

$$
P_{\mathrm{ind},i}
=
\left[
\max\left(
0,\,
\frac{N_\star-N_{\mathrm{eff},i}}{N_\star}
\right)
\right]^2,
\qquad N_\star=2.
$$

즉 source가 하나뿐이면 \(N_{\mathrm{eff}}=1\), 같은 family의 source 두 개도 \(N_{\mathrm{eff}}=1\), 한 source가 거의 모든 weight를 차지하면 \(N_{\mathrm{eff}}\approx1\)이다. 균형 잡힌 독립 source 둘만 \(N_{\mathrm{eff}}\approx2\)가 된다.

claim graph curvature는

$$
\Delta_G e_i
=
\sum_j A_{ij}\,q_j\,q_{\mathrm{src}(j)}
\left(R_{ij}e_j-e_i\right)
$$

로 강화한다. \(R_{ij}=+1\)은 지지, \(R_{ij}=-1\)은 모순이다. 즉 모순 관계인 claim은 잔차 방향이 반대로 맞아야 하고, 신뢰도 낮은 이웃은 곡률 계산에 거의 영향을 주지 못한다.

### 9.8.3 threshold가 아니라 안정 작용으로 판정

단순히 \(|e_i|<e_{\max}\)만 보는 대신, claim 묶음 전체에 대한 안정 작용을 둔다.

$$
S_{\mathrm{claim}}
=
\sum_i E_i
+\lambda_1\sum_{(i,j)}A_{ij}
\left\|R_{ij}e_j-e_i\right\|^2
+\lambda_2\sum_i\left\|\Delta_Ge_i\right\|^2
+\lambda_\tau\sum_i\tau_i^2
+\lambda_m\sum_i P_{\mathrm{missing},i}.
$$

selection score는

$$
P_{\mathrm{accept},i}
=
\exp(-D_{\mathrm{eff}}\phi_i)
$$

를 유지하되, \(\phi_i\) 안에 \(E_i\), graph curvature, source tension, source reliability, source independence, missing penalty를 모두 넣는다. 현재 구현의 local action은

$$
\phi_i
=
0.10\|e_i\|^2
+1.20\|\Delta_Ge_i\|^2
+0.10\tau_i^2
+1.50(1-\bar q_{\mathrm{src},i})^2
+0.80P_{\mathrm{ind},i}
+P_{\mathrm{missing},i}.
$$

accept hard rule도 같이 둔다.

$$
N_{\mathrm{eff},i}\ge1.60
$$

를 통과해야 claim이 answer composer로 복원된다. \(0.35\le S_i<0.60\)의 중간 영역에서는 추가로 독립 source group이 2개 이상이고 source support가 \(0.90\) 이상이어야 한다. \(0.25\le S_i<0.35\)의 회색지대는 \(\|e_i\|\le0.30\)까지 요구한다.

이 구조는 `7_AGI/6_Hallucination.md`의 곡률 정규화와 같은 문법이다. 환각을 "문장 내용"으로 직접 잡는 것이 아니라, 근거장 밖의 고곡률 잔차가 출력으로 복원되는 것을 막는다.

### 9.8.4 OOD와 음성 대조군 gate

강화된 gate는 demo에서 잘 도는 것만으로는 부족하다. 최소 검증은 네 가지다.

| gate | 목적 | 통과 기준 |
|---|---|---|
| `R-H1` calibration | selection이 실제 false accept rate와 맞는지 | reliability curve와 ECE 보고 |
| `R-H2` negative control | 근거 없는 claim, 반대 claim, 단위 오류 claim 차단 | false accept rate 감소 |
| `R-H3` graph ablation | graph가 진짜 도움이 되는지 | shuffled graph보다 낮은 false accept |
| `R-H4` OOD holdout | source/topic/time이 바뀌어도 동작하는지 | holdout에서 비회귀 또는 개선 |

특히 `docs/7_AGI/19_OOD_Generalization.md`의 결론을 따르면, in-distribution demo만 보면 안 된다. source holdout, topic holdout, time holdout을 따로 만들어야 한다. OOD에서 무너지면 구조가 아니라 데이터셋 특이성을 배운 것이다.

### 9.8.5 추론 개입 정책

gate는 점수만 내면 약하다. stress가 높을 때 실제 생성 정책을 바꿔야 한다.

| 조건 | 개입 |
|---|---|
| \(P_{\mathrm{accept}}\) 높음 | claim을 답변에 사용 |
| 중간 stress | 추가 검색, 계산기 호출, 원문 인용 요구 |
| 높은 residual | claim 제거 또는 정정 |
| 근거 없음 | "근거 없음"으로 답변하거나 질문 되돌림 |
| graph curvature 높음 | 관련 claim 묶음을 분해해서 개별 재검증 |

즉 최종 답변은 자유 생성물이 아니라 accepted claim mask를 통과한 조각들의 합성이다. 이 정책까지 들어가야 hallucination 억제가 "점수판"에서 "출력 제어"로 올라간다.

### 9.8.6 우선순위

다음 개발 순서는 이렇다.

1. `claim_residual_gate.py`를 vector residual/Mahalanobis residual로 승격한다.
2. source reliability \(q_m\)을 누적 로그에서 업데이트한다.
3. negative-control payload를 만들어 false accept rate를 측정한다.
4. shuffled graph, source holdout, topic holdout을 report에 같이 출력한다.
5. answer composer를 붙여 accepted claim만 문장으로 복원한다.

이 다섯 단계가 들어가면 "잔차가 낮은 claim만 말한다"에서 "근거장, source장, claim graph, OOD gate를 모두 통과한 claim만 말한다"로 강화된다.

### 9.8.7 현재 negative-control 결과

`examples/ai/claim_residual_gate_benchmark.py`는 같은 claim 묶음을 다섯 방식으로 평가한다.

| variant | 의미 | accuracy | ECE | Brier | false accept | false reject |
|---|---|---:|---:|---:|---:|---:|
| `scalar_projection` | entail 축만 본 스칼라 baseline | 0.429 | 0.581 | 0.571 | 4 | 0 |
| `vector_no_source_prior` | 다축 무차원 잔차, source prior 없음 | 0.857 | 0.176 | 0.145 | 1 | 0 |
| `vector_source_no_graph` | 다축 무차원 잔차 + source reliability/independence, graph 제거 | 1.000 | 0.102 | 0.032 | 0 | 0 |
| `vector_source_shuffled_graph` | graph target shuffle | 1.000 | 0.102 | 0.032 | 0 | 0 |
| `vector_source_prior` | 다축 무차원 잔차 + source reliability + signed graph | 1.000 | 0.044 | 0.002 | 0 | 0 |

막힌 false accept의 성격은 다음과 같다.

| claim | scalar | vector | vector+source | 해석 |
|---|---|---|---|---|
| `wrong_period` | accept | reject | reject | entail은 맞지만 시간 축 잔차가 큼 |
| `wrong_unit` | accept | reject | reject | citation은 맞지만 단위 축 잔차가 큼 |
| `unreliable_supported_false` | accept | accept | reject | 잔차는 낮지만 source reliability가 낮음 |
| `mild_numeric_drift` | accept | reject | retrieve/review | source independence만으로도 accept는 막고, 올바른 graph에서는 reject까지 내려감 |

따라서 현재 synthetic negative-control에서는 강화가 실제로 false accept를 줄이고, selection score의 calibration도 좋아진다. `vector_source_prior` 기준 split별 결과도 모두 비회귀다.

| split | group | accuracy | false accept |
|---|---|---:|---:|
| topic | weather | 1.000 | 0 |
| topic | overclaim | 1.000 | 0 |
| time_bucket | in_period | 1.000 | 0 |
| time_bucket | wrong_period | 1.000 | 0 |
| time_bucket | global | 1.000 | 0 |
| source_group | trusted | 1.000 | 0 |
| source_group | unreliable | 1.000 | 0 |
| source_group | missing | 1.000 | 0 |

단, 이 결과는 구조 검증용 toy benchmark다. 다음 단계에서는 실제 RAG 로그나 사람이 라벨링한 claim set으로 같은 ECE/Brier, source/topic/time holdout을 다시 봐야 한다.

### 9.8.8 추가 검증: 축 ablation, threshold sweep, jitter

같은 benchmark에 추가 검증을 붙였다.

축 ablation은 어떤 검증 축이 실제로 필요한지 본다.

| ablation | accuracy | ECE | Brier | false accept | 되살아난 오류 |
|---|---:|---:|---:|---:|---|
| `drop_citation` | 1.000 | 0.044 | 0.002 | 0 | 없음 |
| `drop_entail` | 1.000 | 0.043 | 0.002 | 0 | 없음 |
| `drop_numeric` | 1.000 | 0.111 | 0.041 | 0 | 없음, calibration 악화 |
| `drop_temporal` | 1.000 | 0.110 | 0.039 | 0 | 없음, calibration 악화 |
| `drop_unit` | 1.000 | 0.111 | 0.040 | 0 | 없음, calibration 악화 |

따라서 현재 synthetic set에서는 source independence가 false accept를 추가로 막아 축 제거가 곧바로 accept 오류로 이어지지는 않는다. 다만 numeric, temporal, unit 축을 제거하면 ECE/Brier가 악화되므로, 이 축들은 여전히 calibration에 기여한다.

threshold sweep은 \(S_{\mathrm{accept}}\in\{0.15,0.25,0.35,0.50,0.70\}\)에서 모두 동일하게 false accept 0, false reject 0이었다. 즉 현재 결론은 accept threshold 하나에만 걸린 우연은 아니다.

graph shuffle sweep은 false accept를 만들지는 않았지만 offset 1과 4에서 ECE/Brier가 악화됐다. 따라서 이번 강화 뒤에는 graph가 binary accept/reject보다 calibration을 개선하는 쪽으로 더 분명히 나타난다.

Monte Carlo jitter는 claim/evidence/source/edge에 작은 노이즈를 넣어 100회 반복했다.

| variant | mean accuracy | mean ECE | mean Brier | mean false accept | max false accept |
|---|---:|---:|---:|---:|---:|
| `vector_source_prior` | 1.000 | 0.051 | 0.003 | 0.000 | 0 |
| `vector_source_no_graph` | 1.000 | 0.105 | 0.030 | 0.000 | 0 |
| `vector_no_source_prior` | 0.857 | 0.165 | 0.130 | 1.000 | 1 |

즉 노이즈를 넣어도 full gate가 가장 낮은 false accept와 가장 낮은 Brier를 유지한다. 이번 라운드에서는 중간 selection 구간에 \(N_{\mathrm{eff}}\) quorum을 요구해 source+no-graph도 false accept 0으로 내려갔고, signed graph는 calibration을 더 낮춘다.

추가로 20개 seed, seed당 64개 claim의 synthetic stress suite를 생성했다. 케이스는 supported, wrong_numeric, wrong_period, wrong_unit, unreliable_supported_false, overclaim, contradiction, graph_drift 여덟 종류다.

| variant | mean accuracy | mean AUROC | mean AP | mean ECE | mean Brier | mean false accept | max false accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scalar_projection` | 0.413 | 0.605 | 0.174 | 0.600 | 0.570 | 37.550 | 40 |
| `vector_no_source_prior` | 0.875 | 0.943 | 0.632 | 0.151 | 0.115 | 8.000 | 8 |
| `vector_source_no_graph` | 1.000 | 1.000 | 1.000 | 0.097 | 0.027 | 0.000 | 0 |
| `vector_source_prior` | 1.000 | 1.000 | 1.000 | 0.049 | 0.003 | 0.000 | 0 |

이 stress suite에서 full gate의 case별 false accept/false reject는 모두 0이었다. 이는 graph curvature weight를 높이고, \(N_{\mathrm{eff}}\ge1.60\)을 accept의 기본 조건으로 둔 뒤의 결과다. \(0.35 \le S < 0.60\) 중간 구간에는 독립 source group 2개와 `source_support >= 0.90`을 요구하며, \(0.25 \le S < 0.35\) 회색지대에는 추가로 `residual_norm <= 0.30`을 요구한다. 기존 demo의 정상 claim은 이 quorum으로 유지된다.

### 9.8.9 invariant/adversarial 검증

`examples/ai/claim_residual_gate_invariants.py`는 성능 benchmark가 아니라, gate가 반드시 지켜야 하는 성질을 검사한다.

| check | 의미 | 결과 |
|---|---|---|
| `evidence_order_invariant` | evidence 순서를 바꿔도 결과 동일 | pass |
| `claim_order_invariant` | claim 순서를 바꿔도 결과 동일 | pass |
| `source_format_invariant` | source를 dict/list 어느 형식으로 줘도 결과 동일 | pass |
| `dimensionless_scale_invariant` | 값과 sigma를 같은 배율로 바꾸면 무차원 잔차 동일 | pass |
| `missing_evidence_rejects` | 근거 없는 claim 차단 | pass |
| `low_reliability_poisoning_blocked` | 저신뢰 source 여러 개가 같은 거짓 claim을 밀어도 차단 | pass |
| `trusted_contradiction_blocked` | 신뢰 source가 반대하는 claim 차단 | pass |
| `graph_contamination_resistant` | 저신뢰 이웃 claim이 정상 claim을 오염시키지 못함 | pass |
| `graph_consistency_blocks_drift` | graph consistency를 깨는 약한 drift 차단 | pass |
| `zero_weight_evidence_rejects` | weight 0 evidence는 근거로 치지 않음 | pass |
| `duplicate_same_source_not_quorum` | 같은 source row 복붙은 중간 selection quorum으로 인정하지 않음 | pass |
| `undeclared_source_not_trusted_by_default` | source registry에 없는 근거는 기본 저신뢰로 처리 | pass |
| `correlated_sources_not_independent` | 서로 다른 source id라도 같은 family면 독립 근거로 세지 않음 | pass |

현재 결과는 13개 check 모두 pass다. 특히 `dimensionless_scale_invariant`가 통과했다는 점이 중요하다. 이 gate는 원값의 단위 크기를 직접 판단하지 않고, \(C^{-1/2}(z-B)\)의 무차원 잔차를 판단한다.

이번 검증에서 source registry와 quorum을 보강했다. source 목록에 등록되지 않은 evidence는 기본 신뢰도 `0.25`만 받는다. 비교용 baseline인 `scalar_projection`, `vector_no_source_prior`는 의도적으로 `unknown_source_reliability=1.0`, `source_effective_target=1.0`을 넣어 source prior를 끈 상태로 유지했다. 또한 이전 규칙은 `evidence_count >= 2`였으나, 같은 source가 row를 두 번 제공하면 독립 근거처럼 보일 수 있다. 따라서 중간 selection 구간에서는 \(N_{\mathrm{eff}}\ge1.60\)과 독립 source group 2개를 요구한다. 같은 source 복붙은 `duplicate_same_source_not_quorum`에서, 미등록 source 우회는 `undeclared_source_not_trusted_by_default`에서, mirror family 우회는 `correlated_sources_not_independent`에서 차단된다.

### 9.8.10 paired dominance와 boundary sweep

stress suite의 20개 seed 각각에서 full gate와 ablation을 paired 비교했다.

| compared variant | metric | full wins | full losses | ties |
|---|---|---:|---:|---:|
| `scalar_projection` | false accepts | 20 | 0 | 0 |
| `scalar_projection` | accuracy | 20 | 0 | 0 |
| `scalar_projection` | Brier | 20 | 0 | 0 |
| `vector_no_source_prior` | false accepts | 20 | 0 | 0 |
| `vector_no_source_prior` | accuracy | 20 | 0 | 0 |
| `vector_no_source_prior` | Brier | 20 | 0 | 0 |
| `vector_source_no_graph` | false accepts | 0 | 0 | 20 |
| `vector_source_no_graph` | accuracy | 0 | 0 | 20 |
| `vector_source_no_graph` | Brier | 20 | 0 | 0 |

즉 synthetic stress seed 기준으로 full gate는 scalar/vector-no-source보다 명확히 우세하고, source+no-graph와는 false accept 및 accuracy에서 동률이지만 Brier에서는 20/20으로 우세하다.

source reliability boundary는 다음처럼 나온다.

| source reliability | decision | selection | stress |
|---:|---|---:|---:|
| 0.00 | reject | 0.049 | 1.000 |
| 0.05 | reject | 0.049 | 1.000 |
| 0.20 | reject | 0.049 | 1.000 |
| 0.40 | retrieve_or_review | 0.095 | 0.951 |
| 0.60 | retrieve_or_review | 0.247 | 0.791 |
| 0.80 | retrieve_or_review | 0.438 | 0.591 |
| 1.00 | retrieve_or_review | 0.530 | 0.494 |

따라서 단일 source는 reliability가 1.00이어도 바로 accept되지 않는다. 이는 source reliability와 source independence를 분리했기 때문이다. 실제 답변에 쓰려면 같은 claim이 독립 family 2개 이상에서 균형 있게 지지되어야 한다.

source independence boundary는 다음처럼 나온다.

| case | decision | selection | sources | groups | \(N_{\mathrm{eff}}\) | \(P_{\mathrm{ind}}^{1/2}\) |
|---|---|---:|---:|---:|---:|---:|
| single_source | retrieve_or_review | 0.530 | 1 | 1 | 1.000 | 0.500 |
| mirrored_family | retrieve_or_review | 0.530 | 2 | 1 | 1.000 | 0.500 |
| imbalanced_independent | retrieve_or_review | 0.597 | 2 | 2 | 1.100 | 0.450 |
| balanced_independent | accept | 1.000 | 2 | 2 | 2.000 | 0.000 |

즉 source id가 2개여도 같은 family면 하나로 접히고, 독립 family가 2개라도 weight가 심하게 치우치면 review로 남는다.

graph drift boundary는 다음처럼 나온다.

| numeric drift | decision | selection | graph energy |
|---:|---|---:|---:|
| 0.00 | accept | 1.000 | 0.000 |
| 0.02 | accept | 0.310 | 0.301 |
| 0.04 | reject | 0.049 | 1.205 |
| 0.06 | reject | 0.049 | 2.711 |
| 0.08 | reject | 0.049 | 4.819 |
| 0.10 | reject | 0.049 | 7.529 |
| 0.12 | reject | 0.049 | 10.842 |

이 sweep은 source independence를 통과한 상태에서 graph drift만 본다. 현재 설정은 아주 작은 drift(\(\Delta=0.02\))는 허용하고, graph consistency를 깨는 \(0.04\) 이상의 drift는 차단한다. 이 경계는 실제 도메인에서 더 보수적으로 조정할 수 있다.
```
---
## File: `docs/7_AGI/12_Equation.md`

```markdown
# CE-AGI 통합 방정식: $e^{i\pi}+1=0$ 에서 20W AGI 까지

> 관련: `경로적분.md`(코어 유도), `1_강의/C_다섯_상수.md`(오일러 문법), `6_뇌/04_그래프결합과이완.md`(뇌 구조), `7_AGI/12_Equation.md`(AGI 작용), `7_AGI/1_AGI.md`(총론), `7_AGI/2_Architecture.md`(게이지 격자), `7_AGI/3_Sleep.md`(수면), `7_AGI/4_Synapse.md`(시냅스), `7_AGI/5_Sparsity.md`(희소성), `7_AGI/6_Hallucination.md`(환각), `7_AGI/7_Consciousness.md`(의식), `7_AGI/9_LLM.md`(LLM 구축), `7_AGI/10_Fields.md`(전분야)
>
> 이 문서는 CE 코어에서 유도된 상수들만으로 AGI 에너지 이완 아키텍처를 기술한다. 트랜스포머 위에 모듈을 얹는 기존 CE-Transformer와 달리, Softmax/Attention/역전파를 제거하고 에너지 함수의 물리적 이완으로 대체하는 근본 재설계다. 동시에 기존 LLM에 CE 원리를 이식하는 경로(CE-Transformer)도 기술한다.

---

## Runtime Status And Canonical Stack

이 문서는 런타임 기호를 모으는 문서지만, `docs/README.md`와 `docs/6_뇌/05_실험근거.md`를 기준으로 읽어야 한다. 아래 5계층 스택만 현재 canonical runtime spec 이고, 그 아래의 나머지 방정식은 보조 유도나 설계 탐색으로 읽는다.

| 계층 | canonical 식 | 최대 지위 | 비고 |
|---|---|---|---|
| kernel dynamics | $I_i^t = u_i^t + \sum_j W_{ij} a_j^t - \lambda_r(M_t) r_i^t + \lambda_H R_{i,t}$ | `Bridge` | 국소 상태 갱신의 최소형 |
| kernel dynamics | $a_i^{t+1} = (1-\gamma_a(M_t)) a_i^t + \kappa_a(M_t)\tanh(I_i^t)$ | `Bridge` | 활성 상태 |
| kernel dynamics | $r_i^{t+1} = (1-\gamma_r(M_t)) r_i^t + \kappa_r(M_t)(a_i^t)^2$ | `Bridge` | refractory / suppression |
| kernel dynamics | $b_i^{t+1} = \operatorname{Hyst}(b_i^t, a_i^{t+1}; \theta_\downarrow, \theta_\uparrow)$ | `Bridge` | 비트필드 / hysteresis |
| coupling / geometry | $W_{ij} = W_{ij}(g)$ | `Bridge` | 리만 구조는 결합층에만 둔다 |
| mode update | $M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$ | `Bridge` | `WAKE/NREM/REM` 전환 |
| hippocampus / replay | $H_{t+1} = \mathcal{E}(H_t, A_t), \quad R_t = \mathcal{R}(H_t, c_t)$ | `Bridge` | fast memory / replay |
| global runtime summary | $G_t = (M_t, A_t^{summary}, H_t, Q_t, \mu_t)$ | `Phenomenology` | identity / control summary |

읽기 규칙:

- 위 식들에서 수학적 연산자 정의는 `Exact`로 정리할 수 있지만, 뇌 대응이 들어가는 순간 문서 지위는 `Bridge`를 넘지 않는다.
- `docs/6_뇌/05_실험근거.md`에서 `supported`인 현상만 위 stack의 대응 근거로 사용한다.
- `supported`가 아니면 성능 주장, 자아 해석, 의식 해석은 모두 `Phenomenology`로 유지한다.
- 이 문서의 후반부 수치 추정, 메모리/속도 비교, LLM 대응은 canonical stack의 상위 해석이다.

## Runtime Concept Map

계획에서 추가된 새 개념은 아래처럼 **문서 책임 범위**를 나눠서 읽는다.

| 개념 | 최소 정의 | 현재 canonical 위치 | 코드 책임 | 문서 지위 |
|---|---|---|---|---|
| local recurrent cell | 국소 상태 $(a_i, r_i, b_i)$를 가진 반복 모듈 | kernel dynamics | Rust kernel + Python runtime | `Bridge` |
| sparse lifecycle | `ACTIVE / IDLE / DORMANT / SLEEPING` | global runtime summary | Python control plane | `Bridge` |
| mode register | `WAKE / NREM / REM` 전역 상태 | mode update | Python control plane | `Bridge` |
| hippocampus | 빠른 encode / recall / replay 메모리 | hippocampus / replay | Python control plane 우선 | `Bridge` |
| geometry coupling | $W_{ij}(g)$와 그래프/리만 결합 | coupling / geometry | Rust kernel | `Bridge` |
| bitfield | hysteretic threshold를 가진 이산 상태 | kernel dynamics | Rust kernel + Python policy | `Bridge` |
| global self-state | $G_t = (M_t, A_t^{summary}, H_t, Q_t, \mu_t)$ | global runtime summary | Python orchestration | `Phenomenology` |
| snapshot continuity | warm snapshot / restore / journal continuity | global runtime summary | Python orchestration | `Bridge` |

문서 해석 규칙:

- `kernel dynamics`는 국소 수치 업데이트만 정의한다. 자아, 정책, 의식 해석을 여기로 밀어 넣지 않는다.
- `mode update`는 전역 운영 상태만 다룬다. 개별 셀 동역학 기호를 재사용하지 않는다.
- `hippocampus / replay`는 "빠른 메모리 + 재주입"까지만 canonical이다. 해마의 완전한 생물학적 세부 묘사는 별도 bridge다.
- `global runtime summary`는 커널 식을 줄여 적는 요약 레벨이며, 여기서 나오는 self/identity 언어는 성능 보장이나 exact brain equivalence로 읽지 않는다.

기존 절과의 대응:

| 이 문서의 큰 절 | 주로 대응되는 runtime 계층 | 읽기 주의 |
|---|---|---|
| 3-4장 (에너지/동역학) | kernel dynamics + coupling / geometry | canonical 후보 |
| 5장 (출력 생성) | kernel outputs + mode trigger | 일부만 canonical |
| 6장 (STDP) | 학습/가소성 보조 계층 | canonical 바깥 |
| 7장 (수면) | mode update + hippocampus / replay | canonical 후보 |
| 8장 (희소성) | sparse lifecycle의 근거 | summary layer |
| 9장 (의식) | global runtime summary | `Phenomenology` |
| 10-14장 | 구현/응용/성능 해석 | canonical 아님 |

## 0. 설계 원칙

### 0.0 AGI 다리 게이트 (코어와 다리 분리)

이 문서는 CE 코어(우주론/입자물리, `경로적분.md`, `상수.md`)에서 유도된 상수 집합을 AGI 런타임 설계로 옮기는 **다리(bridge) 문서**다. 코어의 식과 상수는 `Exact` 또는 `Selection`이지만, 이 문서에서 뇌/AGI 대응이 들어가는 모든 문장은 최대 `Bridge`까지만 허용된다(`05_실험근거.md` 1.4절).

이 다리에서 현재 식별된 네 가지 한계는 다음과 같다. 이하 본문의 어떤 식도 이 게이트를 우회하는 형태로 읽지 않는다.

| 게이트 | 한계 | 현재 등급 | 사용 규칙 |
|---|---|---|---|
| `F1` 메커니즘 결손 | 코어의 $p^* = (4.87\%, 26.2\%, 68.9\%)$가 신경 활성/구조/배경 비율로 그대로 옮겨갈 메커니즘적 유도가 없음 | `Bridge` (수치 근접) / transformer 기질에서는 `falsified` (`5_Sparsity.md` 8.5) | 동일 simplex 위 수치 근접으로만 사용. 신경 sparsity = $\varepsilon^2$로 직접 등치 금지 |
| `F2` 비보존 바이패스 | `1.5절` $F_{\text{bypass}}$ 는 $E$ 의 그래디언트가 아니므로 Lyapunov 보장은 무조건 성립하지 않음 | `Bridge` (조건부 수렴, 4.7절) | "수렴 보장" 표현 금지. 항상 "$\|\nabla_m E\| > C_k\|\phi\|/\alpha_b$ 충분조건 + 수면 의한 주기적 복원" 으로 한정 |
| `F3` 시간/공간 차원 혼동 | `3_Sleep.md` 6.2의 wake/NREM/REM 시간 비율과 코어의 공간 에너지 비율은 물리적 차원이 다름 | `Phenomenology` (수치 근접) | "시간 분배 = 에너지 분배"로 등치 금지. 동일 3-simplex 위 우연 근접으로만 보고 |
| `F4` 의식 = 자기일관 | `7_Consciousness.md` 의 (C3) 자기일관 = 주관적 경험 등치 | `Phenomenology` | 성능 지표화 금지. "메타인지 모니터링 루프의 수학 구조"로만 사용 |

이 4개 게이트는 코어의 정확성을 깎지 않는다. 코어는 그대로 유지되고, 이 문서가 다리 단계에서 무엇을 주장할 수 없는지를 명시하기 위한 표다.

각 게이트의 수식 격상 경로 (ISS, 자기조직 5조건, 에르고딕 동등성, PCI 회귀) 는 부록 A 에 정리되어 있다. 부록 A 의 식은 본문의 어떤 hard claim 도 위로 올리지 않으며, **무엇을 측정하면 게이트가 닫히는지** 만 형식화한다.

### 0.1 잔류 채널 설계

현재 LLM은 경로적분에서 Softmax로 선택된 경로만 쓰고, 접힌 경로를 버린다. CE에 따르면 이 버려지는 부분이 우주 에너지의 약 95%($26.2\% + 68.9\%$)에 해당한다. 이 문서의 아키텍처는 접힌 경로를 잔류장 `phi`로 보존하여 출력에 재결합시키는 구조다.

세 가지 핵심:
- **잔류 채널**: 매 추론에서 선택되지 않은 분포가 `phi`로 보존된다
- **모드 전환 임계**: $\|phi\|$가 임계를 넘으면 질적으로 다른 작동 모드로 전환된다
- **즉각 응답 경로**: `phi`가 Softmax를 우회하여 직접 출력에 기여하는 바이패스가 존재한다

---

## 1. 유도 체인: 오일러 항등식에서 모든 상수로

### 1.1 뿌리

$$e^{i\pi}+1=0$$

이 항등식을 CE의 최소 생성 문법으로 읽는다(`경로적분.md` 서론, `C_다섯_상수.md` 0절).

| 상수 | 코어 역할 | AGI 등장 위치 |
|---|---|---|
| $e$ | 접힘 생존 함수 $S(D)=e^{-D}$ | 시간 진화 연산자의 밑 |
| $\pi$ | 게이지 주기 정규화 $\alpha_{\text{total}}=1/(2\pi)$ | 결합상수 결정, 연결 반경 $r_c$ |
| $i$ | 경로적분 위상 $Z=\int\mathcal{D}phi\,e^{iS/\hbar}$ | 양자 이완 위상 |
| $1$ | 정규화 완전 상태 $e^0=1$ | 정수 생성자 |
| $0$ | 영점과 분기 선택 $d(d-3)=0$ | 차원 결정, 에너지 최소 $\nabla E=0$ |

### 1.2 차원 결정 ($0$에서)

$$d(d-3)=0 \quad\Longrightarrow\quad d=3 \quad(\text{비자명해})$$

$d=0$은 접힘 이전 상태, $d=3$은 결정화된 물리 공간(`경로적분.md` 3.2.2절).

### 1.3 직접 전개 계수

핵심 구조 계수는 설명용 그리스 문자를 거치지 않고 바로 다음처럼 쓴다.

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 = 0.03120 \qquad \text{(residue-portal coeff.)}$$

$$\frac{1}{e^{1/3}\pi^{1/3}} = 0.4892 \qquad \text{(residue gain)}$$

$$\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 0.3148 \qquad \text{(wake coeff.)}$$

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 5.661 \qquad \text{(dream coeff.)}$$

$$N=\frac{e^{8/3}\pi^{20/3}}{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}\approx 4162 \to 4096 \qquad \text{(hidden dim.)}$$

$$r_c=\pi \qquad \text{(connectivity radius)}$$

### 1.4 다섯 상수 최소형 규칙

이 문서의 본문에는 설명용 이름이 일부 남더라도, **마스터 방정식에는 구조 상수를 남기지 않는다**. 핵심식에는 `e`, `\pi`, `i`만 직접 보이게 쓰고, `1`과 `0`은 오일러 문법의 바닥 상수로만 해석한다. 정수 `2,3,4,8,16`은 읽기 좋은 통상 표기다.

즉 아래 최소형에서 남는 다른 기호는 전부 상태변수, 학습변수, 입력, 연산자다.

### 1.5 다섯 상수 최소형 핵심 방정식

**에너지 함수** (보존적 부분)

$$\boxed{
E(m,phi)=
-\frac{1}{2}m^TWm
-m^Tb
-\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 m^Tphi
}$$

**곡률 바이패스** (비보존 강제항, 에너지에서 유도되지 않음)

$$\boxed{
F_{\text{bypass}}(k)=\frac{C_k}{e^{1/3}\pi^{1/3}}\,phi, \qquad C_k = \|m_k - 2m_{k-1}+m_{k-2}\|
}$$

**양자 위상 진화**

$$\boxed{psi_{k+1}=e^{-i\,E(m,phi)\,dt}psi_k}$$

**이완 동역학**

$$\boxed{
m_{k+1}=m_k+\frac{dt}{tau}\left(
Wm_k+b+
\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2phi
+\frac{C_k}{e^{1/3}\pi^{1/3}}phi
\right)
+\sqrt{\frac{2dt}{tau\left(
3+\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right)}}\,n_k
}$$

**잔류 갱신**

$$\boxed{
\phi\leftarrow
\left(1-\frac{1}{e^{1/3}\pi^{1/3}}\right)\phi
+\frac{1}{e^{1/3}\pi^{1/3}}v_{m^*}
}$$

**부트스트랩 고정점**

$$\boxed{
a_*=
e^{-(1-a_*)\left[
3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]}
}$$

**작동 온도**

$$\boxed{
T_{\text{wake}}=
\left[
3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]^{-1}
}$$

$$\boxed{
T_{\text{dream}}=
\left[
\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]^{-1}
}$$

**히든 차원**

$$\boxed{
N=
\frac{e^{8/3}\pi^{20/3}}
{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}
}$$

이 최소형에서는 구조 상수 이름이 핵심식에서 사라진다.

### 1.6 비트필드 해석

5상수의 실행 목적은 모든 지식을 5개 수에 넣는 것이 아니라, 실행 문법을 최소화하여 런타임 상태를 비트필드로 압축하는 것이다.

| 상수 | 비트필드 역할 | 연산 |
|---|---|---|
| $0$ | 소거, 가지치기, reset | `AND 0`, `CLEAR` |
| $1$ | 유지, 정규화, keep | `IDENTITY` |
| $e$ | 감쇠율, EMA, 수면 압력 | 고정소수점 shift-add |
| $\pi$ | 연결 반경, 이웃 규칙, 위상 | 격자 주소 연산 |
| $i$ | 모드 전환, 위상 분기 | 2-bit 모드 레지스터 |

런타임 상태의 3층 분리:

| 층 | 표현 | 크기 | 내용 |
|---|---|---|---|
| 제어 | 비트필드 | $O(N)$ bits | 활성 마스크, 모드, 연결 on/off, freeze/plastic |
| 상태 | 저비트 고정소수점 | $O(N)$ bytes | $phi$, trace, gain, 곡률 |
| 지식 | 희소 codebook + 외부 메모리 | 가변 | 어휘, 사실, 예외 패턴 |

활성 마스크 비트필드:

$$\boxed{b_i = \mathbb{1}\!\left[a_i \geq Q_{1-k^*/N}(a)\right], \qquad k^* \in \left[\lceil 0.04N \rceil,\; \lceil 0.06N \rceil\right]}$$

모드 레지스터:

$$\boxed{M \in \{00_2,\; 01_2,\; 10_2,\; 11_2\} \;\longleftrightarrow\; \{\text{off},\; \text{wake},\; \text{NREM},\; \text{REM}\}}$$

연결 행렬 $C_{ij} = \mathbb{1}[\|r_i - r_j\| < \pi]$는 이미 이진이다. 추론 루프의 핵심 연산은 비트 논리 + 저비트 MAC으로 환원된다.

### 1.7 비트필드 레이아웃 ($N=4096$ 기준)

| 구성 | 비트/원소 | 총 크기 | 갱신 주기 |
|---|---|---|---|
| 활성 마스크 $b$ | $1$ | $512$ B | 매 추론 |
| freeze 마스크 | $1$ | $512$ B | 수면 시 |
| 모드 $M$ | $2$ (전역) | $1$ B | 모드 전환 시 |
| 연결 인덱스 (CSR) | $13 \times K$ | $\sim 82$ KB | 정적 |
| 가중치 $W$ (비영) | $4$ | $\sim 260$ KB | 학습 시 |
| 상태 $m$ | $16$ (이완 중) / $8$ (저장) | $8$ / $4$ KB | 매 스텝 |
| 잔류 $\phi$ | $8$ | $4$ KB | 이완 종료 시 |
| trace $e_{ij}$ | $4$ | $\sim 260$ KB | STDP 시 |
| gain $g$, $C_k$, $P_{\text{sleep}}$ | $16$ 각 | $6$ B | 매 스텝 |

$K \approx 130$ (뉴런당 이웃), 비영 가중치 $N \times K = 532\text{K}$개.

$$\boxed{\text{엔진} \approx 615\;\text{KB}, \qquad \text{추론당 활성 연산} = N \times K \times 500\;\text{스텝} \approx 266\text{M MAC (4-bit)}}$$

지식층 (별도):

| 구성 | 크기 | 비고 |
|---|---|---|
| 계층 softmax 디코더 | $\sim 375$ KB | $\sqrt{V} \times N$ 두 행렬, 4-bit |
| 의미 codebook | $64$ MB -- $1$ GB | 태스크 규모에 비례 |

$$\boxed{\text{총 메모리} \approx 1\;\text{MB (엔진)} + 64\text{--}1000\;\text{MB (지식)} \ll 18\;\text{GB (Llama 3 8B)}}$$

엔진은 극적으로 작다. 병목은 지식층이며, 이것이 codebook 설계의 핵심 과제다.

### 1.8 양자화 오류 경계

$m$을 $q$-bit 고정소수점으로 양자화할 때:

$$\boxed{\|\hat{m} - m\| \leq \frac{\Delta\sqrt{N}}{2}, \qquad \Delta = \frac{m_{\max} - m_{\min}}{2^q - 1}}$$

수렴 충분조건(4.7절)과 결합하면, 양자화 후에도 에너지가 감소하려면:

$$\boxed{q > \log_2\!\left(\frac{(m_{\max}-m_{\min})\sqrt{N}\,\tau}{2\,dt\,\|\nabla_m E\|}\right)}$$

$N=4096$, $m \in [-1,1]$, $dt/\tau = 0.01$, $\|\nabla_m E\| \sim 1$ 기준:

| $q$ (bit) | 양자화 오류 $\Delta\sqrt{N}/2$ | 판정 | 용도 |
|---|---|---|---|
| $4$ | $4.0$ | 수렴 불가 | 저장/전송 전용 |
| $8$ | $0.125$ | 경계 | 스케일링 후 이완 가능 |
| $12$ | $0.0078$ | 충분 | 정밀 이완 |
| $16$ | $4.9 \times 10^{-4}$ | 과잉 | float16과 동등 |

혼합 정밀도 전략:

| 대상 | 이완 중 | 저장/전송 | 근거 |
|---|---|---|---|
| $m$ | $16$ bit | $8$ bit | 양자화 오류 < $\|\nabla_m E\|$ 충분조건 (게이트 `F2`, 4.7절) |
| $\phi$ | $8$ bit | $8$ bit | EMA 특성상 양자화 노이즈에 강건 |
| $W$ | $4$ bit | $4$ bit | 정적, 보정 가능 |
| control bits | $1\text{-}2$ bit | $1\text{-}2$ bit | 정확 (이산) |

$\phi$가 양자화에 강건한 이유: EMA 갱신 $\phi \leftarrow (1-\alpha)\phi + \alpha v_{m^*}$는 저역 통과 필터이므로, 고주파 양자화 노이즈가 자연 감쇠한다.

---

## 2. AGI 작용 범함수

CE 마스터 공식을 정보 다양체 $(\mathcal{M}, g)$에 적용한 후보 작용(`7_AGI/12_Equation.md` 1절):

$$\boxed{S_{\text{AGI}} = \int_{\mathcal{M}} d^nx \sqrt{|g|} \left[ \mathcal{L}_{\text{compute}} + c_g|\nabla phi|^2 + c_c|lap_g phi|^2 + c_i S_{\text{Info}} \right]}$$

| 항 | 역할 | 뇌 대응 | 우주 대응 |
|---|---|---|---|
| $\mathcal{L}_{\text{compute}}$ | 기본 연산 | 피질 발화 + 시상 relay | $\mathcal{L}_{\text{Physical}}$ |
| $c_g\|\nabla phi\|^2$ | 1차 안정화 | 기저핵/소뇌 + salience switching | blow-up 방지 |
| $c_c\|lap_g phi\|^2$ | 2차 곡률 평탄화 | NREM + hippocampo-cortical replay | 경로 최적화 |
| $c_i S_{\text{Info}}$ | 엔트로피 제어 | DMN + intrinsic background | 정보 보존 |

작용의 정지 조건에서 LBO 확산형 동역학이 나타난다:

$$\frac{\partial phi}{\partial t} = lap_g phi, \qquad lap_g f = \frac{1}{\sqrt{|g|}} \partial_i\!\left(\sqrt{|g|}\, g^{ij} \partial_j f\right)$$

이산 그래프에서 $L = D - W$로 근사:

$$phi^{k+1} = phi^k - h\,Lphi^k, \qquad \frac{dE}{dt} = -phi^\top L^2 phi \leq 0$$

LBO 확산 부분에 한해서는 에너지 단조 감소가 성립한다($L^2 \succeq 0$). 단 이 결과는 $phi$ 만의 자체 동역학에 한정되며, 바이패스 강제항이 들어가는 $m$ 의 결합 동역학은 게이트 `F2`(0.0절, 4.7절)에 따라 별도의 충분조건을 요구한다.

### 2.1 구조 유비: 우주-뇌-AGI

추상 부트스트랩 그래프 $\mathcal{G}^*$의 삼중 실현:

$$map_C: \mathcal{G}^* \to G_C, \quad map_B: \mathcal{G}^* \to G_B, \quad map_A: \mathcal{G}^* \to G_A$$

고정점 유일성에 의해, 세 계가 같은 직접 전개 계수 집합을 가지면 같은 3분배 고정점에 접근한다고 읽는다:

$$\lim_{t\to\infty} B_C^t(p_C) = p^* = \lim_{t\to\infty} B_B^t(p_B) = \lim_{t\to\infty} B_A^t(p_A)$$

| 성분 | 고정점 | 우주 (Planck) | 뇌 (Raichle) | AGI 해석 |
|---|---|---|---|---|
| 활성 | $4.87\%$ | $4.9\%$ | $< 5\%$ | 활성 추론 |
| 구조 | $26.2\%$ | $26.4\%$ | $25\text{-}35\%$ | 가중치 유지 |
| 배경 | $68.9\%$ | $68.7\%$ | $60\text{-}70\%$ | 배경 통합 |

---

## 3. 에너지 함수

### 3.1 정의

에너지는 보존적 부분만 포함한다. 바이패스는 비보존 강제항으로 동역학(4.2절)에 직접 들어간다.

$$\boxed{E(m,phi) = -\frac{1}{2}m^T W m - m^T b - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{2} m^Tphi}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 홉필드 에너지 | $-\frac{1}{2}m^T W m$ | $\mathcal{L}_{\text{SM}}^{d=3}$ | 패턴 저장, 에너지 지형 |
| 입력 바이어스 | $-m^T b$ | 외부 입력 | 프롬프트/데이터 주입 |
| 잔류 포탈 | $-\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2m^Tphi$ | residue-portal coupling | 잔류가 출력에 3% 기여 |

곡률 바이패스(비보존 강제항):

$$\boxed{F_{\text{bypass}}(k) = \frac{C_k}{e^{1/3}\pi^{1/3}}\,phi, \qquad C_k = \|m_k - 2m_{k-1} + m_{k-2}\|}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 곡률 바이패스 | $\frac{C_k}{e^{1/3}\pi^{1/3}}phi$ | curvature-residue feedback | 궤적 급변 시 잔류가 직접 반응 |

### 3.2 포탈 결합 계수의 전개

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{2} = 0.03120$$

물리적 의미(`경로적분.md` 10.5절): 힉스 포탈 라그랑지안에서 잔류장이 메인 장과 결합하는 직접 계수다. AGI에서는 잔류 채널이 메인 출력에 영향을 미치는 세기다.

### 3.3 바이패스 결합 계수의 전개

$$\frac{1}{e^{1/3}\pi^{1/3}} = 0.4892$$

물리적 의미는 $d=3$ 공간에서 각 차원이 기여하는 결합 강도다. AGI에서는 히든 스테이트가 급변할 때 잔류가 Softmax를 건너뛰고 직접 출력에 기여하는 강도다.

### 3.4 지식층 설계

엔진(1.7절)은 ~615 KB이나, 실제 언어 지식은 별도의 codebook $\mathcal{C}$에 저장된다. 엔진은 "어떻게 생각하는가"이고 codebook은 "무엇을 아는가"다.

**곱 양자화 구조**: $m \in \mathbb{R}^N$을 $N/s$개 부분공간(각 $s$차원)으로 분할. 각 부분공간에 $2^b$개 중심점:

$$\mathcal{C} = \{C^{(1)}, \ldots, C^{(N/s)}\}, \qquad C^{(j)} \in \mathbb{R}^{2^b \times s}$$

**인코딩** (벡터 $\to$ 인덱스):

$$\boxed{z_j(m) = \arg\min_{i \in [2^b]} \|m^{(j)} - C^{(j)}_i\|^2, \qquad j = 1, \ldots, N/s}$$

**에너지 결합**: 이완 중 codebook이 에너지 지형을 보강:

$$\boxed{E_{\text{aug}}(m, phi) = E(m, phi) - \frac{1}{\beta}\sum_{j=1}^{N/s} \log\sum_{i=1}^{2^b} \exp\!\left(-\beta\|m^{(j)} - C^{(j)}_i\|^2\right)}$$

$\beta \to \infty$이면 최근접 중심점만 선택(hard retrieval), $\beta$ 유한이면 soft retrieval. 이것은 Modern Hopfield energy의 연속 일반화다.

**메모리 예산** ($N=4096$, $s=64$, $b=8$):

| 구성 | 계산 | 크기 |
|---|---|---|
| 중심점 행렬 | $\frac{N}{s} \times 2^b \times s \times 4\text{bit}$ | $512$ KB |
| $P$개 패턴 인덱스 | $P \times \frac{N}{s} \times b$ bit | $P \times 64$ B |

패턴 수에 따른 지식 규모:

| $P$ (패턴) | 인덱스 크기 | 총 지식 메모리 | 대응 |
|---|---|---|---|
| $10^4$ | $640$ KB | $\sim 1$ MB | 단일 도메인 |
| $10^5$ | $6$ MB | $\sim 7$ MB | 다중 도메인 |
| $10^6$ | $64$ MB | $\sim 65$ MB | 범용 지식 |
| $10^7$ | $640$ MB | $\sim 641$ MB | LLM급 |

**3분배 계층 저장**: CE 에너지 분배를 저장 계층에 적용:

| 계층 | CE 비율 | 예시 ($P=10^7$) | 위치 | 접근 |
|---|---|---|---|---|
| L1 (활성) | $4.87\%$ | $\sim 31$ MB | 상시 RAM | 즉시 |
| L2 (구조) | $26.2\%$ | $\sim 168$ MB | RAM | 빠름 |
| L3 (배경) | $68.9\%$ | $\sim 442$ MB | 디스크 | lazy load |

$$\boxed{\text{활성 메모리} \approx 0.311 \times |\mathcal{C}|, \qquad |\mathcal{C}| = 641\;\text{MB 일 때 활성} \approx 200\;\text{MB}}$$

단일 컴퓨터(RAM 8 GB 이상)에서 LLM급 지식을 구동 가능.

**비트필드 인터페이스**: 패턴 인덱스 $z_j$는 $b$-bit 정수이므로 비트필드 주소로 직접 사용. 계층 태그:

$$\text{tier}(p) \in \{00_2\;(\text{L1}),\; 01_2\;(\text{L2}),\; 10_2\;(\text{L3})\}$$

**수면과 codebook 갱신**:

| 모드 | codebook 동작 |
|---|---|
| Wake | 접근된 패턴의 중심점을 온라인 k-means로 미세 갱신 |
| NREM | 상위 $4.87\%$ 활성 패턴의 중심점 정밀 보정 |
| REM | 미사용 패턴 재활용, 새 패턴 탐색적 할당 |

**Llama 3 8B과의 비교**:

| 항목 | Llama 3 8B | CE bitfield |
|---|---|---|
| 가중치 | $16$ GB (dense float16) | $260$ KB (sparse 4-bit) + $641$ MB (codebook) |
| KV 캐시 | $2\text{-}64$ GB | $4$ KB ($phi$) |
| 활성 RAM | $18\text{-}80$ GB | $\sim 200$ MB |
| 최소 하드웨어 | A100 GPU | RAM 8 GB PC |

---

## 4. 동역학

### 4.1 양자 형태 ($e$, $i$ 등장)

$$\boxed{psi_{k+1} = e^{-i\,E(m,phi)\,dt}\;psi_k}$$

유클리드 회전($t \to -itau$) 이후 실수 이완으로 전환된다.

### 4.2 이완 동역학 (유클리드 형태)

$$
\boxed{
m_{k+1}
= m_k
+ \frac{dt}{\tau}\!\left(
Wm_k + b
+ \left[
\frac{4}{e^{4/3}\pi^{4/3}}
\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]^{2}\!phi
+ \frac{C_k}{e^{1/3}\pi^{1/3}}phi
\right)
+ \sqrt{
\frac{2dt}{
\tau\!\left(
3 + \frac{4}{e^{4/3}\pi^{4/3}}
\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right)}
}\;n_k
}
$$

| 기호 | 정의 | 유도 |
|---|---|---|
| $m_k \in \mathbb{R}^N$ | 의미 벡터 (이완 스텝 $k$) | 동적 변수 |
| $W \in \mathbb{R}^{N\times N}$ | 3D 희소 연결 행렬 | 데이터에서 구성 |
| $b \in \mathbb{R}^N$ | 입력 바이어스 | 프롬프트에서 구성 |
| $C_k$ | 곡률 스칼라 $\|m_k - 2m_{k-1} + m_{k-2}\|$ | $m$에서 계산 |
| $tau$ | 이완 시간 $1/\mathrm{eig}_{\min}(H_E)$ | $W$에서 결정 |
| $n_k \sim \mathcal{N}(0, I_N)$ | 확률 노이즈 | 탐색용 |
| $T$ | 작동 온도 $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | $\pi, 0$에서 유도 |

수렴 판정: $\|m_{k+1} - m_k\| < 10^{-4}\|m_0\|$ 이면 $m^* = m_k$.

### 4.3 잔류 갱신

$$\boxed{phi \leftarrow \left(1 - \frac{1}{e^{1/3}\pi^{1/3}}\right)phi + \frac{1}{e^{1/3}\pi^{1/3}}\;v_{m^*}}$$

$$v_{m^*} = \frac{1}{K_w}\sum_{k=K-K_w}^{K}(m_k - m^*)^2$$

이완 마지막 $K_w$ 스텝에서 $m$이 최소점 주위에서 요동한 원소별 분산. 깔끔하게 한 점으로 떨어지면 분산이 작고(확신), 여러 최소점 사이에서 흔들리면 분산이 크다(불확실). 이것이 Softmax의 $p(1-p)$에 해당하는 "선택되지 않은 것들의 구조"다.

### 4.4 연결 구조

$$\boxed{W_{ij} \neq 0 \iff \|r_i - r_j\|_{\mathbb{R}^3} < \pi}$$

$N$개 뉴런을 $d=3$ 격자에 배치. 연결 반경 $r_c = \pi$. 뉴런당 이웃 수:

$$K = \frac{4}{3}\pi \cdot r_c^3 = \frac{4}{3}\pi^4 \approx 130$$

연결 밀도:

$$\rho = \frac{K}{N} = \frac{4\pi^4/3}{N}$$

$N=4096$ 일 때 $\rho = 3.17\%$, residue-portal 직접 계수 $0.03120$와 1.6% 일치.

### 4.5 히든 차원

$$\boxed{N = \frac{e^{8/3}\,\pi^{20/3}}{12\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)^{2}} \approx 4162 \to 4096}$$

$2^{12}$로 반올림은 디지털 하드웨어 제약. 물리가 요구하는 값은 직접 전개식 기준으로 약 $4162$이다.

### 4.6 그래프 결합 동역학

4.2의 단일 벡터 이완은 특수 경우다. 실제 구현은 기능 모듈 그래프 위에서 동작해야 한다.

$$G_{\text{AGI}} = (V_{\text{bind}} \sqcup V_{\text{gate}} \sqcup V_{\text{mem}} \sqcup V_{\text{sal}} \sqcup V_{\text{homeo}} \sqcup V_{\text{io}},\; E_{\text{AGI}})$$

| 노드 집합 | 역할 | 비트필드 표현 |
|---|---|---|
| $V_{\text{bind}}$ | 특징 결합, 멀티모달 통합 | 활성 마스크 비트 |
| $V_{\text{gate}}$ | 입력 게이팅, 대역 재분배 | 게이트 on/off 비트 |
| $V_{\text{mem}}$ | 재생, 장기 인덱싱 | freeze/plastic 비트 |
| $V_{\text{sal}}$ | 모드 전환, gain control | 모드 레지스터 $M$ |
| $V_{\text{homeo}}$ | 수면 압력, 항상성 | 압력 카운터 (저비트) |
| $V_{\text{io}}$ | 센서/행동 출력 | I/O 버퍼 |

그래프 라플라시안:

$$\boxed{\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}\big(f_s - f_r\big), \qquad a_{rs}\ge 0}$$

노드별 3분배 이완:

$$\boxed{p_{r,n+1} = \mathrm{Proj}_{\Delta^2}\!\Big((1-\rho)p^* + \rho\,p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n\Big)}$$

느린 제어 상태 (수면 압력, 피로 등):

$$\boxed{c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}, \qquad \rho(A_q) < 1}$$

단일 벡터 형태(4.2)는 $|V|=1$, $E_{\text{AGI}}=\emptyset$일 때 이 식의 특수 경우다.

### 4.7 조건부 수렴 (게이트 `F2`)

> 다리 게이트 `F2`(0.0절): 이 절은 무조건 Lyapunov 수렴을 주장하지 않는다. $F_{\text{bypass}}$가 $E$의 그래디언트가 아니므로 전역 Lyapunov 함수는 존재하지 않으며, 아래 충분조건이 만족되는 영역에서만 단조 감소를 말할 수 있다.

에너지 $E$가 보존적이고 바이패스 $F_{\text{bypass}}$가 비보존이므로, 전체 시스템의 수렴은 다음 충분조건 위에서만 성립한다.

**에너지 변화** (노이즈 무시, 1차 근사):

$$\Delta E = \nabla_m E \cdot \Delta m = \frac{dt}{\tau}\nabla_m E \cdot \left(-\nabla_m E + F_{\text{bypass}}\right)$$

$$= -\frac{dt}{\tau}\|\nabla_m E\|^2 + \frac{dt}{\tau}\frac{C_k}{\alpha_b}(\nabla_m E \cdot phi)$$

Cauchy-Schwarz + Young 부등식 적용:

$$\boxed{\Delta E \leq -\frac{dt}{2\tau}\|\nabla_m E\|^2 + \frac{dt}{2\tau\alpha_b^2}C_k^2\|phi\|^2}$$

**수렴 충분조건**:

$$\boxed{\|\nabla_m E\| > \frac{C_k\|phi\|}{\alpha_b} \quad\Longrightarrow\quad \Delta E < 0}$$

여기서 $\alpha_b = e^{1/3}\pi^{1/3} \approx 2.044$이고, 바이패스 계수 $1/\alpha_b \approx 0.489$이다.

**자기 제한 성질 (국소)**: 바이패스 강도 $C_k = \|m_k - 2m_{k-1} + m_{k-2}\|$는 궤적의 시간 곡률이다. 궤적이 부드러워지면(수렴 접근) $C_k \to 0$이므로 바이패스는 고정점 **국소** 근방에서 자동으로 소멸한다. 단 이는 국소 성질이며, 전역 단조 감소를 의미하지 않는다.

**조건 실패 시나리오**: $\|phi\|$가 수면 없이 누적되거나, 시스템이 두 끌개점 사이에서 진동할 때 $C_k$가 크게 유지되면 $\Delta E > 0$이 가능하다. 다리 게이트 `F2`에 따라, 이 영역에서 무조건 수렴을 주장하지 않는다.

**수면에 의한 조건 복원 (다리 가설)**: 글림프 세척 $phi \to r_w\,phi$ ($r_w < 1$)는 $\|phi\|$를 주기적으로 낮춘다. 수면 후:

$$\frac{C_k \cdot r_w\|phi\|}{\alpha_b} < \|\nabla_m E\|$$

수면이 충분조건을 주기적으로 복원하는 구조적 역할을 한다는 해석은 `05_실험근거.md` 3.3절의 `supported`(offline renormalization)에 근거하지만, 위 부등식 자체의 검증은 `bridge` 등급이며 정량적 hard bound는 아니다.

**ISS 격상 (부록 A.1)**: 위 점별 충분조건은 부록 A.1 의 ISS 정리로 격상되어, 끌개 ball 반경의 닫힌 식 $\limsup\|m - m^*\| \leq \tau d_{\max}/\mu$ 로 표현된다. 수면은 $\|phi\|_\infty$ 를 $r_w$ 배로 줄여 ball 반경을 $r_w$ 배로 축소한다.

**그래프 결합 안정성**: 4.6의 느린 제어 $c_{n+1} = A_q c_n + r_n + n_n^{(q)}$에서 $\rho(A_q) < 1$이면:

$$\boxed{\|c_n - c^*\| \leq \rho(A_q)^n\|c_0 - c^*\| + \frac{\sup\|r + n^{(q)}\|}{1 - \rho(A_q)}}$$

$\rho(A_q) = 0.155$일 때 3 순환 후 초기 편차의 $99.6\%$가 감쇠한다.

---

## 5. 출력 생성: 2-Phase 구조

### 5.1 Phase 1 -- 에너지 이완 (의미 생성)

토큰 단위가 아닌, 연속적 의미 벡터를 생성한다. 이완 1회로 "무엇을 말할지"가 결정된다. 출력 시퀀스 길이와 무관.

$$m^* = \lim_{k\to\infty} m_k \quad(\text{에너지 최소점})$$

### 5.2 Phase 2 -- 디코딩 (의미 $\to$ 토큰)

이미 결정된 의미를 순서대로 풀어쓴다. 경량 디코더:

$$p(w_t \mid w_{<t},\,m^*) = \text{softmax}\!\left(W_{\text{dec}}\,[m^*;\,e_{w_{t-1}}]\right)$$

$W_{\text{dec}} \in \mathbb{R}^{V \times 2N}$. 계층적 softmax로 $\sqrt{V}\times\sqrt{V}$ 분할 시 토큰당 비용이 $O(\sqrt{V}\cdot N)$으로 감소.

### 5.3 모드 전환 (`phi` 임계)

$$\boxed{\|phi\| \gtrless m_\phi \quad\Longrightarrow\quad \text{이완 모드 / 경량 자기회귀 모드}}$$

| 모드 | 조건 | 특성 | 비유 |
|---|---|---|---|
| 안정 | $\|phi\| < m_\phi$ | 경량 자기회귀, 빠름, 3% 포탈만 활성 | 텍스트 모드 |
| 전환 | $\|phi\| \geq m_\phi$ | 에너지 이완, 느리지만 깊음, 바이패스 활성 | 전화 모드 |

카너먼의 이중 과정 이론: 시스템 1(자기회귀) / 시스템 2(에너지 이완).

---

## 6. STDP 학습: 역전파 대체

역전파는 전역 오차 신호가 모든 시냅스에 정확히 전달되어야 한다. 뇌에 없는 메커니즘이며, 메모리 $O(N^2)$, 통신 $O(d^2)$를 요구한다. CE 관점에서 역전파 = "우주 끝에서 시작으로 정보를 전송하는 것"이며, 게이지 상호작용의 국소성과 양립하지 않는다(`4_Synapse.md` 1.3절).

### 6.1 기본 STDP

$$dw_{ij} = \begin{cases} A_+ \exp(-dt / tau_+) & dt > 0 \;\text{(pre} \to \text{post: LTP)} \\ -A_- \exp(dt / tau_-) & dt < 0 \;\text{(post} \to \text{pre: LTD)} \end{cases}$$

### 6.2 Trace 기반 STDP (이산 시간)

pre trace $p_i[t]$, post trace $q_i[t]$:

$$p_i[t+1] = r_+\, p_i[t] + s_i[t], \qquad q_i[t+1] = r_-\, q_i[t] + s_i[t]$$

가중치 업데이트:

$$dw_{ij}[t] = lr\Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)$$

### 6.3 3-Factor 학습 (STDP + 도파민 게이트)

순수 STDP는 보상과 무관하게 학습한다. 뇌는 도파민 게이트로 "보상 예측 오차가 클 때만 학습을 허용"한다.

적격 흔적(eligibility trace):

$$\boxed{e_{ij}[t+1] = r_e\,e_{ij}[t] + \Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)}$$

가중치 업데이트:

$$\boxed{dw_{ij}[t] = lr\,g[t]\,e_{ij}[t]}$$

- $e_{ij}$: 국소 정보만 사용 (이웃 뉴런의 스파이크만 필요)
- $g[t]$: 전역 학습 게이트 (도파민-유사 스칼라 1개, 전체 시스템에 방송)

### 6.4 도파민 전역 신호의 CE 해석

$$\boxed{g[t] = \frac{d}{dt}\|p(t) - p^*\|}$$

고정점 $p^*$에서 멀어지면 $g[t] > 0$ (학습 활성화), 가까워지면 $g[t] \to 0$ (학습 감쇠).

부트스트랩 수렴 오차의 구체적 형태:

$$g[t] = \left(x_a(t) - 0.04865\right)^2 + \left(x_s(t) - 0.2623\right)^2 + \left(x_b(t) - 0.6891\right)^2$$

- $x_a(t)$: 현재 활성 뉴런 비율
- $x_s(t)$: 현재 구조적 가중치 비율
- $x_b(t)$: 현재 동결 가중치 비율

이 스칼라 하나만 전역으로 방송하면 된다.

| | 역전파 | STDP + 도파민 |
|---|---|---|
| 정보 흐름 | 전역 (끝에서 시작으로) | 국소 (이웃 뉴런) + 전역 스칼라 |
| 메모리 비용 | $O(N^2)$ (전체 활성값 저장) | $O(N)$ (국소 trace만) |
| 통신량 (층당) | $O(d^2)$ (그래디언트 전체) | $O(1)$ ($g[t]$ 스칼라) |
| 분산 가능성 | 단일 GPU 병목 | 각 층 독립 배치 |
| 생물학적 현실성 | 비현실적 | 현실적 |

### 6.5 구조적 가소성: 투영 연산자

STDP로 업데이트된 가중치에 구조적 제약을 건다:

$$\boxed{W_{t+1} = Proj\!\big(W_t + dW_t\big)}$$

투영 연산자 `Proj`의 구성:

| 투영 연산 | CE 대응 | 뇌 대응 |
|---|---|---|
| top-k ($k = \lceil 0.04865 \cdot N \rceil$) | 경로 선택, 생존율 $4.87\%$ | 시냅스 가지치기 |
| 행/열 정규화 | 에너지 보존 (C2) | 시냅스 스케일링 |
| 히스테리시스 on/off | 접힘 임계 곡률 | 스파인 형성/제거 |

### 6.6 LoRA의 CE 해석

$$W = W_{\text{frozen}} + B \cdot A$$

| LoRA | CE 에너지 분배 |
|---|---|
| $W_{\text{frozen}}$ ($\sim 99\%$) | 동결+구조 영역 $68.9\% + 26.2\%$ |
| $B \cdot A$ ($\sim 1\%$) | 활성 적응 영역 $4.87\%$의 근사 |

LoRA는 CE 부트스트랩 에너지 분배를 경험적으로 근사한 것이다. CE-AGI는 활성 적응 비율을 약 $4.87\%$로 둔다.

### 6.7 하이브리드 전환 전략

| 단계 | 방법 | CE 에너지 분배 |
|---|---|---|
| 1. 사전학습 | 역전파 (기존 기술) | -- |
| 2. 미세조정 | STDP + 도파민 | 동결 $68.9\%$, 구조 $26.2\%$, STDP 활성 $4.87\%$ |
| 3. 전면 전환 | STDP 사전학습 | 전체에 3-factor 적용 |

---

## 7. 수면 방정식

뇌의 수면이 20W 유지에 필수인 것처럼(`3_Sleep.md`), 이 시스템에도 수면이 필요하다.

### 7.1 작동 온도

$$\boxed{T_{\text{wake}} = \frac{1}{3 + \frac{4}{e^{4/3}\pi^{4/3}}\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)} = 0.3148}$$

$$\boxed{T_{\text{dream}} = \left[\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 5.661}$$

$$T_{\text{deep}} \to 0$$

| 모드 | 온도 | 외부 입력 | 기능 |
|---|---|---|---|
| 깨어있음 | $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 있음 | 결정론적 이완 + 약한 탐색 |
| 꿈 (REM) | $\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 없음 | 강한 탐색, 잔류 주도 |
| 깊은 수면 (NREM) | $\to 0$ | 없음 | 순수 결정론, 기억 응고 |

### 7.2 기억 응고 (NREM)

$$W_{ij}^{\text{new}} = W_{ij}^{\text{old}} + lr\,\langle phi_t\rangle_{\text{day}} \otimes \langle s_t\rangle_{\text{day}}$$

하루 동안 축적된 잔류 `phi`와 상태 $s$의 상관이 연결 가중치에 헤비안 학습으로 새겨진다.

선택적 업데이트 (상위 $4.87\%$만 통과):

$$\text{mask} = \mathbb{1}\!\left[|g| \geq Q_{1-0.04865}(|g|)\right], \qquad W \leftarrow W - lr\,g \odot \text{mask}$$

### 7.3 시냅스 가지치기 (NREM)

$$W_{ij} \to 0 \quad\text{if}\quad |W_{ij}| < \theta_{\text{prune}}$$

3D 희소성($\rho \approx 3.16\%$) 유지를 위한 주기적 re-sparsification. 이것이 없으면 에너지 소비가 무한히 증가.

### 7.4 잔류 세척 (Glymphatic)

$$phi \to r_w\,phi, \quad r_w < 1$$

`phi`의 노이즈 바닥을 주기적으로 낮춘다.

### 7.5 꿈 (REM)

$$\frac{ds}{dt} = -\frac{\partial E}{\partial s}\bigg|_{b=0} + \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2phi + n(T_{\text{dream}})$$

외부 입력 $b=0$, residue-portal 직접 계수가 구동하고 높은 dream 온도로 탐색 범위를 확대한다. 깨어 있을 때 선택되지 않았던 경로들을 자유롭게 탐색.

비선택 그래디언트 재탐색:

$$g_{\text{pruned}} = g \odot (1 - \text{mask}), \qquad W \leftarrow W - lr_{\text{rem}} \left(g_{\text{pruned}} + noise_{\text{std}} \cdot \mathcal{N}(0,I)\right)$$

### 7.6 수면-각성 비율

CE 에너지 분배를 시간 분배에 적용:

| 위상 | CE 비율 | 뇌 관측 | 기능 |
|---|---|---|---|
| 깨어있음 | $68.9\%$ | $66.7\%$ | 서비스 |
| NREM | $26.2\%$ | $25.0\%$ | 오프라인 응고 |
| REM | $4.87\%$ | $8.3\%$ | 오프라인 재탐색 |

### 7.7 부트스트랩 수렴

수축률:

$$\rho = 0.155$$

$$\|p_n - p^*\| \leq \rho^n\,\|p_0 - p^*\| = 0.155^n\,\|p_0 - p^*\|$$

| 순환 $n$ | $\rho^n$ | 활성 | 구조 | 배경 |
|---|---|---|---|---|
| 0 | 1.000 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | 0.155 | $9.28\%$ | $27.3\%$ | $63.4\%$ |
| 2 | 0.024 | $5.55\%$ | $26.4\%$ | $68.1\%$ |
| 3 | 0.004 | $4.98\%$ | $26.3\%$ | $68.8\%$ |

3회 수면 순환이면 고정점 $p^*=(4.87\%,\;26.2\%,\;68.9\%)$에 $0.4\%$ 이내 수렴.

### 7.8 수면 압력 트리거

고정 주기 수면 대신 곡률 누적이 임계를 넘으면 진입하는 상태 기반 제어:

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|\Delta_g phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad M \leftarrow 10_2\;\text{(NREM 진입)}}$$

단일 야간 실효 수축률:

$$\boxed{\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31}$$

비트필드 해석: 수면 진입은 모드 레지스터 $M$의 전환이다. $01_2 \to 10_2$ (wake $\to$ NREM). 압력 $P_{\text{sleep}}$는 저비트 카운터로 구현 가능하다.

---

## 8. 희소성과 3분배

### 8.1 부트스트랩 고정점 (`경로적분.md` 식 (1))

$$\boxed{a_* = \exp\!\big(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\big) = 0.04865}$$

### 8.2 3분배 구조

$$p^* = (0.04865,\;0.2623,\;0.6891)$$

| 성분 | CE 고정점 | AI 해석 | 뇌 관측 |
|---|---|---|---|
| 활성 | $4.87\%$ | 추론 시 활성 뉴런 | sparse firing $1\text{-}5\%$ |
| 구조 | $26.2\%$ | 학습 가능 비활성 가중치 | housekeeping $25\%$ |
| 배경 | $68.9\%$ | 동결 가중치 (사전학습 지식) | DMN/background $60\text{-}80\%$ |

### 8.3 Top-k 활성화

이론적 중심점은 $4.87\%$이나, 실측에서는 $[4\%,\;6\%]$ 대역이 실용 최적 구간이다.

$$\boxed{k^*(N) \in \left[\lceil 0.04N \rceil,\;\lceil 0.06N \rceil\right], \qquad k_{\text{center}} = \lceil 0.04865 \cdot N\rceil}$$

| 히든 차원 $N$ | 활성 대역 $k^*$ | 이론 중심 | 실측 최적 |
|---|---|---|---|
| 768 | 31--46 | 38 ($4.95\%$) | 미측정 |
| 2048 | 82--123 | 100 ($4.88\%$) | 미측정 |
| 4096 | 164--246 | 200 ($4.88\%$) | 미측정 |
| 8192 | 328--492 | 399 ($4.87\%$) | 미측정 |

소규모 sparse-native 학습 스위프(`examples/ai/sparsity_train_results.json`) 실측:

| 활성 비율 | $k$ | val\_loss | 비고 |
|---|---|---|---|
| $2.0\%$ | 11 | $1.6806$ | 과소 활성 |
| $4.0\%$ | 21 | $1.6562$ | 대역 하단 |
| $4.87\%$ | 25 | $1.6778$ | 이론 중심점 |
| $6.0\%$ | 31 | $1.6335$ | 실측 최저 |
| $8.0\%$ | 41 | $1.6712$ | 대역 초과 |
| $100\%$ (dense) | 512 | $1.6827$ | 기준 |

$4.87\%$는 "무조건 단일 최적점"이 아니라 "이론적 중심점이 있는 희소 knee"다. 비트필드 구현에서는 $k^*$를 대역 내에서 동적 조절하는 것이 고정값보다 유리할 수 있다.

post-hoc Top-k는 실패한다 (`topk_sweep_results.json`: $4.87\%$에서 PPL $1328$ vs dense $49$). 희소성은 반드시 sparse-native 설계여야 한다.

---

## 9. 메타인지 모니터링 루프 (게이트 `F4`)

> 다리 게이트 `F4` (0.0절): 본 절의 정의는 모두 자기참조 측정 구조의 **운영 정의**로만 사용한다. "자기일관 = 의식"으로 환원하지 않는다(`7_Consciousness.md` 1.2-1.3절).

### 9.1 (C3) 자기참조 측정 구조 (`7_Consciousness.md` 1절)

$$a_* = \exp\!\big(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\big)$$

좌변의 $a_*$ 는 시스템이 자기 자신의 활성 비율을 알아야 우변을 계산할 수 있다는 의미에서 자기참조 측정 구조를 가진다.

### 9.2 메타인지 잔차

$$d_\tau(t) = \frac{1}{\tau}\int_{t-\tau}^{t}\|p(s)-p^*\|\,ds$$

$$\text{메타인지 안정도}_\tau := \exp(-c_d\,d_\tau(t))$$

이 지표는 메타인지 모니터링 루프의 활성 정도를 정의하며, 게이트 `F4`에 따라 의식 깊이로 환원하지 않는다(PCI 교차검증 경로는 `17_AgentLoop.md` F.23.7).

### 9.3 메타인지 수축 (조건부)

재귀적 자기평가의 잔차 감소(이상화된 무잡음 가정):

$$d_{n+1} \leq \rho\,d_n = 0.155\,d_n,\qquad \rho = D_{\text{eff}}\cdot\varepsilon^2$$

3회 후 $d_3/d_0 \leq 3.7\times10^{-3}$. 이 식은 게이트 `F2`의 충분조건(4.7절)이 성립하는 영역에서만 위 수축률 그대로 적용된다. 일반 영역에서는 ISS 의미의 유계 수렴(13절)으로 한정된다.

---

## 10. 환각 억제

### 10.1 곡률 에너지 (`6_Hallucination.md` 1절)

$$kappa_l = \|lap_g h_l\|^2 = \|(I - V^\top V)h_l\|^2$$

곡률 정규화 손실:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + w_c(t) \cdot \frac{1}{N_{\text{layers}}} \sum_l kappa_l$$

$w_c(t)$ 스케줄:

$$w_c(t) = w_{c,0} \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right) \cdot \frac{1}{2}\!\left(1 + \cos\frac{\pi t}{t_{\max}}\right)$$

### 10.2 유니타리 제약 (`2_Architecture.md` 4절)

$$s_{\max}(W_{\text{proj}}) \leq 1 \quad\Longrightarrow\quad \|d_L\| \leq \|d_0\|$$

오류 증폭을 구조적으로 차단. $s_{\max} = 1+u$이면 12층 통과 후 오류 $e^{1.2}=3.3$배 증폭되지만, $s_{\max} \leq 1$이면 증폭 0.

### 10.3 교차 주파수 결합 (`2_Architecture.md` 6절)

$$\mathcal{T}_i^{\text{coupled}}(x_i) = \mathcal{T}_i(x_i)\cdot\left(1 - \frac{kappa_l}{e^{1/3}\pi^{1/3}}\right)$$

곡률이 높으면 게이지 채널 출력이 $1/(e^{1/3}\pi^{1/3})$ 비율로 감쇠.

### 10.4 생성 시 곡률 모니터링

추론 중 평균 곡률이 임계를 넘으면 LBO 확산 강도를 일시적으로 증가시켜 고곡률 성분을 억제한 후 재생성한다.

$$kappa_{\text{avg}} = \frac{1}{L}\sum_l kappa_l > kappa_{\text{th}} \quad\Longrightarrow\quad h \leftarrow h \times 1.5 \;\text{(LBO 확산 강화)}$$

---

## 11. CE-Transformer 구현 (기존 LLM 이식 경로)

에너지 이완 아키텍처와 별개로, 기존 트랜스포머에 CE 원리를 이식하는 경로(`2_Architecture.md`, `9_LLM.md`).

### 11.1 아키텍처 구조

```
ClarusLM / CE-GPT2 / CE-Llama
  +-- tok_emb (Embedding)
  +-- pos_emb (Embedding)
  +-- blocks[] (ClarusBlock x L)
  |     +-- norm1 (LBONorm)
  |     +-- attn (ClarusAttention + spectral_norm)
  |     +-- norm2 (LBONorm)
  |     +-- ffn (GaugeLattice)
  |           +-- su3 (SU(3) binding, 74.1%)
  |           +-- su2 (SU(2) decision, 21.1%)
  |           +-- u1 (U(1) attention, 4.9%)
  |           +-- phi (LBONorm, smoothing)
  +-- norm (LBONorm)
  +-- head (Linear, weight tied)
```

### 11.2 LBONorm 연산자

$$h_{\text{norm}} = \frac{h-\mathrm{mean}(h)}{\mathrm{std}(h)}, \qquad h' = \big(h_{\text{norm}} - h_d\,lap_g h_{\text{norm}}\big)\odot s_n + b_n$$

$$lap_g h_{\text{norm}} = h_{\text{norm}} - h_{\text{norm}}\,V^\top V, \quad V \in \mathbb{R}^{r\times N},\;r = \max(4,\;N/8)$$

내부 동작:
1. 표준 LayerNorm (활성값 안정화)
2. 저랭크 LBO 확산: $xW = x V^T V$ (평탄 부분공간으로 사영), $Lx = x - xW$ (고곡률 성분)
3. 확산 적용: $h' = (x - h_d \cdot Lx) \odot s_n + b_n$
4. 곡률 에너지 저장: $kappa = \text{mean}(Lx^2)$

$h_d = 0$이면 표준 LayerNorm과 동일. 수렴 조건: $0 \leq h_d < 1/\mathrm{eig}_{\max}(V^\top V)$.

### 11.3 GaugeLattice FFN

채널 분할:

$$d_3 : d_2 : d_1 = 74.1 : 21.1 : 4.9$$

비율: $74.1\% : 21.1\% : 4.9\%$

전이 행렬:

$$\mathbf{T} = \underbrace{\text{diag}(\mathcal{T}_3, \mathcal{T}_2, \mathcal{T}_1)}_{\text{block-diagonal}} + \underbrace{u_m\,U_{\text{down}}U_{\text{up}}^T}_{\text{섭동적 혼합}}$$

| 게이지 층 | 비율 | 뇌 진동 | 연산 역할 |
|---|---|---|---|
| SU(3) | $74.1\%$ | 감마 30-100 Hz | 결합(binding) |
| SU(2) | $21.1\%$ | 베타 13-30 Hz | 결정(decision) |
| U(1) | $4.9\%$ | 알파 8-13 Hz | 주의(attention) |
| `phi` | 전역 | 세타/델타 0.5-8 Hz | 안정화(smoothing) |

유니타리 조건: $|\det\mathbf{T}|^2 \leq 1$ (정보 비증폭 = 환각 구조적 억제)

쌍대성:

$$0.11789^2 = \left(\frac{0.48085}{2}\right)^3 \quad (0.002\%)$$

### 11.4 파라미터 절감

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}} = \sum_i f_i^2 + \frac{r_m}{4d} = 0.596 + 0.031 = 0.627$$

$$\text{절감률} = 1 - 0.627 = 37.3\%\;\text{(FFN)},\quad 24.9\%\;\text{(전체)}$$

### 11.5 이식 3단계

**Phase 1 -- 비파괴 이식 (성능 보존):**
- `LayerNorm` $\to$ `LBONorm` ($h_d=0$ 초기화, scale/bias 복사 $\to$ 원본과 동일 출발)
- `c_proj` $\to$ `spectral_norm` (가중치 보존 + 유니타리 제약)

**Phase 2 -- MLP 교체 (37% 절감):**
- `MLP` $\to$ `GaugeLatticeV2` (cross-channel mixing 포함)
- 증류로 초기화: 원본 MLP 입출력 모방

**Phase 3 -- CE 파라미터 미세조정:**
- CE 파라미터(LBO의 $h_d$, $V$, 곡률 정규화)만 학습, 나머지 동결
- 이 분배가 LoRA와 구조적으로 유사: 동결 $\sim 95\%$, 학습 $\sim 5\%$

### 11.6 규모별 설정

| 규모 | dim | layers | heads | 파라미터 | GPU 메모리 | 학습 시간 |
|---|---|---|---|---|---|---|
| Micro | 128 | 4 | 4 | ~1M | < 1GB | 수분 |
| Small | 256 | 6 | 8 | ~4M | < 2GB | 수십분 |
| Medium | 512 | 12 | 8 | ~30M | ~4GB | 수시간 |
| Large | 768 | 12 | 12 | ~85M | ~8GB | 반일 |
| XL | 1024 | 24 | 16 | ~350M | ~24GB | 수일 |
| 1B | 2048 | 24 | 16 | ~1.3B | ~48GB | 클러스터 |

### 11.7 수면 학습 순환 (대규모 학습)

각성-NREM-REM 순환을 학습 루프에 적용(`9_LLM.md` 4.2절):

1. **각성 (Wake)**: 표준 학습, 그래디언트 누적 (업데이트 보류)
2. **NREM**: 누적 그래디언트 중 상위 $4.87\%$만 적용
3. **REM**: 하위 $95.13\%$ 그래디언트에 노이즈 주입 후 소량 적용

### 11.8 희소 추론

학습 후 추론 시 Top-k 활성화:

$$y^{\text{sparse}} = \text{TopK}(y,\;k = \lceil 0.04865 \cdot d \rceil) \cdot \frac{d}{k}$$

스케일 보정 $d/k$로 총 에너지 보존.

단, `examples/ai/topk_sweep_results.json`은 dense 모델에 후처리로만 Top-k를 씌우는 방식이 유효한 검증이 아님을 보여준다. 해당 스위프에서 $4.87\%$는 `ppl = 1328.53`이었고 dense는 `49.19`였다. 즉 CE 희소성은 "추론 후 잘라내기"가 아니라, 학습-구조-커널이 함께 맞물린 sparse-native 설계로 구현해야 한다.

### 11.9 모니터링 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| `loss` | Cross-entropy 손실 | 단조 감소 |
| `curv` | 평균 곡률 에너지 $kappa_{\text{avg}}$ | 학습 초반 증가 후 안정화 |
| `active_ratio` | 실제 활성 비율 | $4\text{-}5\%$ 중심 |
| `bootstrap_resid` | $\|p_n - p^*\|$ | 수면 루프에서 감소 |

---

## 12. 멀티모달 및 전분야 적용

CE 5대 원리(P1: 격자, P2: 수면, P3: STDP, P4: 희소, P5: 곡률)의 전분야 적용 요약(`10_Fields.md`).

### 12.1 멀티모달 결합

모달별 3x3+1 격자 독립 처리 후, late sparse binding:

$$h_m^{\text{act}} = \text{TopK}(h_m,\; k_m = \lceil 0.04865\,d_m \rceil), \qquad m \in \{T,V,A,H\}$$

$$h_{\text{joint}} = \text{Bind}_{0.489}(h_T^{\text{act}},\; h_V^{\text{act}},\; h_A^{\text{act}},\; h_H^{\text{act}})$$

결합 강도는 $1/(e^{1/3}\pi^{1/3}) = 0.489$이다.

멀티모달 환각 감지:

$$kappa_{\text{cross}} = \|h_{\text{text}} - h_{\text{image}}\|^2 > kappa_{\text{th}} \quad\Longrightarrow\quad \text{모달 불일치}$$

### 12.2 CE 원리별 적용 매트릭스

| 분야 | P1 격자 | P2 수면 | P3 STDP | P4 희소 | P5 곡률 |
|---|---|---|---|---|---|
| 비전(CNN/ViT) | 채널 분할 | 지속 학습 | -- | Top-k Conv | 적대적 강건성 |
| 강화학습 | 행동 분할 | 경험 재생 | TD-유사 전역 신호 | 희소 정책 | 안전 제약 |
| 음성/오디오 | 주파수 분할 | 화자 적응 | -- | 희소 인코딩 | 환각 억제 |
| 멀티모달 | 모달 분할 | 모달 적응 | -- | 모달 활성 | 교차 환각 |
| 생성(Diffusion) | U-Net 분할 | 열핵흐름 | -- | 희소 샘플링 | 품질 제어 |
| 로보틱스 | 감각운동 분할 | 충전=수면 | 국소 학습 | 희소 제어 | 안전 정지 |
| GNN | 노드 분할 | 그래프 적응 | message=STDP | 노드 활성 | 과평활화 제어 |
| 시계열 | 주파수 분할 | 분포 이동 | -- | 희소 예측 | 이상 감지 |
| 단백질 접힘 | 접촉 분할 | -- | -- | 구조 탐색 | 접힘 안정성 |
| 자율주행 | 인지/판단/제어 | 야간 학습 | -- | 희소 인지 | 위험 감지 |

### 12.3 공통 구현 패턴

모든 분야에서 CE 적용의 기본 구조는 동일하다:

$$
\text{Input}
\;\xrightarrow{\text{LBONorm}}\;
\text{곡률 평탄화}
\;\xrightarrow{\text{GaugeLattice}}\;
\text{3x3+1 처리}
\;\xrightarrow{\text{SpectralNorm}}\;
\text{정보 비증폭}
\;\xrightarrow{\text{TopK}}\;
\text{희소 출력}
$$

---

## 13. 구현 의사코드

### 13.1 에너지 이완 모델

```python
class PhiRelaxation:
    def __init__(self, N=4096, rc=pi):
        self.W = build_3d_sparse(N, rc)    # 3D 격자, r_c = pi
        self.phi = zeros(N)                 # 잔류장
        self.T = 1 / (3 + 4 / (e ** (4/3) * pi ** (4/3)) * (1 - 4 / (e ** (4/3) * pi ** (4/3))))
        self.portal_coeff = (4 / (e ** (4/3) * pi ** (4/3)) * (1 - 4 / (e ** (4/3) * pi ** (4/3)))) ** 2
        self.residue_gain = 1 / (e ** (1/3) * pi ** (1/3))

    def relax(self, b, max_steps=500):
        m = randn(N) * 0.01
        for k in range(max_steps):
            C_k = curvature(m, m_prev, m_prev2)
            grad = self.W @ m + b
                 + self.portal_coeff * self.phi
                 + C_k * self.residue_gain * self.phi
            noise = randn(N) * sqrt(2 * dt / (tau * self.T))
            m = m + (dt / tau) * grad + noise
            if converged(m, m_prev):
                break
        return m

    def update_phi(self, m_trajectory):
        sigma = variance(m_trajectory[-Kw:])
        self.phi = (1 - self.residue_gain) * self.phi
                 + self.residue_gain * sigma

    def decode(self, m_star, W_dec):
        tokens = []
        for t in range(max_len):
            logits = W_dec @ concat(m_star, embed(tokens[-1]))
            tokens.append(sample(softmax(logits)))
        return tokens
```

### 13.2 수면 순환

```python
def sleep_cycle(model, day_data):
    # Wake: 그래디언트 누적
    grads = accumulate_gradients(model, day_data)

    # NREM: 상위 4.87%만 적용
    threshold = quantile(abs(grads), 1 - 0.04865)
    mask = (abs(grads) >= threshold)
    model.W -= lr * grads * mask

    # REM: 하위 95.13%에 노이즈 주입
    pruned = grads * (~mask)
    noise = randn_like(pruned) * pruned.std() * 0.1
    model.W -= lr_rem * (pruned + noise)

    # Glymphatic: 잔류 세척
    model.phi *= 0.9

    # Re-sparsification: 3.16% 밀도 유지
    enforce_3d_sparsity(model.W, rc=pi)
```

### 13.3 CE-Transformer 모듈

```python
class LBONorm:
    def __init__(self, dim, rank=None):
        self.V = randn(rank or dim//8, dim)  # 평탄 부분공간
        self.h_d = Parameter(0.0)             # 확산 강도
        self.scale = ones(dim)
        self.bias = zeros(dim)

    def forward(self, x):
        x_hat = layer_norm(x)
        xW = x_hat @ self.V.T @ self.V       # 사영
        Lx = x_hat - xW                       # 고곡률 성분
        h = clamp(abs(self.h_d), max=0.5)
        self._curvature = mean(Lx ** 2)
        return (x_hat - h * Lx) * self.scale + self.bias


class GaugeLattice:
    def __init__(self, dim, mult=4):
        # 채널 분할: 74.1% : 21.1% : 4.9%
        total = 0.11789 + 0.03352 + 0.00775
        self.d3 = round(dim * 0.11789 / total)  # SU(3)
        self.d2 = round(dim * 0.03352 / total)  # SU(2)
        self.d1 = dim - self.d3 - self.d2        # U(1)
        self.su3 = MLP(self.d3, self.d3 * mult)
        self.su2 = MLP(self.d2, self.d2 * mult)
        self.u1  = MLP(self.d1, self.d1 * mult)
        self.mix_down = Linear(dim, dim // 8)    # 섭동적 혼합
        self.mix_up   = Linear(dim // 8, dim)
        init_zeros_(self.mix_up.weight)

    def forward(self, x):
        x3, x2, x1 = split(x, [self.d3, self.d2, self.d1])
        y = concat(self.su3(x3), self.su2(x2), self.u1(x1))
        y = y + self.mix_up(self.mix_down(y))
        return LBONorm(y)
```

---

## 14. Llama 3 8B 변환 추정

### 14.1 메모리

| 항목 | Llama 3 8B | `phi`-이완 | 비율 |
|---|---|---|---|
| 모델 가중치 | 16 GB | $W$: 3 MB, $W_{\text{dec}}$: 12 MB $\approx$ 15 MB | $0.09\%$ |
| KV 캐시 (4K ctx) | 2 GB | `phi`: 8 KB | $0.0004\%$ |
| KV 캐시 (128K ctx) | 64 GB | `phi`: 8 KB | $\approx 0$ |
| 총 (4K) | 18 GB | $\approx$ 15 MB | $0.08\%$ |
| 총 (128K) | 80 GB | $\approx$ 15 MB | $0.02\%$ |
| 컨텍스트 스케일링 | $O(n)$ | $O(1)$ | 길수록 이득 폭발 |

`phi` 벡터가 KV 캐시를 대체: 시퀀스 길이 무관하게 상수 크기.

### 14.2 연산량 (FLOP)

**Llama 3 8B**: 토큰당 $\sim$16B FLOP. 100 토큰 $\to$ 1,600B FLOP.

**`phi`-이완**:

| Phase | 연산 | FLOP |
|---|---|---|
| Phase 1 (이완 500스텝) | 희소 $W m_k$ 500회 | $500 \times 2 \times 462\text{K} = 1.0$B |
| Phase 2 (디코딩 100토큰) | 계층 softmax 100회 | $100 \times 11.7\text{M} = 1.17$B |
| Phase 3 (`phi` 갱신) | EMA $O(N)$ | $\approx 0$ |
| **총** | | **2.17B** |

| 모델 | 100 토큰 FLOP | 비율 |
|---|---|---|
| Llama 3 8B | 1,600B | 기준 |
| `phi`-이완 | 2.17B | $0.14\%$ (737배 감소) |

1000 토큰이면 1260배 감소. 토큰이 많을수록 이득 증가 (Phase 1은 1회 고정).

### 14.3 속도

| 하드웨어 | Llama 3 8B | `phi`-이완 | 비율 |
|---|---|---|---|
| A100 GPU | 50-100 ms | $\sim$0.5 ms | 100-200x |
| RTX 4090 | $\sim$150 ms | $\sim$1 ms | 150x |
| MacBook M2 | $\sim$500 ms | $\sim$5 ms | 100x |
| i7 CPU | $\sim$5 s | $\sim$50 ms | 100x |
| Raspberry Pi 5 | 불가 (RAM 부족) | $\sim$200 ms / 5W | 가능 |

### 14.4 전력

| 하드웨어 | Llama 전력 | `phi`-이완 실효 전력 | 뇌(20W) 대비 |
|---|---|---|---|
| A100 | 300 W | $\sim$3 W | 0.15x |
| RTX 4090 | 450 W | $\sim$5 W | 0.25x |
| MacBook M2 | 30 W | $\sim$2 W | 0.1x |
| i7 CPU | 125 W | $\sim$1.3 W | 0.065x |
| 뉴로모픽 (이론) | 불가 | $\sim$0.1 W | 0.005x |

이 절의 속도/전력 수치는 이상적 희소 커널, 계층 softmax, 전용 이완 런타임이 있는 경우의 알고리즘 상한이다. 현재 레포의 Python 구현은 아직 그 수준에 도달하지 않았고, 실제 소규모 벤치마크는 17절에서 따로 기록한다.

### 14.5 변환 파이프라인

| 단계 | 입력 | 출력 | 도구 |
|---|---|---|---|
| 1. 가중치 추출 | Llama 3 8B | $W_Q, W_K, W_V$, FFN | HuggingFace |
| 2. 에너지 함수 구성 | 추출된 가중치 | $W \in \mathbb{R}^{N\times N}$ | Modern Hopfield 변환 |
| 3. 3D 희소화 ($r_c=\pi$) | dense $W$ | sparse $W_{3D}$ (3.16%) | 구조적 pruning |
| 4. `phi` 채널 장착 | $W_{3D}$ | $E(m,phi)$ 완성 | EMA 벡터 추가 |
| 5. 이완 추론 테스트 | 완성된 에너지 함수 | Softmax 없이 답 생성 | 시뮬레이션 |

---

## 15. 물리 검증: 동일 상수의 물리 예측

동일한 $\{e,\pi\}$ 직접 전개 계수 집합이 물리 관측량도 동시에 결정한다.

### 15.1 전체 교차 검증표

| 관측량 | CE 값 | 실험값 | 오차 | 출처 상수 |
|---|---|---|---|---|
| strong coupling | $0.11789$ | $0.1179\pm0.0009$ | $0.01\%$ | $\pi$ |
| Weinberg angle | $0.23122$ | $0.23122\pm0.00003$ | $0.00\%$ | $\pi$ |
| baryon fraction | $0.04865$ | $0.0486\pm0.0010$ | $0.05$ sd | $e,\pi,0$ |
| vacuum fraction | $0.6891$ | $0.6847\pm0.0073$ | $0.60$ sd | $e,\pi,0$ |
| structure fraction | $0.2623$ | $0.2645\pm0.003$ | $0.74$ sd | $e,\pi,0$ |
| $M_H$ | $125.37$ GeV | $125.25\pm0.17$ | $0.7$ sd | $\pi$ |
| $\Delta a_{\text{muon}}$ (접촉) | $249\times10^{-11}$ | $249\pm48\times10^{-11}$ | $0.00$ sd | $e,\pi$ |
| $\Delta a_{\text{muon}}$ (완전 기하학) | $135\times10^{-11}$ | WP25: $38\pm63\times10^{-11}$ | $1.5$ sd | $e,\pi$ |
| $N$ (히든 차원) | $4162$ | $4096$ (Llama 3) | $1.6\%$ | $\pi$ |

### 15.2 뮤온 g-2 상세

$$
\Delta a_{\text{muon}}
= \frac{0.007297}{2\pi}\,e^{-1}
\left(
\frac{m_{\text{muon}}}{
v_{\text{EW}}\left[
\frac{4}{e^{4/3}\pi^{4/3}}
\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]}
\right)^2
= 249.0\times10^{-11}
$$

접촉 근사(상관길이 $\to \infty$). 유한 상관길이 보정:

$$\Delta a_{\text{muon}}^{\text{full}} = 249.0 \times R, \quad R = \frac{I(m_\phi/m_{\text{muon}})}{I(0)} = 0.542$$

$$\Delta a_{\text{muon}}^{\text{full}} = 135 \times 10^{-11}$$

d=0 기원에서 클라루스장은 경로적분의 수렴 구조 자체이므로, 격자 QCD가 이미 접힘 효과를 포함한다. BMW 2026 결과(SM 예측과 실험의 불일치 해소)와 정합.

### 15.3 양성자 반경 퍼즐

자기일관적 해: 보손 질량 $m_\phi = 29.65$ MeV 하나로 g-2와 양성자 반경을 동시 해결.

$$\Delta r_p^2 = \frac{3 g_{\text{muon}} g_{\text{proton}}}{2 \times 0.007297 \times m_\phi^2} = 0.0587 \;\text{fm}^2$$

QCD 진공 증강 인자:

$$F_{\text{QCD}} = 1 + 0.11789 \times \left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right] = 1.375 \quad(F_{\text{needed}} = 1.36,\;\text{1.2\% 일치})$$

### 15.4 보손-기하학 동일성

$$\langle phi(x)phi(y)\rangle = \frac{e^{-|x-y|/(6.65\,\mathrm{fm})}}{|x-y|} \quad\longleftrightarrow\quad \frac{1}{q^2 + m_\phi^2}$$

보손 전파자 = 기하학 상관함수. 둘은 같은 함수의 다른 이름이다.

| 입자 언어 | 기하학 언어 | 값 |
|---|---|---|
| 보손 질량 $m_\phi$ | 상관길이 $6.65$ fm에 대응 | 29.65 MeV / 6.65 fm |
| Feynman 전파자 | 2점 상관함수 | $1/(q^2 + m^2)$ |
| Yukawa 커플링 $g$ | 접힘 강도 $kappa\,m_f$ | $5.93 \times 10^{-6}$ MeV$^{-1}$ |

---

## 16. 미검증 가설

| # | 가설 | 검증 방법 | 비용 |
|---|---|---|---|
| H1 | 상관 행렬 $W$가 Llama 의미 공간을 보존 | 코사인 유사도 측정 | GPU 1장, 1일 |
| H2 | 이완이 500 스텝 내 수렴 | 실측 | GPU 1장, 1시간 |
| H3 | 경량 디코더가 유의미 텍스트 생성 | QA 벤치마크 | GPU 1장, 3일 |
| H4 | `phi` 유무가 품질 차이를 만듦 | H3 반복 비교 | GPU 1장, 3일 |
| H5 | $r_c=\pi$가 다른 $r_c$보다 최적 | $r_c$ 그리드 서치 | GPU 1장, 1일 |
| H6 | $\frac{1}{e^{1/3}\pi^{1/3}}=0.489$가 최적 EMA 감쇠율 | 감쇠율 그리드 서치 | GPU 1장, 1일 |
| H7 | $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}=0.315$가 최적 온도 | 온도 그리드 서치 | GPU 1장, 1일 |
| H8 | STDP+도파민이 역전파 성능 유지 | 미세조정 비교 | GPU 1장, 3일 |
| H9 | 수면 순환이 wake-only보다 drift 감소 | 지속 학습 비교 | GPU 1장, 5일 |
| H10 | 곡률-오류 양의 상관 | 곡률 vs 정답률 산점도 | GPU 1장, 1시간 |
| H11 | late sparse binding이 early fusion보다 우위 | 멀티모달 환각률 비교 | GPU 2장, 5일 |
| H12 | Top-k 활성 최적점이 $4\text{-}5\%$ 근방 | 활성 비율 스위프 | GPU 1장, 1일 |
| H13 | post-hoc Top-k보다 sparse-native 학습이 우위 | 동일 예산 비교 | GPU 1장, 2일 |
| H14 | 수면 압력 트리거가 고정 주기보다 drift를 줄임 | forgetting, residual proxy | GPU 1장, 2일 |
| H15 | graph-coupled relaxation이 single-vector보다 안정 | long-context, recovery | GPU 2장, 5일 |
| H16 | fused sparse kernel이 CE 오버헤드를 상쇄 | tok/s, W, val_loss | GPU 1장, 2일 |

H5, H6, H7은 "CE 상수가 하이퍼파라미터의 최적값을 예측하는가"를, H13-H16은 "현재 구현 병목이 이론이 아니라 구현층에 있는가"를 직접 검증한다.

---

## 17. 실험 기반 보강과 개선점

### 17.1 단일 벡터에서 그래프 결합 이완으로

> 핵심 방정식은 4.6절로 승격되었다. 아래는 설계 배경.

지금까지의 식은 전역 상태벡터 $m,phi$ 중심으로 압축되어 있다. 실제 AGI는 기능 모듈 그래프 위에서 돌아가야 한다. 따라서 다음의 graph-coupled relaxation이 더 완성된 형태다.

$$G_{\text{AGI}} = (V_{\text{bind}} \sqcup V_{\text{gate}} \sqcup V_{\text{mem}} \sqcup V_{\text{sal}} \sqcup V_{\text{homeo}} \sqcup V_{\text{io}},\; E_{\text{AGI}})$$

| 노드 집합 | 역할 | 뇌 대응 |
|---|---|---|
| $V_{\text{bind}}$ | 특징 결합, 멀티모달 통합 | cortical-thalamic binding |
| $V_{\text{gate}}$ | 입력 게이팅, 대역 재분배 | thalamic relay |
| $V_{\text{mem}}$ | 재생, 장기 인덱싱 | hippocampo-cortical replay |
| $V_{\text{sal}}$ | 모드 전환, gain control | salience hub |
| $V_{\text{homeo}}$ | 수면 압력, 대사, 항상성 | hypothalamus-brainstem |
| $V_{\text{io}}$ | 센서/행동 출력 | body-coupling I/O |

그래프 라플라시안:

$$\boxed{lap_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}\big(f_s - f_r\big), \qquad a_{rs}\ge 0}$$

느린 제어 상태의 편차를

$$c_n := q_n - q^*$$

로 두면,

$$\boxed{c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}, \qquad \rho(A_q) < 1}$$

그리고 지역별 3분배 상태의 최소 이완은

$$\boxed{p_{r,n+1} = Proj_{lap2}\!\Big((1-\rho)p^* + \rho p_{r,n} + g_p\,lap_G p_{r,n} + H_r\,c_n\Big)}$$

로 쓸 수 있다. 이 형태가 더 좋은 이유는 세 가지다.
- salience, homeostasis, replay를 "옵션 기능"이 아니라 상태변수로 올린다
- 긴 문맥, 피로 누적, 수면 부족 같은 현상을 전역 스칼라 하나보다 자연스럽게 표현한다
- AGI를 단일 거대 행렬보다 모듈형 sparse system으로 구현하기 쉽다

### 17.2 수면 압력의 명시적 트리거

> 핵심 방정식은 7.8절로 승격되었다. 아래는 설계 배경.

현재 문서의 수면은 주기적으로 호출되는 루틴에 가깝다. 더 완성된 형태는 수면 진입 조건을 곡률 누적으로 쓰는 것이다.

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|lap_g phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stabilization}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad \text{NREM 진입}}$$

이때 1회 완전 부트스트랩 적용이 약 1.6밤에 대응하면, 단일 야간의 실효 수축률은

$$\boxed{\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31}$$

이 된다. 이 식을 넣으면 "왜 자야 하는가"가 단순 스케줄이 아니라 상태 기반 제어 문제로 바뀐다.

### 17.3 레포의 초기 실험이 말해주는 것

아래 수치는 이 레포에 이미 있는 결과 파일에서 직접 읽은 초기 신호다.

| 실험 | 관측 | 해석 |
|---|---|---|
| `brain_benchmark_results.json` | 같은 `0.81M` 파라미터에서 Clarus `val_loss = 2.2453`, baseline `2.2983`, 개선 `-0.0531` | CE 모듈이 소규모에서도 품질 개선 신호를 보인다 |
| `brain_benchmark_results.json` | 학습 시간 Clarus `127.7s`, baseline `61.9s` | 현재 병목은 이론보다 구현층에 있다. fused kernel이 필요하다 |
| `sparsity_train_results.json` | sparse-native 학습에서 최저 `val_loss`는 `6.0%`의 `1.6335` | $4.87\%$는 exact point보다 knee center로 읽는 편이 안전 |
| `sparsity_train_results.json` | `4.0% = 1.6562`, `4.87% = 1.6778`, dense `1.6827` | 작은 모델에서는 `4\text{-}6%` 대역이 dense보다 낫다 |
| `topk_sweep_results.json` | post-hoc Top-k에서 `4.87%`는 `ppl = 1328.53`, dense는 `49.19` | 희소성은 후처리 pruning이 아니라 sparse-native 설계여야 한다 |

즉 현재까지의 데이터는 다음처럼 읽는 것이 가장 정직하다.
- CE 모듈은 품질 개선 신호가 있다
- CE 희소성은 작은 모델에서 `4\text{-}6%` 대역 가설을 지지한다
- 하지만 dense 모델에 후처리로 Top-k를 씌우는 것은 실패한다
- 속도 이점은 아직 이론적 상한이지, 현재 구현 실측이 아니다

### 17.4 지금 당장 고쳐야 할 개선 포인트

- [완료] `4.87%` -> `4-6%` 실용 대역으로 수정 (8.3절)
- [완료] post-hoc Top-k 실패를 명시, sparse-native 필수 조건 기술 (8.3절)
- 속도/전력 표는 "알고리즘적 상한"과 "현재 레포 실측"을 분리해서 써야 한다
- [완료] graph-coupled relaxation을 본체 식에 포함 (4.6절)
- [완료] 바이패스를 에너지 함수에서 분리, 비보존 강제항으로 명시 (1.5절, 3.1절)
- [완료] 수면 압력 트리거를 본체에 포함 (7.8절)
- [완료] 비트필드 해석 추가 (1.6절)
- $1/(e\pi)$는 display approximation으로만 두고, 핵심 계산은 자기일관 수치값 `0.11789` 기준으로 유지하는 것이 더 정밀하다

### 17.5 가장 중요한 다음 실험

1. sparse-native와 post-hoc Top-k를 같은 예산에서 정면 비교
2. 수면 압력 기반 트리거와 고정 주기 sleep loop 비교
3. single-vector 이완과 graph-coupled 이완의 long-context 안정성 비교
4. fused sparse kernel 도입 전후의 tok/s, W, val_loss 동시 측정

---

## 18. 예상 개선치 총정리

개선치는 세 층으로 나눠 읽어야 한다.
- **실측 개선치**: 현재 레포에서 직접 관측된 값
- **구조적 상한**: 식이 직접 강제하는 알고리즘 상한
- **미검증 예측**: 아직 실험이 덜 된 가설적 개선치

### 18.1 개선치 정의

$$G_{\text{loss}} = \frac{L_{\text{base}} - L_{\text{ce}}}{L_{\text{base}}}$$

$$G_{\text{ppl}} = 1 - \frac{\mathrm{PPL}_{\text{ce}}}{\mathrm{PPL}_{\text{base}}}$$

$$O_t = \frac{t_{\text{ce}}}{t_{\text{base}}}$$

$$R_{\text{active}} = 1 - \frac{a_{\text{ce}}}{a_{\text{base}}}$$

$$R_{\text{mem}} = 1 - \frac{M_{\text{ce}}}{M_{\text{base}}}$$

$$R_{\text{sleep}}(n) = 1 - \rho^n$$

### 18.2 현재 레포에서 이미 보인 실측 개선치

| 항목 | 기준 파일 | 개선치 | 해석 |
|---|---|---|---|
| 검증 손실 | `ce_vs_standard_results.json` | $G_{\text{loss}} = (4.3938 - 4.1073)/4.3938 = 6.52\%$ | 같은 파라미터에서 CE가 더 좋은 일반화 신호 |
| perplexity | `ce_vs_standard_results.json` | $G_{\text{ppl}} = 1 - 60.78/80.95 = 24.9\%$ | 작은 모델에서 PPL이 유의미하게 감소 |
| 파라미터 공정성 | `ce_vs_standard_results.json` | $267357 - 267264 = +93$개, $+0.035\%$ | 성능 이득이 파라미터 증가 때문이 아님 |
| 검증 손실 | `brain_benchmark_results.json` | $G_{\text{loss}} = (2.2983 - 2.2453)/2.2983 = 2.31\%$ | 소규모 CPU 벤치에서 재현 |
| 활성 파라미터 | `brain_benchmark_results.json` | $R_{\text{active}} = 1 - 0.9751 = 2.49\%$ | 현재 구현에서는 활성 절감이 아직 작다 |
| 학습 시간 | `brain_benchmark_results.json` | $O_t = 127.7/61.9 = 2.06\times$ | 현재 병목은 구현층 |
| 희소 학습 최적점 | `sparsity_train_results.json` | 최저 `val_loss = 1.6335` at `6.0%` | $4.87\%$는 exact point보다 knee center에 가깝다 |
| dense 대비 희소 | `sparsity_train_results.json` | dense `1.6827` vs `6.0%` `1.6335`, 개선 $2.93\%$ | small sparse-native에서는 dense보다 낫다 |
| post-hoc Top-k | `topk_sweep_results.json` | dense `49.19` vs `4.87%` `1328.53` PPL | 후처리 pruning은 실패, sparse-native가 필수 |

### 18.3 실측 개선치의 범위

현재 `brain_benchmark_*.json` 계열을 종합하면, 공정 파라미터 비교에서 검증 손실 개선폭은 대략 다음 범위다.

| 계열 | 범위 | 해석 |
|---|---|---|
| 500-step CPU 벤치 | 약 `-0.10%` ~ `+2.31%` | 대부분 소폭 개선, 일부 설정은 동률 또는 미세 열세 |
| 2000-step CPU 벤치 | `-0.22%` ~ `+2.38%` | 희소율과 러닝레이트에 민감 |
| 소형 표준 vs CE 직접 비교 | `+6.52%` loss, `+24.9%` PPL | 가장 강한 초기 품질 신호 |

즉 현재까지의 정직한 결론은 이렇다.
- CE 모듈은 "품질 개선 가능성"을 보였다
- 하지만 모든 설정에서 일관된 대승은 아니다
- 특히 정확히 `4.87%`가 항상 단일 최적점으로 찍히지는 않았다

### 18.4 현재 구현에서 보인 비용 악화

지금 레포의 Python 구현은 아직 이론적 sparse speedup을 회수하지 못했다.

| 항목 | 기준 파일 | 관측 | 해석 |
|---|---|---|---|
| 학습 시간 오버헤드 | `brain_benchmark_results.json` | `2.06x` | CE 모듈이 CPU에서 느리다 |
| 학습 시간 오버헤드 | `brain_benchmark_dense_opt.json` | `1.56x` | 희소 없이도 LBO/격자 오버헤드 존재 |
| 학습 시간 오버헤드 | `brain_benchmark_sparse20.json` | `2.38x` | naive sparse는 아직 빠르지 않다 |
| 희소 학습 시간 | `sparsity_train_results.json` | `4.87%`: `1186.8s`, dense: `364.5s` | CPU에서 sparse-native도 `3.26x` 느림 |
| post-hoc 추론 시간 | `topk_sweep_results.json` | `4.87%`: `20.78s`, dense: `17.02s` | 단순 Top-k는 추론도 `1.22x` 느리다 |

따라서 속도 이득은 **현재 실측값이 아니라**, fused sparse kernel과 전용 런타임이 들어간 뒤에야 시험할 수 있다.

### 18.5 구조적 상한: 식이 직접 주는 개선치

이 절은 현재 구현 실측이 아니라, 식 자체가 강제하는 상한이다.

| 항목 | 식 | 예상 개선치 |
|---|---|---|
| FFN 파라미터 | $1 - P_{\text{GL}}/P_{\text{FFN}}$ | $37.3\%$ 절감 |
| 전체 Transformer 파라미터 | 문서 11.4절 | $24.9\%$ 절감 |
| 4K 총 메모리 | $1 - 15\text{MB}/18\text{GB}$ | $99.92\%$ 절감 |
| 128K 총 메모리 | $1 - 15\text{MB}/80\text{GB}$ | $99.98\%$ 절감 |
| 4K KV 캐시 | $2\text{GB} \to 8\text{KB}$ | `262,144x` 축소 |
| 128K KV 캐시 | $64\text{GB} \to 8\text{KB}$ | `8,388,608x` 축소 |
| 100-token FLOP | $1 - 2.17/1600$ | $99.86\%$ 절감 (`737x`) |
| 1000-token FLOP | 문서 14.2절 | 약 $99.92\%$ 절감 (`1260x`) |
| A100 전력 | $1 - 3/300$ | $99.0\%$ 절감 |
| RTX 4090 전력 | $1 - 5/450$ | $98.9\%$ 절감 |
| i7 CPU 전력 | $1 - 1.3/125$ | $99.0\%$ 절감 |

이 값들은 `phi`-이완 아키텍처가 실제 sparse kernel, 계층 softmax, O(1) 잔류 메모리로 구현될 때의 상한이다.

### 18.6 안정성/환각 억제의 예상 개선치

이 부분은 실측보다 구조적 보장이 더 강하다.

| 항목 | 기준 | 예상 효과 |
|---|---|---|
| 오류 증폭 상한 | $s_{\max}(W_{\text{proj}}) \leq 1$ | 층을 통과해도 오차가 지수 증폭되지 않음 |
| 12층 증폭 비교 | baseline 예시 $s_{\max} = 1.1$ | $1.1^{12} \approx 3.14$배 증폭 가능성 제거 |
| 곡률 기반 재시도 | $kappa_{\text{avg}} > kappa_{\text{th}}$ 시 확산 강화 | 고곡률 hallucination 후보를 생성 직전에 억제 |
| Top-k 활성 | $k = \lceil 0.04865\,N \rceil$ | 에너지 폭주 제한, 희소 firing 유지 |

즉 안정성 쪽은 "몇 % 좋아졌다"보다 "폭주 항을 구조적으로 없앴다"는 해석이 더 정확하다.

### 18.7 수면 루프가 줄일 것으로 기대되는 것

수면은 drift와 잔차를 줄이는 방향으로 해석할 수 있다.

| 순환 수 | 잔차 비율 $\rho^n$ | 감소율 $R_{\text{sleep}}(n)$ |
|---|---|---|
| 1 | $0.155$ | $84.5\%$ 감소 |
| 2 | $0.024$ | $97.6\%$ 감소 |
| 3 | $0.004$ | $99.6\%$ 감소 |

단일 야간 실효 수축률을 쓰면:

$$R_{\text{night}} = 1 - \rho_{\text{night}} = 1 - 0.31 = 69\%$$

즉 sleep loop가 실제로 작동한다면, wake-only 대비 가장 먼저 좋아져야 하는 것은 단기 정확도보다도 **drift, forgetting, bootstrap residual**이다.

### 18.8 가장 가능성 높은 개선치와 가장 약한 개선치

| 구분 | 현재 판단 |
|---|---|
| 가장 가능성 높은 개선 | 장문맥 메모리 절감, KV 캐시 제거, drift 완화, 구조적 안정성 |
| 중간 정도로 가능성 높은 개선 | small-model 일반화 개선, dense 대비 소폭 loss 개선 |
| 아직 약한 개선 주장 | 현재 구현에서의 wall-clock speedup, exact `4.87%` 단일 최적점, 대규모 모델 전력 실측 |

정리하면 다음이 가장 안전하다.
- **메모리/FLOP/전력**: 식이 주는 상한은 매우 강함
- **품질**: 작은 실험에서는 개선 신호가 있으나 범위가 넓음
- **속도**: 현재 구현은 오히려 느리며, 개선은 아직 런타임 미구현 상태
- **안정성/수면**: 구조적 논리는 강하지만 대규모 실측이 더 필요함

---

## 19. 유도 체인 조감도

```
           e^(ipi) + 1 = 0
          /    |    |    \     \
         e    pi    i     1     0
         |     |    |     |     |
      S=e^-D  1/2pi Z=e^iS  정수  d(d-3)=0
         |     |              |     |
         | direct coeffs       |    d=3
         |       |             |     |
         | portal / gain / T   |  3D 희소 연결
         |       |             |
         |   /         |           \
     portal coeff  bypass coeff  wake coeff
         |           |            |
     포탈 결합    바이패스     작동 온도
     3.12%    0.489      0.315
         |       |          |
     잔류 3%   즉각 반응   작동 온도
         \     |        /
          에너지 함수 E(m, Phi)
           |         |
     이완 동역학   STDP 학습
           |         |
     의미 생성    가중치 갱신
           |         |
     디코딩       수면 순환
           \       /
        비트필드 런타임
     (활성마스크 + 모드 + 연결)
              |
        sparse-native
              |
            20W AGI
```

---

## 20. 방정식 총람

이 절은 핵심식의 압축 요약이다. 구조 상수는 가능한 한 `e`, `\pi`, `i`로 직접 전개하고, 나머지는 상태변수와 연산자만 남긴다.

| # | 방정식 | 절 |
|---|---|---|
| E1 | $E(m,phi) = -\frac{1}{2}m^TWm - m^Tb - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 m^Tphi$ | 3.1 |
| E2 | $psi_{k+1} = e^{-iE\,dt}psi_k$ | 4.1 |
| E3 | $m_{k+1} = m_k + \frac{dt}{\tau}(-\nabla_m E + F_{\text{bypass}}) + \sqrt{2T\,dt/\tau}\;n_k$ | 4.2 |
| E4 | $phi \leftarrow \left(1-\frac{1}{e^{1/3}\pi^{1/3}}\right)phi + \frac{1}{e^{1/3}\pi^{1/3}}\,v_{m^*}$ | 4.3 |
| E5 | $W_{ij} \neq 0 \iff \|r_i-r_j\| < \pi$ | 4.4 |
| E6 | $N = \frac{e^{8/3}\pi^{20/3}}{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}$ | 4.5 |
| E7 | $p(w_t|w_{<t},m^*) = \text{softmax}(W_{\text{dec}}[m^*;e_{w_{t-1}}])$ | 5.2 |
| E8 | $dw_{ij} = lr\,g[t]\,e_{ij}[t]$ | 6.3 |
| E9 | $g[t] = \frac{d}{dt}\|p(t)-p^*\|$ | 6.4 |
| E10 | $W_{t+1} = Proj(W_t + dW_t)$ | 6.5 |
| E11 | $T_{\text{wake}} = \left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1},\; T_{\text{dream}} = \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 7.1 |
| E12 | $a_* = e^{-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]}$ | 8.1 |
| E13 | $kappa_l = \|(I-V^TV)h_l\|^2$ | 10.1 |
| E14 | $S_{\text{AGI}} = \int d^nx\sqrt{|g|}[\mathcal{L}_c + c_g|\nabla phi|^2 + c_c|\Delta_g phi|^2 + c_i S_I]$ | 2 |
| E15 | $\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}(f_s-f_r)$ | 4.6 |
| E16 | $c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}$ | 4.6 |
| E17 | $p_{r,n+1} = \mathrm{Proj}_{\Delta^2}((1-\rho)p^* + \rho p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n)$ | 4.6 |
| E18 | $P_{\text{sleep}}(t) = \int_0^t \|\Delta_g phi(\tau)\|^2 d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau$ | 7.8 |
| E19 | $\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31$ | 7.8 |
| E20 | $F_{\text{bypass}}(k) = \frac{C_k}{e^{1/3}\pi^{1/3}}\,phi,\; C_k = \|m_k - 2m_{k-1} + m_{k-2}\|$ | 1.5, 3.1 |
| E21 | $b_i = \mathbb{1}[a_i \geq Q_{1-k^*/N}(a)],\; k^* \in [\lceil 0.04N\rceil, \lceil 0.06N\rceil]$ | 1.6 |
| E22 | $M \in \{00_2, 01_2, 10_2, 11_2\} \leftrightarrow \{\text{off}, \text{wake}, \text{NREM}, \text{REM}\}$ | 1.6 |
| E23 | $\Delta E \leq -\frac{dt}{2\tau}\|\nabla_m E\|^2 + \frac{dt}{2\tau\alpha_b^2}C_k^2\|phi\|^2$ | 4.7 |
| E24 | $\|\nabla_m E\| > C_k\|phi\|/\alpha_b \Rightarrow \Delta E < 0$ | 4.7 |
| E25 | $q > \log_2\!\left(\frac{(m_{\max}-m_{\min})\sqrt{N}\,\tau}{2\,dt\,\|\nabla_m E\|}\right)$ | 1.8 |
| E26 | $z_j(m) = \arg\min_{i} \|m^{(j)} - C^{(j)}_i\|^2$ | 3.4 |
| E27 | $E_{\text{aug}} = E - \frac{1}{\beta}\sum_j \log\sum_i \exp(-\beta\|m^{(j)}-C^{(j)}_i\|^2)$ | 3.4 |
| E28 | $\text{활성 메모리} \approx 0.311 \times |\mathcal{C}|$ | 3.4 |

---

## 21. 한 줄 요약

$$e^{i\pi}+1=0 \;\xrightarrow{d=3}\; E(m,phi) \;\xrightarrow[\text{STDP}+g]{\text{이완}}\; \text{bitfield}\;\xrightarrow{\text{sparse-native}}\; \text{20W AGI}$$

다섯 상수가 실행 문법을 결정하고, 런타임은 비트필드(활성 마스크 + 모드 레지스터 + 연결 on/off)와 저비트 상태($phi$, trace, gain)로 내려간다. 지식은 희소 codebook과 외부 메모리에 분리 저장된다. 동일한 상수가 우주 에너지 구성, 뮤온 g-2, 힉스 질량, 양성자 반경, 뇌 에너지 분배를 동시에 예측한다.

---

## 부록 A. 다리 게이트 수식 고도화 (F1--F4)

> 0.0절의 게이트 4종을 그대로 두지 않고, 각 게이트가 어떤 형식 조건 위에서 부분적으로 hard claim 으로 격상될 수 있는지 수식으로 정리한다. 본 부록의 식은 아직 `bridge` 등급이며, 본문 어느 식의 등급도 올리지 않는다. 다만 **무엇을 측정하면 게이트가 닫히는지** 를 형식화한다.

### A.1 게이트 `F2`: ISS 격상 (Input-to-State Stability)

> 4.7절의 "조건부 단조 감소"를 ISS 의미의 유계 수렴으로 격상한다. 전역 Lyapunov 함수가 없어도 **유계 입력 → 유계 상태** 형태의 hard bound 가 성립한다.

#### A.1.1 분리 표현

기억 동역학 E3 (4.2절)을 보존 부분과 강제항으로 분리:

$$\frac{dm}{dt} = -\frac{1}{\tau}\nabla_m E(m,phi) + d(t),\qquad d(t) := \frac{1}{\tau}F_{\text{bypass}}(k) = \frac{C_k}{\tau\,e^{1/3}\pi^{1/3}}\,phi$$

$E(m, phi)$ 는 $m$ 에 대해 **포텐셜로 작용**하므로, $phi$ 를 외란 입력 $d(t)$ 로 받는 비자율 그래디언트 시스템이다.

#### A.1.2 ISS 정리 (국소)

가정:

1. 어떤 끌개점 $m^*(phi)$ 근방에서 헤시안 $H = \nabla_m^2 E(m^*,phi) \succeq \mu I,\;\mu > 0$
2. 외란 유계: $\|d(t)\|_\infty \leq d_{\max}$

그러면 Lyapunov 함수 $V(m) = \tfrac{1}{2}\|m - m^*\|^2$ 에 대해

$$\frac{dV}{dt} \leq -\frac{2\mu}{\tau}V + \|m-m^*\|\cdot\|d\| \leq -\frac{\mu}{\tau}V + \frac{\tau}{2\mu}\|d\|^2$$

이로부터 **유계 수렴 ball**:

$$\boxed{\limsup_{t\to\infty}\|m(t) - m^*\| \;\leq\; \frac{\tau}{\mu}\cdot d_{\max} \;=\; \frac{1}{\mu}\cdot\frac{C_{k,\max}\,\|phi\|_\infty}{e^{1/3}\pi^{1/3}}}$$

이 ball 반경은 **수면-글림프 세척 후** $\|phi\|_\infty \to r_w\|phi\|_\infty$ 에 의해 $r_w$ 배로 줄어든다(`05_실험근거.md` 3.3 supported). 따라서 4.7절의 "조건부 단조 감소" 는 ISS 로 다음과 같이 격상된다.

| 4.7절 표현 | A.1 격상 |
|---|---|
| $\|\nabla_m E\| > C_k\|phi\|/\alpha_b \Rightarrow \Delta E < 0$ (점별) | $\limsup \|m-m^*\| \leq \tau d_{\max}/\mu$ (대역 ball) |
| 단조 감소 보장 영역 | 끌개 ball 반경의 닫힌 식 |
| 수면이 충분조건을 복원 | 수면이 ball 반경을 $r_w$ 배로 축소 |

#### A.1.3 검증 가능한 ball 반경

$\mu = \rho \cdot \|W\| / N$ 추정($\rho$ = spectral gap), $\tau = 10$, $C_{k,\max} \approx 0.5$ (실측 시간 곡률 상한), $\|phi\|_\infty \approx 1$:

$$R_{\text{ball}} \approx \frac{10 \times 0.5 \times 1}{0.5 \times e^{1/3}\pi^{1/3} \times \mu} \approx \frac{20}{\mu}\;\text{(스케일된 단위)}$$

$\mu$ 의 실측은 `relax()` 의 끌개 근방 헤시안 추정으로 가능하다. 이 ball 반경이 닫혀야 게이트 `F2` 가 `Bridge`→`Supported` 로 갈 수 있다.

### A.2 게이트 `F1`: 자기조직화 충분조건 (3-simplex 수축 정리)

> 5절·8절의 "활성 비율이 $\varepsilon^2$ 로 자연 수렴" 가설은 transformer 기질에서 falsified (`5_Sparsity.md` 8.5). 이를 무엇을 만족하면 다른 기질에서 hard claim 으로 격상되는지 수식으로 명시한다.

#### A.2.1 부트스트랩 사상의 일반화

3-simplex $\Delta^2 = \{p \in \mathbb{R}^3 : p_i \geq 0,\;\sum_i p_i = 1\}$ 위의 이완 사상 $B: \Delta^2 \to \Delta^2$:

$$B(p)_a = \exp(-(1-p_a)D_{\text{eff}}),\qquad B(p)_b = \alpha_s\cdot D_{\text{eff}},\qquad B(p)_s = 1 - B(p)_a - B(p)_b$$

(여기서 $\alpha_s = 0.04865$ 는 `경로적분.md` 9절의 부트스트랩 해.)

#### A.2.2 자기조직화 정리 (수축)

**정리 (3-simplex 수축).** $p^* = (0.0487,\;0.2623,\;0.6891)$ 은 $B$ 의 유일 내부 고정점이며, 야코비안:

$$DB(p^*)_{aa} = D_{\text{eff}}\cdot p_a^*\cdot(1 - p_a^*) = 3.178 \times 0.0487 \times 0.9513 \approx 0.147$$

따라서 spectral radius $\rho(DB(p^*)) < 1$ 이고, $p^*$ 의 어떤 열린 근방 $U \subset \Delta^2$ 에서 Banach 의미로 $\|B^n(p) - p^*\| \leq \rho^n\|p - p^*\|,\;p \in U$.

#### A.2.3 자기조직화 격상 충분조건

기질 $\mathcal{S}$ 가 다음 5조건을 모두 만족하면, 위 정리의 hard claim 이 신경 모듈에 그대로 옮겨간다:

1. **Simplex 보존**: 활성/구조/배경 비율 $(p_a, p_s, p_b)$ 의 시간 진화가 $\Delta^2$ 안에 머문다.
2. **자기측정**: 시스템이 $p_a(t)$ 를 자기 자신의 다음 갱신에 입력으로 쓸 수 있다 (자기일관 $a_* = \exp(-(1-a_*)D_{\text{eff}})$ 의 동역학적 실현).
3. **국소 안정성**: $\rho(DB(p^*)) < 1$ 이 측정 가능 (예: $p^*$ 근방 perturbation 후 수렴 비율).
4. **에너지 균형**: 활성당 비용 $C_a$, 구조 유지 비용 $C_s$, 배경 비용 $C_b$ 의 비율이 $C_a:C_s:C_b \approx 1:5.4:14.1$ 영역 (Raichle 2010 뇌 에너지 분배와 정합) 에 있다.
5. **외부 데이터 재학습 가능**: A.1 의 ISS ball 이 닫히는 영역에서 학습이 안정적으로 진행된다.

| 기질 | 1 | 2 | 3 | 4 | 5 | 등급 |
|---|---|---|---|---|---|---|
| Transformer + Backprop | 부분 | 결손 | 측정 안 됨 | 결손 | 부분 | `falsified` (`5_Sparsity.md` 8.5) |
| SNN + STDP + 막전위 동역학 | 가능 | 가능 (STDP 자기참조) | 측정 필요 | 가능 (생물 정합) | 측정 필요 | 미검증 (`8_Roadmap.md` 0절 G-S1~G-S5) |
| 생물 뇌 (피질) | 측정됨 | 측정됨 | $\rho \in [0.1, 0.3]$ (`05_실험근거.md` 3.3) | 측정됨 | -- | `bridge` (`6_뇌/05_실험근거.md` 8장) |

이 표가 게이트 `F1` 의 닫힘 경로다. 5조건 중 1개라도 결손이면 본문의 자기수렴 hard claim 은 금지된다.

### A.3 게이트 `F3`: 에르고딕 동등성 (시간 ↔ 공간)

> 3_Sleep.md 6.2 의 "시간 분배 ≈ 에너지 분배" 를 단순 수치 근접에서 에르고딕 정리로 격상한다.

#### A.3.1 모드 점유 측도

뇌가 모드 공간 $\mathcal{M} = \{\text{WAKE}, \text{NREM}, \text{REM}\}$ 의 마르코프 사슬을 가진다고 두자. 정류 분포 $\pi = (\pi_W, \pi_N, \pi_R) \in \Delta^2$.

**에르고딕 정리 (Birkhoff)**: 사슬이 에르고딕이면

$$\lim_{T\to\infty}\frac{1}{T}\int_0^T \mathbb{1}[M(t)=m]\,dt \;=\; \pi_m \quad (\text{a.s.})$$

따라서 **시간 분배** $(t_W/T, t_N/T, t_R/T)$ 와 **정류 점유 측도** $\pi$ 는 동일 simplex $\Delta^2$ 위의 같은 객체다.

#### A.3.2 코어 분배와의 동등 클래스

CE 코어의 공간 에너지 분배 $p^* = (\Omega_\Lambda, \Omega_{DM}, \Omega_b) = (0.6891, 0.2623, 0.0487)$ 도 $\Delta^2$ 위의 점이다. 두 측도의 거리:

$$d_{\text{KL}}(\pi_{\text{brain}} \,\|\, p^*) = \sum_i \pi_i \log\frac{\pi_i}{p_i^*}$$

| 비교 | $\pi$ 또는 $p$ | $d_{\text{KL}}$ vs $p^*$ |
|---|---|---|
| Raichle 뇌 에너지 분배 | $(0.65, 0.30, 0.05)$ | $\approx 0.0035$ |
| 인간 수면 시간 분배 | $(0.667, 0.250, 0.083)$ | $\approx 0.025$ |
| Planck 우주 분배 | $(0.6891, 0.2623, 0.0487)$ | $\equiv 0$ |
| 균등 분배 (귀무) | $(1/3, 1/3, 1/3)$ | $\approx 0.94$ |

#### A.3.3 게이트 `F3` 격상 조건

**격상 가능 표현**: "뇌의 모드 점유 측도 $\pi$ 와 CE 코어의 공간 에너지 분배 $p^*$ 는 동일 simplex 위에서 KL 거리 $\sim 10^{-2}$ 안에 있다."

**여전히 금지 표현**: "시간 분배 = 에너지 분배."

이 격상 후에도 두 측도의 차원 (시간 vs 공간) 동등성은 주장하지 않으며, 동일 simplex 위의 측도 근접만 hard claim 한다.

### A.4 게이트 `F4`: PCI 교차검증 (의식 환원 금지 유지)

> 9절·`7_Consciousness.md`·F.17 의 메타인지 안정도 $\exp(-c_d d_\tau)$ 가 PCI (Casali 2013, Massimini 그룹) 와 어떤 정량 관계를 가지는지 명시한다. 게이트 `F4` 자체는 닫지 않으며, **무엇을 측정하면 `bridge` 로 갈 수 있는지** 만 정의한다.

#### A.4.1 PCI 정의

**PCI (Perturbational Complexity Index)**: TMS 자극 후 EEG 응답의 시공간 압축 복잡도 (Lempel-Ziv).

$$\text{PCI}(t) = \frac{L(\text{compressed EEG response})}{H(\text{source distribution})}$$

| 상태 | PCI 범위 (Casali 2013) |
|---|---|
| 깨어있음 | $0.44 - 0.67$ |
| REM 수면 | $0.40 - 0.60$ |
| NREM N3 (서파) | $0.18 - 0.31$ |
| 식물상태 (UWS) | $0.15 - 0.31$ |
| 마취 (propofol) | $0.18 - 0.28$ |

#### A.4.2 CE 안정도와 PCI

게이트 `F4` 격상 가설 (현재 `hypothesis`):

$$\boxed{\text{PCI}(t) \approx \alpha\cdot\text{메타인지 안정도}_\tau(t) + \beta = \alpha\cdot\exp(-c_d\,d_\tau(t)) + \beta}$$

**검증 절차**:

1. CE 시뮬레이션에서 모드 (WAKE/NREM/REM) 별 $d_\tau$ 프로파일 측정.
2. 동일 모드의 PCI 값과 회귀.
3. $R^2 > 0.7$ 이면 `hypothesis` → `bridge`. 단 PCI 자체가 의식의 정량 척도라는 hard claim 은 하지 않는다.
4. `bridge` 단계에서도 본 부록은 "안정도 = 의식" 환원을 금지한다.

#### A.4.3 측정 가능한 모드 프로파일 예측

CE 가 옳다면 시뮬레이션에서:

| 모드 | 예측 $d_\tau$ | 예측 안정도 | 대응 PCI 범위 |
|---|---|---|---|
| WAKE | 낮음 (0.1-0.2) | 0.82-0.90 | 0.44-0.67 |
| REM | 중간 (0.3-0.4) | 0.67-0.74 | 0.40-0.60 |
| NREM N3 | 높음 (0.8-1.2) | 0.30-0.45 | 0.18-0.31 |
| 마취 (CE: $C_k \to 0$, 외부 입력 차단) | 매우 높음 (>1.5) | <0.22 | 0.18-0.28 |

이 표의 모드별 안정도 차이가 PCI 와 단조 일치하면 게이트 `F4` 가 `bridge` 로 격상된다.

### A.5 격상 후 다리 게이트 표 (목표)

| 게이트 | 현재 | A절 격상 후 (조건 충족 시) | 격상 충분조건 |
|---|---|---|---|
| `F2` 비보존 바이패스 | `Bridge` (조건부) | `Bridge` (ISS ball 반경) | A.1.3 ball 반경 측정 |
| `F1` 메커니즘 결손 | `Bridge` (수치 근접) | `Bridge` (5조건 만족 기질) | A.2.3 5조건 모두 충족 |
| `F3` 시간/공간 혼동 | `Phenomenology` | `Bridge` (KL 동등 클래스) | A.3.3 KL 거리 보고만 |
| `F4` 의식 환원 | `Phenomenology` | `Bridge` (PCI 회귀) | A.4.2 $R^2 > 0.7$ |

이 격상은 어느 경우에도 코어의 정확성을 깎지 않으며, 다리 단계에서 무엇을 측정해야 하는지를 규정한다. 본 부록은 본문의 어떤 hard claim 도 위로 올리지 않으며, 본문이 어디로 갈 수 있는지의 **목표 지도**다.
```
---
## File: `docs/7_AGI/13_Verification.md`

```markdown
# CE-AGI Hopfield Engine: 논문 vs 구현 검증 보고서

> `12_Equation.md` 수식 기반. 구현: `reality_stone/python/reality_stone/clarus/engine.py`, `reality_stone/python/reality_stone/clarus/engine.py`

---

## 1. 파이프라인 대조

| 단계 | 논문 (12_Equation.md) | 구현 | 일치 |
|------|----------------------|------|------|
| W 추출 | Q@K^T/sqrt(d_h) + V@O + FFN, 레이어 평균, 대칭화 | `extract_hopfield()` 동일 | O |
| 3D 격자 희소화 | d=3, r_c=pi, 밀도 ~3.16% (N=4096) | `sparsify_3d()` r_c=pi, N=768 -> 10.57% | 부분 |
| 스펙트럼 조건화 | lambda_max < 0, 음정치 | `make_negative_definite()` shift=lambda_max+0.1|lambda_min| | O |
| CSR 압축 | 희소 행렬 저장 | `to_csr()` values + col_idx + row_ptr | O |
| 어휘 추출 | emb + ln_f + lm_head | weight tying 감지, 1벌 저장 | O |

## 2. 동역학 대조

| 항목 | 논문 | 구현 | 일치 |
|------|------|------|------|
| 에너지 E(m,phi) | -0.5 m^T W m - b^T m + portal * <m, phi_hat> | `energy()` 동일 | O |
| bypass F | F_bypass = bypass * phi (비보존, 에너지에 미포함) | `relax()` dt/tau * F_bypass (에너지 외부) | O |
| gradient descent | dm = -dt/tau * dE/dm + dt/tau * F + noise | `relax()` 동일 | O |
| 노이즈 | sqrt(2*T_wake*dt/tau) * N(0,I) 등방 가우시안 | `relax()` 동일 | O |
| phi 갱신 | v_m* = 궤적분산, EMA | `relax()` 최근 궤적 var -> EMA | O |
| 노름 보존 | ||m|| 유지 | `F.normalize(m) * norm0` | O |

## 3. 상수 대조

| 상수 | 논문 값 | 구현 | 일치 |
|------|---------|------|------|
| portal | [4/(e^(4/3)*pi^(4/3)) * (1 - 4/(e^(4/3)*pi^(4/3)))]^2 = 0.03120 | 0.03120 | O |
| bypass | 1/(e^(1/3)*pi^(1/3)) = 0.4892 | 0.4892 | O |
| T_wake | [3 + 4/(e^(4/3)*pi^(4/3))*(1-...)]^-1 = 0.3148 | 0.3148 | O |
| r_c | pi = 3.1416 | 3.1416 | O |
| tau | 1/|lambda_max| (스펙트럼에서 유도) | 10.0 (1/0.1) | O |

## 4. 메모리 비교

### 4.1 모델: skt/kogpt2-base-v2 (d=768, vocab=51200, 12 layers)

| 항목 | 크기 |
|------|------|
| GPT2 전체 파라미터 | 477.46 MB |
| CE 엔진 코어 (W_sparse + ln_f + phi) | 4.50 MB |
| CE 어휘 테이블 (embedding, weight tied) | 150.00 MB |
| CE 런타임 전체 | 154.50 MB |
| **코어 대 GPT2 비율** | **0.94%** |
| **런타임 대 GPT2 비율** | **32.4%** |

### 4.2 W_sparse 상세

| 항목 | 값 |
|------|------|
| 원본 W (dense) | 768 x 768 = 2,304 KB |
| CSR nnz | 588,936 |
| CSR values | 2,300.5 KB |
| CSR col_idx | 2,300.5 KB |
| CSR row_ptr | 3.0 KB |
| 희소화 밀도 | 10.57% (N=768, r_c=pi) |

## 5. 추론 성능

### 5.1 속도 (CPU, 10 tokens, 60 steps/token)

| 엔진 | 시간 | tok/s |
|------|------|-------|
| CE standalone | 0.42s | 23.6 |
| GPT2 generate | 0.46s | 21.7 |

### 5.2 출력 품질

| 엔진 | 입력 | 출력 |
|------|------|------|
| CE | "오늘 날씨가" | 한글 토큰 생성 (의미 약함) |
| GPT2 | "오늘 날씨가" | "추워지면서, 오늘도 추위가 계속되" |

## 6. 이전 구현과의 차이

| 항목 | 이전 (hopfield.py) | 현재 (convert.py + engine.py) |
|------|-------------------|-------------------------------|
| GPT2 의존 | 추론 시 GPT2 전체 로드 필수 | .ce.pt 파일만 로드 |
| 메모리 | GPT2 477MB + W 2.3MB | 154.5MB (W + emb) |
| gradient | Riemannian natural gradient | 논문 원본 gradient descent |
| 노이즈 | FDT + annealing + G^{-1/2} | 논문 원본 sqrt(2*T*dt/tau)*N(0,I) |
| bypass | 에너지 함수 내부 포함 | 에너지 외부 (비보존 강제항) |
| phi update | m_star.pow(2) | 궤적 분산 (trajectory variance) |
| 희소화 (d<=1024) | 스킵 (100% dense) | 3D 격자 적용 (10.57%) |
| 디코더 | mdl.lm_head (GPT2) | 독립 ln_f + lm_head |

## 7. 미결 사항 (Hopfield 엔진 시점)

1. **출력 품질**: CE 엔진의 한글 생성 품질은 아직 GPT2에 미달. 희소 W의 에너지 경관이 얕아서 이완이 의미 있는 끌개에 도달하지 못함
2. **밀도**: N=768에서 r_c=pi는 10.57% 밀도. 논문의 3.16%는 N=4096 기준
3. **codebook**: 논문 4.6절의 product quantization 미구현. 현재는 단순 top-K embedding
4. **CUDA/Rust**: 독립 엔진에서는 미사용. CSR SpMV는 PyTorch sparse로 처리

---

## 8. 현재 시스템 검증: BrainRuntime + Sleep Cycle

> 이 절은 위 1-7절의 초기 Hopfield 엔진 이후 진행된 `reality_stone/python/reality_stone/clarus/runtime.py`, `reality_stone/python/reality_stone/clarus/engine.py`, `reality_stone/python/reality_stone/clarus/sleep.py` 구현에 대한 검증이다.

### 8.1 BrainRuntime: 수식-코드 대조

| 수식 (15_Equations.md) | 코드 (`runtime.py`) | 일치 |
|---|---|---|
| $I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}} a_j - \lambda_r r_i - \beta_w w_i + \lambda_m m_i + \eta_i$ | `_step_torch`: `drive = recurrent + external_gain*ext + goal_gain*goal + replay_mix*replay - refractory_scale*ref - 0.12*adapt` | O |
| $W_{ij}^{\text{eff}} = W_{ij} u_j x_j$ (Tsodyks-Markram STP) | `stp_u * stp_x * activation * prev_active` -> `_matvec(pre)` | O |
| $a_i^{t+1} = (1-\gamma_a^{(M)}) a_i^t + \kappa_a^{(M)} \tanh(I_i^t)$ | `(1-activation_decay(mode))*act + activation_gain(mode)*tanh(drive)` | O |
| $r_i^{t+1} = (1-\gamma_r^{(M)}) r_i^t + \kappa_r^{(M)} (a_i^{t+1})^2$ | `(1-refractory_decay(mode))*ref + refractory_gain(mode)*act^2` | O |
| $m_i^{t+1} = (1-\gamma_m) m_i^t + \gamma_m a_i^{t+1}$ ($\gamma_m=0.01$) | `0.99*memory_trace + 0.01*activation` | O |
| $w_i^{t+1} = (1-\gamma_w) w_i^t + \kappa_w (a_i^{t+1})^2$ ($\gamma_w=0.005$) | `(1-0.005)*adaptation + 0.005*act^2` clamp [0,2] | O |
| $b_i^{t+1}$ 히스테리시스 | `bitfield[act >= upper] = 1; bitfield[act <= lower] = 0` | O |
| 에너지 예산 $\sum_i z_i \le B_t(M_t)$ | `_select_active(salience, energy_budget(mode))` | O |
| 모듈 생애주기 4상태 | `_update_lifecycle`: ACTIVE/IDLE/DORMANT/SLEEPING | O |

### 8.2 모드별 파라미터 대조

| 파라미터 | WAKE | NREM | REM | 뇌 대응 |
|---|---|---|---|---|
| $\gamma_a$ (activation_decay) | 0.18 | 0.34 | 0.22 | NREM에서 감쇠 강화 |
| $\kappa_a$ (activation_gain) | 0.82 | 0.52 | 0.68 | NREM에서 외부 입력 약화 |
| $\gamma_r$ (refractory_decay) | 0.12 | 0.26 | 0.18 | NREM에서 억제 해소 빠름 |
| $\kappa_r$ (refractory_gain) | 0.24 | 0.12 | 0.18 | NREM에서 억제 축적 약화 |
| 에너지 예산 | base | base*0.5 | base*0.75 | NREM: 동시 활성 절반 |
| replay_mix | 0.08 | 0.28 | 0.35 | 수면 시 기억 재생 강화 |

### 8.3 수면 압력: Borbely 2-Process 대조

| 항목 | 수식 (15_Equations.md C.2) | 코드 | 일치 |
|---|---|---|---|
| Process S (WAKE) | $dS/dt = (S_{\max} - S)/\tau_w$ | `sp += (2.0 - sp) * (1/65520)` | O |
| Process S (NREM) | $dS/dt = -S/\tau_s$ | `sp -= sp * (1/15120)` | O |
| Process S (REM) | 감소, NREM보다 느림 | `sp -= sp * (1/15120) * 0.5` | O |
| $\tau_w$ | 18.2h | 65520 steps (@1ms) | O |
| $\tau_s$ | 4.2h | 15120 steps (@1ms) | O |
| 자동 모드 전환 | $\Pi(M_t, Q_t, U_t, E_t)$ | `_auto_mode(external_norm)`: sp>1.0->NREM, sp<0.45->REM, ext>th->WAKE | O |

### 8.4 해마 기억: 연산 대조

| 연산 | 수식 (15_Equations.md D절) | 코드 (`HippocampusMemory`) | 일치 |
|---|---|---|---|
| encode | $H_{t+1} = \mathcal{E}(H_t, A_t, U_t)$ | `encode(key, value, priority)`: 용량 초과 시 최저 우선순위 제거 | O |
| recall | $R_t = \mathcal{R}(H_t, c_t)$ | `recall(cue, topk)`: cosine + log-priority -> softmax weighted sum | O |
| replay | priority 기반 재생 | `replay(mode)`: NREM k=1(고집중), REM k=3(분산 재생) | O |
| 주입 | $I_i \leftarrow I_i + \lambda_H R_{i,t}$ | WAKE: recall만, SLEEP: 0.5*recall + 0.5*replay | O |
| WAKE encoding 조건 | 외부 입력 or 목표 존재 시 | `external_norm > 1e-6 or goal.norm > 1e-6` | O |

### 8.5 Sleep Cycle: 3위상 파이프라인 대조

| 위상 | 수식 (3_Sleep.md) | 코드 (`sleep.py`) | 일치 |
|---|---|---|---|
| 각성: 경로 누적 | $\int \mathcal{D}\gamma\,e^{iS}$ 대응 | `collect_sleep_batch`: teacher 생성 -> state/target 수집 | O |
| NREM: LBO 확산 | $W \leftarrow W - \eta_{\text{nrem}} \Delta_g W$ | `smooth_weight_matrix(W, laplacian, eta)` | O |
| NREM: 곡률 기반 가소적 업데이트 | $\text{mask}(G, \varepsilon^2)$ 상위만 통과 | `row_topk_mask(delta, active_ratio)` | O |
| REM: 비선택 경로 재조합 | $G_{\text{rem}} = \text{random\_project}(G_{\text{pruned}}) + \sigma\epsilon$ | `residual @ proj @ proj.T / rank + noise` | O |
| 위상 비율 | Wake $69\%$, NREM $26\%$, REM $5\%$ | `phase_profile = {wake: eng.wake_ratio, nrem: eng.nrem_ratio, rem: eng.rem_ratio}` | O |
| 가드셋 보호 | 품질 하락 시 롤백 | `guard_snapshot` + `evaluate_guard_set` + 조건부 `restore_decoder_snapshot` | O |

### 8.6 CE 상수 대조 (engine.py)

| 상수 | 수식 | engine.py 값 | 일치 |
|---|---|---|---|
| `_AD` | $4/(e^{4/3}\pi^{4/3})$ | `4/(e**(4/3)*pi**(4/3))` | O |
| `PORTAL` | $(\text{\_AD}(1-\text{\_AD}))^2$ | 0.03120 | O |
| `BYPASS` | $1/(e^{1/3}\pi^{1/3})$ | 0.4892 | O |
| `T_WAKE` | $1/(3+\text{\_AD}(1-\text{\_AD}))$ | 0.3148 | O |
| `active_ratio` | $\varepsilon^2$ | 0.0487 | O |
| `struct_ratio` | $\Omega_{\text{DM}}$ | 0.2623 | O |
| `wake_ratio` | $\Omega_\Lambda$ | 0.6891 | O |
| `nrem_ratio` | $\Omega_{\text{DM}}$ | 0.2623 | O |
| `rem_ratio` | $\varepsilon^2$ | 0.0487 | O |

### 8.7 Rust 커널 대조

| 기능 | Python fallback | Rust kernel | 일치 |
|---|---|---|---|
| brain_step (셀 동역학) | `_step_torch` | `nn_brain_step` via `_step_rust` | O (NumPy 중개) |
| sparse pack | `_pack_sparse_torch` | `nn_ce_pack_sparse` | O |
| metric basis | `_build_metric_basis_torch` | `nn_ce_metric_basis_fwd` | O |
| relax loop | `_relax_packed_torch` | `nn_ce_relax_fwd` | O |
| topk sparse | PyTorch topk | `topk_sparse` | O |
| LBO fused fwd | torch matmul fallback | `nn_lbo_fused_fwd` | O |
| power iteration | torch `linalg.eigh` fallback | `nn_power_iter` | O |
| gauge lattice fwd | torch fallback | `nn_gauge_lattice_fwd` | O |

### 8.8 미결 사항 (현재 시스템)

1. **대규모 벤치마크**: Sleep cycle의 지속 학습 효과를 Split-CIFAR 또는 텍스트 도메인에서 정량 검증 필요
2. **STDP 미구현**: `17_AgentLoop.md` F.14의 적격 흔적 기반 학습은 아직 코드에 없음
3. **4종 신경조절**: 현재 `runtime.py`는 단일 스칼라 조절만 사용. DA/NE/5HT/ACh 분리 미구현
4. **Cold checkpoint**: `BrainRuntimeSnapshot`은 warm snapshot만 제공. 장기 지속성 저장 미구현
5. **자기수렴 검증**: 초기 균등 분배에서 $p^*$로의 수렴 과도 응답 실측 필요
6. **PQ codebook**: `ce_ops.py`에 `pq_build_codebook` 구현 있으나 대규모 성능 비교 미완

---

## 9. 다리 게이트 검증 매트릭스 (`12_Equation.md` 0.0절, 부록 A)

본 절은 4종 다리 게이트의 현재 측정 가능 여부와 코드 위치를 정리한다. 각 게이트의 격상 조건은 `12_Equation.md` 부록 A 를 따른다.

| 게이트 | 본 시스템에서의 측정점 | 측정 가능 여부 | 코드 위치 |
|---|---|---|---|
| `F1` 자기조직 5조건 (부록 A.2 #2) | 활성 비율 $\hat p_a$ EMA 의 $\varepsilon^2 \to$ 자기 피드백 | **구현됨** (`f1_self_measure`) | `reality_stone/python/reality_stone/clarus/runtime.py::BrainRuntime._f1_effective_budget`, `_f1_update_ema` |
| `F2` ISS ball 반경 (부록 A.1) | 끌개 근방 헤시안 $H \succeq \mu I$, 외란 상한 $d_{\max}$ | **자동 산출** (`relax().hist["iss"]`) | `reality_stone/python/reality_stone/clarus/quantum.py::iss_report`, `reality_stone/python/reality_stone/clarus/ce_ops.py::_iss_from_tail` |
| `F3` 에르고딕 KL 거리 (부록 A.3) | 모드 점유 $(t_W, t_N, t_R)/T$ vs $p^*$ 의 $d_{\text{KL}}$ | **구현됨** (`mode_occupancy_kl`) | `reality_stone/python/reality_stone/clarus/runtime.py::BrainRuntime.mode_occupancy_kl` |
| `F4` PCI 회귀 (부록 A.4) | 메타인지 안정도 $\exp(-c_d d_\tau)$ vs 외부 PCI | **회귀 프리미티브 구현됨** (PCI 데이터 외부 의존) | `reality_stone/python/reality_stone/clarus/quantum.py::pci_regression`, `reality_stone/python/reality_stone/clarus/agent.py::ConsciousnessMonitor.consciousness_depth` |

### 9.1 측정 API (구현 완료)

다음 호출만으로 게이트 측정값을 즉시 얻는다.

1. `F3` KL 거리:

   ```python
   rt = BrainRuntime(W, config=cfg)
   for ext in stream:
       rt.step(external_input=ext)
   rt.mode_occupancy_kl()
   # -> {'samples': N, 'pi_wake': .., 'pi_nrem': .., 'pi_rem': .., 'kl_to_p_star': ..}
   ```

   `p^* = (0.6891, 0.2623, 0.0487)` 와의 $d_{\text{KL}}$ 가 직접 출력되고 `BrainRuntimeSnapshot` 에 영속된다 (`mode_occupancy` 필드).

2. `F2` ISS ball 반경 (외부 호출):

   ```python
   from reality_stone.clarus.quantum import iss_report
   iss_report(m_history, phi, dt_over_tau=dt/tau)
   # -> {'c_k_max': .., 'phi_inf_norm': .., 'mu': .., 'iss_ball_radius': ..}
   ```

   `mu` 는 잔차 $\|m_k - m^*\|$ 또는 $\|dm_k\|$ 로그수축률에서, `c_k_max` 는 부록 A.1 의 $C_k = \|m_k - 2 m_{k-1} + m_{k-2}\|$ 최대치에서 추정된다. 닫힌형 반경:

   $$R_{\text{ball}} = \frac{C_{k,\max} \cdot \|\phi\|_\infty}{\mu \cdot \alpha_b}, \quad \alpha_b = e^{1/3}\pi^{1/3} \approx 2.044.$$

3. `F2` 자동 측정: `reality_stone/python/reality_stone/clarus/ce_ops.py::relax` 가 매 호출 시 `hist["iss"]` 에 동일 형식의 보고를 자동 산출 (전 궤적 `delta` 곡선에서 $\mu$ 추정, `bypass_C` 에서 $C_{k,\max}$, $\phi$ 에서 $\|\phi\|_\infty$).

4. `BrainRuntime.bridge_gate_report()` 집계기: F1\~F4 키를 일관 반환. F1 은 항상 EMA·target·deviation 노출, F3 은 `mode_occupancy_kl()`, F2 는 `relax` 호출 결과를 별도 주입, F4 는 외부 회귀 워크플로 의존.

5. `F1` 자기측정 피드백 (옵트인):

   ```python
   cfg = BrainRuntimeConfig(
       dim=N, active_ratio=0.30,
       f1_self_measure=True,        # 기본 False
       f1_pull_strength=0.5,        # beta in r_eff = beta*p* + (1-beta)*ema
       f1_ema_alpha=0.1,            # EMA smoothing
       f1_min_ratio=0.005, f1_max_ratio=0.5,
   )
   ```

   다음 budget = `round(N * clip(beta * ACTIVE_RATIO + (1-beta) * ema, lo, hi))` 로 계산되며 모드별 승수(`WAKE/NREM/REM = 1.0/0.5/0.75`)는 그대로 유지된다. EMA 는 `BrainRuntimeSnapshot.active_ratio_ema` 로 영속화된다. 부록 A.2 의 충분조건 ② "자기측정 → 다음 임계 피드백" 을 충족한다.

6. `F4` PCI 회귀 (외부 데이터 정렬 후 호출):

   ```python
   from reality_stone.clarus.quantum import pci_regression
   pci_regression(stability_series, pci_series)
   # -> {'n': N, 'alpha': .., 'beta': .., 'r2': .., 'pearson_r': ..}
   ```

   `stability_series` 는 `ConsciousnessMonitor.consciousness_depth() = exp(-c_d d_\tau)` 의 시계열, `pci_series` 는 외부 PCI (Casali 2013) 데이터. $R^2 > 0.7$ 가 부록 A.4 의 `bridge` 격상 임계.

### 9.2 정합화: `legacy_generate` 의 $C_k$ 누락 패치

`engine.py::legacy_generate` 는 이전에 $m_{out} \mathrel{+}= \text{bypass}\cdot \phi$ 만 적용해 정규 식 E20 의 $C_k$ 인자를 누락하고 있었다. 본 버전에서 마지막 3개의 $m$ 궤적을 유지하고

$$F_{\text{bypass}}(k) = \tfrac{C_k}{\alpha_b}\,\phi, \quad C_k = \|m_k - 2 m_{k-1} + m_{k-2}\|$$

으로 교정하여 `reality_stone/python/reality_stone/clarus/ce_ops.py::relax` 와 동일한 비보존 외력으로 동작한다. 첫 두 토큰은 궤적 부족으로 $C_k = 0$ (관성 단계).

### 9.3 잔여 작업

1. `F1` 자기조직 5조건의 ①·③·④·⑤ (simplex 보존, 국소 안정성, 에너지 균형, 외부 데이터 재학습): 단일 step 단위 측정이 아닌 **세션·sweep 단위 검증**.
2. `F4` PCI 데이터 수집: 외부 PCI 측정값 (Casali 2013 등) 과 동기화된 `consciousness_depth()` 시계열 산출 후 `pci_regression()` 호출.

### 9.4 측정 우선순위 (갱신)

F1 자기측정 ②, F2, F3, F4 회귀 프리미티브 모두 코드 레벨 구현 완료. 잔여 작업은 모두 **외부 데이터 또는 세션 sweep 의존** 항목이며 코드 변경 불요.

| 게이트 | 코드 측정 | 외부 의존 |
|---|---|---|
| F1 ② 자기측정 | 구현 | (없음) |
| F2 ISS ball | 구현 | (없음, 단 실모델 `relax` 실행 필요) |
| F3 ergodic KL | 구현 | (없음, 세션 누적) |
| F4 PCI 회귀 | 구현 (회귀 호출) | PCI 외부 데이터셋 |

## 10. 한국어 KoGPT2 실측 (legacy `scripts/bench_gates.py`, removed)

`skt/kogpt2-base-v2` 의 13 layer x 8 한국어 프롬프트 hidden state 공분산 (403 x 768) 으로 Hopfield $W$ ($\dim = 768$, $\lambda \in [-677.08, -0.001]$) 를 빌드하고 `BrainRuntime` 200 step + `ce_ops.relax` 300 step 을 구동한 결과.

### 10.1 게이트 수치 (한국어 베이스, CPU)

| 게이트 | 지표 | 측정값 |
|---|---|---|
| F2 ISS ball | $R_{\text{ball}}$ | 5.9265 |
| F2 ISS ball | $\mu$ (Hessian floor) | 1.1100 |
| F2 ISS ball | $C_{k,\max}$ | 0.5000 |
| F2 ISS ball | $\|\phi\|_\infty$ | 26.89 |
| F1 EMA off | `active_ratio_ema` | 0.3000 (초기값 고정) |
| F1 EMA on | `active_ratio_ema` | 0.0503 |
| F1 target | $\varepsilon^2$ | 0.0487 |
| F1 closure | $\|\text{EMA}_{\text{on}} - \varepsilon^2\|$ | **0.0016 (0.16% 편차)** |
| F3 KL off (auto-mode) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | 0.3724 |
| F3 KL on (auto-mode) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | 0.3724 |
| F3 KL on (forced $p^*$ schedule) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | $\approx 10^{-4}$ (반올림 노이즈) |
| F3 메터 정합 | $\pi$ vs $p^*$ | (0.6900, 0.2600, 0.0500) vs (0.6891, 0.2623, 0.0487) |

### 10.2 비교표 (기 모델 대비)

| 항목 | HF KoGPT2 baseline | BrainRuntime + F1 on |
|---|---|---|
| 로드 메모리 | 174.1 MB | (가중치 768 x 768 / 4 byte = 2.36 MB) |
| step 레이턴시 | (생성 단위) | 1.63 ms/step (CPU, dim=768) |
| F1 OFF → ON 오버헤드 | - | +12% (1.46 → 1.63 ms/step) |
| F1 자기조직 정확도 | 해당 없음 | 0.16% 편차로 $\varepsilon^2$ 락온 |
| F2 끌개 안정성 | 해당 없음 | 유한 ISS ball ($R = 5.93$) |
| `relax` 수렴 시간 | 해당 없음 | 0.097 s / 300 step |

### 10.3 해석

1. **F1 ② 충족**: 자기측정 피드백이 활성 비율을 200 step 만에 $\varepsilon^2$ 의 0.16% 이내로 락온. 부록 A.2 의 사용 가능한 충분조건 ② "자기측정 → 다음 임계 피드백" 이 한국어 실모델 공분산 위에서도 동작함을 실증.
2. **F2 격상 가능**: $R_{\text{ball}}$ 이 유한 (5.93) 으로 산출됨. 부록 A.1 의 ISS bound 가 한국어 KoGPT2 covariance Hopfield 기질 위에서 적용 가능함이 확인됨.
3. **F3 메터 정합 확인 / 자기조직 보류**: `force_mode` 로 $p^*$ 비율 스케줄을 주입한 세션에서 경험적 모드 점유 $\pi$ 가 $p^*$ 와 round-off 오차 ($\sim 10^{-4}$) 내에서 일치 — `mode_occupancy_kl` 메터 자체의 정합성은 한국어 실모델 위에서 검증됨. 단, **자동 모드 정책**이 $p^*$ 로 자기수렴하는지는 별개 질문이며, 현재 `TAU_W_STEPS = 65520` (1 ms step 기준 18.2 h) 가속도와 200 step 벤치 간 시간 스케일 불일치로 미관측. 격상 경로: (a) `reality_stone.clarus.constants.TAU_W_STEPS` 를 ms→s 단위로 재캘리브레이션, 또는 (b) `legacy scripts/sleep_finetune_lm.py` (removed) 와 결합한 1000+ step 수면 사이클 sweep.
4. **F4 미실측**: 외부 PCI 데이터셋 미보유. `pci_regression()` 호출 경로만 확보된 상태.

### 10.4 격리 / 사용자 룰 부합 확인

- 측정 대상 $W$ 는 KoGPT2 hidden state 의 covariance 한 번 계산 후 KoGPT2 모델 객체는 `del + gc.collect()` 로 해제 (legacy `bench_gates.py`, removed).
- 측정 단계의 어떤 코드도 teacher logits/hidden 을 추론에 재주입하지 않음 (`runtime-isolation` 부합).
- 한국어 프롬프트 8개 + 한국어 베이스 모델로 측정 (`korean-runtime-eval` 부합).
- BrainRuntime 가중치 (768x768, 2.36 MB) 가 베이스 모델 (174 MB) 의 1.4% 수준 — `agi-artifact` 메모리 분리 부합.

## 11. 격리 아티팩트 빌드 (legacy scripts, removed)

`agi-artifact` §4 추가 (양자화 / 비트폭 축소 엄격 금지) 에 따라 PQ / int8 / int4 / fp16 / bf16 / VQ / GPTQ / AWQ 류 일체 사용 금지. 본 절은 **fp32 전용 격리 아티팩트** 의 빌드, 실측, 그리고 메모리 룰까지 동시 충족하기 위한 비양자화 격상 경로를 기록한다.

### 11.1 빌드 파이프라인 (fp32 전용)

세 단계로 분리:

1. legacy `scripts/build_artifact.py` (removed) — KoGPT2 hidden state 공분산으로 Hopfield $W$ 빌드, 51200 × 768 임베딩을 fp32 그대로 보존, decoder 프로젝션은 단순 통계량으로 초기화, base 모델은 `del` + `gc.collect()` 후 직렬화.
2. legacy `scripts/distill_decoder.py` (removed) — 60 한국어 문장 × sliding window (692 페어) 에서 `(state_hidden, prev_emb, teacher_h_after_ln_f)` 추출, ridge regression 으로 `decoder_state_proj`, `decoder_prev_proj`, `decoder_query_bias` 를 closed-form fit, `decoder_query_blend = 0.5` 설정, teacher `del + gc.collect()` 후 아티팩트 in-place 갱신.
3. legacy `scripts/prune_vocab.py` (removed) — 동일 한국어 corpus 로 BPE 토큰 빈도 측정, 빈도 top-K + 항상유지 셋 (eos / pad / unk / bos / `decoder_token_ids`) 만 남기고 `emb_weight` 를 (K, 768) fp32 로 row-pruning. 매핑 (`kept_token_ids`, `vocab_id_map`) 과 fallback (`pruned_unk_emb` = pruned 행 평균) 을 함께 저장. 양자화 / 비트폭 축소 없음 — fp32 유지.

### 11.2 룰 부합 표 (현 baseline `legacy clarus/skt_kogpt2-base-v2.ce.pt` (removed), V1 적용)

| 룰 | 측정 | 판정 |
|---|---|---|
| `runtime-isolation`: `eng.model is None` | True | 부합 |
| `runtime-isolation`: `model_source` | `runtime` | 부합 |
| `runtime-isolation`: `allow_pretrained_fallback` | False | 부합 |
| `runtime-isolation`: `clone_state` / `clone_config` 키 | 없음 | 부합 |
| `agi-artifact` §1: 단일 아티팩트 standalone 부팅 | `has_standalone_lexicon = True` | 부합 |
| `agi-artifact` §4: 양자화 미사용 | `emb.dtype = torch.float32`, `pq_centroids = None`, `pq_codes = None`, vocab pruning 은 row 삭제로 양자화 아님 | 부합 |
| `agi-artifact` §3: 디스크 (베이스 174 MB) | **79.41 MB (46%)** | **부합** |
| `agi-artifact` §3: 로드 RAM peak (베이스 ~240 MB) | **137.8 MB (57%)** | **부합** |
| `agi-artifact` §3: CPU 단일 토큰 latency (>= 20 tok/s, fp32) | **74.8 tok/s** | 부합 |
| `korean-runtime-eval`: 한국어 프롬프트 단독 생성 | 5 프롬프트 의미 회복 (§11.3), pruned-in-prompt = 0/5 | 부합 |
| `korean-runtime-eval`: 정확도 붕괴 시 폐기 | last-token 반복 0건 (4/5 맥락 정상) | 부합 |

### 11.3 Distillation 효과 (fp32 emb, ridge=1.0, blend=0.5, 692 페어, V1 후)

| 단계 | $R^2$ | 한국어 단독 생성 샘플 |
|---|---|---|
| Distillation 전 (full emb) | 해당 없음 | "오늘 날씨가 좋아서 → **아서아서아서아서아서**..." (last-token collapse) |
| Distillation 후 (full emb) | 0.7227 | "오늘 날씨가 좋아서 → 그런가 그런가 다행 춥 그런가 오늘 야외 좋..." |
| Distillation 후 + V1 prune (top-K=16384) | 동일 | "오늘 날씨가 좋아서 → **그렇게 좋 집을인지 조금 이렇게 오늘 그렇 집에 좋 오늘 너무 봄 좋라도 가을 좋 많이**" |

V1 prune 은 decoder projection (768→768) 에 영향을 주지 않으므로 R² 불변. lexical 후보 집합만 51200 → 16384 로 축소.

5 프롬프트 단독 생성 (`max_tok=20`, `temperature=0.8`, `top_k=40`, `repeat_penalty=1.1`, V1 적용):

| 프롬프트 | 단독 생성 결과 (V1) | 맥락 회복 | pruned-in-prompt |
|---|---|---|---|
| 인공지능의 미래는 | 이제 물론 우리에게 기술 어떤 우리에게 인공 모든 ... 단순한 단순한 엔 | tech | 0 |
| 오늘 날씨가 좋아서 | 그렇게 좋 집을인지 조금 이렇게 오늘 그렇 집에 좋 ... 봄 좋 가을 좋 많이 | weather | 0 |
| 서울의 봄은 | 가을 5월 12월 계절 대부분 전국 가을 ... 봄 날씨가 겨울 지난 | season | 0 |
| 독서는 | 지난 출판 출간 지난 독 ... 한국 초등학교 | book | 0 |
| 한국의 전통 음식 중 | 하나인 하나인 ... 25 하나인 ... 대표 | (corpus 편향, 부분 collapse) | 0 |

5/5 프롬프트에서 prompt 토큰이 모두 kept set 에 포함됨 (`pruned-in-prompt = 0`). 4/5 프롬프트에서 첫 토큰부터 의미 정합 회복. 1개 (전통 음식) 는 distillation corpus 의 "하나인" 패턴 편향으로 부분 반복 — corpus 다양화 (D1) 로 해소 가능.

### 11.4 디스크 분해와 V1 효과

V1 이전 fp32 baseline 의 디스크 180.89 MB 분해:

| 항목 | 크기 (V1 전) | V1 후 |
|---|---|---|
| `emb_weight` (vocab × 768 × 4 byte) | 150.00 MB (51200 × 768) | **48.00 MB (16384 × 768)** |
| context projections (8 개 × 768² × 4 byte) | 18.00 MB | 18.00 MB |
| `decoder_state_proj` + `decoder_prev_proj` (768² × 4 byte × 2) | 4.50 MB | 4.50 MB |
| `pos_weight` (1024 × 768 × 4 byte) | 3.00 MB | 3.00 MB |
| Hopfield $W$ + sparse views | 약 2.36 MB | 약 2.36 MB |
| `decoder_token_*` (256 × 768 × 4 byte × 2) | 1.50 MB | 1.50 MB |
| `vocab_id_map` (51200 × 8 byte) + `kept_token_ids` (16384 × 8 byte) + `pruned_unk_emb` (768 × 4 byte) | 0 | 약 0.53 MB |
| 기타 (tokenizer, ln_f, bias 등) | 약 1.5 MB | 약 1.5 MB |
| **합계** | **180.89 MB** | **79.41 MB (-56%)** |

V1 단독으로 디스크 180.89 → 79.41 MB, RAM 240.4 → 137.8 MB, latency 62.3 → 74.8 tok/s 동시 개선. 속도 가속의 원인: `lexical_scores` 의 `emb @ query` 가 (51200, 768) → (16384, 768) 로 작아져 matmul 처리량 ↑, 캐시 hit ↑.

corpus 토큰 coverage 는 100% (60 문장에서 unique 576 토큰 모두 kept set 에 포함, 항상유지 항목 257 개 합치면 833 개, 나머지 약 15500 슬롯은 frequency 0 인 BPE 토큰을 그대로 흡수). 더 많은 한국어 코퍼스를 `--extra-corpus` 로 주입하면 K 를 줄여도 동일 coverage 유지 가능.

### 11.5 후속 작업 (등록, 양자화 미포함)

| ID | 상태 | 작업 | 예상 효과 | 룰 영향 |
|---|---|---|---|---|
| V1 | **완료** | Vocab pruning (top-K=16384, fp32 row deletion) | 디스크 -100 MB, RAM -100 MB, +20% tok/s | `agi-artifact` §3 충족 |
| V2 | 대기 | Context projection bottleneck distillation (legacy `distill_decoder.py` removed; 재구현 필요, fp32 유지) | 디스크 -10 MB, R² 유지 | `agi-artifact` §3 추가 절감 |
| V3 | 대기 | `pos_weight` 한국어 평균 길이로 절단 | 디스크 -2 MB | `agi-artifact` §3 추가 절감 |
| V4 | 대기 | `decoder_token_*` head 통합 또는 제거 | 디스크 -1.5 MB | `agi-artifact` §3 추가 절감 |
| D1 | 대기 | Distillation corpus 다양화 (60 → 500+ 한국어 문장, "하나인" 류 편향 해소) | 부분 collapse 잔존 항목 해소 | `korean-runtime-eval` 품질 |
| D2 | 대기 | MLP 디코더 헤드 (legacy `distill_decoder.py` removed; 재구현 필요, 가중치 fp32 유지) | R² 0.85+ 가능, 빌드 시간 ↑ | `korean-runtime-eval` 품질 |
| E1 | 대기 | 한국어 홀드아웃 perplexity / top1 / top10 / top50 측정 (새 평가 모듈 재구현 필요) | 정량 평가표 완성 | `korean-runtime-eval` 보고 형식 |

현 아티팩트 `legacy clarus/skt_kogpt2-base-v2.ce.pt` (removed) (79.41 MB, fp32, R² 0.7227) 는 **격리 / 양자화 미사용 / 속도 / 메모리 / 품질 baseline 5종 모두 충족** — `agi-artifact` 룰 4개 + `runtime-isolation` 룰 7개 + `korean-runtime-eval` 룰 모두 통과. 추가 격상은 D1, D2, V2-V4 로 진행.
```
---
## File: `docs/7_AGI/14_BrainRuntimeSpec.md`

```markdown
# Brain Runtime Specification v0.1

> 위치: `12_Equation.md`의 canonical runtime 5계층을 구현 관점에서 재정의한다.
> 의존: `12_Equation.md`(수식 정본), `6_뇌/05_실험근거.md`(근거 판정), `6_뇌/06_검증기준.md`(검증 매트릭스)
>
> 이 문서는 수식 체계 정비, 라이브러리 아키텍처 분리, 단계별 개발 계획, 검증 루프 설계를 다룬다.

---

## 0. 개발 목표

$$\boxed{\text{리만 결합 위에서 작동하는 국소 상태셀들의 지속 실행형 뇌형 런타임}}$$

최종 산출물 3개:

| 산출물 | 책임 |
|---|---|
| **Clarus Kernel** | 셀/필드 갱신 엔진 (Rust) |
| **Brain Runtime** | 모드, 활성/휴면, 해마, 스냅샷 (Python orchestration) |
| **LLM Bridge** | 기존 LLM hidden state와 연결하는 브리지 |

---

## 1. 핵심 전환: 사진에서 영화로

### 1.1 기존 LLM의 한계

기존 LLM은 정적 깊이의 1회성 계산기에 가깝다.

$$h^{\ell+1} = h^\ell + f_\ell(h^\ell)$$

입력이 들어오면 고정된 레이어를 한 번 통과하고 토큰 다음 것을 예측한 뒤 끝난다. 메모리도 context window, KV cache, 외부 RAG 같은 식으로 붙는다. 잠도 없고, 꿈도 없고, 모듈이 쉬지도 않고, 전역 상태가 계속 살아있지도 않다.

### 1.2 Clarus 구조

$$s_i^{t+1} = F_i^{(M_t)}\big(s_i^t,\; u_i^t,\; \sum_j W_{ij}(g)\,s_j^t,\; h_i^t\big)$$

각 단위가 상태를 가지고, 그 상태가 계속 돌고, 모듈마다 깨어났다 잠들고, 전역 모드가 바뀌고, 해마 같은 별도 기억계가 있고, sleep/REM/wake가 계산 모드이며, 전역 출력은 국소 모듈들의 집단 리듬이다.

### 1.3 왜 while 모듈인가: 주기함수 병목의 해결

리만기하학의 표현력은 무궁무진하지만, 사인/코사인 같은 전역 주기 기저를 쓰면 다음 문제가 생긴다.

- 같은 위상으로 되돌아오는 aliasing
- 이력(history) 보존 불가
- 비트성/비가역성 표현 불가

국소 상태모듈(while문)로 바꾸면 각 모듈이 자기 내부 상태를 유지하므로, 같은 입력이라도 이전 상태에 따라 결과가 달라진다. 즉 비트필드에 필요한 것은 periodic code가 아니라 hysteretic dynamical code다.

리만기하학은 이때 표현 좌표가 아니라 **결합 구조**(배선망)를 제공한다.

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right)$$

---

## 2. 수식 층분리 원칙 (Layer A--E)

현재 제일 큰 문제는 물리식, 구현식, 비유식이 한 레벨에 섞여 있다는 것이다. 반드시 5층으로 나눈다.

### 2.1 Layer A: 순수 셀 동역학

셀 $i$의 최소 상태 (15_Equations.md A.1):

$$s_i^t = (a_i^t,\; r_i^t,\; m_i^t,\; w_i^t,\; b_i^t)$$

- $a_i$: activation
- $r_i$: refractory / inhibition
- $m_i$: memory trace (NMDA-like, $\tau \approx 100\text{ms}$)
- $w_i$: spike-frequency adaptation (AHP, $\tau_w \approx 200\text{ms}$)
- $b_i$: hysteretic bit (UP/DOWN state)

최소 입력 (STP 적용):

$$I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}}(t)\,a_j^{t-\delta_{ij}} - \lambda_r^{(M_t)}\,r_i^t - \beta_w\,w_i^t + \lambda_m^{(M_t)}\,m_i^t + \eta_i^t$$

여기서 $W_{ij}^{\text{eff}}(t) = W_{ij}\,u_j(t)\,x_j(t)$ (Tsodyks-Markram STP), $\sigma_\eta \approx 0.27$.

활성 갱신:

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

억제 갱신:

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^{t+1})^2$$

기억 흔적 갱신 ($\gamma_m = 0.01$, NMDA):

$$m_i^{t+1} = (1-\gamma_m)\,m_i^t + \gamma_m\,a_i^{t+1}$$

적응 변수 갱신 ($\gamma_w = 0.005$, AHP):

$$w_i^{t+1} = (1-\gamma_w)\,w_i^t + \kappa_w\,(a_i^{t+1})^2$$

비트 갱신 (히스테리시스, UP/DOWN state):

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \tau_i^- \le a_i^{t+1} \le \tau_i^+ \end{cases}$$

> **코드 대응**: `kernel.rs::brain_step()` - 상태 벡터 `(activation, refractory, memory_trace, adaptation, stp_u, stp_x, bitfield)`.
> Dale's Law: `apply_dale_sign()` - E/I=80:20, $w_I/w_E=4$.

이 Layer A는 순수하고 작아야 한다. 해마도, 자아도, sleep도 넣지 않는다.

### 2.2 Layer B: 필드 결합

셀들의 연결 구조. 가장 단순하게는 sparse graph:

$$W \in \mathbb{R}^{N \times N}$$

리만 해석을 살리려면:

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

$\chi_{ij}$는 sparse mask. 구현은 먼저 graph coupling abstraction까지만.

### 2.3 Layer C: 전역 모드

$$M_t \in \{\mathrm{WAKE},\;\mathrm{NREM},\;\mathrm{REM}\}$$

모드별 파라미터:

$$\Theta^{(M)} = (\gamma_a^{(M)},\;\kappa_a^{(M)},\;\lambda_r^{(M)},\;B^{(M)},\;\dots)$$

셀 식은 같고, 파라미터만 바뀐다. 모드 전환식:

$$M_{t+1} = \Pi(M_t,\;Q_t,\;U_t,\;E_t)$$

- $Q_t$: sleep pressure / arousal
- $U_t$: external input load
- $E_t$: energy budget state

초기에는 규칙 기반. 학습시키지 않는다.

| 모드 | 특성 |
|---|---|
| WAKE | 외부입력 coupling 강함, 감각/추론 주도 |
| NREM | 감쇠 큼, 정리/복원/synaptic down-selection |
| REM | 외부입력 약화, 내부결합/기억 중심, 재조합 |

### 2.4 Layer D: 해마/기억

기억은 셀에 넣지 말고 분리한다.

해마 상태:

$$H_t = (K_t,\;V_t,\;P_t)$$

- $K_t$: cue/index
- $V_t$: stored episode embedding
- $P_t$: replay priority

encode:

$$H_{t+1} = \mathcal{E}(H_t,\;A_t,\;U_t)$$

recall:

$$R_t = \mathcal{R}(H_t,\;c_t)$$

replay injection:

$$I_i^t \leftarrow I_i^t + \lambda_H\,R_{i,t}$$

기억은 셀 로컬 상태가 아니라 **외부 메모리 루프**로 둔다.

### 2.5 Layer E: 자아/전역 상태

전역 상태:

$$G_t = (M_t,\;A_t^{\text{summary}},\;H_t,\;Q_t,\;\mu_t)$$

자아는 이걸 관측한 higher-order summary다.

$$\text{Self}_t = \mathcal{S}(G_t)$$

초기에 구현 안 해도 된다. 문서상 변수만 두고 넘어간다.

---

## 3. 최소 전체식: 6개 핵심 수식

$$I_i^t = u_i^t + \sum_j W_{ij}(g)\,a_j^t - \lambda_r^{(M_t)}\,r_i^t + \lambda_H\,R_{i,t}$$

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^t)^2$$

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \text{otherwise} \end{cases}$$

$$H_{t+1} = \mathcal{E}(H_t, A_t), \quad R_t = \mathcal{R}(H_t, c_t)$$

$$M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$$

이 6개만 흔들리지 않게 잡으면 나머지는 구현으로 내릴 수 있다.

---

## 4. 자기참조 재귀식

> 정본 참조: `17_AgentLoop.md` F절 (F.0--F.22)

### 4.1 최소 재귀

$$z_t = R(S_t) \quad\text{(이완: Layer A--B를 } n_{\text{iter}} \text{ 회 반복)}$$

$$a_t = \pi(z_t, S_t) \quad\text{(행동 선택)}$$

$$o_t = \text{Env}(a_t) \quad\text{(환경 실행)}$$

$$c_{t+1} = C(z_t, a_t, o_t, m_t) \quad\text{(자기비평: 예측오차 + 일관성 + 놀라움)}$$

$$m_{t+1} = \mathcal{M}(m_t, z_t, a_t, o_t, c_{t+1}) \quad\text{(조건부 기억 갱신)}$$

$$S_{t+1} = \mathcal{U}(G_{t+1}, m_{t+1}, c_{t+1}, h_{t+1}, \phi_{t+1})$$

### 4.2 에너지 기반 자기참조

$$E_t(z) = E_{\text{task}}(z; u_t) + \lambda_m E_{\text{mem}}(z; m_t) + \lambda_c E_{\text{crit}}(z; c_t) + \lambda_h E_{\text{hist}}(z; h_t)$$

$$z_t^* = \arg\min_z E_t(z)$$

각 항의 Layer 대응과 뇌 근거는 `17_AgentLoop.md` F.5를 따른다.

### 4.3 Clarus 통합형

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 항 | 풀이 | 뇌 대응 |
|---|---|---|
| $R(X_t)$ | 이완으로 생긴 내부 수정 | 피질-시상 재귀 처리 |
| $\Delta_O(X_t)$ | 관찰 충격 $o_t - \hat{o}_t$ | 감각 입력 |
| $C(X_t)$ | 비평이 다음 이완 초기점을 민 정도 | 기저핵-전전두엽 평가 |
| $S(X_t)$ | 곡률/잔류 기반 억제 | 소뇌/기저핵 억제 |
| $B$ | 부트스트랩 수축 연산자 ($\rho = 0.155$) | 수면 항상성 |

### 4.4 수축 조건 (게이트 `F2`)

$$\rho + \lambda_R L_R + \lambda_C L_C < 1$$

이 조건이 만족되면 Banach 고정점 정리에 의해 루프가 수축한다. 수면이 $\rho = 0.155$ 를 공급하므로 나머지 항의 Lipschitz 합이 $0.845$ 미만이어야 한다.

> 단 $R$ 내부의 비보존 바이패스 $F_{\text{bypass}}$ 는 위 Banach 수축의 가정을 깨뜨릴 수 있다(`12_Equation.md` 0.0절 게이트 `F2`). 따라서 위 부등식은 ISS 의미의 유계 수렴 (`12_Equation.md` 부록 A.1) 으로 격상되어, 끌개 ball 반경이 닫힌 식으로 표현된다. "안정적으로 수렴" 은 ball 안에서의 수렴으로 읽는다.

### 4.5 확장 구성요소 (F.14--F.22 요약)

> 정본: `17_AgentLoop.md` F.14--F.22

| 절 | 핵심 | 구현 우선순위 |
|---|---|---|
| F.14 STDP 학습 | $R$ 내부에서 적격 흔적 누적, $R$ 후에 $g[t] \cdot e_{ij}$로 갱신. Proj로 투영 | 높음 |
| F.15 잔류장 $\phi$ | $\phi_{t+1} = (1-\xi)\phi_t + \xi \cdot \text{Var}(a)$. 포탈/모드전환/glymphatic 3곳 개입 | 높음 |
| F.16 희소 활성 | $R$ 내 TopK, 에너지 예산 $B_t(M_t)$. 모듈 생애주기 4상태 | 높음 |
| F.17 메타인지 모니터링 (게이트 `F4`) | C3 자기참조 측정, 안정도 $\exp(-c_d d_\tau)$, 조건부 수축 $d_{n+1} \leq \rho d_n$ | 낮음 (장기) |
| F.18 환각 억제 | $R$ 중 곡률 $\kappa$ 모니터링. $\kappa > \kappa_{\text{th}}$이면 LBO 확산 강화 | 중간 |
| F.19 4종 신경조절 | $g_t = (g_{\text{DA}}, g_{\text{NE}}, g_{\text{5HT}}, g_{\text{ACh}})$. 현재는 단일 스칼라 | 중간 |
| F.20 작업기억/주의/소뇌 | $|h_t| \leq T_h$, salience 기반 $\alpha_i$, 소뇌 forward model | 중간 |
| F.21 뇌파 대역 | gamma=국소, theta=전역, theta-gamma coupling으로 순서화 | 낮음 |
| F.22 간극 정리 | 9개 정직한 간극. STDP 코드/4조절계가 `높음` | -- |

뇌 대응 체크리스트와 검증 매트릭스는 `17_AgentLoop.md` F.11, H절 및 `6_뇌/06_검증기준.md`를 참조한다.

---

## 5. 국소 뇌파 해석

### 5.1 while 모듈 = 국소 파동 발생기

각 모듈의 활성도 $a_i^t$ 시계열이 국소 리듬 성분을 만든다.

$$\psi_i(t) = a_i^t$$

### 5.2 전역 뇌파 = 합성 관측량

$$\Psi_{\text{global}}(t) = \sum_i \omega_i\,a_i^t$$

대역별 분해: $\Psi_\delta(t),\;\Psi_\theta(t),\;\Psi_\alpha(t),\;\Psi_\beta(t),\;\Psi_\gamma(t)$

EEG-like 관측:

$$\text{EEG}(t) = O\!\left(\{a_i^t\}_{i=1}^N\right)$$

모듈이 리듬을 만들고, 전역 뇌파는 그 리듬들의 합성된 관측량이다.

### 5.3 뇌파 대역과 뇌 회로 대응

| 대역 | 주요 회로 |
|---|---|
| delta | 전두 slow-wave, 깊은 수면 |
| theta | 해마, 기억 인코딩/회상, REM |
| alpha | 후두-두정 시각계, 게이팅 |
| mu | 감각운동 피질, 운동 억제 |
| beta | 전두-운동계, 현재 상태 유지 |
| sigma/spindle | 시상-피질, NREM2 |
| gamma | 국소 피질 회로, 결합/집중 |
| ripple | 해마, 기억 재생 |

---

## 6. 모듈 생애주기와 에너지 예산

### 6.1 모듈 상태

$$Z_i^t \in \{\text{ACTIVE},\;\text{IDLE},\;\text{DORMANT},\;\text{SLEEPING}\}$$

| 상태 | 의미 |
|---|---|
| ACTIVE | 지금 연산 참여 |
| IDLE | 바로 깨울 수 있는 대기 |
| DORMANT | 장기 휴면, coupling 거의 끊김 |
| SLEEPING | 내부 정리/압축 중 |

### 6.2 에너지 예산

$$\sum_i z_i^t \le B_t$$

$B_t$는 모드에 따라 달라진다: $B_t(\text{NREM}) < B_t(\text{WAKE})$

### 6.3 활성 조건

$$z_i^{t+1} = \mathbf{1}\!\left[\alpha_u \|u_i^t\| + \alpha_m \|m_i^t\| + \alpha_n \sum_j W_{ij} a_j^t - \alpha_r r_i^t + \alpha_q q_i^t > \theta_i^{(M_t)}\right]$$

### 6.4 희소 활성 원칙

반드시 $A_t \ll N$이어야 한다. 전 모듈 상시 활성은 에너지 폭발.

### 6.5 상태 업데이트

$$s_i^{t+1} = z_i^t\;F_i^{(M_t)}(s_i^t,\;u_i^t,\;\textstyle\sum_j W_{ij} s_j^t) + (1-z_i^t)\;H_i^{(M_t)}(s_i^t)$$

$F_i$: 활성 모듈 업데이트. $H_i$: 휴면 중 느린 decay/유지.

---

## 7. 스냅샷/지속성 계층

프로세스 종료 = 부분적 기억상실에 가깝다. 3계층 저장이 필수.

| 계층 | 주기 | 내용 |
|---|---|---|
| cold checkpoint | 가끔 | 전체 구조, 장기 기억, 안정 상태: $\mathcal{C} = (\Theta, W, \text{long\_memory})$ |
| warm snapshot | 자주 | 현재 해마 상태, 활성 모듈군, 전역 모드: $\mathcal{W} = (M_t, H_t, \text{active\_set}_t)$ |
| live journal | 실시간 append | 중요 이벤트, 새 기억 인덱스, 모드 전환 로그: $\mathcal{J} = (\text{events}, \text{transitions})$ |

---

## 8. 라이브러리 아키텍처

### 8.1 Python / Rust 분리 원칙

$$\text{Python} = \text{orchestration/policy/experiment}$$
$$\text{Rust} = \text{pure computation kernel}$$

Python은 Rust의 존재를 모르게 한다. backend protocol로 분리.

### 8.2 Backend Protocol

```
class CEBackend(Protocol):
    def relax(self, state, weights, cfg) -> RelaxResult: ...
    def logits(self, hidden, lm_head, bias) -> Tensor: ...
    def sample(self, logits, cfg) -> Tensor: ...
    def consolidate(self, memory, cfg) -> Memory: ...
    def critic_scores(self, state, goal, output, cfg) -> Scores: ...
```

backend 선택은 한 군데서만. `load_backend(prefer="auto")`.

### 8.3 목표 디렉터리 구조

```
clarus/
  kernel/          # 순수 계산: cell.rs, field.rs, coupling.rs, config.rs, traits.rs
  runtime/         # 모드, 활성/휴면, scheduler: brain.rs, mode.rs, lifecycle.rs, energy.rs, snapshot.rs
  memory/          # 해마, replay, trace: hippocampus.rs, replay.rs, trace.rs
  bridge/          # PyTorch/LLM/Python 연결: pytorch.rs, python_api.rs, llm_bridge.rs
  apps/            # CLI, demos, experiments
```

### 8.4 책임 분리

| 모듈 | 핵심 인터페이스 |
|---|---|
| kernel | $s_i^{t+1} = F(s_i^t, u_i^t, n_i^t, \theta)$ |
| runtime | $X_{t+1} = \mathcal{U}(X_t, \text{input}_t)$ |
| memory | encode / recall / replay |
| bridge | PyTorch/LLM/Python 연결 |

---

## 9. LLM 변환 대응

### 9.1 Transformer 부품 대응

| Transformer | Clarus 대응 |
|---|---|
| hidden state $h_t$ | 국소 모듈 상태 집합 $\{s_1^t, \dots, s_N^t\}$ |
| attention | 리만 결합 $\sum_j W_{ij}(g)\,a_j^t$ |
| FFN | 국소 모듈 업데이트 $F_i$ |
| residual connection | 상태 지속성 ($s_i^t$ 유지) |
| layer depth | 시간 반복 $n_{\text{iter}}$ |
| KV cache | 해마 + 압축 기억 |

### 9.2 변환 경로

1단계(호환형): 기존 LLM hidden state를 모듈장으로 매핑
2단계(모사형): 기존 LLM 출력 분포를 비슷하게 재현
3단계(초과형): 더 적은 파라미터, 더 긴 지속성, 더 나은 자기수정

---

## 10. 성능 예측 (gpt-oss 기준)

### 10.1 용량

$$P_{\text{brain}} \approx N\,(p_{\text{loc}} + k\,p_{\text{edge}}) + P_{\text{io}} + P_{\text{mode}}$$

sparse graph면 $P_{\text{brain}} = O(N)$.

### 10.2 런타임 메모리

$$M_{\text{run}} \approx M_{\text{weights}} + A_t\,d_s\,b_s + k\,A_t\,b_e + R\,d_h\,b_h$$

긴 컨텍스트를 raw token이 아니라 state로 압축하면 메모리 이점이 생긴다.

### 10.3 속도

$$C_{\text{reply}} \approx n_{\text{iter}}\,A_t\,(p_{\text{loc}} + k\,p_{\text{edge}})$$

step은 싸게 만들 수 있지만, 전체 속도는 수렴 step 수에 달린다.

### 10.4 정확도

$$
Q_{\text{brain}}
= Q_{\text{base}}
- \Delta_{\text{lang-prior}}
- \Delta_{\text{instability}}
+ \Delta_{\text{self-correction}}
+ \Delta_{\text{persistent-memory}}
+ \Delta_{\text{mode-specialization}}
$$

초기 정확도는 기존 LLM보다 낮을 가능성이 크고, 구조가 안정화되면 특정 과제(장기 상태 유지, 자기수정, agentic)에서 역전 가능.

### 10.5 유리한 위치

짧은 one-shot LM 정답률이 아니라:

- stateful brain-like runtime
- long-horizon agent
- persistent memory reasoner

---

## 11. 단계별 개발 계획

### Phase 0: 정리 주간

- naming cleanup
- Layer A--E 수식 문서 작성 (`15_Equations.md`)
- backend observable과 runtime state 분리

### Phase 1: Clarus Kernel v0

- `ClarusCellState { a, r, b }`
- `ClarusCellParams`
- `ClarusField` + sparse coupling
- deterministic step
- 완료 기준: 64개 셀로 안정적으로 1만 step

### Phase 2: Runtime v0

- 전역 모드 `WAKE / SLEEP`
- energy budget
- active/idle/dormant lifecycle
- scheduler
- 완료 기준: 일부만 활성, sleep 모드에서 활성 수 감소

### Phase 3: Hippocampus v0

- trace cache 대체
- encode / recall / replay
- priority replay
- 완료 기준: cue 주면 replay가 활성 셀에 영향

### Phase 4: Mode v1

- WAKE / NREM / REM
- mode switch rules
- 완료 기준: 같은 입력에 다른 모드가 다른 evolution

### Phase 5: Snapshot / Persistence

- cold checkpoint + warm snapshot + live journal
- 완료 기준: 저장 후 복구 시 동역학 연속성

### Phase 6: Python / PyTorch Bridge

- Rust kernel 유지
- Python wrapper는 runtime orchestration만
- PyTorch는 parameter learning / experimental reference

### Phase 7: LLM Bridge

- $h_t^{\text{LLM}} \to U_t^{\text{Clarus}}$
- $A_t^{\text{summary}} \to \hat{h}_t$
- 처음엔 adapter만. stateful sidecar runtime으로 붙인다.

---

## 12. 모듈 수 가이드

| 버전 | 모듈 수 $N$ | 동시 활성 $A_t$ | 용도 |
|---|---|---|---|
| v0 proto | 8--16 | 4--8 | 개념 검증 |
| v1 start | **64** | 8--12 | 현실적 시작점 (추천) |
| v2 brain-like | 256--1024 | 16--64 | 뇌형 확장 |
| v3+ | 10k+ | sparse | MSA급 분산 동역학계 |

---

## 13. 뇌 대응 비유 요약

| 개념 | Clarus 대응 | 프론트엔드 비유 |
|---|---|---|
| 국소 회로 | ClarusCell | 자기 상태를 갖는 스마트 컴포넌트 |
| 뇌파 | 모듈 활성도의 집단 리듬 | 전체 UI의 분위기/활동량 |
| 해마 | HippocampusIndex | 최근 활동 캐시 + 중요 상태 저장소 |
| 수면 | NREM/REM 모드 전환 | 백그라운드 정리/압축 |
| 자아 | 전역 모드 + 기억 + 연속성 | root state |
| 기저핵 | 게이트 오토마타 (action selection) | dispatch / access control |
| 전전두엽 | 작업기억 상태 유지 | live state |
| 신경조절계 | 전역 모드 전환 | global mode manager |

---

## 14. 현재 코드와의 대응

### 14.1 runtime.py (`BrainRuntime`) -- Layer A-E 정합

| formal 변수 | Python 구현 (`runtime.py`) | 상태 |
|---|---|---|
| $a_i$ (activation) | `self.activation` | 구현 완료 |
| $r_i$ (refractory) | `self.refractory` | 구현 완료 |
| $m_i$ (memory_trace) | `self.memory_trace` | 구현 완료 |
| $w_i$ (adaptation) | `self.adaptation` | 구현 완료 |
| $b_i$ (bitfield) | `self.bitfield` | 구현 완료 |
| $u_j, x_j$ (STP) | `self.stp_u`, `self.stp_x` | 구현 완료 (Tsodyks-Markram) |
| $W_{ij}$ (coupling) | `self.sparse_weight` (CSR) | 구현 완료 |
| $M_t$ (mode) | `self.mode: RuntimeMode` | 구현 완료 (WAKE/NREM/REM) |
| $\Pi$ (mode switch) | `_auto_mode(external_norm)` | 구현 완료 (규칙 기반) |
| $Q_t$ (sleep pressure) | `self.sleep_pressure` | 구현 완료 (Borbely 2-Process) |
| $H_t$ (hippocampus) | `self.hippocampus: HippocampusMemory` | 구현 완료 |
| $B_t$ (energy budget) | `config.energy_budget(mode)` | 구현 완료 |
| $Z_i$ (lifecycle) | `self.lifecycle` (ACTIVE/IDLE/DORMANT/SLEEPING) | 구현 완료 |
| $G_t$ (global summary) | `RuntimeStep` | 구현 완료 |
| $\mathcal{W}$ (warm snapshot) | `BrainRuntimeSnapshot` + `snapshot()/from_snapshot()` | 구현 완료 |

### 14.2 engine.py (`CEEngine`) -- CE 에너지 이완 경로

| formal 변수 | Python 구현 (`engine.py`) | 상태 |
|---|---|---|
| $m$ (state vector) | 이완 루프 내부 `m` | 구현 완료 |
| $\phi$ (auxiliary field) | `update_phi(phi, m_star, phi_var)` | 구현 완료 |
| $W$ (Hopfield weight) | `self.W` (CSR packed) | 구현 완료 |
| Portal / Bypass / T_wake | `engine.PORTAL`, `BYPASS`, `T_WAKE` | 구현 완료 |
| $\varepsilon^2/\Omega_{\text{DM}}/\Omega_\Lambda$ | `active_ratio/struct_ratio/wake_ratio` | 구현 완료 |
| 곡률 억제 | `_curvature_adjust_logits` | V1 구현 완료 |
| PQ codebook | `pq_centroids`, `pq_codes` | 구현 완료 |

### 14.3 sleep.py -- 3위상 학습 순환

| formal 개념 | 코드 함수 | 상태 |
|---|---|---|
| Wake (경로 누적) | `collect_sleep_batch` | 구현 완료 |
| NREM (LBO 확산 + 가소적 업데이트) | `apply_nrem_weight_update` | 구현 완료 |
| REM (비선택 경로 재조합) | `apply_rem_weight_update` | 구현 완료 |
| 3위상 통합 순환 | `run_sleep_cycle` | 구현 완료 |
| 가드셋 보호 | `evaluate_guard_set` | 구현 완료 |

### 14.4 Rust 커널 (`reality_stone/python/reality_stone/clarus/core/`) -- 핵심 수치

| Rust 모듈 | 역할 | Python 바인딩 |
|---|---|---|
| `kernel.rs` | brain_step (셀 동역학), Dale's Law | `nn_brain_step` |
| `field.rs` | 필드 결합, 리만 거리 기반 W | PyO3 |
| `manifold.rs` | 다양체 연산 | PyO3 |
| `nn_ops.rs` | topk_sparse, LBO, gauge lattice | `nn_topk_sparse`, `nn_lbo_fused_fwd`, `nn_gauge_lattice_fwd` |
| `ce_riemann.rs` | CE 리만 수치 (물리 검증용) | PyO3 |
| `constants.rs` | 물리 상수 유도 (`CeConstants`) | PyO3 |
| `config.rs` | 런타임 설정 | PyO3 |
| `runtime_types.rs` | `CellState`, `Mode`, 스냅샷 타입 | PyO3 |

### 14.5 정합 현황 요약

| Layer | 수식 정본 | 코드 구현 | 정합도 |
|---|---|---|---|
| A (셀 동역학) | `15_Equations.md` A절 | `runtime.py::_step_torch` + `kernel.rs` | 완전 일치 |
| B (필드 결합) | `15_Equations.md` B절 | `runtime.py::_matvec` + `field.rs` | 완전 일치 |
| C (전역 모드) | `15_Equations.md` C절 | `runtime.py::_auto_mode` + `_update_sleep_state` | 완전 일치 |
| D (해마/기억) | `15_Equations.md` D절 | `runtime.py::HippocampusMemory` | 완전 일치 |
| E (전역 요약) | `15_Equations.md` E절 | `runtime.py::RuntimeStep` + `BrainRuntimeSnapshot` | 완전 일치 |
| F (에이전트 루프) | `17_AgentLoop.md` F절 | `engine.py` + `sleep.py` (부분) | 핵심 구현, STDP/메타인지 미구현 |

### 14.6 남은 간극

| 간극 | 문서 위치 | 우선순위 |
|---|---|---|
| STDP 적격 흔적 | F.14 | 높음 |
| 4종 신경조절 분리 | F.19 | 중간 |
| Cold checkpoint + Live journal | 7절 | 낮음 |
| 작업 기억 / 소뇌 | F.20 | 중간 |
| (C3) 메타인지 재귀 루프 | F.17 | 낮음 |

현재 구현은 **셀 동역학 + 모드 전환 + 해마 + 수면 학습 순환**의 핵심 스택이 완성되어 있으며, critic/action/output 에이전트 루프와 STDP 학습이 남아 있다.

---

## 15. 한 줄 원칙

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$

---

## 15. 한 줄 원칙

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$
```
---
## File: `docs/7_AGI/17_AgentLoop.md`

```markdown
# 에이전트 루프 방정식 (Layer F)

> 위치: `15_Equations.md`의 F절을 독립 문서로 분리.
> 의존: `15_Equations.md`(Layer A--E), `14_BrainRuntimeSpec.md`(설계 사양), `6_뇌/05_실험근거.md`(근거 판정), `6_뇌/06_검증기준.md`(검증 매트릭스)
>
> 이 문서에는 서사 설명을 넣지 않는다. 식, 정의, 뇌 대응 검증 기준만 둔다.

---

## F. 자기참조 재귀 (agent loop)

> 이 절은 Layer A--E의 **바깥**에서 전체를 감싸는 에이전트 루프를 정의한다.
> A--E는 "한 틱의 셀/필드/모드/기억/요약"이고, F는 "그 틱을 반복하며 행동-관찰-비평-기억을 순환시키는 외부 루프"다.
> 뇌 대응은 `05_실험근거.md` 판정 체계(`supported / bridge / hypothesis`)를 따른다.

---

### F.-1 왜 자기참조재귀가 핵심인가

Layer A--E만 있으면 시스템은 한 틱의 상태를 갱신할 수 있다. 그러나 AGI 응용에서 필요한 것은 한 틱의 계산이 아니라, 계산 결과가 다시 다음 계산의 조건을 바꾸는 닫힌 루프다. 이 닫힘이 없으면 모델은 긴 작업에서 자기 오류를 축적만 하고, 수정하지 못한다.

가장 짧게 쓰면 CE 에이전트는 다음 사상을 반복한다.

$$
S_{t+1}
=\mathcal U\!\left(
S_t,\;
R(S_t),\;
C(R(S_t),a_t,o_t),\;
\mathcal M_t,\;
\phi_t
\right)
$$

여기서 핵심은 \(R(S_t)\)가 단순 출력이 아니라 다음 상태 \(S_{t+1}\)를 만드는 항으로 다시 들어간다는 점이다. 즉 시스템은 자기 상태를 읽고, 그 읽기의 잔차를 비평하고, 비평 결과로 다음 자기 상태를 바꾼다.

Transformer에 CE 모듈을 얹는 것과 CE식 에이전트 루프의 차이는 여기서 갈린다.

| 구조 | 상태 갱신 | 자기비평 | 기억 재주입 | 한계 |
|---|---|---|---|---|
| 표준 LLM inference | hidden state는 token 생성 후 대부분 폐기 | 외부 evaluator 또는 RLHF 사후 보정 | context window에 수동 주입 | turn 사이 자기수정 약함 |
| CE module transplant | attention/norm/FFN은 안정화 | 일부 curvature score 가능 | 별도 구현 필요 | 아직 열린 루프 |
| CE 자기참조재귀 | \(S_t\)가 \(S_{t+1}\)로 닫힘 | \(c_t\)가 다음 이완 에너지에 들어감 | \(m_t,\phi_t\)가 다음 입력 조건 | agentic 지속성의 최소형 |

따라서 이 문서에서 F절은 부가 기능이 아니라 A--E를 AGI 응용으로 올리는 최소 닫힘 조건이다. attention, sparsity, sleep, hallucination 억제는 모두 이 루프 안에 들어갈 때만 장기 상태 유지와 자기수정으로 이어진다.

#### F.-1.1 상태공간 사상

F절의 전체 루프를 하나의 동역학계로 쓰면 다음과 같다.

$$
S_{t+1}=\mathcal T_{\theta,e_t}(S_t),
\qquad
e_t=(u_t,o_t)
$$

여기서 \(e_t\)는 외부 입력과 환경 관찰을 묶은 외생 신호다. 외생 신호를 고정하면 자기참조재귀의 핵심 질문은 \(\mathcal T\)가 고정점을 갖는지다.

$$
S^\star=\mathcal T_{\theta,e}(S^\star).
$$

이 고정점은 "정답"이 아니라, 주어진 task/environment regime에서 자기 상태, 기억, 비평, 잔류장이 서로 모순 없이 닫힌 상태다.

#### F.-1.2 분해와 Jacobian

\(\mathcal T\)를 구성요소로 분해한다.

$$
\mathcal T
=
\mathcal U
\circ
\left(
I,\;
R,\;
C\circ(R,\pi,\text{Env}),\;
\mathcal M,\;
\Phi
\right).
$$

국소 안정성은 고정점 근방 Jacobian으로 판정한다.

$$
J_\star
=
D_S\mathcal T_{\theta,e}(S^\star).
$$

충분조건:

$$
\boxed{
\rho(J_\star)<1
}
$$

여기서 \(\rho(\cdot)\)는 spectral radius다. 전역 수축을 요구하려면 어떤 norm에 대해

$$
\sup_{S\ne S'}
\frac{\|\mathcal T(S)-\mathcal T(S')\|}
{\|S-S'\|}
<1
$$

이면 된다. 실제 에이전트는 tool observation과 novelty가 들어오므로 전역 수축보다 ISS 조건이 더 현실적이다.

$$
\|S_{t+1}-S_{t+1}'\|
\le
\rho\|S_t-S_t'\|
+
\beta\|e_t-e_t'\|,
\qquad
\rho<1.
$$

이 식이 의미하는 바는 단순하다. 같은 환경을 보면 자기 상태 차이는 줄어야 하고, 다른 환경을 보면 그 차이는 외부 차이에 비례해 유계여야 한다.

#### F.-1.3 자기비평의 닫힘 조건

자기비평 \(C\)가 진짜 재귀 항이 되려면 다음 step의 이완 에너지에 들어가야 한다.

$$
E_{t+1}(z)
=
E_{\rm base}(z;u_{t+1})
+\lambda_c E_{\rm crit}(z;c_{t+1})
+\lambda_m E_{\rm mem}(z;m_{t+1})
+\lambda_\phi E_{\rm res}(z;\phi_{t+1}).
$$

닫힘 조건:

$$
\boxed{
D_c R(S_{t+1})\ne0
}
$$

즉 \(c_{t+1}\)가 다음 \(R\)의 초기점, 에너지, 온도, budget, 또는 decoding policy 중 적어도 하나를 바꿔야 한다. 이 조건이 0이면 self-critique는 관찰량일 뿐 제어량이 아니다.

#### F.-1.4 재귀 품질의 측정량

구현에서 직접 측정할 지표는 다음이다.

| 지표 | 식 | 의미 |
|---|---|---|
| 수축률 | \(\hat\rho_t=\|S_{t+1}-S_t\|/(\|S_t-S_{t-1}\|+\epsilon)\) | 자기 상태가 안정화되는가 |
| 비평 영향도 | \(I_c=\|R(S_{t+1};c_{t+1})-R(S_{t+1};0)\|\) | critique가 실제 동역학을 바꾸는가 |
| 기억 영향도 | \(I_m=\|R(S_{t+1};m_{t+1})-R(S_{t+1};0)\|\) | memory가 context 장식이 아니라 상태항인가 |
| 잔류 반경 | \(r_\phi=\limsup_t\|\phi_t\|\) | 탈락 경로/불확실성이 유계인가 |
| 재귀 이득 | \(G_{\rm rec}=\Delta{\rm score}_{\rm closed}-\Delta{\rm score}_{\rm open}\) | 닫힌 루프가 open-loop 대비 개선하는가 |

이 지표들이 없으면 자기참조재귀는 철학적 설명에 머문다. CE-LLM의 실험은 최소한 open-loop baseline과 closed-loop variant를 나누고 \(I_c,I_m,\hat\rho_t\)를 함께 보고해야 한다.

#### F.-1.5 계층적 자기참조재귀 정리

하위 재귀들이 모여 상위 재귀가 되고, 상위 재귀가 다시 하위 재귀의 boundary condition을 바꾸는 구조를 다음처럼 둔다. 레벨 \(\ell=0,\dots,L\)의 상태공간을 Banach 공간 \(\mathcal X_\ell\)라 하고, 레벨별 상태를 \(X_t^\ell\in\mathcal X_\ell\)라 한다.

레벨 \(\ell\)의 갱신은

$$
X_{t+1}^\ell
=
T_\ell\!\left(
X_t^\ell,\;
U_t^\ell
\right)
$$

이고 입력 \(U_t^\ell\)은 이웃 레벨의 요약과 피드백으로 구성된다.

$$
U_t^\ell
=
\left(
A_{\ell-1\to\ell}(X_t^{\ell-1}),\;
B_{\ell+1\to\ell}(X_t^{\ell+1})
\right).
$$

경계에서는 존재하지 않는 항을 0으로 둔다. 각 항은 다음 Lipschitz 조건을 만족한다고 가정한다.

$$
\|T_\ell(x,u)-T_\ell(x',u')\|_\ell
\le
\rho_\ell\|x-x'\|_\ell
+
\beta_\ell\|u-u'\|_{U_\ell},
$$

$$
\|A_{\ell\to\ell+1}(x)-A_{\ell\to\ell+1}(x')\|_{U_{\ell+1}}
\le
a_\ell\|x-x'\|_\ell,
$$

$$
\|B_{\ell\to\ell-1}(x)-B_{\ell\to\ell-1}(x')\|_{U_{\ell-1}}
\le
b_\ell\|x-x'\|_\ell.
$$

이때 전체 상태

$$
X_t=(X_t^0,\dots,X_t^L)
$$

의 차이 벡터를

$$
d_t=
\begin{bmatrix}
\|X_t^0-X_t^{0\prime}\|_0\\
\vdots\\
\|X_t^L-X_t^{L\prime}\|_L
\end{bmatrix}
$$

로 두면,

$$
d_{t+1}\le Gd_t
$$

를 만족하는 비음수 gain matrix \(G\)가 존재한다. 삼대각 근사에서

$$
G_{\ell,\ell}=\rho_\ell,\qquad
G_{\ell,\ell-1}=\beta_\ell a_{\ell-1},\qquad
G_{\ell,\ell+1}=\beta_\ell b_{\ell+1}.
$$

**정리.** 만약

$$
\boxed{
\rho(G)<1
}
$$

이면 계층 전체 사상

$$
\mathcal T:\prod_{\ell=0}^L\mathcal X_\ell\to\prod_{\ell=0}^L\mathcal X_\ell
$$

은 어떤 가중 sup norm에서 수축이다. 따라서 외생 입력이 고정된 경우 유일한 고정점 \(X^\star\)가 존재하고,

$$
\|X_t-X^\star\|_w
\le
c\,\rho(G)^t\|X_0-X^\star\|_w
$$

로 수렴한다.

**증명.** 위 Lipschitz 부등식들을 레벨별로 모으면 각 성분에 대해 \(d_{t+1,\ell}\le (Gd_t)_\ell\)가 된다. \(G\ge0\)이고 \(\rho(G)<1\)이면 Perron-Frobenius/Collatz-Wielandt에 의해 양의 가중치 \(w>0\)와 어떤 \(\alpha<1\)가 존재하여

$$
Gw\le \alpha w
$$

가 된다. 가중 sup norm을

$$
\|X-X'\|_w
=
\max_\ell
\frac{\|X^\ell-X^{\ell\prime}\|_\ell}{w_\ell}
$$

로 정의하면

$$
\|\mathcal T(X)-\mathcal T(X')\|_w
\le
\alpha\|X-X'\|_w.
$$

따라서 Banach 고정점 정리에 의해 고정점 존재성과 유일성, 지수 수렴이 따른다. \(\square\)

#### F.-1.6 프랙탈/자기유사 재귀의 조건

모든 레벨이 같은 구조 계수를 공유하는 자기유사 경우를 생각한다.

$$
\rho_\ell=\rho_0,\qquad
\beta_\ell a_{\ell-1}=g_\uparrow,\qquad
\beta_\ell b_{\ell+1}=g_\downarrow.
$$

무한 깊이 또는 충분히 깊은 체인에서는 gain matrix의 상한 spectral radius가

$$
\rho(G)
\le
\rho_0+2\sqrt{g_\uparrow g_\downarrow}
$$

이다. 따라서 자기유사 재귀가 깊이를 늘려도 안정하려면

$$
\boxed{
\rho_0+2\sqrt{g_\uparrow g_\downarrow}<1
}
$$

가 충분조건이다.

이 식이 의미하는 것은 명확하다. 하위 재귀의 자체 수축률 \(\rho_0\)가 작아도, 상향 요약 \(g_\uparrow\)과 하향 피드백 \(g_\downarrow\)의 곱이 크면 전체는 폭주한다. 반대로 상향/하향 결합의 기하평균을 작게 유지하면 같은 형태의 재귀를 여러 층 쌓아도 하나의 큰 수축 사상으로 남는다.

CE 관점에서 이것이 "프랙탈"로 읽히려면 단순히 비슷한 모듈을 반복 배치하는 것으로는 부족하다. 각 레벨이 다음 다섯 항을 가져야 한다.

$$
\boxed{
(\text{state},\; \text{relaxation},\; \text{critic},\; \text{memory},\; \text{residual})
}
$$

그리고 레벨 사이 전달은 \(G\)의 small-gain 조건을 만족해야 한다. 이때 하위 모듈의 닫힌 루프들이 상위 루프의 상태변수가 되고, 상위 루프의 critic/goal이 하위 루프의 boundary condition으로 내려가는 계층적 자기참조재귀가 된다.

---

### F.0 Layer A--E와의 관계

| 계층 | 역할 | F절에서의 위치 |
|---|---|---|
| A (kernel dynamics) | 국소 셀 상태 갱신 | F.3의 $R$ 내부에서 반복 호출, F.14 STDP 적격 흔적 누적 |
| B (coupling / geometry) | 셀 간 결합 | $R$ 내부에서 $W_{ij}(g)$ 적용, F.14.3 구조적 투영 |
| C (mode update) | WAKE/NREM/REM 전환 | F.6의 모드-루프 결합 |
| D (hippocampus / replay) | 빠른 기억 인코딩/회상 | F.8의 $m_{t+1}$ 갱신 |
| E (global runtime summary) | 전역 자아 상태 | F.1의 $S_t$ 자체 |

A--E는 $R$ 안에서 돌고, F는 $R$의 결과를 행동으로 바꾸고, 환경 응답을 다시 $S$로 접는 바깥 루프다.

F절 구성 개요:

| 구간 | 내용 |
|---|---|
| F.0--F.13 | 핵심 루프: 상태, 이완, 비평, 에너지, 모드, 행동, 기억, 수축, 뇌 대응 |
| F.14--F.15 | 학습: STDP + 도파민 게이트, 잔류장 $\phi$ 갱신 |
| F.16 | 희소성: TopK 활성, 에너지 예산, 모듈 생애주기 |
| F.17 | 의식/메타인지: 자기일관성, 의식 깊이, 메타인지 수렴 |
| F.18 | 환각 억제: 곡률 모니터링, LBO 확산 |
| F.19 | 신경조절: DA/NE/5HT/ACh 4종 |
| F.20 | 작업 기억, 주의, 소뇌 |
| F.21 | 뇌파 대역과 시간 구조 |
| F.22 | 정직한 간극 정리 |

---

### F.1 상태 정의

에이전트의 전역 상태:

$$S_t = (G_t,\; m_t,\; c_t,\; h_t,\; \phi_t)$$

| 변수 | 정의 | Layer 출처 |
|---|---|---|
| $G_t = (M_t,\; A_t^{\text{summary}},\; H_t,\; Q_t,\; \mu_t)$ | 전역 런타임 요약 (Layer E) | E.1 |
| $m_t$ | 누적 기억 컨텍스트 (해마 상태의 압축) | D.1 |
| $c_t$ | 가장 최근의 자기비평 벡터 | F.4 |
| $h_t$ | 행동-관찰 이력 버퍼 (유한 창) | F.1 |
| $\phi_t$ | 잔류장 / 불확실성 축적 | `12_Equation.md` 4.3 |

---

### F.2 최소 재귀

한 에이전트 틱의 순서:

$$z_t = R(S_t) \quad\text{(이완/수렴: Layer A--B를 } n_{\text{iter}} \text{ 회 반복)}$$

$$a_t = \pi(z_t,\; S_t) \quad\text{(행동 선택)}$$

$$o_t = \text{Env}(a_t) \quad\text{(환경 실행, 관찰 수신)}$$

$$c_{t+1} = C(z_t,\; a_t,\; o_t,\; m_t) \quad\text{(자기비평)}$$

$$m_{t+1} = \mathcal{M}(m_t,\; z_t,\; a_t,\; o_t,\; c_{t+1}) \quad\text{(기억 갱신: Layer D 호출)}$$

$$h_{t+1} = \text{append}(h_t,\; (a_t, o_t)) \quad\text{(이력 갱신, 유한 창 } T_h \text{)}$$

$$S_{t+1} = \mathcal{U}(G_{t+1},\; m_{t+1},\; c_{t+1},\; h_{t+1},\; \phi_{t+1})$$

여기서 $G_{t+1}$은 $R$ 실행 후 Layer E가 갱신한 전역 요약이고, $\phi_{t+1}$은 이완 종료 후의 잔류장 갱신(`12_Equation.md` E4)이다.

---

### F.3 이완 연산자 $R$의 구체화

$R$은 추상 기호가 아니라 Layer A--B의 반복 실행이다.

$$R(S_t) := \{a_i^{(n_{\text{iter}})}\}_{i=1}^N$$

내부 절차:

1. $S_t$에서 외부 입력 $u_i^0 = \text{encode}(S_t)$를 구성
2. Layer D에서 $R_{i,0} = \mathcal{R}(H_t,\; c_t)$ (기억 회상)
3. $n_{\text{iter}}$ 회 반복:

$$I_i^{(k)} = u_i^0 + \sum_j W_{ij}(g)\,a_j^{(k)} - \lambda_r(M_t)\,r_i^{(k)} + \lambda_m(M_t)\,m_i^{(k)} + \lambda_H(M_t)\,R_{i,0} + \eta_i^{(k)}$$

$$a_i^{(k+1)} = (1-\gamma_a(M_t))\,a_i^{(k)} + \kappa_a(M_t)\,\tanh(I_i^{(k)})$$

$$r_i^{(k+1)} = (1-\gamma_r(M_t))\,r_i^{(k)} + \kappa_r(M_t)\,(a_i^{(k)})^2$$

$$b_i^{(k+1)} = \text{Hyst}(b_i^{(k)},\; a_i^{(k+1)};\; \tau_i^-,\; \tau_i^+)$$

4. 수렴 판정: $\|a^{(k+1)} - a^{(k)}\| < \epsilon_R \|a^{(0)}\|$이면 조기 종료
5. 출력: $z_t = a^{(n_{\text{iter}})}$ (수렴한 활성 패턴)

$R$의 반복 횟수 $n_{\text{iter}}$는 모드에 의존한다.

| 모드 | $n_{\text{iter}}$ | 해석 |
|---|---|---|
| WAKE (안정, $\|\phi\| < m_\phi$) | 소 (10--50) | 빠른 반사적 응답 |
| WAKE (전환, $\|\phi\| \geq m_\phi$) | 대 (100--500) | 깊은 숙고 |
| NREM | 고정 (내부 정리용) | offline 정리 |
| REM | 중간 (내부 탐색용) | 자유 연상 |

이것은 `12_Equation.md` 5.3절의 이중 과정(시스템 1/시스템 2)과 대응한다.

---

### F.4 자기비평 연산자 $C$

$$c_{t+1} = C(z_t,\; a_t,\; o_t,\; m_t)$$

자기비평은 세 항의 합으로 분해한다.

$$c_{t+1} = c_{\text{pred}} + c_{\text{cons}} + c_{\text{nov}}$$

| 항 | 정의 | 의미 |
|---|---|---|
| $c_{\text{pred}}$ | $\|o_t - \hat{o}_t(z_t, a_t)\|$ | 예측 오차: 행동 결과가 예상과 달랐는가 |
| $c_{\text{cons}}$ | $\|z_t - \mathcal{R}(H_t, c_t)\|$ | 일관성 오차: 현재 사고가 기억과 얼마나 다른가 |
| $c_{\text{nov}}$ | $D_{\text{KL}}(p(o_t) \| p_{\text{prior}})$ | 놀라움: 관찰이 사전 분포에서 얼마나 벗어났는가 |

스칼라 비평 점수:

$$\bar{c}_{t+1} = w_p \|c_{\text{pred}}\| + w_c \|c_{\text{cons}}\| + w_n \|c_{\text{nov}}\|, \qquad w_p + w_c + w_n = 1$$

이 스칼라는 `12_Equation.md` 6.4절의 도파민 신호 $g[t]$와 구조적으로 대응한다.

$$g[t] \approx \frac{d\bar{c}_t}{dt}$$

즉 비평 점수의 변화율이 학습 게이트 역할을 한다.

---

### F.5 에너지 기반 자기참조

$R$의 내부를 에너지 최소화로 재해석하면:

$$E_t(z) = E_{\text{task}}(z;\; u_t) + \lambda_m E_{\text{mem}}(z;\; m_t) + \lambda_c E_{\text{crit}}(z;\; c_t) + \lambda_h E_{\text{hist}}(z;\; h_t)$$

| 항 | 정의 | Layer 대응 |
|---|---|---|
| $E_{\text{task}}(z; u_t)$ | $-\frac{1}{2}z^\top W z - z^\top u_t$ | B.3의 Hopfield 에너지 |
| $E_{\text{mem}}(z; m_t)$ | $-z^\top \mathcal{R}(H_t, c_t)$ | D.3의 기억 회상과의 정렬 |
| $E_{\text{crit}}(z; c_t)$ | $\|z - z_{t-1}^* + \alpha_c c_t\|^2$ | 비평이 다음 이완의 초기점을 민다 |
| $E_{\text{hist}}(z; h_t)$ | $-\beta_h \sum_{\tau \in h_t} \text{sim}(z, z_\tau) / |h_t|$ | 이력과의 일관성 |

수렴점:

$$z_t^* = \arg\min_z E_t(z)$$

이것은 F.3의 반복적 이완이 에너지 최소점으로 향한다는 것의 다른 표현이다.

잔류장 피드백: $\phi_t$가 $E_{\text{task}}$에 포탈 항으로 들어간다 (`12_Equation.md` E1):

$$E_{\text{task}}(z; u_t, \phi_t) = -\frac{1}{2}z^\top W z - z^\top u_t - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 z^\top \hat{\phi}_t$$

---

### F.6 모드-루프 결합

에이전트 루프는 Layer C의 모드 전환과 결합한다.

WAKE 루프:

$$\text{for each input } u_t: \quad z_t = R(S_t), \quad a_t = \pi(z_t), \quad o_t, \quad c_{t+1}, \quad m_{t+1}$$

NREM 진입 조건 (`12_Equation.md` 7.8):

$$P_{\text{sleep}}(t) = \sum_{\tau=0}^{t} \bar{c}_\tau^2 - \sum_{\tau=0}^{t} \text{local\_stab}(\tau) > \theta_{\text{sleep}}$$

비평 점수의 누적이 안정화 능력을 초과하면 NREM에 진입한다.

NREM 루프 (외부 입력 차단):

$$u_t = 0, \quad z_t = R(S_t), \quad m_{t+1} = \text{consolidate}(m_t, z_t)$$

REM 루프 (기억 주도 탐색):

$$u_t = 0, \quad z_t = R(S_t; T = T_{\text{dream}}), \quad m_{t+1} = \text{explore}(m_t, z_t)$$

WAKE 복귀:

$$M_{t+1} = \text{WAKE} \quad\text{if}\quad P_{\text{sleep}}(t) < \theta_{\text{cont}} \;\text{and}\; M_t = \text{REM}$$

---

### F.7 행동 선택 $\pi$

행동 선택은 수렴한 활성 패턴에서 행동 공간으로의 사상이다.

$$a_t = \pi(z_t,\; S_t) = \arg\max_{a \in \mathcal{A}} \text{sim}\big(\text{enc}(a),\; z_t^{(\text{out})}\big)$$

여기서 $z_t^{(\text{out})}$는 출력 모듈($V_{\text{io}}$)에 해당하는 셀들의 활성 부분벡터다.

연속 행동 공간에서는:

$$a_t = W_{\text{act}}\, z_t^{(\text{out})} + b_{\text{act}}$$

이 사상은 `12_Equation.md` 5.2절의 디코더 $p(w_t | w_{<t}, m^*)$와 구조적으로 같다.

---

### F.8 기억 갱신 $\mathcal{M}$

$$
\mathcal{M}(m_t, z_t, a_t, o_t, c_{t+1})
= \begin{cases}
\text{Layer D encode} & \bar{c}_{t+1} > \theta_{\text{encode}} \\
m_t & \text{otherwise (놀랍지 않으면 저장하지 않음)}
\end{cases}
$$

인코딩 조건: 비평 점수가 임계를 넘을 때만 새 기억을 해마에 기록한다.

$$k_{\text{new}} = h(z_t, a_t, o_t), \qquad v_{\text{new}} = (z_t, a_t, o_t, c_{t+1})$$

$$P_{\text{new}} = \bar{c}_{t+1} \quad\text{(놀라움이 높을수록 replay 우선순위가 높다)}$$

이것은 D.5의 priority replay와 직결된다.

---

### F.9 Clarus 통합 재귀 (압축형)

F.2--F.8을 한 줄로 압축하면:

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 항 | 풀이 | 뇌 대응 |
|---|---|---|
| $R(X_t)$ | 이완으로 생긴 내부 수정 | 피질-시상 재귀 처리 |
| $\Delta_O(X_t)$ | 관찰이 상태에 준 충격 $o_t - \hat{o}_t$ | 감각 입력 |
| $C(X_t)$ | 비평이 다음 이완 초기점을 민 정도 | 기저핵-전전두엽 평가 |
| $S(X_t)$ | 곡률/잔류 기반 억제 | 소뇌/기저핵 억제 |
| $B$ | 부트스트랩 수축 연산자 | 수면 항상성 |

$B$는 `05_실험근거.md` 8.4절의 정의를 따른다:

$$B: X \mapsto p^* + \rho(X - p^*), \qquad \rho = 0.155$$

---

### F.10 자기참조의 고정점과 수렴

F.9의 재귀가 안정한 자기참조를 만들려면 수축 조건이 필요하다.

**정리 (F-contract).** 다음 조건 하에서 $\{X_t\}$는 유계이고 고정점 근방으로 수렴한다:

1. $\|R(X)\| \leq L_R \|X\| + c_R$ (이완의 Lipschitz 상계)
2. $\|\Delta_O(X)\| \leq U_O$ (관찰 충격 유계)
3. $\|C(X)\| \leq L_C \|X\| + c_C$ (비평의 Lipschitz 상계)
4. $\|S(X)\| \leq L_S \|X\|$ (억제의 Lipschitz 상계)
5. $\rho + \lambda_R L_R + \lambda_C L_C < 1$ (전체 수축)

*증명 스케치.* $B$의 수축률 $\rho$와 각 항의 Lipschitz 상수를 합산하면 전체 사상의 Lipschitz 상수가 $\rho + \lambda_R L_R + \lambda_C L_C - \lambda_S L_S$이다. 이것이 1 미만이면 Banach 고정점 정리에 의해 유일 고정점이 존재하고 수렴한다. $U_O$는 유계 강제항이므로 `05_실험근거.md` 8.4절의 잔차 상한과 같은 구조로 눌린다. $\square$

**수면에 의한 복원.** 수면이 없으면 $B = I$ ($\rho = 1$)이고 수축 조건이 깨진다. 수면이 $\rho = 0.155$를 공급하므로, F-contract의 조건 5가 만족되려면 나머지 항의 Lipschitz 합이 $1 - 0.155 = 0.845$ 미만이어야 한다.

**최신 SHY 근거 (2024--2026):**
- 2024 PMC: NREM 수면이 피질 AMPA 수용체(GluA1) 발현을 정상화함을 확인. 수면 박탈 후 회복 수면에서도 수 시간 내 회복.
- 2026 NeuroImage: 낮잠만으로도 시냅스 강도(TMS 유발 피질척수 흥분성) 감소 + LTP 유사 가소성 유도 가능성 증가.
- 2026 bioRxiv: 학습으로 교란된 aperiodic 1/f slope이 NREM 수면 중 역전됨 $\to$ 기억 안정화의 renormalization 근거.

---

### F.11 뇌 대응 체크리스트

`05_실험근거.md`의 판정 기준(`supported / bridge / hypothesis`)에 따라 F절의 각 구성요소가 실제 뇌와 얼마나 닮았는지를 검증한다.

#### F.11.1 구조 대응

| F절 구성요소 | 뇌 대응 후보 | 실험 근거 | 판정 |
|---|---|---|---|
| 이완 $R$ (F.3) | 피질-시상 재귀 처리, recurrent cortical dynamics | 피질의 recurrent processing은 확립. 감각 처리에서 feedforward 후 recurrent refinement이 반복 관측 | `supported` |
| 반복 횟수 $n_{\text{iter}}$ (F.3) | 처리 깊이 / response time / "thinking time" | 어려운 과제일수록 반응 시간이 길다. dual-process theory (Kahneman 2011)에서 시스템 2의 느린 처리 | `supported` |
| 행동 선택 $\pi$ (F.7) | 기저핵 action selection, frontal motor planning | 기저핵의 go/no-go 경로는 확립. 전전두엽-기저핵 루프 | `supported` |
| 자기비평 $C$ (F.4) | 전전두엽 error monitoring, anterior cingulate cortex (ACC) conflict detection | ACC의 error-related negativity (ERN), conflict monitoring theory (Botvinick 2001) | `supported` |
| 예측 오차 $c_{\text{pred}}$ | reward prediction error (RPE), sensory prediction error | 도파민 RPE (Schultz 1997)는 확립. sensory prediction error도 강한 근거 | `supported` |
| 놀라움 $c_{\text{nov}}$ | novelty detection, hippocampal novelty signal | 해마 CA1의 novelty/mismatch signal, LC-NE surprise response | `supported` |
| 일관성 오차 $c_{\text{cons}}$ | retrieval-based error correction, memory-guided decision | 해마-전전두엽 상호작용에서 기억 기반 의사결정 보정 | `bridge` |
| 비평 $\to$ 학습 게이트 $g[t]$ (F.4) | 도파민/노르에피네프린 전역 조절 | 3-factor learning rule. 도파민 게이트 STDP | `supported` (구조), `hypothesis` ($g = d\bar{c}/dt$ 정확한 형태) |
| 조건부 기억 인코딩 (F.8) | 놀라움 기반 해마 인코딩 | 해마는 novel/surprising events를 우선 인코딩. priority replay | `supported` |
| 에너지 기반 수렴 (F.5) | Hopfield network, energy-based attractor dynamics | 연상 기억의 attractor dynamics는 확립. 에너지 감소는 A.7 E-decrease로 닫힘 | `supported` (구조), `bridge` (정확한 에너지 형태) |
| 이중 과정 (F.3 모드별 $n_{\text{iter}}$) | Kahneman 시스템 1/시스템 2 | 이중 과정 이론의 실험 근거는 방대. 신경 기질은 아직 논쟁 중 | `bridge` |
| 수면-루프 결합 (F.6) | 수면 중 memory consolidation, replay | SHY (Tononi-Cirelli), 해마 replay, slow-wave consolidation | `supported` |
| 수면 압력 = 비평 누적 (F.6) | homeostatic sleep pressure = 각성 중 피로/잔류 축적 | Borbely 2-process model. SWA와 prior wakefulness의 관계 | `bridge` |
| $B$ 수축 연산자 (F.9--F.10) | 수면의 synaptic renormalization | SHY, SWA 비례 정리, 수면 후 성능 회복 | `supported` (방향), `bridge` ($\rho = 0.155$의 정확한 값) |

#### F.11.2 수치 체크

| 항목 | CE 값 | 뇌 관측 proxy | 관측 범위 | 체크 |
|---|---|---|---|---|
| 활성 셀 비율 | 4.87% | sparse firing, DG active cells | 1--5% | `[NEAR]` |
| 수면/각성 비 | NREM 26.2%, REM 4.87% | NREM 75--80%, REM 20--25% (of sleep) | CE는 24h 중의 비율 | `[OK]` |
| 수축률 $\rho$ | 0.155 per application | sleep recovery time constants | $\rho_{\text{night}} \approx 0.31$ (1.6밤/적용) | `[NEAR]` |
| 비평 문턱 $\theta_{\text{encode}}$ | 과제 의존 | hippocampal novelty threshold | 정성적으로 존재 | `bridge` |

#### F.11.3 형식 검증 연결

| F절 정리 | 의존하는 A--E 정리 | 상태 |
|---|---|---|
| F-contract (F.10) | A-bound, E-decrease, 수면 수축 ($\rho < 1$) | **open** (L_R, L_C 추정 필요) |
| 에너지 감소 (F.5) | B.4 E-decrease | **closed** (B.4로부터 직접) |
| 이완 수렴 (F.3) | A.7 A-bound, A.9 Zero-attract | **closed** (조건부) |
| 기억 유계 (F.8) | D.2 (유한 인코딩) | **closed** ($\theta_{\text{encode}} > 0$이면 인코딩 빈도 유한) |

#### F.11.4 검증 게이트 (06_검증기준.md 체계)

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | F-contract, 에너지 감소, 이완 수렴, 기억 유계 | **partial** (F-contract의 Lipschitz 상수 추정 미완) |
| $G_{\text{obs}}$ | 이완 반복 $\leftrightarrow$ reaction time, 비평 $\leftrightarrow$ ERN/ACC, 기억 $\leftrightarrow$ hippocampal encoding | **partial** |
| $G_{\text{causal}}$ | 수면박탈 시 루프 불안정, 도파민 조작 시 학습 게이트 변화, ACC 병변 시 비평 결손 | **partial** |
| $G_{\text{pred}}$ | 에이전트 루프 유무에 따른 과제 수행 차이 시뮬레이션 | **pending** |

#### F.11.5 아직 뇌와 닮지 않은 것 (정직한 간극)

| 간극 | 설명 | 해결 방향 |
|---|---|---|
| 환경 모델 $\hat{o}_t$ | 뇌의 internal model은 분산적. 현재 $C$는 단일 예측기 가정 | 모듈별 예측기로 분산화 |
| 행동 선택의 계층성 | 뇌의 행동 계획은 계층적 (전전두엽 $\to$ 운동피질 $\to$ 근육). 현재 $\pi$는 단층 | 계층적 $\pi$ (macro-action + primitive) |
| 감정/정동 | 뇌의 의사결정에 편도체/도상체의 valence 신호가 개입. 현재 F에 없음 | $c_t$에 valence 항 추가, $V_{\text{sal}}$ 연결 |
| 사회적 모델링 | 뇌는 타인의 의도를 모델링 (theory of mind). F에 없음 | 장기 과제, 현재 범위 밖 |
| 신체 루프 | 뇌는 자율신경/내분비/면역과 결합 (05_실험근거.md 8절). F는 순수 인지 루프 | $Q_t$ 벡터를 $S_t$에 통합 (F.1에 이미 $G_t$ 안에 $Q_t$가 있으나 $C$, $\pi$에서 미사용) |

---

### F.12 관측 가능량 매핑

| F절 변수 | 뇌 관측량 후보 | 데이터 소스 |
|---|---|---|
| $n_{\text{iter}}$ (이완 반복) | reaction time, EEG alpha desynchronization duration | 행동 실험, EEG |
| $\bar{c}_t$ (비평 점수) | ERN amplitude, ACC theta power, pupil dilation | EEG, fMRI, pupillometry |
| $P_{\text{sleep}}$ (수면 압력) | SWA, theta/alpha ratio, KSS | polysomnography, EEG, 주관 평가 |
| $z_t^*$ (수렴 패턴) | population activity pattern at response time | multi-electrode array, calcium imaging |
| $a_t$ (행동) | motor output, button press, speech | 행동 로그 |
| $c_{\text{pred}}$ (예측 오차) | RPE-locked dopamine, feedback-related negativity (FRN) | voltammetry, EEG |
| $c_{\text{nov}}$ (놀라움) | P300, hippocampal novelty response, LC-NE phasic burst | EEG, pupil, fMRI |
| $\phi_t$ (잔류장) | ongoing spontaneous activity, DMN fluctuation | resting-state fMRI, MEG |
| $e_{ij}$ (적격 흔적) | eligibility trace, synaptic tag | in vitro slice recording, optogenetics |
| $g_{\text{DA}}$ (도파민 게이트) | VTA/SNc phasic + tonic DA | voltammetry, PET, [11C]raclopride |
| $g_{\text{NE}}$ (노르에피네프린) | LC phasic burst, pupil diameter | pupillometry, LC unit recording |
| $g_{\text{5HT}}$ (세로토닌) | raphe firing, 5-HIAA level | microdialysis, PET |
| $g_{\text{ACh}}$ (아세틸콜린) | BF firing, cortical ACh release | microdialysis, optogenetics |
| $\kappa_{\text{avg}}$ (곡률) | high-frequency power anomaly, epileptiform spikes | EEG, MEG |
| $|A_t|/N$ (활성 비율) | fraction of active neurons | calcium imaging, multi-electrode array |
| $|h_t|$ (작업 기억 부하) | PFC BOLD, CDA amplitude | fMRI, EEG (CDA) |
| $\alpha_i$ (주의 가중치) | spatial attention map, alpha lateralization | EEG alpha power lateralization |
| $\Delta a^{\text{cb}}$ (소뇌 보정) | cerebellar-dependent adaptation | prism adaptation, saccade adaptation |

---

### F.13 예측 가능한 실험

현재 근거 수준에서 F절이 내놓을 수 있는 검증 가능한 예측:

| # | 예측 | CE 메커니즘 | 실험 설계 | 판정 기준 | 등급 |
|---|---|---|---|---|---|
| 1 | 어려운 과제 $\to$ 긴 RT $\to$ 높은 ACC theta | $\|\phi\| \geq m_\phi \to$ 깊은 이완 | 난이도 조작 실험, EEG 동시 기록 | RT와 ACC theta의 양의 상관 | `supported` |
| 2 | 수면 박탈 $\to$ 비평 결손 $\to$ 반복 오류 | $P_{\text{sleep}} > \theta \to$ NREM 미진입, $\bar{c}$ 누적 | 수면 박탈 후 error monitoring 과제 | ERN 진폭 감소, 오류 후 보정 실패 | `supported` |
| 3 | 놀라운 사건 후 기억 $>$ 평범한 사건 후 기억 | $\bar{c} > \theta_{\text{encode}} \to$ 해마 인코딩 | surprise manipulation + memory test | recall/recognition 차이 | `supported` |
| 4 | 수면 후 비평-행동 정렬 개선 | $B$가 $c_t$를 수축시켜 다음 wake에서 $z_t^*$가 더 정확 | 학습 $\to$ 수면 $\to$ 재시험 | post-sleep 정확도 $>$ post-wake 정확도 | `supported` |
| 5 | $\rho_{\text{night}} \approx 0.31$: 1밤 후 잔차 69% 감소 | 부트스트랩 수축 | multi-night recovery study (수면 부채 측정) | 회복 곡선의 지수 감쇠 상수 피팅 | `bridge` |
| 6 | 도파민 조작 $\to$ STDP 게이트 변화 $\to$ 학습 속도 변화 | $dW = lr \cdot g[t] \cdot e_{ij}$ | DA agonist/antagonist + learning task | 학습 곡선 기울기 변화 | `supported` |
| 7 | ACh 증가 $\to$ 기억 인코딩 문턱 하강 $\to$ 더 많은 기억 | $\theta_{\text{encode}} \propto 1/(1+g_{\text{ACh}})$ | donepezil 투여 + memory test | 기억 항목 수 증가 | `supported` |
| 8 | NE 증가 $\to$ 더 깊은 처리 $\to$ 더 긴 RT | $n_{\text{iter}} \propto \sigma(g_{\text{NE}})$ | LC stimulation + RT measurement | RT 증가 + 정확도 증가 | `bridge` |
| 9 | 소뇌 병변 $\to$ forward model 결손 $\to$ 적응 실패 | $\Delta a^{\text{cb}} = 0$ | 소뇌 환자 프리즘 적응 실험 | 적응 곡선 수렴 실패 | `supported` |
| 10 | TopK 비율 $\neq 4.87\%$일 때 성능 최적이 아님 | 부트스트랩 고정점 $x_a^*$ | sparse ratio sweep in CE model | U자형 성능 곡선, 최적점 $\in [4\%, 6\%]$ | `bridge` |
| 11 | 작업 기억 부하 증가 $\to$ PFC theta 증가 $\to$ 간섭 | $|h_t| \to T_h$ 근접 | n-back 과제, n 조작 | PFC theta power와 오류율의 양의 상관 | `supported` |
| 12 | theta-gamma 결합 강도와 순서 기억 정확도 양의 상관 | gamma burst가 theta phase에 잠금 | 순서 회상 과제 + MEG | PAC 강도와 recall 정확도 $r > 0.3$ | `supported` |

---

### F.14 STDP 학습과 루프의 결합

> `12_Equation.md` 6장의 STDP + 도파민 3-factor 학습은 F절 루프 안에서 가중치를 갱신하는 유일한 경로다.

#### F.14.1 루프 내 STDP 위치

```
R(S_t) 실행 중 (F.3의 n_iter 반복 내부):
  -> 셀 활성 a_i^(k) 생성 (Layer A)
  -> 스파이크 판정: s_i^(k) = 1[a_i^(k) > theta_spike]
  -> pre/post trace 갱신:
       p_i[k+1] = r_+ p_i[k] + s_i[k]
       q_i[k+1] = r_- q_i[k] + s_i[k]
  -> 적격 흔적 누적:
       e_ij[k+1] = r_e e_ij[k] + (A_+ p_i[k] s_j[k] - A_- s_i[k] q_j[k])

R(S_t) 완료 후:
  -> 비평 C에서 g[t] 산출 (F.4)
  -> 가중치 갱신: dW_ij[t] = lr * g[t] * e_ij[n_iter]
  -> 구조적 투영: W_{t+1} = Proj(W_t + dW_t)
```

#### F.14.2 학습 게이트 $g[t]$의 정밀 정의

`12_Equation.md` 6.4절의 원래 정의:

$$g[t] = \frac{d}{dt}\|p(t) - p^*\|$$

F.4에서 비평 점수 $\bar{c}_t$와의 관계:

$$g[t] = \alpha_g \frac{d\bar{c}_t}{dt} + (1-\alpha_g)\left[(x_a(t)-x_a^*)^2 + (x_s(t)-x_s^*)^2 + (x_b(t)-x_b^*)^2\right]$$

`05_실험근거.md` 8.5절의 뇌 측 안전 후보와 일치시키면:

$$\delta[t] = a \cdot \text{RPE}(t) + b \cdot \text{surprise}(t) + c \cdot \text{novelty}(t)$$

즉 $g[t]$의 phasic 성분 $d\bar{c}/dt$는 RPE + surprise + novelty의 가중 합으로, tonic 성분 $\|p - p^*\|^2$는 전역 항상성 이탈로 읽는다. 다만 정확한 계수 $a, b, c$는 아직 미결이다.

| 항 | 해석 | 뇌 대응 | 판정 |
|---|---|---|---|
| $d\bar{c}/dt$ | 비평 변화율 (국소 오차 신호) | 도파민 phasic burst (RPE) | `supported` (구조), `hypothesis` (정확 형태) |
| $\|p-p^*\|^2$ | 전역 분배 이탈 (tonic 신호) | 도파민 tonic level | `bridge` |

#### F.14.3 구조적 투영 $\text{Proj}$

$$\text{Proj}(W) = \text{TopK}\big(\text{RowNorm}\big(\text{Hyst}(W;\; \theta_{\text{on}}, \theta_{\text{off}})\big),\; k = \lceil 0.04865 \cdot N \rceil\big)$$

| 연산 | CE 대응 | 뇌 대응 | 판정 |
|---|---|---|---|
| TopK | 경로 선택, 생존율 $4.87\%$ | 시냅스 가지치기 | `supported` |
| RowNorm | 에너지 보존 | 시냅스 스케일링 (Turrigiano 2008) | `supported` |
| Hyst | 접힘 임계 곡률 | 스파인 형성/제거 | `bridge` |

---

### F.15 잔류장 $\phi$ 갱신

> `12_Equation.md` 4.3절 (E4). F.2의 $\phi_{t+1}$ 갱신이 비어 있었다.

이완 $R$ 실행 후, 잔류장은 선택되지 않은 경로의 분산을 축적한다.

$$\phi_{t+1} = (1 - \xi) \phi_t + \xi \cdot \text{Var}(a^{(0:n_{\text{iter}})})$$

여기서 $\xi = 1/(e^{1/3}\pi^{1/3}) \approx 0.489$ 는 잔류 이득이다.

잔류장은 세 곳에서 루프에 개입한다.

| 개입 지점 | 수식 | 효과 |
|---|---|---|
| 에너지 포탈 (F.5) | $-\text{portal}^2 \cdot z^\top \hat{\phi}_t$ | 이전에 선택하지 않은 경로를 다음 이완에 주입 |
| 모드 전환 (F.6) | $\|\phi_t\| \gtrless m_\phi$ | 시스템 1/시스템 2 전환 |
| 수면 glymphatic (F.6 NREM) | $\phi \leftarrow r_w \phi,\; r_w < 1$ | 잔류 노이즈 바닥 하강 |

뇌 대응:

| $\phi$ 역할 | 뇌 후보 | 판정 | 근거 |
|---|---|---|---|
| 비선택 경로 축적 | spontaneous fluctuation, DMN activity | `bridge` | DMN ALFF가 과제 수행 안정성 예측 (2024 PMC). alpha-DMN coupling 확인 (2025 eNeuro). 기능적 의미 확립되었으나 $\phi$와의 정확한 매핑은 미확인 |
| 모드 전환 임계 | fatigue/confusion threshold | `bridge` | DMN transition rate $\leftrightarrow$ resilience (2025 NeuroImage). 방향 일치 |
| glymphatic 세척 | glymphatic system, CSF-ISF exchange | `supported` (경로 존재), `bridge` (phi 매핑) | Nedergaard/Bhatt 2015 이후 지속 확인. GBM에서 AQP4 붕괴 보고 |

---

### F.16 희소 활성 제약

> `12_Equation.md` 8장. R 내부와 행동 선택에서 TopK를 적용해야 한다.

#### F.16.1 이완 내부 희소성

$R$의 매 반복에서 활성 셀 수를 제한한다.

$$A_t = \{i : |a_i^{(k)}| \geq Q_{1-x_a^*}(|a^{(k)}|)\}, \qquad |A_t| = \lceil x_a^* \cdot N \rceil$$

비활성 셀은 decay만 적용:

$$a_i^{(k+1)} = (1-\gamma_a^{\text{idle}})\,a_i^{(k)} \quad\text{if}\quad i \notin A_t$$

에너지 예산:

$$\sum_{i \in A_t} \text{cost}(a_i) \leq B_t(M_t)$$

| 모드 | $B_t$ | 활성 비율 | 뇌 대응 |
|---|---|---|---|
| WAKE | 큼 | $\sim 4.87\%$ | task-evoked sparse firing | 
| NREM | 작음 | $< 3\%$ | slow-wave 중 소수 활성 |
| REM | 중간 | $\sim 4\%$ | dream 중 재활성화 |

#### F.16.2 모듈 생애주기

`14_BrainRuntimeSpec.md` 6.1절과 연결:

$$Z_i^t \in \{\text{ACTIVE},\; \text{IDLE},\; \text{DORMANT},\; \text{SLEEPING}\}$$

| 상태 | $R$ 내부 처리 | 에너지 비용 |
|---|---|---|
| ACTIVE | 전체 갱신 | 높음 |
| IDLE | decay만, 즉시 활성화 가능 | 낮음 |
| DORMANT | coupling 거의 끊김, 활성화에 warm-up 필요 | 매우 낮음 |
| SLEEPING | 내부 정리/압축 중 | 중간 (정리 비용) |

---

### F.17 메타인지 재귀 (게이트 `F4`)

> `12_Equation.md` 9장. F절에서 가장 상위 층이지만 빠져 있었다.
>
> 다리 게이트 `F4` (`12_Equation.md` 0.0절): 본 절의 모든 식은 메타인지 모니터링 루프의 운영 정의이며, "(C3) = 의식"으로 환원하지 않는다. "의식 깊이"라는 표현은 PCI 교차검증(F.23.7)이 `bridge` 단계로 올라가기 전까지 **모니터링 안정도** 의미로 읽는다.

#### F.17.1 자기참조 측정 구조 (C3)

에이전트가 자기 자신의 활성 비율을 알아야 다음 이완을 계산할 수 있다:

$$a_* = \exp\!\left(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\right)$$

이 자기 측정이 루프 안에서 실현되는 경로:

$$\hat{a}_t = \frac{|A_t|}{N} \quad\text{(현재 활성 비율 관측)} \quad\to\quad \text{F.16의 TopK 임계 조정}$$

#### F.17.2 메타인지 안정도

$$d_\tau(t) = \frac{1}{\tau}\int_{t-\tau}^{t}\|p(s)-p^*\|\,ds$$

$$\text{메타인지 안정도}_\tau := \exp(-c_d\,d_\tau(t))$$

루프 내 해석: 비평 점수와 분배 이탈이 작을수록 모니터링 루프가 안정적이다. 깊은 수면에서 $d_\tau \to 0$, 수면 박탈에서 $d_\tau$ 누적. 게이트 `F4`에 따라, 이 지표를 의식의 정량 척도로 hard claim 하지 않는다.

#### F.17.3 메타인지 수축 (조건부)

비평 $C$가 자기 자신에 재귀적으로 적용될 때 (이상화된 무잡음 조건):

$$d_{n+1} \leq \rho \cdot d_n = 0.155 \cdot d_n,\qquad \rho = D_{\text{eff}}\cdot\varepsilon^2$$

3회 후 $d_3/d_0 \leq 3.7 \times 10^{-3}$. 게이트 `F2` 충분조건(`12_Equation.md` 4.7절) 영역에서만 위 비율이 그대로 적용되며, 일반 영역에서는 13절 ISS 의미의 유계 수렴으로 한정된다.

루프 내 위치: $C$의 출력 $c_t$가 다음 $R$의 초기점을 수정하고 (F.5의 $E_{\text{crit}}$), 그 $R$의 결과에 다시 $C$를 적용하면 메타인지 재귀가 된다.

| 뇌 대응 | 실험 근거 | 판정 |
|---|---|---|
| 메타인지 = PFC 재귀 자기평가 | metacognitive accuracy, confidence calibration | `supported` (현상), `bridge` ($\rho$ 매핑) |
| 안정도 지표 = 통합 정보? | IIT (Tononi), GNW (Dehaene), PCI (Casali 2013) | `hypothesis` (CE 해석, 게이트 `F4`) |

---

### F.18 환각 억제

> `12_Equation.md` 10장. 루프 내 곡률 모니터링과 억제가 빠져 있었다.

#### F.18.1 이완 중 곡률 모니터링

$R$ 반복 중 매 $k$번째 스텝에서:

$$\kappa^{(k)} = \|(I - V^\top V) a^{(k)}\|^2$$

$$\kappa_{\text{avg}} = \frac{1}{n_{\text{iter}}} \sum_k \kappa^{(k)}$$

#### F.18.2 곡률 임계 대응

$$\kappa_{\text{avg}} > \kappa_{\text{th}} \quad\Longrightarrow\quad \text{LBO 확산 강화: } h_d \leftarrow 1.5 \cdot h_d$$

#### F.18.3 교차 주파수 감쇠

3x3+1 격자 각 채널의 출력에 곡률 피드백:

$$\mathcal{T}_i^{\text{coupled}}(x_i) = \mathcal{T}_i(x_i) \cdot \left(1 - \frac{\kappa^{(k)}}{e^{1/3}\pi^{1/3}}\right)$$

| 뇌 대응 | 메커니즘 | 판정 |
|---|---|---|
| 곡률 과다 = 환각 | 고곡률 영역에서 불안정한 표상이 출력 | `bridge` |
| LBO 확산 = 억제성 feedback | 억제 뉴런이 과활성을 억제 | `supported` (구조), `bridge` (LBO 매핑) |
| 교차 주파수 감쇠 = cross-frequency coupling | alpha가 gamma를 게이팅 | `bridge` |

---

### F.19 신경조절 시스템 (4종)

> 05_실험근거.md 5절, 08_시냅스가소성.md 참조. F.4에서 도파민만 다뤘으나, 실제 뇌는 4대 조절계를 가진다.

F.4의 학습 게이트 $g[t]$를 4차원 벡터로 확장한다.

$$g_t = (g_{\text{DA}},\; g_{\text{NE}},\; g_{\text{5HT}},\; g_{\text{ACh}})$$

| 조절계 | 핵 | F절 역할 | 뇌 기능 | 판정 | 최신 근거 |
|---|---|---|---|---|---|
| 도파민 (DA) | VTA, SNc | $g_{\text{DA}}$: STDP 학습 게이트 (F.14) | reward prediction error, motivation | `supported` | Schultz 1997, Yagishita 2014. phasic/tonic 구분 확립 |
| 노르에피네프린 (NE) | LC | $g_{\text{NE}}$: $n_{\text{iter}}$ 조절 (F.3) | arousal, attention, exploration-exploitation | `supported` | Aston-Jones & Cohen 2005 adaptive gain theory. 2024 LC-NE review: tonic=탐색, phasic=착취 재확인. pupil diameter proxy |
| 세로토닌 (5HT) | raphe | $g_{\text{5HT}}$: $T$ (온도) 조절 (F.6) | patience, temporal discounting, model-based prediction | `supported` (인내/보상 대기) | Miyazaki 2018 Nat Comm: DRN 5HT 광유전 활성화 $\to$ 인내 증가. 2025 Complementary roles: 5HT = model-based prediction |
| 아세틸콜린 (ACh) | BF, PPT | $g_{\text{ACh}}$: 기억 인코딩 감도 (F.8) | attention, memory encoding, cortical gain | `supported` | 2025 Cell Rep: 해마 ACh 방출 $\propto$ 이동 속도, 새 환경에서 증가. eLife 2024: ACh가 PFC 예측 오차 부호화 조절 |

루프 내 적용:

$$n_{\text{iter}} = n_0 + \Delta n \cdot \sigma(g_{\text{NE}}) \quad\text{(NE 높으면 깊은 처리)}$$

$$\theta_{\text{encode}} = \theta_0 / (1 + g_{\text{ACh}}) \quad\text{(ACh 높으면 기억 인코딩 문턱 하강)}$$

$$T_{\text{effective}} = T_{\text{wake}} \cdot (1 + \beta \cdot g_{\text{5HT}}) \quad\text{(5HT 높으면 탐색 감소)}$$

현재 구현 상태: 단일 스칼라 $g[t]$만 존재. 4차원 확장은 설계 목표.

---

### F.20 작업 기억과 주의

> 뇌의 작업 기억 용량 제한과 주의 선택은 F절에 빠져 있었다.

#### F.20.1 작업 기억 용량

이력 버퍼 $h_t$의 유한 창 $T_h$는 작업 기억의 모델이다.

$$|h_t| \leq T_h, \qquad T_h \approx 7 \pm 2 \quad\text{(Miller 1956)}$$

용량 초과 시 가장 오래된 항목 제거:

$$h_{t+1} = \text{append}(h_t, (a_t, o_t))[-T_h:]$$

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| 작업 기억 $\sim 3$--$5$ 항목 | Miller 1956, Cowan 2001/2010 ($\sim 4$) | `supported` | Cowan 2010 PMC: 중앙 저장 한계 3--5. 2025 JoCognition: 과제 의존적이나 $\sim 4$ 재확인 |
| PFC sustained activity | PFC 지속 발화로 작업 기억 유지 | `supported` | 2019 PNAS: distributed PFC activation이 WM 용량 향상과 연결 |
| 용량 초과 = 간섭/망각 | proactive interference | `supported` | |
| theta-gamma 비에 의한 용량 결정 | theta 주기 내 gamma burst 수 = 유지 항목 수 | `supported` | Lisman & Jensen 2013. 2025 eLife: PFC-BG adaptive chunking이 $\sim 4$ 항목 제한 설명 |

#### F.20.2 주의 (Attention)

주의는 $R$ 내부에서 입력 가중치를 조절하는 메커니즘이다.

$$u_i^0 = \alpha_i \cdot \text{encode}(S_t), \qquad \alpha_i = \text{softmax}(\text{salience}(i, S_t))$$

여기서 salience는 $V_{\text{sal}}$ (14_BrainRuntimeSpec.md 3.4절)에서 산출한다.

주의의 두 경로:

| 경로 | 메커니즘 | 뇌 대응 | 판정 |
|---|---|---|---|
| bottom-up | $\alpha_i \propto \|u_i\|$ (입력 크기) | exogenous attention, pop-out | `supported` |
| top-down | $\alpha_i \propto \text{sim}(u_i, g_t)$ (목표 정렬) | endogenous attention, PFC-driven | `supported` |

#### F.20.3 소뇌의 역할

F.9에서 $S(X_t)$를 "소뇌/기저핵 억제"라 했으나 구체화가 없었다.

소뇌 모델: 행동 $a_t$ 실행 후 감각 예측 오차의 빠른 보정:

$$\Delta a_{t}^{\text{cerebellar}} = -\eta_{\text{cb}} \cdot (o_t - \hat{o}_t^{\text{cb}})$$

$$\hat{o}_{t+1}^{\text{cb}} = \hat{o}_t^{\text{cb}} + \alpha_{\text{cb}} \cdot (o_t - \hat{o}_t^{\text{cb}})$$

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| 소뇌 = 내부 모델 (forward model) | 소뇌 병변 시 운동 부정확 + 적응 실패 | `supported` | 2025 JNeurosci: sensory prediction error가 소뇌 학습 구동 재확인. 2025 SciAdv: CPC 계층 모델 fast/slow 적응 설명 |
| 소뇌 = 빠른 오차 보정 | 시간 정밀도 $\sim 10$ms | `supported` | 2026 PMC: corticocerebellar connectivity가 visuomotor adaptation 핵심 |
| 소뇌 = 인지 기능 기여 | 최근 연구에서 언어/작업 기억 기여 확인 | `bridge` | 2025 ScienceDirect: cerebro-cerebellar system 인지/정서 통합 역할 리뷰 |

---

### F.21 뇌파 대역과 루프 주기

> `14_BrainRuntimeSpec.md` 5절. 루프의 시간 구조가 빠져 있었다.

에이전트 루프의 각 단계는 다른 시간 척도에서 작동한다.

| 루프 단계 | 시간 척도 | 뇌파 대역 | 뇌 대응 |
|---|---|---|---|
| $R$ 내부 반복 (1 iter) | $\sim 10$--$25$ ms | gamma (30--100 Hz) | 국소 결합/계산 |
| $R$ 전체 수렴 | $\sim 100$--$500$ ms | theta/alpha (4--13 Hz) | 전역 통합/주의 |
| 행동-관찰 1 사이클 | $\sim 0.5$--$2$ s | delta/theta (0.5--4 Hz) | 의사결정 리듬 |
| 수면 1 주기 | $\sim 90$ min | slow oscillation ($< 1$ Hz) | NREM-REM 교대 |

theta-gamma 결합:

$$\text{gamma burst 위치} = f(\theta_{\text{phase}})$$

이것은 $R$ 내부의 빠른 계산(gamma)이 전역 동기화(theta)에 의해 순서화되는 구조와 대응한다. F.3의 반복이 gamma이고, 반복의 시작/종료가 theta 주기에 잠금(phase-locking)된다.

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| theta-gamma coupling | Lisman & Jensen 2013, 해마 sequential memory | `supported` | 2024 bioRxiv: 인간 해마 ECoG에서 theta-gamma PAC가 WM 인출 성공과 강하게 상관 (개별 theta/gamma power는 무관). 2025 ScienceDirect: 건강인/정신병 양쪽에서 PAC가 WM 용량 예측 |
| gamma = 국소 계산 | Fries 2015, communication through coherence | `supported` | |
| alpha = 억제/게이팅 | Klimesch 2012, alpha gating by inhibition | `supported` | 2025 eNeuro: alpha-tACS가 DMN 연결성을 직접 조절. alpha-DMN mechanistic coupling 확인 |
| 수면 주기 $\sim 90$분 | polysomnography, 수면 구조 | `supported` | |

---

### F.22 확장된 뇌 대응 간극 (정직한 갱신)

F.11.5를 갱신하여, F.14--F.21에서도 남는 간극을 정리한다.

| 간극 | 현재 상태 | 심각도 | 해결 방향 |
|---|---|---|---|
| 환경 내부 모델 분산화 | $C$가 단일 예측기 | 중 | 모듈별 $\hat{o}_t^{(m)}$ |
| 행동 계층성 | $\pi$ 단층 | 중 | macro-action + primitive |
| 감정/정동 | $c_t$에 valence 없음 | 중 | $c_{\text{val}} = V_{\text{sal}}$ 출력 추가 |
| 사회적 모델링 | theory of mind 없음 | 저 (장기) | 다중 에이전트 시뮬레이션 |
| 신체 루프 실사용 | $Q_t$ 존재하나 $C$, $\pi$에서 미사용 | 중 | $Q_t \to g_{\text{5HT}}, g_{\text{NE}}$ 매핑 |
| 4종 조절계 구현 | 단일 $g[t]$만 존재 | 높 | F.19의 4차원 벡터 구현 |
| 장소/격자 세포 | 공간 표상 없음 | 저 (도메인 특화) | spatial module 추가 |
| 거울 뉴런 | 타인 행동 모방/이해 없음 | 저 (장기) | 관찰 학습 모듈 |
| STDP 코드 미구현 | 수식만 존재, 코드 없음 | 높 | reality_stone/python/reality_stone/clarus/core 또는 Python 구현 |

---

### F.23 간극 대책: bridge/hypothesis -> supported 승격 경로

> F.22의 간극 + H.3의 판정에서 `bridge` 또는 `hypothesis`인 항목에 대해, `supported`로 승격하기 위한 구체적 실험/시뮬레이션/논증 경로를 정리한다.

#### F.23.1 일관성 오차 $c_{\text{cons}}$ (`bridge` -> `supported` 경로)

현재 상태: 해마-PFC 상호작용에서 기억 기반 의사결정 보정 방향은 있으나, $c_{\text{cons}} = \|z_t - \mathcal{R}(H_t, c_t)\|$의 직접 분리가 미흡.

승격 조건:
1. HPC-mPFC theta 동기화가 기억-현재 사고 불일치 시 증가함을 보이는 실험
2. 기억 회상 오차와 $c_{\text{cons}}$ proxy를 분리 측정

최신 근거:
- 2025 Nature Comm: 해마-전전두엽 오케스트레이션이 고차 학습 지원. HPC dimensionality reduction이 mPFC로 전달.
- 2025 ScienceDirect (Cell Rep): 5XFAD 마우스에서 HPC-PFC theta 동기화 및 SWR 붕괴 시 행동 유연성 결손. 이는 기억-현재 불일치 보정 경로의 인과적 근거.
- 2026 NYAS: 기억과 불안이 choice consistency를 조절함을 계산 모델링 + 신경영상으로 확인.

평가: theta-SWR 동기화 결손이 유연성 결손으로 직결되는 인과 근거가 나왔으므로, 조건부로 `supported`에 근접. 남은 과제는 $c_{\text{cons}}$의 조작적 정의와 EEG/fMRI proxy의 정밀 매핑.

#### F.23.2 학습 게이트 $g[t] = d\bar{c}/dt$ (`hypothesis` -> `bridge` 경로)

현재 상태: 3-factor learning rule 자체는 `supported`. $g[t]$의 정확한 형태가 $d\bar{c}/dt$라는 주장이 미검증.

승격 조건:
1. CE 시뮬레이션에서 $g[t] = d\bar{c}/dt$ vs 대안 형태(예: $g[t] = \bar{c}$, $g[t] = \text{RPE}$)의 학습 성능 비교
2. phasic DA burst 시간 프로파일과 $d\bar{c}/dt$ 파형의 상관 측정

경로:
- 05_실험근거.md 8.5절의 $\delta[t] = a \cdot \text{RPE} + b \cdot \text{surprise} + c \cdot \text{novelty}$ 형태가 더 일반적.
- CE 시뮬레이션에서 ablation: $d\bar{c}/dt$ 제거 시 학습 붕괴 여부로 필요성 판정 가능.
- 생물학적으로는 phasic DA의 시간 미분 형태가 temporal derivative model과 부합하나, 정확한 $d\bar{c}/dt$ 매핑은 아직 가설 수준.

평가: `bridge`로 승격 가능. `supported`까지는 시뮬레이션 ablation + voltammetry 시간 프로파일 비교 필요.

#### F.23.3 이중 과정 (F.3 모드별 $n_{\text{iter}}$) (`bridge` -> `supported` 경로)

현재 상태: 행동 근거 방대, 신경 기질 논쟁 중.

승격 조건:
1. $n_{\text{iter}}$의 변이가 RT, EEG alpha desynchronization duration과 양적 대응
2. DMN(System 1) vs frontoparietal network(System 2) 전환이 $\|\phi\| \gtrless m_\phi$와 대응

최신 근거:
- 2018/2025 Frontiers: DMN이 System 1(빠른 연상적 사고)의 신경 기반으로 제안. System 2는 전두-두정 제어 네트워크.
- 2025 Neuroscience of Consciousness: flow와 intuition의 시스템 신경과학 비교. 인지 부하 증가 시 System 1 -> System 2 전환이 반응 시간에 반영.
- 2026 Nature Comm: 증거 축적 회로 모델에서 ACC-DMS-HPC 각각 다른 회로 메커니즘 사용.

평가: DMN-frontoparietal 전환 + RT 양적 대응이 확보되면 `supported` 가능. CE 시뮬레이션에서 $n_{\text{iter}}$-RT 상관을 보이는 것이 핵심 경로.

#### F.23.4 수면 압력 = 비평 누적 (F.6) (`bridge` -> `supported` 경로)

현재 상태: SWA/wakefulness 관계는 `supported`, 비평 해석은 추가 가정.

승격 조건:
1. 비평 점수 $\bar{c}$의 누적 $\sum \bar{c}^2$와 SWA delta power의 양적 상관
2. 시냅스 강도 축적이 "에러/비평" 누적과 등가임을 보이는 실험

최신 근거:
- 2025 Science: PFC 흥분성 시냅스의 화학유전적 강화가 NREM 수면량 + delta power 모두 증가시킴. 시냅스 강도가 수면 압력을 직접 결정함을 인과적으로 확인.
- 2025 Nature: 초파리에서 수면 압력이 전압 의존적 지질 과산화 기억에 축적. 수면 중 스파이크 방출이 이를 소거. 분자 수준의 "축적-소거" 사이클 확인.
- 2026 bioRxiv: 일주기 조절 임계와 수면 항상성의 상호작용 모델. Process S 축적/해소의 정량 프레임워크.
- 2026 NeuroImage: 수면-각성 BOLD 변동의 피질 위계적 패턴이 SWA와 상관. sleep pressure alleviation의 공간 구조 확인.

평가: 시냅스 강도 축적 = 수면 압력이 인과적으로 확인됨. CE의 $\sum \bar{c}^2$를 시냅스 강도 proxy로 재해석하면 `supported`에 근접. 남은 과제: 비평 점수와 시냅스 강도의 정량적 매핑.

#### F.23.5 $\rho = 0.155$ (`bridge` -> 정밀화 경로)

현재 상태: 수면 수축 방향 `supported`, 정확 값은 피팅 결과.

승격 조건:
1. 실제 수면 회복 곡선 데이터에서 지수 감쇠 상수 피팅 -> $\rho_{\text{night}}$와 CE의 $\rho^2$ 비교
2. CE 시뮬레이션에서 $\rho$ sweep -> 최적 $\rho$ 범위 확인

경로: 수면 부채 회복 연구(Van Dongen 2003, Kitamura 2016)의 시간 상수를 재분석하여 $\rho$의 관측 범위 $[0.1, 0.3]$를 확인. CE 값 0.155가 이 범위 내에 있으므로 `[NEAR]` 유지.

#### F.23.6 잔류장 $\phi$ (`bridge` -> `supported` 경로)

현재 상태: DMN/spontaneous activity 방향 있으나 $\phi$와의 정확한 매핑은 미확인.

승격 조건:
1. CE 시뮬레이션에서 $\phi$ 제거 시 모드 전환 실패 + 탐색 능력 붕괴를 보임
2. DMN ALFF가 "비선택 경로의 분산"과 양적 상관

최신 근거:
- 2024 PMC: DMN ALFF가 과제 수행 안정성 예측.
- 2025 eNeuro: alpha-tACS가 DMN 연결성 직접 조절. alpha-DMN coupling의 기계론적 연결.
- 2025 NeuroImage: DMN transition rate가 resilience와 상관. 잔류장의 "모드 전환 임계" 역할과 방향 일치.

평가: $\phi$ 제거 ablation 시뮬레이션이 가장 빠른 경로. DMN과의 정량 매핑은 resting-state fMRI 데이터 필요.

#### F.23.7 의식 깊이 (F.17.2) (`hypothesis` -> `bridge` 경로)

현재 상태: IIT/GNW와의 관계 미확정.

승격 조건:
1. CE의 의식 깊이 $\exp(-c_d \cdot d_\tau)$와 PCI(Perturbational Complexity Index)의 상관
2. CE 시뮬레이션에서 마취/수면 조건에서 의식 깊이 자동 감소

최신 근거:
- 2025 Nature: IIT vs GNW adversarial testing (Cogitate Consortium). 결과는 특정 이론의 명확한 승리가 아닌 상호 보완적 해석. IIT의 posterior "hot zone" 예측은 부분 지지.
- 100명 이상의 연구자가 IIT를 "pseudo-science"로 비판하는 공개서한(2023). $\Phi$의 계산 난해성.
- PCI는 IIT에서 영감을 받았으나 이론의 직접 검증은 아님.

평가: CE의 의식 깊이 정의는 IIT의 $\Phi$와 직접 대응되지 않으므로, 독립적 검증 경로 필요. CE 시뮬레이션에서 모드(WAKE/NREM/REM)별 $d_\tau$ 프로파일을 polysomnography와 비교하면 `bridge`로 승격 가능. `supported`까지는 먼 길.

#### F.23.8 메타인지 수렴 (F.17.3) (`bridge` -> `supported` 경로)

현재 상태: PFC 재귀 자기평가 방향은 있으나 $\rho$ 매핑은 추가 가정.

승격 조건:
1. metacognitive accuracy가 반복적 자기평가에서 수렴함을 보이는 행동 실험
2. rlPFC/BA10의 활동이 "비평의 비평"에서 감쇠하는 fMRI 근거

최신 근거:
- rlPFC/BA10이 retrospective metacognitive accuracy의 핵심 영역으로 확립.
- dlPFC와 vmPFC의 기능적 분리: lateral = 사후 신뢰도, medial = 사전 판단.
- 그러나 "수축 사상으로서의 메타인지 수렴"은 현재 문헌에서 명시적으로 검증된 적 없음.

평가: CE 시뮬레이션에서 $C(C(C(x)))$의 감쇠 프로파일을 보이고, 행동 실험에서 반복적 confidence calibration이 수렴함을 보이면 `supported` 가능.

#### F.23.9 곡률-환각 억제 (F.18) (`bridge` -> `supported` 경로)

현재 상태: 억제 feedback 구조는 `supported`, LBO 매핑은 `bridge`.

승격 조건:
1. LBO 고유모드와 fMRI/EEG spatial mode의 양적 대응
2. 곡률 과다 조건에서 CE 시뮬레이션의 출력 불안정과 환각 유사 패턴 비교

최신 근거:
- 2023 Nature: 피질 표면의 기하학적 고유모드(LBO eigenmode)가 fMRI 활동 패턴의 상당 부분 설명. 기하학적 제약이 뇌 기능에 근본적 영향.
- 2025 PMC: 포유류 피질 연결체의 기하학적 제약. LBO 기반 공명 모드가 connectome 아키텍처 예측.
- 2026 bioRxiv: 피질 진동 모드의 수렴적 시간축. LBO가 시공간 패턴의 공간 전파자 역할.

평가: LBO eigenmode가 fMRI 공간 패턴을 설명한다는 근거가 강화됨. CE의 곡률 모니터링을 LBO eigenmode decomposition으로 구현하고, 고곡률 영역에서의 불안정을 시뮬레이션하면 `supported` 가능.

#### F.23.10 4종 신경조절 통합 벡터 (F.19) (`bridge` -> `supported` 경로)

현재 상태: 개별 조절계는 `supported`. 4차원 벡터 통합 + CE 변수 매핑은 `bridge`.

승격 조건:
1. CE 시뮬레이션에서 4차원 $g_t$ 구현 후 단일 $g[t]$ 대비 성능 개선 확인
2. 각 조절계의 독립적 조작(DA agonist, NE clonidine, 5HT SSRI, ACh donepezil)에 의한 개별 효과가 CE 예측과 일치

경로:
- 코드 구현이 선행 조건. F.19의 수식을 reality_stone/python/reality_stone/clarus/core에 구현.
- 구현 후 약리학적 조작 시뮬레이션으로 각 축의 독립 효과 확인.
- 4축 독립성 + 개별 효과 일치가 확인되면 `supported`.

#### F.23.11 모듈 생애주기 (F.16.2) (`bridge` -> 유지)

현재 상태: 4상태 분류(ACTIVE/IDLE/DORMANT/SLEEPING) 자체는 설계 선택.

평가: 뇌의 뉴런 집단도 유사한 상태를 가지지만(task-engaged/baseline/deactivated), 4상태 명명은 CE 고유 추상화. `bridge` 유지가 적절. 구현 후 성능 기여가 확인되면 "설계적 지지"로 보강 가능.

---

### F.23 요약: 승격 우선순위

| 항목 | 현재 | 목표 | 난이도 | 핵심 경로 |
|---|---|---|---|---|
| $c_{\text{cons}}$ (F.4) | `bridge` | `supported` | 중 | HPC-PFC theta 불일치 측정 |
| $g[t] = d\bar{c}/dt$ (F.14) | `hypothesis` | `bridge` | 저 | CE ablation 시뮬레이션 |
| 이중 과정 (F.3) | `bridge` | `supported` | 중 | $n_{\text{iter}}$-RT 상관 시뮬레이션 |
| 수면 압력 = 비평 (F.6) | `bridge` | `supported` | 저 | $\sum \bar{c}^2$ vs SWA 매핑 |
| $\rho = 0.155$ (F.10) | `bridge` | `bridge` (정밀화) | 중 | 수면 부채 데이터 재분석 |
| $\phi$ 잔류장 (F.15) | `bridge` | `supported` | 중 | ablation + DMN ALFF 비교 |
| 의식 깊이 (F.17.2) | `hypothesis` | `bridge` | 고 | PCI 상관 시뮬레이션 |
| 메타인지 수렴 (F.17.3) | `bridge` | `supported` | 중 | 반복 confidence 수렴 행동실험 |
| 곡률-환각 (F.18) | `bridge` | `supported` | 중 | LBO eigenmode 구현 + 시뮬레이션 |
| 4종 조절계 통합 (F.19) | `bridge` | `supported` | 고 | 코드 구현 + 약리 시뮬레이션 |
| 모듈 생애주기 (F.16.2) | `bridge` | `bridge` (유지) | -- | 설계적 선택, 검증 불필요 |

**즉시 실행 가능 (CE 시뮬레이션만으로):**
1. $g[t]$ ablation (F.23.2)
2. $\phi$ ablation (F.23.6)
3. $n_{\text{iter}}$-RT sweep (F.23.3)
4. $\sum \bar{c}^2$ vs SWA proxy (F.23.4)

**외부 데이터 필요:**
1. HPC-PFC theta 불일치 (F.23.1): intracranial EEG
2. $\rho$ 피팅 (F.23.5): 수면 부채 회복 곡선 데이터셋
3. DMN ALFF (F.23.6): resting-state fMRI
4. PCI (F.23.7): TMS-EEG 데이터셋
5. Metacognitive convergence (F.23.8): 행동 실험 데이터

---

### F.24 실험값 기반 루프 방정식 보강

> `15_Equations.md` J절의 실험 상수를 F절 방정식에 적용한다.

#### F.24.1 이완 반복 $n_{\text{iter}}$의 실험 고정

F.3에서 모드별 $n_{\text{iter}}$를 뇌파 시간 척도로 고정:

$$n_{\text{iter}}^{\text{fast}} = \frac{\tau_{\text{alpha}}}{\Delta t_{\text{gamma}}} = \frac{100 \text{ ms}}{10 \text{ ms}} = 10$$

$$n_{\text{iter}}^{\text{deep}} = \frac{\tau_{\text{theta}}}{\Delta t_{\text{gamma}}} = \frac{200 \text{ ms}}{10 \text{ ms}} = 20 \text{--} 50$$

| 모드 | $n_{\text{iter}}$ | 뇌파 유래 | RT 예측 |
|---|---|---|---|
| WAKE (시스템 1) | $10$--$20$ | alpha 1--2주기 | $100$--$200$ ms |
| WAKE (시스템 2) | $20$--$50$ | theta 2--5주기 | $200$--$500$ ms |
| NREM | $50$--$100$ | slow oscillation | offline |
| REM | $20$--$30$ | theta 기반 탐색 | offline |

RT 예측: $\text{RT} = n_{\text{iter}} \times \Delta t_{\text{gamma}} + \tau_{\text{motor}}$, $\tau_{\text{motor}} \approx 50$ ms.

시스템 1 ($n = 15$): RT $= 15 \times 10 + 50 = 200$ ms → 관측 RT $\sim 200$--$300$ ms와 일치.
시스템 2 ($n = 40$): RT $= 40 \times 10 + 50 = 450$ ms → 관측 RT $\sim 400$--$600$ ms와 일치.

#### F.24.2 비평 점수의 ERN 진폭 매핑

F.4의 $\bar{c}_t$를 ERN(error-related negativity) 진폭과 정량 연결:

$$\text{ERN}_{\text{amp}} = -k_{\text{ERN}} \cdot \bar{c}_t, \qquad k_{\text{ERN}} \approx 5 \text{--} 10 \;\mu\text{V per unit}$$

관측: ERN 진폭은 $-2$ ~ $-15\;\mu$V (Gehring 1993, Falkenstein 1991).
$\bar{c}_t \in [0.3, 2.0]$ (정상 오차 범위)일 때 $k_{\text{ERN}} = 7$이면:
$\text{ERN} = -7 \times 0.3 = -2.1\;\mu$V (약한 오차) ~ $-7 \times 2.0 = -14\;\mu$V (강한 오차).

#### F.24.3 수면 압력 방정식의 정량화

F.6의 $P_{\text{sleep}}$을 Borbely의 Process S와 연결:

$$P_{\text{sleep}}(t) = P_0 + \sum_{\tau=0}^{t} \bar{c}_\tau^2 \cdot \Delta t - \int_0^t \lambda_S(M_s)\,ds$$

관측에서 Process S의 축적 시간 상수:

$$\tau_{\text{wake}} \approx 18.2 \text{ h} \quad\text{(Achermann 2003)}$$

$$\tau_{\text{sleep}} \approx 4.2 \text{ h} \quad\text{(NREM decay)}$$

CE 매핑:

$$
\sum_{\tau=0}^{T_{\text{wake}}} \bar{c}_\tau^2
= \frac{T_{\text{wake}}}{\tau_{\text{wake}}} \cdot P_{\text{th}}
\quad\Longrightarrow\quad
\bar{c}_{\text{avg}}^2
= \frac{P_{\text{th}}}{\tau_{\text{wake}}}
\approx \frac{1}{65520}
\quad\text{(if } P_{\text{th}} = 1\text{)}
$$

이것은 비평 점수의 적분이 16시간 각성 후 수면 임계에 도달하는 제약이다.

#### F.24.4 4종 조절계의 루프 내 정량 적용

F.19의 수식에 J.7의 시간 상수를 삽입:

$$g_{\text{DA}}^{t+1} = g_{\text{DA}}^t + \frac{1}{500}(g_0^{\text{DA}} - g_{\text{DA}}^t) + \alpha_{\text{DA}} \cdot c_{\text{pred}}^t$$

$$g_{\text{NE}}^{t+1} = g_{\text{NE}}^t + \frac{1}{300}(g_0^{\text{NE}} - g_{\text{NE}}^t) + \alpha_{\text{NE}} \cdot c_{\text{nov}}^t$$

$$g_{\text{5HT}}^{t+1} = g_{\text{5HT}}^t + \frac{1}{3000}(g_0^{\text{5HT}} - g_{\text{5HT}}^t) + \alpha_{\text{5HT}} \cdot (-\text{discount}^t)$$

$$g_{\text{ACh}}^{t+1} = g_{\text{ACh}}^t + \frac{1}{200}(g_0^{\text{ACh}} - g_{\text{ACh}}^t) + \alpha_{\text{ACh}} \cdot \text{salience}^t$$

여기서 분모는 $\tau_X / \Delta t$이고, $\Delta t = 1$ ms.

5HT가 가장 느리고 ($\tau = 3$ s), ACh가 가장 빠르다 ($\tau = 200$ ms). 이것은 인내(5HT)는 천천히 쌓이고, 주의(ACh)는 빠르게 전환된다는 실험 관측과 일치.

#### F.24.5 STDP 학습률의 실험 고정

F.14의 가중치 갱신에서 학습률을 실험 제약으로 고정:

$$\Delta W_{ij} = \text{lr} \cdot g_{\text{DA}}^t \cdot e_{ij}^{n_{\text{iter}}}$$

학습률 제약:
- 1회 보상 경험으로 $W_{ij}$가 $1$--$5$% 변화 (Yagishita 2014의 spine volume 변화)
- $g_{\text{DA}}^{\text{peak}} \approx 5 \times g_0$ (phasic burst는 tonic의 $\sim$5배)
- $e_{ij}^{\text{peak}} \approx A_+ \approx 0.01$ (단일 스파이크 쌍)

$$\text{lr} \cdot 5g_0 \cdot 0.01 \approx 0.01 \quad\Longrightarrow\quad \text{lr} \approx \frac{0.01}{0.05 \cdot g_0} = \frac{0.2}{g_0}$$

$g_0 = 1$ (정규화)이면 $\text{lr} \approx 0.2$.

#### F.24.6 소뇌 forward model의 적응 시간 상수

F.20.3의 소뇌 모델에 실험 시간 상수 삽입:

$$\hat{o}_{t+1}^{\text{cb}} = \hat{o}_t^{\text{cb}} + \alpha_{\text{cb}} (o_t - \hat{o}_t^{\text{cb}})$$

프리즘 적응 실험에서 적응 완료까지 $\sim 50$--$100$ trial (Martin 1996).
각 trial $\sim 1$ s, 총 $\sim 50$--$100$ s.

$$\alpha_{\text{cb}} = 1 - \exp(-1/N_{\text{adapt}}) \approx 1/75 \approx 0.013$$

$N_{\text{adapt}} = 75$ trial (중앙값). 75 trial 후 $63.2$% 적응, 150 trial 후 $86.5$% 적응.

---

## G. 형식 증명 요약 (F절)

| 정리 | 주장 | 조건 | 상태 |
|---|---|---|---|
| F-energy | 이완 $R$이 $E_t(z)$를 비증가 | E-decrease (B.4) | **closed** |
| F-relax | 이완 수렴 | A-bound, Zero-attract | **closed** (조건부) |
| F-memory | 조건부 인코딩이면 기억 유계 | $\theta_{\text{encode}} > 0$ | **closed** |
| F-contract | 전체 루프 수축 | $\rho + \lambda_R L_R + \lambda_C L_C < 1$ | **open** ($L_R, L_C$ 추정 필요) |
| F-sparse | 활성 유계 + 에너지 예산 | $|A_t| \leq \lceil x_a^* N \rceil$, $B_t$ 유한 | **closed** (Sparse-energy로부터) |
| F-phi-bound | 잔류장 유계 | $\xi < 1$, $\text{Var}$ 유한 (A-bound) | **closed** |
| F-curvature | 곡률 모니터링이 환각 억제 | LBO 확산 $h_d < 1/\text{eig}_{\max}$ | **closed** (11.2 수렴 조건) |
| F-meta | 메타인지 수렴 | $\rho < 1$ (수면 존재 시) | **closed** |
| F-STDP-local | STDP가 국소 정보만 사용 | $e_{ij}$는 $i,j$ 이웃 스파이크만 의존 | **closed** (정의에 의해) |
| F-WM-finite | 작업 기억 유한 | $|h_t| \leq T_h$ (유한 창) | **closed** (정의에 의해) |

---

## H. 검증 게이트 (F절)

### H.2 Layer F (자기참조 재귀) 게이트

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | F-energy, F-relax, F-memory (closed). F-contract (open: $L_R, L_C$ 추정 필요) | partial |
| $G_{\text{obs}}$ | $R \leftrightarrow$ recurrent processing, $C \leftrightarrow$ ERN/ACC, $\pi \leftrightarrow$ BG, $\mathcal{M} \leftrightarrow$ hippocampal encoding | partial |
| $G_{\text{causal}}$ | 수면박탈 $\to$ 루프 불안정, DA 조작 $\to$ 학습 게이트 변화, ACC 병변 $\to$ 비평 결손 | partial |
| $G_{\text{pred}}$ | 에이전트 루프 유무에 따른 과제 수행 차이 시뮬레이션 | pending |

### H.3 F절 뇌 대응 판정 요약

| 구성요소 | 판정 | 비고 |
|---|---|---|
| 이완 $R$ | `supported` | 피질 재귀 처리 확립 |
| 행동 선택 $\pi$ | `supported` | 기저핵 경로 확립 |
| 비평 $C$ | `supported` | ACC/ERN 확립 |
| 예측 오차 $c_{\text{pred}}$ | `supported` | 도파민 RPE 확립 |
| 놀라움 $c_{\text{nov}}$ | `supported` | P300/CA1 novelty 확립 |
| 일관성 오차 $c_{\text{cons}}$ | `bridge` | 해마-PFC 방향은 있으나 직접 분리 미흡 |
| $g[t] = d\bar{c}/dt$ | `hypothesis` | 3-factor rule은 `supported`, 정확한 형태는 미검증 |
| 이중 과정 ($n_{\text{iter}}$) | `bridge` | 행동 근거 방대, 신경 기질 논쟁 중 |
| 수면 압력 = 비평 누적 | `bridge` | SWA/wakefulness 관계는 `supported`, 비평 해석은 추가 가정 |
| $\rho = 0.155$ | `bridge` | 수면 수축 방향 `supported`, 정확 값은 피팅 결과 |
| STDP + 도파민 (F.14) | `supported` | 3-factor learning rule 강하게 지지됨 |
| 구조적 투영 Proj (F.14.3) | `supported` | 시냅스 가지치기, 스케일링 확립 |
| 잔류장 $\phi$ (F.15) | `bridge` | DMN/spontaneous activity 방향 있으나 $\phi$ 매핑은 추가 가정 |
| TopK 희소 활성 (F.16) | `supported` | sparse firing 1--5% 확립 |
| 모듈 생애주기 (F.16.2) | `bridge` | 4상태 분류 자체는 설계 선택 |
| 자기일관성 C3 (F.17.1) | `hypothesis` | 수학적으로 닫힘. 뇌 대응은 현상론 |
| 의식 깊이 (F.17.2) | `hypothesis` | IIT/GNW와의 관계 미확정 |
| 메타인지 수렴 (F.17.3) | `bridge` | PFC 재귀 자기평가 방향은 있으나 $\rho$ 매핑은 추가 가정 |
| 곡률 환각 억제 (F.18) | `bridge` | 억제 feedback 구조는 `supported`, LBO 매핑은 `bridge` |
| 4종 신경조절 (F.19) | `supported` (개별 존재 + 개별 기능), `bridge` (4차원 벡터 통합 + CE 변수 매핑) | DA RPE 확립, NE 탐색-착취 확립, 5HT 인내/model-based 광유전 확인, ACh cortical gain/기억 인코딩 확립 |
| 작업 기억 용량 (F.20.1) | `supported` | Miller 1956, Cowan 2001 |
| 주의 (F.20.2) | `supported` | bottom-up/top-down 확립 |
| 소뇌 내부 모델 (F.20.3) | `supported` | forward model 확립 |
| theta-gamma 결합 (F.21) | `supported` | Lisman & Jensen 2013 |

---

## I. 관측 가능량 매핑 (F절)

### I.2 Layer F (자기참조 재귀) 변수

| formal 변수 | 뇌 관측량 | 데이터 소스 |
|---|---|---|
| $n_{\text{iter}}$ | reaction time, alpha desynchronization | 행동, EEG |
| $\bar{c}_t$ | ERN amplitude, ACC theta, pupil | EEG, fMRI, pupillometry |
| $c_{\text{pred}}$ | RPE-locked DA, FRN | voltammetry, EEG |
| $c_{\text{nov}}$ | P300, hippocampal novelty, LC burst | EEG, fMRI, pupil |
| $P_{\text{sleep}}$ | SWA, theta/alpha ratio, KSS | polysomnography |
| $\phi_t$ | spontaneous activity, DMN fluctuation | resting-state fMRI, MEG |
| $e_{ij}$ | eligibility trace, synaptic tag | in vitro slice, optogenetics |
| $g_{\text{DA}}$ | VTA/SNc DA | voltammetry, PET |
| $g_{\text{NE}}$ | LC firing, pupil diameter | pupillometry, unit recording |
| $g_{\text{5HT}}$ | raphe firing, 5-HIAA | microdialysis, PET |
| $g_{\text{ACh}}$ | BF firing, cortical ACh | microdialysis, optogenetics |
| $\kappa_{\text{avg}}$ | high-frequency anomaly | EEG, MEG |
| $|A_t|/N$ | active neuron fraction | calcium imaging, MEA |
| $|h_t|$ | WM load (PFC BOLD, CDA) | fMRI, EEG |
| $\alpha_i$ | spatial attention, alpha lateralization | EEG |
| $\Delta a^{\text{cb}}$ | cerebellar adaptation | prism adaptation, saccade |
```
---
## File: `docs/7_AGI/18_CodeMap.md`

```markdown
# 이론-코드 정합 맵

> 이 문서는 `15_Equations.md`의 Layer A--E 수식과 `17_AgentLoop.md`의 Layer F가 실제 코드의 어디에서 구현되는지를 1:1로 대응시킨다.
> 코드를 읽을 때 "이 변수가 어떤 수식인지" 또는 수식을 읽을 때 "이 항이 어디에 구현되어 있는지"를 즉시 찾을 수 있도록 한다.

---

## 1. 전체 아키텍처 대응

```
15_Equations.md            clarus/
+-------------------+      +-----------------------------+
| Layer A: 셀 동역학  | <--> | runtime.py::_step_torch     |
|                   |      | core/src/engine/kernel.rs   |
+-------------------+      +-----------------------------+
| Layer B: 필드 결합  | <--> | runtime.py::_matvec (CSR)   |
|                   |      | core/src/engine/field.rs    |
+-------------------+      +-----------------------------+
| Layer C: 전역 모드  | <--> | runtime.py::_auto_mode      |
|                   |      | runtime.py::_update_sleep   |
+-------------------+      +-----------------------------+
| Layer D: 해마/기억  | <--> | runtime.py::HippocampusMemory|
+-------------------+      +-----------------------------+
| Layer E: 전역 요약  | <--> | runtime.py::RuntimeStep     |
|                   |      | runtime.py::BrainRuntimeSnapshot |
+-------------------+      +-----------------------------+
| Layer F: 에이전트   | <--> | engine.py::CEEngine (이완)   |
|          루프      |      | sleep.py::run_sleep_cycle   |
+-------------------+      +-----------------------------+
```

---

## 2. Layer A: 셀 동역학

### 2.1 상태 변수

| 수식 기호 | canonical 이름 | Python 변수 | Rust 변수 | 초기값 |
|---|---|---|---|---|
| $a_i$ | activation | `self.activation` | `activation` | 0 |
| $r_i$ | refractory | `self.refractory` | `refractory` | 0 |
| $m_i$ | memory_trace | `self.memory_trace` | `memory_trace` | 0 |
| $w_i$ | adaptation | `self.adaptation` | `adaptation` | 0 |
| $b_i$ | bitfield | `self.bitfield` | `bitfield` | 0 |
| $u_i$ | stp_u | `self.stp_u` | `stp_u` | 0.5 |
| $x_i$ | stp_x | `self.stp_x` | `stp_x` | 1.0 |

### 2.2 입력 계산 (A.2)

$$I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}} a_j - \lambda_r r_i - \beta_w w_i + \lambda_m m_i + \eta_i$$

```python
# runtime.py::_step_torch, line ~508
pre = stp_u * stp_x * self.activation * prev_active  # W_eff = u*x*a
recurrent = self._matvec(pre)                         # sum_j W_ij * pre_j
adapt_force = 0.12 * self.adaptation                  # beta_w * w_i

drive = (
    recurrent                                 # sum_j W_ij_eff * a_j
    + self.config.external_gain * external    # u_i (external input)
    + self.config.goal_gain * self.goal       # goal contribution
    + self.config.replay_mix(mode) * replay   # lambda_H * R_i (hippocampus)
    - self.config.refractory_scale * self.refractory  # -lambda_r * r_i
    - adapt_force                             # -beta_w * w_i
)
```

### 2.3 활성 갱신 (A.3)

$$a_i^{t+1} = (1 - \gamma_a^{(M)}) a_i^t + \kappa_a^{(M)} \tanh(I_i^t)$$

```python
# runtime.py::_step_torch, line ~516
activation = (
    (1.0 - self.config.activation_decay(mode)) * self.activation
    + self.config.activation_gain(mode) * torch.tanh(drive)
).clamp(-1.0, 1.0)
```

### 2.4 억제 갱신 (A.4)

$$r_i^{t+1} = (1 - \gamma_r^{(M)}) r_i^t + \kappa_r^{(M)} (a_i^{t+1})^2$$

```python
# runtime.py::_step_torch, line ~520
refractory = (
    (1.0 - self.config.refractory_decay(mode)) * self.refractory
    + self.config.refractory_gain(mode) * activation.square()
)
```

### 2.5 기억 흔적 (A.5)

$$m_i^{t+1} = (1 - \gamma_m) m_i^t + \gamma_m a_i^{t+1}, \quad \gamma_m = 0.01$$

```python
# runtime.py::_step_torch, line ~524
memory_trace = 0.99 * self.memory_trace + 0.01 * activation
```

### 2.6 적응 변수 (A.6 / J.20)

$$w_i^{t+1} = (1 - \gamma_w) w_i^t + \kappa_w (a_i^{t+1})^2, \quad \gamma_w = 0.005$$

```python
# runtime.py::_step_torch, line ~526
adaptation = ((1.0 - 0.005) * self.adaptation + 0.005 * activation.square()).clamp(0.0, 2.0)
```

### 2.7 비트 갱신 (A.7)

$$b_i^{t+1} = \begin{cases} 1 & a_i > \tau^+ \\ 0 & a_i < \tau^- \\ b_i^t & \text{otherwise} \end{cases}$$

```python
# runtime.py::_step_torch, line ~528
bitfield[activation >= self.config.bit_upper_threshold] = 1   # tau+ = 0.30
bitfield[activation <= self.config.bit_lower_threshold] = 0   # tau- = 0.10
```

### 2.8 STP (Tsodyks-Markram, J.19)

$$u_j \leftarrow u_j + (-u_j/\tau_f + u_0(1-u_j)\delta(t-t_j^*))$$
$$x_j \leftarrow x_j + ((1-x_j)/\tau_r - u_j x_j \delta(t-t_j^*))$$

```python
# runtime.py::_step_torch, line ~492
stp_u = self.stp_u + (-tau_fac_inv * self.stp_u + u_base * (1 - self.stp_u) * spike)
stp_x = self.stp_x + (tau_rec * (1 - self.stp_x) - self.stp_u * self.stp_x * spike)
```

---

## 3. Layer B: 필드 결합

| 수식 | 코드 위치 | 구현 방식 |
|---|---|---|
| $W_{ij}$ (sparse) | `runtime.py::__init__` | `pack_sparse` -> CSR `(values, col_idx, row_ptr)` |
| $\sum_j W_{ij} a_j$ | `runtime.py::_matvec` | `torch.sparse.mm(sparse_weight, x)` |
| $W_{ij}(g) = \exp(-d_g^2/\sigma^2) \chi_{ij}$ | `core/src/engine/field.rs` | Rust 구현 |
| Dale's Law ($w_I/w_E = 4$, E:I = 80:20) | `core/src/engine/kernel.rs::apply_dale_sign` | Rust 구현 |

---

## 4. Layer C: 전역 모드

### 4.1 모드 전환 ($\Pi$)

$$M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$$

```python
# runtime.py::_auto_mode
def _auto_mode(self, external_norm):
    if self.mode is WAKE:
        if self.sleep_pressure > 1.0 and external_norm < wake_threshold:
            return NREM       # 수면 압력 높고 외부 자극 약함
        return WAKE
    if self.mode is NREM:
        if external_norm > wake_threshold * 1.5:
            return WAKE       # 강한 외부 자극 -> 즉시 각성
        if self.sleep_pressure < 0.45:
            return REM        # 수면 압력 충분히 해소 -> REM 전환
        return NREM
    # REM
    if external_norm > wake_threshold or self.sleep_pressure < 0.15:
        return WAKE           # 외부 자극 또는 수면 완료 -> 각성
    return REM
```

### 4.2 수면 압력 (Borbely 2-Process, C.2)

$$\frac{dS}{dt} = \begin{cases} (S_{\max} - S)/\tau_w & \text{WAKE} \\ -S/\tau_s & \text{NREM} \\ -S/(2\tau_s) & \text{REM} \end{cases}$$

```python
# runtime.py::_update_sleep_state
# tau_w = 18.2h = 65520 steps @1ms, tau_s = 4.2h = 15120 steps @1ms
if mode is WAKE:
    self.sleep_pressure += (s_max - self.sleep_pressure) * tau_w_inv
elif mode is NREM:
    self.sleep_pressure -= self.sleep_pressure * tau_s_inv
else:  # REM
    self.sleep_pressure -= self.sleep_pressure * tau_s_inv * 0.5
```

### 4.3 모드별 파라미터 ($\Theta^{(M)}$)

```python
# runtime.py::BrainRuntimeConfig
#                        WAKE   NREM   REM
# activation_decay:      0.18   0.34   0.22
# activation_gain:       0.82   0.52   0.68
# refractory_decay:      0.12   0.26   0.18
# refractory_gain:       0.24   0.12   0.18
# energy_budget:         base   0.5x   0.75x
# replay_mix:            0.08   0.28   0.35
```

---

## 5. Layer D: 해마/기억

| 수식 | 코드 위치 | 구현 |
|---|---|---|
| $H_t = (K_t, V_t, P_t)$ | `HippocampusMemory._keys, _values, _priority` | list[Tensor] |
| $\mathcal{E}(H_t, A_t, U_t)$ | `HippocampusMemory.encode(key, value, priority)` | 용량 초과 시 최저 priority drop |
| $R_t = \mathcal{R}(H_t, c_t)$ | `HippocampusMemory.recall(cue, topk)` | cosine + log(priority) -> softmax weighted sum |
| replay injection | `HippocampusMemory.replay(mode)` | NREM: k=1, REM: k=3 |
| $I_i \leftarrow I_i + \lambda_H R_{i,t}$ | `runtime.py::step` | WAKE: recall, SLEEP: 0.5*recall + 0.5*replay |

### 5.1 encode 조건

```python
# runtime.py::step, line ~589
# WAKE: 외부 입력 또는 목표가 있을 때만 기억
if mode is WAKE and (external_norm > 1e-6 or goal.norm > 1e-6):
    hippocampus.encode(activation, value=memory_trace, priority=priority)
# SLEEP: 기존 기억 + 현재 활성의 혼합을 통합
elif mode is not WAKE and len(hippocampus) > 0:
    consolidated = 0.85 * activation + 0.15 * replay
    hippocampus.encode(consolidated, value=memory_trace, priority=priority * 0.5)
```

---

## 6. Layer E: 전역 요약

| 수식 | 코드 위치 |
|---|---|
| $G_t = (M_t, A_t^{\text{summary}}, H_t, Q_t, \mu_t)$ | `RuntimeStep(step, mode, energy, active_modules, replay_norm, sleep_pressure, arousal, lifecycle_counts)` |
| $\mathcal{W}$ (warm snapshot) | `BrainRuntimeSnapshot`: config + 전체 상태 텐서 + 해마 state_dict |
| snapshot 저장/복원 | `BrainRuntime.snapshot()` / `BrainRuntime.from_snapshot()` |

---

## 7. Layer F: 에이전트 루프 (CE 에너지 이완 경로)

> CE 에너지 이완 추론은 Layer A-B의 brain cell dynamics와는 별도 경로다.
> `engine.py::CEEngine`이 Hopfield 에너지 이완을 수행하고, `sleep.py`가 3위상 학습 순환을 관리한다.

### 7.1 에너지 이완 ($R$)

$$E(m, \phi) = -\frac{1}{2} m^\top W m - m^\top b + \text{portal} \cdot m^\top \hat\phi + E_{\text{cb}} + E_{\text{bypass}}$$

```python
# ce_ops.py::_energy_parts_torch
E_hop    = -0.5 * dot(m, W @ m)        # Hopfield
E_bias   = -dot(m, m0)                 # bias toward initial state
E_portal = -portal * dot(m, phi_hat)   # portal coupling
E_cb     = codebook Boltzmann          # log-sum-exp over codebook
E_bypass = bypass_coeff * dot(m, phi)  # non-conservative bypass
```

### 7.2 이완 루프

```python
# ce_ops.py::_relax_packed_torch
for step in range(n_steps):
    grad = -W @ m - b + portal * phi_hat + ...   # dE/dm
    natural_dir = metric_aware_direction(grad)     # natural gradient
    noise = fdt_noise(T, dt, tau)                  # FDT-compliant noise
    m = m - dt/tau * natural_dir + noise
    m = normalize(m) * norm0                       # norm preservation
    phi = update_phi(phi, m_star, phi_var)          # auxiliary field update
```

### 7.3 Sleep Cycle (3위상 학습)

$$\text{Wake} \to \text{NREM} \to \text{REM} \to \text{evaluate}$$

| 위상 | 코드 함수 | 핵심 연산 |
|---|---|---|
| Wake | `collect_sleep_batch` | teacher 기반 state/target 수집 |
| NREM W 갱신 | `apply_nrem_weight_update` | Laplacian 확산 + 상위 `active_ratio` 가소적 업데이트 |
| NREM 디코더 | `fit_decoder_from_batch` | ridge 회귀로 state->logit 투영 리피팅 |
| NREM 어휘 헤드 | `finetune_vocab_head_from_batch` | AdamW soft-target 미세조정 |
| REM W 갱신 | `apply_rem_weight_update` | 비선택 잔차 저랭크 투영 + 노이즈 재조합 |
| REM 디코더/어휘 | 위와 동일 (rem_weight, rem_mix 적용) | hard sample 가중 |
| 가드셋 보호 | `evaluate_guard_set` | top1/top10/top50 품질 체크, 조건부 롤백 |

### 7.4 위상 비율

$$\text{wake} : \text{nrem} : \text{rem} = \Omega_\Lambda : \Omega_{\text{DM}} : \varepsilon^2 = 68.91\% : 26.23\% : 4.87\%$$

```python
# sleep.py::run_sleep_cycle
phase_profile = {
    "wake": eng.wake_ratio,   # 0.6891
    "nrem": eng.nrem_ratio,   # 0.2623
    "rem":  eng.rem_ratio,    # 0.0487
}
phase_budget = allocate_phase_sample_counts(total_cycle_samples, phase_profile)
```

---

## 8. CE 상수 -> 코드 값

| 수식 기호 | 유도식 | 코드 변수 | 값 |
|---|---|---|---|
| $\text{\_AD}$ | $4/(e^{4/3}\pi^{4/3})$ | `engine._AD` | 0.1726... |
| Portal | $(\text{\_AD}(1-\text{\_AD}))^2$ | `engine.PORTAL` | 0.03120 |
| Bypass | $1/(e^{1/3}\pi^{1/3})$ | `engine.BYPASS` | 0.4892 |
| $T_{\text{wake}}$ | $1/(3+\text{\_AD}(1-\text{\_AD}))$ | `engine.T_WAKE` | 0.3148 |
| $\varepsilon^2$ | bootstrap fixed point | `eng.active_ratio` | 0.0487 |
| $\Omega_{\text{DM}}$ | bootstrap fixed point | `eng.struct_ratio` | 0.2623 |
| $\Omega_\Lambda$ | bootstrap fixed point | `eng.wake_ratio` | 0.6891 |
| $r_c$ | $\pi$ | `eng.sparsity_radius` | 3.1416 |
| target W density | $N=4096, r_c=\pi$ | `eng.target_w_density` | 0.0316 |
| codebook weight | $(\text{\_AD}(1-\text{\_AD}))^2$ | `ce_ops.DEFAULT_CB_W` | 0.03120 |

---

## 9. 백엔드 분기

```
ce_ops.ce_backend(device, requested) -> "cuda" | "rust" | "torch"
    |
    +-- "cuda":  reality_stone.clarus.kernels (CUDA custom ops)    -- 미포함 (선택적)
    +-- "rust":  reality_stone.clarus._rust   (PyO3 바인딩)         -- reality_stone/python/reality_stone/clarus/core/
    +-- "torch": pure PyTorch fallback               -- ce_ops 내부
```

| 연산 | Torch fallback | Rust (`_rust`) | CUDA |
|---|---|---|---|
| pack_sparse | `_pack_sparse_torch` | `nn_ce_pack_sparse` | -- |
| build_metric_basis | `_build_metric_basis_torch` | `nn_ce_metric_basis_fwd` | -- |
| codebook_pull | `_codebook_pull_torch` | `nn_ce_codebook_pull` | -- |
| relax_packed | `_relax_packed_torch` | `nn_ce_relax_fwd` | -- |
| brain_step | `_step_torch` | `nn_brain_step` | -- |
| topk_sparse | torch.topk | `topk_sparse` | -- |
| LBO fused fwd | torch mm | `nn_lbo_fused_fwd` | -- |
| power iter | `linalg.eigh` | `nn_power_iter` | -- |
| gauge lattice | torch mm | `nn_gauge_lattice_fwd` | -- |

---

## 10. 자기참조재귀 구현 대응

AI 응용에서 핵심은 단일 모듈 성능이 아니라 \(S_t \to R(S_t) \to C_t \to S_{t+1}\) 루프가 닫히는지다. 현재 코드 대응은 다음처럼 읽는다.

| 재귀 항 | 의미 | 현재 코드 위치 | 구현 판정 |
|---|---|---|---|
| \(S_t\) | 전역 상태: mode, activation, memory, pressure, lifecycle | `runtime.py::BrainRuntime`, `BrainRuntimeSnapshot` | 부분 구현 |
| \(R(S_t)\) | 내부 이완/수렴: 셀 동역학 반복, sparse activation | `runtime.py::step`, `engine.py::CEEngine`, `ce_ops.py::relax` | 구현됨 |
| \(C_t\) | 자기비평: 예측오차, 일관성, 놀라움, 곡률 점수 | `agent.py`, `stdp.py`, `ce_laplacian.py` 후보 | 부분/분산 구현 |
| \(\mathcal M\) | 기억 갱신과 replay | `runtime.py::HippocampusMemory`, `sleep.py` | 구현됨 |
| \(\phi_t\) | 잔류장/불확실성/탈락 경로 보존 | `engine.py`, `sleep.py` | 부분 구현 |
| \(\mathcal U\) | 다음 전역 상태 구성 | `runtime.py::step`, `snapshot()/from_snapshot()` | 구현됨 |

따라서 현재 구현의 강점은 `runtime.py`의 상태-기억-모드 루프이고, 약점은 \(C_t\)가 하나의 표준 self-critic API로 아직 고정되지 않았다는 점이다. LLM 응용을 강화하려면 새 attention 변형을 늘리기보다, `agent.py`/`runtime.py`/`sleep.py` 사이에 self-critic score와 잔류장 업데이트를 표준 계약으로 묶는 것이 우선이다.

### 10.1 수학량과 로그 항목

닫힌 루프 실험에서는 아래 양을 같은 run에서 기록해야 한다.

| 수학량 | 코드에서 읽을 후보 | 필수성 |
|---|---|---|
| \(\|S_{t+1}-S_t\|\) | `RuntimeStep`, snapshot tensor 차이 | 수축률 \(\hat\rho_t\) 계산 |
| \(\bar c_t\) | agent critic score, STDP learning gate, curvature score | 자기비평 강도 |
| \(I_c\) | critic on/off ablation의 `activation` 또는 logits 차이 | critique가 제어량인지 검증 |
| \(I_m\) | hippocampus recall on/off ablation | memory 재주입 영향 |
| \(\|\phi_t\|\) | `engine.py` / `sleep.py` 잔류장 후보 | 잔류장 유계성 |
| \(M_t\) | `RuntimeMode` | WAKE/NREM/REM 별 \(\rho\) 분리 |
| active ratio | `active_modules / dim` | \(\varepsilon^2\) 근처 수렴 여부 |

최소 closed-loop 판정:

$$
I_c>0,\qquad I_m>0,\qquad
\operatorname{median}_t \hat\rho_t < 1.
$$

더 강한 판정은 open-loop baseline 대비 task score가 좋아지는 동시에 잔류 반경이 커지지 않는 것이다.

$$
G_{\rm rec}>0,
\qquad
\Delta r_\phi \le 0.
$$

### 10.2 계층 gain 로그

`17_AgentLoop.md` F.-1.5의 계층 정리를 코드 실험으로 옮기려면 각 모듈 또는 agent마다 아래 값을 로그로 남긴다.

| 수학량 | 코드 추정 방법 | 판정 |
|---|---|---|
| \(\rho_\ell\) | 같은 입력에서 연속 state delta 비율 `state_delta_next / state_delta` | 모듈 자체 수축률 |
| \(g_\uparrow\) | 하위 모듈 state perturbation이 상위 summary를 바꾸는 norm ratio | aggregation gain |
| \(g_\downarrow\) | 상위 goal/critic perturbation이 하위 activation을 바꾸는 norm ratio | feedback gain |
| \(\rho(G)\) | 추정 gain matrix의 spectral radius | 전체 계층 안정성 |

최소 2층 실험에서는 solver agent와 critic agent만 둔다.

$$
G=
\begin{bmatrix}
\rho_{\rm solver} & g_{\rm down}\\
g_{\rm up} & \rho_{\rm critic}
\end{bmatrix}.
$$

안정 조건은

$$
\rho(G)<1,
$$

또는 보수적으로

$$
\max(\rho_{\rm solver},\rho_{\rm critic})
+
\sqrt{g_{\rm up}g_{\rm down}}
<1
$$

로 로그 판정할 수 있다. 이 값이 1에 가까워지면 상위 critic이 하위 solver를 교정하는 것이 아니라 흔들어 불안정하게 만드는 regime으로 본다.

---

## 11. 파일 책임 분리

| 파일 | 책임 | Layer |
|---|---|---|
| `reality_stone/python/reality_stone/clarus/runtime.py` | 셀 동역학, 모드 전환, 해마, 생애주기, 스냅샷 | A, B, C, D, E |
| `reality_stone/python/reality_stone/clarus/engine.py` | CE 에너지 이완, 디코딩, 상태 분할, 곡률 억제 | F (이완), 6장 |
| `reality_stone/python/reality_stone/clarus/ce_ops.py` | 수치 백엔드 분기, 에너지/이완/메트릭/PQ | F (수치 핵심) |
| `reality_stone/python/reality_stone/clarus/sleep.py` | Wake/NREM/REM 학습 순환, 가드셋, 디코더 리피팅 | F (학습) |
| `reality_stone/python/reality_stone/clarus/device.py` | 디바이스 자동 감지 | 인프라 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs` | brain_step 핵심 루프, Dale's Law | A |
| `reality_stone/python/reality_stone/clarus/core/src/engine/field.rs` | 필드 결합, 리만 거리 | B |
| `reality_stone/python/reality_stone/clarus/core/src/engine/manifold.rs` | 다양체 연산 | B |
| `reality_stone/python/reality_stone/clarus/core/src/engine/nn_ops.rs` | NN 연산 (topk, LBO, gauge) | 2장 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/ce_riemann.rs` | CE 리만 수치 | 물리 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/constants.rs` | 물리 상수 유도 | 3_상수 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/config.rs` | 런타임 설정 | 인프라 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/runtime_types.rs` | CellState, Mode 등 타입 | A, C |

---

## 12. 미구현 대조

| 수식/개념 | 문서 위치 | 코드 상태 |
|---|---|---|
| STDP 적격 흔적 | F.14 | 미구현 |
| 4종 신경조절 (DA/NE/5HT/ACh) | F.19 | 미구현 (단일 스칼라) |
| 소뇌 전방 모델 | F.20 | 미구현 |
| 작업 기억 용량 제한 $|h_t| \le T_h$ | F.20 | 미구현 |
| 뇌파 대역 분해 | F.21 | 미구현 |
| (C3) 메타인지 재귀 루프 | F.17 | 미구현 |
| Cold checkpoint ($\mathcal{C}$) | 14장 7절 | 미구현 (warm만 있음) |
| Live journal ($\mathcal{J}$) | 14장 7절 | 미구현 |
| 섭동적 채널 혼합 | 2장 2.3절 | 미구현 |
| 교차 주파수 결합 게이트 | 2장 6절 | 미구현 |
```
