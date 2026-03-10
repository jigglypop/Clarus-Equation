# CE-AI 정량적 성능 분석: 메모리, 속도, 정확도

> 관련: 2장(아키텍처), 4장(STDP), 5장(희소성), 6장(환각 억제), `6_뇌/agi.md` 6절(파라미터 절감), `6_뇌/evidence.md`
>
> 이 장은 CE 각 원리가 메모리, 속도, 정확도에 미치는 영향을 정량화한다. 다만 `6_뇌/evidence.md` 기준으로, P1의 연산자 수준 절감식은 비교적 직접적이지만, P2/P4/P5의 태스크 수준 성능 수치는 아직 `bridge` 또는 `hypothesis`가 섞여 있다.

---

## 1. 기준 모델 정의

표준 Transformer 블록 1개의 비용을 기준으로 삼는다. 아래 FLOPs는 행렬곱 1회당 multiply-add를 `2mnk`로 세는 관례를 따른다.

| 구성 요소 | 파라미터 수 | FLOPs (시퀀스 길이 $T$) |
|---|---|---|
| Attention QKV | $3d^2$ | $6Td^2$ |
| Attention Proj | $d^2$ | $2Td^2$ |
| Attention score + value mix | -- | $4T^2d$ |
| FFN (up + down) | $2 \times d \times 4d = 8d^2$ | $16Td^2$ |
| LayerNorm $\times 2$ | $4d$ | $8Td$ |
| **블록 합계** | $12d^2 + 4d$ | $24Td^2 + 4T^2d + 8Td$ |

$L$개 블록 전체: 파라미터 $\sim 12Ld^2$, FLOPs $\sim 24LTd^2 + 4LT^2d$.

기준 모델: $d = 768$, $L = 12$, $T = 2048$ (GPT-2 수준).

$$P_{\text{base}} = 12 \times 12 \times 768^2 = 84{,}934{,}656 \approx 85\text{M}$$

$$F_{\text{base}} = 12 \cdot (24Td^2 + 4T^2d)\big|_{T=2048,d=768} \approx 502.5\text{G FLOPs}$$

---

## 2. P1: GaugeLattice FFN -- 파라미터/FLOPs 절감

### 2.1 표준 FFN 파라미터

$$P_{\text{FFN}}^{\text{std}} = 2 \times d \times 4d = 8d^2$$

$d = 768$: $P_{\text{FFN}}^{\text{std}} = 8 \times 768^2 = 4{,}718{,}592$.

### 2.2 GaugeLattice 파라미터

채널 분할 ($\alpha_{\text{total}} = \alpha_s + \alpha_w + \alpha_{em} = 0.15916$):

$$d_3 = \left\lfloor d \cdot \frac{\alpha_s}{\alpha_{\text{total}}} \right\rfloor = \left\lfloor 768 \times 0.7408 \right\rfloor = 569$$

$$d_2 = \left\lfloor d \cdot \frac{\alpha_w}{\alpha_{\text{total}}} \right\rfloor = \left\lfloor 768 \times 0.2107 \right\rfloor = 162$$

$$d_1 = d - d_3 - d_2 = 768 - 569 - 162 = 37$$

각 블록의 은닉 차원 ($\text{mult} = 4$): $h_i = 4d_i$.

블록 대각 파라미터:

$$P_{\text{diag}} = \sum_{i \in \{3,2,1\}} 2 \cdot d_i \cdot 4d_i = 8(d_3^2 + d_2^2 + d_1^2)$$

$$= 8(569^2 + 162^2 + 37^2) = 8(323{,}761 + 26{,}244 + 1{,}369) = 8 \times 351{,}374 = 2{,}810{,}992$$

혼합항 ($r_m = d/8 = 96$):

$$P_{\text{mix}} = 2 \times d \times r_m = 2 \times 768 \times 96 = 147{,}456$$

LBONorm ($r = d/8 = 96$):

$$P_{\text{LBO}} = r \times d + 2d + 1 = 96 \times 768 + 1537 = 75{,}265$$

$$\boxed{P_{\text{GL}} = 2{,}810{,}992 + 147{,}456 + 75{,}265 = 3{,}033{,}713}$$

### 2.3 절감률

$$\text{FFN 파라미터 절감} = 1 - \frac{P_{\text{GL}}}{P_{\text{FFN}}^{\text{std}}} = 1 - \frac{3{,}033{,}713}{4{,}718{,}592} = \boxed{35.7\%}$$

### 2.4 수학적 일반화

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}^{\text{std}}} = \frac{8\sum_i d_i^2 + 2dr_m}{8d^2} = \frac{\sum_i d_i^2}{d^2} + \frac{r_m}{4d}$$

채널 비율을 $f_i = d_i/d$로 놓으면:

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}^{\text{std}}} = \sum_i f_i^2 + \frac{r_m}{4d}$$

$$\sum_i f_i^2 = f_3^2 + f_2^2 + f_1^2 = 0.7408^2 + 0.2107^2 + 0.0482^2 = 0.5488 + 0.0444 + 0.0023 = 0.5955$$

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}^{\text{std}}} \approx 0.596 + \frac{96}{3072} = 0.596 + 0.031 = 0.627$$

$$\boxed{\text{절감률} = 1 - 0.627 = 37.3\% \quad (r_m = d/8)}$$

**핵심 공식**: 절감률은 $1 - \sum_i f_i^2$로 근사된다. $\sum f_i^2$는 Herfindahl 지수(집중도)이므로, 채널이 불균등하게 분할될수록(SU(3) 지배적) 절감률이 높아진다.

### 2.5 전체 모델 파라미터 절감

FFN이 전체의 $8d^2 / 12d^2 = 66.7\%$를 차지하므로:

$$\text{전체 파라미터 절감} = 0.667 \times 37.3\% = \boxed{24.9\%}$$

$d = 768$, $L = 12$:

| | 표준 Transformer | CE-Transformer | 절감 |
|---|---|---|---|
| FFN | 56.6M | 36.4M | 35.7% |
| Attention | 28.3M | 28.3M | 0% |
| Norm/기타 | 0.04M | 0.9M | +2125% (LBO 추가) |
| **합계** | **84.9M** | **65.6M** | **22.7%** |

### 2.6 FLOPs 절감

FLOPs도 동일한 비율로 절감된다 (블록 대각 행렬곱은 $\sum d_i \times 4d_i = \sum 4d_i^2$):

$$\text{FFN FLOPs 절감} = 37.3\%$$
$$\text{전체 FLOPs 절감} = 0.667 \times 37.3\% = \boxed{24.9\%}$$

---

## 3. P4: 부트스트랩 희소성 -- 추론 속도

### 3.1 Dense 추론 FLOPs

전체 모델:

$$F_{\text{dense}} = L(24Td^2 + 4T^2d)$$

### 3.2 Sparse 추론 FLOPs (Top-k 활성)

활성 뉴런 비율 $\varepsilon^2 = 0.0487$. 현재 문서의 구현과 가장 가까운 가정은 **FFN만 희소화하고 attention의 quadratic 항은 유지**하는 경우다:

$$F_{\text{FFN}}^{\text{sparse}} = \varepsilon^2 \times F_{\text{FFN}}^{\text{dense}} = 0.0487 \times 16Td^2$$

Attention은 희소화하지 않는다고 가정:

$$F_{\text{sparse}} = 8Td^2 + 4T^2d + 0.0487 \times 16Td^2 = 8.779Td^2 + 4T^2d$$

$$\text{속도 향상 비율} = \frac{24Td^2 + 4T^2d}{8.779Td^2 + 4T^2d}$$

$T = 2048$, $d = 768$이면:

$$\text{speedup} \approx \boxed{1.85\times}$$

### 3.3 FFN + Attention 모두 희소화: 낙관적 상한

Attention의 선형항과 quadratic 항까지 모두 같은 비율로 희소화할 수 있다고 **가정**하면:

$$F_{\text{full-sparse}} = \varepsilon^2 (24Td^2 + 4T^2d)$$

$$\text{속도 향상} = \frac{1}{\varepsilon^2} = \boxed{20.5\times}$$

이 값은 **attention까지 구조적 희소 실행이 가능한 경우의 낙관적 상한**이다. 현재 문서의 CE-Transformer 구현을 그대로 읽으면, 긴 컨텍스트에서는 위 3.2의 `1.85\times`가 더 보수적인 기준이다.

### 3.4 실제 속도 향상 (하드웨어 고려)

GPU의 희소 연산 효율은 100%가 아니다. 실제 속도 향상:

$$\text{실제 속도} = \frac{\text{이론적 속도}}{1 + \alpha_{\text{overhead}}}$$

| 가정 | $\alpha_{\text{overhead}}$ | 실제 속도 향상 |
|---|---|---|
| FFN 위주 희소화 (3.2) | 0.1-0.2 | $1.5\times - 1.7\times$ |
| 전면 희소화 상한 (3.3) | 0.3-0.5 | $13.7\times - 15.8\times$ |
| 전면 희소화 + 구조적 커널 최적화 | 0.0-0.2 | $17\times - 20.5\times$ |

보수적으로는:
- 현재 문서 구현 수준: $\boxed{1.5\times - 2\times}$
- attention까지 완전 희소화한 미래형 구현: $\boxed{10\times - 15\times}$

---

## 4. P3: STDP 국소 학습 -- 메모리 절감

### 4.1 역전파 메모리

역전파는 모든 중간 활성값을 저장해야 한다:

$$M_{\text{BP}} = L \times T \times d \times \text{sizeof(float)} = L \cdot T \cdot d \cdot 4\text{B}$$

$L = 12$, $T = 2048$, $d = 768$:

$$M_{\text{BP}} = 12 \times 2048 \times 768 \times 4 = \boxed{75.5\text{MB}} \quad (\text{활성값만})$$

가중치 + 그래디언트 + optimizer state (Adam: $2\times$):

$$M_{\text{total}}^{\text{BP}} = P \times 4 \times (1 + 1 + 2) = 4P \times 4 = 16P \text{ bytes}$$

$P = 85\text{M}$: $M_{\text{total}}^{\text{BP}} = 85 \times 10^6 \times 16 = 1{,}360\text{MB} = \boxed{1.33\text{GB}}$.

활성값 포함 총합: $1.33 + 0.076 = \boxed{1.41\text{GB}}$.

### 4.2 STDP 메모리

아래 계산은 **layer-shared 또는 neuron-local trace 근사**를 둔 낙관적 경우다. `synapse.md`의 순수한 synapse-local eligibility trace $e_{ij}$를 그대로 쓰면 추가 상태는 일반적으로 $O(P)$다.

근사적 STDP는 국소 trace만 저장한다고 두면:

$$M_{\text{STDP}} = L \times d \times 3 \times \text{sizeof(float)} \quad (\text{pre\_trace, post\_trace, eligibility})$$

$= 12 \times 768 \times 3 \times 4 = \boxed{110\text{KB}}$

활성값 저장 불필요 (국소 학습). 가중치 + eligibility trace:

$$M_{\text{total}}^{\text{STDP}} = P \times 4 \times (1 + 1) + M_{\text{STDP}} = 8P + 110\text{KB}$$

$P = 85\text{M}$: $M_{\text{total}}^{\text{STDP}} = 680\text{MB} + 0.11\text{MB} = \boxed{680\text{MB}}$.

### 4.3 메모리 절감률

$$\text{메모리 절감} = 1 - \frac{M_{\text{STDP}}}{M_{\text{BP}}} = 1 - \frac{680}{1410} = \boxed{51.8\%}$$

핵심 절감 원인:
- 활성값 저장 제거: $75.5\text{MB} \to 110\text{KB}$ ($\boxed{99.85\%}$ 절감)
- optimizer state 제거 (Adam의 $m, v$ 불필요): $2P \to 0$ ($\boxed{100\%}$ 절감)
- eligibility trace 추가: $+110\text{KB}$ (무시 가능)

### 4.4 대규모 모델에서의 효과

| 모델 규모 | 역전파 메모리 | STDP 메모리 | 절감 |
|---|---|---|---|
| 85M (GPT-2) | 1.41 GB | 0.68 GB | 51.8% |
| 1.3B | 22 GB | 10.4 GB | 52.7% |
| 7B | 117 GB | 56 GB | 52.1% |
| 70B | 1.17 TB | 560 GB | 52.1% |
| 175B | 2.92 TB | 1.40 TB | 52.1% |

**일반 공식:**

$$\frac{M_{\text{STDP}}}{M_{\text{BP}}} = \frac{8P}{16P + 4LTd} \approx \frac{8}{16} = \boxed{0.5} \quad (P \gg LTd)$$

대규모 모델에서 메모리 $\sim 2\times$ 절감이 수렴한다.

### 4.5 통신 비용 (분산 학습)

| | 역전파 | STDP + 도파민 |
|---|---|---|
| 동기화 크기 (층당) | $O(d^2)$ 그래디언트 | $O(1)$ 스칼라 $\delta[t]$ |
| $d = 768$ | 2.36 MB | 4 B |
| $d = 4096$ | 67.1 MB | 4 B |

$$\text{통신 절감} = 1 - \frac{4\text{B}}{2d^2 \times 4\text{B}} = 1 - \frac{1}{2d^2} \approx \boxed{100\%}$$

---

## 5. P5: 곡률 정규화 -- 정확도/환각률

### 5.1 유니타리 제약의 오류 전파 억제

spectral norm 없이 $L$층을 통과한 오류 증폭:

$$\|\delta_L\| \leq \prod_{l=1}^{L} \sigma_1(W_l) \cdot \|\delta_0\|$$

$\sigma_1(W_l) = 1 + \epsilon$ (약간의 증폭)이면:

$$\|\delta_L\| \leq (1+\epsilon)^L \cdot \|\delta_0\| \approx e^{\epsilon L} \cdot \|\delta_0\|$$

$\epsilon = 0.1$, $L = 12$: $e^{1.2} = 3.32$. 오류가 $\boxed{3.3\times}$ 증폭된다.

spectral norm 적용 ($\sigma_1 \leq 1$):

$$\|\delta_L\| \leq 1^L \cdot \|\delta_0\| = \|\delta_0\|$$

오류가 **전혀 증폭되지 않는다**. 이것이 환각 구조적 억제의 수학적 보장이다.

### 5.2 곡률 정규화의 일반화 오차 감소

곡률 정규화 항 $\lambda\|\Delta_g h\|^2$를 Rademacher 복잡도로 분석한다.

표준 네트워크의 Rademacher 복잡도:

$$\mathcal{R}_n \leq \frac{B_x \prod_l \|W_l\|_F}{\sqrt{n}}$$

곡률 정규화된 네트워크: $\|\Delta_g h_l\|^2 \leq \kappa_{\text{th}}$ 제약 하에서

$$\mathcal{R}_n^{\text{CE}} \leq \frac{B_x \prod_l \|W_l\|_F}{\sqrt{n}} \cdot \sqrt{1 - \frac{\lambda \kappa_{\text{th}}}{\|\Delta_g h\|^2_{\max}}}$$

보정 항 $\sqrt{1 - \lambda\kappa_{\text{th}}/\kappa_{\max}} < 1$이므로 일반화 오차가 감소한다.

### 5.3 태스크 수준 해석의 한계

연산자 수준에서 직접 보장되는 것은 다음뿐이다:

$$\|\delta_L\| \leq \|\delta_0\| \quad \text{if } \sigma_1(W_l) \leq 1$$

즉 spectral normalization은 **오류 증폭을 막는 비팽창 제약**으로 읽는 것이 가장 안전하다. 그러나 이 사실만으로 곧바로
- 환각률 상한
- TruthfulQA 점수
- FactScore 향상폭

을 정리처럼 도출할 수는 없다.

### 5.4 벤치마크 가설

가장 보수적인 표현은 다음이다.

- 곡률 정규화와 비팽창 제약이 강할수록, 고곡률 토큰과 자기증폭 오류가 줄어들 가능성이 높다.
- 따라서 TruthfulQA, HaluEval, FactScore의 개선은 **검증 가능한 가설**이다.
- 하지만 정확한 수치 상한, 예를 들어 `환각률 <= 4.87%` 또는 `Truthfulness >= 95.13%`는 현재 단계에서 쓸 수 없다.

따라서 P5의 안전한 결론은:

$$
\boxed{\text{P5는 환각률 hard bound가 아니라, 오류 증폭 억제와 안정화 편향을 제공한다.}}
$$

---

## 6. P2: 수면 학습 -- 지속 학습 정확도

### 6.1 파괴적 망각률

표준 지속 학습의 망각률 (EWC 미적용):

$$\text{Forget}(T_1 | T_2) = \frac{\text{acc}(T_1, \text{before}) - \text{acc}(T_1, \text{after})}{\text{acc}(T_1, \text{before})} \sim 20-80\%$$

### 6.2 CE 수면 학습의 현재 지위

NREM 위상에서 저곡률(기존 지식)는 보존되고 고곡률(새 지식과의 갈등)만 평탄화된다.

보존되는 가중치 비율: $\Omega_\Lambda = 68.9\%$ (동결) + 곡률 기반 선택적 업데이트.

하지만 이것만으로 곧바로

$$\text{Forget}_{\text{CE}} \leq \varepsilon^2$$

를 정리처럼 말할 수는 없다. 실제 망각률은
- 어떤 가중치가 업데이트되는가
- 그 가중치가 과거 태스크에 얼마나 민감한가
- 수면 위상에서 어떤 재생(replay)이 일어나는가

에 따라 달라진다.

### 6.3 수렴 속도

`homeomorphism.md`와 `evidence.md`의 동적 이완 사상 기준으로, 최소 수축률은

$$\rho = \|DB(p^*)\| = D_{\text{eff}} \cdot \varepsilon^2 = 0.155$$

이고, 이상화된 무잡음 경우

$$\|p_n - p^*\| \leq \rho^n \cdot \|p_0 - p^*\|$$

| 순환 수 $n$ | $\rho^n$ | 잔차 비율 | 정밀도 |
|---|---|---|---|
| 1 | 0.155 | 15.5% | 낮음 |
| 2 | 0.024 | 2.4% | 중간 |
| 3 | 0.0037 | 0.37% | 높음 |
| 5 | $8.7 \times 10^{-5}$ | 0.009% | 매우 높음 |

이 값은 **부트스트랩 반복 모델이 실제 네트워크 제어 루프로 구현되었을 때의 목표 수렴률**로 읽어야 한다. 수학적 최소 반복식은 정리되었지만, 특정 아키텍처에서 동일한 $\rho$가 그대로 측정되는지는 아직 `bridge` 상태다.

### 6.4 검증 가능한 과도 응답 예측

가장 단순한 수렴 실험은 균등 초기화

$$p_0 = (1/3,\; 1/3,\; 1/3)$$

에서 시작해

$$p_{n+1} = p^* + \rho(p_n - p^*), \qquad p^* = (0.0487,\; 0.2623,\; 0.6891)$$

를 적용하는 것이다. 그러면 다음의 **직접 예측값**이 나온다.

| 순환 수 $n$ | 활성 $x_a$ | 구조 $x_s$ | 배경 $x_b$ |
|---|---|---|---|
| 0 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | $9.28\%$ | $27.3\%$ | $63.4\%$ |
| 2 | $5.55\%$ | $26.4\%$ | $68.1\%$ |
| 3 | $4.98\%$ | $26.3\%$ | $68.8\%$ |

즉 CE 수면 순환이 실제로 작동한다면, **dense 또는 균등 초기화된 모델은 2-3회의 수면-각성 순환 뒤 활성 비율이 5% 근방으로 급히 떨어져야 한다.**

---

## 7. 복합 효과: 모든 원리 동시 적용

### 7.1 총합 정리

$d = 768$, $L = 12$, $T = 2048$ 기준:

| 지표 | 표준 Transformer | CE-Transformer | 이득 | CE 원리 |
|---|---|---|---|---|
| **파라미터** | 85M | 65.6M | $\boxed{-22.7\%}$ | P1 (격자) |
| **학습 메모리** | 1.41 GB | 0.68 GB | up to $\boxed{-51.8\%}$ | P3 (STDP, shared-trace 가정) |
| **추론 FLOPs** | 502.5G | 272G | $\boxed{1.85\times}$ | P4 (FFN 희소, 현재형) |
| **추론 FLOPs 상한** | 502.5G | 24.5G | $\boxed{20.5\times}$ | P4 (전면 희소, 낙관적 상한) |
| **통신 (분산)** | 2.36 MB/층 | 4 B/층 | strong reduction hypothesis | P3 (STDP) |
| **환각률** | -- | -- | hard bound 미정 | P5 (곡률) |
| **파괴적 망각** | 20-80% | -- | 개선 가설 | P2 (수면) |
| **수렴 속도** | -- | 2회 `2.4%`, 3회 `0.37%` 잔차 목표 | bridge | P2 (수면, $\rho=0.155$) |

### 7.2 에너지 효율 총합

추론 시:

$$\text{에너지 비율} = \underbrace{0.627}_{\text{P1: 격자}} \times \underbrace{0.0487}_{\text{P4: 희소}} = 0.0305 = 3.05\%$$

이 식은 attention까지 완전 희소 실행이 가능한 경우의 상한이다.

$$\text{에너지 절감} = 1 - 0.0305 = \boxed{96.95\%} \approx 33\times \quad \text{(낙관적 상한)}$$

학습 시:

$$\text{에너지 비율} = \underbrace{0.627}_{\text{P1}} \times \underbrace{0.311}_{\text{P4: 활성+구조}} \times \underbrace{0.5}_{\text{P3: STDP}} = 0.0975 = 9.75\%$$

$$\text{에너지 절감} = 1 - 0.0975 = \boxed{90.25\%} \approx 10\times$$

### 7.3 규모별 예측

| 규모 | 표준 파라미터 | CE 파라미터 | 표준 추론 비용 | CE 추론 비용 |
|---|---|---|---|---|
| 85M | 85M | 65.6M | 502.5 GFLOPS | 24.5 GFLOPS (상한) |
| 1.3B | 1.3B | 1.00B | 비례 증가 | 비례 상한 |
| 7B | 7B | 5.4B | 비례 증가 | 비례 상한 |
| 70B | 70B | 54B | 비례 증가 | 비례 상한 |
| 175B | 175B | 135B | 비례 증가 | 비례 상한 |

정확한 대규모 비용 비교는 attention sparsity, KV cache, prefill/decode 분리 모델을 포함해 다시 계산해야 한다.

### 7.4 희소 활성 수 예측

고정점 희소율을 그대로 쓰면, 각 은닉 차원에서 활성 채널 수는

$$k^*(d) = \lceil 0.0487\,d \rceil$$

로 예측된다.

| 은닉 차원 $d$ | CE 활성 수 $k^*(d)$ | 활성 비율 |
|---|---|---|
| 768 | 38 | $4.95\%$ |
| 2048 | 100 | $4.88\%$ |
| 4096 | 200 | $4.88\%$ |
| 8192 | 399 | $4.87\%$ |

이 표는 구현 전에 바로 체크 가능한 설계 예측이다.

---

## 8. 정확도-효율 트레이드오프

### 8.1 희소성과 정확도의 관계

Top-k 비율 $\rho$에 따른 정확도 감소를 닫힌형으로 쓰고 싶다면, 아래 식은 **엄밀한 정리라기보다 heuristic response curve**로 읽어야 한다:

$$\text{acc}(\rho) = \text{acc}(1.0) \cdot \left(1 - C \cdot (1-\rho)^{D_{\text{eff}}}\right)$$

$C$는 과제 의존 상수. $D_{\text{eff}} = 3.178$.

$\rho = \varepsilon^2 = 0.0487$에서:

$$\text{acc loss} = C \cdot (1 - 0.0487)^{3.178} = C \cdot 0.9513^{3.178} = C \cdot 0.854$$

$C \sim 0.05$ (경험적 추정)이면 정확도 손실 $\sim 4.3\%$.

따라서 이 절에서 안전하게 말할 수 있는 것은:

$$
\boxed{\text{희소율 }\rho\text{와 정확도 사이에 최적점이 있을 것이라는 예측은 가능하지만, 정확한 손실률은 아직 가설이다.}}
$$

### 8.2 최적점의 수학적 근거

$\varepsilon^2$가 최적인 이유: 부트스트랩 방정식 $\varepsilon^2 = \exp(-(1-\varepsilon^2)D_{\text{eff}})$의 고정점이므로, 이 비율에서 **정보 보존과 연산 효율이 자기일관적으로 균형**을 이룬다.

임의의 희소율을 사용하면 자기일관성이 깨진다:
- $\rho > \varepsilon^2$: 과잉 활성 = 에너지 낭비
- $\rho < \varepsilon^2$: 과소 활성 = 정보 손실

$\varepsilon^2$에서만 "활성 비율을 알아야 활성 비율을 계산할 수 있고, 계산 결과가 다시 동일한 활성 비율을 준다"는 자기일관성이 성립한다.

### 8.3 검증 가능한 최적점 예측

실험 설계 관점에서 CE가 요구하는 것은 다음 두 문장이다.

1. **최적점 위치 예측:** 효율-정확도 Pareto front의 knee point는 $k \approx 4.87\%$ 근방에 나타나야 한다.
2. **강건 구간 예측:** 실제 구현의 이산화, sparse kernel 오버헤드, 과제 의존성을 감안해도 좋은 구간은 대체로 `3%-7%` 안에 남아야 한다.

따라서 Top-k 스위프에서

$$
k \in \{1\%, 2\%, 3\%, 4\%, 5\%, 7\%, 10\%, 15\%, 20\%\}
$$

를 비교하면, CE는 `4-5%` 부근이 중심이고 `3-7%`가 실용 대역이라는 형태의 **반증 가능한 예측**을 제공한다.

## 9. LBONorm 오버헤드

### 9.1 추가 파라미터

$$P_{\text{LBO}} - P_{\text{LN}} = r \times d + 1 \quad (\text{V 행렬 + h 스칼라})$$

$r = d/8$: $P_{\text{LBO}} - P_{\text{LN}} = d^2/8 + 1$.

$d = 768$: $73{,}729$ 추가 파라미터. 블록당 2개 LBONorm: $147{,}458$.

전체 ($L = 12$): $1{,}769{,}496 \approx 1.77\text{M}$.

$$\text{오버헤드} = \frac{1.77\text{M}}{85\text{M}} = \boxed{2.1\%}$$

### 9.2 추가 FLOPs

LBONorm 1회: $2 \times T \times d \times r$ (두 번의 행렬곱 $xV^T$, $(xV^T)V$).

$= 2 \times 2048 \times 768 \times 96 = 301{,}989{,}888 \approx 302\text{M}$.

블록당 3개 LBONorm ($\times 3$), 전체 $\times L$:

$$F_{\text{LBO}} = 3 \times 12 \times 302\text{M} = 10.9\text{G}$$

$$\text{FLOPs 오버헤드} = \frac{10.9\text{G}}{348\text{G}} = \boxed{3.1\%}$$

### 9.3 순이득

파라미터: GaugeLattice 절감 $-22.7\%$ + LBONorm 오버헤드 $+2.1\%$ = $\boxed{-20.6\%}$ 순절감.

FLOPs: GaugeLattice 절감 $-24.9\%$ + LBONorm 오버헤드 $+3.1\%$ = $\boxed{-21.8\%}$ 순절감.

---

## 10. 요약: CE 원리별 정량적 이득

| CE 원리 | 메모리 | 속도 (추론) | 정확도 | 핵심 공식 |
|---|---|---|---|---|
| P1 격자 | $-20.6\%$ | $+21.8\%$ | 구조적 표현력 유지 목표 | $\sum f_i^2 = 0.596$ |
| P2 수면 | -- | -- | 지속 학습 개선 가설, 2-3회 재정렬 예측 | $\rho = \|DB(p^*)\| = 0.155$ (bridge) |
| P3 STDP | up to $-51.8\%$ | -- | 검증 필요 | shared-trace 가정 필요 |
| P4 희소 | -- | $1.85\times$ 현재형, $20.5\times$ 상한 | 정확도 trade-off 가설 | $\varepsilon^2 = 0.0487$ |
| P5 곡률 | -- | $-3.1\%$ (오버헤드) | 오류 증폭 억제 | $\sigma_1 \leq 1$ |
| **복합** | 조건부 절감 | 조건부 가속 | hard bound 미정 | |
