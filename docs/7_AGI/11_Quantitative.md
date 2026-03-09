# CE-AI 정량적 성능 분석: 메모리, 속도, 정확도

> 관련: 2장(아키텍처), 4장(STDP), 5장(희소성), 6장(환각 억제), `6_뇌/agi.md` 6절(파라미터 절감)
>
> 이 장은 CE 각 원리가 메모리, 속도, 정확도에 미치는 정량적 이득을 수학적으로 유도한다. 모든 계산은 CE 상수 $\alpha_s = 0.11789$, $\varepsilon^2 = 0.04865$, $D_{\text{eff}} = 3.178$에서 연역된다.

---

## 1. 기준 모델 정의

표준 Transformer 블록 1개의 비용을 기준으로 삼는다:

| 구성 요소 | 파라미터 수 | FLOPs (시퀀스 길이 $T$) |
|---|---|---|
| Attention QKV | $3d^2$ | $6Td^2$ |
| Attention Proj | $d^2$ | $2Td^2$ |
| FFN (up + down) | $2 \times d \times 4d = 8d^2$ | $16Td^2$ |
| LayerNorm $\times 2$ | $4d$ | $8Td$ |
| **블록 합계** | $12d^2 + 4d$ | $24Td^2 + 8Td$ |

$L$개 블록 전체: 파라미터 $\sim 12Ld^2$, FLOPs $\sim 24LTd^2$.

기준 모델: $d = 768$, $L = 12$, $T = 2048$ (GPT-2 수준).

$$P_{\text{base}} = 12 \times 12 \times 768^2 = 84{,}934{,}656 \approx 85\text{M}$$

$$F_{\text{base}} = 24 \times 12 \times 2048 \times 768^2 \approx 348\text{G FLOPs}$$

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

전체 모델: $F_{\text{dense}} = 24LTd^2$.

### 3.2 Sparse 추론 FLOPs (Top-k 활성)

활성 뉴런 비율 $\varepsilon^2 = 0.0487$. FFN의 연산만 희소화한다고 가정:

$$F_{\text{FFN}}^{\text{sparse}} = \varepsilon^2 \times F_{\text{FFN}}^{\text{dense}} = 0.0487 \times 16Td^2$$

Attention은 희소화하지 않는다고 가정 ($8Td^2$ 유지):

$$F_{\text{sparse}} = 8Td^2 + 0.0487 \times 16Td^2 = 8Td^2 + 0.779Td^2 = 8.779Td^2$$

$$\text{속도 향상 비율} = \frac{F_{\text{dense}}}{F_{\text{sparse}}} = \frac{24Td^2}{8.779Td^2} = \boxed{2.73\times}$$

### 3.3 FFN + Attention 모두 희소화

Attention에서도 Top-k 활성($\varepsilon^2$) 적용:

$$F_{\text{full-sparse}} = \varepsilon^2 \times 24Td^2 = 0.0487 \times 24Td^2 = 1.169Td^2$$

$$\text{속도 향상} = \frac{24}{1.169} = \boxed{20.5\times}$$

이것이 5장에서 예측한 "$\sim 20\times$ 절감"의 정확한 값이다.

### 3.4 실제 속도 향상 (하드웨어 고려)

GPU의 희소 연산 효율은 100%가 아니다. 실제 속도 향상:

$$\text{실제 속도} = \frac{\text{이론적 속도}}{\text{희소 오버헤드}} = \frac{20.5\times}{1 + \alpha_{\text{overhead}}}$$

| 하드웨어 | $\alpha_{\text{overhead}}$ | 실제 속도 향상 |
|---|---|---|
| 구조적 희소성 (2:4, A100) | 0.0 | $20.5\times$ |
| 비구조적 희소성 (마스크) | 0.3-0.5 | $13.7\times - 15.8\times$ |
| Top-k 선택 오버헤드 | 0.1-0.2 | $17.1\times - 18.6\times$ |

보수적 추정: $\boxed{10\times - 15\times}$ 실제 속도 향상.

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

STDP는 국소 trace만 저장한다:

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

### 5.3 환각률의 이론적 상한

CE 유니타리 제약 + 곡률 정규화 하에서 환각률의 상한:

$$P(\text{hallucination}) \leq \varepsilon^2 = \boxed{4.87\%}$$

유도: 환각 = 잠재 공간에서 고곡률 영역에 진입. 유니타리 제약으로 오류 증폭이 차단되므로, 환각은 초기 입력 단계의 오류에서만 발생한다. 초기 입력의 선택률이 $\varepsilon^2$이므로, 이것이 환각률의 상한.

### 5.4 TruthfulQA 예측

표준 LLM의 TruthfulQA 점수: $\sim 30-50\%$ (사실성).

CE 제약 후 예측:

$$\text{Truthfulness}_{\text{CE}} \geq 1 - \varepsilon^2 = 95.13\%$$

개선폭: $\boxed{45-65\%p}$ 향상 (이론적 상한).

실제 예상: 곡률 정규화가 모든 환각을 제거하지는 못하므로, $\sim 70-85\%$ 수준이 현실적이다.

---

## 6. P2: 수면 학습 -- 지속 학습 정확도

### 6.1 파괴적 망각률

표준 지속 학습의 망각률 (EWC 미적용):

$$\text{Forget}(T_1 | T_2) = \frac{\text{acc}(T_1, \text{before}) - \text{acc}(T_1, \text{after})}{\text{acc}(T_1, \text{before})} \sim 20-80\%$$

### 6.2 CE 수면 학습의 망각률

NREM 위상에서 저곡률(기존 지식)는 보존되고 고곡률(새 지식과의 갈등)만 평탄화된다.

보존되는 가중치 비율: $\Omega_\Lambda = 68.9\%$ (동결) + 곡률 기반 선택적 업데이트.

이론적 망각률:

$$\text{Forget}_{\text{CE}} \leq \varepsilon^2 = \boxed{4.87\%}$$

유도: 각 수면 순환에서 업데이트되는 가중치 비율이 $\varepsilon^2$이므로, 기존 지식에 영향을 미치는 비율도 최대 $\varepsilon^2$다.

### 6.3 수렴 속도

부트스트랩 수축률 $|B'(p^*)| = 0.155$:

$$\|p_n - p^*\| \leq 0.155^n \cdot \|p_0 - p^*\|$$

| 순환 수 $n$ | $0.155^n$ | 잔차 비율 | 정밀도 |
|---|---|---|---|
| 1 | 0.155 | 15.5% | 낮음 |
| 2 | 0.024 | 2.4% | 중간 |
| 3 | 0.0037 | 0.37% | 높음 |
| 5 | $8.7 \times 10^{-5}$ | 0.009% | 매우 높음 |

$\boxed{3\text{회}}$ 순환이면 $0.4\%$ 이내 수렴.

---

## 7. 복합 효과: 모든 원리 동시 적용

### 7.1 총합 정리

$d = 768$, $L = 12$, $T = 2048$ 기준:

| 지표 | 표준 Transformer | CE-Transformer | 이득 | CE 원리 |
|---|---|---|---|---|
| **파라미터** | 85M | 65.6M | $\boxed{-22.7\%}$ | P1 (격자) |
| **학습 메모리** | 1.41 GB | 0.68 GB | $\boxed{-51.8\%}$ | P3 (STDP) |
| **추론 FLOPs** | 348G | 17G | $\boxed{20.5\times}$ | P4 (희소) |
| **통신 (분산)** | 2.36 MB/층 | 4 B/층 | $\boxed{\sim 600{,}000\times}$ | P3 (STDP) |
| **환각률** | 50-70% | $\leq 4.87\%$ | $\boxed{\geq 10\times}$ 감소 | P5 (곡률) |
| **파괴적 망각** | 20-80% | $\leq 4.87\%$ | $\boxed{\geq 4\times}$ 감소 | P2 (수면) |
| **수렴 속도** | -- | 3회 순환 | 고정 | P2 (수면) |

### 7.2 에너지 효율 총합

추론 시:

$$\text{에너지 비율} = \underbrace{0.627}_{\text{P1: 격자}} \times \underbrace{0.0487}_{\text{P4: 희소}} = 0.0305 = 3.05\%$$

$$\text{에너지 절감} = 1 - 0.0305 = \boxed{96.95\%} \approx 33\times$$

학습 시:

$$\text{에너지 비율} = \underbrace{0.627}_{\text{P1}} \times \underbrace{0.311}_{\text{P4: 활성+구조}} \times \underbrace{0.5}_{\text{P3: STDP}} = 0.0975 = 9.75\%$$

$$\text{에너지 절감} = 1 - 0.0975 = \boxed{90.25\%} \approx 10\times$$

### 7.3 규모별 예측

| 규모 | 표준 파라미터 | CE 파라미터 | 표준 추론 비용 | CE 추론 비용 |
|---|---|---|---|---|
| 85M | 85M | 65.6M | 348 GFLOPS | 17 GFLOPS |
| 1.3B | 1.3B | 1.00B | 5.3 TFLOPS | 259 GFLOPS |
| 7B | 7B | 5.4B | 29 TFLOPS | 1.4 TFLOPS |
| 70B | 70B | 54B | 287 TFLOPS | 14 TFLOPS |
| 175B | 175B | 135B | 717 TFLOPS | 35 TFLOPS |

$\boxed{70\text{B CE-Transformer} \approx 7\text{B 표준 Transformer의 추론 비용}}$

---

## 8. 정확도-효율 트레이드오프

### 8.1 희소성과 정확도의 관계

Top-k 비율 $\rho$에 따른 이론적 정확도 감소:

$$\text{acc}(\rho) = \text{acc}(1.0) \cdot \left(1 - C \cdot (1-\rho)^{D_{\text{eff}}}\right)$$

$C$는 과제 의존 상수. $D_{\text{eff}} = 3.178$.

$\rho = \varepsilon^2 = 0.0487$에서:

$$\text{acc loss} = C \cdot (1 - 0.0487)^{3.178} = C \cdot 0.9513^{3.178} = C \cdot 0.854$$

$C \sim 0.05$ (경험적 추정)이면 정확도 손실 $\sim 4.3\%$.

$$\boxed{\text{정확도 95.7\%를 유지하면서 속도 20.5배 향상}}$$

### 8.2 최적점의 수학적 근거

$\varepsilon^2$가 최적인 이유: 부트스트랩 방정식 $\varepsilon^2 = \exp(-(1-\varepsilon^2)D_{\text{eff}})$의 고정점이므로, 이 비율에서 **정보 보존과 연산 효율이 자기일관적으로 균형**을 이룬다.

임의의 희소율을 사용하면 자기일관성이 깨진다:
- $\rho > \varepsilon^2$: 과잉 활성 = 에너지 낭비
- $\rho < \varepsilon^2$: 과소 활성 = 정보 손실

$\varepsilon^2$에서만 "활성 비율을 알아야 활성 비율을 계산할 수 있고, 계산 결과가 다시 동일한 활성 비율을 준다"는 자기일관성이 성립한다.

---

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
| P1 격자 | $-20.6\%$ | $+21.8\%$ | 동등 | $\sum f_i^2 = 0.596$ |
| P2 수면 | -- | -- | 망각 $\leq 4.87\%$ | $\|B'\| = 0.155$ |
| P3 STDP | $-51.8\%$ | -- | 동등 (검증 필요) | $O(N) \to O(1)$ 통신 |
| P4 희소 | -- | $20.5\times$ | $-4.3\%$ (추정) | $\varepsilon^2 = 0.0487$ |
| P5 곡률 | -- | $-3.1\%$ (오버헤드) | 환각 $\leq 4.87\%$ | $\sigma_1 \leq 1$ |
| **복합** | $\boxed{-52\%}$ | $\boxed{20\times}$ | $\boxed{+45\%p}$ (환각) | |
