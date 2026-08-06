# CE-AI 정량적 성능 분석: 메모리, 속도, 정확도

> 관련: 2장(아키텍처), 4장(STDP), 5장(희소성), 6장(환각 억제), `7_AGI/12_Equation.md` 6절(파라미터 절감), `6_뇌/05_실험근거.md`
>
> 이 장은 CE 각 원리가 메모리, 속도, 정확도에 미치는 영향을 정량화한다. 다만 `6_뇌/05_실험근거.md` 기준으로, P1의 연산자 수준 절감식은 비교적 직접적이지만, P2/P4/P5의 태스크 수준 성능 수치는 아직 `bridge` 또는 `hypothesis`가 섞여 있다.

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

채널 분할은 `2_Architecture.md`와 같은 공학 benchmark
$w=(0.1180,0.03352,0.00775)$를 사용한다. 합은
$w_{\rm total}=0.15927$이다. 첫 성분만 Track-A 외부 $\alpha_s$ 입력이며,
나머지 성분과 neural channel 대응은 물리 결합의 canonical 예측이 아니다.

$$d_3 = \operatorname{round}\!\left(d\frac{0.1180}{0.15927}\right)
=\operatorname{round}(768\times0.740880)=569$$

$$d_2 = \operatorname{round}\!\left(d\frac{0.03352}{0.15927}\right)
=\operatorname{round}(768\times0.210460)=162$$

$$d_1 = d - d_3 - d_2 = 768 - 569 - 162 = 37$$

각 블록의 은닉 차원 ($\text{mult} = 4$): $h_i = 4d_i$.

블록 대각 파라미터:

$$P_{\text{diag}} = \sum_{i \in \{3,2,1\}} 2 \cdot d_i \cdot 4d_i = 8(d_3^2 + d_2^2 + d_1^2)$$

$$= 8(569^2 + 162^2 + 37^2) = 8(323{,}761 + 26{,}244 + 1{,}369) = 8 \times 351{,}374 = 2{,}810{,}992$$

혼합항 ($r_m = d/8 = 96$):

$$P_{\text{mix}} = 2 \times d \times r_m = 2 \times 768 \times 96 = 147{,}456$$

LBONorm ($r = d/8 = 96$):

$$P_{\text{LBO}} = r \times d + 2d + 1 = 96 \times 768 + 1537 = 75{,}265$$

$$P_{\text{kernel}}:=P_{\text{diag}}+P_{\text{mix}}=2{,}958{,}448,$$

$$\boxed{P_{\text{replacement}}:=P_{\text{kernel}}+P_{\text{LBO}}
=3{,}033{,}713}.$$

### 2.3 절감률

FFN operator만 비교하면

$$1-\frac{P_{\text{kernel}}}{P_{\text{FFN}}^{\text{std}}}
=37.30\%.$$

추가 LBONorm까지 replacement 비용에 포함한 순절감은

$$1-\frac{P_{\text{replacement}}}{P_{\text{FFN}}^{\text{std}}}
=\boxed{35.71\%}.$$

### 2.4 수학적 일반화

반올림 전 연속 channel fraction에서 FFN kernel 비는

$$\frac{P_{\text{kernel}}}{P_{\text{FFN}}^{\text{std}}}
\simeq\frac{8d^2\sum_i f_i^2+2dr_m}{8d^2}
=\sum_i f_i^2+\frac{r_m}{4d}.$$

채널 비율을 $f_i = d_i/d$로 놓으면:

$$\sum_i f_i^2
=0.740880^2+0.210460^2+0.048660^2
=0.595565$$

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}^{\text{std}}}
\approx0.595565+\frac{96}{3072}=0.626815$$

$$\boxed{\text{kernel 절감률}\approx1-0.626815=37.32\%
\quad(r_m=d/8)}.$$

LBONorm까지 포함하면 연속근사는

$$1-\left(0.626815+\frac{75{,}265}{4{,}718{,}592}\right)
=35.72\%,$$

정수 channel $(569,162,37)$을 그대로 쓴 정확값은 위 2.3절의 $35.71\%$다.

**핵심 공식**: 큰 $d$에서 kernel의 지배항은 $1-\sum_i f_i^2$다.
$\sum f_i^2$는 Herfindahl 지수(집중도)이므로, 나머지 rank와 norm 비용을
고정하면 channel이 불균등할수록 block-diagonal kernel의 파라미터 수가
작아진다. 이는 성능 보존을 뜻하지 않는다.

### 2.5 전체 모델 파라미터 절감

표준 block에서 FFN이 지배항의 $8d^2/12d^2=66.7\%$를 차지한다. 그러나
CE 쪽에는 LBONorm 추가비용이 있으므로 $37.3\%$를 단순 곱하지 않고 항별로
합산한다.

$$
P_{\rm CE,block}=4d^2+P_{\rm kernel}+P_{\rm LBO}=5{,}393{,}009,
$$

$$
1-\frac{P_{\rm CE,block}}{12d^2+4d}=\boxed{23.84\%}.
$$

$d = 768$, $L = 12$:

| | 표준 Transformer | CE-Transformer | 절감 |
|---|---|---|---|
| FFN/Gauge kernel | 56.6M | 35.5M | 37.3% |
| Attention | 28.3M | 28.3M | 0% |
| Norm/기타 | 0.04M | 0.90M | LBONorm 추가 |
| **합계** | **84.97M** | **64.72M** | **23.84%** |

### 2.6 FLOPs 절감

블록 대각 FFN kernel의 dense matmul FLOPs만 보면 같은 $37.3\%$ 산술이
적용된다. 그러나 LBONorm, gather/scatter, sparse-kernel overhead가 있으므로
전체 FLOPs가 같은 비율로 줄어든다고 단정하지 않는다.

$$\text{FFN kernel의 이상적 matmul 절감}=37.3\%.$$

$0.667\times37.3\%=24.9\%$는 추가 연산과 memory traffic을 무시한 전체
block의 낙관적 상한이다. 실제값은 동일 hardware에서 측정한다.

---

## 3. P4: 부트스트랩 희소성 -- 추론 속도

### 3.1 Dense 추론 FLOPs

전체 모델:

$$F_{\text{dense}} = L(24Td^2 + 4T^2d)$$

### 3.2 Sparse 추론 FLOPs (Top-k 활성)

Track-A manifest 기반 설계 활성비는 $x=0.0486382585$다. 현재 문서의
구현과 가장 가까운 가정은 **FFN만 희소화하고 attention의 quadratic 항은
유지**하는 경우다:

$$F_{\text{FFN}}^{\text{sparse}}=xF_{\text{FFN}}^{\text{dense}}
=0.0486382585\times16Td^2$$

Attention은 희소화하지 않는다고 가정:

$$F_{\text{sparse}}=8Td^2+4T^2d+0.0486382585\times16Td^2
=8.778212Td^2+4T^2d$$

$$\text{속도 향상 비율} = \frac{24Td^2 + 4T^2d}{8.7782121363Td^2 + 4T^2d}$$

$T = 2048$, $d = 768$이면:

$$
F_{\rm dense}=502.511173632\ {\rm G},\qquad
F_{\rm sparse}=281.863525046\ {\rm G},
$$

$$\boxed{\text{speedup}=1.7828173\times}$$

### 3.3 FFN + Attention 모두 희소화: 낙관적 상한

Attention의 선형항과 quadratic 항까지 모두 같은 비율로 희소화할 수 있다고 **가정**하면:

$$F_{\text{full-sparse}} = \varepsilon^2 (24Td^2 + 4T^2d)$$

$$\text{속도 향상} = \frac{1}{\varepsilon^2} = \boxed{20.5\times}$$

이 값은 **attention까지 구조적 희소 실행이 가능한 경우의 낙관적 상한**이다. 현재 문서의 CE-Transformer 구현을 그대로 읽으면, 긴 컨텍스트에서는 위 3.2의 `1.7828\times`가 더 보수적인 기준이다.

### 3.4 실제 속도 향상 (하드웨어 고려)

GPU의 희소 연산 효율은 100%가 아니다. 아래 식과 표는 측정값이 아니라
가정한 overhead 모델이다.

$$\text{실제 속도} = \frac{\text{이론적 속도}}{1 + \alpha_{\text{overhead}}}$$

| 가정 | $\alpha_{\text{overhead}}$ | 실제 속도 향상 |
|---|---|---|
| FFN 위주 희소화 (3.2) | 0.1-0.2 | $1.5\times - 1.7\times$ |
| 전면 희소화 상한 (3.3) | 0.3-0.5 | $13.7\times - 15.8\times$ |
| 전면 희소화 + 구조적 커널 최적화 | 0.0-0.2 | $17\times - 20.5\times$ |

실제 wall-clock 속도라고 부르려면 동일 hardware, batch, sequence length,
kernel과 warm-up을 고정한 benchmark가 필요하다. 현재 문서에는 그 측정이
없다.

---

## 4. P3: STDP 국소 학습 -- 메모리 절감

### 4.1 역전파 메모리

역전파는 모든 중간 활성값을 저장해야 한다.

$$M_{\text{BP}} = L \times T \times d \times \text{sizeof(float)} = L \cdot T \cdot d \cdot 4\text{B}$$

$L = 12$, $T = 2048$, $d = 768$:

$$M_{\text{BP}} = 12 \times 2048 \times 768 \times 4 = \boxed{75.5\text{MB}} \quad (\text{활성값만})$$

가중치 + 그래디언트 + optimizer state (Adam: $2\times$):

$$M_{\text{total}}^{\text{BP}} = P \times 4 \times (1 + 1 + 2) = 4P \times 4 = 16P \text{ bytes}$$

$P = 85\text{M}$: $M_{\text{total}}^{\text{BP}} = 85 \times 10^6 \times 16 = 1{,}360\text{MB} = \boxed{1.33\text{GB}}$.

활성값 포함 총합: $1.33 + 0.076 = \boxed{1.41\text{GB}}$.

### 4.2 STDP 메모리

아래 계산은 **layer-shared 또는 neuron-local trace 근사**를 둔 낙관적 경우다. `08_시냅스가소성.md`의 순수한 synapse-local eligibility trace $e_{ij}$를 그대로 쓰면 추가 상태는 일반적으로 $O(P)$다.

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

**일반 공식**

$$\frac{M_{\text{STDP}}}{M_{\text{BP}}} = \frac{8P}{16P + 4LTd} \approx \frac{8}{16} = \boxed{0.5} \quad (P \gg LTd)$$

대규모 모델에서 메모리 $\sim 2\times$ 절감이 수렴한다.

### 4.5 통신 비용 (분산 학습)

| 배치 | 역전파 | STDP + 도파민 |
|---|---|---|
| 공유 global model의 data parallel | $O(P)$ gradient/all-reduce | 업데이트된 $O(P)$ weights 또는 동등한 sufficient statistic 동기화 필요 |
| 완전 local model | $O(P)$ 동기화 | $\delta[t]$는 $O(1)$이지만 replica가 서로 다른 모델로 발산 |

따라서 4 B 통신 주장은 global weight를 공유하지 않는 경우에만 가능하다.
동일 모델을 유지하는 분산 학습의 통신 절감률은 압축·동기화 프로토콜을
명시하기 전에는 미정이다.

---

## 5. P5: 곡률 정규화 -- 정확도/환각률

### 5.1 유니타리 제약의 오류 전파 억제

전체 layer map을 $\mathcal T_l$이라 두면 일반 상계는

$$
\boxed{\|\delta_L\|\leq
\prod_{l=1}^L\operatorname{Lip}(\mathcal T_l)\,\|\delta_0\|}.
$$

순수 선형층만 있으면
$\operatorname{Lip}(\mathcal T_l)=\sigma_1(W_l)$다. 그러나
Transformer에는 activation, attention, normalization과 residual이 있다.
특히 $h\mapsto h+F_l(h)$이면

$$\operatorname{Lip}(\mathcal T_l)\leq1+\operatorname{Lip}(F_l),$$

이므로 $F_l$ 안의 한 weight에 $\sigma_1(W_l)\leq1$을 걸어도 전체 layer가
비팽창이라고 결론낼 수 없다. LayerNorm은 variance floor와 gain, attention은
softmax Jacobian과 모든 projection norm까지 포함해야 한다. 오직 전체
$\mathcal T_l$에 대해 $\operatorname{Lip}(\mathcal T_l)\leq1$을
검증했을 때만 $\|\delta_L\|\leq\|\delta_0\|$가 성립한다.

개별 weight의 spectral norm은 full-layer 비팽창의 충분조건이 아니다.
설령 full-layer 조건을 확인해도 다리 게이트 `F4`에 따라 환각률 hard
bound로 환산하지 않는다(§5.3 참조).

### 5.2 곡률 정규화의 일반화 오차 감소

곡률 정규화 항 $\lambda\|\Delta_g h\|^2$는 가설공간을 줄일 수 있지만,
그 감소량은 data geometry와 parameter-to-function map을 통해 증명해야 한다.

표준 네트워크의 Rademacher 복잡도:

$$\mathcal{R}_n \leq \frac{B_x \prod_l \|W_l\|_F}{\sqrt{n}}$$

기존
$\sqrt{1-\lambda\kappa_{\rm th}/\kappa_{\max}}$ 계수는 Rademacher 정리에서
나오지 않으며 radicand가 음수가 될 수도 있어 삭제한다. 현재 안전한
주장은 위 generic norm bound뿐이다. 곡률 제약으로 더 강한 bound를
주려면 covering number 또는 contraction lemma에 들어가는 명시적
함수클래스를 먼저 정의해야 한다.

### 5.3 태스크 수준 해석의 한계

연산자 수준에서 직접 보장되는 것은 다음뿐이다.

$$
\|\delta_L\|\leq\|\delta_0\|
\quad\text{if every full layer satisfies }
\operatorname{Lip}(\mathcal T_l)\leq1.
$$

개별 weight spectral normalization은 그 조건의 한 구성요소일 뿐이다.
전체 비팽창을 확인하더라도 그것만으로 곧바로
- 환각률 상한
- TruthfulQA 점수
- FactScore 향상폭

을 정리처럼 도출할 수는 없다.

### 5.4 벤치마크 가설

가장 보수적인 표현은 다음이다.

- 곡률 정규화와 비팽창 제약이 강할수록, 고곡률 토큰과 자기증폭 오류가 줄어들 가능성이 높다.
- 따라서 TruthfulQA, HaluEval, FactScore의 개선은 **검증 가능한 가설**이다.
- 우주 분율을 환각률 또는 truthfulness hard bound로 쓰는 것은 금지한다.

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

설계상 배경 가중치 비율: $\Omega_\Lambda=69.027\%$ + 곡률 기반 선택적 업데이트.

하지만 이것만으로 곧바로

$$\text{Forget}_{\text{CE}} \leq \varepsilon^2$$

를 정리처럼 말할 수는 없다. 실제 망각률은
- 어떤 가중치가 업데이트되는가
- 그 가중치가 과거 태스크에 얼마나 민감한가
- 수면 위상에서 어떤 재생(replay)이 일어나는가

에 따라 달라진다.

### 6.3 수렴 속도

정준 비선형 단체 사상 $B_p$의 고정점 선형화에는

$$q_\star=\rho\!\left(DB_p(p^\star)\right)=D_np_a^\star
=0.1545681540116411$$

이 성립한다. 따라서 고정점 근방에서는

$$\|p_n-p^\star\|_1=q_\star^n\|p_0-p^\star\|_1
+O(\|p_0-p^\star\|_1^2),$$

이며 이것은 전역 부등식이 아니다. $U=\{p\in\Delta^2:p_a\leq0.13\}$에서는
$q_U=0.2001757361$인 엄밀한 균일 상계가 따로 성립한다.

| 순환 수 $n$ | 국소 선형 주항 $q_\star^n$ | $U$에서의 균일 상계 $q_U^n$ |
|---|---:|---:|
| 1 | 0.1545681540 | 0.2001757361 |
| 2 | 0.0238913142 | 0.0400703253 |
| 3 | 0.0036928363 | 0.0080211069 |
| 5 | $8.82267\times10^{-5}$ | $3.21408\times10^{-4}$ |

이 값들은 $B_p$의 수학적 수렴률이다. 특정 아키텍처나 수면 회복에서 같은
수치가 측정된다는 주장은 아직 `bridge`다.

### 6.4 검증 가능한 과도 응답 예측

비교용 affine 실험을 다음과 같이 정의한다. 균등 초기화

$$p_0 = (1/3,\; 1/3,\; 1/3)$$

에서 시작하고

$$p_{n+1}=p^*+\rho_c(p_n-p^*),\qquad
p^*=(0.0486382585,0.2610881744,0.6902735671)$$

를 적용한다. 이 affine update는 runtime controller benchmark이며 canonical nonlinear
self-map $B_p$ 자체가 아니다. 따라서 $\rho_c$는 별도 제어 하이퍼파라미터다.
아래 표는 비교 편의를 위해서만 $\rho_c=q_\star$로 둔다.

이 경우 다음의 **결정론적 코드 기대값**이 나온다.

| 순환 수 $n$ | 활성 $x_a$ | 구조 $x_s$ | 배경 $x_b$ |
|---|---|---|---|
| 0 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | $9.2643\%$ | $27.2255\%$ | $63.5102\%$ |
| 2 | $5.5440\%$ | $26.2814\%$ | $68.1746\%$ |
| 3 | $4.9690\%$ | $26.1355\%$ | $68.8955\%$ |

이는 affine benchmark 구현의 단위 테스트이지, 실제 모델이나 생물학적 수면의
과도응답 예측은 아니다.

---

## 7. 복합 효과: 모든 원리 동시 적용

### 7.1 총합 정리

$d = 768$, $L = 12$, $T = 2048$ 기준:

| 지표 | 표준 Transformer | CE-Transformer | 이득 | CE 원리 |
|---|---|---|---|---|
| **파라미터** | 85M | 65.6M | $\boxed{-22.7\%}$ | P1 (격자) |
| **학습 메모리** | 1.41 GB | 0.68 GB | up to $\boxed{-51.8\%}$ | P3 (STDP, shared-trace 가정) |
| **추론 FLOPs** | 502.511G | 281.864G | $\boxed{1.7828\times}$ | P4 (FFN 희소, 현재형) |
| **추론 FLOPs 상한** | 502.511G | 24.441G | $\boxed{20.5600\times}$ | P4 (전면 희소, 낙관적 상한) |
| **통신 (분산)** | $O(P)$ | global model이면 $O(P)$ 동기화 필요 | 절감 미정 | P3 (STDP) |
| **환각률** | -- | -- | hard bound 미정 | P5 (곡률) |
| **파괴적 망각** | 20-80% | -- | 개선 가설 | P2 (수면) |
| **분배 사상 수렴** | -- | 국소 주항 2회 `2.389%`, 3회 `0.3693%`; $U$ 균일 상계는 각각 `4.007%`, `0.8021%` | bridge (런타임 매핑) | P2 ($B_p$ 수학량) |

### 7.2 에너지 효율 총합 (조건부 상한)

> 다리 게이트 `F1` (`12_Equation.md` 0.0절): 아래 수식은 활성 비율이
> Track-A target $x=0.0486382585$로 자기수렴한다는 가설 아래의 알고리즘적
> 상한이다. transformer + backprop에서는 기각됐으며 현재는 설계값이다.

추론 시 (자기수렴 가설 + attention 완전 희소화 가정):

$$\text{에너지 비율}=0.627\times0.0486382585
=0.0304962=3.04962\%$$

$$
\text{에너지 절감}=1-0.0304962
=\boxed{96.95038\%},\qquad
\text{비율}=32.791\times
\quad\text{(이중 가설 상한)}
$$

학습 시:

$$\text{에너지 비율}=0.627\times
\underbrace{(0.0486382585+0.2610881744)}_{0.3097264329}
\times0.5=0.0970992=9.70992\%$$

$$
\text{에너지 절감}=1-0.0970992
=\boxed{90.29008\%},\qquad
\text{비율}=10.299\times
\quad\text{(조건부)}
$$

### 7.3 규모별 예측

| 규모 | 표준 파라미터 | CE 파라미터 | 표준 추론 비용 | CE 추론 비용 |
|---|---|---|---|---|
| 85M | 85M | 65.6M | 502.511 GFLOPS | 24.441 GFLOPS (상한) |
| 1.3B | 1.3B | 1.00B | 비례 증가 | 비례 상한 |
| 7B | 7B | 5.4B | 비례 증가 | 비례 상한 |
| 70B | 70B | 54B | 비례 증가 | 비례 상한 |
| 175B | 175B | 135B | 비례 증가 | 비례 상한 |

정확한 대규모 비용 비교는 attention sparsity, KV cache, prefill/decode 분리 모델을 포함해 다시 계산해야 한다.

### 7.4 희소 활성 설계값

Track-A 희소율을 외부 설계 target으로 쓰면, 각 은닉 차원에서 활성 채널 수는

$$k^*(d)=\lceil0.0486382585\,d\rceil$$

로 정해진다.

| 은닉 차원 $d$ | 설계 활성 수 $k^*(d)$ | 활성 비율 |
|---|---|---|
| 768 | 38 | $4.9479\%$ |
| 2048 | 100 | $4.8828\%$ |
| 4096 | 200 | $4.8828\%$ |
| 8192 | 399 | $4.8706\%$ |

정수 올림 때문에 실현 비율은 정준 연속값 $4.8638258516\%$보다 약간
커진다. 이 표는 구현 전에 바로 체크 가능한 설계값이지 성능 예측은 아니다.

---

## 8. 정확도-효율 트레이드오프

### 8.1 희소성과 정확도의 관계

Top-k 비율 $\rho$에 따른 정확도 감소를 닫힌형으로 쓰고 싶다면, 아래 식은 **엄밀한 정리라기보다 heuristic response curve**로 읽어야 한다:

$$\text{acc}(\rho) = \text{acc}(1.0) \cdot \left(1 - C \cdot (1-\rho)^{D_{\text{eff}}}\right)$$

$C$는 과제 의존 상수. $D_{\text{eff}} = 3.178$.

$x=0.0486382585$에서:

$$\text{acc loss}=C(1-x)^{D_n}=C\times0.85346$$

$C \sim 0.05$ (경험적 추정)이면 정확도 손실 $\sim 4.3\%$.

따라서 이 절에서 안전하게 말할 수 있는 것은:

$$
\boxed{\text{희소율 }\rho\text{와 정확도 사이에 최적점이 있을 것이라는 예측은 가능하지만, 정확한 손실률은 아직 가설이다.}}
$$

### 8.2 최적점의 수학적 근거 (코어 측)

$\varepsilon^2$는 코어 부트스트랩 방정식 $\varepsilon^2 = \exp(-(1-\varepsilon^2)D_{\text{eff}})$의 고정점이며, **이 식이 신경 활성/구조/배경에 그대로 적용된다는 가정 하에서** 자기일관 균형점이 된다.

가정 하의 해석:
- $\rho > \varepsilon^2$: 과잉 활성 = 에너지 낭비
- $\rho < \varepsilon^2$: 과소 활성 = 정보 손실

> 다리 게이트 `F1`: 위 자기일관 해석은 코어의 우주론·입자물리 식이 신경 모듈에 옮겨갈 메커니즘적 유도가 닫힌 후에야 hard claim 으로 올릴 수 있다. 현 시점 transformer 기질에서는 이 자기일관이 측정되지 않았다(`5_Sparsity.md` 8.5절 falsified). 따라서 본 절의 "$\varepsilon^2$에서 자기일관" 문장은 **검증해야 할 예측**이지 정리가 아니다.

### 8.3 검증 가능한 최적점 예측

실험 설계 관점에서 CE가 요구하는 것은 다음 두 문장이다.

1. **최적점 위치 가설:** 효율-정확도 Pareto front의 knee point를 manifest target $k\approx4.864\%$ 근방에서 사전등록한다.
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
| P2 수면 | -- | -- | 지속 학습 개선 가설; 분배 사상과 수면 회복률의 동일시는 미검증 | $q_\star=0.1545681540$, $q_U=0.2001757361$ (런타임 매핑은 bridge) |
| P3 STDP | up to $-51.8\%$ | -- | 검증 필요 | shared-trace 가정 필요 |
| P4 희소 | -- | $1.7828\times$ FLOP 비, $20.56\times$ 전면희소 상한 | 정확도 trade-off 가설 | $x=0.0486382585$ |
| P5 곡률 | -- | overhead 미측정 | full-layer Lipschitz 검증 필요 | 개별 $\sigma_1$만으로 불충분 |
| **복합** | 조건부 절감 | 조건부 가속 | hard bound 미정 | |
