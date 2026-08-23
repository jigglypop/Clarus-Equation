# U6 원시 스펙트럼·절대척도 대체 경로

Status: COMPLETE

기준일: 2026-08-15  
독립 계산: `verify_primordial_entropy_routes.py`

## 1. 목표와 공통 경계

목표 주장은 두 개다. 첫째, 하나의 perturbation action에서
$A_s,n_s,r$과 scale dependence를 함께 계산한다. 둘째, 같은 관측
$H_0$를 다시 단위변환한 값이 아니라 독립 동역학에서 late-time vacuum
scale을 얻는다. 기존 projector와 phase-area gate의 반례는 이 목표들을
삭제하지 않고, 각각 “projector 선택”과 “항등식 중복 계수”라는 route만
제외한다.

무차원 CE core와 $M_{\rm Pl}$만으로 dimensionful scale을 쓰려면

$$
M=M_{\rm Pl}F(D,\delta,\ldots)
$$

형태가 필요하다. 차원해석은 함수 $F$를 고르지 않으므로, scale을
지수적으로 작게 만드는 RG 흐름·Euclidean transfer·flux quantization 또는
stochastic volume law 중 하나가 실제 dynamics로 추가되어야 한다.

## 2. P1 — $R+R^2$ action과 Mukhanov--Sasaki 경로

가장 먼저 projector family를 하나의 inflation action으로 교체한다.

$$
S=\frac{M_{\rm Pl}^2}{2}\int d^4x\sqrt{-g}
\left(R+\frac{R^2}{6M^2}\right).
$$

Einstein frame에서는

$$
V(\phi)=V_0\left(1-e^{-\sqrt{2/3}\phi/M_{\rm Pl}}\right)^2,
\qquad
V_0=\frac34M^2M_{\rm Pl}^2.
$$

$y=e^{\sqrt{2/3}\phi/M_{\rm Pl}}$라 두면 potential slow-roll에서

$$
\epsilon_V=\frac{4}{3(y-1)^2},
\qquad
\eta_V=\frac{4(2-y)}{3(y-1)^2},
$$

$$
N_*=\frac34\left[y_*-y_{\rm end}
-\log\left(\frac{y_*}{y_{\rm end}}\right)\right],
\qquad
y_{\rm end}=1+\frac{2}{\sqrt3}.
$$

따라서 $n_s=1-6\epsilon_V+2\eta_V$, $r=16\epsilon_V$이고

$$
A_s=\frac{V_*}{24\pi^2M_{\rm Pl}^4\epsilon_V}
$$

가 amplitude normalization을 정한다. 독립 계산은 다음과 같다.

| $N_*$ | $n_s$ | $r$ | $V_0/M_{\rm Pl}^4$ | $M/M_{\rm Pl}$ |
|---:|---:|---:|---:|---:|
| 50 | 0.96156963 | 0.00419231 | $1.3395\times10^{-10}$ | $1.3364\times10^{-5}$ |
| 55 | 0.96497722 | 0.00349830 | $1.1151\times10^{-10}$ | $1.2193\times10^{-5}$ |
| 60 | 0.96782714 | 0.00296389 | $9.4285\times10^{-11}$ | $1.1212\times10^{-5}$ |

여기서는 $A_s=2.099\times10^{-9}$를 normalization 입력으로 썼다. 따라서
$A_s$는 교차예측이 아니며, 살아 있는 교차량은 reheating이 정한 $N_*$의
$n_s,r$, running과 non-Gaussianity다.

- 새 모델 선택: $R+R^2$ action 하나.
- 연속 자유도: $M$과 reheating history. $A_s$로 $M$을 고정하면 하나가
  소모된다.
- target-aware: 기존 CE 문서에 이 potential이 있으나 현재 amplitude와
  비교한 뒤 분석됐으므로 관측 검증은 exploratory다.
- kill test: 허용 reheating 전체에서 joint $n_s,r$ likelihood 실패,
  $M$을 만드는 후속 mechanism과 EFT control의 불일치.
- missing lemma: $M/M_{\rm Pl}\simeq(1.1$--$1.3)\times10^{-5}$를 관측
  amplitude 없이 생성하는 CE scale dynamics.

## 3. P2 — phase-area law의 변분적 존재구성

기존 law를 바로 제1원리라고 부르지 않고, 어떤 dynamics가 그 law를
정지해로 만들 수 있는지 먼저 구성한다. dimensionless recursion coordinate
$N\in[0,N_e]$와 $s(N)=\log S(N)$를 두고

$$
I_s=\int_0^{N_e}dN\,\frac K2
\left(\frac{ds}{dN}-\kappa\right)^2,
\qquad K>0
$$

를 택한다. $s(0)=s_0$와 natural endpoint condition을 주면 최소해는

$$
s(N_e)=s_0+\kappa N_e.
$$

따라서

$$
\kappa=\frac{\pi^2}{2},
\qquad
s_0=-\pi\delta(1-q_{\rm ext})
$$

를 **선택하면** 기존 phase-area 식을 정확히 재현한다. 이는 positive
quadratic action을 가진 변분적 존재구성이지만 $K,\kappa,s_0$의 미시적
기원을 유도하지 않는다. $\pi^2/2$는 unit four-ball volume과 같지만,
그 기하량이 $ds/dN$의 preferred slope가 된다는 사상은 추가 전제다.

de Sitter identity

$$
S_{\rm dS}=\pi\left(\frac{M_P}{H}\right)^2
$$

를 쓰면

$$
H=M_P\sqrt\pi\,e^{-s/2}.
$$

외부 $\alpha_s=0.11789$와 기존 integer choices를 넣은 독립 계산은

| phase law | $\log S$ | $H$ readout (km/s/Mpc) |
|---|---:|---:|
| leading term only | 282.268966692 | 51.56037193 |
| boundary correction 포함 | 281.737688630 | 67.24834592 |

을 준다. 이 큰 이동은 correction이 작은 장식이 아니라 목표 수치의 핵심
구조임을 뜻한다. 또한

$$
\frac{d\log H}{d\alpha_s}=-60.3126,
\qquad
N_e\to N_e+1:\quad H\to0.084805H
$$

라서 scale/scheme의 $\alpha_s$와 integer counting을 정확히 고정해야 한다.

- 새 모델 선택: 위 $s$-flow action 하나.
- 연속 자유도: 완전히 고정하면 0이지만 $K,\kappa,s_0$의 세 구조 선택이
  존재한다. $K$는 정지값에는 없고 fluctuation에 나타난다.
- target-aware: 예. 현재 $H_0$를 본 뒤 law와 correction을 검토했다.
- 독립 교차량: frozen law가 주는 $H_0$ 하나와 $\alpha_s$ 변화 민감도.
  $\rho_\Lambda^{1/4}$는 같은 $H_0$의 거듭제곱 재표현이므로 별도 계수 금지.
- kill test: 미시 transfer operator가 $\kappa,s_0$를 산출하지 못함,
  independently fixed input convention에서 $H_0$가 동결 오차범위 밖,
  integer/correction을 결과 뒤 변경.
- missing lemma: covariant horizon microstate measure에서 $I_s$와 두
  boundary coefficient를 유도하는 spectral/Euclidean 계산.

## 4. P3 — RG dimensional transmutation

dimensionless coupling $g$의 asymptotically-free one-loop 흐름

$$
\mu\frac{dg}{d\mu}=-\frac{b}{16\pi^2}g^3
$$

은

$$
\Lambda_{\rm RG}=\mu
\exp\left[-\frac{8\pi^2}{b g^2(\mu)}\right]
$$

을 생성한다. 이 route는 작은 dimensionful scale을 action/RG에서 만드는
구조적으로 정당한 방법이다. $\Lambda_{\rm RG}$를 scalaron mass $M$ 또는
vacuum relaxation scale에 연결하려면 field content가 $b$를 정하고 threshold
matching과 vacuum energy operator를 함께 계산해야 한다.

- 새 모델 선택: 특정 gauge/matter content 하나.
- 연속 자유도: UV coupling과 threshold masses; $b$는 field content의
  discrete 선택이다.
- target-aware: 필요한 $b$를 목표 scale에서 역산하면 예.
- 교차량: 같은 field content의 beta function, thresholds, relics와
  laboratory coupling running.
- kill test: 독립적으로 정한 field content가 필요한 exponent를 못 만들거나
  generated condensate가 올바른 sign/EOS/stress를 주지 못함.
- missing lemma: CE의 $D,\delta$가 어떤 renormalizable field content와
  beta coefficient를 고정하는지에 대한 표현론적 사상.

## 5. P4 — four-form flux와 sequestering

covariant four-form 후보는

$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R-\Lambda_{\rm bare}
-\frac{1}{2\cdot4!}F_{\mu\nu\rho\sigma}F^{\mu\nu\rho\sigma}
\right]
$$

이고, EOM은 $F_4=c\,\epsilon_4$를 준다. 따라서

$$
\rho_\Lambda=\Lambda_{\rm bare}+\frac12c^2.
$$

membrane이 있으면 $c=ne$처럼 flux가 양자화될 수 있다. 이 route는
vacuum stress를 action과 conserved flux로 명확히 만들고 radiative
correction을 sequestering 구조로 분리할 수 있다는 장점이 있다. 그러나
$e,n,\Lambda_{\rm bare}$의 cancellation을 CE 고정점이 정하지 않으면
절대값 예측은 아니다.

- 새 모델 선택: four-form/membrane sector 하나.
- 자유도: bare vacuum, flux quantum, integer branch와 membrane history.
- target-aware: vacuum에 가까운 branch만 고르면 예.
- 교차량: membrane transition spectrum, residual vacuum distribution,
  gravitational response와 radiative-stability law.
- kill test: flux spacing이 관측 허용 폭보다 크거나, 필요한 branch가
  metastability/empty-universe 조건을 위반.
- missing lemma: $q_{\rm ext}$ 또는 CE topology가 flux quantum과 branch를
  관측값 없이 선택하는 법칙.

## 6. P5 — causal-volume stochastic vacuum

spacetime four-volume $V_4$와 cosmological term을 conjugate 변수로 두고
cell count가 Poisson fluctuation을 가진다면

$$
\Delta V_4\sim\sqrt{V_4},
\qquad
\Delta\Lambda\sim\frac1{\sqrt{V_4}}\sim H^2
$$

형태의 late-time order relation을 얻을 수 있다. 이는 현재 epoch에서
$\rho_\Lambda\sim M_{\rm Pl}^2H^2$의 크기를 동적으로 추적할 가능성을 주지만
정확한 계수·부호·상관시간은 stochastic dynamics에 남는다.

- 새 모델 선택: causal-cell measure와 conjugacy law 하나.
- 자유도: cell volume, fluctuation coefficient, sign process와 initial state.
- target-aware: order relation은 아니지만 coefficient fit은 예.
- 교차량: redshift-dependent $w(z)$, sign/change statistics, large-angle metric
  correlations.
- kill test: expansion history·CMB에서 stochastic fluctuations가 허용범위를
  넘거나 안정한 양의 late-time branch가 생기지 않음.
- missing lemma: CE branching process를 diffeomorphism-invariant spacetime
  cell measure로 승격하는 구성.

## 7. 경로 결합과 우선순위

| 경로 | 직접 닫는 것 | 아직 필요한 핵심 | 독립 kill test 강도 |
|---|---|---|---|
| P1 $R+R^2$ | $n_s,r$ 공동 계산 | $M$ scale와 reheating | 높음 |
| P2 phase-flow action | 기존 entropy law의 변분 존재 | $\kappa,s_0$ 미시 유도 | 중간 |
| P3 RG transmutation | dimensionful scale 생성 원리 | field content/threshold | 높음 |
| P4 four-form | covariant vacuum stress/flux | branch·quantum 선택 | 중간 |
| P5 stochastic volume | coincidence/order tracking | 계수·부호·noise bound | 높음 |

가장 생산적인 결합은 P1의 scalaron mass를 P3의 independently specified RG
sector에서 생성하는 경로다. late-time vacuum은 같은 mass를 재사용하지
말고 P2, P4 또는 P5 가운데 하나를 별도 교차량과 함께 시험해야 한다.
P2를 유지할 경우 $H_0$와 $\rho_\Lambda$를 두 성공으로 세지 않고, 하나의
entropy-scale relation으로만 계산한다.

현재 어느 route도 zero-input 절대척도를 닫지 못했지만, 목표는 유지된다.
다음 공식 math lane은 P1+P3의 field-content lemma와 P2의 transfer-operator
lemma를 우선 검산하고, 실패하면 P4/P5의 branch-selection lemma로 넘어가야
한다.
