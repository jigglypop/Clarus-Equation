# Phase A 식 (1)–(4) 무차원성 독립 감사

Status: COMPLETE

대상: `00-contract.md`의 식 (1)–(4)와 명시된 향후 `tanh` 후보

판정: **PASS — 단, 정규화된 합성 좌표라는 계약 안에서만 통과한다.** 식 (1)–(4)의 합·곱·Gaussian NLL·차이는 모두 차원 벡터 $(0,0,0,0)$로 닫힌다. 전역 `DimensionlessChecker` 등록은 이번 격리 benchmark 단계에서는 필요하지 않으며, manifest의 단위 계약과 focused test로 만드는 **격리 certificate가 적합하다**. 정본 또는 공용 CE 코어로 승격할 때만 전역 registry 등록을 다시 심사한다.

## 1. 기준과 정규화 지도

기본 차원 벡터는 $d=(M,L,T,\Theta)$이고 $\mathbf 0=(0,0,0,0)$이다. Phase A V1은 물리량을 직접 계산하는 모델이 아니라 정규화된 이산시간 합성계다. 물리 좌표를 나중에 연결할 경우에는 단순히 이름만 무차원으로 선언하지 말고 다음 좌표변환을 고정해야 한다.

$$
\widetilde x=S_x^{-1}x_{\rm phys},\qquad
\widetilde u=S_u^{-1}u_{\rm phys},\qquad
\widetilde y=S_y^{-1}y_{\rm phys}.
$$

$S_x,S_u,S_y$는 각 성분의 양의 기준 scale을 대각에 가진다. 이에 따라

$$
\widetilde A=S_x^{-1}A_{\rm phys}S_x,\qquad
\widetilde B=S_x^{-1}B_{\rm phys}S_u,\qquad
\widetilde C=S_y^{-1}C_{\rm phys}S_x
$$

를 사용한다. 연속시간 계수에서 출발한다면 $A_d,B_d$를 만들 때 물리적 $\Delta t$까지 이산화에 포함해야 한다. V1에서 timestep은 정수 index이므로 그 자체는 무차원이다.

## 2. 원자량 감사

| 원자량·인자 | 차원 벡터 | 정규화 또는 조건 | 판정 |
|---|---:|---|---|
| $x_t,x_{t+1}$ | $\mathbf 0$ | $S_x^{-1}x_{\rm phys}$ | PASS |
| $u_t$ | $\mathbf 0$ | $S_u^{-1}u_{\rm phys}$ | PASS |
| $y_t$ | $\mathbf 0$ | $S_y^{-1}y_{\rm phys}$ | PASS |
| $z_t$ | $\mathbf 0$ | 유한 집합의 label/index | PASS |
| $A_z,\widehat A_z,\widehat A$ | $\mathbf 0$ | 정규화 좌표의 discrete-step operator | PASS |
| $B,\widehat B$ | $\mathbf 0$ | $S_x^{-1}B_{\rm phys}S_u$ | PASS |
| $C$ | $\mathbf 0$ | $S_y^{-1}C_{\rm phys}S_x$; identity/mask/mix도 수치 좌표에서는 무차원 | PASS |
| $\epsilon_t$ | $\mathbf 0$ | $S_x^{-1}\epsilon_{\rm phys}$ | PASS |
| $\nu_t$ | $\mathbf 0$ | $S_y^{-1}\nu_{\rm phys}$ | PASS |
| $\sigma$ | $\mathbf 0$ | 합성 V1의 정규화 state noise scale, $\sigma>0$ | PASS |
| $\sigma^2I$ | $\mathbf 0$ | 정규화 좌표의 covariance | PASS |
| residual $r=y-\widehat y$ 또는 $x_{t+1}-\widehat x_{t+1}$ | $\mathbf 0$ | 같은 정규화 chart에서 뺄셈 | PASS |
| standardized residual $r/\sigma$ | $\mathbf 0$ | 동일 state scale의 양의 $\sigma$ | PASS |
| $\lVert r\rVert^2/\sigma^2$ | $\mathbf 0$ | norm과 variance가 같은 정규화 좌표를 사용 | PASS |
| $2\pi\sigma^2$ 및 $\log(2\pi\sigma^2)$ | $\mathbf 0$ | log 인자는 양의 순수수 | PASS |
| Gaussian NLL | $\mathbf 0$ | dimensionless density 좌표에서 계산 | PASS |
| $\Delta_s$ | $\mathbf 0$ | 같은 NLL convention의 차이 | PASS |
| 향후 $b_z$ | $\mathbf 0$ | state-reference scale로 정규화 | PASS 조건부 |
| 향후 $A_zx+Bu+b_z$ | $\mathbf 0$ | 세 항이 모두 $\mathbf 0$ | PASS 조건부 |
| 향후 $\tanh(A_zx+Bu+b_z)$ | $\mathbf 0$ | `tanh` 전체 인자가 $\mathbf 0$ | PASS 조건부 |

## 3. 식별 감사

### 식 (1)

$$
d(A_{z_t}x_t)=\mathbf0+\mathbf0=\mathbf0,\qquad
d(Bu_t)=\mathbf0+\mathbf0=\mathbf0.
$$

따라서 $x_{t+1}$, $A_{z_t}x_t$, $Bu_t$, $\epsilon_t$의 차원이 모두 $\mathbf0$여서 덧셈이 닫힌다. Gaussian covariance도 $d(\sigma^2I)=\mathbf0$이다.

### 식 (2)

$d(Cx_t)=\mathbf0$이고 $d(\nu_t)=\mathbf0$이므로 $y_t=Cx_t+\nu_t$가 닫힌다. unknown mix 여부는 식별 가능성을 바꾸지만 차원 판정은 바꾸지 않는다.

### 식 (3)

$d(\widehat A_{z_t}x_t)=d(\widehat Bu_t)=\mathbf0$이므로 예측 state도 $\mathbf0$이다. pooled와 factorized model의 parameter 수 차이는 통계적 공정성 문제이지 차원 문제가 아니다.

### 식 (4)와 Gaussian NLL

좌표당 NLL을

$$
\frac12\left[\log(2\pi\sigma^2)+\frac{r_i^2}{\sigma^2}\right]
$$

로 쓰면 log 인자와 표준화 잔차가 모두 $\mathbf0$이다. 유한 합과 graph-seed별 차이 $\Delta_s=\mathrm{NLL}_{\rm pooled,s}-\mathrm{NLL}_{\rm factorized,s}$도 $\mathbf0$이다.

연속 물리변수의 raw density에 직접 로그를 취하면 density의 기준측도와 단위 문제가 생긴다. V1은 정규화된 좌표의 density를 사용하므로 통과한다. 미래 물리 데이터에서는 $S_x$ 또는 whitening/Jacobian convention을 고정하고 두 모델에 동일하게 적용해야 NLL 비교가 의미 있다.

## 4. 남은 조건과 fail-closed 항목

1. 계약 표의 “$\sigma$를 state reference scale로 나눔”은 scalar·등방 합성 V1에는 충분하다. 서로 다른 물리 단위나 scale을 가진 state에는 일반적으로 $\widetilde\Sigma=S_x^{-1}\Sigma_{\rm phys}S_x^{-T}$가 필요하므로 scalar $\sigma^2I$를 자동 가정하면 안 된다.
2. 향후 nonlinear generator는 $b_z$를 manifest의 dimensionless 항목으로 추가하고, $A_zx+Bu+b_z$ 전체를 kernel 진입 전에 검사해야 한다.
3. NLL의 $\sigma$를 truth로 둘지 train residual로 추정할지는 차원과 무관하지만 비교 protocol에 고정해야 한다. 어느 쪽이든 $\sigma>0$, finite, 동일 좌표·동일 convention이어야 한다.
4. 단위 metadata가 없거나 정규화 scale이 0/음수/non-finite이면 certificate를 `false`로 닫아야 한다.

## 5. checker 적용 범위 판정

기존 `dimensionless.py`는 `Quantity`와 정확한 rational 차원 벡터를 다루는 저수준 도구이고, `tests/test_dimensionless.py`의 15개 회귀가 통과했다. 반면 `DimensionlessChecker`의 전역 formula registry는 현재 문자열 공식과 수동 휴리스틱 중심이며 배열별 scale, covariance, matrix mapping, Gaussian 기준측도를 자동 검증하지 않는다.

따라서 이번 단계의 판정은 다음과 같다.

- **전역 등록: 보류.** Phase A는 정본·공용 코어가 아니라 격리 benchmark이고, 전역 문자열 등록은 실제 array-level 단위 계약보다 강한 검증을 제공하지 않는다.
- **격리 certificate: 필수이며 충분.** manifest에서 모든 state/input/noise scale과 normalized flag를 검증하고, 식 (1)–(4)의 원자량을 `DIMENSIONLESS`로 batch audit하는 focused test가 적합하다.
- **승격 조건:** 식이 정본 또는 공용 API에 들어갈 때 전역 registry 항목과 회귀 test를 추가하되, 격리 runtime certificate를 대체하지 않는다.

코드 회귀 원문은 `artifacts/dimensionless-check.log`에 보존했다.

## 6. 경계

무차원성 통과는 오직 **차원 정합성**을 뜻한다. 이는 $A_z,B$의 식별성, 인과 support의 진실성, Gaussian noise 가정, NLL의 통계적 적절성, pooled 대비 성능 우위, 생물학적 타당성, 기억 또는 AGI 주장을 정당화하지 않는다.
