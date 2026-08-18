# Population-manifold diffusion loop

이 문서는 population manifold 위의 state diffusion이 관측 또는 잠재 readout을 예측하는지 점검한 좁은 loop다. 독자는 manifold embedding·time series split·baseline의 기본을 아는 독자를 전제로 하며, manifold 비유와 실제 데이터의 구조 해석을 분리한다.

최소 명제와 state shape·timebase를 정한 뒤 누수 대조, AML310 출력, 실패 해석과 다음 순위를 읽는다. dataset provenance·seed·split·metric·threshold·OOD 조건이 없는 확장은 계획이며, 현재 실험은 일반 뇌·AGI 동역학 증거가 아니다.


> 상태: `synthetic PASS / AML310 exploratory FAIL / AML32 untouched`

## 1. 검증한 최소 명제

최소 명제는 현재 manifold state와 diffusion update를 입력으로 다음 관측·잠재 증분을 출력하는 조건부 contract다. embedding shape·normalization·sampling interval이 정의역이며, 수식의 존재는 manifold 원인 해석을 보장하지 않는다.

관측 뉴런 활동을 train 구간에서만 표준화하고 PCA한 뒤, 지연 상태

$$
q_t=[z_t,z_{t-2},z_{t-4}]
$$

를 만든다. 과거 library 상태 $q_s$와 query $q_t$ 사이의 Gaussian
kernel

$$
K_{ts}=\exp\!\left(-\frac{\lVert q_t-q_s\rVert^2}{\epsilon_t}\right)
$$

으로 가까운 $k$개 analog의 미래 증분을 운반했다.

$$
\widehat X_{t+h}
=X_t+
\frac{\sum_{s\in N_k(t)}K_{ts}(X_{s+h}-X_s)}
     {\sum_{s\in N_k(t)}K_{ts}}.
$$

마지막 보강에서는 출력도 관측 뉴런별 활동이 아니라 train-only PCA
잠재상태로 제한했다.

$$
\widehat z_{t+h}
=z_t+
\frac{\sum_{s\in N_k(t)}K_{ts}(z_{s+h}-z_s)}
     {\sum_{s\in N_k(t)}K_{ts}}.
$$

이는 anatomical graph, biological monad, CloudCell을 증명하는 식이 아니다.
검증 대상은 오직 **비선형 analog transport가 더 단순한 예측기를
일관되게 능가하는가**이다.

## 2. 누수 차단과 대조군

누수 차단은 time ordering·entity split·feature provenance를 고정하고, 대조군은 diffusion 항의 추가 기여를 분리한다. baseline·seed·metric·threshold가 등록되지 않거나 OOD에서 역전되면 결과는 기각 또는 미완성이다.

- 시간순 60/20/20 분할과 경계 embargo를 사용했다.
- 결측치 대치, 표준화, PCA, kernel library는 test를 보지 않는다.
- $k\in\{4,8,16,32,64,128,256\}$와 ridge는 validation에서만 골랐다.
- 대조군은 persistence와 multivariate linear latent transition이다.
- 19개 circular future shift는 상태의 자기상관과 query geometry를
  보존하는 null이다.
- 독립 기록 하나를 복제 단위로 삼았다. 뉴런 수를 표본 수로 세지 않았다.
- 기록 통과 기준은
  $\Delta R^2_{\mathrm{diff-best}}>0.01$,
  출력 차원의 60% 이상에서 linear보다 우세,
  shift $p\le0.05$이다.
- 패널 기준은 AML310 네 기록 중 최소 3개 통과이다.

합성 nonlinear manifold는 통과했고 white noise는 통과하지 않았다.
test block을 변조해도 PCA/hash 및 validation 선택이 바뀌지 않는 누수
회귀시험도 통과했다.

## 3. AML310 결과

AML310 결과는 지정 panel·split·seed에서 계산한 관측·잠재 output metric이다. 두 output을 같은 분모로 섞지 않고 uncertainty·baseline·ablation과 비교하며, 구조 해석은 개입 증거 전에는 가설이다.

### 3.1 관측 활동 증분 출력

관측 증분은 time-indexed activity tensor를 입력으로 한 다음 관측량 예측이다. sampling window·label provenance·metric 분모가 바뀌면 수치가 달라지며 OOD horizon은 failure 조건이다.

| horizon | 기록별 $\Delta R^2_{\mathrm{diff-best}}$ | 통과 |
|---|---:|---:|
| $h=1$ | -0.0038, -0.0027, -0.0028, -0.0046 | 0/4 |
| $h=6$ | -0.0091, -0.0482, -0.0169, +0.0180 | 1/4 |

$h=6$의 한 기록만 통과했으므로 3/4 패널 기준에는 미달했다.

### 3.2 잠재상태 증분 출력

잠재 증분은 embedding producer가 만든 normalized state를 consumer metric으로 변환한다. latent coordinate의 회전·scale ambiguity를 통제하지 않으면 관측 output과 같은 의미로 해석할 수 없다.

| horizon | 기록별 $\Delta R^2_{\mathrm{diff-best}}$ | 통과 |
|---|---:|---:|
| $h=1$ | -0.0089, -0.0107, -0.0293, -0.0214 | 0/4 |
| $h=6$ | -0.0381, -0.0542, -0.1202, -0.0041 | 0/4 |

잠재 출력에서는 모든 기록에서 diffusion이 strongest baseline보다
낮았다. $h=1$에는 여덟 잠재축 모두에서 linear transition보다
열세였다.

## 4. 왜 맞지 않았는가

실패 해석은 residual·baseline·ablation이 가리키는 가정 위반을 분리한다. 단일 실패는 모든 manifold 접근의 반례가 아니며, data quality·state definition·timebase를 새 fixture에서 다시 기각 가능하게 한다.

핵심 실패는 kernel 자체가 미래 정보를 전혀 못 찾은 것이 아니다.
일부 기록에서는 persistence보다 나았다. 그러나 같은 train-only 상태를
사용하는

$$
\widehat z_{t+h}=Az_t+b
$$

또는 persistence 중 더 강한 기준선을 일관되게 넘지 못했다. 현재 데이터의
짧은 시간척도에서는 부드러운 저차 선형 전이/자기상관으로 설명되는 몫이
크고, nearest-neighbor geometry가 추가하는 고유 비선형 이득은 재현되지
않았다. 기록마다 고른 $k$가 32에서 256까지 크게 달라진 점도 하나의
공통 diffusion scale이라는 해석과 맞지 않는다.

## 5. 판정과 다음 순위

판정은 현재 metric evidence와 구조 가설·후속 계획을 분리한다. 다음 순위는 dataset·split·threshold·rollback을 가진 실험 계약이며, 실행 전에는 supported나 완료가 아니다.

AML310에서 관측 출력은 0/4와 1/4, 잠재 출력은 0/4와 0/4였다.
그러므로 현재 diffusion 가설은 **탐색 패널에서 반증**한다.
실패한 사양을 확인 패널에 반복 적용해 유리한 결과만 찾지 않기 위해
`AML32_moving` 일곱 기록은 이 가설에 사용하지 않았다.

남은 가장 좁고 강한 후보는

$$
x_i(t+h)
=f_i\!\left(x_i(t),x_i(t-1),x_i(t-2)\right)+\epsilon
$$

의 local temporal-memory 명제이다. 다음 루프는 비선형 current-only
기준선과 재학습한 circular-history null을 추가한 뒤, 규칙을 고정하고
untouched AML32에서 확인한다. 이것이 통과해도 증명 범위는
**현재 측정을 조건으로 한 과거 신호의 held-out 예측정보**까지이며,
CloudCell, monad 또는 AGI 구조까지 확장되지 않는다.
