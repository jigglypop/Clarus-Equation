# 인과적 재귀 기하 복원 우회 연구 경로

Status: COMPLETE

## 1. 목표와 강제·자유 구조

목표량은 정확한 latent graph 자체가 아니라, 등록된 관측·개입 정의역에서 보지 않은 intervention rollout, quotient-level 장기예측, cue-to-target 접근 energy와 recall을 낮은 자유도로 교차 예측하는 구조다. 수학이 강제하는 부분은 SCC condensation 정리, 관측-only latent gauge, exact lumpability 식, 정적 metric 방향성 no-go다. 자유 부분은 관측 chart, 개입 interface, edge threshold와 window, quotient state, control horizon, context 수와 gate, scale별 effective-edge operator다.

$n$은 관측 state 차원, $m$은 control 차원, $k$는 quotient block 수, $d$는 metric 차원, $p=d(d+1)/2$는 SPD 성분 수라 쓴다. `target-aware`는 test endpoint나 목표값을 본 뒤 후보를 만든 경우를 뜻한다. 아래 후보는 현재 결과값을 보지 않고 구성했으므로 기본값은 `아니오`지만, grid에서 test 성능으로 선택하면 즉시 `예`로 바뀐다.

## 2. 후보 요약과 순위

| 순위 | Route | 주 대상 | 새 공리 최대 1개 | fitted dof / 선택 규모 | target-aware | 독립 교차 예측 | 죽이는 반증 시험 |
|---:|---|---|---|---|---|---|---|
| 1 | `R1` 완전관측 개입 LTI 식별 | `H1`, `N1`의 좁은 잔존 | 고정 chart의 LTI+known intervention protocol을 하나의 model-class 공리로 둠 | $n(n+m)$; noisy support penalty 후보 $L_\lambda$ | 아니오; test로 $\lambda$를 고르면 예 | unseen target·dose의 1-step 계수와 multi-step rollout을 같은 $A,B$로 예측 | design rank 부족, coefficient가 split마다 불안정, 최선 matched VAR와 독립 seed CI 차이 0 이하 |
| 2 | `R3` controllability-Gramian 기억 energy | `H3`, `H4` | recall cue를 등록된 control input과 energy로 해석 | frozen $A,B$이면 metric 추가 dof 0; horizon 후보 $L_T$ | $T$ preregistration이면 아니오 | cue별 $x^TW_T^{-1}x$가 basin-entry time·recall 순위를 예측 | $W_T$ rank 결손, nonlinear validity 붕괴, weight-only baseline 동률, geometry shuffle 무효 |
| 3 | `R2` exact-lumpable SCC quotient | `H2` | quotient state를 등록된 block aggregate $Qx$로 정의 | $\bar A,\bar B$ 직접 fit 시 $k(k+m)$; $A,B,Q$에서 유도 시 추가 0 | 아니오 | 서로 다른 microstate의 aggregate rollout과 module intervention propagation | $\inf_{\bar A}\lVert QA-\bar A Q\rVert$가 허용치 초과, 같은 압축률 최선 대조보다 Pareto 우위 없음 |
| 4 | `R5` scale-indexed effective-edge hierarchy | `H2` 다중스케일 확장 | 모든 scale에 공통인 하나의 $\Phi$가 새 edge semantics를 정의 | $q_\Phi$; 사후 window/threshold 탐색은 $\prod_\ell L_{W,\ell}L_{\tau,\ell}$ | schedule preregistration이면 아니오 | level별 reciprocal edge가 해당 horizon의 양방향 held-out intervention transfer를 예측 | 같은 semantics 반복으로 singleton화, scale perturbation 불안정, 새 edge가 독립 rollout에 재현되지 않음 |
| 5 | `R4` gauge-fixed context SPD mixture | `H5`, `H3` | context label/gate를 outcome과 독립적으로 관측하거나 고정 | $Kp+q$; 단일 metric 대비 $(K-1)p+q$, grid $L_KL_rL_g$ | rank·$K$를 test로 고르면 예 | unseen context switch의 NLL, path-cost ordering과 recall을 동시에 예측 | $g_j$가 같거나 단일 metric과 동률, encoder unfreeze 시 이득 소멸, basis/gate shuffle 무효 |

순위는 식별성 기반, 추가 자유도, 교차 예측 강도 순이다. `R1`은 나머지 경로의 $A,B$를 독립적으로 고정할 수 있어 먼저 온다. `R3`은 metric을 별도 latent story가 아니라 control energy로 정의해 추가 dof가 가장 낮다. `R2`는 강하지만 exact lumpability가 희귀할 수 있다. `R5`와 `R4`는 각각 scale 선택과 mixture 자유도가 커서 뒤에 둔다.

## 3. `R1`: 완전관측 interventional LTI anchor

### 경로

1. 합성계에서 observed coordinate, sampling interval, state/control 단위를 고정한다.
2. intervention target과 amplitude가 training design $Z=[X;U]$를 full row rank로 만들도록 사전 설계한다.
3. training split에서 식 (10)의 $A,B$와 uncertainty를 추정하고 freeze한다.
4. graph는 fixed chart의 nonzero $A_{ij}$ support 또는 training-only calibration threshold로 정의한다.
5. test에서는 새 target·dose·initial state의 rollout만 연다.

### 지위와 자유도

식별성은 `11-math.md` 5절의 조건부 정리 후보다. rollout 우위는 [미완성] 비교 가설이다. exact noiseless design이면 $n(n+m)$ coefficient 외의 선택은 없다. noisy support를 위해 $L_\lambda$개 penalty를 비교하면 validation-only nested selection을 쓰고 test에서 다시 고르지 않는다. hidden $z_t$를 추가할 경우 feature 수만큼 coefficient가 늘며, $z_t$가 관측되지 않으면 이 route는 `G/F 대 z` 분리 주장을 하지 않는다.

### 교차 예측과 kill test

같은 $A,B$가 one-step coefficient, multi-step rollout, impulse-response sign과 intervention propagation time을 함께 맞혀야 한다. 정확한 graph support가 틀려도 rollout만 맞으면 결과 명칭은 effective predictive dynamics로 제한한다. design 최소 singular value가 사전 하한 아래이거나, coefficient support가 독립 graph seed에서 재현되지 않거나, matched VAR/Neural ODE 대비 primary endpoint의 독립 seed interval이 0을 넘지 못하면 우위 경로를 죽인다.

## 4. `R2`: SCC가 아니라 lumpability를 시험하는 quotient

### 경로

1. `R1`에서 freeze한 $A$의 zero/nonzero semantics로 SCC partition과 $Q$를 test outcome을 보지 않고 만든다.
2. training에서 $\bar A,\bar B$를 fit하거나 $QA, QB$에서 유도한다.
3. 식 (1)의 algebraic residual과 같은 $Qx$를 가진 microstate pair의 next-state dispersion을 측정한다.
4. compression ratio가 같은 community, spectral, balanced random, learned soft partition을 동일 예산으로 비교한다.
5. module intervention의 sign, latency, aggregate magnitude를 blind test에서 예측한다.

### 지위와 자유도

식 (1)이 exact하면 quotient closure는 조건부 정리 후보다. SCC가 그 조건을 만족한다는 문장과 대조군보다의 우위는 [미완성]이다. $Q$와 $A,B$가 freeze되면 quotient dynamics의 추가 fitted dof는 0이다. 직접 fit하면 $k(k+m)$개다. 네 partition family 중 가장 좋은 것만 사후 보고하면 선택 자유도가 생기므로 SCC를 primary, 나머지를 명시적 대조로 모두 보고한다.

### 교차 예측과 kill test

한-step residual이 작다는 사실과 long-horizon aggregate rollout이 모두 맞아야 한다. 특히 같은 quotient state지만 서로 다른 microstate를 시작점으로 한 paired test가 필요하다. `11-math.md` 식 (12)--(13) 형태의 dispersion이 허용치보다 크거나, SCC label shuffle이 성능을 떨어뜨리지 않거나, 같은 compression ratio의 최선 대조군보다 Pareto 우위가 없으면 SCC를 필수 predictive unit으로 쓰는 경로를 죽이고 해석용 topology로만 남긴다.

## 5. `R3`: controllability-Gramian 접근 energy

### 경로

1. encoder, $A,B$, control norm과 horizon $T$를 recall outcome을 열기 전에 freeze한다.
2. $W_T$의 rank와 condition number를 계산하고 reachable target만 사전등록한다.
3. $g_T=W_T^{-1}$과 cue-target displacement의 energy를 계산한다.
4. 이 scalar 하나로 unseen cue의 basin-entry time, 성공률과 interference sensitivity를 예측한다.
5. activity-only, weight-only, Euclidean, parameter-matched nonlinear scalar baseline과 geometry shuffle을 비교한다.

### 지위와 자유도

$W_T\succ0$에서 최소 energy가 $x^TW_T^{-1}x$라는 명제는 `11-math.md` 8절의 조건부 정리 후보다. 그 energy가 생물학적 또는 AGI 기억 회상을 추가로 예측한다는 부분은 [미완성]이다. $A,B,T$가 frozen이면 $g_T$의 $p$개 성분은 derived quantity라 추가 fitted dof가 0이다. $L_T$개 horizon 중 성능이 좋은 것을 고르면 look-elsewhere가 $L_T$이므로 하나의 primary $T$를 고정하고 나머지는 sensitivity로 보고한다.

### 교차 예측과 kill test

energy 순위가 recall accuracy뿐 아니라 basin-entry time과 minimum cue magnitude를 같은 방향으로 예측해야 한다. $W_T$가 singular하면 global SPD 주장을 즉시 중단하고 reachable subspace로 좁힌다. $A,B$를 계산한 weight-only baseline이 같은 예측을 내면 metric은 편리한 reparameterization일 뿐 독립 기전이 아니다. encoder unfreeze 후에만 이득이 나타나거나 식 (15) gauge shuffle에서 해석이 바뀌면 geometry 귀속을 죽인다.

## 6. `R5`: 비자명 hierarchy를 위한 scale별 새 semantics

### 경로

1. 첫 SCC condensation 뒤 같은 edge를 다시 SCC화하지 않는다.
2. 각 level에서 coarse intervention response나 registered horizon $W_\ell$의 transfer operator로 effective edge를 새로 정의한다.
3. 모든 level에 같은 update 형식 $\Phi$를 쓰고 threshold/window schedule을 preregister한다.
4. 새 reciprocal edge가 나타날 때에만 상위 SCC를 구성한다.
5. 각 상위 edge가 해당 horizon의 held-out bidirectional influence를 예측하는지 검사한다.

### 지위와 자유도

같은 semantics의 재귀 SCC는 첫 condensation 뒤 singleton으로 끝난다는 P0 경계를 가진다. 식 (4)의 새 $\Phi$를 둔 경로는 [미완성] 모델 선택이다. scale마다 window와 threshold를 독립 탐색하면 선택 규모가 $\prod_\ell L_{W,\ell}L_{\tau,\ell}$로 폭증한다. 이를 피하려면 하나의 deterministic schedule과 shared $q_\Phi$ parameter를 training에서만 고정해야 한다.

### 교차 예측과 kill test

상위 edge는 단순 reconstruction이 아니라 그 level의 새로운 intervention latency와 양방향 transfer sign을 예측해야 한다. window를 한 bin 움직였을 때 hierarchy가 전면 교체되거나, 상위 edge가 blind seed에서 재현되지 않거나, one-level quotient와 같은 예측을 내면 multiscale 효능 경로를 죽인다. 새 semantics를 명시하지 않은 모든 반복은 `CGM-D1`의 별도 산출로 부르지 않는다.

## 7. `R4`: context mixture는 마지막에 시험

### 경로

1. context label 또는 gate feature를 recall outcome과 독립적으로 고정한다.
2. encoder를 freeze하고 $K=1$ single metric과 $K>1$ mixture를 같은 effective rank, optimizer step, FLOP로 비교한다.
3. metric은 Cholesky/SPD parameterization으로 보존하되 결과는 gauge-invariant cost와 distance로만 보고한다.
4. training context 조합과 다른 switch order를 blind test로 둔다.
5. basis label shuffle, gate shuffle, geometry shuffle와 unrestricted weight baseline을 수행한다.

### 지위와 자유도

단일 metric보다의 OOD 우위는 [미완성]이다. $K$개 full SPD basis는 $Kp$개, gate는 $q$개 parameter를 가지며 단일 metric 대비 최소 $(K-1)p+q$개가 늘어난다. $K$, rank, gate family를 각각 $L_K,L_r,L_g$개 비교하면 선택 규모는 $L_KL_rL_g$다. test 성능으로 고르면 target-aware 후보가 되므로 nested validation과 고정 test가 필요하다. $K!$ label symmetry와 식 (15)의 continuous representation gauge는 해석에서 quotient한다.

### 교차 예측과 kill test

같은 mixture가 unseen context NLL, route ordering, recall과 interference를 동시에 개선해야 한다. $g_1=\cdots=g_K$이면 single metric으로 정확히 축소되므로 strict-improvement 주장은 즉시 죽는다. encoder를 고정했을 때 이득이 사라지거나, parameter-matched unrestricted quadratic form이 동률이거나, gate/basis shuffle이 성능을 바꾸지 않으면 문맥별 geometry 해석을 죽인다.

## 8. `CGM-H6` 교차도메인 적용 규칙

`CGM-H6`는 별도 수학 route가 아니라 위 후보의 replication gate다. 합성계에서 먼저 `R1`의 식별성과 `R2`/`R3`의 endpoint를 고정한 뒤, 신경 자료에서는 raw edge 일치가 아니라 자료가 실제 제공하는 intervention-conditioned prediction만 같은 이름으로 채점한다. 도메인마다 threshold, window, context 수 또는 encoder를 다시 고르면 retuning 한 번마다 선택 자유도를 기록한다. 한 도메인의 hyperparameter를 freeze한 transfer와 domain-specific retuning을 별도 결과로 보고하며, 둘 중 하나만 성공해도 `같은 규약이 모두 작동한다`는 강한 문장은 남지 않는다.

## 9. 권고 실행 순서와 재개 조건

첫 실행 주제는 `R1`이다. full-rank intervention design에서 고정 chart의 $A,B$조차 재현되지 않으면 SCC나 metric 해석으로 넘어가지 않는다. 다음은 같은 freeze된 dynamics로 `R3`의 Gramian energy를 계산하는 저자유도 기억 시험이고, 병렬로 `R2`의 lumpability residual을 계산한다. `R5`는 한 단계 SCC가 실제로 예측 신호를 가진 뒤에만 열며, `R4`는 single metric과 weight-only baseline이 고정된 뒤에만 연다.

재개 조건은 명확하다. `R1`에는 독립 graph seed와 known intervention이 있는 full-state 합성계, `R2`에는 paired microstate 및 module intervention, `R3`에는 controllable cue interface와 delayed recall, `R5`에는 사전 고정 scale operator, `R4`에는 outcome-independent context와 nested split이 필요하다. 어느 route도 SCC, metric 또는 connectome을 의식의 충분조건으로 사용하지 않는다.

Status: COMPLETE
