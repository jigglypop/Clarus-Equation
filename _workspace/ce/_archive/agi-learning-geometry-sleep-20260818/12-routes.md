# 학습된 계산 기하와 수면 재정렬: 대안 경로

Status: COMPLETE

## 출발점

`LGS-N1`--`N3`의 P0 반례 때문에 구조적으로 같은 기하, weight-only 유일 기하, 보편 $L/v$ 공식을 회복하는 경로는 없다. 아래는 등록된 미완성 가설을 더 좁게 시험하는 대안이며, 모두 test endpoint를 본 뒤 선택하지 않는 것을 전제로 한다.

| Route | 대상 | 추가 가정 (최대 1개) | fitted dof / 선택 규모 | target-aware | 독립 교차 예측 | kill test |
|---|---|---|---|---|---|---|
| R1 context-conditioned cost map | H1, H4, H6 | $\Phi_c(W,A,c;\theta)$와 action/cost protocol을 training 전에 고정 | $q_\theta$; model family $L_\Phi$, context feature set $L_c$ | 아니오; test로 고르면 예 | 동일 $\theta$가 unseen context의 cost, next-state NLL, hitting time을 같이 예측 | raw $W$+activity+latency matched baseline 대비 paired held-out gain CI가 0을 못 넘음, 또는 geometry shuffle 동률 |
| R2 critical-path timing | H2 | task completion은 preregistered required-DAG의 critical path | edge delay parameter $r$ 또는 fixed measured delays; architecture alternatives $L_{\rm DAG}$ | 아니오; RT를 보고 DAG를 선택하면 예 | new task의 RT와 dual-task/join manipulation effect를 동일 delay vector로 예측 | 단순 length/$v$ 또는 parameter-matched latency model이 동률, parallel/join intervention 방향 실패 |
| R3 selective practice cost | H3 | pair usage dose가 training 전에 기록되고 untrained matched pair가 존재 | dose slope 1 plus declared covariates $q$; threshold bins $L_b$ | 아니오; dose cutpoint를 test에서 고르면 예 | distance/cost change, energy proxy, error/RT를 같은 pair에서 함께 예측 | global scaling, exposure-only, raw-$W$ baseline이 설명하거나 trained/untrained 차이가 없음 |
| R4 sleep selective rearrangement | H5, H6 | sleep/wake matched protocol과 predeclared local/global metric estimator | estimator $q_g$, sleep interaction 1, replay coupling 1; window choices $L_W$ | 아니오; post-sleep endpoint로 window를 고르면 예 | next-day trajectory fidelity와 generalization을 same-night selective change가 동시 예측 | global scaling 또는 time-awake/no-sleep control과 구별 불가, replay coupling shuffle이 무효 |

## R1: context를 조건으로 둔 operational geometry

고정 state/action chart에서 $W,A,c$를 측정하고 training-only로 선언한 $\Phi_c$가 $w_c$를 낸다. APSP 또는 control-energy는 이 derived cost에서만 계산한다. 모든 $\theta$, scaling, context encoding, missingness 처리는 holdout 전에 고정한다. raw $W$, local activity, latency, Euclidean latent distance와 parameter-matched unrestricted predictor를 필수 baseline으로 둔다.

자유도는 $q_\theta$와 후보 family의 $L_\Phi L_c$ 선택이다. 이는 $W\to g$의 유일성 정리가 아니라 fixed procedure가 added prediction을 갖는지의 가설이다. 같은 frozen map이 context switch 이후 cost ordering과 trajectory endpoint 모두를 맞히지 못하면 route를 기각한다.

## R2: 길이가 아니라 edge delay의 critical path

각 task를 required precedence DAG로 등록하고 edge delay $\tau_e$를 independent calibration에서 고정 또는 train split에서 fit한다. 예측치는

$$
\widehat T=t_0+\max_{p}\sum_{e\in p}\tau_e+t_{\rm integrate}.
$$

단일 serial graph가 아니라 branch join을 조작하여 critical path의 변화만 예측하는 것이 핵심이다. $L_{\rm DAG}$개의 가능한 task graph를 반응시간을 보고 택하면 look-elsewhere로 회계한다. positive result도 cognitive mechanism 증명이 아니라 지정된 completion model의 predictive success다.

## R3: 자주 쓰는 쌍의 선택적 비용 감소

동일 초기 cost와 complexity를 가진 state pair를 matched하고, training phase의 usage dose를 outcome과 분리해 기록한다. primary는 pair-level $\Delta d_c$ 또는 independent transition-cost estimate, secondary는 error/RT/energy proxy다. global gain 혹은 uniform all-pair scaling은 null/대조이며 pair identity permutation과 usage-dose shuffle을 수행한다.

선형 dose slope 하나만 primary로 두고 $L_b$개 bin/cutpoint 탐색은 secondary sensitivity로 남긴다. 모든 pair의 비용이 같이 줄거나 untrained matched pair와 차이가 없으면 selective specialization을 기각한다.

## R4: 수면의 선택적 재정렬

수면 전후의 $W,A,c$에서 R1과 같은 predeclared estimator를 적용하고, local affected circuit과 non-affected circuit을 동시에 추적한다. sleep/wake, time-awake, firing-rate, global scaling, replay-ablation/matched-sham을 대조한다. primary estimator window와 threshold는 한 번만 선언한다; $L_W$개 window 중 best를 고르면 target-aware selection이다.

다음 날 trajectory fidelity와 generalization을 둘 다 예측하고, global scaling보다 selective model이 낫고 replay-coupling shuffle이 성능을 떨어뜨릴 때만 route가 살아남는다. 이것은 NREM 뒤 REM이라는 고정 직렬 알고리즘을 전제하지 않으며, 지역 비동기 maintenance와 주기적 global synchronization도 동등한 경쟁 모델로 남긴다.

## 해석 제한과 재개 조건

네 route는 AGI 충분조건, 해부학적 기질의 유일성, Riemannian geometry 또는 수면 알고리즘을 증명하지 않는다. 재개에는 independent graph/animal/session split, declared unit/time scale, training-only selection ledger, intervention/no-sleep controls가 필요하다. 어떤 route든 matched baseline에서 strict held-out gain이 없으면 geometry를 독립 기전으로 주장하지 않는다.
