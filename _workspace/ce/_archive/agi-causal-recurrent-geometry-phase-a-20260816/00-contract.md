# Phase A: 개입 조건부 인과 재귀 기하 식별 benchmark 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-connectome-geometric-memory-20260816

Mode: light follow-up

## 1. 질문

고정된 관측 좌표에서 완전히 관측되는 유한 이산시간 선형계와 관측된 문맥을 사용해, 알려진 개입 입력이 있을 때 계수 $A_z,B$를 학습 데이터만으로 복원하고 보지 않은 intervention의 다음 상태를 예측할 수 있는 최소 benchmark를 구현한다. 동시에 관측-only latent graph 유일복원이 불가능하다는 선행 no-go를 executable negative fixture로 보존한다.

이번 run의 핵심 비교 질문은 다음과 같다.

> 문맥별 전이 $A_z$를 분리한 estimator가 같은 데이터와 parameter 회계를 받는 pooled transition baseline보다 fresh development graph에서 held-out intervention prediction을 개선하는가.

strict 우위는 정리가 아니라 development 비교 결과다. confirmation seed는 이 run에서 실행하지 않는다.

## 2. 정의역과 표기

Phase A V1의 생성계는 다음과 같다.

$$
x_{t+1}=A_{z_t}x_t+Bu_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2I).
\tag{1}
$$

$x_t\in\mathbb R^n$은 정규화된 무차원 상태, $u_t\in\mathbb R^m$은 정규화된 무차원 개입, $z_t\in\{0,\ldots,K-1\}$는 관측된 문맥 label이다. $A_z$와 $B$는 무차원 discrete-step 계수다. 정확한 계수·support는 고정 chart의 known-identity regime에서만 평가한다.

관측은 다음 세 type을 구분한다.

$$
y_t=Cx_t+\nu_t.
\tag{2}
$$

| 관측 type | $C$ | 허용 주장 |
|---|---|---|
| known identity | $I$ | 선언한 선형 class에서 계수·support와 prediction 평가 |
| known mask | 알려진 row selector | 관측 subspace prediction만 평가 |
| unknown mix | 미지의 가역 행렬 | exact latent edge 금지; 관측·개입 prediction만 평가 |

Phase A V1의 학습 estimator는 known identity만 소비한다. 다른 regime은 claim certificate가 exact-edge 승격을 거부하는 fail-closed fixture로 구현한다.

## 3. 무차원 계약

식 (1)의 모든 합은 동일한 무차원 state 단위다.

| 코어 인자 | 차원 벡터 $(M,L,T,\Theta)$ | 정규화 |
|---|---|---|
| $x_t,y_t$ | $(0,0,0,0)$ | future physical data에서는 $x/x_{\rm ref}$ |
| $u_t$ | $(0,0,0,0)$ | future physical data에서는 $u/u_{\rm ref}$ |
| $A_z,B,C$ | $(0,0,0,0)$ | discrete step와 기준 scale에 흡수 |
| $\epsilon_t,\nu_t,\sigma$ | $(0,0,0,0)$ | state reference scale로 나눔 |
| Gaussian residual $(y-\hat y)/\sigma$ | $(0,0,0,0)$ | $\sigma>0$을 명시 |

후속 nonlinear generator가 $\tanh(A_zx+Bu+b_z)$를 사용하면 $\tanh$의 전체 인자는 반드시 무차원이어야 한다. 이번 V1은 선형 anchor만 구현하며 dimensionless certificate가 false인 입력을 거부한다. 무차원성은 식별성이나 모델 적합성을 증명하지 않는다.

## 4. 계약 주장

| Claim ID | 시작 지위 | 계약 |
|---|---|---|
| PA-N1 | 선행 [정리: no-go]의 executable fixture | 서로 다른 latent support를 가진 similarity-related LTI가 동일 관측열을 만들 수 있음을 exact fixture로 재현 |
| PA-T1 | 선행 [정리: 조건부]의 executable fixture | noiseless known-identity full-rank design에서 $[A\ B]$를 수치 오차 내 복원 |
| PA-T2 | [정리 후보: 구현 계약] | rank-deficient design은 exact identification certificate를 거부 |
| PA-D1 | [정의] | anatomy, latent causal support와 predictive transition을 같은 field로 합치지 않음 |
| PA-D2 | [정의] | 모든 state·input·noise와 likelihood residual은 무차원 |
| PA-I1 | [미완성: 구현] | generator는 graph seed와 trajectory seed를 분리하고 동일 seed에서 deterministic |
| PA-I2 | [미완성: 구현] | estimator는 training의 $x_t,u_t,z_t,x_{t+1}$만 소비하며 truth $A_z,B$나 test outcome을 보지 않음 |
| PA-I3 | [미완성: 구현] | unknown-mix와 known-mask에서 exact-edge claim이 fail closed |
| PA-H1 | [미완성: development 비교] | factorized context model이 pooled baseline보다 held-out intervention Gaussian NLL을 개선 |
| PA-H2 | [미완성: integrity 비교] | intervention tag/time shuffle은 causal input prediction을 악화시킴 |
| PA-X1 | 활성 제외 | Phase A 결과를 SCC 효능, 기억, 생물학, 의식 또는 AGI 증거로 사용하지 않음 |

## 5. 구현 승인 후보

Gate가 허용할 때 다음 새 격리 표면만 구현한다.

1. reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py
2. tests/test_causal_recurrent_geometry_benchmark.py
3. experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
4. examples/agi/causal_recurrent_geometry_development_run.py
5. `.gitignore`의 정확한 예외 `!experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json` 한 줄

기존 runtime, __init__.py, SCC/metric 구현, 정본과 default flag는 수정하지 않는다. `.gitignore`의 기존 사용자 변경은 보존하고 위 exact-path 예외만 추가한다. current infinite-tail, dirty V15와 untracked V16/V17 자산을 import하지 않는다.

## 6. estimator와 대조군

candidate는 문맥별 transition과 공유 input matrix를 최소제곱 또는 고정 ridge로 학습한다.

$$
\widehat x_{t+1}
=
\widehat A_{z_t}x_t+\widehat Bu_t.
\tag{3}
$$

pooled baseline은 모든 문맥에 하나의 $\widehat A$를 사용한다. candidate와 baseline은 같은 trajectory, input, loss와 ridge 값을 사용한다. candidate의 추가 자유도는 $(K-1)n^2$이며 결과에 명시한다. V1의 목표는 우위 정리가 아니라 이 차이를 회계한 positive/negative benchmark anchor다.

## 7. split과 endpoint

seed namespace는 manifest에서 고정한다.

- pilot: sample-size나 코드 sanity에만 사용하며 점수에 포함하지 않음;
- development: 구현·모델 선택에 사용 가능한 fresh graph seeds;
- confirmation: manifest에 예약하되 이번 run에서 열지 않음.

primary endpoint는 graph seed별 held-out intervention Gaussian NLL의

$$
\Delta_s
=
\operatorname{NLL}_{\rm pooled,s}
-
\operatorname{NLL}_{\rm factorized,s}
\tag{4}
$$

이며 positive가 candidate 개선이다. development 판정은 평균, median, paired bootstrap interval과 seed-level 값을 모두 보고한다. edge error, coefficient error와 free rollout은 secondary다. frame을 독립 통계 단위로 세지 않는다.

## 8. 정확도·실패 규칙

- noiseless full-rank fixture의 최대 계수 오차: $10^{-10}$ 이하;
- deterministic replay: 같은 manifest·seed에서 직렬화된 결과 동일;
- finite/domain check: NaN, infinity, $\sigma\le0$, 빈 context, shape mismatch를 거부;
- no-future/no-hidden: estimator API에 truth coefficient나 test target을 전달하지 않음;
- exact-edge certificate: known identity, full-rank와 선언한 model class를 모두 만족할 때만 true;
- development의 $\Delta$가 양수가 아니면 PA-H1은 STOP이며 estimator 구현 성공으로 뒤집지 않음;
- shuffled input이 intact와 동률이면 PA-H2는 STOP;
- confirmation 결과는 이번 run에서 없음으로 유지.

## 9. 구현 전 감사 질문

1. 공유 $B$와 문맥별 $A_z$가 full-rank design에서 유일하게 식별되는 정확한 design 조건은 무엇인가.
2. pooled baseline과 candidate의 dof 차이를 어떤 판정표에 노출해야 하는가.
3. held-out intervention의 NLL에서 noise scale을 truth로 둘지 train residual로 추정할지 결정한다.
4. graph seed, trajectory seed와 intervention seed가 서로 누수되지 않는지 검사한다.
5. observation-only no-go, rank deficiency와 unknown-mix refusal이 positive benchmark와 같은 API에서 fail closed인지 검사한다.

## 10. 종료 조건

이 run은 다음이 모두 성립할 때만 COMPLETE다.

1. independent math lane과 formal audit가 구현 범위를 승인한다.
2. 새 네 표면의 focused tests, Ruff와 compile check가 통과한다.
3. development result와 모든 실패/경계가 31-validation에 원문으로 남는다.
4. confirmation은 실행되지 않았음이 manifest와 report에 명시된다.
5. 제품 runtime, 정본, 기존 seed와 user changes가 보존된다.
