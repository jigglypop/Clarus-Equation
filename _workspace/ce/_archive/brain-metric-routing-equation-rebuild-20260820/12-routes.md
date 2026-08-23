# 후속 경로와 반증 조건

Status: COMPLETE

## 선택 원칙

새 식의 목적은 양성 결과를 만들기 위한 seed나 decoder 탐색이 아니다. 실제 기록에서 서로 다른 두 estimand,

$$
G^{o\leftarrow A}
\quad\text{and}\quad
R^{A\to B},
$$

를 같은 세션·시간축에서 독립 검정하는 것이다. 한쪽 실패를 다른 쪽이 보상하지 않는다.

## Route A — 실제 same-session 관측 검정

가장 먼저 실행할 경로다.

### 입력 적격성

- 같은 세션에서 source neural activity, target neural activity, future 행동 출력, task context와 nuisance가 동기화되어야 한다.
- source/target unit과 trial/block provenance가 보존되어야 한다.
- pseudopopulation, 서로 다른 session의 행 결합, 불명확한 시간 정렬은 `APPARATUS_INVALID`다.

### 추정

1. training block에서만 source chart $\psi_j$를 적합한다.
2. 사전 지정 future behavior/output likelihood로 $G^{o\leftarrow A}$를 추정한다.
3. 별도 target-neural likelihood로 $R^{A\to B}$를 추정한다.
4. context별 metric과 route model은 동일 held-out block에서 각각 자신의 adverse controls와 비교한다.
5. raw SPD는 session 밖으로 평균하지 않고 session-level invariant scalar만 animal hierarchy에 넣는다.

### 필수 falsifier

- metric: shared-context, gain-only, diagonal, nuisance-only, reference-dominated.
- route: target-history-only, block-preserving source shift, reverse direction, global-state augmented. Negative-lag/time-reverse는 시간 대칭·leakage·공통구동을 찾는 사전 지정 방향성 진단이며, 단독으로 0이어야 하는 strict null은 아니다.
- transform: orthogonal, diagonal rescale, condition-number가 제한된 affine rechart.

### 최대 주장

두 gate가 모두 통과해도 “동일 실제 recording에서 context-sensitive output Fisher geometry와 lagged conditional predictive transfer가 함께 관측되었다”까지다. 인과 라우팅이나 metric mediation은 아니다.

## Route B — randomized perturbation으로 causal routing 검사

Route A와 독립된 입력이 필요하다.

- source 영역의 시간 고정 perturbation, sham, 반대 방향·비표적 자극을 무작위 배정한다.
- perturbation 이전에 chart, horizon, metric estimator, route model을 동결한다.
- $do(A)$가 target future와 behavior에 미치는 효과를 추정하되, $G$가 매개한다는 주장은 별도 mediator intervention 없이는 금지한다.
- rescue 또는 closed-loop reversal이 있으면 특정 route의 causal necessity/sufficiency를 더 강하게 검정할 수 있다.

최대 주장은 “이 intervention이 이 source-target transfer와 output을 바꾸었다”까지다. synaptic path의 유일성은 structural measurement 없이 성립하지 않는다.

## Route C — longitudinal structure producer

가장 강하지만 현재 입력이 없다.

- 같은 animal/cell/synapse의 $W_0,W_1$.
- 독립 calibration에서의 $G_0,G_1$.
- 학습 전후 held-out neural trajectory와 behavior.
- randomized plasticity/connectivity intervention, sham, gain/noise controls.

이 조건이 있어야 $\Delta W\to\Delta G$를 검정할 수 있다. 그 뒤에도 $\Delta G\to\Delta x$ mediation은 독립 mediator intervention 또는 충분한 causal state model이 필요하다.

## 폐기·보류 경로

| 경로 | 상태 | 이유 |
|---|---|---|
| $C^{-1}$을 primary brain metric으로 사용 | RETIRED | 비선형 chart tensor law를 만족하지 않고 task output을 식별하지 않음 |
| SPD 하나로 route/dynamics 예측 | RETIRED | 같은 $G$, 다른 $R$ 완전 반례와 G2 simulator `STOP` |
| 새 BrainRuntime seed/controller 탐색 | DEFERRED | 실제자료 식 검정이 우선이며 C1은 이미 `STOP` |
| SCC lesion을 기억/의식 경로로 해석 | STRUCTURE_UNDEFINED_STOP | dense support와 matched lesion 문제, biological identity 부재 |
| PFC pseudopopulation을 same-unit route로 재사용 | RETIRED | 시간축·동시성·same-unit가 없음 |

## 다음 실행 순서

1. 이 equation run의 독립 수학·형식지위 audit.
2. PASS 후 canonical brain paper의 식과 주장 지위 갱신.
3. 로컬 실제 시계열의 schema/time alignment eligibility만 판정.
4. 적격할 때 Route A를 별도 preregistered empirical run으로 실행.
5. mammalian same-session dataset에서 독립 replication.

실제자료가 적격하지 않으면 새 synthetic seed로 돌아가지 않고 `BLOCKED_INPUT`을 기록한다.
