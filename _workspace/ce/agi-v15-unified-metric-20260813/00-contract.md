# 00-contract — AGI V15 Unified Metric Agent

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-clarus-field-20260812`

## 1. 질문

[연구 질문] 세계모형, 기억, 계획, 비평, 목표를 서로 다른 에이전트나 독립 head로 두지 않고, 하나의 정보다양체 $(\mathcal M,g_t)$와 하나의 공통 계량장 $g_t$에서 얻는 기하 연산으로 구성할 수 있는가. 현재 유한·재귀 SCC의 노드는 별도 정신이 아니라 같은 다양체의 chart 또는 수치 표본으로 해석하고, 노드 수 증가를 atlas/mesh refinement로 읽을 수 있는가.

[범위] 이번 run은 AGI 달성을 주장하지 않는다. 형식적 기하 core, 완전한 반례와 no-go, 최소 구현, 합성 계량 검사를 만든다. 장기계획·연속학습·도구사용을 아우르는 AGI 과제 채점은 다음 run의 사전등록 대상으로 남긴다.

## 2. 정의역과 기호

[정의] $\mathcal M$은 연결된 매끄러운 $d$차원 정보다양체다. 구현 baseline은 한 chart에 들어가는 유한 점 집합 $z_i\in\mathbb R^d$와 그 위의 대칭 양의 정부호 행렬 $g_i\in\operatorname{SPD}(d)$를 사용한다. 2차원은 시각화·곡률 fixture이고 본체의 차원은 임의의 유한 $d\ge2$다.

[정의] 좌표변환 $y=f(x)$의 Jacobian을 $J=\partial y/\partial x$라 하면 metric 성분은

$$
g_y=J^{-T}g_xJ^{-1}
$$

로 변환한다. 점·접벡터·공변량도 같은 좌표변환 규칙을 따라야 한다.

[정의] 수치 구현은

$$
g_\theta(z)=L_\theta(z)L_\theta(z)^T+\varepsilon_g I,
\qquad \varepsilon_g>0
$$

또는 입력된 SPD 행렬을 사용한다. $g$의 고유값은 사전등록된 $0<m\le\lambda_{\min}(g)\le\lambda_{\max}(g)\le M<\infty$ 범위에 투영한다. $m,M,\varepsilon_g$는 정규화된 정보좌표에서 무차원이다.

[정의] 현재 Clarus의 gate 기호와 metric을 구분한다. 계량은 $g_{\mu\nu}$, hard write indicator는 $\chi\in\{0,1\}$로 쓴다. geodesic surprise는

$$
\delta_g^2(z,\hat z)=d_g(z,\hat z)^2,
\qquad
\chi=\mathbf 1[\delta_g^2/\ell_0^2>\theta]
$$

다. $\ell_0>0$은 정보거리 기준척도이며 지표함수의 인자는 무차원이어야 한다.

## 3. 하나의 $g$에서 읽는 다섯 기능

다음은 별도 학습 head가 아니라 같은 $g$의 함수형으로만 허용한다.

| 이름 | 허용된 기하 객체 | 독립 역할 파라미터 |
|---|---|---:|
| world | Levi-Civita connection, exponential/log map, metric heat kernel | 0 |
| memory | 관측 source가 남긴 지속적 SPD metric deformation, parallel transport/holonomy | 0 |
| planning | $d_g$ 또는 metric edge length에 대한 최소작용 경로 | 0 |
| critic | geodesic prediction residual, curvature/geodesic deviation | 0 |
| goal | 외부 경계조건이 $g$에 기록한 metric basin의 invariant readout | 0 |

[공리: 외부 source] 관측과 사용자 선호는 무에서 $g$로 나오지 않는다. 이들은 공통 update law의 source 또는 경계조건으로 들어간다. source가 들어온 뒤에는 별도 persistent goal/memory head를 두지 않고 $g_t$만 지속 상태로 보존한다.

## 4. 검사 명제

- **UM-1 [후보 정리: 좌표 공변성].** metric, 점, 접벡터를 함께 변환하면 local quadratic length, curve length, geodesic distance와 그 거리만 쓰는 plan/critic readout은 좌표변환에 불변이다. 구현은 최소한 임의의 invertible affine chart change에서 수치 검사를 통과해야 한다.
- **UM-2 [후보 정리: SPD·유계].** eigenvalue projection 또는 $LL^T+\varepsilon_gI$는 metric의 양의 정부호를 보존한다. metric update 뒤에도 $[m,M]$ 투영을 적용하면 condition number는 $M/m$ 이하이다.
- **UM-3 [후보 조건부 정리: 고정 metric 장].** fixed Riemannian manifold에서 $\phi_0,r\in L^2(g)$이면 $\partial_t\phi=\kappa\Delta_g\phi-\lambda\phi+r$는 $L^2(g)$ energy bound를 갖고, 추가로 $\phi_0,r\ge0$이면 positivity를 보존한다. compact manifold에서는 유계 source가 $L^2$ 조건을 만족한다. 완비 비콤팩트 공간의 점별 유계 source만으로는 $L^2$ 결론이 나오지 않는다. 시간가변 $g_t$에는 별도 metric-rate 조건이 필요하며 자동 상속을 금지한다.
- **UM-4 [후보 no-go].** 고정된 $g$와 isometry-equivariant 연산만으로는 metric의 isometry가 교환하는 두 후보 중 하나를 의미 있는 목표로 유일하게 선택할 수 없다. 목표 선택에는 symmetry-breaking source/boundary 또는 현재 상태가 필요하다.
- **UM-5 [미완성/가설: SCC 연속체].** 유한 SCC node를 chart/sample로 읽고 graph operator가 $\Delta_g$에 수렴한다는 주장은 sampling, bandwidth, overlap transition과 consistency 조건을 명시할 때만 조건부 정리 후보이다. node 수 증가가 지능 증가를 뜻한다는 주장은 하지 않는다.
- **UM-6 [예측: 역할 통합].** 동일한 $g$에서 파생된 world/memory/planning/critic/goal readout이 별도 역할 가중치 없이 좌표변환 동등성과 합성 metric barrier fixture를 통과할 수 있는지 검사한다. 통과해도 AGI 효능으로 승격하지 않는다.

## 5. 열린 경로

route 레인은 공리 추가가 후보당 1개 이하인 구조적으로 다른 경로를 최소 세 개 비교한다.

1. discrete atlas/point-cloud metric core,
2. conformal metric plane과 curvature flow,
3. continuous learned SPD metric 또는 sub-Riemannian 대안.

정적 Riemannian distance의 대칭성이 비가역·방향성 행동 비용을 표현하지 못하면 time-dependent metric, tangent velocity 또는 sub-Riemannian control distribution 가운데 필요한 최소 구조를 분리한다. 이를 몰래 $g$ 하나의 산출로 부르지 않는다.

## 6. 구현 승인 후보와 계량 항목

감사 전에는 구현하지 않는다. Gate가 허용하면 기존 `ClarusField`를 파괴하지 않는 opt-in module로 다음 최소 범위를 구현한다.

- SPD metric validation/projection과 condition-number certificate,
- affine chart covariance of local lengths,
- metric edge lengths와 shortest-path planning,
- geodesic surprise hard gate,
- 외부 source에 의한 bounded metric deformation 하나,
- metric-only symmetric-goal no-go regression,
- $g=I$ 대 deformed-$g$의 합성 barrier ablation,
- public certificate에 AGI, biological, cosmological, continuum-limit 미검증 지위 노출.

[사전 고정 계량] 좌표 불변 오차 $\le10^{-10}$, SPD 최소 고유값 $\ge m-10^{-12}$, 최대 고유값 $\le M+10^{-12}$, plan cost 좌표 불변 상대오차 $\le10^{-10}$를 구현 허용 오차로 둔다. 합성 barrier는 성공·실패 여부보다 Euclidean과 metric plan이 사전 구성된 cost를 정확히 최소화하는지를 검사한다. 성능 튜닝은 허용하지 않는다.

## 7. killing 조건과 경계

- affine chart change 뒤 invariant readout이 허용 오차 밖으로 변하면 “하나의 기하 객체” 구현 주장은 사망한다.
- 역할별 숨은 weight, 별도 persistent state 또는 목표 label을 readout에 직접 전달하면 “$g$만 공유” 주장은 사망한다.
- symmetric metric에서 source 없이 유일 목표를 반환하면 equivariance 위반 또는 숨은 tie-break이며 의미 목표 생성 주장으로 사용할 수 없다.
- time-dependent $g_t$의 안정성을 fixed-$g$ energy proof에서 인용하면 자동 상속 주장은 사망한다.
- 합성 fixture 통과는 AGI, 뇌, 우주 물리 증거가 아니다.

## 8. 외부 자료와 재현

외부 관측값·외부 데이터셋을 사용하지 않는다. 따라서 10-sources 레인은 스킵한다. 상세 수치 스크립트와 로그는 이 run의 `artifacts/`에만 둔다. 구현·수치 검증은 Gate 이후에 수행한다.

Status: COMPLETE
