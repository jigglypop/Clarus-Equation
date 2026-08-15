# 30-implementation — AGI V15 Unified Metric Agent

Status: COMPLETE

## 1. 승인 범위

`20-audit.md`의 Gate가 허용한 R-A finite metric-graph baseline만 구현했다. 기존 `ClarusField`, BrainRuntime, SCC direct-limit, V10–V14 코드는 변경하지 않았다. full geodesic, Levi-Civita connection, curvature, heat kernel, continuum limit, 비가역 world dynamics와 AGI task loop는 구현하지 않았다.

## 2. 코드 산출

| 파일 | 변경 | 형식 경계 |
|---|---|---|
| `reality_stone/python/reality_stone/clarus/unified_metric.py` | 하나의 SPD metric state, affine tensor transport, fixed-chart projection, edge/path, deformation, surprise, goal-tie readout와 certificate | finite local quadratic + metric graph only |
| `reality_stone/python/reality_stone/clarus/__init__.py` | V15 타입과 함수의 guarded public export | 기존 optional import 구조 유지 |
| `tests/test_unified_metric.py` | 입력 검증, 공변성, projection 경계, 다섯 readout, tie no-go, certificate 검사 | 합성·단위 회귀 17개 |
| `examples/agi/unified_metric_demo.py` | identity 대 barrier metric의 결정론적 예시 | smoke demonstration |
| `dimensionless_checker.py`, `tests/test_dimensionless.py` | $d_g^2/\ell_0^2$와 $M/m$ 등록·검사 | 무차원 정합만 확인 |
| `docs/7_AGI/1_AGI.md`, `18_CodeMap.md`, `28_Nested_Infinite_SCC_V9.md` | 현재 V15 지위, 코드 대응, SCC-to-atlas 경계 반영 | AGI·continuum 승격 없음 |

## 3. 유일 persistent state와 다섯 readout

`UnifiedMetricState`에는 `metric` 필드 하나만 있다. immutable points와 adjacency는 substrate이고, source tensor·관측·예측·후보집합은 transient call input이다. 역할별 persistent state와 역할별 학습 weight는 없다.

| 사용자 개념 | 구현 함수 | 정확한 구현 의미 |
|---|---|---|
| world | `edge_lengths` | symmetric metric-cost substrate; 미래 transition 아님 |
| memory | `metric_deformation`, `apply_source_metric` | 외부 source 전후 tensor 변형 |
| planning | `shortest_path` | finite graph minimum cost |
| critic | `surprise_gate` | local quadratic error의 무차원 hard gate |
| goal | `minimum_cost_targets` | 외부 source가 만든 비용의 모든 최소화점 |

대칭 graph에서는 goal minimizer tie를 모두 반환한다. `shortest_path`가 여러 개인 경우에는 한 representative를 노출하되 `unique=False`와 tie policy를 함께 반환한다. 이 representation tie를 의미 목표의 선택으로 사용하지 않는다.

## 4. metric 갱신과 공변성의 분리

affine chart change $y=Jx+b$는 projection 없이

$$
g_y=J^{-T}g_xJ^{-1}
$$

를 적용한다. local quadratic length, endpoint-average edge length와 고정 graph path cost는 이 경로에서 공변이다.

수치 안정화는 한 chart에서 spectral projection을 적용한다.

$$
P_{[m,M]}(g)=Q\operatorname{diag}(\operatorname{clip}(\lambda_i,m,M))Q^T.
$$

이 연산은 일반 affine-covariant가 아니므로 `projection_affine_covariant=False`를 공개한다. source update는 현재 metric과 source를 이 fixed-chart 범위에 투영한 뒤

$$
g^+=(1-\alpha)g+\alpha g_{\mathrm{source}},
\qquad 0\le\alpha\le1
$$

로 결합한다. 두 endpoint가 $mI\preceq g\preceq MI$이면 convexity로 결과도 같은 범위에 있다.

## 5. 무차원 게이트

critic hard gate는

$$
\chi=\mathbf 1\left[\frac{(z-\hat z)^Tg(z-\hat z)}{\ell_0^2}>\theta\right]
$$

만 사용한다. `reference_scale=$\ell_0$`는 양수이고 $	heta$는 무차원이다. 경계에서 strict inequality를 사용하므로 정확히 같으면 gate는 0이다.

## 6. 공개 certificate

동적 certificate는 실제 최소·최대 고유값, condition number, configured bound와 fixed-chart bound 충족 여부를 낸다. 다음은 하드코딩된 거짓 경계다.

- projection affine covariance,
- full geodesic·connection·curvature·heat kernel,
- continuum limit,
- irreversible world dynamics,
- AGI·생물학·우주론 evidence.

`geometry_scope`는 `finite-point-local-quadratic+metric-graph`, `world_scope`는 `metric_cost_substrate`, `persistent_state`는 `metric_only`, `role_parameter_count`는 0이다.

Status: COMPLETE
