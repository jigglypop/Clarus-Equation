# 20-audit — AGI V15 Unified Metric Agent

Status: COMPLETE

입력 레인 상태를 확인했다. `00-contract.md`와 `11-math.md`, `12-routes.md`는 COMPLETE이고, `10-sources.md`는 외부 관측을 사용하지 않아 사유와 함께 SKIPPED다. 독립 검산 로그 `artifacts/verify_unified_metric_math.log`와 경로 계산 `artifacts/route-explorer-calculations.md`를 대조했다.

## 1. 주장별 지위 감사

| Claim ID | 레인 판정 | 감사 판정 | 근거와 경계 |
|---|---|---|---|
| UM-1 | affine tensor/readout 공변성 정리 | **일치** | $v_y=Jv_x$, $g_y=J^{-T}g_xJ^{-1}$에서 quadratic length가 정확히 불변이다. 고정 adjacency의 endpoint-average edge와 shortest cost까지 보존된다. 변환 후 Euclidean neighbor graph를 재구성하는 경우는 포함하지 않는다. |
| UM-2A | SPD·condition bound 정리 | **일치** | $LL^T+\varepsilon I$의 양의 정부호와 fixed-chart spectral clipping의 $M/m$ 상계는 완전하다. |
| UM-2 affine update | 일반 공변성 없음 | **no-go 경계 승인** | $g=I$, $J=\operatorname{diag}(10,1)$ 반례에서 clipping 뒤 길이제곱이 1에서 10으로 변한다. spectral projection은 fixed-chart 안정화일 뿐 tensor-natural 연산이 아니다. |
| UM-3F | fixed-metric $L^2$ energy·positivity 조건부 정리 | **수정 후 일치** | 최신 계약은 $\phi_0,r\in L^2(g)$에서 energy bound, 추가로 $\phi_0,r\ge0$에서 positivity로 정확히 분리한다. compact 공간의 bounded source는 이 범위에 들어간다. |
| UM-3 광범위 부모 | 삭제·축소 | **처리 완료** | 완비 비콤팩트 $\mathbb R^d$, $r=1$이면 $r\notin L^2$이고 해의 $L^2$ 노름이 무한대다. 활성 계약은 더 이상 이 부모 범위를 주장하지 않는다. |
| UM-3 time-varying | 자동 상속 no-go | **일치** | $\partial_t d\mu_g=\tfrac12\operatorname{tr}_g(\dot g)d\mu_g$의 volume 항이 추가된다. $g_t=e^{2ct}I$ constant-mode 반례는 metric-rate 조건 없이 fixed-$g$ 감쇠를 상속할 수 없음을 보인다. |
| UM-4 | source-free unique-goal no-go | **정리 승인** | 후보를 고정점 없이 교환하는 metric isometry 아래 equivariant singleton selector는 존재할 수 없다. 구현은 모든 minimizer tie를 보존해야 한다. |
| UM-4D | 정적 metric 방향성 no-go | **정리 승인** | 역경로의 길이가 같아 $d_g(x,y)=d_g(y,x)$다. 정적 $g$의 거리만으로 비가역 world transition을 정할 수 없다. |
| UM-5 | SCC/graph continuum | **미완성 일치** | 동일 finite endpoint metric을 가진 smooth interpolation들의 연속 거리가 2.0000과 1.02936으로 달라진다. sampling, overlap cocycle, quadrature/operator consistency와 direct-limit compatibility가 필요하다. |
| UM-6 | 다섯 metric readout | **구현 예측으로 일치** | 같은 $g$에서 finite cost/deformation/path/surprise/minimizer를 읽는 구조 시험이다. 의미론·비가역 예측·AGI를 결론내지 않는다. |

## 2. 명칭과 대상 경계

“세계모형·기억·계획·비평·목표가 $g$에 의해서만 달라진다”는 문장은 구현 단계에서 다음처럼 좁혀야 한다.

| 이름 | 이번 구현에서 허용되는 정확한 대상 |
|---|---|
| world | 고정 graph 위 metric edge-cost substrate |
| memory | 외부 source 전후의 metric deformation; 별도 persistent memory 없음 |
| planning | metric graph shortest path와 cost |
| critic | local quadratic prediction surprise |
| goal | 외부 source가 metric에 기록한 cost의 tie-preserving minimizer 집합 |

미래 transition law, source 선택법, 의미 목표의 기원은 $g$에서 자동 산출되지 않는다. drift나 time orientation을 숨겨 넣고 “$g$만 사용했다”고 부르는 것을 금지한다. SCC 노드는 finite graph sample로만 읽는다. atlas/continuum, Laplace–Beltrami 수렴, node 수 증가에 따른 지능 향상은 인증하지 않는다.

## 3. 경로 판정 감사

R-A discrete atlas/point-cloud를 finite baseline으로 선택한 것은 현재 계약에 가장 적은 추가 구조를 요구하므로 승인한다. 다만 A-A sampling-atlas consistency는 이번 finite 구현에서 검증되지 않으므로 구현 명칭은 `finite metric graph`로 제한한다.

R-C learned full SPD는 $C^2$ regularity와 hidden-role audit가 필요하므로 이번 구현에서 제외한다. R-B conformal plane은 analytic control로만 보존한다. $9{:}1$ anisotropy에 대한 최선 상대 오차 $0.624695\ldots$가 일반 baseline으로서의 표현 한계를 보인다. R-D sub-Riemannian 경로는 intrinsic spectral gap과 bracket rank 검사가 없으므로 제외한다.

## 4. 자유도·무차원 감사

finite 상태 자유도는 $N d(d+1)/2$이며 다섯 역할별로 복제하지 않는다. 구현 config의 $m,M,\alpha$와 정규화된 metric은 무차원이다. surprise hard gate에는 $d_g^2/\ell_0^2$만 넣는다. $d_g^2$ 자체에 차원을 부여하고 그대로 threshold 또는 sigmoid에 넣는 구현은 차원 게이트 실패다.

metric source는 외부 입력 공리다. source를 관측에서 어떻게 생성하는지, source가 올바른 목표를 표현하는지는 이번 run의 산출이 아니다.

## 5. 문제 목록과 해소 상태

- **P0 없음.** UM-3의 광범위 부모는 계약에서 제거됐고 정확한 $L^2$/positivity 조건만 활성이다.
- **P1-1 해소 조건:** affine readout covariance와 fixed-chart spectral projection을 certificate에서 별도 필드로 표시한다.
- **P1-2 해소 조건:** world를 비가역 predictor가 아니라 `metric_cost_substrate`로 명명한다.
- **P1-3 해소 조건:** full geodesic, connection, curvature, heat kernel, continuum limit를 구현하지 않았으면 모두 명시적으로 false로 노출한다.
- **P1-4 해소 조건:** goal readout은 candidate 순서와 무관한 모든 최소화점 집합을 반환하고 singleton tie-break를 숨기지 않는다.
- **P1-5 해소 조건:** 유일한 persistent semantic state가 metric인지 구조 검사로 고정하고 역할별 persistent state·가중치를 추가하지 않는다.
- **P2:** 본체 명칭은 $d$차원 `metric graph`; 2D plane은 시각화/analytic control로만 쓴다.

집계: 감사 명제 28개, 정리·조건부 정리 6개, no-go 정리 3개, 명시 공리 3개(외부 source, fixed chart, finite graph topology), 구현 예측 1개, 미완성 5개(full geodesic/connection/curvature, continuum, directed dynamics, source semantics, AGI utility), 제거된 부모 범위 1개.

## 6. 구현 승인 범위

다음 opt-in finite baseline만 승인한다.

1. finite points와 고정 symmetric connected topology,
2. SPD validation, fixed-chart spectral projection과 condition certificate,
3. projection 없는 affine chart change와 local/edge/path cost covariance,
4. endpoint-average metric edge와 shortest path,
5. $d_g^2/\ell_0^2$ local surprise hard gate,
6. projected SPD source와 현재 metric의 convex deformation,
7. metric-only persistent state와 source-free symmetric-goal tie regression,
8. 공개 certificate의 비승인 항목을 모두 false로 고정.

기존 `ClarusField`, BrainRuntime, SCC direct-limit 코드는 수정하지 않는다. 새 module은 opt-in이어야 한다. 합성 barrier fixture와 회귀검사 외 과제 utility 주장은 이번 run에 포함하지 않는다.

Gate: PASS
