# 12-routes — AGI V15 Unified Metric Agent

Status: COMPLETE

## 0. 고정 판정 기준

표적은 특정 상수 맞추기가 아니라 구조 시험이다. 유한 표본과 연속 다양체에서 barrier의 양의 metric 변형 $\Delta g\succeq0$가 국소 비용을 낮추지 않아야 하며, worldㆍmemoryㆍplanningㆍcriticㆍgoal은 역할별 상태ㆍheadㆍ가중치 없이 같은 $g_t$만 읽어야 한다. 수치 효과크기는 사후 맞추지 않고 계약의 covariance/plan 허용오차 $10^{-10}$을 쓴다.

모든 후보는 계약의 외부 source/boundary 공리를 공유한다. 아래에는 후보별 **추가 공리를 정확히 하나 이하**만 둔다. 모든 후보가 다섯 기능이라는 구조 표적을 본 뒤 계약에 포함된 경로이므로 `target-aware=예(구조)`다. 다만 맞춰야 할 숨은 수치 target은 없고, 아직 결과를 본 수치 tuning도 없다.

상세 유도와 반례는 `artifacts/route-explorer-calculations.md`에 분리했다.

## 1. 비교표

| 순위 | 경로 | 후보별 추가 공리 | persistent state dof / search dof | target-aware | 핵심 교차예측 | killing test | 구현 난이도 | 판정과 AGI 경계 |
|---|---|---|---|---|---|---|---|---|
| 1 | **R-A discrete atlas / point cloud** | A-A: shape-regular sampling, $h_N\to0$, overlap cocycle, 보간/quadrature와 reversible adjacency 일관성 | $N d(d+1)/2$; 최소 연속 search 1개(neighbor scale), categorical 선택은 사전등록 | 예(구조), 수치 target 없음 | 같은 $\Delta g$가 edge lengthㆍgraph heatㆍmemory deformationㆍplanㆍcriticㆍgoal basin을 함께 바꿈 | 운반된 adjacency에서 affine/plan 오차 $>10^{-10}$; refinement 비수렴; hidden interpolation; directed SCC를 LB로 동일시; node-id goal | **중간** | **PRIMARY**: 계약의 finite baseline과 가장 직접적으로 맞는다. 통과해도 graph 기반 기하 certificate일 뿐 AGI나 continuum 지능이 아니다. |
| 2 | **R-C continuous learned full SPD** | A-C: 영역에서 $g_\theta\in C^2$, 1ㆍ2차 미분 유계 | network $p$개; 출력은 $N d(d+1)/2$; 역할 head 0개. architecture 수 $H$와 seed 수도 search로 기록 | 예(구조), 수치 target 없음 | 한 source update가 connection/heat, persistent deformation, numerical geodesic, residual/curvature, goal basin을 동시에 바꿈 | clip 경계 비매끄러움; transformed-chart 재-clipping/$+\epsilon I$; hold-out chart/barrier 실패; hidden role head | **높음** | **SECONDARY**: anisotropy와 연속장을 표현하지만 자유도와 regularity 부담이 크다. metric learner 자체는 세계모델ㆍ자율목표ㆍ장기계획의 증거가 아니다. |
| 3 | **R-B conformal plane / curvature flow** | A-B: $d=2$, 사전 고정 $g_0$에 대해 $g=e^{2u}g_0$; readout에는 $g$만 전달 | $N$ scalar; flow/goal variant를 둘 이상 보면 categorical search 증가 | 예(구조), 수치 target 없음 | 하나의 $u$가 connectionㆍheatㆍpersistent $u$ㆍweighted geodesicㆍcurvature critic을 정확식으로 묶음 | $g_0$ 대비 anisotropy ratio $\ne1$ fixture; $g_0$ 미운반 affine 시험; time-varying 안정성 자동상속 | **낮음~중간** | **ANALYTIC CONTROL, 일반 baseline은 기각**: 저자유도 검산에는 좋지만 anisotropic memory/barrier와 $d>2$ 일반성을 표현하지 못한다. 곡률 flow는 학습이나 목표 의미론이 아니다. |
| 4 | **R-D $g$-derived sub-Riemannian** | A-D: $d\ge3$에서 $\operatorname{Ric}_g^{\sharp}$의 고정 $k$-cluster에 uniform spectral gap | full-SPD dof와 동일, 추가 연속 state 0; $k$를 훑으면 $d-1$ categorical trials. 독립 $H$는 금지된 $Nk(d-k)$ 추가 dof | 예(구조), 수치 target 없음 | $g$에서 유도된 $H_g$ 하나가 horizontal heatㆍholonomyㆍdistanceㆍresidualㆍgoal basin을 묶음 | spectral gap 폐쇄; bracket-rank 부족; raw matrix eigenvectors; forward/backward 비대칭 주장 | **매우 높음** | **DEFER / 진단용**: 접근가능성 구조가 꼭 필요한 후속 benchmark 전에는 과도하다. 대칭 이차비용으로 비가역 행동을 만들 수 없으며 controllability는 AGI가 아니다. |

## 2. 후보별 짧은 판정

### R-A — 먼저 구현할 finite core

계약이 이미 $N$개 점과 $g_i\in\operatorname{SPD}(d)$를 주므로 추가 표현 변환이 가장 적다. $y=Jx+b$, $g_y=J^{-T}g_xJ^{-1}$일 때

$$
\ell_{ij}^2=(z_j-z_i)^T\frac{g_i+g_j}{2}(z_j-z_i)
$$

는 정확히 불변이고, 같은 edge set을 운반하면 shortest-path cost도 불변이다. 최소 구현은 SPD projection/certificate, invariant edge length, shortest path, geodesic-surprise hard gate, bounded source deformation, symmetric-goal no-go, identity/deformed barrier ablation이다.

다만 affine 변환 뒤 Euclidean $k$-NN을 새로 만들면 adjacency가 달라질 수 있다. `adjacency transport`와 `neighbor reconstruction`을 별개 시험으로 기록해야 한다. 또한 node 수 증가를 atlas refinement라고 부르려면 A-A 아래 distanceㆍheatㆍtransport의 refinement 수렴이 실제로 확인되어야 한다.

유한 node의 $g_i$ 값만으로 continuum geometry가 정해지지는 않는다. node 사이에 지지되는 smooth conformal bump는 모든 $g_i$를 그대로 두면서 그 사이 distance와 curvature를 바꿀 수 있다. 그러므로 interpolation/mesh는 숨은 구현 세부가 아니라 A-A가 세는 추가 구조다. Laplace-Beltrami heat는 reversible/self-adjoint이므로 arbitrary directed SCC도 그대로 기하 core로 볼 수 없다.

### R-C — 표현력은 크지만 두 번째

full SPD 연속장은 conformal route가 놓치는 anisotropy를 표현한다. 그러나 $L_\theta L_\theta^T+\epsilon I$와 eigenvalue clip이 SPD라는 사실만으로 Levi-Civita/curvature가 안정적으로 계산되는 것은 아니다. clip threshold를 건너면 $C^2$가 깨질 수 있으므로 smooth bounded parameterization 또는 threshold margin 검사가 필요하다.

최소 교차검증은 barrier 위치와 affine chart를 학습에서 빼고도, 명시적 tensor transform

$$
g_\theta^J(y)=J^{-T}g_\theta(J^{-1}(y-b))J^{-1}
$$

과 다섯 readout이 일치하는지 보는 것이다. spectral projection이나 $+\epsilon I$는 일반 affine 공변이 아니므로 canonical chart에서 최종 tensor를 완성한 뒤 그 tensor만 변환해야 한다. 예를 들어 $g=I$, $J=\operatorname{diag}(10,1)$에서 변환 후 lower clip $m=0.1$을 다시 하면 첫 방향 squared length가 1에서 10으로 바뀐다. 역할별 loss head나 latent state가 남으면 성능과 관계없이 one-$g$ 주장은 중단한다.

### R-B — 정확한 저차원 control, 일반 경로 아님

$g=e^{2u}I$인 평면에서는 $K=-e^{-2u}\Delta u$와 $L_g(\gamma)=\int e^u\lVert\dot\gamma\rVert dt$가 정확하다. 그래서 수식ㆍ수치 구현의 analytic control로 가치가 있다.

반면 $g_0$-orthonormal frame에서 relative eigenvalue가 항상 같아 anisotropy ratio가 1이다. $A=\operatorname{diag}(1,9)$를 $cI$로 근사하는 최선의 상대 Frobenius 오차도

$$
\min_c\frac{\lVert A-cI\rVert_F}{\lVert A\rVert_F}
=\sqrt{\frac{32}{82}}=0.624695\ldots
$$

로 0이 되지 않는다. 따라서 anisotropic barrier/memory가 필요한 순간 해상도를 늘리는 대신 경로를 기각해야 한다. $g_0$를 좌표변환 때 함께 운반하지 않거나 역할 readout의 두 번째 상태로 쓰는 것도 각각 covariance/one-$g$ 위반이다.

### R-D — sub-Riemannian의 허용선

좌표행렬 $g$의 고유벡터는 congruence 변환에서 보존되지 않으므로 이를 horizontal direction으로 쓰는 단순안은 즉시 기각한다. 살아남는 최소안은 $g$에서 내재적으로 계산되는 $\operatorname{Ric}^{\sharp}_g$의 spectral subbundle뿐이며, 그마저 uniform gap이 닫히면 중단한다. 별도 distribution $H$를 학습하면 점당 $k(d-k)$ dof의 두 번째 persistent geometric object가 생겨 현재 계약 밖이다.

또한 대칭 control set과 이차 metric 비용은

$$
d_{SR}(x,y)=d_{SR}(y,x)
$$

를 만족한다. 비가역 행동비용이 필요하면 drift, time-dependent metric, Finsler/Randers 구조 중 하나를 명시해 새 계약으로 넘어가야 한다.

## 3. 사전등록 killing-test 순서

1. canonical chart에서 SPD/eigenvalue bound와 condition certificate를 만든다.
2. 그 최종 tensor와 운반된 graph를 변환하며 affine local length 및 plan 상대오차 $\le10^{-10}$을 검사한다. 변환 chart에서 clipping/$+\epsilon I$를 재적용하지 않는다.
3. source-free symmetric fixture에서 유일 goal을 선택하지 않는지 검사.
4. identity 대 source-deformed barrier에서 하나의 $g$만 바꾸고 다섯 readout을 함께 검사.
5. 역할별 parameter/state/goal label을 직접 전달하면 성능과 무관하게 실패.
6. time-varying update는 metric-rate를 별도 검사한다. $\beta=\sup\operatorname{tr}_g\dot g<4\lambda$ 같은 충분조건 없이 fixed-$g$ energy 정리를 자동 상속하지 않는다.
7. R-A는 refinement, R-C는 hold-out chart/barrier, R-B는 anisotropy, R-D는 gap/bracket rank를 각각 추가 검사.

## 4. 최종 route decision

V15의 최소 구현 경로는 **R-A**로 고정한다. **R-C**는 R-A의 finite invariant certificate와 hidden-role audit가 통과한 뒤 연속ㆍanisotropic 확장으로만 연다. **R-B**는 저차원 analytic control로 유지하되 일반 one-$g$ 해결책으로 세지 않는다. **R-D**는 접근가능성 benchmark가 별도로 사전등록되기 전까지 구현하지 않는다.

이 판정은 route 선택일 뿐 완결ㆍ승격 판정이 아니다. 어느 경로의 synthetic test가 통과해도 AGI, 생물 뇌의 등가성, 우주론, 장기계획, continual learning, tool use, SCC continuum limit은 검증되지 않는다.
