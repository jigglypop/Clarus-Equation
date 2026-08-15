# AGI V15 Unified Metric Agent: 하나의 계량 상태를 공유하는 유한 기하 core

Status: COMPLETE

최종 판정: **하나의 SPD metric state에서 다섯 finite readout을 계산하는 기하 core를 구현했다. 정적 Riemannian metric 하나만으로 비가역 세계동역학이나 source-free 의미 목표를 생성한다는 강한 주장은 no-go 정리에 의해 성립하지 않는다. 현재 산출은 AGI가 아니라 다음 일반행동 과제를 위한 finite metric-graph primitive다.**

## 1. 초록

[모델 선택] 본 연구는 세계모형, 기억, 계획, 비평, 목표를 독립 persistent module로 두지 않고 하나의 정보다양체 metric $g_t$를 공유하게 하는 구조를 시험한다. [정리] tensor와 좌표를 함께 affine 변환하면 local quadratic length와 고정 graph의 metric path cost가 불변이며, $LL^T+\varepsilon I$와 fixed-chart spectral projection은 각각 SPD와 condition bound를 보장한다. [정리: no-go] 정적 Riemannian distance는 대칭이어서 비가역 world transition을 혼자 정하지 못하고, source-free 대칭 metric은 equivariant한 유일 목표를 선택하지 못한다. [구현 산출] 하나의 metric state에서 cost substrate, deformation memory, shortest-path planning, dimensionless surprise critic, tie-preserving goal readout을 계산하는 opt-in finite core를 만들었다. [수치 산출] 128개 비직교 affine trial에서 local·edge·path 최대 상대오차는 각각 $6.344\times10^{-16}$, $6.301\times10^{-16}$, $4.778\times10^{-16}$이었고 관련 회귀 묶음은 모두 완료됐다. [한계] full geodesic, connection, curvature, continuum SCC limit, 비가역 world dynamics, 생물학·우주론·AGI 효능은 구현하거나 검증하지 않았다.

## 2. 서론

이 절은 “다섯 기능이 하나의 리만기하학 평면이며 $g$에 의해서만 달라진다”는 아이디어를 검사 가능한 문장으로 바꾼다. [정의] 본체는 2차원 평면으로 제한하지 않는 $d$차원 정보다양체 $(\mathcal M,g_t)$다. 2차원은 시각화와 analytic control에만 사용한다. [모델 선택] 유한 구현은 $N$개 표본 $z_i\in\mathbb R^d$와 각 표본의 $g_i\in\operatorname{SPD}(d)$를 사용한다.

[구조 가설] 다섯 기능은 별도 persistent state를 가지지 않고 같은 $g$를 읽는다. 관측과 사용자 선호는 [공리: 외부 입력]으로 공통 metric update에 들어간다. 그 이후의 persistent semantic state는 $g$ 하나다. 이 구분은 “모든 것이 $g$로 저장된다”와 “모든 의미가 $g$에서 무입력으로 생긴다”를 분리한다. 전자는 finite baseline으로 구현했다. 후자는 목표 대칭 no-go 때문에 성립하지 않는다.

SCC 노드 수의 증가는 별도 정신의 무한 복제가 아니라 하나의 metric substrate를 더 조밀하게 표본화하는 후보 해석을 갖는다. [미완성] node 수 $N\to\infty$가 Riemannian atlas 또는 Laplace–Beltrami operator로 수렴한다는 결론에는 sampling, overlap, quadrature, operator consistency와 direct-limit compatibility가 더 필요하다.

## 3. 정의와 표기

이 절은 증명과 구현이 사용하는 객체를 정의한다. $G=(V,E)$는 $N<\infty$개의 node를 가진 고정된 대칭 연결 graph다. $z_i\in\mathbb R^d$, $d\ge2$는 한 chart의 node 좌표이고 $g_i$는 node $i$의 대칭 양의 정부호 covariant tensor다. 구현의 metric과 좌표는 기준 정보척도로 정규화한 무차원량이다.

[정의] affine chart change는 식 (1)이다.

$$
y=Jx+b,
\qquad
g_y=J^{-T}g_xJ^{-1},
\qquad J\in GL(d).
\tag{1}
$$

[정의] node $i$의 접벡터 $v$에 대한 local quadratic length는 식 (2)다.

$$
q_i(v)=v^Tg_iv.
\tag{2}
$$

[정의] 고정 edge $(i,j)$의 구현 비용은 endpoint metric의 산술평균으로 정한다.

$$
\bar g_{ij}=\frac{g_i+g_j}{2},
\qquad
\ell_{ij}=\sqrt{(z_j-z_i)^T\bar g_{ij}(z_j-z_i)}.
\tag{3}
$$

식 (3)은 finite metric-graph edge cost다. 일반적으로 연속 metric의 정확한 geodesic length라고 부르지 않는다.

[정의] prediction surprise와 hard gate는 식 (4)다.

$$
\delta_g^2=(z-\hat z)^Tg_i(z-\hat z),
\qquad
\chi=\mathbf 1\left[\frac{\delta_g^2}{\ell_0^2}>\theta\right].
\tag{4}
$$

$\ell_0>0$은 기준 정보거리다. 식 (4)의 threshold 인자는 무차원이며 경계에서는 strict inequality 때문에 $\chi=0$이다. Clarus-field의 gate와 metric 표기가 충돌하지 않도록 metric은 $g$, hard gate는 $\chi$로 쓴다.

## 4. 공리

이 절은 metric 하나만으로 나오지 않는 선택을 드러낸다.

1. **A1 — finite topology [모델 선택].** node와 대칭 연결 adjacency는 외부에서 고정한다. adjacency weight는 이번 edge 비용에 사용하지 않고 양의 값은 topology만 선언한다.
2. **A2 — 외부 metric source [외부 입력].** 관측과 선호는 transient SPD target tensor 또는 그 원자료로 들어온다. source의 의미가 올바르다는 결론은 이번 수학의 산출이 아니다.
3. **A3 — fixed-chart numerical bound [모델 선택].** 한 계산 chart에서 $0<m\le M<\infty$를 고정하고 source update 전에 metric 고유값을 $[m,M]$로 projection한다.
4. **A4 — 공통 update [모델 선택].** 역할별 update를 두지 않고 식 (5)의 하나의 source law를 사용한다.

$$
g^+=(1-\alpha)g+\alpha g_{\mathrm{source}},
\qquad 0\le\alpha\le1.
\tag{5}
$$

5. **A5 — goal candidate boundary [외부 입력].** goal readout의 후보집합과 현재 source node는 query로 들어온다. node label 자체를 선호 tie-break로 사용하지 않는다.

[미완성] source tensor를 원시 관측과 자연어 목표에서 학습하는 법칙은 아직 없다. 따라서 A2는 향후 agent 학습의 핵심 빈칸이다.

## 5. 정리와 증명

이 절은 coordinate covariance, SPD stability, fixed-metric field energy와 두 no-go를 증명한다.

### 5.1 affine readout 공변성

**정리 1 [정리].** 식 (1)에 따라 point, tangent vector와 metric을 함께 운반하면 식 (2)는 불변이다. 같은 adjacency를 운반하면 식 (3)의 모든 edge length와 그 합의 최솟값인 graph shortest cost도 불변이다.

**증명.** $v_y=Jv_x$를 식 (2)에 대입하면

$$
q_y(v_y)
=(Jv_x)^T(J^{-T}g_xJ^{-1})(Jv_x)
=v_x^Tg_xv_x.
\tag{6}
$$

edge displacement도 $z_j-z_i\mapsto J(z_j-z_i)$로 변하고 endpoint 평균은 tensor 변환과 선형으로 교환한다. 따라서 각 $\ell_{ij}$가 불변이다. 고정된 path의 비용은 edge length의 합이므로 불변이고, 같은 path 집합 위의 최솟값도 불변이다. □

[경계] chart 변환 후 Euclidean $k$-NN을 다시 계산하면 path 집합 자체가 달라질 수 있다. 정리 1은 adjacency transport에 대한 정리이며 neighbor reconstruction에 대한 정리가 아니다.

### 5.2 SPD와 condition bound

**정리 2 [정리].** $\varepsilon_g>0$이면 식 (7)의 $g$는 SPD이고 $\lambda_{\min}(g)\ge\varepsilon_g$다.

$$
g=LL^T+\varepsilon_gI.
\tag{7}
$$

**증명.** $v\ne0$에 대해

$$
v^Tgv=\lVert L^Tv\rVert_2^2+\varepsilon_g\lVert v\rVert_2^2>0.
\tag{8}
$$

Rayleigh quotient를 최소화하면 최소 고유값 하한도 나온다. □

**정리 3 [조건부 정리].** 한 고정 chart에서 대칭 metric의 고유값을 $[m,M]$로 clipping하면 결과는 SPD이고 condition number는 $M/m$ 이하이다. 식 (5)의 두 endpoint가 같은 Loewner bound $mI\preceq g,g_{\mathrm{source}}\preceq MI$를 만족하면 $g^+$도 같은 bound를 만족한다.

**증명.** clipping된 모든 고유값은 정의상 $[m,M]$에 있다. 식 (5)에 $mI$와 $MI$의 부등식을 각각 선형결합하면 같은 lower/upper bound가 보존된다. □

**반례 경계.** spectral clipping $P$는 일반적으로

$$
P(J^{-T}gJ^{-1})\ne J^{-T}P(g)J^{-1}
\tag{9}
$$

이다. $g=I$, $J=\operatorname{diag}(10,1)$인 수치 fixture에서 변환된 벡터의 length squared는 projection 전 1, projection 후 25였다. 따라서 정리 1의 affine readout과 정리 3의 fixed-chart 안정화는 서로 다른 certificate다.

### 5.3 fixed metric field energy

**정리 4 [조건부 정리].** 고정 Riemannian manifold에서 $\phi_0,r\in L^2(g)$이고 boundary energy flux가 0이며

$$
\partial_t\phi=\kappa\Delta_g\phi-\lambda\phi+r,
\qquad \lambda>0,
\tag{10}
$$

라고 하자. 충분히 정칙한 해는 식 (11)을 만족한다.

$$
\frac{d}{dt}\lVert\phi\rVert_{L^2(g)}^2
\le-\lambda\lVert\phi\rVert_{L^2(g)}^2
+\lambda^{-1}\lVert r\rVert_{L^2(g)}^2.
\tag{11}
$$

$\phi_0,r\ge0$이면 positivity도 보존된다.

**증명.** 식 (10)에 $\phi$를 곱해 적분하고 부분적분하면 diffusion 항은 $-\kappa\lVert\nabla_g\phi\rVert_2^2$다. source 항에 Young 부등식

$$
2\langle r,\phi\rangle
\le\lambda^{-1}\lVert r\rVert_2^2+\lambda\lVert\phi\rVert_2^2
$$

을 적용하면 식 (11)이 나온다. Gronwall 부등식이 energy bound를 준다. heat semigroup의 positivity가 비음 초기값과 source를 보존한다. □

[반례 경계] 완비 비콤팩트 $\mathbb R^d$에서 bounded source $r=1$은 $L^2$가 아니다. $phi_0=0$이면 해가 $(1-e^{-\lambda t})/\lambda$인 공간상 상수라 모든 $t>0$에서 $L^2$ norm이 무한대다. 따라서 “완비+점별 bounded source”라는 넓은 부모 범위는 활성 주장에 남기지 않았다.

[시간가변 경계] $g=g_t$에서는

$$
\partial_td\mu_{g_t}
=\frac12\operatorname{tr}_{g_t}(\dot g_t)d\mu_{g_t}
\tag{12}
$$

가 energy에 volume-rate 항을 추가한다. fixed-$g$ 정리는 metric update에 자동 상속되지 않는다. $\operatorname*{ess\,sup}\operatorname{tr}_g\dot g\le\beta<4\lambda$ 같은 별도 조건이 한 충분조건 후보지만 이번 코드에는 PDE를 구현하지 않았다.

### 5.4 source-free goal no-go

**정리 5 [no-go 정리].** 후보집합 $C$에서 한 점을 반환하는 selector $F(g)$가 metric isometry에 equivariant하다고 하자. $C$의 후보들을 교환하고 어느 후보도 고정하지 않는 isometry $\varphi$가 있으면 source-free singleton $F(g)$는 존재하지 않는다.

**증명.** $\varphi^*g=g$이면 equivariance는

$$
F(g)=F(\varphi^*g)=\varphi^{-1}F(g)
\tag{13}
$$

를 요구한다. 식 (13)은 $F(g)$가 $\varphi$의 고정점이어야 함을 뜻하지만 가정상 후보 고정점이 없다. 모순이다. □

따라서 대칭 graph에서 허용되는 invariant 출력은 모든 minimizer의 집합이다. 한 점을 선택하려면 A2 또는 A5 같은 symmetry-breaking 입력이 필요하다.

### 5.5 정적 metric 방향성 no-go

**정리 6 [no-go 정리].** 정적 Riemannian metric의 distance는 대칭이다. 따라서 그 distance만으로 비가역 world transition을 고유하게 결정할 수 없다.

**증명.** curve $\gamma(t)$의 역경로를 $\bar\gamma(t)=\gamma(1-t)$라 하면 $\dot{\bar\gamma}$의 부호만 바뀐다. metric length의 quadratic form은 부호에 불변이므로 $L_g(\bar\gamma)=L_g(\gamma)$다. 양방향 path에 대한 infimum을 취하면

$$
d_g(x,y)=d_g(y,x).
\tag{14}
$$

비가역 transition은 대칭 distance만으로 복원되지 않는다. □

drift, time-dependent metric, asymmetric control, Finsler/Randers 항 또는 외생 time orientation이 필요하다. 이 가운데 어느 하나를 도입하면 정적 $g$ 하나의 산출이라고 부르지 않고 별도 공리로 기록해야 한다.

## 6. 구현 산출

이 절은 정리 1–3과 공리 A1–A5를 finite 코드에 대응시킨다. [구현 산출] `UnifiedMetricState`의 field는 `metric` 하나다. immutable point/topology와 transient query는 persistent semantic state로 세지 않는다. [구현 산출] certificate는 persistent state field count 1과 role parameter count 0을 출력한다.

다섯 사용자 개념의 현재 정확한 의미는 다음과 같다.

| 개념 | 하나의 $g$에서 읽는 값 | 구현 범위 |
|---|---|---|
| world | 식 (3)의 edge cost | metric-cost substrate; 미래예측기 아님 |
| memory | $g_t-g_{t_0}$ | source deformation; 별도 memory state 없음 |
| planning | edge cost 합의 최솟값 | finite graph shortest path |
| critic | 식 (4)의 surprise | local hard gate |
| goal | candidate path cost의 argmin 집합 | 모든 tie 보존; 의미 source는 외부 입력 |

[구현 산출] `MetricPath`는 path가 비유일할 때 `unique=False`를 반환한다. path sequence는 재현을 위한 lowest-index representative일 뿐 goal preference가 아니다. `MetricGoalReadout`은 후보 순서와 무관하게 모든 minimizer를 반환한다.

[구현 산출] 공개 certificate는 `geometry_scope=finite-point-local-quadratic+metric-graph`, `world_scope=metric_cost_substrate`를 기록한다. projection affine covariance, full geodesic, connection, curvature, heat kernel, continuum, irreversible dynamics, AGI·생물학·우주론 evidence는 모두 false다.

## 7. 수치 검증과 내부 비교

이 절은 계산 구현의 정합성을 계량하며 AGI task 성능과 분리한다. seed 150013의 128개 affine trial에서 SPD tensor와 invertible non-orthogonal Jacobian을 생성했다. adjacency는 재구성하지 않고 운반했다.

| invariant readout | 최대 상대오차 | 사전 허용치 |
|---|---:|---:|
| local quadratic | $6.344\times10^{-16}$ | $10^{-10}$ |
| finite edge | $6.301\times10^{-16}$ | $10^{-10}$ |
| shortest cost | $4.778\times10^{-16}$ | $10^{-10}$ |

identity diamond graph에서는 candidate 1과 2가 모두 goal minimizer였고 source-to-target path도 비유일했다. node 1에 $4I$ barrier를 적용하면 minimizer는 node 2, representative plan은 $(0,2,3)$, cost는 $2\sqrt2$였다. 같은 deformation은 node 1의 memory delta와 critic surprise도 동시에 증가시켰다.

source projection fixture에서 결과 metric의 고유값 범위는 $[0.875,1.25]$, condition number는 $1.4285714286$이었다. projection no-go fixture의 covariance defect 24도 음성 대조로 남겼다.

회귀검사는 다음 범위에서 수행했다.

| 검사 범위 | 결과 |
|---|---:|
| V15 focused | 17 passed |
| dimensionless + Clarus-field + V15 | 46 passed |
| SCC compatibility | 114 passed |
| runtime/public compatibility | 34 passed, 기존 warning 2 |
| CE core slice | 89 passed, 기존 warning 2 |
| local-cloud compatibility | 89 passed |
| geometry compatibility | 40 passed, 1 skipped |

새 구현·검사·예시·artifact의 정적 검사와 compile은 완료됐다. 저장소 전체 suite는 기존 dirty fixture/policy 실패가 알려져 있어 이번 범위에서 재실행하지 않았다. CE의 별도 constants 하네스는 bootstrap residual $2.08\times10^{-17}$, 12개 scored row 중 11개가 1-sigma 이내이고 1개가 CAUTION인 기존 상태를 재현했다. 이 결과는 V15와 독립이다.

## 8. SCC 무한재귀와 리만 구조의 관계

이 절은 기존 infinite-SCC 목표에 V15가 무엇을 추가하는지 구분한다. [모델 선택] finite SCC node는 하나의 metric graph의 sample 또는 chart candidate로 해석할 수 있다. node 수가 증가하면 동일한 $g$의 수치 해상도를 증가시키는 방향을 시험할 수 있다. 이는 동일 agent를 무한 복제하는 것과 다르다.

[정리: 선행 조건부] direct-limit update에는 level embedding $J_n$과 update $F_n$의 exact compatibility

$$
J_nF_n=F_{n+1}J_n
\tag{15}
$$

가 필요하다. V15 metric에도 추가로 tensor transport와 graph operator consistency가 필요하다. finite sample metric은 continuum을 유일하게 식별하지 않는다. 실제로 endpoint의 metric 값이 같은 두 smooth conformal interpolation이 거리 2.0000과 1.02936을 줄 수 있다.

[미완성] 다음 항목이 증명·검증되기 전에는 $N\to\infty$를 Riemannian continuum으로 부르지 않는다.

- fill distance $h_N\to0$와 sampling density,
- chart overlap transition과 cocycle,
- adjacency·quadrature의 일관된 운반,
- graph distance/heat/transport의 수렴,
- 식 (15)의 metric-state/update compatibility.

따라서 무한 SCC는 AGI의 원천이 아니라, 유한 일반 agent가 이미 가진 기하 상태를 임의 정밀도로 근사하는 후보 계산 구조다.

## 9. AGI 지위와 다음 단계

이 절은 구현 성과를 AGI 목표에 연결하되 승격하지 않는다. [산출] V15는 별도 역할 state를 하나의 공유 metric state로 줄였다. [미완성] V15는 아직 관측에서 metric source를 학습하지 않으며, 행동을 선택해 환경을 바꾸고 새 관측으로 metric을 갱신하는 closed agent loop가 없다. [미완성] 비가역 transition, temporal credit assignment, continual learning, tool use, compositional OOD와 long-horizon planning 성능도 채점하지 않았다.

다음 AGI 단계는 `V16 Metric-Source General Agent`로 분리해야 한다. 첫 과제는 source law를 학습하되 persistent 역할 head를 새로 만들지 않는 것이다. 최소 loop는 식 (16)이다.

$$
o_t,a_{t-1},g_t
\xrightarrow{\ U_\theta\ }
g_{t+1}
\xrightarrow{\ \text{five readouts}\ }
(\hat o_{t+1},m_t,\gamma_t,c_t,G_t)
\xrightarrow{\ \pi\ }a_t.
\tag{16}
$$

$U_\theta$와 policy $\pi$는 새로 필요한 동역학이다. 이들이 공통 $g$를 쓰더라도 “$g$만으로 무입력 산출”은 아니다. 사전등록 benchmark는 memory horizon, latent dynamics, counterfactual planning, compositional OOD, continual task switch를 하나의 model checkpoint에서 평가하고 GRU, Transformer, retrieval agent와 compute를 맞춰야 한다.

## 10. 관측 비교

이 run은 외부 관측값과 외부 데이터셋을 사용하지 않았다. 따라서 생물학·우주론 관측 비교를 수행하지 않는다. metric을 뇌 또는 시공간의 실제 물리 metric으로 동일시하는 단계는 [공리: 물리 사상]이며 이번 run에는 채택하지 않았다.

## 11. 미완성 과제와 한계

이 절은 현재 certificate 밖의 항목을 명시한다. 열린 P0는 없다. 감사의 P1은 구현 명칭과 false certificate로 경계를 고정했지만 연구 과제 자체는 남는다.

- [미완성] 원시 관측·자연어 목표에서 외부 metric source를 학습하는 공통 update law.
- [미완성] drift 또는 time-dependent metric을 포함하는 비가역 world dynamics와 별도 안정성 정리.
- [미완성] continuous $C^2$ SPD field의 connection, curvature, numerical geodesic과 regularity 검증.
- [미완성] SCC refinement에서 distance, heat, transport와 update가 수렴하는 continuum theorem.
- [미완성] strong baseline과 compute-matched AGI multi-task 채점.
- [한계] spectral projection은 fixed-chart 안정화이며 affine-natural update가 아니다.
- [한계] 하나의 $g$는 parameter count를 줄이는 구조 제약이지 충분한 지능의 증명이 아니다.
- [한계] 외부 source가 없으면 대칭 목표를 유일하게 선택할 수 없다.

경로 선택에서는 R-A finite metric graph만 구현했다. learned full SPD R-C는 두 번째 후보다. conformal 2D R-B는 analytic control로만 남긴다. sub-Riemannian R-D는 intrinsic spectral gap과 accessibility benchmark가 별도로 필요하다.

## 12. 재현성

이 절은 검산 명령과 artifact 경로를 제공한다. 저장소 루트는 `C:/Users/dongh/OneDrive/Desktop/Clarus-Equation`이고 run은 `_workspace/ce/agi-v15-unified-metric-20260813`이다.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_unified_metric.py -q -p no:cacheprovider
.venv\Scripts\python.exe examples/agi/unified_metric_demo.py
.venv\Scripts\python.exe _workspace/ce/agi-v15-unified-metric-20260813/artifacts/verify_unified_metric_math.py
.venv\Scripts\python.exe _workspace/ce/agi-v15-unified-metric-20260813/artifacts/validate_unified_metric.py
```

수학 반례 로그는 `artifacts/verify_unified_metric_math.log`, 128-trial 계량 로그는 `artifacts/validate_unified_metric.log`, 전체 명령과 결과는 `31-validation.md`에 있다.

## 13. 참조

외부 문헌과 외부 데이터는 사용하지 않았다. 내부 근거는 본 run의 `00-contract.md`, `11-math.md`, `12-routes.md`, `20-audit.md`, `30-implementation.md`, `31-validation.md`와 선행 `_workspace/ce/agi-clarus-field-20260812`, `docs/7_AGI/28_Nested_Infinite_SCC_V9.md`다.

Status: COMPLETE
