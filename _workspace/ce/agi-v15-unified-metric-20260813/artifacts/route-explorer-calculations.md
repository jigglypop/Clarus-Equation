# V15 one-$g$ route explorer: detailed calculations

Status: COMPLETE

이 문서는 `12-routes.md`의 판정을 뒷받침하는 계산 노트다. 외부 관측값이나 외부 데이터는 쓰지 않았고, 모든 후보는 계약에 이미 들어 있는 외부 source/boundary 공리를 공유한다. 아래의 후보별 공리는 그 공리에 더하는 공리이며 각 후보당 최대 하나다.

## 1. 고정 표적과 공통 자유도

구조 표적은 다음 한 문장으로 고정한다. 유한 표본과 연속 다양체 수준에서, barrier 안의 양의 변형 $\Delta g\succeq0$는 국소 선요소를 감소시키지 않아야 하고, worldㆍmemoryㆍplanningㆍcriticㆍgoal의 다섯 readout은 역할별 상태나 역할별 가중치 없이 같은 균일 타원 metric $g_t$에서만 계산되어야 한다. 보편적인 수치 효과크기는 주장하지 않으며, 좌표 등가 오차와 plan 상대 오차는 계약의 $10^{-10}$을 그대로 쓴다.

계약이 강제하는 부분은 SPD와 $[m,M]$ 고유값 경계, tensor 변환법칙, geodesic surprise hard gate, 외부 source/boundary가 먼저 $g_t$에 기록된 뒤 역할별 상태가 남지 않는다는 조건, 대칭 공간에서 source 없이 유일 goal을 선택하지 못한다는 no-go다. 경로가 선택해야 하는 부분은 표본화/보간, metric family, 공통 update law, 연결ㆍ열핵의 수치근사, 그리고 하나의 사전등록된 불변 goal scalar다.

각 경로의 `state dof`와 `search dof`를 분리한다. $N$개 표본과 차원 $d$에서 full SPD field의 표본 상태 자유도는

$$
q_{\mathrm{SPD}}=N\frac{d(d+1)}2.
$$

이 수는 다섯 역할별로 곱하지 않는다. 후보를 본 뒤 bandwidth, flow, goal scalar, network 폭을 고르는 행위는 별도의 look-elsewhere 자유도다.

## 2. 공통 불변량과 두 no-go

### 2.1 affine edge-length 불변성

$y=Jx+b$, $g_y=J^{-T}g_xJ^{-1}$이고 $\Delta y=J\Delta x$라 하자. endpoint tensor의 산술평균을 $\bar g_{ij}=(g_i+g_j)/2$라 하면

$$
\ell_{ij}^2=\Delta x_{ij}^{T}\bar g_{ij}\Delta x_{ij}
$$

이고

$$
\begin{aligned}
(\ell_{ij}')^2
&=(J\Delta x_{ij})^T(J^{-T}\bar g_{ij}J^{-1})(J\Delta x_{ij})\\
&=\Delta x_{ij}^T\bar g_{ij}\Delta x_{ij}=\ell_{ij}^2.
\end{aligned}
$$

따라서 같은 edge set을 변환하면 edge length와 그 합의 최솟값인 graph shortest path가 정확히 불변이다. 반대로 변환 뒤 Euclidean $k$-NN을 다시 만들면 edge set 자체가 달라질 수 있다. 따라서 affine 시험은 adjacency도 함께 운반하는 시험과 adjacency 재구성 시험을 구분해야 한다.

이 정확성은 **완성된 tensor를 그대로 변환할 때만** 성립한다. spectral clipping이나 $+\epsilon I$를 좌표마다 다시 적용하는 연산은 일반 affine 공변이 아니다. 예를 들어 $g_x=I$, $J=\operatorname{diag}(10,1)$이면 정확한 변환은 $g_y=\operatorname{diag}(0.01,1)$이다. $\Delta x=e_1$, $\Delta y=10e_1$에 대해 exact squared length는 양쪽 모두 1이다. 그런데 $y$-chart에서 lower clip $m=0.1$을 다시 적용하면 $\widetilde g_y=\operatorname{diag}(0.1,1)$이고

$$
\Delta y^T\widetilde g_y\Delta y=10.
$$

따라서 projection certificate는 canonical chart의 SPD/bound certificate이고, covariance test에서는 canonical chart에서 projection을 끝낸 **최종 tensor**만 변환해야 한다. $+\epsilon I$도 $I$ 대신 함께 변환되는 reference tensor를 쓰지 않으면 같은 문제가 생긴다.

### 2.2 source-free unique-goal no-go

goal readout $F$가 metric isometry $\varphi$에 대해 equivariant라면

$$
F(\varphi^*g)=\varphi^{-1}F(g).
$$

$\varphi^*g=g$인 경우 $F(g)=\varphi^{-1}F(g)$여야 한다. 고정점 없는 isometry가 존재하거나 후보점들을 비자명하게 순열하면 한 점짜리 $F(g)$는 존재할 수 없다. 그러므로 대칭 fixture에서 유일 goal이 나오면 hidden tie-break, node label, source 또는 boundary가 들어간 것이다. 모든 경로에 동일한 killing test로 쓴다.

### 2.3 방향성 no-go

Riemannian curve length는

$$
L_g(\gamma)=\int_0^1\sqrt{g_{\gamma(t)}(\dot\gamma(t),\dot\gamma(t))}\,dt
$$

이고 역경로 $\bar\gamma(t)=\gamma(1-t)$도 같은 길이를 갖는다. 따라서 정적 metric에서는

$$
d_g(x,y)=d_g(y,x).
$$

대칭 control set $u\leftrightarrow-u$와 이차 비용을 쓰는 sub-Riemannian 거리도 같다. forward/backward 비용 차이를 주장하려면 drift, time-dependent metric, Finsler/Randers 항 가운데 하나가 더 필요하다. 그것을 넣은 순간 정적 one-$g$만으로 방향성을 얻었다는 주장은 중단한다.

### 2.4 time-varying heat의 metric-rate 항

compact domain의 compatible boundary condition에서

$$
\partial_t\phi=\kappa\Delta_{g_t}\phi-\lambda\phi+r
$$

와 $E(t)=\frac12\int\phi^2d\mu_{g_t}$를 쓰면

$$
E'=-\kappa\lVert\nabla\phi\rVert_{L^2(g_t)}^2-\lambda\lVert\phi\rVert_{L^2(g_t)}^2
+\langle\phi,r\rangle_{L^2(g_t)}
+\frac14\int\operatorname{tr}_{g_t}(\dot g_t)\phi^2d\mu_{g_t}.
$$

$\beta=\sup_{t,z}\operatorname{tr}_{g_t}\dot g_t<4\lambda$이면 effective damping $a=\lambda-\beta/4>0$을 얻는다. 이 항을 빼고 fixed-$g$ energy를 복사하면 탈락이다. 또한 noncompact complete domain에서 bounded $r$만으로 $L^2$ 결론을 내릴 수 없다. $\mathbb R^d$의 $r=1$은 bounded지만 정상해 $1/\lambda$가 $L^2$가 아니기 때문이다.

## 3. R-A: discrete atlas / point cloud

### 3.1 유일한 추가 공리 A-A

**A-A (sampling-atlas consistency).** 표본열은 fill distance $h_N\to0$인 shape-regular cover를 이루고, overlap transition은 cocycle를 만족하며, 보간ㆍquadrature와 reversible/symmetric adjacency는 같은 cover에서 일관되게 운반된다.

이 공리가 없으면 유한 graph 구현은 가능하지만 UM-5의 continuum/atlas 해석은 미완성이다.

유한한 node tensor 값은 continuum field를 식별하지 못한다. 실제로 node가 없는 open interval에 지지되는 smooth bump $u\ge0$, $u\not\equiv0$를 잡으면 $g^{(0)}=dx^2$와 $g^{(1)}=e^{2u(x)}dx^2$는 모든 저장 node에서 같지만 그 interval을 가로지르는 거리는

$$
d_{g^{(1)}}(a,b)=\int_a^b e^{u(x)}dx>b-a=d_{g^{(0)}}(a,b)
$$

로 다르다. 따라서 interpolation/mesh는 단순 구현 세부가 아니라 finite-to-continuum 주장에 필요한 추가 구조다. 또한 Laplace-Beltrami heat는 volume measure에 대해 self-adjoint/reversible이므로 arbitrary directed SCC를 그대로 그 이산화라고 부를 수 없다. directed SCC를 쓰려면 reversible core로 제한하거나 방향 dynamics를 별도 상태로 명시해야 한다.

### 3.2 최소 구성

- 상태: 각 표본의 $g_i\in\operatorname{SPD}(d)$, 총 $N d(d+1)/2$ dof.
- world: $\ell_{ij}$로 만든 density-corrected graph Laplacian과 graph heat kernel. 국소 connection/parallel transport는 overlap transition과 metric 차분에서 근사한다.
- memory: source가 만든 $g_i(t)-g_i(t_0)$의 지속 변형 및 loop transport/holonomy. 별도 memory vector는 금지한다.
- planning: $\ell_{ij}$의 Dijkstra/A* 최솟값. heuristic을 쓰면 metric lower bound만 써야 한다.
- critic: $d_{G_t}(z_{t+1},\hat z_{t+1})^2/\ell_0^2$와 discrete geodesic-deviation residual.
- goal: 공통으로 사전등록한 invariant scalar $\Phi[g]$의 basin. node id나 목표 label을 readout에 직접 넣지 않는다.

### 3.3 자유도와 교차예측

필수 상태 dof는 $N d(d+1)/2$다. 계약 밖의 최소 연속 search dof는 bandwidth 또는 neighbor scale 하나다. neighbor 규칙, edge quadrature, graph-Laplacian normalization, goal scalar를 각각 여러 개 시험하면 그 개수가 그대로 categorical look-elsewhere가 된다. 이 네 선택은 첫 실행 전에 하나씩 고정해야 한다.

한 barrier 변형 $g\mapsto g+\Delta g$, $\Delta g\succeq0$가 주어지면 다음 교차예측을 동시에 낸다.

1. barrier를 가로지르는 모든 고정 edge의 $\ell_{ij}$는 감소하지 않는다.
2. shortest-path cost는 고정 graph에서 감소하지 않지만, 경로 자체는 barrier 밖으로 우회할 수 있다.
3. graph heat kernel과 critic residual도 같은 edge weight 변화에 반응해야 한다.
4. memory readout은 정확히 같은 $\Delta g_i$를 보고해야 한다.

### 3.4 killing tests

- 운반된 adjacency에서 affine local-length 오차 또는 plan 상대오차가 $10^{-10}$을 넘으면 즉시 탈락.
- $h_N\downarrow0$에서 distance, heat, transport가 정한 기준해로 수렴하지 않거나 atlas overlap에서 불연속이면 continuum 해석 탈락.
- finite $g_i$만으로 interpolation을 숨긴 채 continuum connection/curvature/distance가 유일하다고 주장하면 탈락.
- detailed balance가 없는 directed SCC를 Laplace-Beltrami discretization이라고 부르면 continuum geometry 주장이 탈락.
- metric을 고정한 채 역할별 weight만 바꾸어 barrier 통과 여부가 바뀌면 one-$g$ 주장이 탈락.
- 대칭 graph에서 node index로 유일 goal을 고르면 goal readout 탈락.
- graph가 연결인데 표본 refinement 때문에 artificial disconnection이 지속되면 해당 sampling rule 탈락.

## 4. R-B: conformal plane / curvature flow

### 4.1 유일한 추가 공리 A-B

**A-B (fixed conformal class).** $d=2$이고 runtime metric은 사전 고정된 reference tensor $g_0$의 conformal class $g=e^{2u}g_0$ 안에 머문다. $g_0$는 parameterization에만 쓰고 다섯 readout에는 인스턴스화된 $g$만 전달한다.

$g_0$를 역할 readout의 추가 상태로 쓰면 one-$g$ 경계 위반이다. 좌표변환 때 $g_0$도 tensor로 운반해야 하며, 매 chart에서 다시 $I$로 놓으면 covariance가 깨진다.

### 4.2 평면에서의 정확식

$g_0=I$인 chart에서

$$
g_{ij}=e^{2u}\delta_{ij},\qquad g^{ij}=e^{-2u}\delta^{ij},\qquad d\mu_g=e^{2u}dx.
$$

Levi-Civita connection, scalar Laplacian, Gaussian curvature는

$$
\Gamma^k_{ij}=\delta^k_i\partial_j u+\delta^k_j\partial_i u-\delta_{ij}\partial^k u,
$$

$$
\Delta_g f=e^{-2u}\Delta f,\qquad K_g=-e^{-2u}\Delta u.
$$

계획 비용은

$$
L_g(\gamma)=\int_0^1 e^{u(\gamma(t))}\lVert\dot\gamma(t)\rVert_2\,dt.
$$

따라서 scalar field $u$ 하나로 connection/heat, 지속 변형 $u_t-u_0$, planning length, curvature critic를 읽을 수 있다. normalized curvature flow의 후보식은

$$
\partial_tu=-(K_g-\bar K_g)+s_g,
$$

이며 $s_g$는 계약의 외부 source가 metric에 들어오는 공통 항이다. 이 식의 안정성을 fixed-$g$ heat energy에서 자동 상속했다고 주장해서는 안 된다.

### 4.3 표현력 하한

$N$개 표본에서 상태 dof는 $N$으로 full SPD의 $3N$보다 작다. 그러나 $g_0$-orthonormal frame에서 모든 상대 고유값이 같으므로 anisotropy ratio는 항상 1이다. 목표 tensor가 $A=\operatorname{diag}(1,r)$이고 conformal 근사 $cI$를 쓰면 Frobenius 최적값은 $c=(1+r)/2$이고

$$
\min_c\frac{\lVert A-cI\rVert_F}{\lVert A\rVert_F}
=\frac{|r-1|}{\sqrt{2(1+r^2)}}.
$$

$r=9$이면 하한은

$$
\sqrt{\frac{32}{82}}=0.624695\ldots.
$$

즉 9:1 anisotropic barrier fixture를 conformal family로 맞추는 것은 수치해상도 문제가 아니라 구조적으로 불가능하다.

### 4.4 killing tests

- 상대 tensor $g_0^{-1}g$의 condition number가 1이 아닌 ground-truth fixture를 요구하면 이 경로는 즉시 탈락.
- 일반 affine 변환 뒤 $g_0$를 운반하지 않고 $e^{2u}I$로 재투영해 $10^{-10}$ covariance를 넘으면 탈락.
- Ricci/curvature-flow step 뒤 $[m,M]$ 경계를 벗어나거나 metric-rate 조건 없이 fixed-metric energy 감소를 주장하면 해당 안정성 주장은 탈락.
- 같은 $u$인데 역할별 scalar field가 따로 생기면 one-$g$ 주장이 탈락.

search dof는 flow variant와 source regularization을 사후 선택하는 경우 늘어난다. 시간 단위를 고정해 주 flow coefficient를 1로 두고, flow 하나와 goal scalar 하나를 사전등록하지 않으면 이 경로는 낮은 상태 dof의 이점을 잃는다.

## 5. R-C: continuous learned full SPD

### 5.1 유일한 추가 공리 A-C

**A-C (regularity).** 계산 영역에서 $g_\theta$는 $C^2$이고, 계약의 균일 ellipticity와 함께 1ㆍ2차 미분이 유계다.

Levi-Civita connection에는 1차 미분, curvature에는 2차 미분이 필요하므로 단순히 표본별 SPD라는 사실만으로는 world/critic readout이 정의되지 않는다.

### 5.2 최소 구성과 dof

계약의

$$
g_\theta(z)=L_\theta(z)L_\theta(z)^T+\epsilon_gI
$$

또는 smooth bounded parameterization을 쓴다. network parameter가 $p$개이면 모델 search/state capacity는 $p$ dof이고, 표본 출력은 $N d(d+1)/2$개다. 다섯 역할별 head는 0개여야 한다. 좌표변환한 field는 재학습하지 않고

$$
g_\theta^J(y)=J^{-T}g_\theta(J^{-1}(y-b))J^{-1}
$$

로 정의한다.

world는 자동미분한 connection/heat operator, memory는 source update가 남긴 $g_{\theta,t}-g_{\theta,t_0}$와 parallel transport, planning은 수치 geodesic, critic은 geodesic residual/curvature, goal은 같은 사전등록 $\Phi[g_\theta]$를 쓴다.

hard spectral clipping은 고유값이 clip threshold를 지나는 곳에서 일반적으로 $C^2$가 아니다. 또한 clipping과 좌표마다 다시 적용하는 $+\epsilon_gI$는 일반 affine 공변이 아니다. 따라서 canonical chart에서 smooth bounded tensor를 완성한 뒤 그 최종 tensor를 명시적으로 변환해야 한다. clipping된 tensor로 SPD test는 통과해도 connection/curvature/covariance test는 실패할 수 있다. smooth bounded parameterization을 쓰거나 threshold와의 양의 margin을 별도로 확인해야 한다.

### 5.3 killing tests

- $L L^T$만 확인하고 derivative regularity를 확인하지 못하면 curvature readout은 미완성.
- chart를 변환한 hold-out에서 explicit tensor transform과 network readout의 local length/plan이 $10^{-10}$ 안에 맞지 않으면 covariance 탈락.
- transformed chart에서 spectral clipping 또는 $+\epsilon I$를 다시 적용해 exact tensor transform과 달라지면 general-affine 주장이 탈락.
- barrier 위치와 affine chart를 hold-out했을 때만 실패하면 synthetic fixture memorization이며 일반 manifold 주장이 탈락.
- network에 world/memory/critic/goal별 loss head 또는 latent state가 남으면 one-$g$ 주장이 탈락.
- source update 뒤 다섯 readout 가운데 하나가 별도 parameter 없이는 변하지 않으면 shared-field cross-prediction 탈락.

architecture, 폭/깊이, optimizer, loss weight, projection 방식을 둘 이상 비교하면 look-elsewhere는 매우 커진다. $p$만 보고 저자유도라 할 수 없으며, 시도한 architecture 수 $H$와 seed 수를 함께 기록해야 한다.

## 6. R-D: $g$-derived sub-Riemannian diagnostic

### 6.1 raw matrix-eigenvector 구성의 반례

좌표행렬 $g$의 고유벡터는 tensor 불변량이 아니다. $g=I$도 비직교 $J$ 아래에서는 $g'=J^{-T}J^{-1}$이 되어 좌표행렬 고유방향이 달라진다. 더 강하게 모든 SPD form은 한 좌표에서 $I$로 만들 수 있다. 따라서 `가장 작은 eigenvalue의 좌표 고유벡터`로 horizontal distribution을 정하는 규칙은 UM-1을 통과할 수 없다.

### 6.2 유일한 추가 공리 A-D와 내재적 대안

**A-D (intrinsic spectral gap).** $d\ge3$에서 intrinsic self-adjoint endomorphism $A_g=\operatorname{Ric}_g^{\sharp}$의 사전 고정된 $k$-dimensional spectral cluster와 나머지 spectrum 사이에 영역 전체의 양의 gap이 있다.

이때 $H_g$를 그 spectral subbundle로 잡으면 좌표 공변이고 별도 distribution state가 없다. gap이 닫히면 $H_g$가 불연속일 수 있으므로 경로를 중단한다. $d=2$에서는 $\operatorname{Ric}^{\sharp}=K I$라 일반적으로 방향을 고를 수 없다는 점도 즉시 경계다.

독립 distribution $H$를 따로 학습하면 점당 Grassmann 자유도 $k(d-k)$가 추가된다.

$$
q_H=N k(d-k).
$$

이는 $g$에서 유도되지 않은 두 번째 persistent geometric object이므로 현재 one-$g$ 주장을 벗어난다.

### 6.3 최소 구성과 killing tests

수평 곡선 $\dot z\in H_g$만 허용하고 $g|_{H_g}$의 이차 비용을 쓴다. world는 sub-Laplacian/hypoelliptic heat, memory는 horizontal holonomy, planning은 sub-Riemannian distance, critic은 horizontal prediction residual, goal은 같은 $\Phi[g]$로 읽는다. 상태 dof는 full SPD route와 같고 추가 연속 dof는 0이지만, $k\in\{1,\ldots,d-1\}$를 훑으면 $d-1$개의 categorical 시도가 생긴다.

- intrinsic spectral gap의 최솟값이 0 또는 수치오차 이하이면 탈락.
- Lie brackets가 full tangent rank를 만들지 못해 연결된 점 사이 거리가 무한대로 남으면 일반 planning route 탈락.
- forward/backward 비용이 다르다고 보고하면 대칭 control no-go와 충돌하므로 탈락.
- raw matrix eigenvectors, fixed coordinate axes, 별도 learned $H$를 쓰면 one-$g$/covariance 주장 탈락.

이 경로는 curvature의 2차 미분과 bracket 계산까지 필요하므로 네 후보 중 구현 난이도가 가장 높다.

## 7. 공통 최소 시험 순서

1. SPD/eigenvalue certificate.
2. canonical chart에서 projection을 끝낸 최종 tensor와 transported adjacency를 사용한 affine local-length $10^{-10}$ 시험. transformed chart에서 projection/$+\epsilon I$를 재적용하지 않는다.
3. identity metric 대 source-deformed anisotropic barrier ablation.
4. 같은 $g$를 입력으로 다섯 readout을 한 번에 계산하고 역할별 state/weight가 0개임을 검사.
5. source-free symmetric-goal no-go regression.
6. time-varying update에서는 $\partial_tg$ 또는 이산 metric-rate를 별도 기록하고 $\beta=\sup\operatorname{tr}_g\dot g<4\lambda$ 같은 충분조건 없이 fixed-$g$ stability를 자동 상속하지 않음.
7. atlas refinement 또는 hold-out chart/barrier 위치에서 교차예측을 재검사.

이 순서의 통과는 finite implementation certificate일 뿐 AGI, 생물학적 뇌, 우주론, SCC continuum limit의 검증이 아니다.
