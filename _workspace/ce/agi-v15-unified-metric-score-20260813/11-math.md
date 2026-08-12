# AGI V15 unified metric: independent mathematical verification

이 문서는 `00-contract.md`의 F1--F5만 독립적으로 검산한다. 선행 run의
결론은 증거로 상속하지 않았다. 동결된 `unified_metric.py`에서 실제로 사용한
연산을 먼저 다음과 같이 고정했다.

- 노드 $i$의 점과 계량은 각각 $z_i\in\mathbb R^d$와
  $g_i\in\mathbb S_{++}^d$이다.
- 국소 제곱길이는 $q_i(v)=v^Tg_iv$이다.
- 무향 edge $\{i,j\}$의 길이는

$$
\ell_{ij}=\sqrt{(z_j-z_i)^T\frac{g_i+g_j}{2}(z_j-z_i)}.
$$

- vertex path $P=(v_0,\ldots,v_k)$의 비용은
  $C(P)=\sum_{r=0}^{k-1}\ell_{v_rv_{r+1}}$이고, 최단 비용은
  $D(s,t)=\min_{P:s\leadsto t}C(P)$이다.
- `make_state(..., project=False)`는 SPD 여부만 검사한다.
  `project_metric`은 현재 좌표계에서 고윳값을 $[1/4,4]$로 clipping한다.

재현 fixture는 `artifacts/verify_scored_math.py`, 원시 실행 결과는
`artifacts/verify_scored_math.log`에 있다. 수치 fixture는 증명의 대체물이
아니며, 아래의 상징 논증과 실제 구현의 대응 여부만 검사한다.

## F1. 재투영 없는 affine tensor transport

### [정리] F1

$J\in GL(d)$, $b\in\mathbb R^d$에 대해

$$
y_i=Jz_i+b,\qquad h_i=J^{-T}g_iJ^{-1}
$$

로 놓고 graph topology를 고정하자. 변위도 $w=Jv$로 수송하면 모든 국소
제곱길이, 모든 edge 길이, 모든 고정 vertex path 비용 및 모든 최단 비용이
보존된다.

### 증명

국소식에는 직접 대입하여

$$
q_i'(w)
=(Jv)^T(J^{-T}g_iJ^{-1})(Jv)
=v^Tg_iv
=q_i(v)
$$

를 얻는다. Edge 변위는 $y_j-y_i=J(z_j-z_i)$이고, 평균 계량은 행렬 수송의
선형성으로

$$
\frac{h_i+h_j}{2}
=J^{-T}\frac{g_i+g_j}{2}J^{-1}
$$

이다. 따라서 edge 제곱길이도

$$
(y_j-y_i)^T\frac{h_i+h_j}{2}(y_j-y_i)
=(z_j-z_i)^T\frac{g_i+g_j}{2}(z_j-z_i)
$$

로 동일하다. 양변이 양수이므로 제곱근인 $\ell_{ij}$도 같다. 같은 vertex
sequence를 갖는 모든 path의 각 항이 같아 $C'(P)=C(P)$이고, 고정된 같은
path 집합에서 최소화하므로 $D'(s,t)=D(s,t)$이다. 이것으로 네 종류의
보존을 모두 보였다.

`affine_chart_change`는 위의 $y_i,h_i$를 그대로 계산한다. Fixture는 수송
전후 모두 `make_state`를 사용했고 `project_metric`을 호출하지 않았으므로,
증명에 없는 비공변 연산이 끼어들지 않았다.

### [산출] F1 killing fixture

- 비직교 $3\times3$ Jacobian과 비영 offset을 사용했다.
- 국소식 상대오차: $1.1102230246251565\times10^{-16}$.
- edge 최대 상대오차: $6.867587011347192\times10^{-16}$.
- chain 최단 path 상대오차: $0$.
- 최대값은 $6.867587011347192\times10^{-16}<10^{-10}$이다.

판정: **PASS**.

## F2. fixed-chart spectral clipping의 affine 비공변성

### [no-go 정리] F2

$\Pi$를 대칭행렬의 고윳값을 $[1/4,4]$로 clipping하는 구현의
`project_metric`이라 하자. 일반적인 $J\in GL(d)$에 대해

$$
\Pi(J^{-T}gJ^{-1})
=J^{-T}\Pi(g)J^{-1}
$$

는 성립하지 않는다.

### 완전 반례

다음 SPD 계량과 invertible Jacobian을 택한다.

$$
g=\begin{pmatrix}9&0\\0&1\end{pmatrix},
\qquad
J=\begin{pmatrix}3&0\\0&1\end{pmatrix}.
$$

먼저 수송한 뒤 clipping하면

$$
J^{-T}gJ^{-1}=I,
\qquad
\Pi(J^{-T}gJ^{-1})=I.
$$

반대 순서에서는

$$
\Pi(g)=\begin{pmatrix}4&0\\0&1\end{pmatrix},
\qquad
J^{-T}\Pi(g)J^{-1}
=\begin{pmatrix}4/9&0\\0&1\end{pmatrix}.
$$

따라서 두 결과의 entrywise 최대 절대 defect는 $5/9$이다. 단 하나의
허용 입력에서 등식이 실패했으므로 일반 affine 공변성 주장은 완전히
반박된다.

### [산출] F2 killing fixture

실제 `project_metric`과 `affine_chart_change`의 두 합성 순서를 실행한 defect는
$0.5555555555555556=5/9>10^{-3}$이었다.

판정: **PASS**.

## F3. 정적 metric-graph 비용의 방향 대칭성

### [no-go 정리] F3

동결 구현이 받아들이는 모든 정적 상태에서

$$
C(v_0,\ldots,v_k)=C(v_k,\ldots,v_0),
\qquad
D(s,t)=D(t,s).
$$

따라서 이 비용 하나만으로는 같은 edge의 정방향과 역방향에 서로 다른
비용을 주는 비가역 동역학을 표현할 수 없다.

### 증명

생성자에서 호출하는 `normalized_graph_laplacian`은 adjacency의 대칭성과
연결성을 검사한다. `edge_lengths`는 같은 endpoint 평균 계량을 이용해
$\ell_{ij}$를 계산한 뒤 양쪽 array entry에 같은 값을 기록하므로
$\ell_{ij}=\ell_{ji}>0$이다. 이에 따라

$$
C(P^{\mathrm{rev}})
=\sum_{r=0}^{k-1}\ell_{v_{r+1}v_r}
=\sum_{r=0}^{k-1}\ell_{v_rv_{r+1}}
=C(P).
$$

Path reversal은 $s\leadsto t$ path 집합과 $t\leadsto s$ path 집합 사이의
전단사이고 비용을 보존한다. 양쪽 집합에서 최소를 취하면
$D(s,t)=D(t,s)$이다.

### [산출] F3 killing fixture

비균일 SPD node metric을 가진 5-node chain에서 구현이 반환한 양방향
최단 비용은 모두 $5.911821337545026$이고 상대오차는 $0<10^{-12}$였다.
반환 path의 vertex sequence도 정확히 서로의 역순이었다.

판정: **PASS**.

## F4. 대칭 diamond의 equivariant singleton goal no-go

### [no-go 정리] F4

외생 symmetry-breaking source가 없고, source vertex $0$을 고정하면서 후보
$1,2$를 교환하는 대칭을 가진 diamond에서 equivariant한 목표 readout은
$\{1,2\}$ 중 singleton 하나만 선택할 수 없다.

### 증명

점들을

$$
z_0=(0,0),\quad z_1=(1,1),\quad z_2=(1,-1),\quad z_3=(2,0)
$$

로 두고 edge를 $01,02,13,23$으로 둔다. 모든 node metric은 $I$이다.
$x$축 반사와 node permutation

$$
\sigma(0)=0,\quad \sigma(1)=2,\quad
\sigma(2)=1,\quad \sigma(3)=3
$$

의 결합은 점, graph, identity metric, source vertex 및 후보 집합을 모두
보존한다. 이 전체 입력을 $X$라 하자. Equivariant singleton selector
$S$가 존재하면 입력이 불변이므로

$$
S(X)=S(\sigma X)=\sigma S(X)
$$

여야 한다. 그러나 $S(X)=\{1\}$이면 우변은 $\{2\}$이고,
$S(X)=\{2\}$이면 우변은 $\{1\}$이다. 두 가능한 singleton 모두 모순이다.
따라서 대칭을 깨지 않는 완전한 minimizer readout은 두 원소를 함께
보존해야 한다.

### [산출] F4 killing fixture

후보를 역순 `[2, 1]`로 공급해도 실제 `minimum_cost_targets`는 두 비용을
모두 $\sqrt2=1.4142135623730951$로 계산하고
`minimizers == (1, 2)`, `unique == false`를 반환했다.

판정: **PASS**.

## F5. finite endpoint tensor의 continuum 비식별성

### [no-go 정리] F5

두 sample point에서의 metric tensor만으로 그 사이의 연속 Riemannian
metric이나 적분 길이를 유일하게 결정할 수 없다.

### 구성 및 증명

$M=[0,1]\times\mathbb R$에서 다음 두 smooth SPD metric을 정의한다.

$$
g^{(0)}=dx^2+dy^2,
$$

$$
g^{(1)}=\left(1+\sin^2(\pi x)\right)^2dx^2+dy^2.
$$

$1+\sin^2(\pi x)\ge1$이므로 둘 다 모든 점에서 양의 정부호이다. 또한
$x=0,1$에서는 sine 항이 0이므로 두 endpoint tensor는 모두 $I$로
동일하다. $\gamma(x)=(x,0)$의 길이는 각각

$$
L_{g^{(0)}}(\gamma)=\int_0^1 1\,dx=1,
$$

$$
L_{g^{(1)}}(\gamma)
=\int_0^1\left(1+\sin^2(\pi x)\right)dx
=\frac32.
$$

차이는 $1/2>10^{-2}$이다. 더 강하게, 임의의 절대연속 연결곡선
$\eta(t)=(x(t),y(t))$에 대해 두 번째 길이 integrand는

$$
\sqrt{(1+\sin^2(\pi x))^2\dot x^2+\dot y^2}
\ge (1+\sin^2(\pi x))|\dot x|
$$

이고, 우변 적분은 해당 함수의 원시함수를 이용하면 $3/2$ 이상이다.
$\gamma$가 등호를 달성하므로 두 번째 geodesic distance 자체도 $3/2$이다.
첫 번째는 $1$이다.

구현은 두 endpoint의 동일한 $I$만 받아 평균하므로 두 연속 보간 모두에
동일한 finite edge cost $1$을 부여한다. 즉 차이는 수치 오차가 아니라
입력 표현의 비식별성이다.

### [산출] F5 killing fixture

구현의 공통 finite endpoint edge cost는 $1$이었다. 200,000 midpoint로
두 번째 적분을 독립 재현한 값은 $1.5$이며 차이는 $0.5>10^{-2}$였다.

판정: **PASS**.

## 증명 채점

| 항목 | 상징 논증 | 실제 killing fixture | 점수 |
|---|---:|---:|---:|
| F1 affine readout covariance | 완전 | 재현 | 1/1 |
| F2 clipping 비공변성 | 완전 반례 | 재현 | 1/1 |
| F3 정적 비용 대칭성 | 완전 | 재현 | 1/1 |
| F4 equivariant singleton no-go | 완전 | 재현 | 1/1 |
| F5 continuum 비식별성 | 완전 구성 | 재현 | 1/1 |
| **합계** |  |  | **5/5** |

이 점수는 계약에서 정의한 `MATH PASS`만 확정한다. Held-out correctness,
OOD, oracle navigation utility 및 A1--A4는 다른 lane의 실행 점수이며 이
문서로 선취하지 않는다.

Status: COMPLETE
