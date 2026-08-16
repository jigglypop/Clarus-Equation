# 인과적 재귀 기하 복원 수학 독립 검산

Status: COMPLETE

PREDECESSOR: `_workspace/ce/_archive/agi-v15-unified-metric-20260813/40-final-report.md`

## 1. 대상, 정의역과 전제

이번 레인은 계약의 `CGM-D1`, `CGM-N1`, `CGM-H1`--`CGM-H6`, `CGM-X1`, `CGM-X2`를 유한 유향 그래프, 유한차원 이산시간 동역학, 대칭 양의 정부호 계량의 범위에서 검산한다. 관측만으로 얻는 동치와 개입으로 깨지는 동치를 구분하고, 그래프 support는 고정된 관측 좌표계에 상대적인 대상으로 취급한다. 외부 신경 자료나 생물학적 사상은 사용하지 않았다.

[정의] 두 잠재 상태모형이 모든 허용 입력 아래 같은 관측분포를 만들면 관측 동치라 한다. 관측만 허용한 경우에는 입력집합이 하나의 수동 정책으로 축소된다. 정확한 잠재 좌표, 그 좌표에서의 edge support와 계량 성분은 이 동치류 안에서 불변일 때만 식별 가능하다.

[정의] 분할의 block-sum 사상을 $Q$라 하고 선형계가 $x_{t+1}=Ax_t+Bu_t$를 만족한다고 하자. quotient state $y_t=Qx_t$가 모든 미시상태에 대해 닫혀 있으려면 어떤 $\bar A,\bar B$가 존재하여

$$
QA=\bar A Q,
\qquad
QB=\bar B
\tag{1}
$$

를 만족해야 한다. 식 (1)을 이 레인의 선형 predictive sufficiency 또는 exact lumpability 조건으로 쓴다.

[정의] 잠재 encoder $W$와 SPD 계량 $g$가 만드는 관측 가능한 이차 비용은

$$
c(x)=x^TW^TgWx=x^TKx,
\qquad K=W^TgW
\tag{2}
$$

다. loss가 식 (2)에만 의존하면 $W$와 $g$ 각각이 아니라 $K$만 직접 식별된다.

## 2. 주장별 판정표

아래 P 등급은 수학 레인의 결함 등급이다. `없음`은 해당 범위에서 P0--P2 결함을 찾지 못했다는 뜻이며 최종 승격 판정은 아니다. 강한 부모 문장에 반례가 있고 좁은 실험 가설이 남는 경우 두 범위를 분리했다.

| Claim ID | 수학 레인 지위 | P 등급 | 정확한 판정 범위 |
|---|---|---|---|
| `CGM-D1` | [정리 후보] | 없음 | 유한 고정 유향 그래프에서 maximal SCC 분할은 유일하고 self-edge를 제거한 condensation은 DAG다. |
| `CGM-N1` | 강한 부모는 [미완성], 대체 no-go는 [정리 후보] | P0 | 일반 관측 시계열만으로 정확한 잠재 좌표와 그 좌표의 $G$를 유일 복원한다는 문장은 latent similarity와 관측 Markov 동치 반례로 거짓이다. |
| `CGM-H1` | [미완성] | P0(무조건 개선), P1(좁은 가설) | 개입이 있다는 사실만으로 분리모형의 strict 성능 우위가 보장되지는 않는다. 완전관측 LTI와 full-rank excitation에서는 $A,B$의 유일 식별만 살아난다. |
| `CGM-H2` | [미완성] | P0(자동 충분성), P1(비교 가설) | SCC quotient는 위상적 상호도달성만 보존하며 predictive sufficiency나 다른 압축보다의 우위를 자동 보장하지 않는다. 식 (1)이 붙은 좁은 quotient만 정확하다. |
| `CGM-H3` | [미완성] | P1, 단 joint encoder 해석에는 P0 경계 | 고정 representation에서 metric feature의 추가 예측력은 시험 가능하다. $W,g$를 함께 학습하면서 개선을 $g$의 독립 기억 효과로 귀속하는 문장은 식 (2)의 gauge 때문에 식별되지 않는다. |
| `CGM-H4` | [미완성] | P1 | 느린 제약으로 회상을 모델링하는 것은 허용되는 모델 선택이지만 보편 기억기전은 아니다. controllability Gramian은 접근 에너지라는 좁은 operational metric을 준다. |
| `CGM-H5` | [미완성] | P0(항상 우위), P1(이질 문맥 가설) | 두 문맥 계량이 같으면 모든 mixture가 단일 metric과 정확히 같으므로 strict 우위는 보장되지 않는다. rank·gate·label permutation 자유도를 회계한 OOD 비교는 남는다. |
| `CGM-H6` | [미완성] | P1 | 동일 규약의 합성계·신경계 교차도메인 성공은 수학 산출이 아니라 blind replication 가설이다. 좌표, 측정과 intervention interface를 도메인별로 고정해야 한다. |
| `CGM-X1` | 활성 주장 제외 유지 | P1; 무연결 공리계의 함의로 읽으면 P0 | 의식의 operational predicate와 SCC를 잇는 공리가 없다. 그래프 공리만 둔 형식모형에서 의식 predicate를 false로 배정할 수 있어 SCC로부터의 논리적 함의가 없다. |
| `CGM-X2` | 활성 주장 제외 유지, no-go는 [정리 후보] | P0 | 정적 SPD metric의 거리는 대칭이므로 비가역 drift를 정하지 못한다. source-free 대칭계의 유일 의미 목표도 선행 V15의 isometry 반례를 피하지 못한다. |

## 3. `CGM-D1`과 SCC 반복의 정확한 경계

유향 그래프에서 $u\sim v$를 $u$에서 $v$로 가는 path와 $v$에서 $u$로 가는 path가 모두 존재한다는 관계로 정의한다. 길이 0 path, path의 방향쌍, path concatenation으로 각각 반사성, 대칭성, 추이성이 나오므로 $\sim$은 동치관계다. 그 동치류는 정의상 maximal SCC이며 동치류 분할은 유일하다.

condensation에 서로 다른 block $C_1,\ldots,C_r$의 유향 cycle이 있다고 하자. 각 quotient edge를 원그래프 path와 block 내부 path로 이어 붙이면 모든 $C_i$가 서로 도달 가능하다. 그러면 이 block들의 합집합이 하나의 더 큰 strongly connected set이므로 maximality에 모순이다. 따라서 intra-block self-edge를 제거한 condensation은 DAG다.

이 정리에는 중요한 즉시 귀결이 있다. DAG의 SCC는 모두 singleton이므로 같은 edge semantics로

$$
G_{k+1}=\operatorname{Cond}(G_k)
\tag{3}
$$

를 반복하면 첫 축약 뒤 비자명 SCC hierarchy가 끝난다. 네 node 예에서는 첫 분할이 $\{0,1\},\{2,3\}$이고 condensation edge가 첫 block에서 둘째 block으로 하나일 때, 다음 SCC는 두 singleton이다. `artifacts/verify_cgm_math.py`는 self-loop를 제외한 네 labeled node의 모든 $2^{12}=4096$개 그래프에서도 이 경계를 확인했다. 이 유한 열거는 일반 증명을 대신하지 않는다.

따라서 동일 semantics의 `SCC를 SCC로 재귀 압축하면 비자명 다중스케일이 자동 생성된다`는 부모 범위에는 P0 반례가 있다. 살릴 수 있는 좁은 경로는 scale $\ell$마다 새로운 effective edge operator

$$
E_{\ell+1}=\Phi_{\ell}(G_{\ell},X,W_{\ell},u)
\tag{4}
$$

를 명시하는 것이다. 시간창 $W_{\ell}$, coarse intervention response 또는 quotient dynamics가 reciprocal edge를 새로 만들 수는 있지만, 이는 SCC 정리의 산출이 아니라 별도 공리와 선택 자유도다.

## 4. `CGM-N1`: 두 개의 명시적 관측 동치 반례

### 4.1 잠재 similarity 반례

선형 잠재계 $x_{t+1}=Ax_t$, $y_t=Cx_t$와 임의의 $H\in GL(n)$에 대해

$$
x'_t=Hx_t,
\qquad
A'=HAH^{-1},
\qquad
C'=CH^{-1}
\tag{5}
$$

로 두면 모든 $t$에서 $C'(A')^tHx_0=CA^tx_0$다. 정확한 fixture는

$$
A=
\begin{pmatrix}
1/2&0\\
0&1/3
\end{pmatrix},
\quad
H=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},
\quad
A'=
\begin{pmatrix}
1/2&-1/6\\
0&1/3
\end{pmatrix}.
\tag{6}
$$

$A$에는 node 사이 off-diagonal edge가 없고 $A'$에는 $2\to1$ edge가 있다. 그러나 $C=(1,2)$, $C'=(1,1)$, $x_0=(2,3)^T$, $x'_0=Hx_0$이면 $t=0,\ldots,6$의 관측이 정확히

$$
8,\ 3,\ 7/6,\ 17/36,\ 43/216,\ 113/1296,\ 307/7776
\tag{7}
$$

로 같고 등식은 모든 $t$에 성립한다. 따라서 decoder와 잠재 chart가 고정되지 않은 관측-only 계에서 정확한 latent support는 식별되지 않는다.

### 4.2 causal direction 반례

$\rho=1/2$라 하자. $X\sim N(0,1)$와 독립인 $\epsilon\sim N(0,3/4)$에 대해 $Y=\rho X+\epsilon$인 $X\to Y$ 모형과, $Y\sim N(0,1)$와 독립인 $\epsilon'\sim N(0,3/4)$에 대해 $X=\rho Y+\epsilon'$인 $Y\to X$ 모형은 둘 다

$$
\operatorname{Cov}(X,Y)=
\begin{pmatrix}
1&1/2\\
1/2&1
\end{pmatrix}
\tag{8}
$$

를 만든다. 수동 관측분포만으로 두 causal direction을 구별할 수 없다. 비가우시안성, 시간순서, 구조 공리 또는 개입은 이 동치를 깰 수 있지만 그 중 하나가 명시되어야 한다.

## 5. 개입·완전관측 선형계에서 살아남는 좁은 정리

**조건부 정리 후보.** 고정된 관측 좌표에서 완전히 관측되는 LTI 계가

$$
x_{t+1}=Ax_t+Bu_t+\epsilon_t
\tag{9}
$$

를 만족하고, hidden confounder가 없으며 $\mathbb E[\epsilon_t\mid x_t,u_t]=0$라고 하자. $Z=[X;U]$의 population second moment가 양의 정부호이면 conditional mean으로 $[A\ B]$가 유일하게 식별된다. noiseless finite design에서 $\operatorname{rank}Z=n+m$이면

$$
[A\ B]=YZ^T(ZZ^T)^{-1}
\tag{10}
$$

로 유일하다.

**검산.** $\Theta_1Z=\Theta_2Z$이면 $(\Theta_1-\Theta_2)ZZ^T=0$이다. full row rank에서 $ZZ^T$가 가역이므로 $\Theta_1=\Theta_2$다. scratch fixture는

$$
[A\ B]=
\begin{pmatrix}
1/2&1/4&1\\
0&1/3&2
\end{pmatrix}
\tag{11}
$$

를 exact rational arithmetic으로 복원한다. 반대로 rank-deficient design에는 $\Delta\Theta=(2,-1,7;0,0,0)$인 nonzero null direction이 있어 $(\Theta+\Delta\Theta)Z=\Theta Z$다.

이 정리는 `CGM-H1`의 성능 우위를 증명하지 않는다. 고정 좌표의 $A,B$ support를 식별할 수 있다는 좁은 명제다. $z_t$가 관측되고 $z_t$와 $x_t$의 등록된 interaction feature까지 포함한 design이 full rank일 때에는 같은 증명이 context coefficient에 확장된다. $z_t$가 숨었거나 design rank가 부족하면 분리는 다시 비식별적이다. 또한 true model이 이미 parameter-matched baseline에 포함되거나 $z_t$가 상수이면 두 모형의 Bayes rollout error가 같을 수 있어 strict improvement는 보장되지 않는다.

## 6. SCC의 threshold·window 민감도와 predictive sufficiency 반례

weighted edge가 $0\leftrightarrow1$에 각각 $3/5$, $1\leftrightarrow2$에 각각 $2/5$라고 하자. threshold $1/2$에서는 SCC가 $\{0,1\},\{2\}$지만 threshold $3/10$에서는 $\{0,1,2\}$ 하나다. event edge가 시간 $1,2,3,4$에 차례로 $0\to1$, $1\to0$, $1\to2$, $2\to1$이면 window $t\le2$와 $t\le4$도 같은 두 분할을 만든다. 따라서 SCC는 edge estimator, threshold와 window를 고정한 뒤의 정확한 위상량이지 그 선택들에 불변인 물리 모듈이 아니다. 여러 threshold와 window 중 endpoint가 좋은 조합을 고르면 선택 수의 곱만큼 look-elsewhere가 생긴다.

SCC가 하나라는 사실은 quotient가 predictive sufficient하다는 뜻도 아니다. 다음 계를 보자.

$$
A=
\begin{pmatrix}
0&1\\
2&0
\end{pmatrix},
\qquad
Q=\begin{pmatrix}1&1\end{pmatrix}.
\tag{12}
$$

두 node는 하나의 SCC다. $x=(1,0)^T$와 $x'=(0,1)^T$는 모두 $Qx=Qx'=1$이지만

$$
QAx=2,
\qquad
QAx'=1.
\tag{13}
$$

따라서 현재 compressed state가 같아도 다음 compressed state가 다르다. 실제로 $QA=(2,1)$은 어떤 scalar $\bar A$에 대해서도 $\bar A Q=(\bar A,\bar A)$와 같을 수 없다.

반대로 식 (1)이 성립하면 $y_{t+1}=\bar A y_t+\bar B u_t$이므로 quotient는 정확히 닫힌다. 모든 $x,u$에서 선형 quotient dynamics가 닫힌다고 가정하면 계수 비교로 식 (1)이 필요하므로 이 조건은 이 범위에서 필요충분하다. SCC partition은 식 (1)을 자동 함의하지 않는다. `CGM-H2`가 살아남으려면 lumpability residual, long-horizon aggregate prediction과 held-out intervention propagation을 별도 endpoint로 측정해야 한다.

## 7. SPD gauge, geometry--weight 비식별성과 방향성 no-go

좌표변환 $y=h(x)$의 Jacobian을 $J$라 하면 같은 intrinsic metric의 성분은

$$
g_y=J^{-T}g_xJ^{-1}
\tag{14}
$$

로 바뀐다. 따라서 raw matrix entry, eigenvector 방향과 서로 다른 latent chart의 $g_t-g_0$를 곧바로 기억량으로 읽을 수 없다. 고정된 chart 또는 invariant distance·energy가 필요하다. 이는 coordinate gauge이며 올바르게 quotient하면 같은 intrinsic geometry를 표현한다.

encoder와 metric을 함께 학습할 때는 더 직접적인 factorization 비식별성이 있다. 임의의 $S\in GL(r)$에 대해

$$
W'=SW,
\qquad
g'=S^{-T}gS^{-1}
\tag{15}
$$

이면 $(W')^Tg'W'=W^TgW$다. scratch fixture의

$$
W=I,
\quad g=\operatorname{diag}(4,9),
\quad S=
\begin{pmatrix}1&1\\0&1\end{pmatrix}
\tag{16}
$$

에서는 $W'=S$, $g'=\begin{pmatrix}4&-4\\-4&13\end{pmatrix}$이 같은 $K=\operatorname{diag}(4,9)$를 만든다. 일반적인 자유도는 $GL(r)$의 $r^2$ continuous gauge이며 stabilizer가 있으면 orbit 차원은 줄어든다. loss가 $K$만 볼 때 geometry 개선과 ordinary weight 개선을 분리해 귀속할 수 없다. encoder freeze, gauge fixing, 독립 cost measurement 또는 cross-fitted dynamics가 최소 경계다.

정적 SPD metric은 방향도 만들지 않는다. 모든 displacement $v$에 대해

$$
v^Tgv=(-v)^Tg(-v)
\tag{17}
$$

이고 curve reversal도 길이를 보존하므로 $d_g(x,y)=d_g(y,x)$다. fixture의 정방향·역방향 비용은 둘 다 $25$다. 비가역성은 $F$, drift, control constraint, time orientation, Finsler/Randers 같은 비대칭 구조 중 하나에서 와야 한다. time-dependent SPD metric도 각 시각의 접벡터 부호 대칭만으로 방향을 고르지 못하므로 외생 시간방향이나 dynamics가 여전히 필요하다. 이는 선행 V15의 `CGM-X2` no-go를 보존한다.

## 8. 살아남는 좁은 기억 정리: controllability Gramian

고정 representation의 이산 선형 제어계

$$
x_{k+1}=Ax_k+Bu_k,
\qquad x_0=0
\tag{18}
$$

에서 $T$ step 도달행렬을 $R_T=[A^{T-1}B,\ldots,B]$라 쓰면

$$
W_T=R_TR_T^T
=\sum_{k=0}^{T-1}A^kBB^T(A^k)^T
\tag{19}
$$

다. $W_T\succ0$이면 $R_Tu=x$를 만족하는 control 중 $\lVert u\rVert_2^2$ 최소값은

$$
E_T(x)=x^TW_T^{-1}x,
\qquad
u^*=R_T^TW_T^{-1}x
\tag{20}
$$

다. 실제로 $R_Tu^*=x$이고 임의의 feasible $u=u^*+v$에서 $R_Tv=0$이므로 $u^*$와 $v$가 직교하여 $\lVert u\rVert^2=\lVert u^*\rVert^2+\lVert v\rVert^2$다. 따라서

$$
g_T=W_T^{-1}
\tag{21}
$$

는 cue/control energy로 정의된 operational SPD metric이다. finite horizon에는 $A$의 안정성이 필요 없고, $T\to\infty$ Gramian에는 spectral stability가 필요하다.

exact fixture에서 $A=\operatorname{diag}(1/2,1/3)$, $B=(1,1)^T$, $T=2$이면

$$
W_2=
\begin{pmatrix}
5/4&7/6\\
7/6&10/9
\end{pmatrix},
\qquad
W_2^{-1}=
\begin{pmatrix}
40&-42\\
-42&45
\end{pmatrix}.
\tag{22}
$$

$x=(1,0)^T$의 최적 control은 $(6,-2)^T$이고 최소 energy는 $40=x^TW_2^{-1}x$다. 반대로 $B=(1,0)^T$이면 $W_2=\operatorname{diag}(5/4,0)$라 $(0,1)^T$는 도달 불가능하고 global SPD metric은 존재하지 않는다. 이때 reachable subspace의 pseudoinverse 비용만 정의할 수 있으며 unreachable 방향의 비용은 무한대로 취급해야 한다.

이 정리는 `기억=metric`을 증명하지 않는다. $g_T$는 $A,B,T$에서 유도되므로 weight dynamics와 독립 상태가 아니고, $A$나 $B$가 학습 뒤 바뀌면 함께 바뀐다. nonlinear 계에서는 operating trajectory 주위의 local Gramian과 validity radius가 필요하다. 살아남는 문장은 `고정 representation과 등록된 cue/control interface에서 controllability가 성립하면 attractor 접근 최소 energy는 SPD quadratic form이며 recall accessibility의 사전 고정 predictor 후보가 된다`다. weight-only baseline이 같은 $A,B$에서 이 energy를 이미 계산하거나 held-out recall을 동일하게 예측하면 geometry의 독립 효능 주장은 죽는다.

## 9. 숨은 공리와 자유도

1. exact graph support는 latent invariant가 아니라 고정 관측 chart, sampling interval과 edge semantics에 상대적이다.
2. intervention은 target, dose, timing, off-target effect와 reset 가능성을 알아야 식별성에 기여한다. 개입 label만으로 충분하지 않다.
3. SCC threshold $\tau$, time window $W$, edge estimator, transitive reduction 여부는 외부 선택이다. $L_\tau L_W L_E$ 조합을 본 뒤 고르면 그만큼 선택 규모가 생긴다.
4. $d$차원 SPD 한 개는 $p=d(d+1)/2$개 성분을 가진다. $K$개 context metric과 gate parameter $q$는 단일 metric보다 $(K-1)p+q$개 이상의 명시 자유도를 추가한다. context label permutation만으로도 $K!$개의 동일 parameterization이 있다.
5. identical context metric $g_1=g_2$에서는 모든 convex mixture가 단일 metric과 같으므로 `mixture가 항상 strict 개선한다`는 부모 문장은 완전 반례를 가진다.
6. `CGM-H6`의 통계 단위는 frame이 아니라 독립 graph, seed, animal 또는 session이어야 한다. 도메인별 retuning 횟수는 fitted freedom으로 센다.

## 10. P0/P1/P2 발견과 부모 범위

| ID | 등급 | 발견 | 무너지는 정확한 부모 범위 | 좁은 잔존 범위 |
|---|---|---|---|---|
| `M-P0-1` | P0 | 식 (5)--(8)의 관측 동치 | 관측-only로 정확한 latent coordinate와 edge direction/support를 유일 복원 | 관측·개입·모형 제약 아래 동치류 또는 고정 chart의 LTI coefficient 식별 |
| `M-P0-2` | P0 | true model이 baseline에 포함되거나 $z_t$가 상수이면 strict 개선 0 | 개입이 있으면 분리모형이 항상 rollout을 개선 | 등록된 이질 문맥·제약 아래의 비교 가설 |
| `M-P0-3` | P0 | 식 (12)--(13) | SCC membership/aggregate가 자동 predictive sufficient | 식 (1)을 만족하는 exact lumpable quotient |
| `M-P0-4` | P0 | condensation DAG의 SCC는 singleton | 같은 edge semantics의 SCC 반복이 비자명 hierarchy를 계속 생성 | 식 (4)의 scale별 새 effective-edge 공리가 있는 hierarchy |
| `M-P0-5` | P0 | 식 (15)--(16) | joint encoder loss 개선을 독립 metric-memory 효과로 귀속 | fixed/cross-fitted representation의 invariant energy 비교 |
| `M-P0-6` | P0 | $g_1=g_2$ mixture | 문맥 mixture가 단일 metric보다 항상 strict 우위 | 이질 문맥과 matched dof에서의 OOD 예측 가설 |
| `M-P0-7` | P0 | 식 (17) | 정적 SPD metric 하나가 비가역 전이와 방향을 결정 | 별도 $F$/control/time orientation이 있는 directed model |
| `M-P1-1` | P1 | noisy·partial·latent 계의 식별 조건 미고정 | `CGM-H1`, `CGM-H6` | 개입 설계와 blind split을 고정해야 재개 가능 |
| `M-P1-2` | P1 | threshold/window/partition 선택 회계 필요 | `CGM-H2` | preregistered sensitivity surface와 lumpability residual 필요 |
| `M-P1-3` | P1 | operational metric과 기억 기전의 bridge 없음 | `CGM-H3`, `CGM-H4` | Gramian energy의 held-out recall·ablation 검증 필요 |
| `M-P1-4` | P1 | 의식 predicate와 bridge 공리 없음 | `CGM-X1` | 독립 operational 정의와 반증 가능한 연결 명제 필요 |
| `M-P2-1` | P2 | $G$, $g$, $G_t$와 graph/metric 표기가 문맥상 충돌 가능 | 전체 | graph는 $\mathcal G$, metric은 $g$처럼 최종 집필에서 분리 권고 |

## 11. 교차 예측과 재현

수학적으로 유망한 교차 예측은 세 가지다. 첫째, 식 (10)으로 식별한 계수는 training에 없던 intervention target과 amplitude의 one-step 및 multi-step conditional mean을 동시에 예측해야 한다. 둘째, SCC quotient가 실제 충분통계라면 식 (1)의 residual과 서로 다른 microstate에서의 aggregate rollout error가 함께 작아야 한다. 셋째, 식 (21)의 energy가 기억 접근성을 포착한다면 fixed $A,B$에서 cue별 basin-entry time 또는 성공률을 순위 예측하되 weight-only 모델에 이미 포함된 정보를 넘어서야 한다.

재현 명령은 다음과 같다.

```powershell
python _workspace/ce/agi-connectome-geometric-memory-20260816/artifacts/verify_cgm_math.py
```

스크립트는 Python 표준라이브러리의 exact rational arithmetic만 사용한다. 원시 요약은 `artifacts/verify_cgm_math.log`에 있다.

Status: COMPLETE
