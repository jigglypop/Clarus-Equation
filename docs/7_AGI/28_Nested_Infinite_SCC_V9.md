# V9 중첩 무한 SCC: 직접극한 recurrent tower 정본

> 날짜: 2026-08-12
>
> 형식 수학: **SURVIVES — NISCC-1~8 및 converse exhaustion 정리**
>
> isolated unit 구현: **IMPLEMENTED — focused `-W error` pytest 162 passed**
>
> V9 인과 메커니즘: **UNTESTED — 이론/설계 단계**
>
> 개발 실행: **0/256, BLOCKED — pre-run gate 미완료** *(당시 시점 기술, stale — 이후 별도 run에서 256-seed 실행 1회 집행, 판정 STOP; §13 사후 기록 참조)*
>
> 생물학·AGI 승격: **금지 — 직접 증거 없음**

이 문서에서 “뇌 자체를 무한 SCC로 본다”는 말은 물리적 뉴런이 무한하다는 뜻이
아니다. 정확한 뜻은 **유한한 강연결 graph view를 계속 생성할 수 있는 규칙의
직접합집합을 하나의 countably infinite SCC인 이상적 template으로 정의한다**는
것이다. 모든 실제 실행은 유한 prefix 또는 유한 causal cone만 계산한다.

핵심 판정은 다음 네 문장으로 닫힌다.

1. 하나의 고정 유향 그래프에서 서로 다른 maximal SCC는 중첩될 수 없다.
2. $G_0\subseteq G_1\subseteq\cdots$이고 각 $G_n$이 강연결이면
   $G_\infty=\bigcup_nG_n$도 강연결이다.
3. 이 topology만으로 수렴·기억·예측·지능은 나오지 않는다. 호환 동역학과 하나의
   level-independent contraction certificate가 별도로 필요하다.
4. V1~V8과 ACBSM 어디에도 이 다중 레벨 인과 상태의 성공 증거는 없다. 따라서
   V9는 이름이나 그래프 그림이 아니라 새 상태 메커니즘으로 새로 시험해야 한다.

## 1. 형식 객체와 “중첩”의 정확한 뜻

**[정의 NISCC-D1]** 비어 있지 않은 유향 그래프들의 열을

$$
G_n=(V_n,E_n),\qquad n\in\mathbb N
$$

이라 하고, $i_n:G_n\hookrightarrow G_{n+1}$을 injective graph embedding으로
둔다. 각 레벨을 그 image로 바꾸어

$$
V_n\subseteq V_{n+1},\qquad E_n\subseteq E_{n+1}
$$

로 쓴다. 모든 $G_n$이 강연결이면 $(G_n,i_n)_{n\in\mathbb N}$을
**중첩 SCC tower**라 부른다. 여기서 $G_n$은 자기 level-$n$ graph view에서는
유일한 maximal SCC일 수 있지만, 고정된 더 큰 $G_m$ 또는 $G_\infty$ 안에서는
일반적으로 maximal SCC가 아니라 strongly connected subgraph다.

**[정의 NISCC-D2]** 직접극한의 graph representative를

$$
G_\infty=(V_\infty,E_\infty),\qquad
V_\infty=\bigcup_{n=0}^{\infty}V_n,\quad
E_\infty=\bigcup_{n=0}^{\infty}E_n
$$

으로 정의한다. 유한 레벨이 무한히 자주 새 vertex를 하나 이상 추가하면 tower를
**properly infinite**라 한다. 단지 edge annotation이나 시간을 계속 추가하는 것은
proper vertex infinity가 아니다.

**[정의 NISCC-D3]** quotient tower는 포함과 다른 대상이다. 그것은 surjection

$$
\pi_{n+1,n}:Z_{n+1}\twoheadrightarrow Z_n
$$

과 projective compatibility를 사용한다. 반대로 이 문서의 graph direct union과
state direct limit는 injection을 사용한다. generator는 요청한 level·node·operator를
만드는 유한 규칙일 뿐, injection도 quotient도 자동으로 증명하지 않는다.

**[정의 NISCC-D4]** positive-delay forward event unroll은 vertex $(v,t)$와
시간을 엄격히 증가시키는 edge로 이루어진다. 이것은 recurrent template,
scale-indexed tower, time-translation quotient 어느 것과도 같은 graph가 아니다.

## 2. graph 직접합집합 정리와 역정리

**[정리 NISCC-1]** 포함으로 중첩된 비어 있지 않은 강연결 유향 그래프들의
직접합집합 $G_\infty$는 강연결이다.

**증명.** 임의의 $u,v\in V_\infty$를 택한다. 어떤 $r,s$에 대해
$u\in V_r$, $v\in V_s$이고, $k=\max(r,s)$로 두면 중첩성 때문에 둘 다
$V_k$에 속한다. $G_k$가 강연결이므로 $u\to^*v$와 $v\to^*u$인 유한
path가 $E_k\subseteq E_\infty$ 안에 존재한다. 따라서 모든 두 vertex가 서로
도달 가능하다. $\square$

**[정리 NISCC-1C: converse exhaustion]** 비어 있지 않은 유향 그래프가
countable하고 강연결인 것은, 그 그래프가 증가하는 유한 강연결 subgraph 열의
합집합인 것과 동치다. countably infinite graph이면 그 열은 properly infinite로
고를 수 있다.

**증명 개요.** 역방향은 NISCC-1과 countable union of finite sets에서 즉시
따른다. 정방향에서는 vertex와 edge를 열거하고 root $r$을 하나 고른다. 각 vertex
$v$마다 유한 witness path $r\to^*v$, $v\to^*r$을 고른다. stage $n$에서
처음 $n+1$개 vertex·edge와 그 endpoint들, 그리고 그들에 대한 두 root witness
path를 누적한다. 이 subgraph는 유한하고 모든 vertex가 $r$과 상호 도달
가능하므로 강연결이다. 모든 vertex와 edge가 언젠가 포함되므로 합집합은 원래
graph와 같다. 무한 graph에서는 새 vertex가 처음 등장하는 stage들의 subsequence를
취하면 proper exhaustion이다. explicit enumeration을 쓰거나 통상적인 ZFC를
전제로 한다. $\square$

**[정리 NISCC-2]** 모든 $V_n$이 유한하고 tower가 properly infinite이면
$V_\infty$는 countably infinite다. 외부 vertex가 없는 standalone
$G_\infty$는 정확히 하나의 maximal SCC, 즉 $V_\infty$ 전체를 갖는다.

**증명 개요.** 각 vertex를 `(birth level, 그 level 안의 순번)`에 대응시키면
자연수 쌍으로 열거된다. 무한히 많은 genuine birth 때문에 유한할 수 없다.
NISCC-1에 의해 전체가 한 mutual-reachability class다. 더 큰 ambient graph에
넣으면 외부 reciprocal path가 이 집합을 더 큰 SCC와 합칠 수 있으므로
standalone 조건은 필수다. $\square$

**[산출 BRAIN-N1: 수학적 모델]** NISCC-1/1C/2에서 countably infinite인
standalone one-SCC template과 그 finite strong exhaustion은 구성 가능하다.
exhaustion의 root·열거·witness path·level boundary는 유일하지 않으며, 이 산출은
물리적 neuron count나 biological identity를 포함하지 않는다.

## 3. 고정 graph와 시간 unroll의 no-go

**[정리 NISCC-3: no-go]** 하나의 고정 유향 그래프에서 서로 다른 maximal SCC는
서로소이며, overlap하거나 proper하게 중첩될 수 없다.

**증명.** mutual reachability는 equivalence relation이고 maximal SCC는 그
equivalence class다. 두 class가 vertex 하나라도 공유하면 같은 class다. 따라서
$\varnothing\ne C\subseteq D$인 두 SCC는 $C=D$다. $\square$

그러므로 “중첩 SCC”라는 표현은 graph view, induced domain, threshold, color,
resolution 또는 quotient map이 index와 함께 선언됐을 때만 유효하다. 이 정본의
유한 $G_n$들은 고정 limit graph 안의 여러 maximal SCC가 아니다.
고정 graph SCC의 표준 알고리즘 배경은
[Tarjan 1972](https://doi.org/10.1137/0201010)을 따른다.

**[정리 NISCC-4: no-go]** 모든 edge가 시간을 엄격히 증가시키는 forward event
unroll은 horizon이 무한해도 DAG이며, 모든 SCC는 singleton이다.

**증명.** 양의 길이 path를 따라 시간은 계속 증가하므로 시작 event로 돌아오는
cycle이 존재할 수 없다. 서로 다른 두 event가 상호 도달 가능할 수도 없다.
template self-loop $v\to v$조차

$$
(v,0)\to(v,1)\to(v,2)\to\cdots
$$

로 풀리며 event SCC는 모두 singleton이다. recurrence는 template 또는 따로
선언한 time-translation quotient에 있다. $\square$

## 4. direct-limit 동역학

**[정의 NISCC-D5]** 각 level state를 normalized metric space
$(X_n,d_n)$, update를 $F_n:X_n\to X_n$, injective isometric embedding을
$J_n:X_n\hookrightarrow X_{n+1}$로 둔다. algebraic direct limit의 원소는
$(n,x)$의 equivalence class

$$
(n,x)\sim(m,y)
\iff
J_{k,n}x=J_{k,m}y
\quad\text{인 어떤 }k\ge\max(n,m)
$$

이며 $[n,x]$로 쓴다.

**[조건부 정리 NISCC-5A]** 정확한 inclusion compatibility

$$
J_nF_n=F_{n+1}J_n
\tag{1}
$$

가 모든 level의 선언된 invariant image에서 성립하면

$$
F_{\mathrm{alg}}[n,x]=[n,F_nx]
\tag{2}
$$

는 representative와 무관한 well-defined update다. canonical prescription
(2)가 well defined이려면 (1)이 필요하다.

**증명 개요.** 같은 class의 두 representative를 common level $k$로 올리고
$F_k$를 적용한다. (1)을 반복 적용하면 두 image가 다시 같은 class가 된다.
역으로 $[n,x]=[n+1,J_nx]$의 두 representative가 같은 image를 가져야 하고,
canonical injection의 injectivity가 바로 (1)을 강제한다. $\square$

**[조건부 정리 NISCC-5B]** 모든 level에 공통인 하나의 유한 상수 $L$이 있어

$$
d_n(F_nx,F_ny)\le Ld_n(x,y)
\tag{3}
$$

이면 $F_{\mathrm{alg}}$은 direct-limit metric에서 $L$-Lipschitz이고, 그
metric completion $\overline X$ 위의 유일한 $L$-Lipschitz self-map
$\overline F$로 연장된다.

**증명 개요.** 두 algebraic state를 common level로 보내 (3)을 적용하면
direct-limit Lipschitz bound가 나온다. dense algebraic sequence $x_j\to x$의
$F_{\mathrm{alg}}x_j$가 Cauchy이므로 그 limit로 연장을 정의한다. 같은 bound가
sequence 선택과 무관함과 uniqueness를 보장한다. $\square$

weight tying 또는 같은 함수 이름을 모든 level에서 쓰는 것만으로 (1)이 증명되지는
않는다. 새 좌표, boundary state, bias, nonlinearity와 down-message까지 embedded
image를 invariant하게 보존해야 한다.

## 5. 수축, 고정점과 truncation bound

**[조건부 정리 NISCC-6A]** $\overline X$가 complete하고
$\overline F:\overline X\to\overline X$가 하나의 level-independent
$0\le q<1$에 대해

$$
d(\overline Fx,\overline Fy)\le qd(x,y)
\tag{4}
$$

를 만족하면 유일한 fixed point $x^*$가 존재하며

$$
d(\overline F^t x,x^*)\le q^t d(x,x^*)
$$

이다. 이는 Banach contraction theorem의 직접 적용이다.

**[조건부 정리 NISCC-6B]** finite-prefix map을 같은 complete metric space로
lift한 $\widehat F_n$이 있고, 선언된 invariant domain 전체에서

$$
\sup_z d(\overline Fz,\widehat F_nz)\le\epsilon_n
\tag{5}
$$

이면 두 rollout은

$$
d(x_t,y_t)
\le q^t d(x_0,y_0)
+\frac{1-q^t}{1-q}\epsilon_n
\tag{6}
$$

를 만족한다. $\widehat F_n$의 fixed point $y_n$이 존재하면

$$
d(y_n,x^*)\le\frac{\epsilon_n}{1-q}.
\tag{7}
$$

시간별 defect만 있는 경우에는 certified initial error와 실제 매 step의 defect를
사용해

$$
E_{t+1}\le qE_t+\eta_t,
\qquad
E_t\le q^tE_0+\sum_{s=0}^{t-1}q^{t-1-s}\eta_s
\tag{8}
$$

를 재귀적으로 유지해야 한다. 현재 state 한 점에서 측정한 defect 하나는 (5)가
아니며, 그것을 $1-q$로 나눠 fixed-point certificate나 depth deactivation에
사용할 수 없다.

**[완전 반례 NISCC-7A: topology는 안정성이 아니다]** 같은 proper bidirected
path tower 위에서 $F_n(x)=2x$를 쓰면 모든 비영 state가 발산하고,
$F_n(x)=-x$를 쓰면 모든 비영 state가 period two로 진동한다. graph tower는
두 경우 모두 동일하고 각 level도 강연결이다. 따라서 strong connectivity에서
수렴·fixed point attractor·기억·예측을 산출하는 부모 주장은 성립하지 않는다.

**[완전 반례 NISCC-7B: 호환성 없는 limit]** append-zero embedding에서
$F_n=I$와 $F_n=-I$를 level마다 번갈아 쓰면 같은 direct-limit state의 두
representative가 각각 $x$와 $-x$로 보내진다. $x\ne0$에서 둘은 다르므로
canonical limit update가 존재하지 않는다.

**[완전 반례 NISCC-7C: 모든 finite certificate는 uniform certificate가 아니다]**
$X_n=\mathbb R^n$에서 compatible weighted backward shift를

$$
(B_nx)_i=\frac{i}{i+1}x_{i+1}\ (1\le i<n),
\qquad (B_nx)_n=0
$$

로 둔다. 모든 $B_n$은 nilpotent라서 $\rho(B_n)=0$이고
$q_n=(n-1)/n<1$인 finite contraction이다. 그러나 $q_n\to1$이며 completed
$\ell^2$ shift는 $\lVert B\rVert=\rho(B)=1$이다. 따라서 모든 finite
principal truncation의 통과로 uniform strict contraction, bounded resolvent,
infinite-limit fixed-point bound를 주장할 수 없다.

## 6. lazy generator, causal cone과 quotient 경계

**[정리 NISCC-8A]** total generator `Gen(n)`이 유한 $G_n$, embedding,
birth level, complete incoming adjacency와 nesting·strong-connectivity certificate를
반환하면, 유한하게 많은 level/index query는 가장 큰 필요한 prefix까지만 생성해
답할 수 있다. NISCC-1/2가 materialization 없이 이상적 union의 graph 지위를
보장한다. 몇 개 prefix를 test한 사실만으로 모든 level의 invariant가 증명되지는
않으므로 generator rule 자체의 증명 또는 inductive certificate가 필요하다.

**[조건부 정리 NISCC-8B]** exact compatibility와 compatible readout
$R_mJ_{m,n}=R_n$이 있으면 모든 finite time $t$에

$$
F_m^tJ_{m,n}=J_{m,n}F_n^t
$$

이고, level $n$에서 태어난 finite-support state의 readout은 더 큰 어느 level에서
계산해도 정확히 같다. 이 정리는 nonzero infinite tail을 가진 임의의 completed
state에는 적용되지 않는다.

**[조건부 정리 NISCC-8C: exact finite causal cone]** limit graph의 in-degree가
유한하고, synchronous local update에서 정보가 tick당 최대 한 edge만 건너며,
finite query set $S$의 complete predecessor가 알려져 있다고 하자. horizon
$T<\infty$의 backward causal cone

$$
\mathcal C_T(S)=
\{u:\ u\to^*s\text{인 길이 }\le T\text{ path가 어떤 }s\in S\text{에 존재}\}
$$

은 유한하며 $S$의 time-$T$ state는 이 cone 안의 initial state와 input에만
의존한다. 최대 in-degree가 $\Delta$이면

$$
|\mathcal C_T(S)|\le |S|\sum_{k=0}^{T}\Delta^k.
$$

**증명 개요.** 유한 집합의 한-step predecessor가 유한이고 이를 $T$번
귀납한다. local update의 의존성이 tick마다 graph distance를 최대 하나만 줄이므로
distance $>T$인 vertex는 time-$T$ query에 영향을 줄 수 없다. 유한 cone은
tower의 어떤 finite prefix에 포함된다. $\square$

이 정리는 infinite-horizon 또는 exact fixed-point query를 유한하게 만들지 않는다.
그때는 별도의 contraction/tail bound, spatial decay 또는 exact quotient가 필요하다.

**[정리: exact quotient 판정]** surjection $Q:X\twoheadrightarrow Z$에 대해

$$
QF=\Phi Q
$$

인 유일한 macro update $\Phi:Z\to Z$가 존재할 필요충분조건은

$$
Qx=Qy\Longrightarrow QFx=QFy
\tag{9}
$$

인 fiber invariance다. 필요성은 같은 aggregate에 $\Phi$를 적용하면 나오고,
충분성은 $\Phi(z)=QF(x)$를 fiber representative와 무관하게 정의하면 나온다.
그러므로 유한 generator는 exact quotient가 아니며, compression 보존은 (9) 또는
semiconjugacy defect와 별도의 error bound를 요구한다.

## 7. 무차원 계약

**[정의 DIM-V9]** graph 정리들은 조합론적이라 물리 단위가 없다. state·동역학
정리에 들어가는 metric과 값은 다음과 같이 먼저 무차원화한다.

| 원시량 | V9 core 입력 | 차원 상태 |
|---|---|---|
| 물리 시간·delay $t,\tau$ | $t/t_0$, $\tau/t_0$ 또는 integer tick | 무차원 |
| rate $\nu$ | $\nu/\nu_0$ | 무차원 |
| energy $E$ | $E/E_0$ | 무차원 |
| reward·cost $r,c$ | $r/r_0,c/c_0$ | 무차원 |
| edge strength·state | 고정된 positive block/state scale로 정규화 | 무차원 |
| $L,q,\rho(M)$ | normalized metric/operator의 비 | 무차원 |
| $\epsilon_n,E_t,\eta_t$ | 동일한 normalized comparison metric | 서로 같은 type |

**[산출 DIM-V9]** 위 typing 아래 $\epsilon_n/(1-q)$, 식 (6)~(8),
normalized `tanh` 인자와 probability/fixed-point kernel은 무차원이다. positive
finite reference scale은 training data만으로 고정해야 하며, development H20 target을
본 뒤 scale을 바꾸면 새 model이다. 이 게이트는 차원 정합만 보장하며 수렴·예측력·
생물학적 정당성을 보장하지 않는다.

## 8. V1~V8 및 ACBSM에서 V9로 이어지는 계보

**[경험식: 저장된 synthetic 연구 결과]** `PASS`는 각 버전의 제한된 저장 gate만을
뜻하며 AGI나 생물학의 증거가 아니다.

| 연구 | 보존된 결과 | V9에 걸리는 경계 |
|---|---|---|
| V1 | programmed edge는 복원했지만 prediction·lesion·control gate 실패 | graph recovery는 state mechanism 증거가 아님 |
| V2 | target confounding을 제거한 좁은 matched-basis task에서 validation/test PASS; equal-probe dense와 사실상 동률 | 동일 정보·동일 capacity control 필수 |
| V3 | rank-one loading은 복원했지만 prediction conjunction 실패 | episode별 자유 timescale fit 금지 |
| V4 | pooled scalar AR의 one-step true-state-reentry gate PASS | prefix-only H20 free rollout 증거가 아님 |
| V5 | 최초 genuine prefix-only H20 free rollout; 네 comparator/robustness clause 실패 | frozen V5는 필수 failed-parent comparator |
| V6 | 등록만 있고 구현·실행 없음 | 증거 없음 |
| V7 | sparse ablation 하나만 양호; 전체 gate FAIL, test 미개봉 | 부분 통과로 route를 승격하지 않음 |
| V8 | prospective R1 confirmation에서 candidate $0.5377851901$, V5 $0.5400745832$, improvement $0.0022893931$, 95% interval $[-0.0031918816,0.0077706678]$; confirmation FAIL | output shrinkage family를 V9로 재명명하지 않음; test `81100..81355` 미개봉 |
| ACBSM | 8-fold training screen `HOLD`; rank-two가 모든 selected fold에서 rank-one으로 collapse; fresh `82100..82355` 미개봉 | rank-one observer는 comparator이며 nested multiscale state의 성공 증거가 아님 |

V9가 새 메커니즘이려면 적어도 둘 이상의 persistent level state와 실제 cross-scale
message가 output을 인과적으로 매개해야 한다. SCC decomposition, nested 그림,
독립 residual AR mode의 증식, $P+g(S-P)$ output gain, analytic posterior 또는
V5/ACBSM bypass는 V9가 아니다.

**[산출: RBE와의 설계 해석]** RBE가 방대한 weight enumeration을 유한한
basis·coordinate·generator representation으로 바꾸고 필요한 항만 query한다는
철학이라면, 이 V9 설계는 무한히 unfolding되는 agent/event instance의 enumeration을
유한 tower generator·호환 update·필요한 causal cone query로 바꾼다. 공통점은
“개체를 모두 materialize하지 않고 생성 규칙과 관계 구조를 보존한다”는 engineering
analogy다. RBE의 압축률·오차·외부 검증 여부는 NISCC 정리의 전제도 V9의 경험
증거도 아니다. 또한 어느 쪽의 generator도 fiber invariance를 보이지 않는 한
exact quotient를 뜻하지 않는다.

## 9. V9 finite engineering design과 falsifier

**[정의 V9-1 후보]** 선택된 engineering route는 유한 parameter generator가
중첩 tower의 다음 shell을 결정적으로 만들고, 실제 tick에는 adaptive finite prefix만
실행하는 previous-tick/Jacobi recurrent controller다. output은 immutable
`TowerStateToken`만 읽으며 raw event, target, hidden simulator state, persistence,
V5/V8 forecast 또는 ACBSM posterior를 직접 읽지 않는다.

**[예측 V9-P1: 개발 전 고정할 항목]** 같은 현재 input에 서로 다른 legal history를
주면 tower state와 output이 달라지고, full-state reset은 그 차이를 제거하며,
state swap은 donor history의 방향으로 output을 옮겨야 한다.

**[예측 V9-P2: 개발 전 고정할 항목]** retained upper level의 reset, upward/downward
message cut, one-tick shift, sign flip과 state shuffle은 실제 next-update tensor를
바꾸고 preregistered held-out degradation을 만들어야 한다. 각 lesion array는 별도
storage를 가져야 하며 label-only alias는 허용하지 않는다.

**[예측 V9-P3: 개발 전 고정할 항목]** candidate는 동일 raw stream·normalizer·
training budget을 받는 V5, ACBSM-rank1, flat finite recurrent, matched monolithic
SCC, compute-matched fixed-depth, maximum-depth, zero-bridge와 symmetric-dense control을
상대로 미리 정한 효과 하한을 넘어야 한다. 한 depth로 collapse하거나 matched
monolithic control과 동률이면 V9 mechanism route는 STOP이다.

**[미완성 V9-1]** 위 예측은 아직 시험되지 않았다. architecture, optimizer,
normalizer, comparison identity, intervention, budget, exact 256개 seed와 모든 hash를
freeze한 implementation lock 및 독립 pre-run audit가 없다. 따라서 development
registration은 HOLD, 실행은 **0/256 BLOCKED**, confirmation/test는 만들지 않는다.
V8 test `81100..81355`와 ACBSM fresh block `82100..82355`는 계속 닫아 둔다.
*(사후 주석: 위 문단은 이 문서 작성 시점의 기술이며 stale이다. 이후 별도 run에서
pre-run gate가 통과되어 256-seed 개발 실행이 1회 집행되었고 판정은 STOP이었다.
§13 사후 기록을 따른다.)*

## 10. neuroscience가 허용하는 해석의 상한

**[경험식: primary-source 동기]** 현재 출처는 유한한 recurrent anatomy와
multiscale organization의 동기까지만 제공한다.

- FlyWire v630의 confidence $\ge50$, neuron pair당 최소 5 synapse인 directed
  graph에서 `127,978` neurons 중 `93.3%`가 한 giant SCC였다
  ([Lin et al. 2024](https://doi.org/10.1038/s41586-024-07968-y)). 이는 그 graph의
  giant recurrent core 증거이지 nested maximal SCC나 stable dynamics의 증거가 아니다.
- 더 뒤의 adult-fly reconstruction도 `139,255` proofread neurons과 `54.5` million
  assigned chemical synapses인 유한 substrate다
  ([Dorkenwald et al. 2024](https://doi.org/10.1038/s41586-024-07558-y)).
- larval fly의 “nested recurrent architecture”는 hierarchical connectivity
  clustering과 최대 5-hop return-cascade census를 뜻하며 SCC tower가 아니다
  ([Winding et al. 2023](https://doi.org/10.1126/science.add9330)).
- BANC의 13 CNS networks는 directed weights를 symmetrize한 weighted undirected
  graph의 spectral clustering 결과이지 SCC가 아니다
  ([Bates et al. 2026](https://doi.org/10.1038/s41586-026-10735-w)).
- C. elegans chemical graph의 giant SCC는 `237/279`, electrical edge를 reciprocal로
  더한 union에서는 `274/279`였다. edge layer와 direction convention에 따라 SCC가
  바뀐다는 증거다
  ([Varshney et al. 2011](https://doi.org/10.1371/journal.pcbi.1001066)).
- mouse cortico-basal-ganglia-thalamic return loop는 finite recurrent pathway의
  해부·생리 근거지만 infinite tower나 V9 state mediation을 보이지 않는다
  ([Foster et al. 2021](https://doi.org/10.1038/s41586-021-03993-3)).
- adult human neuron count도 약 $86.1\pm8.1$ billion이라는 유한 추정이다
  ([Azevedo et al. 2009](https://doi.org/10.1002/cne.21974)).

**[미완성 BRAIN-N3]** 생물학적 brain이 이 tower의 inclusions, chosen geometry,
compatible update, contraction 또는 causal cross-level state를 실제로 구현한다는
직접 측정·intervention은 없다. literal infinite physical neuron/agent branch는
기각한다. graph theorem을 cognition, consciousness 또는 AGI로 승격하지 않는다.

## 11. 구현·검증 자원과 현재 허용 범위

**[산출 BRAIN-N2: 조건부 engineering design]** 독립 source·math·route lane과
status audit는 다음 isolated formal/unit 범위를 승인했다.

- deterministic finite-prefix generator와 nesting/SCC/birth/hash certificate;
- exact compatibility fixture와 explicit rejection;
- complete-predecessor를 포함한 finite causal-cone query;
- uniform 또는 recursive error-envelope helper;
- finite $D_{\max}$, previous-tick controller와 immutable state token;
- reset/up-cut/down-cut/time-shift/sign/shuffle의 실제 unit intervention;
- deterministic non-evidence fixture와 unit/property test.

이 범위는 형식 정리와 implementation contract를 검사할 뿐, predictive superiority,
V9 confirmation, biological identity 또는 AGI를 검사하지 않는다. 선행 run은 runtime
adapter와 evidence seed 실행을 승인하지 않았다. 2026-08-12 후속 light run은 기본
경로를 바꾸지 않는 opt-in runtime adapter와 unit/property 검증만 별도로 승인했다.

연구 원장:

- [계약](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/00-contract.md)
- [출처·V1~V8 계보 감사](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/10-sources.md)
- [수학 증명](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/11-math.md)
- [대안 route·개발 설계](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/12-routes.md)
- [형식 지위 감사](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/20-audit.md)
- [isolated 구현 잠금](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/30-implementation.md)
- [독립 unit 검증](../../_workspace/ce/agi-v9-nested-infinite-scc-20260811/31-validation.md)
- [runtime integration 계약](../../_workspace/ce/agi-v9-runtime-integration-20260812/00-contract.md)
- [runtime integration 수학 감사](../../_workspace/ce/agi-v9-runtime-integration-20260812/11-math.md)
- [runtime integration 지위 감사](../../_workspace/ce/agi-v9-runtime-integration-20260812/20-audit.md)
- [runtime integration 구현 기록](../../_workspace/ce/agi-v9-runtime-integration-20260812/30-implementation.md)
- [runtime integration 검증 기록](../../_workspace/ce/agi-v9-runtime-integration-20260812/31-validation.md)

구현된 opt-in unit surface:

- [finite tower generator](../../reality_stone/python/reality_stone/clarus/nested_scc_tower.py)
- [finite controller](../../reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py)
- [graph·dynamics unit tests](../../tests/test_nested_scc_tower.py)
- [controller·intervention unit tests](../../tests/test_adaptive_scc_tower_controller.py)
- [deterministic non-evidence demo](../../examples/agi/nested_scc_tower_demo.py)
- [opt-in runtime action path](../../reality_stone/python/reality_stone/clarus/agent.py)
- [runtime integration tests](../../tests/test_agent.py)
- [runtime integration demo](../../examples/agi/nested_scc_runtime_agent_demo.py)

**[산출: isolated unit 구현]** generator는 bidirected path shell의 등록된 유한
prefix, deterministic manifest, nesting·SCC audit, complete predecessor rule,
finite causal cone, strictly forward event-unroll audit와 schedule-specific contraction
certificate를 구현한다. default previous-tick/Jacobi fixture의 level-independent
row-sum Lipschitz 상계는 **global coordinate sup metric**, 즉 구현 식별자
`global_coordinate_sup`에서 $q=0.54$다. 이 metric과 schedule을 바꾸면 같은
certificate가 아니다. 이 수치는 default finite map의 unit certificate이며
infinite-limit 성능 수치가 아니다.

**[산출: 호환성 ceiling]** active upward boundary coupling 아래 일반적인
append-zero unit-cube inclusion은 명시적으로 `REFUSED`된다. exact compatibility는
zero-state/zero-input invariant singleton 또는 upward gain이 0인 별도 fixture에서만
통과한다. controller는 이 실패를 숨기지 않고 한 level씩 유한 $D_{\max}$까지
보수적으로 grow한 뒤 `exhausted`를 보고한다. sampled defect로 level을 제거하지 않고,
truncation bound나 infinite-horizon convergence를 주장하지 않는다.

**[산출: state-only unit 경로]** immutable token은 controller identity, episode
generation, tick, active depth, 모든 state와 delay-message buffer의 hash에 묶인다.
forecast/policy readout은 token만 받으며, stale·foreign·mutated token은 fail closed한다.
reset, up/down cut, one-tick shift, sign flip과 shuffle은 실제 다음 update가 소비하는
별도 storage의 tensor를 바꾼다. token hash는 이 in-process research fixture의
integrity check일 뿐 외부 authentication·security claim이 아니다.

**[산출: sealed state lifecycle]** generator의 spec·operator·manifest·array write
flag와 controller가 붙잡은 live generator identity·parameter identity는 public unit
boundary에서 다시 검사된다. `observe`, `reset_episode`, `load_state_dict`는 새 상태,
message, trace, token과 snapshot payload를 전부 검증한 뒤 한 번에 commit한다. seal,
schema, overflow, token, intervention 또는 snapshot 검증이 실패하면 이전 snapshot을
보존한다. 이는 해당 Python process의 unit atomicity이지 분산 transaction 보장이
아니다.

**[산출: exact schema·snapshot unit integrity]** unit API는 등록된 정수·문자열·
SHA-256 digest·permutation을 exact built-in schema로 재검증하고 observation과
permutation을 canonical tuple, 허용된 실수 입력을 finite canonical float로 고정한다.
bool, numeric text, built-in subclass, forged dataclass, lying container와 nonfinite 값은
fail closed한다. `observation_scales=()`인 **정확한 empty tuple**만 default-scale
sentinel이며 `False`, `0`, `None`, text와 list는 sentinel이 아니다. snapshot은 매
process에서 무작위로 생성된 key의 HMAC-SHA-256 tag에 묶이고, pending intervention을
포함한 same-process round trip 뒤 token·trace·state·readout continuation을 그대로
재현한다. 이 tag는 process-local provenance/integrity일 뿐 외부 인증이 아니며,
snapshot은 cross-process persistence format이 아니다.

**[산출: 2026-08-12 predecessor isolated unit lock]** 두 focused test 파일을 모든 warning을
error로 승격해 실행한 결과는 **162 passed in 3.90s**였다. 별도의 독립
schema/contract 감사도 **579 checks PASS**였지만, 둘 다 finite unit surface만
검사한다. 위 다섯 Python 파일은 Ruff check **All checks passed**, Ruff format check
**5 files already formatted**였으며 deterministic non-evidence demo는 exit code 0이었다.
독립 predecessor BFS는 exact finite cone과 일치했고, 각
reset/cut/shift/flip/shuffle fixture는 intact와 다른 실제 `state_after`를 만들었다.

| 잠금 대상 | SHA-256 |
|---|---|
| `nested_scc_tower.py` | `18A13966CBBE69F244D686D7F5C7DC58A2D1A8F20057BD79F8DCE12B01138F81` |
| `adaptive_scc_tower_controller.py` | `9204DDDBF893A0C15DC34DE503E1E9C853A14FAAD164FA5AFB0F32BB1822E028` |
| `test_nested_scc_tower.py` | `18FAC6E512D928CD8E50DD58BCE3977D3812452EE37373DB0ADE1E5059CB032A` |
| `test_adaptive_scc_tower_controller.py` | `7D7D3FBC04E441E7F33711C5B1D0026E9FB2EADE24142D01FB62B23BCFF877ED` |
| `nested_scc_tower_demo.py` | `595D518A1BB1B5689054451F7891E7A98BC5785905AB4560F4D9A28747967896` |

**[산출: opt-in runtime integration]** `RuntimeAgentConfig.nested_scc_enabled=True`이면
현재 observation과 고정 action embedding의 cosine similarity를 dimensionless action
evidence로 만든다. evidence는 Cauchy--Schwarz에 의해 각 좌표가 $[-1,1]$에 놓이며,
zero norm은 정확히 0으로 정의한다. 이어 `CausalEvent`가 tower state를 갱신하고,
행동은 발급된 token에 대한 `read_policy`의 `selected_action`에서만 나온다. 같은 현재
observation에 다른 legal history를 준 unit fixture는 서로 다른 policy를 만들었다.
이는 state mediation과 실행 가능한 유한 agent loop의 산출이지 task utility나 AGI
능력의 증거가 아니다. 기본값은 계속 `False`이고 legacy action path는 유지된다.
belief control과 V9를 동시에 켜는 결합은 아직 정리·검증되지 않아 fail closed한다.

**[산출: UNIT-P2 cleanup]** 실제 update와 grow-only depth decision에 들어가지 않던
`depth_error_tolerance`와 `hysteresis_ticks`는 삭제했다. 오해를 부르던
`generated_parameter_count`는 `serialized_operator_scalar_count`로 바꾸고, 두 operator
template·coordinate scale·여섯 dynamics coefficient를 직렬화한 metadata일 뿐 model
capacity, trainable-parameter parity 또는 MAC/compute 산정이 아니라고 코드에 고정했다.
이름이
underscore로 시작하는 field는 adversarial unit test가 seal을 검증하려고 변조하는
private implementation detail이며 public API가 아니다. snapshot은 위에서 고정한
same-process continuation에만 유효하다.

**[산출: 2026-08-12 runtime integration lock]** agent·tower·controller·runtime contract와
dimensionless 관련 집중 suite는 기존 PyTorch sparse constructor가 내는 두 `UserWarning`
category만 모듈 단위로 제외하고 모든 나머지 warning을 error로 승격해 **210 passed in
4.41s**였다. dimensionless 단독 suite는 **10 passed in 2.86s**, Ruff check는 변경된
실행 파일·테스트·demo에서 **All checks passed**, Ruff format check는 **8 files already
formatted**, deterministic runtime demo는 exit code 0이었다. demo의 세 tick은 active
depth $1\to2\to3$, action $0\to1\to2$를 token policy에서 산출했다.

| 후속 잠금 대상 | SHA-256 |
|---|---|
| `agent.py` | `34983C5A3AFA8A1DED9DA302CB44D4C6333AF78653626DEB80DEB579E70A64AC` |
| `clarus/__init__.py` | `6B2E16C1859B0107A9B77EADCD1E90D9E0BE00F1F740D3316F2EB04D8F8F08FA` |
| `nested_scc_tower.py` | `3C101AD966FE9AEE8D1F41E9319AB55D88D027B94FEF4584203A416F081652E7` |
| `adaptive_scc_tower_controller.py` | `9204DDDBF893A0C15DC34DE503E1E9C853A14FAAD164FA5AFB0F32BB1822E028` |
| `test_agent.py` | `0851595453E60C536D55648DFAE454186653062FBC34950A784D0FEFBB8F923C` |
| `test_nested_scc_tower.py` | `EFF7DF726B78CCE777106835DC45546FCB424B547E40A226DC11E2FD7D8EC126` |
| `test_adaptive_scc_tower_controller.py` | `7D7D3FBC04E441E7F33711C5B1D0026E9FB2EADE24142D01FB62B23BCFF877ED` |
| `nested_scc_runtime_agent_demo.py` | `1AB2F48503B6EEAFE3489BC1EFCC962241FB7F9F4392BA6C268E3DECB452CEDA` |

따라서 이 실행은 고정 fixture의 unit integrity와 unit/property causal effect만
검증했다. development seed, H20 benchmark, matched control, predictive score,
biological data는 하나도 열거나 실행하지 않았다. V9 data 상태는 계속 **0/256
BLOCKED**이며 biological/AGI route는 **UNTESTED / NOT AUTHORIZED**다.
*(사후 주석: 마지막 문장의 data 상태는 이 절 작성 시점 기술이며 stale이다.
이후 개발 실행 1회가 집행되었고 판정은 STOP이다. §13 사후 기록을 따른다.)*

## 12. 최종 경계표

| 주장 | CE 지위 | 살아 있는 정확한 범위 | 넘어갈 수 없는 경계 |
|---|---|---|---|
| nested finite strong views의 direct union | **[정리]** | nonempty, injectively nested level이면 union도 strong | 각 finite level이 fixed limit의 별도 maximal SCC라는 뜻이 아님 |
| countable strong graph의 finite exhaustion | **[정리]** | explicit enumeration/ZFC에서 양방향 동치 | exhaustion의 uniqueness·생물학적 level을 주지 않음 |
| proper finite tower의 infinite SCC | **[정리]** | infinitely many vertex births, standalone union | literal infinite physical neurons가 아님 |
| fixed graph의 nested maximal SCC | **[정리: no-go]** | distinct SCC는 disjoint | graph view를 숨긴 중첩 표현 금지 |
| forward infinite unroll | **[정리: no-go]** | strict positive delay이면 DAG·singleton SCC | event infinity를 recurrent SCC라 부르지 않음 |
| direct-limit update | **[조건부 정리]** | isometric embedding과 exact compatibility | weight tying·topology만으로 성립하지 않음 |
| generic append-zero exactness | **[완전 반례]·[산출: unit]** | zero invariant singleton 또는 structural zero upward gain만 exact | nonzero upward boundary coupling의 일반 주장은 `REFUSED` |
| unique fixed point와 truncation bound | **[조건부 정리]** | complete self-map, uniform $q<1$, common metric/domain·defect | finite $q_n<1$ 목록이나 한 점 defect는 불충분 |
| finite causal cone | **[조건부 정리]** | finite in-degree, local tick, complete predecessors, finite horizon | infinite-horizon/fixed-point 유한 계산을 보장하지 않음 |
| lazy generator | **[정리]·[산출: unit]** | total certified rule이면 finite query theorem; 현재 코드는 등록된 finite $D_{\max}$ prefix fixture | exact quotient·unbounded implementation을 자동 보장하지 않음 |
| sealed finite controller | **[산출: unit]·[미완성]** | exact schema, atomic lifecycle, same-process HMAC continuation과 opt-in RuntimeAgent 연결 | cross-process snapshot, adaptive truncation, capacity/MAC, 성능·생물학·AGI를 보장하지 않음 |
| V9 nested causal state | **[산출: state-mediated runtime path]·[미완성]·[예측]** | observation→tower state→token policy 인과 경로는 unit 검증; utility는 새 development registration에서만 시험 가능 | 현재 성능 성공·confirmation 주장 금지; "0/256 BLOCKED"는 당시 시점 기술로 stale — §13 사후 기록(실행 1회, 판정 STOP) 참조 |
| brain-wide nested representation | **[미완성]** | finite biological analogue라는 falsifiable hypothesis | biological identity·의식·AGI·물리적 무한 승격 금지 |
| V15 metric-atlas reading | **[공리: 모델 선택]·[미완성]** | 각 finite SCC node를 하나의 metric-graph sample로 읽을 수 있음 | sampling/overlap/operator 수렴 없이 Riemannian atlas·Laplace--Beltrami·지능 증가로 승격 금지 |

따라서 V1~V9 연구의 이 문서 작성 시점 종착점은 다음과 같았다.

```text
NESTED-SCC MATHEMATICS       SURVIVES
ISOLATED GENERATOR/UNIT CODE IMPLEMENTED — PREDECESSOR LOCK PRESERVED
OPT-IN RUNTIME STATE PATH    IMPLEMENTED — 210 RELATED TESTS PASS
V9-1 TASK UTILITY            UNTESTED
256-SEED DEVELOPMENT RUN     0/256 — BLOCKED
V9 CONFIRMATION              NOT REGISTERED / BLOCKED
BIOLOGICAL OR AGI CLAIM      UNTESTED / NOT AUTHORIZED
```

위 상태 블록에서 `V9-1 TASK UTILITY`, `256-SEED DEVELOPMENT RUN`,
`V9 CONFIRMATION` 세 행은 stale이다. 이후 경과는 §13 사후 기록에 있다.

## 13. 사후 기록 (2026-08-12): 첫 256-seed 개발 실행과 STOP 판정

이 절은 §9, §11, §12의 "0/256 BLOCKED" 기술 이후의 사실 경과를 기록한다. 원
기록은 당시 시점 기술로 본문에 그대로 보존하되, 현재 상태로는 stale이다. 별도
run `agi-v9-loop-engineering-20260812`에서 preregistration과 authorization을
포함한 pre-run gate가 통과되어(Gate: PASS), development seed `0..255`에 대한
256-seed 개발 실행이 1회 집행되었다.

**[산출: 2026-08-12 개발 실행 결과]** 등록된 gate 판정은 **STOP**이다. V9 arm의
평균 정확도는 $0.3457$이고, 가장 강한 등록 대조군인 matched monolithic 대조군은
$0.6116$이었다. paired mean improvement는 $-0.2659$(95% bootstrap 구간
$[-0.2788,-0.2524]$)로, 임계 $\ge 0.02$를 만족하지 못해 FAIL이다. 반면
upper-reset loss $0.0635$와 cross-cut loss $0.0635$는 임계 $\ge 0.05$를 통과했고,
causal integrity counter는 모두 0으로 PASS였다. confirmation seed
`10000..10255`는 개봉되지 않았고 봉인 상태로 유지된다. 모든 수치는 무차원
정확도·정확도 차이다.

이 결과에서 살아남는 진술과 죽는 진술은 다음과 같이 나뉜다. cross-level state가
reset/cut 대조 대비 output 행동을 실제로 바꾼다는 인과 기여 진술은 [산출]로
살아남는다. 반면 이 등록된 mechanism/task에 대한 V9의 task utility 우위,
confirmation 개시, AGI·생물학 함의는 기각된다. 따라서 §12 표의 V9 nested causal
state 행은 "실행 1회 집행, 판정 STOP, confirmation 미개봉"으로 읽어야 한다.

**[미완성: 사후 진단과 재설계 조건 — 후속 가설]** post-development 진단은 모든
V9 level이 같은 약한 local recurrence를 공유했고 readout이 지연된 약한
복사본들을 평균했다는 것이다. 이 진단은 사후적이며 이번 점수를 수리하지 못하고
confirmation을 열지 못한다. 후속 재설계는 다음 여섯 조건을 요구한다.

1. local temporal state를 기저 객체로 삼는다.
2. monadic 객체는 neuron이나 shell이 아니라 typed transition kernel이다.
3. cross-level state는 full이 local-only와 cloud-only를 동일 최적화·compute
   예산에서 이길 때만 인정한다.
4. level별 timescale은 candidate와 matched control 양쪽에 포함한다.
5. readout은 모든 level을 평균하지 않고 local과 shared increment를 분리해
   노출한다.
6. 이번 STOP의 seed block과 confirmation block은 재사용하지 않는다.

후속 architecture는 별도의 새 model이며 새 계약과 새 seed role을 요구한다.
이번 점수의 수리가 아니다.

출처:

- [post-development audit](../../_workspace/ce/agi-v9-loop-engineering-20260812/artifacts/post-development-audit.md)
- [최종 폐쇄 보고](../../_workspace/ce/agi-v9-loop-engineering-20260812/40-final-report.md)
