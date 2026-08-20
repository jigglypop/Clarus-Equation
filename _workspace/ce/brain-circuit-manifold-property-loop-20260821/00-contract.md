# Research contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brain-circuit-manifold-equations-20260821`

## 1. 목적과 주장 상한

이 run은 predecessor의 A6-P passive flow-pullback과 A6-C finite-horizon
minimum-energy 식을 **무작위이되 완전히 재현 가능한 smooth delayed network**에서
property test한다. 검증 순서는 `식 동결 -> 수치 반증 시도 -> 실패 원인 분류 -> 같은
시험으로 식만 1회 개정 -> 재시험`이다.

이 run은 실제 뇌 반응, 해부학적 피질 주름, AGI 성능 또는 생물학적 회로 형성을
검증하지 않는다. 성공해도 지위는 `MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED`를
넘지 않는다. 실제 BrainRuntime은 hard selection, clamp, STP, refractory/adaptation,
backend별 delay semantics를 포함하므로 이번 smooth A6 시험과 동일시하지 않는다.

## 2. predecessor evidence

| frozen file | SHA-256 | 이 run이 상속하는 사실 |
|---|---|---|
| `00-contract.md` | `54b5d7716c6e57df9113ecaf770bcc61452e8e5019efce92f39bf2940581286b` | A6.1--A6.10a 정의와 범위 |
| `11-math.md` | `ec6a23a6093df33204306759e73b86d63086e3a8728d2ae5eb9067ba9fccaf83` | passive/active 기하 분리와 반례 |
| `20-audit.md` | `7af493542bedd94986c719e38b121d9d68b966344ddebae9f0e8ebb8e3f6993c` | 수학 gate PASS, anatomy bridge BLOCKED_INPUT |
| `31-validation.md` | `23531a1fdaf6fab5882a42c61e4bc2692b34a9063c84a3615c6627ec0860c2e9` | 작은 deterministic witness PASS |
| `40-final-report.md` | `80878ce5793b8526c7517dbdc74332e991db3cec0a5b2594454c4ab82df6f3a6` | BA-A6-P/C claim ceiling |

새 empirical response/confirmation asset은 열지 않는다. A3--A5의 threshold, clip,
RMS, ridge, horizon 또는 cohort도 다시 조정하지 않는다.

Revision 1 재실행 환경은 `.codex/hooks/python.cmd`가 선택한
`C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe`, Python
`3.11.9`, NumPy `2.4.6`, Windows `10.0.26200`으로 동결한다. 최종 receipt에는 실제
`sys.executable`, Python/NumPy/platform version, 실행한 source SHA-256과 contract
SHA-256을 기록한다. 환경이 다르면 같은 receipt로 합치지 않는다.

## 3. 동결된 smooth delayed fixture

시드 목록은 다음 여덟 개로 고정한다.

```text
104729, 130363, 169399, 200003, 250007, 300017, 350003, 400009
```

각 시드는 순서대로 `(q,D,T)=(3,1,3),(4,2,4)`를 번갈아 쓴다. 모든 수는
dimensionless normalized fixture다.

- `tanh` activation, `lambda_i in [0.35,0.90]`.
- signed `W_ij in [-0.35,0.35]`, frozen efficacy `p_ij in [0.20,0.90]`.
- heterogeneous integer delay `d_ij in {0,...,D}`.
- net offset `c_i=b_i-theta_i in [-0.30,0.30]`; 별도 receipt가 없으므로 `b`와
  `theta`의 독립 식별을 주장하지 않는다.
- fixed history `a_0,...,a_{-D} in [-0.40,0.40]`.
- deterministic time-varying `B_n`, SPD `R_n`, fixed input path `u_n`, SPD terminal
  metric `G_T`.
- `C_Gamma`는 diagonal을 항상 포함하고 off-diagonal support는 seed RNG의 고정
  Bernoulli(0.35) draw로 정한다. nonzero entry의 절댓값은 `[0.04,0.12]`이고 부호도
  같은 RNG로 고정한다. 따라서 `||C_Gamma||_infinity<=0.48`이며 전체
  `epsilon in [0,1]`에서 `|W_ij(epsilon)|<=0.47`이다.
- circuit path는 `W(epsilon)=W_0+epsilon C_Gamma`, 기준점은 `epsilon=0.25`.

상태는

\[
\xi_n=(a_n,a_{n-1},\ldots,a_{n-D})\in\mathbb R^{q(D+1)}
\]

이고, nonlinear update와 접선식은 predecessor A6.1--A6.3c를 그대로 사용한다.
초기 current slot injection을 `iota`, terminal current slot projection을 `P`라 하면

\[
J_T=P\Phi(T,0)\iota,\qquad
g_T=J_T^\top G_TJ_T.
\tag{P1}
\]

## 4. 고정 property tests

오차 함수는 행렬에 대해

\[
\operatorname{err}(X,Y)=
\frac{\lVert X-Y\rVert_F}{\max(1,\lVert Y\rVert_F)}
\tag{P2}
\]

로 고정한다. 중앙차분 간격은 모두 `h=2^-17`이다.

### P-A — passive tangent

초기의 current activation만 열별로 `+/-h` perturb하고 과거 history는 고정한다.
nonlinear endpoint finite-difference `J_FD`와 (P1)의 analytic `J_T`에 대해 모든
시드에서

\[
\operatorname{err}(J_T,J_{FD})\le5\times10^{-7}
\tag{P3}
\]

이어야 한다. 수치 rank 인증은 singular values에 대해

\[
\tau(M)=10^{-10}\max(1,\sigma_{\max}(M)),\qquad
r_\tau(M)=\#\{\sigma_k(M)>\tau(M)\}
\tag{P4}
\]

를 쓴다. exact mathematical rank와 `r_tau`를 구분해 기록한다. exact fixture
`J=diag(1,0)`은 `EXACT_RANK_DEFICIENT`이고 ridge를 금지한다. randomized floating
fixture에서 `r_tau<q`이면 exact rank theorem 대신 `PASSIVE_RANK_UNCERTIFIED`로
기록한다. generalized eigenvalue와 determinant-ratio는 pre/post 두 metric 모두
`r_tau=q`일 때만 계산한다.

### P-B — total circuit response

trajectory derivative `dot a`, 그리고 trajectory 변화를 포함한 `dot A_n`,
`dot B_n`, `dot Phi`, `dot J`,

\[
\dot g=\dot J^\top G_TJ+J^\top G_T\dot J
\tag{P5}
\]

를 predecessor A6.7a-1--A6.7b로 계산한다. `epsilon +/- h` 중앙차분과 비교해

\[
\operatorname{err}(\dot J,\dot J_{FD})\le2\times10^{-6},\qquad
\operatorname{err}(\dot g,\dot g_{FD})\le3\times10^{-6}
\tag{P6}
\]

이어야 한다. 직접 `C_Gamma` 항만 남긴 frozen-trajectory partial derivative를 total
response로 통과시키지 않는다.

### P-C — coordinate covariance

다음 deterministic non-orthogonal matrix를 쓴다. `S`의 diagonal은
`linspace(0.75,1.25,q)`, `S[0,1]=0.25`, `S[1,0]=-0.10`이며 나머지
off-diagonal은 0이다. 모든 fixture에서 `kappa_2(S)<4`이고
`||S^T S-I||_F>0.1`이어야 한다. 이어서

\[
L=\operatorname{diag}(S,\ldots,S)
\]

로 delay history 전체를 재차트한다. 이는 새로운 componentwise-tanh 생물학 모델을
만드는 것이 아니라, 같은 map을

\[
f_y(y)=Lf_x(L^{-1}y)
\]

로 표현하는 접선 좌표변환이다. 따라서

\[
A_y=LA_xL^{-1},\quad \mathcal B_y=L\mathcal B_x,\quad
\iota_y=L\iota_xS^{-1},\quad
P_y=SP_xL^{-1},\quad G_{T,y}=S^{-\top}G_{T,x}S^{-1}
\tag{P7}
\]

이고

\[
J_y=SJ_xS^{-1},\qquad g_y=S^{-\top}g_xS^{-1}.
\tag{P8}
\]

두 covariance residual과 `v_y=S v_x`의 quadratic length residual은 모두
`1e-10` 이하여야 한다. 반대로 이 non-orthogonal `S`로 `J`만 변환하고 terminal
`G_T`의 numeric entries를 새 chart에서도 그대로 두는 adverse control은 residual이
`1e-5` 이상이어야 한다. 같은 chart를
pre/post에 일관되게 적용했을 때만 generalized eigenvalues와 determinant ratio를
불변량으로 읽는다.

### P-D — Gramian과 minimum energy

이번 A6-C target은 **augmented terminal state**다.

\[
H_T=[\Phi(T,1)\mathcal B_0\ \cdots\ \mathcal B_{T-1}],\qquad
\mathcal W_c=H_T\bar R_T^{-1}H_T^\top.
\tag{P9}
\]

current endpoint만 시험할 경우에는 모든 곳에서 `H_a=P H_T`와
`W_{c,a}=P W_cP^T`를 써야 하며 둘을 혼합하지 않는다. 모든 시드에서 symmetry
residual은 `1e-12` 이하, 최소 고유값은
`-1e-11 max(1,||W_c||_2)` 이상이어야 한다.

reachable target `v=H_TU`에 대해 `A=H_T Rbar^-1/2`의 weighted least-norm
해를 독립 계산한다. residual은 `1e-9` 이하, 그 energy와
`v^T W_c^dagger v`의 상대오차는 `1e-8` 이하여야 한다. inverse derivative는
`r_tau(W_c)=dim(W_c)`이고 `kappa_2(W_c)<=1e8`일 때만 허용한다.

그 eligible domain에서는 predecessor A6.10a의

\[
\dot H_k=\dot\Phi(T,k+1)\mathcal B_k
+\Phi(T,k+1)\dot{\mathcal B}_k,
\]

\[
\dot W_c=\dot H\bar R^{-1}H^\top+H\bar R^{-1}\dot H^\top,
\qquad
\dot E=-v^TW_c^{-1}\dot W_cW_c^{-1}v
\tag{P9a}
\]

도 fixed baseline target `v`에 대해 `epsilon +/- h` 중앙차분과 비교한다. normalized
energy-derivative error는 `5e-6` 이하여야 한다. eligible하지 않은 seed에는 이
inverse 식을 적용하지 않고 rank/condition status를 기록한다.

상태 chart에서는

\[
H_y=LH_x,\quad W_{c,y}=LW_{c,x}L^\top,\quad v_y=Lv_x,
\quad E_y(v_y)=E_x(v_x)
\tag{P10}
\]

이며 well-conditioned case의 energy residual은 `1e-10` 이하여야 한다.

### P-E — reachability killing controls

1. `B=0`, `v!=0`: `UNREACHABLE`, energy `+infinity`.
2. rank-one `H`와 `v` orthogonal to `Im H`: `UNREACHABLE`.
3. singular values `(1,1e-12,...)`: operational rank deficient; ridge나 inverse
   derivative 금지.

`pinv`가 숫자를 반환했다는 이유만으로 unreachable target을 finite-energy로
분류하면 P0다.

### P-F — state-dependent efficacy adverse branch

별도 exact smooth fixture는 `q=2,D=1`과 다음 값을 쓴다.

```text
lambda = [0.80, 0.65]
W = [[0.70, -0.20], [0.35, 0.50]]
C = [[0.12, -0.08], [0.05, 0.11]]
delay = [[0, 1], [1, 0]]
xi = [0.45, -0.30, -0.20, 0.35]
c = [0.05, -0.08]
alpha = [[0.10, -0.20], [0.30, -0.10]]
beta[0,0] = [0.90, 0.00, -0.40, 0.20]
beta[0,1] = [-0.20, 0.60, 0.30, -0.50]
beta[1,0] = [0.50, -0.70, 0.40, 0.10]
beta[1,1] = [-0.60, 0.20, 0.10, 0.80]
```

여기서

\[
p_{ij}(\xi)=\frac{1+\tanh(\alpha_{ij}+\beta_{ij}^\top\xi)}2
\tag{P11}
\]

를 쓰면 full top-row derivative는

\[
(A_n)_{i\alpha}=(1-\lambda_i)\delta_{\alpha,(i,0)}
+\lambda_i\phi_i'(h_i)
\sum_jW_{ij}\left[p_{ij}\mathbf1_{\alpha=(j,d_{ij})}
+a_j^{n-d_{ij}}\frac{\partial p_{ij}}{\partial\xi_\alpha}\right].
\tag{P12}
\]

circuit response도

\[
\dot h_i=\sum_j\left[C_{ij}p_{ij}a_j
+W_{ij}(\dot p_{ij}a_j+p_{ij}\dot a_j)\right],\qquad
\dot p_{ij}=\nabla p_{ij}\cdot\dot\xi
\tag{P13}
\]

를 포함해야 한다. one-step full tangent formula는 error `<=5e-7`를 만족하고,
`partial p / partial xi`를 고의로 뺀 frozen-p 식은 error `>=1e-3`로 실패해야
한다. 같은 exact fixture를 두 step 전진시킨 circuit-state response도 full
`dot p` 식은 central-FD error `<=5e-7`를 만족하고, `dot p`만 고의로 생략한
response는 error `>=1e-5`로 실패해야 한다. 이는 frozen-p A6를 plastic efficacy에
그대로 재사용할 수 없다는 경계이지, 원래 범위의 A6 반증이 아니다.

## 5. 판정과 재귀 수정 규칙

`PROPERTY_PASS`는 여덟 시드가 P-A--P-D를 모두 통과하고 P-E/P-F adverse
control이 지정된 방향으로 실패할 때만 준다.

- analytic equation mismatch: `P0_FORMULA`, 수식과 구현을 함께 1회 개정한 뒤
  **같은 시드ㆍ간격ㆍ허용오차ㆍcontrol**을 재실행한다.
- ill-conditioned/rank boundary만 발생: `P1_DOMAIN_BOUNDARY`, 식을 억지로
  통과시키지 않고 정의역/상태를 좁힌다.
- fixture 또는 receipt 불일치: `P2_APPARATUS`, 결과를 폐기하고 outcome-blind
  apparatus만 고친다.
- tolerance, seed, scale, dimension, horizon 또는 adverse separation을 결과를 본 뒤
  완화하지 않는다.
- 한 번의 formula revision 뒤 같은 gate가 다시 실패하면 이번 branch는
  `STOP_MATH_PROPERTY`다. 다른 식은 새 successor contract에서 시작한다.

성공해도 A6-P는 full-rank smooth flow의 conditional pullback theorem이고 A6-C는
고정 actuator/cost/horizon의 endpoint minimum-energy form일 뿐이다. actual cortical
folding bridge는 계속 `BLOCKED_INPUT`이다.
