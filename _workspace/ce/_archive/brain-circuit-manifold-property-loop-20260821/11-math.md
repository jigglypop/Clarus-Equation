# Mathematics

Status: COMPLETE

## 1. 검산 대상

predecessor의 smooth, frozen-`p,B,R` branch에서

\[
\xi_{n+1}=F_n(\xi_n,u_n;W),\qquad
\delta\xi_{n+1}=A_n\delta\xi_n+\mathcal B_n\delta u_n
\]

를 쓴다. 여기서 `A_n=D_xi F_n`이고 지연 상태의 shift row까지 포함한다. 이번
검산은 세 객체를 구별한다.

1. `J_T=P Phi(T,0) iota`: initial current activation에서 terminal current
   activation으로 가는 flow derivative.
2. `g=J_T^T G_T J_T`: full-rank일 때만 Riemannian인 passive pullback.
3. `W_c=H Rbar^-1 H^T`: augmented terminal state의 fixed-horizon control
   Gramian. 이는 endpoint minimum-energy form이지 passive metric과 같은 객체가 아니다.

## 2. tangent와 circuit response

frozen efficacy에서는 top block이

\[
(A_n)_{i,(j,d)}=(1-\lambda_i)\delta_{ij}\mathbf1[d=0]
+\lambda_i\phi_i'(h_i^n)W_{ij}p_{ij}^n\mathbf1[d=d_{ij}]
\]

이다. 회로 `W(epsilon)=W0+epsilon C_Gamma`를 켤 때 `dot A`는 direct edge
항뿐 아니라 reference trajectory가 움직여 `phi''(h) dot h`가 생기는 total
derivative다. 따라서

\[
\dot\Phi(T,0)=\sum_{r=0}^{T-1}
\Phi(T,r+1)\dot A_r\Phi(r,0)
\]

와

\[
\dot g=\dot J^T G_TJ+J^TG_T\dot J
\]

를 중앙차분으로 별도 검산한다. direct-edge partial response가 우연히 작은 예제에서
가까워도 total derivative로 인정하지 않는다.

## 3. 좌표 공변성의 정확한 범위

arbitrary invertible `S` 뒤에 componentwise `tanh`를 다시 적용하면 이는 좌표변환이
아니라 새 모델이다. 같은 nonlinear map의 재차트는 delay lift

\[
L=\operatorname{diag}(S,\ldots,S),\qquad
f_y(y)=Lf_x(L^{-1}y)
\]

로 정의해야 한다. 이때

\[
A_y=LA_xL^{-1},\quad \mathcal B_y=L\mathcal B_x,
\quad \iota_y=L\iota_xS^{-1},\quad P_y=SP_xL^{-1},
\quad J_y=SJ_xS^{-1}
\]

이고 terminal metric도

\[
G_{T,y}=S^{-T}G_{T,x}S^{-1}
\]

로 바꿔야

\[
g_y=S^{-T}g_xS^{-1}
\]

가 된다. `G`의 numeric entries를 모든 chart에서 그대로 고정하면 다른 geometric
reference를 선택한 것이며 covariance 반례가 아니다. adverse control에는 명시적으로
non-orthogonal `S`를 써야 이 오류가 노출된다. 방향도 `v_y=S v_x`로 옮겼을 때만
quadratic length를 비교한다.

## 4. terminal space를 섞지 않는 제어식

augmented target에는

\[
H=[\Phi(T,1)\mathcal B_0,\ldots,\mathcal B_{T-1}],
\qquad W_c=H\bar R^{-1}H^T
\]

를 쓴다. current activation target이라면 먼저 `H_a=PH`로 투영하고
`W_{c,a}=P W_cP^T`를 써야 한다. `H`의 target과 `P W_c P^T`의 target을 섞는
식은 ill-typed다.

reachable `v`의 weighted least-norm 해는

\[
A=H\bar R^{-1/2},\quad y^*=A^\dagger v,\quad
U^*=\bar R^{-1/2}y^*
\]

이고

\[
(U^*)^T\bar R U^*=v^TW_c^\dagger v.
\]

state chart `L` 아래에는 `H_y=LH`, `W_{c,y}=LW_cL^T`, `v_y=Lv`이므로
reachable subspace에서 energy가 불변이다.

full operational rank와 `kappa_2(W_c)<=10^8`인 점에서는 product rule로

\[
\dot W_c=\dot H\bar R^{-1}H^T+H\bar R^{-1}\dot H^T,
\qquad
\dot E=-v^TW_c^{-1}\dot W_cW_c^{-1}v
\]

를 쓴다. `dot H`에는 `dot Phi B`와 `Phi dot B`가 모두 들어간다. rank-changing
지점에서 이 inverse derivative를 연장하지 않는다.

## 5. exact rank와 operational certification

floating-point SVD cutoff

\[
\tau(M)=10^{-10}\max(1,\sigma_{max}(M))
\]

는 exact algebraic rank 정의가 아니라 이 run의 수치 판정 규약이다. 따라서
`r_tau<dimension`은 `operationally rank uncertified`라고 쓰며 exact singularity
정리로 부르지 않는다. 반면 `diag(1,0)`처럼 exact construction으로 영공간이
주어진 fixture는 `EXACT_RANK_DEFICIENT`라고 쓴다. generalized-eigenvalue와
determinant-ratio는 비교하는 두 metric이 모두 operationally full rank일 때만
허용한다. full inverse response는 `r_tau=dimension`이고
`kappa_2(W_c)<=10^8`인 영역에만 제한한다. near-singular 또는 subspace-changing
지점에서는 finite pre/post comparison과 reachability status만 허용한다.

## 6. state-dependent efficacy의 필수 항

`p_ij=p_ij(xi)`이면 product rule로

\[
\frac{\partial}{\partial\xi_\alpha}(p_{ij}a_j^{n-d_{ij}})
=p_{ij}\mathbf1_{\alpha=(j,d_{ij})}
+a_j^{n-d_{ij}}\frac{\partial p_{ij}}{\partial\xi_\alpha}
\]

가 된다. 회로 response에도 `dot p=grad p dot dot xi`가 들어간다. 이를 생략한
frozen-`p` Jacobian은 plastic efficacy model의 Jacobian이 아니다. 이번 adverse
fixture는 full formula의 finite difference 일치와 omitted-term formula의 의도된
불일치를 동시에 요구한다. one-step tangent와 two-step circuit-state response를
분리해 각각 `partial p / partial xi`와 `dot p` 누락을 공격한다.

## 7. 반례와 지위

- `J=diag(1,0)`: bounded smooth map이어도 pullback은 degenerate하다.
- `B=0`, `v!=0`: `v^T W_c^dagger v=0`이라는 숫자는 최소에너지가 아니라
  infeasible target을 잘못 읽은 것이다. 올바른 값은 `+infinity`다.
- rank-one `H`와 image에 수직인 `v`: pseudoinverse quadratic form만으로는
  reachability를 판정할 수 없다.
- `p(xi)`인데 derivative 항을 뺌: one-step central difference가 이를 반증한다.
- `S`로 state만 바꾸고 `G_T`는 바꾸지 않음: metric covariance를 고의로 깨는
  adverse control이다.

현재 수학 판정은 `READY_FOR_PROPERTY_TEST`다. frozen smooth branch 안에서는
pre-test P0가 없으며, 결과가 식 mismatch를 보일 때만 같은 gate로 1회 개정한다.
