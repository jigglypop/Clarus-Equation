# 수학 레인

Status: COMPLETE

## 판정

초안의 단일 “reachability metric” 표현은 REVISE였다. Gramian pseudoinverse는 fixed-horizon terminal minimum-energy quadratic form이지 일반적인 local Riemann/sub-Riemannian metric이 아니다. 또한 $p_{ij}^n$의 상태 의존성을 허용하면서 $\partial p/\partial a$를 누락하면 Jacobian이 틀린다.

정정본은 다음 두 대상을 분리하므로 수학적으로 닫힌다.

$$
g_{\rm pass}(T)=J_T^\top G_TJ_T
$$

는 full-rank flow derivative가 만드는 **수동적 pullback Riemann metric**이고,

$$
E_T^*(v)=v^\top\mathcal W_c(T)^\dagger v
$$

는 reachable terminal displacement에 대한 **능동적 endpoint minimum-energy form**이다.

## 1. 타입과 전제

뉴런 수는 $q$, horizon은 $T$, 최대 지연은 $D$로 분리한다.

$$
a_n\in[-1,1]^q,
\qquad
\xi_n=(a_n,a_{n-1},\ldots,a_{n-D})\in\mathbb R^{q(D+1)}.
$$

계약의 모든 state와 input은 고정 reference scale로 무차원화한다. $R_n\succ0$이고, 기준 trajectory·history·$p_n$·$B_n$은 tangent perturbation에 대해 고정한다. $\phi_i$는 기준 drive에서 $C^1$이다. circuit-response 미분까지 사용할 때는 필요한 구간에서 $C^2$와 reference trajectory의 $\varepsilon$-미분가능성을 요구하며 $p,B,b,\theta,\lambda,d,G_T$는 그 branch에서 고정한다.

## 2. delayed tangent system 검산

$$
h_i^n=\sum_jW_{ij}p_{ij}^na_j^{n-d_{ij}}+(B_nu_n)_i+b_i-\theta_i,
$$

$$
a_i^{n+1}=(1-\lambda_i)a_i^n+\lambda_i\phi_i(h_i^n)
$$

를 미분하면 current row는

$$
(A_n)_{i,(j,d)}
=(1-\lambda_i)\delta_{ij}\mathbf1[d=0]
+\lambda_i\phi_i'(h_i^n)W_{ij}p_{ij}^n\mathbf1[d=d_{ij}],
$$

$$
(\mathcal B_n)_{i\mu}
=\lambda_i\phi_i'(h_i^n)(B_n)_{i\mu}.
$$

history row는

$$
(A_n)_{(i,d+1),(j,d)}=\delta_{ij},
\qquad d=0,\ldots,D-1
$$

인 shift register다. 따라서 $A_n\in\mathbb R^{q(D+1)\times q(D+1)}$, $\mathcal B_n\in\mathbb R^{q(D+1)\times m}$이다.

$b_i$와 $\theta_i$는 이 동역학에서 $c_i=b_i-\theta_i$로만 나타난다. 독립 calibration receipt가 없으면 둘은 구조적으로 식별 불가능하다. 즉 이 식 자체가 개별 threshold를 추정해 주는 것이 아니라, 측정된 threshold가 있으면 이를 넣을 자리를 제공한다.

### P0 반례: activity-dependent efficacy 누락

$q=1$, $D=0$, $\lambda=W=1$, $p(a)=(a+1)/2$, $\phi=\tanh$, $a=1/2$로 둔다. $p=0.75$, $h=0.375$, $\phi'(h)=0.8715799750$이다. frozen-$p$ 식은

$$
A_{\rm frozen}=\phi'(h)Wp=0.6536849813
$$

를 주지만 실제 미분은

$$
\frac{d}{da}\tanh(p(a)a)
=\phi'(h)W\left(p+a\frac{dp}{da}\right)
=0.8715799750.
$$

따라서 frozen-$p$ A6.3은 $\delta p=0$이라는 조건부 식이다. 가소성을 켜면 $p,e$를 상태에 넣어야 한다.

## 3. passive flow-pullback 정리

고정 history와 input path 아래 nonlinear flow map을 $F_T:a_0\mapsto a_T$라 한다. 그 derivative가

$$
J_T=D F_T(a_0)=P\Phi(T,0)\iota
$$

이고 terminal reference metric이 $G_T\succ0$이면

$$
g_{\rm pass}=F_T^*G_T=J_T^\top G_TJ_T.
$$

임의의 $v\ne0$에 대해

$$
v^\top g_{\rm pass}v=(J_Tv)^\top G_T(J_Tv).
$$

$J_T$가 injective이면 $J_Tv\ne0$이므로 우변이 양수이고 $g_{\rm pass}\succ0$이다. 반대로 $J_Tv=0$인 $v\ne0$가 있으면 $v^\top g_{\rm pass}v=0$이므로 positive definite가 아니다. 따라서

$$
g_{\rm pass}\text{가 Riemannian}
\iff \operatorname{rank}J_T=q.
$$

rank loss 시 결과는 degenerate PSD tensor다. 작은 ridge를 넣어 수치적으로 invertible하게 만드는 것은 원래 flow의 immersion 성질을 복원하지 않는다.

### 방향·주축·부피 response

초기 기준계량 $G_0\succ0$에 대해

$$
s(v)=\sqrt{\frac{v^\top g_{\rm pass}v}{v^\top G_0v}}
$$

는 방향 $v$의 infinitesimal stretch다. 회로 전후 계량이 모두 SPD이면

$$
g_{\rm post}v_m=\Lambda_mg_{\rm pre}v_m
$$

의 $\sqrt{\Lambda_m}$가 principal stretch ratio다. 같은 coordinate chart를 쓸 때

$$
\Delta\log V_g
=\frac12\log\frac{\det g_{\rm post}}{\det g_{\rm pre}}
=\frac12\sum_m\log\Lambda_m.
$$

예를 들어 $G=I$, $J=\operatorname{diag}(2,1/2)$이면 $g=\operatorname{diag}(4,1/4)$다. 한 방향은 2배 늘고 다른 방향은 절반으로 줄지만 $\det g=1$이라 전체 volume은 보존된다. 따라서 scalar 하나만으로는 anisotropic pull을 설명할 수 없다.

## 4. 회로 형성의 1차 응답 정리

$W(\varepsilon)=W_0+\varepsilon C_\Gamma$로 새 circuit $\Gamma$를 켠다. 나머지 coefficient와 initial history를 고정하면

$$
\dot h_i^n
=\sum_j\left[(C_\Gamma)_{ij}p_{ij}^na_j^{n-d_{ij}}
+W_{ij}(\varepsilon)p_{ij}^n\dot a_j^{n-d_{ij}}\right],
$$

$$
\dot a_i^{n+1}
=(1-\lambda_i)\dot a_i^n
+\lambda_i\phi_i'(h_i^n)\dot h_i^n,
\qquad \dot a_i^n=0\;(n\le0).
$$

따라서

$$
(\dot A_n)_{i,(j,d)}
=\lambda_i\left[
\phi_i''(h_i^n)\dot h_i^nW_{ij}(\varepsilon)p_{ij}^n
+\phi_i'(h_i^n)(C_\Gamma)_{ij}p_{ij}^n
\right]\mathbf1[d=d_{ij}],
$$

이고 shift block derivative는 0이다. 또한

$$
(\dot{\mathcal B}_n)_{i\mu}
=\lambda_i\phi_i''(h_i^n)\dot h_i^n(B_n)_{i\mu}
$$

이며 history row는 0이다. 이 total derivative에 product rule을 적용하면

$$
\dot\Phi(T,0)
=\sum_{r=0}^{T-1}
\Phi(T,r+1)\dot A_r\Phi(r,0).
$$

따라서

$$
\dot J_T=P\dot\Phi(T,0)\iota,
$$

$$
\dot g_\Gamma
=\dot J_T^\top G_TJ_T+J_T^\top G_T\dot J_T
$$

이다. 이것이 “회로 하나가 생겼을 때 manifold가 얼마나 끌리는가”의 local tensor 답이다. $g\succ0$이면

$$
\frac{d}{d\varepsilon}\log V_g
=\frac12\operatorname{tr}(g^{-1}\dot g)
$$

도 성립한다. delay $d_{ij}$는 정수 topology 변수라 미분하지 않고, 서로 다른 delay-augmented maps의 finite comparison으로 처리한다.

## 5. finite-horizon minimum-energy 정리

$$
\delta\xi_{n+1}=A_n\delta\xi_n+\mathcal B_n\delta u_n
$$

에서 zero initial perturbation을 두고

$$
H_T=
\begin{bmatrix}
\Phi(T,1)\mathcal B_0&\cdots&\mathcal B_{T-1}
\end{bmatrix},
\quad
\bar R_T=\operatorname{diag}(R_0,\ldots,R_{T-1})
$$

라 하면 $v=H_TU$이고

$$
\mathcal W_c(T)=H_T\bar R_T^{-1}H_T^\top\succeq0.
$$

$\bar R_T\succ0$이므로 $\operatorname{Im}\mathcal W_c=\operatorname{Im}H_T$다. $y=\bar R_T^{1/2}U$로 치환하면

$$
\min_{H_TU=v}U^\top\bar R_TU
$$

는 $H_T\bar R_T^{-1/2}y=v$를 만족하는 최소 Euclidean norm 문제다. Moore–Penrose minimum-norm identity에 따라

$$
E_T^*(v)=
\begin{cases}
v^\top\mathcal W_c(T)^\dagger v,&v\in\operatorname{Im}H_T,\\
+\infty,&v\notin\operatorname{Im}H_T.
\end{cases}
$$

이 정리는 endpoint value만 증명한다. physical/anatomical metric이나 nonlinear global reachability를 증명하지 않는다.

## 6. 회로 형성의 control-energy 응답

$\mathcal W_c\succ0$이고 $B,R$이 $\varepsilon$에 대해 고정되면

$$
\dot H_T=
\begin{bmatrix}
\dot\Phi(T,1)\mathcal B_0+\Phi(T,1)\dot{\mathcal B}_0&
\cdots&\dot{\mathcal B}_{T-1}
\end{bmatrix},
$$

따라서

$$
\dot{\mathcal W}_c
=\dot H_T\bar R_T^{-1}H_T^\top
+H_T\bar R_T^{-1}\dot H_T^\top.
$$

inverse derivative $\dot W^{-1}=-W^{-1}\dot W W^{-1}$를 적용하면

$$
\dot E_T^*(v)
=-v^\top\mathcal W_c^{-1}\dot{\mathcal W}_c
\mathcal W_c^{-1}v.
$$

음수면 지정된 actuator·cost·horizon 아래에서 그 endpoint direction이 쉬워진다. rank가 바뀌면 이 derivative는 쓰지 않고 reachable subspace와 finite energy를 따로 비교한다.

## 7. 반례와 수치 경계

1. **입력 없음:** $B_n=0$이면 $\mathcal W_c=0$이다. 회로 $W$가 커도 external-input reachability는 없다.
2. **포화:** $\phi'(h)=0$이면 $\mathcal B_n=0$이고 tangent propagation도 rank를 잃을 수 있다.
3. **rank loss:** $J=\operatorname{diag}(1,0)$이면 $J^\top J=\operatorname{diag}(1,0)$으로 Riemann metric이 아니다.
4. **nonnormal transient:** 

   $$
   A=\begin{bmatrix}0.9&10\\0&0.9\end{bmatrix},
   \quad B=\begin{bmatrix}0\\1\end{bmatrix},
   \quad T=2
   $$

   에서

   $$
   \mathcal W_c=
   \begin{bmatrix}100&9\\9&1.81\end{bmatrix},
   \qquad E^*(e_1)=0.0181.
   $$

   eigenvalue 둘은 모두 $0.9$지만 off-diagonal transient가 energy를 크게 바꾼다.
5. **actuator dependence:** scalar one-step system은 $E^*(v)=Rv^2/B^2$다. 같은 $W$라도 $B,R$이 바뀌면 energy geometry가 바뀐다.
6. **hard threshold:** projection boundary나 binary event threshold에서는 ordinary derivative가 정의되지 않는다. smooth model, saltation matrix 또는 nonlinear finite perturbation을 사용해야 한다.

## 8. 무차원성

| 양 | 정규화 후 차원 |
|---|---|
| $a,h,\theta,b,W,p,\lambda$ | 1 |
| tick $n$, integer delay $d$ | 1 |
| $u,B,R,A,\mathcal B,J,G,g,\mathcal W_c$ | 1 |
| $s,\Lambda,\rho,\Delta\log V_g$ | 1 |

log에는 양의 에너지 비, determinant 비, generalized eigenvalue처럼 무차원인 비만 들어간다. raw voltage·current·seconds를 사용할 경우 각각의 기준척도로 먼저 정규화하고 $R$의 단위를 그 선택과 일치시켜야 한다.

## 9. 최종 수학 지위

- A6.1--A6.3: **[정의 + 조건부 미분 산출]**. frozen-$p,B$와 smoothness가 필요하다.
- A6.4--A6.7: **[조건부 정리]**. full-rank domain에서 passive pullback Riemann geometry다.
- A6.7a--A6.7c: **[조건부 산출]**. 회로 parameterization과 동일 chart가 필요하다.
- A6.8--A6.10a: **[조건부 정리]**. fixed LTV model의 endpoint minimum-energy다.
- A6.11: **[공리: 모델 선택]**. 별도 augmented/hybrid 검증이 필요하다.
- A6.12: **[정의]**. 실제 cortical embedding 자료가 없으므로 A6-P와의 bridge는 **[미완성]**이다.
