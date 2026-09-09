# 30. 전역 flat coframe connection과 zero boost deficit

29장의 cell-local transport는 각 pair를 맞추는 데에는 충분했지만, 세 transport를 이어도 identity가 되지 않는 negative control을 남겼다. 이 장은 다른 입력에서 출발한다. 다섯 fine simplex가 하나의 affine Minkowski 공간에 함께 매장돼 있다는 사실을 **선언된 모형 입력**으로 두고, 각 cell의 coframe을 그 공통 공간과 비교한다. 그러면 fixed flat witness에서 전역 cocycle와 hinge holonomy를 계산할 수 있다.

이 단계의 필요성은 “pairwise matching”과 “connection”을 구별하는 데 있다. 먼저 cell coframe $F_c$와 spin lift $G_c$를 정하고, 그 비로 shared-tetrahedron transition을 유도한다. 다음으로 inverse, triple cocycle, gauge covariance와 future/outward 부호를 확인한다. 마지막으로 열 internal triangle의 telescoping loop와 trace-gated boost deficit을 계산하고, 이것이 intrinsic 또는 curved Regge reconstruction이 아니라는 경계를 적는다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-GLOBAL-CONNECTION-31A--E`를 읽기 전용 입력으로 삼는다. 결과는 fixed globally embedded flat witness의 **gauge-dependent affine/Levi--Civita representative**다. common global affine coframe 자체를 선언했기 때문에, intrinsic 또는 gauge-independent connection을 재구성한 결과가 아니다.

## 30.1 선언한 공통 coframe과 transition

각 fine cell $c$는 unique boundary tetrahedron에서 정한 proper-orthochronous Lorentz coframe $F_c\in SO^+(1,3)$를 갖고, $SL(2,\mathbb C)$ lift $G_c$를 갖는다고 둔다. 여기서 $F_c$는 cell-local 성분을 공통 affine Minkowski 성분으로 보내는 행렬이다. 이 공통 target space가 존재한다는 명제는 24장의 fixed flat embedding에 추가한 선언이며, 좌표와 무관하게 도출한 사실이 아니다.

shared tetrahedron을 사이에 두고 source cell을 $s$, target cell을 $t$라 하면 source 성분을 target 성분으로 바꾸는 transition은

$$
H_{t\leftarrow s}=F_t^{-1}F_s\in SO^+(1,3),
\qquad
h_{t\leftarrow s}=G_t^{-1}G_s\in SL(2,\mathbb C).
\tag{1}
$$

이다. 각 $F_c,G_c$가 proper-orthochronous coframe과 그 lift이므로 식 (1)의 두 transition도 같은 군에 속한다.

**보조정리 30.1 (inverse와 cocycle).** 식 (1)은 inverse relation과 triple cocycle를 만족한다.

**증명.** 행렬의 역과 결합법칙만 사용하면

$$
\begin{aligned}
H_{s\leftarrow t}H_{t\leftarrow s}
 &=F_s^{-1}F_tF_t^{-1}F_s
 &&\text{식 (1)을 대입}\\
 &=\mathbf1,
\end{aligned}
$$

이고, 세 cell $s,t,u$에 대해서는

$$
\begin{aligned}
H_{u\leftarrow t}H_{t\leftarrow s}
 &=F_u^{-1}F_tF_t^{-1}F_s
 &&\text{식 (1)을 두 번 대입}\\
 &=F_u^{-1}F_s\\
 &=H_{u\leftarrow s}.
\end{aligned}
\tag{2}
$$

같은 계산에서 $F$를 $G$로 바꾸면 $h_{s\leftarrow t}h_{t\leftarrow s}=\mathbf1$ 및 $h_{u\leftarrow t}h_{t\leftarrow s}=h_{u\leftarrow s}$를 얻는다. $\square$

coframe section을 cell마다 $F_c\mapsto F_cK_c$, $G_c\mapsto G_ck_c$로 바꾸면 $K_c\in SO^+(1,3)$, $k_c\in SL(2,\mathbb C)$에 대해

$$
H_{t\leftarrow s}\mapsto K_t^{-1}H_{t\leftarrow s}K_s,
\qquad
h_{t\leftarrow s}\mapsto k_t^{-1}h_{t\leftarrow s}k_s.
\tag{3}
$$

가 된다. 따라서 closed-loop holonomy는 시작 cell의 gauge로 conjugate될 뿐이다. 이 gauge covariance는 gauge-independent reconstruction과 다르다. 후자를 주장하려면 coframe의 선택 없이 정의되는 자료가 별도로 필요하다.

## 30.2 shared tetrahedron에서 보존되는 것

shared tetrahedron의 global tangent vector를 $e$라 하자. 두 cell 성분은 $F_s^{-1}e$와 $F_t^{-1}e$이며,

$$
H_{t\leftarrow s}F_s^{-1}e
=F_t^{-1}F_sF_s^{-1}e
=F_t^{-1}e
\tag{4}
$$

가 성립한다. 고정 embedding의 exact shared shape/tangent matching은 식 (4)의 대상이 되는 같은 global tangent를 제공한다.

future normal도 같은 global vector $n_t$로 읽히므로 같은 계산으로 보존된다. 반면 outward normal은 cell incidence에 의존한다. 28장의 부호 규약에 따라 두 shared incidence의 global outward normal은 서로 반대이므로,

$$
H_{t\leftarrow s}\bigl(F_s^{-1}N^{\rm out}_{s}\bigr)
=-F_t^{-1}N^{\rm out}_{t}.
\tag{5}
$$

이다. outward-to-outward 등식을 요구하면 부호가 틀린다. 식 (5)의 antipode가 올바른 shared-face relation이다.

## 30.3 열 hinge loop의 telescoping과 zero deficit

internal spacelike triangle 하나에는 세 fine cell이 incident한다. 순서를 $(c_0,c_1,c_2)$로 잡으면 dual loop holonomy는

$$
\begin{aligned}
\mathcal H
 &=H_{c_0\leftarrow c_2}H_{c_2\leftarrow c_1}H_{c_1\leftarrow c_0}\\
 &=F_{c_0}^{-1}F_{c_2}F_{c_2}^{-1}F_{c_1}
   F_{c_1}^{-1}F_{c_0}\\
 &=\mathbf1.
\end{aligned}
\tag{6}
$$

spin lift도 같은 방식으로 $\mathfrak h=\mathbf1$로 telescope한다. 이 fixed $1\to5$ witness에는 internal triangle이 열 개이므로 식 (6)은 열 loop 각각에 적용한다. 여기서 identity는 declared common affine frame의 결과다. curved geometry의 deficit을 새로 계산해 없앤 것이 아니다.

spacelike hinge의 tangent plane을 $\mathcal H$가 고정하는지 먼저 검사한다. 그 다음 orthogonal $1+1$ plane에서 holonomy가 real boost라고 판정된 경우에만

$$
\cosh\delta=\frac{\operatorname{tr}\mathcal H-2}{2},
\qquad
\delta=\operatorname{arcosh}\!\left(\frac{\operatorname{tr}\mathcal H-2}{2}\right)
\tag{7}
$$

로 boost deficit을 읽는다. trace-domain gate $({\operatorname{tr}\mathcal H-2})/2\ge1$를 통과하지 못하면 clamp해서 deficit을 선언하지 않는다. 여기서는 식 (6)으로 $\operatorname{tr}\mathcal H=4$이므로 gate의 좌변은 $1$이고 $\delta=0$이다.

## 30.4 29장의 $2.828$ loop는 왜 반대 결과가 아닌가

29장의 50개 cell-local edge rotation은 공통 coframe의 비로 만든 link가 아니다. 그 section에서 pairwise loop residual 최대값은 약 $2.8284271247$이며, 이는 pairwise edge alignment를 global connection으로 읽을 수 없다는 negative control이다.

반대로 식 (6)의 loop는 five declared affine coframe의 transition으로 만든 열 internal-hinge holonomy다. 두 loop는 링크의 정의와 loop의 기저가 다르다. 따라서 $2.8284271247$을 열 identity holonomy의 반례나 Regge curvature로 해석하면 안 된다.

## 30.5 재현 출력과 남은 경계

base witness에서 최대 cell-frame, transition, cocycle, holonomy residual은 각각 약 $1.0213\times10^{-15}$, $1.81494\times10^{-15}$, $7.2466\times10^{-16}$, $7.8679\times10^{-16}$다. positive rational scale $10^{-500}$과 $10^{500}$에서도 transition은 보존된다. focused [global-connection test](../../tests/test_proper_vertex_one_to_five_global_connection.py)의 **6 passed**는 이 고정 witness의 재현 근거일 뿐, 앞 절의 행렬 증명 또는 물리적 curvature reconstruction의 증거가 아니다.

Regge state와 proper single-vertex 범위를 구분하는 외부 문맥은 [Engle--Zipfel (2015), arXiv:1502.04640](https://arxiv.org/abs/1502.04640) 및 [Engle--Vilensky--Zipfel (2015), arXiv:1505.06683](https://arxiv.org/abs/1505.06683)을 따른다. 이 문헌은 이 장의 declared affine coframe을 intrinsic connection이나 five-vertex amplitude로 승격하지 않는다.

[31장](31_exact_oriented_bivector와_full_shape_반례.md)은 이 fixed flat gauge에서 exact oriented classical bivector와 cross-cell transport를 추가하고, 이전 local section의 full labelled-shape 반례를 제시한다. 그러나 global Regge spinor phase, global EPRL boundary state, $Y_\gamma$, proper projector, Lorentzian $SL(2,\mathbb C)$ integral, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum limit과 Einstein--Hilbert/two-DOF IR은 열려 있다. 이 장의 끝점은 고정 globally embedded flat gauge에서의 affine/Levi--Civita representative다.
