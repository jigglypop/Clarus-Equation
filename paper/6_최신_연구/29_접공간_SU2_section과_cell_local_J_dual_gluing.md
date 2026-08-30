# 29. 접공간 $SU(2)$ section과 cell-local $J$-dual gluing

28장이 future normal과 회전 없는 boost를 정했다면, 다음 질문은 그 normal의 3차원 rest space 안에서 어떤 방향축을 쓸 것인가이다. 이 장은 고정 Lorentzian $1\to5$ 유리수 증인에서 right-handed tangent triad와 그 $SU(2)$ lift를 만들고, 한 fine cell 안에서 공유 triangle을 읽는 두 tetrahedron 사이의 $J$-dualized spinor matching을 구성한다.

이 순서가 필요한 까닭은 sorted Cholesky chart가 저절로 $SU(2)$ lift 가능한 방향을 주지 않기 때문이다. 먼저 15개 chart 가운데 7개가 improper라는 유한 반례를 보인다. 그 뒤 coordinate section으로 right-handed frame을 새로 정하고, 각 cell의 10개 tetrahedron pair에 edge-aligned transport와 $U(1)$ phase convention을 붙인다. 마지막으로 reverse regression과 nontrivial loop 반례를 통해 이 국소 구성이 global Regge transport가 아님을 분명히 한다.

이 장은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-TANGENT-FRAMES-29A--D` 및 `C4-LORENTZIAN-ONE-TO-FIVE-CELL-LOCAL-GLUING-30A--G`를 읽기 전용 입력으로 삼는다. 결론은 고정 좌표, vertex ordering, quaternion sign 및 phase convention에 의존하는 **cell-local section**이다. global cocycle, discrete Levi--Civita/Regge holonomy와 Regge phase를 얻었다는 결론은 포함하지 않는다.

## 29.1 sorted Cholesky chart가 주는 유한 반례

각 spacelike tetrahedron의 future-normal rest space에서 sorted edge 세 개를 행으로 둔 행렬을 $E$라 하자. 그 normalized Gram matrix가 $EE^{\mathsf T}=LL^{\mathsf T}$가 되도록 Cholesky row matrix $L$을 고르면, Cholesky axes와 rest-edge axes를 비교하는 직교 행렬은

$$
O=E^{\mathsf T}(L^{\mathsf T})^{-1},
\qquad O^{\mathsf T}O=\mathbf1.
\tag{1}
$$

이다. 직교라는 사실은 orientation-preserving을 뜻하지 않는다. 이 fixed witness의 15개 $O$를 계산하면 determinant가 $+1$인 것은 8개, $-1$인 것은 7개다.

$SU(2)$의 adjoint action은 $U\sigma_jU^\dagger=R_{ij}\sigma_i$로 $R\in SO(3)$만 낸다. 그러므로 $\det O=-1$인 7개 chart에는 $SU(2)$ lift가 없다. “모든 sorted-Cholesky chart가 canonical $SU(2)$ lift를 갖는다”는 부모 명제는 이 한 rational witness에서 반례다. 이 개수는 임의 triangulation의 orientation 통계나 물리 gauge obstruction을 뜻하지 않는다.

## 29.2 right-handed coordinate section

반례 뒤에는 lift 가능한 축을 별도로 선택해야 한다. tetrahedron $t=(v_0,v_1,v_2,v_3)$에서 28장의 boost로 세 sorted edge를 future-normal rest space로 보낸 것을 $e_1,e_2,e_3$라 하자. 첫 두 edge가 독립이라는 이 witness의 전제 아래 Gram--Schmidt로

$$
\begin{aligned}
r_1&=\frac{e_1}{\lVert e_1\rVert},\\
\widetilde r_2&=e_2-r_1(r_1\cdot e_2),\\
r_2&=\frac{\widetilde r_2}{\lVert\widetilde r_2\rVert},\\
r_3&=r_1\times r_2,
\qquad R_t=(r_1,r_2,r_3)\in SO(3).
\end{aligned}
\tag{2}
$$

로 정한다. 마지막 cross product가 right-handedness를 강제하므로, 식 (1)의 improper chart를 lift하려 하지 않는다.

quaternion의 첫 nonzero 성분을 양수로 택하는 canonical sign 규약으로 $U_t\in SU(2)$를 골라

$$
U_t\sigma_jU_t^\dagger=(R_t)_{ij}\sigma_i.
\tag{3}
$$

를 만족시킨다. 28장의 positive-Hermitian boost lift를 $X_t$라 하면 full local section은

$$
G_t=X_tU_t\in SL(2,\mathbb C),
\qquad
\Lambda_t\operatorname{diag}(1,R_t)e_0=n_t.
\tag{4}
$$

이다. 식 (4)는 각각의 tetrahedron에서 full Lorentz frame 하나를 제공한다. vertex sorting, 앞의 두 edge, quaternion sign을 바꾸면 section도 바뀐다. 따라서 이것은 right $SU(2)$ gauge를 물리적으로 고정한 결과나 gauge-independent observable이 아니다.

## 29.3 한 cell 안의 edge-aligned transport

한 fine cell에는 다섯 tetrahedron이 있고, unordered pair마다 common triangle이 하나 있다. 그러므로 cell 하나에서 10개, 다섯 cell에서 50개의 cell-local $SO(3)/SU(2)$ transport record가 생긴다. 각 triangle의 두 tangent axis를 $(a_1,a_2)$, outward normal을 $\nu_a$라 쓰고 다른 tetrahedron 쪽 자료를 $(b_1,b_2,\nu_b)$라 하자. 첫 edge axis를 맞추고 outward normal을 antipodal하게 보내려면

$$
s_{ab}=-\epsilon_a\epsilon_b,
\qquad
R_{ab}=(b_1,s_{ab}b_2,-\nu_b)(a_1,a_2,\nu_a)^{\mathsf T}
\tag{5}
$$

로 둔다. 여기서 $\epsilon_a,\epsilon_b\in\{+1,-1\}$는 두 face frame의 orientation sign이다. 양쪽 괄호 안 행렬은 같은 orientation을 가지도록 $s_{ab}$가 보정하므로 $R_{ab}\in SO(3)$이며,

$$
R_{ab}\nu_a=-\nu_b
\tag{6}
$$

가 성립한다. 이 증인에서는 second tangent sign을 보존하는 record가 24개, 뒤집는 record가 26개다. 각각에 canonical-sign $SU(2)$ lift $U_{ab}$를 붙였지만, $R_{ab}$는 cell-local edge-aligned section일 뿐 shared internal face의 cell-independent connection이 아니다. [31장](31_exact_oriented_bivector와_full_shape_반례.md)의 24/26 검사처럼 이 section은 full labelled triangle gluing이나 Regge phase를 주지 않는다.

## 29.4 $J$-dual과 phase convention

정규화 spinor $\xi=(z_0,z_1)$에 anti-linear map을

$$
J\xi=(-\overline z_1,\overline z_0)
\tag{7}
$$

로 정의한다. 정의를 두 번 적용하면

$$
\begin{aligned}
J^2\xi
 &=J(-\overline z_1,\overline z_0)
 &&\text{식 (7)을 다시 적용}\\
 &=(-z_0,-z_1)
 &&\text{복소켤레 계산}\\
 &=-\xi.
\end{aligned}
\tag{8}
$$

또한 $\lVert J\xi\rVert^2=|z_1|^2+|z_0|^2=\lVert\xi\rVert^2$다. Pauli direction을 $n_i(\xi)=\xi^\dagger\sigma_i\xi$로 정의하면, 성분은 $(2\operatorname{Re}\overline z_0z_1,2\operatorname{Im}\overline z_0z_1,|z_0|^2-|z_1|^2)$다. 식 (7)을 대입하면 세 성분 모두 부호가 바뀌므로

$$
n(J\xi)=-n(\xi).
\tag{9}
$$

식 (6)과 식 (9)는 $U_{ab}\xi_{ab}$와 $J\xi_{ba}$가 같은 Pauli direction을 갖게 한다. 두 정규화 spinor의 nonzero overlap을

$$
z=\langle J\xi_{ba},U_{ab}\xi_{ab}\rangle,
\qquad
\lambda_{ab}=\overline{z/|z|}
\tag{10}
$$

로 두면, 이 장에서 **정의로**

$$
\lambda_{ab}U_{ab}\xi_{ab}=J\xi_{ba}
\tag{11}
$$

가 되도록 representative를 고른다. $\lambda_{ab}$는 한 local record의 남은 $U(1)$ phase를 제거하는 convention이다. [31장](31_exact_oriented_bivector와_full_shape_반례.md)의 full-shape 반례 뒤에도 이 정의는 유지한다. Regge action phase를 정의하거나 계산하거나 일치시킨 결과가 아니다.

## 29.5 reverse regression과 loop 반례

ordered pair를 뒤집으면 section이 최소한 역연산과 양립해야 한다. 50개 record에서 직접 확인한 regression은

$$
R_{ba}=R_{ab}^{\mathsf T},
\qquad
U_{ba}=\pm U_{ab}^{-1}.
\tag{12}
$$

이다. 두 번째 부호는 $SU(2)\to SO(3)$ double cover와 canonical-sign 선택에서 생긴다. 식 (12)는 reverse pair의 회귀 검사이며, triple transport의 cocycle 조건은 아니다.

실제로 fine cell 안의 ordered tetrahedron triple $(a,b,c)$에 대해

$$
\left\|R_{ca}R_{bc}R_{ab}-\mathbf1\right\|
\tag{13}
$$

를 50개 loop에서 계산하면 최대값은 약 $2.8284271247>1$이다. 따라서 pairwise local section만으로 trivial triangle loop를 주장할 수 없다. 이 반례는 선택한 section의 nontrivial loop를 보일 뿐 Regge curvature, deficit angle 또는 discrete Levi--Civita holonomy를 계산한 것은 아니다.

## 29.6 재현 범위와 남은 단계

tangent-frame과 bra--ket focused tests, 그리고 앞선 boundary/frame-lift focused tests를 함께 실행하면 **30 passed**다. tangent-frame의 최대 residual은 약 $1.95\times10^{-14}$이고 $J$-dualized matching의 최대 residual은 약 $1.36\times10^{-15}$다. 이는 15개 right-handed tangent section, 50개 cell-local transport, 식 (8)--(12)의 구현 회귀 근거다. 수학적 정의와 유도는 코드의 성공 횟수와 다른 층에 있다.

[30장](30_전역_flat_coframe_connection과_zero_deficit.md)은 declared common affine flat embedding에서 gauge-dependent global coframe cocycle와 zero-deficit hinge transport를 추가했다. 그러나 이것은 intrinsic/gauge-independent shared-face cocycle이나 curved Regge holonomy가 아니다. global Regge phase, face bivector와 EPRL orientation equation, globally glued spin network, $Y_\gamma$, proper projector, Lorentzian $SL(2,\mathbb C)$ integral, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum limit과 Einstein--Hilbert/two-DOF IR은 여전히 열려 있다. 그러므로 이 장은 local coordinate section과 $J$-dualized matching의 끝점이다.
