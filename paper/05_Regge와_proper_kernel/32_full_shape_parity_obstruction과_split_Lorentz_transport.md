# 32. full shape parity obstruction과 split Lorentz transport

31장은 이전 edge-aligned section이 두 번째 labelled edge까지 맞추지 못하는 24/26 반례를 보였다. 이 장은 그 실패를 정확한 determinant 부호로 다시 분류한다. 두 labelled triangle edge와 outward normal antipode를 모두 요구하면 map은 유일하지만, 그 map이 항상 proper rotation인 것은 아니다.

먼저 fixed pointwise label convention에서 full-shape map의 determinant를 유도한다. 다음으로 세 tetrahedron의 negative parity cycle이 per-tetrahedron parity repair를 막는다는 것을 보인다. 그 뒤 genuinely proper인 full Lorentz transition과 그 Wigner $SO(3)$ factor를 분리해, Lorentz cocycle가 닫혀도 Wigner section은 cocycle가 아니라는 사실을 수치 witness와 함께 적는다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-REGGE-FACES-33A--E`를 읽기 전용 입력으로 삼는다. 결론은 fixed rational witness와 globally sorted labels에 한정한 parity obstruction 및 split transport다. global Regge phase, EPRL orientation 또는 quantum amplitude의 결과가 아니다.

## 32.1 labelled full-shape map의 exact determinant

한 cell의 공유 triangle에서 source와 target face의 ordered orthonormal frame을 각각

$$
S=(a_1,a_2,n_s),
\qquad T=(b_1,b_2,-n_t)
\tag{1}
$$

로 둔다. $a_1,a_2$와 $b_1,b_2$는 globally sorted label에서 정한 같은 두 triangle edge의 방향이고, $n_s,n_t$는 각 tetrahedron의 outward face normal이다. 두 labelled edge와 normal antipode를 모두 보내는 유일한 map은

$$
R_{t\leftarrow s}=TS^{\mathsf T}.
\tag{2}
$$

이다.

이 부호를 float determinant로 판정하지 않는다. face의 opposite vertex 쪽 inward vector를 $d$라 하고, future normal을 $N$이라 하면 exact four-dimensional determinant

$$
D=\det(N,e_1,e_2,d)
\tag{3}
$$

의 부호는 proper Lorentz rest frame에서 $(e_1,e_2,d)$의 orientation을 정한다. outward normal은 inward의 반대이므로 face sign을 $\omega=-\operatorname{sgn}D$로 둔다. 식 (1)의 마지막 열에 $-n_t$를 썼으므로

$$
\begin{aligned}
\det R_{t\leftarrow s}
 &=\frac{\det T}{\det S}
 &&\text{식 (2)}\\
 &=\frac{-\omega_t}{\omega_s}\\
 &=-\omega_s\omega_t
 &&\text{$\omega_s^2=1$}.
\end{aligned}
\tag{4}
$$

따라서 모든 50 map은 exact하게 $\det R=+1$ 또는 $-1$로 분류된다. 이 witness에서는 24개가 $SO(3)$ proper rotation이고 26개가 improper reflection이다. $SU(2)$는 $SO(3)$만 double-cover하므로 26 reflection에는 $SU(2)$ lift가 없다.

## 32.2 negative cycle이 막는 global parity repair

각 tetrahedron에 parity 변수 $p_t\in\{+1,-1\}$를 한 번만 부여해 link 부호를 고치려 한다고 하자. pointwise labels를 유지하면서 link $(s,t)$를 proper로 만들려면 식 (4)의 부호 $q_{st}$에 대해

$$
p_sp_t=q_{st}
\tag{5}
$$

가 필요하다. 다음 명시 cycle을 보자.

$$
(1,3,4,5)\to(2,3,4,5)\to(0,3,4,5)\to(1,3,4,5).
\tag{6}
$$

세 edge의 exact 부호는 $(+,+,-)$다. 식 (5)를 세 번 곱하면 좌변의 각 $p_t$는 두 번 나타나므로 $+1$이어야 한다. 반면 우변은 $-1$이다. 따라서 모순이다.

$$
(p_ap_b)(p_bp_c)(p_cp_a)=+1\ne(+)(+)(-)=-1.
\tag{7}
$$

그러므로 global per-tetrahedron parity assignment는 없다. wedge마다 label 순서를 따로 바꾸면 어떤 link는 고칠 수 있다. 그러나 같은 tetrahedron의 convention이 wedge마다 달라지므로, 그것은 pointwise-labelled face gluing이 아니다. 이 반례가 삭제하는 것은 이 fixed ordering에서의 global pointwise-labelled $SU(2)$ face transport뿐이다. 일반 Regge/EPRL theory가 불가능하다는 주장은 아니다.

## 32.3 full Lorentz connection과 Wigner section

30장의 declared affine coframe에서는 full Lorentz transition

$$
H_{t\leftarrow s}=F_t^{-1}F_s\in SO^+(1,3)
\tag{8}
$$

가 proper-orthochronous이고 두 four-dimensional labelled triangle edge를 모두 transport한다. 식 (8)의 products는 exact affine construction에서 flat cocycle를 이룬다.

그러나 $H$에서 공간 회전을 읽는 일은 별도의 section 선택이다. $u=H e_0$라 하고 $B(u)$를 $e_0$을 $u$로 보내는 canonical pure boost라 하면

$$
W(H)=B(u)^{-1}H
=\begin{pmatrix}1&0\\0&R_W\end{pmatrix},
\qquad R_W\in SO(3).
\tag{9}
$$

가 된다. $R_W$는 각 link에서 proper rotation이지만 $H\mapsto R_W$는 group homomorphism가 아니다. boost $B(u)$가 link와 intermediate time axis에 의존하므로, Lorentz cocycle를 Wigner factor의 cocycle로 내릴 수 없다.

실제 fixed witness에서 Wigner factor는 이전 local rotation과 24개에서 일치하고 26개에서는 각 mismatch가 $\sqrt8$이다. full Lorentz cocycle residual은 $8\times10^{-12}$보다 작게 닫히지만 Wigner loop residual은 최소 약 $1.350083183\times10^{-6}$, 최대 약 $2.291688314\times10^{-4}$로 영이 아니다. 최대 witness는 cell $(0,1,5,3,4)$, omitted vertices $(1,3,4)$다. 이 값들은 curved holonomy나 Regge phase가 아니라 canonical-boost split이 만드는 local section의 noncocycle를 뜻한다.

positive scale $10^{-500}$과 $10^{500}$에서도 parity와 transport split은 보존된다. focused [Regge-face test](../../tests/test_proper_vertex_one_to_five_regge_faces.py)의 **5 passed**는 이 fixed witness의 재현 근거이지 위 반례의 범위를 넓히거나 quantum amplitude를 증명하지 않는다.

## 32.4 EPRL ceiling

[Engle--Zipfel (2015), arXiv:1502.04640](https://arxiv.org/abs/1502.04640)과 [Engle--Vilensky--Zipfel (2015), arXiv:1505.06683](https://arxiv.org/abs/1505.06683)은 proper Lorentzian single-vertex와 그 asymptotic 범위의 외부 문맥이다. 이 장의 parity obstruction이나 split Lorentz/Wigner transport가 그 문헌의 full Regge-like spinor data를 구성했다고 말할 수 없다.

global Regge spinor/action phase, full EPRL critical orientation, global pointwise-labelled $SU(2)$ face transport와 EPRL boundary state/network, $Y_\gamma$, proper projector, Lorentzian $SL(2,\mathbb C)$ integral, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum limit과 Einstein--Hilbert/two-DOF IR은 열려 있다.

[33장](33_incidence_spinor와_Cartan_dual_EH_sector.md)은 pointwise face normal에 cell--tetrahedron incidence sign을 넣은 새 노드 정의에서 local $SU(2)$ section을 구성한다. 이는 이 장의 fixed pointwise-labelled $24/26$ no-go를 반박하지 않는다. incidence closure와 Cartan-dual fixed-cell sector certificate도 physical Regge phase, EH action/dynamics, 또는 EPRL amplitude의 증명이 아니다.
