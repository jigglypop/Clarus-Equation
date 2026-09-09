# 31. exact oriented bivector와 full shape 반례

30장은 선언된 공통 affine coframe의 transition이 flat loop에서 telescope함을 보였다. 그러나 connection만으로 shared triangle의 두 labelled edge와 spinor phase가 Regge 자료가 되는 것은 아니다. 이 장은 각 cell/triangle wedge의 고전 oriented bivector를 exact matrix convention으로 구성하고, 이전 local section이 full labelled triangle shape를 모두 glue하지 못한다는 반례를 함께 제시한다.

먼저 exact wedge, Hodge dual, cell orientation sign을 정의한다. 다음으로 outward-normal, left rest-frame, right rest-frame이라는 세 경로가 같은 signed normal-plane bivector를 준다는 것을 확인한다. 그 뒤 declared flat connection으로 cross-cell bivector를 transport하고 internal loop를 닫는다. 마지막으로 24개 성공과 26개 실패를 분리해, 무엇을 삭제하고 무엇을 남기는지 밝힌다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-BIVECTORS-32A--F`를 읽기 전용 입력으로 삼는다. 범위는 fixed flat affine embedding의 고전 bivector 자료다. full EPRL critical orientation, Regge action phase 또는 양자 amplitude의 결과가 아니다.

## 31.1 exact matrix convention과 cell orientation

signature를 $(-+++)$, Levi--Civita 기호를 $\epsilon_{0123}=+1$로 고정한다. globally sorted triangle의 exact edge를 $e_1,e_2$라 하면 antisymmetric matrix

$$
\Sigma_{IJ}=(e_1\wedge e_2)_{IJ}=e_{1I}e_{2J}-e_{2I}e_{1J}
\tag{1}
$$

를 만든다. 면적과 canonical normal-plane bivector는

$$
A^2=\frac18\Sigma_{IJ}\Sigma^{IJ},
\qquad B_0=\frac12\star\Sigma
\tag{2}
$$

로 정한다. 식 (1)에서 $\Sigma_{IJ}=-\Sigma_{JI}$이고, reverse label은 wedge의 부호를 바꾼다. 이들은 float 계산보다 먼저 유리수 행렬에서 성립하는 등식이다.

cell 안에서 triangle 양쪽의 omitted vertex displacement를 $d_L,d_R$라 쓴다. cell의 고전 orientation sign은

$$
\varepsilon_c=-\operatorname{sgn}\det(e_1,e_2,d_L,d_R),
\qquad B_c=\varepsilon_c B_0
\tag{3}
$$

로 둔다. $d_L,d_R$를 맞바꾸면 determinant와 $\varepsilon_c$가 반전한다. 식 (3)은 fixed coordinate embedding의 signed classical orientation 규약이다. EPRL critical orientation이나 action phase를 정의하지 않는다.

## 31.2 같은 bivector를 얻는 세 경로

한 cell 안에서 triangle을 공유하는 left/right tetrahedron의 outward unit normal을 $N_L^{\rm out},N_R^{\rm out}$라 하자. 먼저 두 normal에서 normal-plane bivector를 만들고 unit norm으로 맞춘다.

$$
\widehat B_N=-\operatorname{unit}\!\left(N_L^{\rm out}\wedge N_R^{\rm out}\right).
\tag{4}
$$

다음으로 각 tetrahedron의 rest frame에서 time axis를 $e_0$, triangle outward spatial normal을 $\nu_L,\nu_R$라 쓴다. incidence sign을 포함한 rest-simple representatives를 global frame으로 되돌리면

$$
\begin{aligned}
\widehat B_L
 &=F_L\left[-e_0\wedge(0,\varepsilon_L\nu_L)\right]F_L^{\mathsf T},\\
\widehat B_R
 &=F_R\left[-e_0\wedge(0,\varepsilon_R\nu_R)\right]F_R^{\mathsf T}.
\end{aligned}
\tag{5}
$$

fixed witness의 outward/future convention과 식 (3)에서 세 경로는

$$
\widehat B_N=\widehat B_c=\widehat B_L=-\widehat B_R
\tag{6}
$$

로 합친다. 마지막 부호는 같은 triangle을 두 tetrahedron의 서로 반대 normal side에서 읽기 때문에 생긴다. 따라서 signed classical orientation equation은 $\widehat B_L+\widehat B_R=0$이다.

각 future normal $n$에 대해 unit bivector는 선형 simplicity도 만족한다.

$$
n_I(\star\widehat B_c)^{IJ}=0.
\tag{7}
$$

식 (7)은 이 고전 normal-plane bivector가 주어진 tetrahedron normal에 단순하다는 뜻이다. face bivector의 EPRL orientation equation이나 quantum critical-point equation은 아니다.

## 31.3 flat coframe 아래의 bivector transport

30장의 $H_{t\leftarrow s}=F_t^{-1}F_s$를 사용하면, canonical bivector의 cell 성분은

$$
B_t=H_{t\leftarrow s}B_sH_{t\leftarrow s}^{\mathsf T}.
\tag{8}
$$

로 transport한다. $SL(2,\mathbb C)$ lift의 induced Lorentz adjoint action도 같은 식을 준다. fixed witness에서는 40개 cross-cell comparison에서 exact $B_0$가 일치하고, canonical 및 signed representative가 식 (8)과 일치한다. 이는 common affine embedding에서 같은 고전 bivector를 다른 cell gauge로 쓴 pure-gauge consistency다.

열 internal triangle의 세-cell loop도 식 (8)을 세 번 적용하면 30장의 transition telescope 때문에 시작 bivector로 돌아온다. base witness의 최대 wedge, cross-cell transport, loop residual은 각각 약 $1.304219\times10^{-15}$, $2.230806\times10^{-15}$, $5.43944\times10^{-16}$다. scale $10^{-500}$과 $10^{500}$에서도 scale-free route가 보존된다. focused [bivector test](../../tests/test_proper_vertex_one_to_five_bivectors.py)의 **6 passed**는 이 산출의 재현 근거일 뿐, intrinsic curvature나 quantum amplitude의 증거가 아니다.

## 31.4 full labelled shape의 반례와 남는 명제

29장의 edge-aligned local section은 첫 labelled triangle edge와 outward normal antipode를 맞춘다. 그러나 두 번째 labelled edge까지 같은 rotation으로 맞추는 full shape gluing을 검사하면, 50개 record 중 24개만 통과하고 26개가 실패한다. 실패한 second-edge mismatch는 최소 약 $0.622905$, 최대 $2$이며, concrete witness $\mathrm{triangle}=(2,3,4)$에서는 약 $1.521777$이다.

이 반례가 삭제하는 명제는 좁다. 선택한 local edge-aligned section이 unique full Regge gluing 또는 Regge/action phase를 준다는 주장은 더 이상 유지하지 않는다. 일반 Regge/EPRL 모형이 불가능하다는 no-go는 얻지 않았다. 남는 결과는 first-edge alignment, normal antipode, 그리고

$$
\lambda_{ab}U_{ab}\xi_{ab}=J\xi_{ba}
\tag{9}
$$

를 정한 **local $U(1)$ representative convention**이다. 여기서 $\lambda_{ab}$는 29장에서 정의한 overlap phase이며 Regge phase가 아니다.

full Regge-like boundary data와 proper single-vertex 범위를 구분하는 외부 문맥은 [Engle--Zipfel (2015), arXiv:1502.04640](https://arxiv.org/abs/1502.04640) 및 [Engle--Vilensky--Zipfel (2015), arXiv:1505.06683](https://arxiv.org/abs/1505.06683)을 따른다. 이 장은 그 문헌이 다루는 full orientation 또는 amplitude 조건을 달성했다고 주장하지 않는다.

## 31.5 남은 경계

[32장](32_full_shape_parity_obstruction과_split_Lorentz_transport.md)은 24/26 full-shape map의 exact parity obstruction과 full Lorentz/Wigner split을 추가한다. 이 결과도 global Regge spinor/action phase, full EPRL critical orientation, global EPRL state/network, $Y_\gamma$, proper projector, Lorentzian $SL(2,\mathbb C)$ integral, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum limit과 Einstein--Hilbert/two-DOF IR을 대체하거나 증명하지 않는다.
