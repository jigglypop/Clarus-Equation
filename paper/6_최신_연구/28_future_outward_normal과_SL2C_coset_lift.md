# 28. future/outward normal과 $SL(2,\mathbb C)/SU(2)$ coset lift

27장의 방향 spinor와 국소 LS 벡터는 tetrahedron 내부의 자료다. 그것만으로는 Lorentzian 4차원 공간에서 어느 시간 방향을 택하는지, 한 fine cell에서 어느 쪽이 바깥인지를 정하지 못한다. 이 장은 고정 유리수 Lorentzian $1\to5$ 증인에서 exact cofactor로 normal line을 만들고, 그 미래 단위 대표의 회전 없는 $SL(2,\mathbb C)/SU(2)$ 대표를 구성한다.

이 단계가 필요한 까닭은 future와 outward가 서로 다른 부호 자료이기 때문이다. 먼저 cofactor가 세 tangent를 정확히 annihilate함을 보이고 미래 unit normal을 정한다. 다음으로 incidence별 outward sign을 두어 shared tetrahedron의 반례를 확인한다. 마지막으로 pure boost와 positive-Hermitian $SL(2,\mathbb C)$ lift를 유도한 뒤, 기계 잔차와 proper/EPRL에 남은 일을 분리한다.

이 글은 안정화된 원장 `C4-LORENTZIAN-ONE-TO-FIVE-FRAME-LIFTS-28A--D`를 읽기 전용 입력으로 쓴다. 전제는 24장의 비퇴화 Lorentzian gluing 증인이고, 결론은 그 한 유한 유리수 증인의 normal 및 rotation-free coset 자료다. 임의 triangulation, full $SU(2)$ frame, proper vertex 또는 다섯-vertex EPRL 진폭의 정리는 아니다.

## 28.1 exact cofactor normal

좌표공간은 $\mathbb Q^4$, metric은 $\eta=\operatorname{diag}(-1,1,1,1)$로 둔다. spacelike tetrahedron $t=(v_0,v_1,v_2,v_3)$의 정렬된 꼭짓점 좌표를 $p_{v_a}$라 하고, $q_i=p_{v_i}-p_{v_0}$, $i=1,2,3$를 세 tangent vector라 한다. 24장의 gluing 전제에서 이들이 span하는 부분공간은 양의 정부호다. 따라서 그 $\eta$-직교 여공간은 timelike한 한 직선이다.

$q_i$를 행으로 둔 $3\times4$ 행렬에서 $I$번째 열을 지운 행렬을 $Q_{\widehat I}$라 하자. 다음 유리수 covector가 그 직선을 정확히 준다.

$$
(c_t)_I=(-1)^I\det Q_{\widehat I},\qquad
c_t^\sharp=(-c_{t,0},c_{t,1},c_{t,2},c_{t,3}).
\tag{1}
$$

필요하면 둘에 동시에 $-1$을 곱해 $(c_t^\sharp)^0>0$로 정한다.

**보조정리 28.1.** $c_t(q_i)=0$이 $i=1,2,3$에 대해 정확히 성립한다.

**증명.** 식 (1)을 대입하고 마지막 행으로 Laplace 전개하면

$$
\begin{aligned}
c_t(q_i)
 &=\sum_{I=0}^3(-1)^I\det Q_{\widehat I}(q_i)^I
 &&\text{cofactor 정의}\\
 &=-\det\!\begin{pmatrix}q_1\\q_2\\q_3\\q_i\end{pmatrix}
 &&\text{마지막 행 전개의 전체 부호}\\
 &=0
 &&\text{마지막 행이 이미 있는 $i$번째 행과 같음}.
\end{aligned}
$$

그러므로 $c_t^\sharp$는 orthogonal complement의 영이 아닌 벡터다. tetrahedron이 spacelike이므로 이 벡터는 timelike이고, 정한 부호 때문에 future-directed다. $\square$

정규화에는 float을 쓰기 전에 exact near-null gate를 둔다. $M_t=\max_I|(c_t^\sharp)^I|$이면

$$
\frac{M_t^2}{-\eta(c_t^\sharp,c_t^\sharp)}\le10^{24}
\tag{2}
$$

를 요구하고, 통과한 뒤에만

$$
n_t=\frac{c_t^\sharp}{\sqrt{-\eta(c_t^\sharp,c_t^\sharp)}},
\qquad \eta(n_t,n_t)=-1,\qquad n_t^0>0
\tag{3}
$$

로 만든다. 식 (2)는 분모가 너무 작아 normalization이 부동소수 오차를 증폭하는 경우를 막는다. base witness의 최대값은 정확히 $500000/499573$이며 gate보다 작다. 이 수치는 이 증인의 안전한 정규화 산출이지 일반 condition-number 정리가 아니다.

## 28.2 outward sign은 future choice가 아니다

cell $v$ 안의 tetrahedron incidence $(v,t)$에서, $t$에 없는 cell 꼭짓점까지의 displacement를 $d_{vt}$라 하자. 비퇴화성 때문에 $c_t(d_{vt})\ne0$이고, outward side를

$$
\varepsilon_{vt}=-\operatorname{sgn}\bigl(c_t(d_{vt})\bigr),
\qquad N^{\rm out}_{vt}=\varepsilon_{vt}n_t
\tag{4}
$$

로 정의한다. 음수 부호는 $N^{\rm out}_{vt}$가 opposite vertex가 놓인 쪽의 반대, 즉 cell 밖을 향하게 하는 규약이다.

internal tetrahedron이 두 fine cell에 공유되는 경우가 반례다. globally sorted tetrahedron에서 cofactor와 $n_t$를 한 번 정했으므로 양쪽 incidence의 future representative는 같다. 그러나 두 opposite vertex는 hyperplane의 반대쪽에 있으므로, 어떤 $a>0$에 대해

$$
\begin{aligned}
c_t(d_+)&=a,\qquad c_t(d_-)=-a
&&\text{두 cell이 서로 반대쪽에 있음},\\
\varepsilon_+&=-1,\qquad\varepsilon_-=+1
&&\text{식 (4)의 정의},\\
N^{\rm out}_{+t}&=-n_t,\qquad N^{\rm out}_{-t}=+n_t
&&\text{outward normal의 결과}.
\end{aligned}
\tag{5}
$$

따라서 같은 future normal과 반대 outward normal은 동시에 참이다. 하나의 outward normal은 past-directed일 수 있다. base witness에서는 열 internal tetrahedron마다 $a=9/2500$이고, 두 exact evaluation은 부호만 반대다. 이 결과는 shared face의 quantum bra--ket dualization이나 spinor phase gluing을 구성했다는 뜻이 아니다.

## 28.3 pure boost

식 (3)의 미래 normal을 $n=(\gamma,\boldsymbol\nu)$라 쓴다. $\gamma>0$이고 unit 조건은 $\gamma^2-\lVert\boldsymbol\nu\rVert^2=1$이다. 회전을 넣지 않는 변환을

$$
\Lambda(n)=
\begin{pmatrix}
\gamma&\boldsymbol\nu^{\mathsf T}\\
\boldsymbol\nu&I_3+\dfrac{\boldsymbol\nu\boldsymbol\nu^{\mathsf T}}{1+\gamma}
\end{pmatrix}
\tag{6}
$$

로 정의한다. 첫 번째 열에서 $\Lambda(n)e_0=n$이다. $K=I_3+\boldsymbol\nu\boldsymbol\nu^{\mathsf T}/(1+\gamma)$라 쓰면 $K\boldsymbol\nu=\gamma\boldsymbol\nu$이다. 따라서 $\Lambda^{\mathsf T}\eta\Lambda$의 혼합 block은 $-\gamma\boldsymbol\nu^{\mathsf T}+\boldsymbol\nu^{\mathsf T}K=0$, 시간 block은 $-\gamma^2+\lVert\boldsymbol\nu\rVert^2=-1$이다. 공간 block도

$$
\begin{aligned}
-\boldsymbol\nu\boldsymbol\nu^{\mathsf T}+K^2
 &=I_3+\left(\frac{2}{1+\gamma}
 +\frac{\lVert\boldsymbol\nu\rVert^2}{(1+\gamma)^2}-1\right)
 \boldsymbol\nu\boldsymbol\nu^{\mathsf T}
 &&\text{$K$를 전개}\\
 &=I_3
 &&\text{$\lVert\boldsymbol\nu\rVert^2=(\gamma-1)(\gamma+1)$}.
\end{aligned}
$$

그러므로 $\Lambda(n)^{\mathsf T}\eta\Lambda(n)=\eta$다. $e_0$에서 연속적으로 출발하고 $\Lambda^0{}_0>0$이므로 proper orthochronous 성분에 속한다.

## 28.4 positive-Hermitian $SL(2,\mathbb C)$ lift

Pauli 행렬을 $\sigma_i$라 하고 $\boldsymbol\nu\cdot\boldsymbol\sigma=\sum_i\nu^i\sigma_i$라 하자. 식 (6)의 boost lift를

$$
X(n)=\frac{(1+\gamma)\mathbf1+
\boldsymbol\nu\cdot\boldsymbol\sigma}{\sqrt{2(1+\gamma)}}
\tag{7}
$$

로 둔다. 분자의 고유값은 $(1+\gamma)\pm\lVert\boldsymbol\nu\rVert$로 모두 양수이므로 $X(n)$은 positive Hermitian이다. Pauli 항등식 $(\boldsymbol\nu\cdot\boldsymbol\sigma)^2=\lVert\boldsymbol\nu\rVert^2\mathbf1$에서

$$
\begin{aligned}
\det X(n)
 &=\frac{(1+\gamma)^2-\lVert\boldsymbol\nu\rVert^2}{2(1+\gamma)}
 &&\text{$a\mathbf1+\boldsymbol b\cdot\boldsymbol\sigma$의 determinant}\\
 &=1
 &&\text{unit-timelike 조건},
\end{aligned}
\tag{8}
$$

$$
\begin{aligned}
X(n)X(n)^\dagger
 &=\frac{\bigl((1+\gamma)\mathbf1+\boldsymbol\nu\cdot\boldsymbol\sigma\bigr)^2}{2(1+\gamma)}
 &&\text{$X(n)$이 Hermitian}\\
 &=\gamma\mathbf1+\boldsymbol\nu\cdot\boldsymbol\sigma
 &&\text{Pauli 항등식과 unit 조건}.
\end{aligned}
\tag{9}
$$

따라서 $X(n)\in SL(2,\mathbb C)$이고, Hermitian vector $x^0\mathbf1+x^i\sigma_i$의 작용 $x\mapsto XxX^\dagger$는 $e_0$을 $n$으로 보낸다. 이 작용에서 유도한 Lorentz 행렬은 식 (6)과 같다. 같은 normal을 보내는 모든 lift는 $X(n)u$, $u\in SU(2)$ 꼴이다. positive-Hermitian 조건은 right gauge orbit에서 회전 없는 대표 하나만 고른다. tangent triad나 full $SU(2)$ frame은 만들지 않는다.

## 28.5 구현 재현과 남은 경계

[frame-lift 모듈](../../examples/physics/proper_vertex_one_to_five_frame_lifts.py)과 focused [회귀 테스트](../../tests/test_proper_vertex_one_to_five_frame_lifts.py)는 위 유한 증인을 materialize한다. base scale, $7/3$, $10^{-100}$에서 15개 unique tetrahedron과 25 incidences(5 boundary, 20 internal)를 확인한다. positive rational rescaling은 normal direction, boost 및 lift를 바꾸지 않는다.

focused test의 **8 passed**는 exact tangent annihilation, timelike/future/outward 분리, shared internal의 정확한 반대 부호, scale invariance, near-null rejection, boost 및 lift residual을 회귀한 결과다. 이는 식 (1), 식 (5), 식 (6)--(9)의 수학 증명과 구별되는 구현 재현 근거다.

[Engle--Zipfel (2015), arXiv:1502.04640](https://arxiv.org/abs/1502.04640)은 Lorentzian proper vertex의 비퇴화 경계 자료와 단일-vertex semiclassical 분석의 외부 문맥을 준다. 이 장은 그 문헌의 full boundary frame이나 proper orientation을 재구성하지 않으며, 이 certificate를 단일-vertex 또는 multi-vertex amplitude의 증명으로 읽지 않는다.

독립 tetrahedron $SU(2)$ tangent frame과 cell-local $J$-dualized matching은 [29장](29_접공간_SU2_section과_cell_local_J_dual_gluing.md)의 coordinate section에서 추가했다. 그러나 global shared-face cocycle, discrete Levi--Civita/Regge holonomy와 Regge phase, face bivector 및 EPRL orientation equation, 27장의 local LS vector와 global frame의 결합, globally glued spin network, $Y_\gamma$, proper projector, Lorentzian group integral, standard proper EPRL five-vertex amplitude 및 multicell saddle/Hessian은 아직 구성하지 않았다. 이 장의 끝점은 normal과 rotation-free coset lift다.
