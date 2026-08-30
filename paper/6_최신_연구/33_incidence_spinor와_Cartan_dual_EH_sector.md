# 33. incidence spinor와 Cartan dual EH sector

32장의 $24/26$ 결과는 globally sorted pointwise label과 tetrahedron face normal만을 노드 자료로 삼으면 26개 full-shape map이 improper라는 정확한 반례였다. 이 장의 목적은 그 반례를 없던 일로 만들지 않으면서, 실제 wedge가 갖는 cell--tetrahedron incidence orientation을 노드 자료에 포함할 때 어떤 $SU(2)$ section이 닫히는지를 재구성하는 데 있다. 이어서 그 section 위에서 유한 $Y_\gamma$ 표본과 proper-projector의 부호 반례를 확인하고, Cartan dual branch가 어떤 고전 sector를 선택하는지 분리한다.

이 순서는 필요하다. pointwise obstruction 뒤에 spinor, projector, Einstein--Hilbert(EH)라는 말을 바로 붙이면 서로 다른 세 부호를 혼동하기 쉽다. 그래서 먼저 incidence normal을 정의하고 100개 노드의 link/cycle을 닫는다. 다음으로 50개 homogeneous $Y_\gamma$ 표본과 50개 유한 projector를 읽고, 마지막에 독립 rank-$6$ 연속체 재구성으로 $(\omega,\nu,\mu)$를 판정한다. 여기의 EH는 **orientation/Plebanski sector**의 이름이며 EH action이나 dynamics의 유도가 아니다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-INCIDENCE-SPINORS-34A--F`를 읽기 전용 입력으로 사용한다. 수치는 고정된 다섯 fine 4-simplex의 재현 witness를 뜻한다. global Regge spinor/action phase나 proper EPRL amplitude의 결론으로 읽으면 안 된다.

## 33.1 pointwise 반례와 incidence 해법은 서로 다른 명제다

32장에서 한 face의 normal을 $n_{tf}$라 쓰고, 두 labelled edge와 $-n_{tf}$를 보내는 map의 determinant를 검사했다. 그 정의에서는 50 map 중 24개만 $SO(3)$에 있고 26개는 reflection이다. 세 link 부호 $(+,+,-)$의 cycle도 있으므로 tetrahedron마다 한 번 정한 parity로 이를 동시에 고칠 수 없다.

그러나 cell $c$ 안에서 tetrahedron $t$가 차지하는 outward incidence sign을 $\epsilon_{ct}\in\{\pm1\}$로 두면, oriented incidence normal은

$$
\widetilde n_{ctf}=\epsilon_{ct}n_{tf}
\tag{1}
$$

이다. 이제 노드는 face $f$ 하나가 아니라 $(c,t,f)$다. 이 한 항이 빠진 옛 pointwise 자료와 새 자료를 동일시할 수 없다. 같은 cell에서 shared face의 두 incidence는 반대 방향을 가져, full-shape frame

$$
S=(e_1,e_2,\widetilde n_{csf}),\qquad
T=(e_1,e_2,-\widetilde n_{ctf})
\tag{2}
$$

의 orientation이 일치한다. 따라서

$$
\begin{aligned}
R_{(c,t,f)\leftarrow(c,s,f)}&=TS^{\mathsf T},\\
R^{\mathsf T}R&=I,\\
\det R&=+1.
\end{aligned}
\tag{3}
$$

식 (3)은 26 reflection을 parity로 수선한 것이 아니다. 그것은 normal의 정의역을 pointwise face에서 oriented incidence로 바꾼 결과다. 그러므로 32장의 “fixed pointwise-labelled global $SU(2)$ gluing 없음”은 그대로 남고, 여기서 얻는 것은 incidence-labelled section이다.

Pauli map $\xi\mapsto\xi^\dagger\boldsymbol\sigma\xi$로 식 (1)의 단위 방향을 projective spinor line $[\xi_{ctf}]$로 올린다. $R$의 $SU(2)$ lift를 $U$라 하면 phase를 다음 방정식을 만족하도록 **정의**할 수 있다.

$$
e^{i\lambda}U\xi_{csf}=J\xi_{ctf}.
\tag{4}
$$

여기서 $J$는 표준 anti-linear spinor dual이다. $\lambda$는 linkwise $U(1)$ convention이며 Regge action phase가 아니다.

## 33.2 무엇이 실제로 닫혔는가

다섯 cell에는 각 $5\times4$ oriented incidence가 있어 노드 수는 $100$이다. 그 위에 50개의 within-cell proper $SO(3)/SU(2)$ link와, internal shared tetrahedron을 common rest gauge에서 비교하는 40개의 cross-cell identity link를 둔다. 10개 internal triangle마다 alternating six-cycle 하나가 생긴다.

cycle $\ell_1\cdots\ell_6$의 회전, lift, phase-corrected spinor product를 각각 곱하면 확인한 최대 residual은

$$
\begin{array}{c|cccc}
\text{검사}&\text{link}&SO(3)\ \text{cycle}&SU(2)\ \text{cycle}&U(1)\ \text{cycle}\\ \hline
\max\ \text{residual}&1.77663\!\times\!10^{-15}&1.85293\!\times\!10^{-15}&5.93625\!\times\!10^{-16}&7.62119\!\times\!10^{-16}
\end{array}
\tag{5}
$$

이다. positive scale $10^{-500}$ 및 $10^{500}$에서 같은 section을 재현한 결과도 gauge convention의 안정성만 뜻한다. curved holonomy, global cocycle의 물리적 해석, 또는 Regge phase가 식 (5)에서 따라오지 않는다.

## 33.3 유한 $Y_\gamma$ 표본과 projector가 말하는 부호

level $3$, $\gamma=0.274$에서 각 incidence face에 finite integer spin witness를 붙이고, lowest $SU(2)$ type

$$
(k,p)=(j,\gamma j)
\tag{6}
$$

의 homogeneous $Y_\gamma$ coherent-state formula를 한 점에서 평가했다. 50개 표본의 최대 residual은 $2.66802\times10^{-15}$다. 이는 published homogeneous formula의 finite sample이지 infinite-dimensional principal-series representation 전체나 globally contracted $Y_\gamma$ state가 아니다.

같은 50 incidence에 positive spectral proper-projector matrix를 유한 차원에서 만들었다. 각 matrix는 rank $j$이고 zero eigenvalue 하나를 갖고, 최대 idempotence residual은 $2.05441\times10^{-15}$다. 그런데 선택한 원래 branch의 Engle--Zipfel sector scalar는 모두

$$
q_{ab}<0,\qquad \min|q_{ab}|=0.00207851
\tag{7}
$$

이다. 그래서 positive projector는 선택 coherent state를 죽이고 그 $J$-dual을 보존한다. 이는 “현재 선택한 incidence boundary가 positive proper sector”라는 주장에 대한 유한 반례다. projector의 정의나 proper vertex amplitude 자체의 반례는 아니다.

## 33.4 Cartan dual이 바꾸는 것

원래 frame에 Cartan involution

$$
X\longmapsto (X^\dagger)^{-1}
\tag{8}
$$

을 적용하고 parity-related bivector candidate를 함께 택한다. 이 dual frame들은 여전히 proper-orthochronous이고 $\beta$ sign은 원래 branch와 일치한다. critical equation, orientation equation, parity-bivector relation을 별도로 검사한 뒤 sector scalar만 다시 계산하면 50개 모두 $q_{ab}>0$가 된다. 따라서 dual branch의 positive projector는 선택 coherent state를 보존한다.

이는 원래 branch와 gauge-equivalent한 이름 바꾸기가 아니다. 다섯 cell의 dual solution은 원래 해와 분리되어 있으며, 이 절의 결론은 고정 coframe/bivector 입력에 대한 projector-positive reconstruction candidate다. EH action을 평가했거나 stationary action equation의 dynamics를 증명한 것은 아니다.

## 33.5 독립 rank-$6$ sector 판정

projector 부호만으로 Plebanski sector를 정하지 않는다. 각 cell에서 여섯 antisymmetric coordinate-face component를 독립적으로 재구성해 coordinate-face matrix의 rank가 $6$임을 확인한다. reconstructed tensor의 orientation scalar 부호를 $\omega$, Hodge-tetrad 비교를 $\nu$, 그리고 $\mu=\omega\nu$로 둔다.

$$
\begin{aligned}
\text{원래 branch}:\quad &(\omega,\nu,\mu)=(-1,+1,-1),\\
\text{Cartan dual}:\quad &(\omega,\nu,\mu)=(+1,+1,+1).
\end{aligned}
\tag{9}
$$

식 (9)의 순서는 $(\omega,\nu,\mu)$이며, dual의 $(+1,+1,+1)$만이 이 fixed-cell certificate에서 positive EH orientation/Plebanski sector candidate를 고른다. 이것은 전역 EH action, dynamics, 또는 IR two-degree-of-freedom theorem이 아니다.

## 33.6 남겨 둔 경계

[Engle--Zipfel (2015), arXiv:1502.04640](https://arxiv.org/abs/1502.04640)은 proper Lorentzian single-vertex의 nondegenerate Regge-like boundary data와 asymptotic 문맥을 제시한다. 위의 local incidence certificate는 그 문헌이 요구하는 global construction을 대신하지 않는다.

남은 항목은 LS 자료와 incidence spinor의 global boundary state/network contraction, physical global Regge spinor/action phase, full principal-series $Y_\gamma$ representation, proper sector boundary projector insertion, gauge-fixed single-vertex $SL(2,\mathbb C)$ integral, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum dynamics, 그리고 EH/two-DOF IR이다. 이 장의 100-node closure, finite samples, $q$-flip, rank-$6$ certificate는 그 목록을 닫지 않는다.

[34장](34_single_cell_proper_kernel_contract.md)은 이 목록 가운데 한 cell의 finite LS vector, Eq.-(53) target projector endpoint, root-gauge-fixed relative frame만 contract로 정한다. $\alpha$ evaluation, principal-series action, pointwise integrand 및 Haar integral을 수행하지 않으므로 이 경계는 유지된다.
