# 37. QFT M0--M2: 공변 기준장과 고전 제약의 경계

이 문서는 양자론과 일반상대론을 한 식으로 합쳤다고 주장하지 않는다. M0--M2의 첫 결과, 곧 일반공변 작용에서 출발한 네 개의 기준 스칼라와 그 고전 정준 제약을 독자가 재현할 수 있게 정리한다. 먼저 왜 기준장이 필요한지 읽고, 다음으로 작용과 ADM 분해를 따라가며, 마지막으로 전역 좌표ㆍ양자화ㆍ관측량에서 아직 남은 경계를 확인하는 순서가 좋다.

필요한 이유는 “누가 어느 시간에 무엇을 보았는가”를 좌표 라벨이 아니라 장의 관계로 써야 하기 때문이다. 그러나 기준장을 넣었다는 사실만으로 보편 플랑크 틱, 관측자 독립 시간, 양자 이상항 없는 중력, 또는 숨은 중첩 성분의 에너지 원천이 생기지는 않는다. 여기서 확정한 고전 결과와 보류한 양자 결과의 공식 지위는 읽기 전용 원장 [E54](../검증_원장/참조_양자_보존_원장.md#qnb-e54-a)에 있다.

## 1. 무엇을 고정했는가

시공간은 $M=\mathbb R\times\Sigma$, 계량 부호는 $(-,+,+,+)$로 둔다. $g_{\mu\nu}$는 중력 계량, $\chi$는 보통 물질 스칼라, $X^A$ ($A=0,1,2,3$)는 사건을 서로 비교할 때 사용할 네 개의 기준 스칼라다. $X^0$를 시계, $X^i$를 막대라고 부를 수 있지만 이는 허용된 국소 patch에서의 역할 이름이지, 모든 관찰자가 공유하는 절대 시간의 선언이 아니다.

M1의 기준선은 양의 internal metric을 둔 다음 작용이다. $V(\chi)=m^2\chi^2/2+\lambda\chi^4/4!$, $m^2,\lambda\geq0$, $\mu_X^2>0$로 둔다.

$$
\begin{aligned}
S={}&\int_M d^4x\,\sqrt{-g}\left[
 {M_P^2\over2}(R-2\Lambda)
-{1\over2}\nabla_\mu\chi\nabla^\mu\chi-V(\chi)
-{\mu_X^2\over2}\delta_{AB}\nabla_\mu X^A\nabla^\mu X^B\right]\\
&+M_P^2\int_{\partial M}d^3y\,\sqrt{|h|}\,K .
\end{aligned}
\tag{37.1}
$$

마지막 항은 유한한 시간 경계를 둘 때 유도 계량을 고정하는 Dirichlet 변분 원리를 위한 Gibbons--Hawking--York 항이다. 이 경계항의 원전은 [Gibbons--Hawking (1977)](https://doi.org/10.1103/PhysRevD.15.2752)이다. 이 문서의 정준 계산은 경계가 없는 compact $\Sigma$에서 하므로, 경계 전하를 없앴다고 주장하지 않는다.

자연단위 $\hbar=c=1$에서 $[x]=-1$이며 $[M_P]=[\mu_X]=[m]=1$, $[\Lambda]=2$, $[\chi]=1$, $[X]=[\lambda]=0$이다. 따라서 (37.1)의 각 항은 질량차원 $4$를 가져야 한다. 이 선택은 기준장 $X^A$를 무차원 라벨로 쓰되 그 기울기와 $\mu_X$가 응력과 정준 운동량의 차원을 운반하게 한다.

## 2. ADM 분해가 주는 것

공간 계량 $q_{ab}$, lapse $N$, shift $N^a$를 도입하고

$$
K_{ab}={1\over2N}(\dot q_{ab}-D_aN_b-D_bN_a)
\tag{37.2}
$$

로 정의한다. 여기서 $D_a$는 $q_{ab}$의 공변미분이다. (37.1)을 $3+1$로 분해하면 계량의 정준 운동량은

$$
\pi^{ab}={M_P^2\sqrt q\over2}\left(K^{ab}-q^{ab}K\right)
\tag{37.3}
$$

가 된다. $1/2$는 Einstein--Hilbert 항의 정규화에 따른 계수다. 이 계수를 빼면 아래 해밀토니안 제약의 중력 항도 바뀌므로, 식 (37.3)은 단순한 표기 선택이 아니다. ADM 정준 형식의 출발점은 [Arnowitt--Deser--Misner](https://arxiv.org/abs/gr-qc/0405109)에서 확인할 수 있다.

$N$과 $N^a$에는 시간미분이 없으므로 이들은 동역학 장이 아니라 제약 승수다. 그 결과 해밀토니안 및 운동량 제약은

$$
\begin{aligned}
\mathcal C={}&{2\over M_P^2\sqrt q}\left(\pi^{ab}\pi_{ab}-{\pi^2\over2}\right)
-{M_P^2\sqrt q\over2}\left({}^{(3)}R-2\Lambda\right)
+{p_\chi^2\over2\sqrt q}
+\sqrt q\left({q^{ab}\partial_a\chi\partial_b\chi\over2}+V\right)\\
&+{\delta^{AB}P_AP_B\over2\mu_X^2\sqrt q}
+{\mu_X^2\sqrt q\over2}\delta_{AB}q^{ab}\partial_aX^A\partial_bX^B\approx0,\\
\mathcal C_a={}&-2D_b\pi^b{}_a+p_\chi\partial_a\chi+P_A\partial_aX^A\approx0.
\end{aligned}
\tag{37.4}
$$

여기서 $p_\chi$와 $P_A$는 각각 $\chi$와 $X^A$의 운동량이다. $q_{ab}$의 여섯 성분, $\chi$ 하나, $X^A$ 넷에서 시작해 nondynamical lapse/shift를 제거하고 네 개의 first-class secondary constraint를 적용하면 물리적 configuration 자유도는 $11-4=7$이다. 즉 graviton 둘, $\chi$ 하나, 기준 스칼라 넷이다. 이 수는 기준장이 gauge fixing 뒤에 저절로 사라진다는 뜻이 아니다.

## 3. 고전 대수는 닫히지만 양자 대수는 아직 아니다

$H[N]=\int_\Sigma N\mathcal C$, $D[\vec N]=\int_\Sigma N^a\mathcal C_a$로 놓으면 smooth classical phase space에서

$$
\begin{aligned}
\{D[\vec N],D[\vec M]\}&=D[[\vec N,\vec M]],\\
\{D[\vec N],H[M]\}&=H[\mathcal L_{\vec N}M],\\
\{H[N],H[M]\}&=D[q^{ab}(N\partial_bM-M\partial_bN)].
\end{aligned}
\tag{37.5}
$$

이는 hypersurface-deformation algebra의 고전 결과이며 [Teitelboim (1973)](https://doi.org/10.1016/0003-4916(73)90196-1)의 범위에 속한다. 구조함수 $q^{ab}$가 나타난다는 점이 보통 Lie 대수와 다르다. 따라서 (37.5)는 양자 연산자 교환자에 regularization을 택해도 이상항이 없다는 증명이 아니다. M2의 고전 부분은 완료했지만, quantum anomaly와 physical inner product는 미완성이다.

기준장 응력은

$$
T^{(X)}_{\mu\nu}=\mu_X^2\delta_{AB}\left(\nabla_\mu X^A\nabla_\nu X^B-{1\over2}g_{\mu\nu}\nabla_\rho X^A\nabla^\rho X^B\right)
\tag{37.6}
$$

이고 $X$의 고전 장방정식 위에서만 $\nabla^\mu T^{(X)}_{\mu\nu}=0$이다. 이 on-shell 항등식은 반고전 Ward identity나 양자 중력의 응력 보존을 대신하지 않는다. 또한 kinetic 부호가 양수여도 특정 상태의 $\nabla_\mu X^A$는 preferred frame을 만들 수 있다. 작용의 공변성과 상태의 대칭성은 별도 문제다.

## 4. 전역 막대의 반례와 국소 patch

$\Sigma$가 compact이고 경계가 없다고 하자. 세 전역 $\mathbb R$-값 막대 $X^i$가 모든 곳에서 rank $3$이라면 $X:\Sigma\to\mathbb R^3$는 local submersion이라 상이 open이다. 반면 compact 공간의 연속상은 compact이므로, 비어 있지 않은 compact open subset of $\mathbb R^3$가 생긴다는 모순이 난다. 그러므로 이 조건의 전역 막대는 존재하지 않는다. 이 명제는 모든 전역 네 스칼라 map을 금지하는 것이 아니라, 전역 좌표계처럼 쓸 수 있는 세 막대의 요구만 막는다.

따라서 M3는 local reference atlas, compact-target rod, mixed embedding 가운데 하나를 명시적으로 택해야 한다. 국소 기준 patch의 최소 판정식은 법선과 공간 접선 방향으로 만든

$$
B^A{}_{\bar\mu}=\left(n^\rho\partial_\rho X^A,\ \partial_aX^A\right),
\qquad
\det B=N^{-1}\det(\partial_\mu X^A)\ne0
\tag{37.7}
$$

이다. 이 행렬식이 영이 아닌 patch에서만 $X^A$를 사건을 비교하는 기준으로 쓸 수 있다. 전역 chart를 몰래 가정하면 이 반례를 피해 갈 수 없다.

## 5. 스칼라 시계는 조건부 제곱근이다

$T=X^0$를 timelikeㆍmonotonic patch에서 택하고 $\mathcal C_{\setminus P_T}$를 $\mathcal C$에서 $P_T^2/(2\mu_X^2\sqrt q)$ 항만 뺀 나머지로 둔다. 그러면

$$
P_T=s\sqrt{-2\mu_X^2\sqrt q\,\mathcal C_{\setminus P_T}},
\qquad s\in\{+1,-1\}
\tag{37.8}
$$

이라고 쓸 수 있다. 다만 이는 (37.7)의 patch에서 radicand가 음이 아니고, $s$를 고정하며, $P_T\ne0$이고

$$
q^{ab}\partial_aT\partial_bT<\left[{P_T\over\mu_X^2\sqrt q}\right]^2
\tag{37.9}
$$

를 만족할 때만 허용된다. 즉 보통의 스칼라 시계는 자동으로 전역 선형 제약 $P_T+H=0$을 주지 않는다. 이 구별은 scalar-clock의 국소 분기와 dust에 의한 deparametrization을 비교한 [Giesel--Thiemann (2012)](https://arxiv.org/abs/1206.3807), [Giesel--Vetter (2016)](https://arxiv.org/abs/1610.07422)와도 맞닿아 있다.

## 6. 사라지는 기준장의 유혹과 다음 kill gate

$\mu_X\to0$을 취하면 기준장이 관측에 거의 영향을 주지 않으면서 QFT를 회복할 것처럼 보일 수 있다. 그러나 fixed gradient에서는 $T^{(X)}_{\mu\nu}$와 $P_A$가 작아질 수 있는 반면 Legendre map의 rank가 퇴화한다. fixed $P_A$에서는 (37.4)의 $P_A^2/(2\mu_X^2\sqrt q)$가 발산한다. 따라서 이 극한은 regular한 “기준장을 끈” 한계가 아니다.

비교를 위해 Brown--Kuchař dust는 dust momentum을 선형 시간 제약으로 풀 수 있는 경로를 제공한다. 하지만 dust는 실제 응력과 물질 흐름을 지니며 caustic 문제도 부담한다. [Brown--Kuchař (1995)](https://arxiv.org/abs/gr-qc/9409001)는 그 장점과 비용의 원전이다. 현재 기준선은 dust를 정답으로 채택하지 않고, 네 양의 kinetic 스칼라를 QFT 회복의 공변 baseline으로 유지한다.

다음 M2 quantum kill gate는 명확하다. 하나의 regularization과 연산자 영역을 고정한 뒤, 물리적 inner product와 함께 제약 연산자들이

$$
[\hat C[\xi],\hat C[\eta]]
=i\hbar\hat C[[\xi,\eta]_{\rm HD}]+\mathcal A[\xi,\eta],
\qquad \mathcal A[\xi,\eta]=0
\tag{37.10}
$$

를 만족함을 보이거나, 남는 이상항을 제어할 수 없음을 확인해야 한다. 실패하면 M3의 관계적 관측량, M4의 측정, M6의 반고전 회복으로 진행하지 않는다. 이 단계는 양자론과 일반상대론 사이의 다리를 이미 건넜다는 선언이 아니라, 그 다리가 무너지는 정확한 위치를 고정한 결과다.

## 참고와 지위

이 글의 식과 상태 표기는 [E54 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e54-a)을 읽기 전용으로 따른다. 현재 확정 범위는 M1의 공변 baseline과 M2의 고전 ADMㆍHDA 계산이며, 예측은 없다. 양자 이상항, 전역 branch 일관성, 관계적 관측량의 dressing/경계 gluing, 표준 QFT 및 반고전 Einstein 회복은 아직 열려 있다.
