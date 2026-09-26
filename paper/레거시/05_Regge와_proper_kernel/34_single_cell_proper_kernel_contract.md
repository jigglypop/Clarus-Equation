# 34. single-cell proper kernel contract

이 장은 33장의 Cartan-dual positive branch에서 **정확히 한** fine 4-simplex만 떼어 proper-vertex kernel에 넣을 수 있는 유한 입력을 계약으로 정리한다. 목표는 amplitude 값을 내는 일이 아니다. 다섯 tetrahedron의 LS recoupling-coordinate intertwiner, 열 face의 target projector, 그리고 root gauge-fixed critical frame이 서로 어떤 endpoint에 놓이는지를 독자가 추적할 수 있게 만드는 일이다.

이 순서가 필요한 까닭은 “projector를 만들었다”와 “proper EPRL integral을 계산했다”가 전혀 다른 말이기 때문이다. 먼저 한 cell의 조합론과 LS vector의 한계를 세고, 이어 Engle--Zipfel Eq. (53)에서 $\alpha$와 $\Pi_{ba}$의 자리를 고정한다. 마지막으로 root gauge fixing이 남기는 네 $SL(2,\mathbb C)$ factor를 세고, 아직 적분자가 없다는 경계를 적는다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-PROPER-KERNEL-35A--D`를 읽기 전용 입력으로 쓴다. 여기서 ‘kernel contract’는 유한 boundary tensor와 endpoint convention의 산출이며, pointwise integrand나 Haar integral의 다른 이름이 아니다.

## 34.1 한 cell에서 실제로 있는 자료

고정 $1\to5$ witness에서 fine 4-simplex 하나를 고르면 tetrahedron은 5개다. 임의의 두 tetrahedron이 face 하나를 공유하므로 unoriented face는

$$
\binom52=10
\tag{1}
$$

개, 방향을 주어 endpoint를 구별한 incidence는 $2\times10=20$개다. 각 tetrahedron에 one normalized LS recoupling-coordinate vector를 둔다.

$$
|\iota_a\rangle
=\sum_k c_{a,k}|k\rangle,
\qquad \sum_k|c_{a,k}|^2=1.
\tag{2}
$$

식 (2)는 four-valent invariant recoupling basis에서 만든 **다섯 유한 vector**다. 이 finite level의 spin-weighted geometric closure defect는 영으로 확인되지 않았다. 따라서 식 (2)를 exact spin-weighted closure, 독립 tetrahedron frame의 gluing, 또는 Haar group average의 수치 적분으로 읽지 않는다.

## 34.2 Eq. (53)의 endpoint를 뒤집지 않기

[Engle--Zipfel의 Eq. (53)](https://arxiv.org/abs/1502.04640)는 proper Lorentzian vertex의 face factor에 invariant bilinear pairing $\alpha$와 positive projector를 둔다. 이 장의 ordered face $(a,b)$, $a<b$ convention에서는 projector가 **target $ba$ endpoint**의 ket에 먼저 작용한다.

$$
\alpha\!\left(
X_a\mathcal I|j_{ab},\xi_{ab}\rangle,
X_b\mathcal I\Pi_{ba}^{(+)}|j_{ba},\xi_{ba}\rangle
\right).
\tag{3}
$$

여기서 $\alpha$는 아직 수치 평가하지 않은 두 principal-series state의 invariant pairing이다. 식 (3)이 고정하는 것은 $\Pi_{ba}^{(+)}$가 두 번째 인자의 target ket에 작용한다는 endpoint policy다. 논문의 covariance 항등식 없이 projector를 $ab$ endpoint나 pairing 밖으로 옮겨서는 안 된다. 이 선택은 face마다 두 endpoint의 spin이 같다는 finite label check와 함께 보존된다.

Cartan-dual critical data에서는 열 face 모두

$$
q_{ba}>0,\qquad
\left\|\Pi_{ba}^{(+)}|j_{ba},\xi_{ba}\rangle\right\|=1
\tag{4}
$$

이며 target-projector residual과 critical spinor equation residual이 tolerance 아래로 닫힌다. 식 (4)는 선택 target ket 보존의 finite matrix certificate다. full proper-projector operator나 principal-series matrix coefficient를 materialize한 것이 아니다.

## 34.3 root gauge fixing이 세는 차원

Cartan-dual critical frame을 $G_a\in SL(2,\mathbb C)$, $a=0,\ldots,4$라 하자. root $r$ 하나를 택해

$$
\widehat G_a=G_r^{-1}G_a,\qquad \widehat G_r=I
\tag{5}
$$

로 둔다. $SL(2,\mathbb C)$ 하나는 실수 6차원이므로, gauge fixing 전 다섯 factor는 30 real dimensions이고 식 (5) 뒤에는 네 factor, 곧

$$
4\times6=24
\tag{6}
$$

real dimensions가 남는다.

common-left transform $G_a\mapsto LG_a$에 대해

$$
(LG_r)^{-1}(LG_a)=G_r^{-1}G_a=\widehat G_a
\tag{7}
$$

이므로 relative frame은 변하지 않는다. fixed regression에서는 root identity, 식 (7)의 relative-frame equality, projector invariance, $\beta$ sign invariance를 따로 확인한다. 이 결과는 gauge convention이 일관된다는 뜻이지 $dG$ Haar measure를 정의하거나 적분을 수행했다는 뜻이 아니다.

## 34.4 contract가 멈추는 곳

한 cell에서 만든 것은 다섯 LS vector, 열 face의 Eq.-(53) target endpoint policy, positive-sector target projector, 그리고 24-real-dimensional formal relative-frame domain이다. [33장](33_incidence_spinor와_Cartan_dual_EH_sector.md)의 fixed-cell EH orientation/Plebanski candidate와도 양립하지만, EH action/dynamics를 계산하지 않는다.

특히 $\alpha$의 값은 평가하지 않았고, full principal-series $Y_\gamma$ action, pointwise proper-vertex integrand, Haar measure 및 gauge-fixed $SL(2,\mathbb C)$ integral도 만들지 않았다. LS bra--ket의 global pairing/contraction, physical Regge spinor/action phase, global boundary state/network, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum dynamics와 EH/two-DOF IR은 모두 미완성이다. 이 한-cell contract를 그 결과들의 축약으로 쓰지 않는다.

[35장](35_CP1_pointwise_proper_coefficient.md)은 fixed Cartan-dual frames에서 Eq.-(53) target endpoint의 compact $\mathbb{CP}^1$ alpha pairing을 수치로 평가한다. 그 product는 Haar measure에 대한 relative coefficient일 뿐 pointwise Haar-density integrand나 noncompact integral을 제공하지 않으므로, 이 절의 ceiling은 바뀌지 않는다.
