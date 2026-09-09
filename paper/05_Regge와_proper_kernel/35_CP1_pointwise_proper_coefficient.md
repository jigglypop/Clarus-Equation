# 35. $\mathbb{CP}^1$ pointwise proper coefficient

34장은 한 cell의 projector endpoint와 relative critical frame을 contract로 고정했지만, 그 face factor의 수를 아직 계산하지 않았다. 이 장은 Cartan-dual classical frame tuple에서 열 face의 published homogeneous-function/$\mathbb{CP}^1$ pairing을 유한 quadrature로 평가한다. 얻는 대상은 **product Haar measure에 대한 pointwise coefficient function**이며, Haar density를 포함한 integrand나 Haar integral의 값이 아니다.

이 구분을 먼저 둬야 한다. $\mathbb{CP}^1$의 compact angular integral은 한 face pairing을 정의하는 부분이고, root gauge-fix 뒤 남은 $SL(2,\mathbb C)^4$의 noncompact Haar integration은 별도의 일이다. 따라서 Eq. (53)의 target $\Pi_{ba}$ placement를 유지한 채, 먼저 한 face formula, 다음 열 compact quadrature와 product, 마지막 수치 오차의 범위를 차례로 적는다.

이 글은 안정화된 `C4-LORENTZIAN-ONE-TO-FIVE-CP1-KERNEL-36A--D`를 읽기 전용 입력으로 쓴다. level $3$, $\gamma=0.274$의 rounded spins는 exact spin-weighted closure를 만족하지 않으므로 finite-spin stationary point certificate가 아니다.

## 35.1 한 face의 $\mathbb{CP}^1$ formula

ordered face $(a,b)$에서 $p=\gamma j$와 $d_j=2j+1$를 둔다. $z\in\mathbb{CP}^1$의 normalized representative를 쓰고

$$
\Omega(z)=\frac14d\Omega_{S^2}
\tag{1}
$$

로 둔다. 그러면 $\int_{\mathbb{CP}^1}\Omega=\pi$다. source spin-$j$ vector의 degree-$2j$ homogeneous polynomial을 $P_s$라고 쓰고, target vector에는 linear $\epsilon_j$ dual을 먼저 적용해 $P_{\epsilon t}$를 쓴다. 이는 anti-linear $J$와 다른, alpha bilinear pairing 안의 linear map이다.

Eq. (53)의 convention에서 target ket은 먼저 projector를 받는다.

$$
|t\rangle=\Pi^{(+)}_{ba}|j_{ba},\xi_{ba}\rangle.
\tag{2}
$$

고정 frame $X_a,X_b$에서 이 장이 쓰는 alpha face pairing은 다음처럼 정규화한다.

$$
\begin{aligned}
C_{ab}(X_a,X_b)
={}&c_{ab}d_j\int_{\mathbb{CP}^1}\!\Omega(z)\\
&\times\langle X_a^{-1}z,X_a^{-1}z\rangle^{-1-j-ip}
\langle X_b^{-1}z,X_b^{-1}z\rangle^{-1-j+ip}\\
&\times P_s\!\left(\overline{X_a^{-1}z}\right)
P_{\epsilon t}\!\left(X_b^{-1}z\right),\\
c_{ab}={}&\frac{\sqrt{j^2+p^2}}{\pi(j-ip)}.
\end{aligned}
\tag{3}
$$

두 radial factor의 밑은 spinor의 squared Hermitian norm이다. 두 exponent의 $\mp ip$와 두 frame의 $X^{-1}$가 식 (3)의 방향 convention을 고정한다. 식 (2) 때문에 $\Pi_{ba}$를 source $ab$ endpoint로 옮기지 않으며, projector 뒤의 target polynomial에 linear $\epsilon_j$를 적용한다.

## 35.2 compact quadrature와 product

각 $\mathbb{CP}^1$는 두 real coordinate를 가지므로 열 face에 대해 열 compact quadrature를 수행한다. coarse/fine chart의 face coefficient 차이의 최대값은

$$
\max_{ab}|C^{\rm fine}_{ab}-C^{\rm coarse}_{ab}|
=1.51757\times10^{-14}
\tag{4}
$$

이다. fine pairing에 Eq.-(53) target projector를 넣은 값과 target coherent state만 넣은 값의 최대 차이는

$$
\max_{ab}|C^{\Pi}_{ab}-C^{\rm coh}_{ab}|
=1.37327\times10^{-15}
\tag{5}
$$

다. Cartan-dual branch에서 projector가 target ket을 보존한다는 34장의 finite check와 일관된다.

ten face coefficient의 product와 integer-spin graphical ordering sign을 곱한 output은

$$
\prod_{a<b}C^{\Pi}_{ab}
=0.9909730266819072+0.1334356978266045\,i,
\qquad |\cdot|=0.9999163090,
\tag{6}
$$

이며 sign은 $+1$이다. 식 (6)은 root-gauge-fixed Cartan-dual classical frames에서의 product-Haar-measure-relative numerical coefficient다. exact value, integrand density, 또는 amplitude 값이라고 부르지 않는다.

## 35.3 수치 결과가 보장하지 않는 것

식 (4)는 coarse/fine 두 discretization 사이의 경험적 안정성 estimate다. rigorous error bound, exact quadrature, stationary-phase closure의 정리가 아니다. 특히 rounded finite spins의 spin-weighted closure가 exact하지 않으므로 식 (6)에서 finite-spin stationary point를 선언할 수 없다.

또한 이 장은 product Haar density를 넣지 않았고 noncompact $SL(2,\mathbb C)^4$ Haar measure 또는 gauge-fixed integral을 materialize/evaluate하지 않았다. 열 compact $\alpha$ pairing을 넘는 LS/global boundary bra--ket contraction, physical Regge spinor/action phase, proper EPRL five-vertex amplitude, multicell Hessian, curved/refinement/continuum dynamics와 EH/two-DOF IR은 여전히 미완성이다.
