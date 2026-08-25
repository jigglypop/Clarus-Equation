# Mathematics lane — from repeated self-nonidentity to one clock field

Status: LANE_COMPLETE — R2 conditional derivation; full objective open

## 1. Kinematic definition and dimensions

Use metric signature $(-,+,+,+)$. Let $T$ have time dimension and define

$$
X=-\frac12g^{\mu\nu}\nabla_\mu T\nabla_\nu T.
$$

Then $X$, $X_*$, $\delta=X/X_*-1$, and $\kappa$ are dimensionless;
$[\rho_\infty]=[M^4]$ and $[\Gamma]={\rm time}^{-1}$. On homogeneous FLRW,

$$
X=\frac12\dot T^2.
$$

The discrete statement $T_{n+1}\ne T_n$ has a continuum readout $X>0$.
It does not by itself pick a time arrow; the future branch $\dot T>0$ is separate
Cauchy data.

For the fixed-dephasing operational predecessor, define

$$
\theta=\Gamma T.
$$

Then the dimensionless opportunity fraction is

$$
c(\theta)=1-e^{-\theta}=1-e^{-\Gamma T}.
$$

Thus route B uses $\rho_\infty c(\theta)$ only after adding the independent
energy-density scale $\rho_\infty$ and clock-conversion rate $\Gamma$.

## 2. General $P(T,X)$ identities

For

$$
S=\int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R+P(T,X)\right]+S_b+S_r,
$$

variation gives

$$
\nabla_\mu(P_X\nabla^\mu T)+P_T=0,
$$

$$
T_{\mu\nu}^{(T)}
=P_X\nabla_\mu T\nabla_\nu T+Pg_{\mu\nu}.
$$

Thus

$$
\rho_T=2XP_X-P,\qquad p_T=P.
$$

The identity

$$
\nabla_\mu T^\mu{}_\nu
=\left[\nabla_\mu(P_X\nabla^\mu T)+P_T\right]\nabla_\nu T
$$

proves on-shell covariant conservation. In FLRW,

$$
\frac{d}{dt}\left(a^3P_X\dot T\right)=a^3P_T.
$$

## 3. Route A: shift-symmetric kinetic condensate

Take

$$
P_A(X)=M^4\left[-1+\frac\kappa2\delta^2\right],
\qquad
\delta=\frac{X}{X_*}-1.
$$

Then

$$
P_X=\frac{\kappa M^4}{X_*}\delta,\qquad
P_{XX}=\frac{\kappa M^4}{X_*^2},
$$

$$
p_A=M^4\left(-1+\frac\kappa2\delta^2\right),
$$

$$
\rho_A=M^4\left(1+2\kappa\delta+\frac{3\kappa}{2}\delta^2\right).
$$

The exact conserved-current solution on $\dot T>0$ is

$$
\delta\sqrt{1+\delta}=Qa^{-3},\qquad Q>0,
$$

or

$$
\delta'=-\frac{6\delta(1+\delta)}{2+3\delta}.
$$

At $\delta=0$,

$$
\dot T=\sqrt{2X_*}\ne0,\qquad \rho_A=M^4,\qquad p_A=-M^4.
$$

Thus a nonzero self-nonidentity flow can have $w=-1$. For $0<\delta\ll1$,

$$
\rho_A=M^4+2\kappa M^4Qa^{-3}+O(a^{-6}),
\qquad
p_A=-M^4+O(a^{-6}).
$$

This is the conditional $\Lambda+$dust theorem.

The exact stability quantities are

$$
\rho_X=P_X+2XP_{XX}
=\frac{\kappa M^4}{X_*}(2+3\delta),
$$

$$
c_s^2=\frac{\delta}{2+3\delta}.
$$

Hence the physical two-derivative branch is $\delta\ge0$. At $\delta=0$,
$c_s^2=0$ and a higher-spatial-derivative EFT completion is needed to control
the cutoff. For $\delta\gg1$,

$$
\delta\propto a^{-2},\qquad \rho_A\propto a^{-4},\qquad w_A\to\frac13.
$$

Therefore the quadratic route is CDM-like only if $\delta$ is already small over
the structure-formation domain.

## 4. Route B: anchored soft measurement clock

Take

$$
P_B(T,X)=\rho_\infty\left[
\frac\kappa2\delta^2-\left(1-e^{-\Gamma T}\right)
\right].
$$

Its pressure and energy density are

$$
p_B=\rho_\infty\left[
\frac\kappa2\delta^2-\left(1-e^{-\Gamma T}\right)
\right],
$$

$$
\rho_B=\rho_\infty\left[
\left(1-e^{-\Gamma T}\right)
+2\kappa\delta+\frac{3\kappa}{2}\delta^2
\right].
$$

This admits the exact decomposition

$$
\rho_V=V(T),\quad p_V=-V(T),\qquad
\rho_K=\rho_\infty\left(2\kappa\delta+\frac{3\kappa}{2}\delta^2\right),
\quad p_K=\frac{\kappa\rho_\infty}{2}\delta^2.
$$

For $\delta>0$ the kinetic part is positive. Its equation of state is

$$
w_K=\frac{\delta}{4+3\delta},
$$

so it is dustlike for $\delta\ll1$.

Because

$$
P_T=-\rho_\infty\Gamma e^{-\Gamma T}<0,
$$

the exact current equation is

$$
\frac{d}{dt}\left(a^3J\right)
=-a^3\rho_\infty\Gamma e^{-\Gamma T},
\qquad
J=P_X\dot T.
$$

Consequently

$$
a^3(t)J(t)=a_i^3J_i-
\int_{t_i}^{t}a^3(s)\rho_\infty\Gamma e^{-\Gamma T(s)}\,ds.
$$

### No-free-generation theorem

If $J_i=0$ at $T_i=0$, then every sufficiently near future time has $J<0$,
hence $\delta<0$ and $c_s^2<0$. Therefore a closed one-field theory cannot
create both increasing $V$ and positive dust current from zero. The corrected
matching supplies the initial canonical current as retained fold data:

$$
\boxed{
a_i^3J_i=\Pi_{\rm fold}
=\mathcal R_\Pi[\mu_F,C_{\rm self};\rho_\infty,\Gamma,L_c]>0.
}
$$

This is a Neumann-type Cauchy datum paired with $T|_{\Sigma_*}=0$. It is not a
second bulk field and is not added again to $\rho_B$; its bulk energy is already
the positive $\delta_i$ contribution to $\rho_K$. The map $\mathcal R_\Pi$ and
its dimensionful normalization remain a physical matching axiom.

### Global positive-current condition

For a finite validation interval $[t_i,t_f]$, the stable branch exists as long as

$$
\boxed{
\Pi_{\rm fold}>
\sup_{t\in[t_i,t_f]}
\int_{t_i}^{t}a^3(s)\rho_\infty\Gamma
e^{-\Gamma T(s)}\,ds.
}
$$

For an infinite de Sitter future with
$T\simeq\sqrt{2X_*}\,t$ and
$H_\infty=\sqrt{\rho_\infty/(3M_{\rm Pl}^2)}$, convergence requires

$$
\boxed{\Gamma\sqrt{2X_*}>3H_\infty.}
$$

This is only the asymptotic convergence condition. A global theorem also requires
the preceding supremum with $t_f=\infty$ and a strictly larger
$\Pi_{\rm fold}$.

For the dimensionless numerical branch with $X_*=1/2$, write

$$
N=\ln a,\qquad \tau=H_0T,\qquad
\gamma=\Gamma/H_0,\qquad
q=e^{3N}u\sqrt{1+u/\kappa}.
$$

Suppose the finite solve reaches $N_f>0$ with $q_f>0$ and
$V_f:=\rho_V(N_f)/\rho_{\mathrm{crit},0}>0$.  Total continuity gives

$$
\frac{dE^2}{dN}=-3\left[\rho_b+\frac43\rho_r+\rho_K+p_K\right]\le0,
$$

so $E\le E(0)=1$ on the positive branch, while Friedmann positivity and
monotonic $V$ give $E\ge\sqrt{V_f}$.  Up to any first putative zero of $q$,
$q>0$ and strict monotonicity of
$F(u)=u\sqrt{1+u/\kappa}$ imply $u>0$.  Hence

$$
\tau'=\frac{\sqrt{1+u/\kappa}}{E}\ge1,
\qquad
-q'=\frac{\gamma e^{3N-\gamma\tau}}{2E}.
$$

For the sufficient condition $\gamma>3$, integration from $N_f$ therefore
bounds all remaining current loss by

$$
\boxed{
\Delta q_{\rm tail}\le
\frac{\gamma e^{3N_f-\gamma\tau_f}}
{2\sqrt{V_f}(\gamma-3)}.
}
$$

If $q_f>\Delta q_{\rm tail}$, the assumed first zero cannot be reached.  This
first-zero contradiction closes the bootstrap and proves $q(N)>0$ for every
$N\ge N_f$.  The inequality $\gamma>3$ is a conservative sufficient condition
for this particular bound, not a physical necessity and not an identity with
$\Gamma\sqrt{2X_*}>3H_\infty$.

The component exchange is

$$
\dot\rho_K+3H(\rho_K+p_K)
=-\dot V,
\qquad
\dot\rho_V=+\dot V,
$$

so the increasing opportunity readout is paid for by the initial kinetic
inventory, while the total stress is conserved.

## 5. Shift identifiability

Under $T\mapsto T+\Delta$,

$$
\rho_\infty(1-e^{-\Gamma T})
\mapsto
\rho_\infty-\rho_\infty e^{-\Gamma\Delta}e^{-\Gamma T}.
$$

No single redefinition of $\rho_\infty$ preserves both the constant and exponential
coefficients unless $\Delta=0$. Thus the old shift--amplitude degeneracy is absent
provided $\Sigma_*$ is a physically defined matching surface and $T|_{\Sigma_*}=0$.
If $\Sigma_*$ is only a coordinate convention, the anchor has no empirical content.
This conclusion is action-relative: a generic independent vacuum counterterm and an
independent exponential coefficient would restore enough freedom to absorb the anchor.
The tied coefficients inherited from $c(\theta)=1-e^{-\theta}$ therefore need
radiative protection or an explicit renormalization condition.

## 6. Soft constrained-clock limit

Let $u=\kappa\delta$ stay finite while $\kappa\to\infty$. Then

$$
\rho_K\to2\rho_\infty u,\qquad p_K\to0,\qquad c_s^2\to0,
$$

and $X\to X_*$. This is the irrotational-dust constrained-clock limit. It is an
IR limit, not a proof of caustic-free nonlinear evolution.

## 7. Formal verdict

| claim | status | reason |
|---|---|---|
| nonzero $X_*$ can give $w=-1$ | [정리] | direct stress-tensor evaluation |
| positive small $\delta$ gives dustlike energy | [정리: domain restricted] | exact $\rho_K,p_K,c_s^2$ |
| increasing $V$ and positive dust arise from zero | [삭제: 완전 반례] | current sign forces $\delta<0$ |
| fold-matched positive current can feed increasing $V$ | [조건부 정리] | exact current integral and $\Pi_{\rm fold}$ bound |
| old exponential shift degeneracy remains | [반증] | anchored two-term function is not rescaled into itself |
| future orientation follows from $X>0$ | [미완성] | $\dot T=\pm\sqrt{2X}$; branch data needed |
| one field is a quantum-path derivation | [미완성] | operational-to-EFT map and scales are not derived |

Route B survives the gate only with a nonzero positive initial current, the global
positivity inequality, the future branch, and an explicit IR/cutoff domain.

## 8. R1 명시적 Gaussian 저장소 유도

### 8.1 단순 선형 결합의 하한 반례

정준 저장소 스칼라 $\phi$에 $gT\phi$를 선형으로 결합하면 균일 mode의
potential은

$$
\frac12m^2\phi^2+gT\phi
=\frac12m^2\left(\phi+\frac{gT}{m^2}\right)^2
-\frac{g^2T^2}{2m^2}
$$

가 된다. 기존 clock potential은 $T\to+\infty$에서 유한값으로 포화하므로
마지막 항을 상쇄하지 못한다. 따라서 $g\ne0$인 선형 경로는 물질 Hamiltonian의
하한을 잃으며 음성대조군으로 제거한다.

### 8.2 유계 source를 가진 확대 작용

서명 $(-,+,+,+)$에서 다음 규제된 이산 저장소로 시작한다.

$$
S_{\rm tot}=\int d^4x\sqrt{-g}\left\{
P(T,X)-\sum_A\left[
\frac12\nabla_\mu\phi_A\nabla^\mu\phi_A
+\frac12m_A^2\phi_A^2+s_A(T)\phi_A
\right]\right\},
\tag{R1.1}
$$

$$
s_A(T)=\mu_A^3F_A(\Gamma T),\qquad
F_A(0)=0,\qquad \|F_A\|_\infty<\infty.
\tag{R1.2}
$$

질량차원은

$$
[T]=-1,\ [X]=0,\ [P]=4,\ [\phi_A]=1,\ [m_A]=[\Gamma]=[\mu_A]=1,
\ [s_A]=3,\ [s'_A]=4.
\tag{R1.3}
$$

따라서 식 (R1.1)의 모든 벌크 밀도는 차원 4다. 연속 스펙트럼 극한에서는

$$
\sum_A\frac{\mu_A^6}{m_A^2}\|F_A\|_\infty^2<\infty
\tag{R1.4}
$$

에 해당하는 UV 수렴조건과 진공에너지 재규격화 조건이 필요하다.

변분하면

$$
\nabla_\mu(P_X\nabla^\mu T)+P_T-
\sum_As'_A(T)\phi_A=0,
\qquad
(\Box-m_A^2)\phi_A=s_A(T).
\tag{R1.5}
$$

균일 FLRW에서는

$$
\dot J+3HJ=P_T-\sum_As'_A\phi_A,
\qquad
\ddot\phi_A+3H\dot\phi_A+m_A^2\phi_A+s_A(T)=0.
\tag{R1.6}
$$

$T\ge0$, $\delta\ge0$인 선택 가지에서 clock 에너지는 양수이고, 저장소와
상호작용을 완성제곱하면

$$
\rho_{\rm matter}\ge
-\sum_A\frac{\mu_A^6\|F_A\|_\infty^2}{2m_A^2}.
\tag{R1.7}
$$

이는 규제된 cell과 고정 배경에서 물질부가 유한한 하한을 가진다는 뜻이다.
일반 GR 전체에 전역 양의 Hamiltonian이 존재한다는 주장이 아니다.

### 8.3 CTP 적분과 kernel 조건

$s_r=(s_++s_-)/2$, $s_a=s_+-s_-$로 두고 응답을

$$
\langle\phi_A(x)\rangle=\phi_A^h(x)+
\int d^4y\sqrt{-g_y}\,D_{R,A}(x,y)s_A(T_y)
\tag{R1.8}
$$

로 정의한다. 이 부호 규약에서 Gaussian 저장소의 영향작용은

$$
S_{\rm IF}=-\int_{xy}s_a(x)D_R(x,y)s_r(y)
+\frac{i}{2}\int_{xy}s_a(x)N(x,y)s_a(y)
\tag{R1.9}
$$

이다. 여기서 $D_R=\sum_AD_{R,A}$는

$$
(\Box_x-m_A^2)D_{R,A}(x,y)
=\frac{\delta^{(4)}(x-y)}{\sqrt{-g_x}},
\qquad D_R(x,y)=0\quad(x\notin J^+(y))
\tag{R1.10}
$$

을 만족하고,

$$
N(x,y)=\frac12\sum_A
\langle\{\phi_A^h(x),\phi_A^h(y)\}\rangle_i,
\qquad
\int fNf\ge0
\tag{R1.11}
$$

이다. $[D_R]=[N]=2$이고 이중적분을 포함한 식 (R1.9)는 무차원이다.
배경 주위 선형 kernel

$$
K_R(x,y)=\sum_As'_A(\bar T_x)D_{R,A}(x,y)s'_A(\bar T_y)
\tag{R1.12}
$$

의 차원은 10이므로 $\int d^4yK_R\,\delta T$는 clock 방정식의 차원 5와
일치한다. 열평형을 별도 채택할 때에만 KMS/FDR을 요구한다. 유한 mode bath는
재귀를 가지므로 비가역적 coarse-grained arrow에는 연속 스펙트럼과 시간척도
분리가 최소 추가조건이다. 전체 계의 유니터리 진화 자체가 근본적으로
비가역이 되었다는 뜻은 아니다.

### 8.4 초기 Gaussian 상태와 고전 matching

유한 coarse-graining cell의 균일 mode를 $(q,p)$로 smear하면

$$
\bar q=0,\qquad \bar p=V_ca_i^3\Pi_F>0,
\qquad
\mathbf V+\frac{i\hbar}{2}\mathbf\Omega\succeq0
\tag{R1.13}
$$

인 변위 Gaussian 상태가 존재한다. 따라서 평균 $\langle T_i\rangle=0$과
평균 $\langle J_i\rangle=\Pi_F>0$는 양자역학적으로 양립한다. 정확한
연산자값 $T_i=0$과 분산 0을 뜻하지 않는다. local unsmeared kernel은
$\delta(0)$를 만들므로 cell 크기 $L_c$ 또는 UV cutoff가 필수다.

또한

$$
\langle P_X\dot T\rangle
\ne P_X(\langle T\rangle,\langle X\rangle)\langle\dot T\rangle
\tag{R1.14}
$$

가 일반적이다. R0의 고전 $\delta_i$와 연결하려면 renormalized composite
current와 좁은 semiclassical packet 조건
$|\langle J\rangle-J_{\rm cl}|\le\epsilon_J$를 별도로 검증해야 한다.
식 (R1.13)은 $\Pi_F$의 존재 가능한 준비를 보일 뿐 그 수치를 예측하지 않는다.

### 8.5 총 Ward 장부

상호작용 응력을

$$
T_{\rm int}^{\mu\nu}=-g^{\mu\nu}\sum_As_A(T)\phi_A
\tag{R1.15}
$$

까지 포함하면 온셸에서

$$
\nabla_\mu T_T^{\mu\nu}
=\sum_As'_A\phi_A\nabla^\nu T=:Q_T^\nu,
\qquad
\nabla_\mu(T_\phi^{\mu\nu}+T_{\rm int}^{\mu\nu})=-Q_T^\nu.
\tag{R1.16}
$$

따라서 $T+\phi+$상호작용의 총응력은 정확히 보존된다. 균일계에서는

$$
\dot\rho_T+3H(\rho_T+p_T)=-\sum_As'_A\phi_A\dot T,
\tag{R1.17}
$$

$$
\dot\rho_\phi+3H(\rho_\phi+p_\phi)=-\sum_As_A\dot\phi_A,
\qquad
\dot\rho_{\rm int}=\sum_A(s'_A\phi_A\dot T+s_A\dot\phi_A),
\tag{R1.18}
$$

이고 세 식의 합은 0이다. bath를 명시적으로 Einstein source에 포함한 뒤
영향함수의 응력을 독립 성분으로 다시 더하거나 $\Pi_F$를 별도 fluid로 더하면
이중계산이다.

### 8.6 새 안정성 falsifier

기존 두-미분 bulk의 fixed-background 장파장 선형식에는

$$
\rho_X\,\delta\ddot T+3H\rho_X\,\delta\dot T
-\frac{P_X}{a^2}\nabla^2\delta T-P_{TT}\delta T+\cdots=0,
\tag{R1.19}
$$

$$
P_{TT}=\rho_\infty\Gamma^2e^{-\Gamma T}>0,
\qquad
m_{\rm eff}^2\simeq-\frac{P_{TT}}{\rho_X}<0.
\tag{R1.20}
$$

가 남는다. 이 식은 metric mixing과 bath self-energy를 생략했으므로 곧바로
완전한 우주론 불안정성 정리는 아니지만, 기존의 ghost·gradient 검사만으로
“안정”이라고 부를 수 없음을 보이는 필수 falsifier다. 동시에
$c_s^2\to0$에서는 strong-coupling cutoff 또는 ghost-condensate형 $k^4$
완성이 필요하다. full coupled scalar perturbation의 모든 고윳모드가 Hubble
시간과 CMB/LSS 범위에서 허용되는지 확인하기 전 R1의 안정성은 미완성이다.

## 9. R2-A 포화 readout의 장파장 성장률

### 9.1 고정 FLRW 배경의 정확한 2차 작용

$T=\bar T(t)+\pi$로 놓고 일반 $P(T,X)$를 2차까지 전개한 뒤 배경 방정식을
사용하면

$$
S_\pi^{(2)}=\frac12\int d^4x\,a^3\left[
A\dot\pi^2-\frac{B}{a^2}(\nabla\pi)^2+C\pi^2
\right],
\tag{R2.1}
$$

$$
A=P_X+2XP_{XX},\qquad B=P_X,
\qquad
C=P_{TT}-a^{-3}\frac{d}{dt}
\left(a^3P_{TX}\dot{\bar T}\right).
\tag{R2.2}
$$

따라서

$$
A\ddot\pi+(3HA+\dot A)\dot\pi
-\frac{B}{a^2}\nabla^2\pi-C\pi=0.
\tag{R2.3}
$$

현재 $P(T,X)=K(X)-V(T)$에서는 $P_{TX}=0$이고

$$
A=\frac{\kappa\rho_\infty}{X_*}(2+3\delta),\qquad
B=\frac{\kappa\rho_\infty}{X_*}\delta,
\tag{R2.4}
$$

$$
m_{\rm eff}^2=-\frac{C}{A}
=-\frac{\Gamma^2X_*e^{-\Gamma\bar T}}
{\kappa(2+3\delta)}<0.
\tag{R2.5}
$$

이다. $\delta\ge0$에서 ghost와 gradient 조건은 통과하지만 질량항의 부호는
음이다.

### 9.2 포화·단조 증가·전역 무타키온 no-go

비상수 $V\in C^2[0,\infty)$가 단조 증가하고 위에서 유계라면 전 구간에서
$V''\ge0$일 수 없다. 만약 $V''\ge0$이면 $V'$는 비감소다. 어느 한 점에서
$V'>0$이면 이후 적어도 선형으로 발산해 상계성과 모순이고, 그렇지 않으면
$V'\equiv0$이어서 비상수 조건과 모순이다. 따라서

$$
\text{비상수 포화}
+\text{단조 증가}
+\text{전역 }m_{\rm eff}^2\ge0
\tag{R2.6}
$$

의 세 조건은 이 분리형 $K(X)-V(T)$에서 동시에 만족할 수 없다. 이 반례는
목표를 버리게 하지 않고 판정량을 “음의 부호의 존재”에서 “관측 시간 동안의
실제 증폭”으로 바꾼다.

### 9.3 Hubble-normalized 성장 게이트

기존 무차원 배경 변수 $N=\ln a$, $\tau=H_0T$, $u=\kappa\delta$를 쓰면
$k=0$ 고정배경 식은

$$
\pi''+f_N\pi'-r\pi=0,
\tag{R2.7}
$$

$$
r(N)=\frac{|m_{\rm eff}^2|}{H^2}
=\frac{\gamma^2X_*e^{-\gamma\tau}}
{\kappa(2+3u/\kappa)E^2},
\tag{R2.8}
$$

$$
f_N=3+\frac{E'}E+
\frac{3u'/\kappa}{2+3u/\kappa}.
\tag{R2.9}
$$

고정계수 양의 성장근은 부동소수점 상쇄를 피한 형태

$$
\lambda_+(N)=
\frac{2r}{\sqrt{f_N^2+4r}+f_N}
\tag{R2.10}
$$

로 계산한다. Hubble 마찰만 사용한 별도 느린 성장 진단은

$$
S_{\rm tach}:=\int\frac{|m_{\rm eff}^2|}{3H}\,dt
=\int\frac{r(N)}3\,dN
\tag{R2.11}
$$

이다.

동결한 $\kappa=10^{17}$ 배경을 $a=10^{-4}$부터 $1$까지 독립 재적분한
R2 artifact는 $\gamma=3.5,5,10,20,30$ 전부에서

$$
\max r=3.0455\times10^{-18},\qquad
\max\int\lambda_+dN=2.4783\times10^{-18},
\tag{R2.12}
$$

$$
\max(\pi/\pi_i-1)=2.2737\times10^{-18},\qquad
\log G_{\rm const}<2.4839\times10^{-17}
\tag{R2.13}
$$

을 얻었다. $\gamma=10$의 3000/6000 step 비교에서 최대 상대 변화는
$3.93\times10^{-7}$이었다. 즉 이 동결 배경에서 fixed-metric 타키온은
Hubble 시간에 사실상 자라지 않는다. 이는 $\kappa=10^{17}$이라는 외부
hierarchy에 강하게 의존하며 metric mode를 제거한 결과가 아니다. 직접 해는
$\pi_i=1$, $\pi_i'=0$을 쓴 선택 초기조건이고, $\lambda_+$ 적분과
고정계수 상계도 이 비교식의 진단량이다. 임의 초기 섭동과 모든 물리 모드의
성장 정리로 승격하지 않는다.

### 9.4 bath와 $k^4$가 자동 해결책이 아닌 이유

정상 부호 Gaussian bath를 저주파에서 정적으로 제거하면

$$
P_{\rm eff}=P+\sum_A\frac{s_A^2}{2m_A^2},
\qquad
V_{\rm eff}=V-\sum_A\frac{s_A^2}{2m_A^2}.
\tag{R2.14}
$$

따라서 안정화에는

$$
V_{\rm eff}''\ge0,\qquad
V_{\rm eff}'\ge0
\tag{R2.15}
$$

를 동시에 만족하는 source가 필요하다. 그러나 $V_{\rm eff}$도 비상수·단조
증가·포화라면 식 (R2.6)의 구조적 한계를 피하지 못한다. bath는 불안정
구간을 옮기거나 약화할 수 있을 뿐 자동 안정화 증명이 아니다.

또한 나이브한 $(\Box T)^2$는 고차 시간미분 ghost를 만들 수 있다. clock
foliation의 $u_\mu=-\nabla_\mu T/\sqrt{2X}$와 extrinsic curvature를 써

$$
\Delta S_{k^4}=-\frac{\bar M^2}{2}
\int d^4x\sqrt{-g}\,(\delta K)^2
\tag{R2.16}
$$

같은 unitary-gauge EFT 연산자를 degeneracy 조건 아래 채택해야 한다. 분리
극한의 분산식은

$$
\omega^2=c_s^2q^2+\frac{\bar M^2}{A}q^4+m_{\rm eff}^2
\tag{R2.17}
$$

다. 양의 $k^4$는 UV gradient를 안정화하지만 $q=0$의 음의 질량을 없애지
않는다. 중력 mixing은 추가 Jeans형 $q^2$ 항을 만들 수 있으므로 lapse와
shift를 제거한 ADM reduced kinetic/gradient matrix, bath pole, cutoff
hierarchy 및 gauge-invariant CMB/LSS mode가 다음 필수 gate다.

### 9.5 고정계량 $\pi$와 물리적 단일-clock mode의 분리

$\pi=\delta T$는 시간 미분동형사상 아래 변하고 unitary gauge에서는
$\pi=0$으로 둘 수 있으므로, 식 (R2.5)의 음의 질량은 그 자체로 물리적
타키온 pole이 아니다. Einstein 중력과 $P(T,X)$ clock만 남긴 단일-clock
부분계에서 lapse와 shift 제약을 제거하면

$$
S_\zeta^{(2)}=\int d^4x\,a^3Q_s\left[
\dot\zeta^2-c_s^2\frac{(\nabla\zeta)^2}{a^2}
\right],
\qquad
Q_s=\frac{X(P_X+2XP_{XX})}{H^2},
\qquad
c_s^2=\frac{P_X}{P_X+2XP_{XX}}.
\tag{R2.18}
$$

즉 이 부분계의 $\zeta$에는 독립적인 질량항이 없고, 장파장 해는

$$
\zeta=C_1+C_2\int^t\frac{dt'}{a^3(t')Q_s(t')}
\tag{R2.19}
$$

이다. $\mathcal A:=\rho_\infty/\rho_{\rm crit,0}$로 쓰면 현재 동결 배경에
평가한 계수는

$$
\frac{Q_s}{M_{\rm Pl}^2}
=\frac{3\kappa\mathcal A(1+\delta)(2+3\delta)}{E^2},
\tag{R2.20}
$$

$$
\frac{d\ln(a^3Q_s)}{dN}
=3+\frac{\delta'}{1+\delta}
+\frac{3\delta'}{2+3\delta}-2\frac{E'}E,
\tag{R2.21}
$$

$$
-\frac{d}{dN}\ln\frac1{Ha^3Q_s}
=\frac{d\ln(a^3Q_s)}{dN}+\frac{E'}E.
\tag{R2.22}
$$

R2 artifact에서 $a\in[10^{-4},1]$, $\gamma=3.5,5,10,20,30$에 대해

$$
\min c_s^2=9.2138\times10^{-19}>0,
\qquad
\min\frac{Q_s}{M_{\rm Pl}^2}=3.3167\times10^5>0,
\tag{R2.23}
$$

$$
\min\frac{d\ln(a^3Q_s)}{dN}=3.93909,
\qquad
\min\left[-\frac{d}{dN}\ln\frac1{Ha^3Q_s}\right]=3.469545
\tag{R2.24}
$$

를 얻었다. 따라서 이 유한 구간의 **clock+GR 단일-clock 부분계**는 ghost와
gradient 조건을 통과하고 두 번째 $\zeta$ mode의 $\dot\zeta$와 적분함수도
감소한다. 유한 구간에서 이 사실만으로 $\zeta_2$ 자체의 무한 미래 수렴을
주장하지 않는다. 실제
배경의 baryon과 radiation을 배경에만 넣고 그 섭동을 끈 계산은 닫힌
게이지 불변 우주론이 아니다. Gaussian 저장소까지 포함하면 entropy mode와
retarded pole이 추가되므로 식 (R2.18)은 전체 안정성의 필요조건일 뿐이다.

### 9.6 작은 sound speed의 두 cutoff를 구분한다

$X_*=1/2$, $\delta\to0$에서
$\varphi=\sqrt{A_0}\,\pi=2\sqrt{\kappa\rho_\infty}\,\pi$로 정준화하면

$$
\mathcal L_3=\frac{\dot\varphi^3-
\dot\varphi(\nabla\varphi)^2}{4\sqrt{\kappa\rho_\infty}},
\tag{R2.25}
$$

$$
\mathcal L_4=\frac{\dot\varphi^4-2\dot\varphi^2(\nabla\varphi)^2
+(\nabla\varphi)^4}{32\kappa\rho_\infty}.
\tag{R2.26}
$$

따라서 bare derivative scale은

$$
\Lambda_3=2(\kappa\rho_\infty)^{1/4},
\qquad
\Lambda_4=(32\kappa\rho_\infty)^{1/4}.
\tag{R2.27}
$$

작은 $c_s$에서 $x=c_s\tilde x$와
$\tilde\varphi=c_s^{3/2}\varphi$로 이차 작용을 정준화하면 spatial cubic의
강결합 **에너지**와 **물리 파수** cutoff는 서로 다르다.

$$
\boxed{\Lambda_E\sim\Lambda_3c_s^{7/4}},
\qquad
\boxed{q_{\rm sc}=\frac{\Lambda_E}{c_s}
\sim\Lambda_3c_s^{3/4}}.
\tag{R2.28}
$$

$\Lambda_E$를 물리 파수와 직접 비교하면 $c_s$ 한 인자를 잃으므로 금지한다.
$H_0=67.4\,{\rm km\,s^{-1}Mpc^{-1}}$, reduced
$M_{\rm Pl}=2.435\times10^{27}\,{\rm eV}$를 사용한 tree-level
power counting은

$$
\Lambda_3=79.7\text{--}80.6\ {\rm eV},
\qquad
\Lambda_4=94.8\text{--}95.8\ {\rm eV},
\tag{R2.29}
$$

$$
\min\Lambda_E=1.3336\times10^{-14}\ {\rm eV},
\qquad
\min q_{\rm sc}=1.3893\times10^{-5}\ {\rm eV}.
\tag{R2.30}
$$

전 구간에서

$$
\min\frac{\Lambda_E}{H}=9.2757\times10^{18},
\qquad
\min\frac{q_{\rm sc}}{(1\,{\rm Mpc}^{-1})/a}
=2.1725\times10^{24}.
\tag{R2.31}
$$

따라서 이 power counting 안에서는 관측구간의 배경 시간척도와 선형
우주론 파수가 두-미분 strong coupling보다 충분히 낮다. 이는 loop,
nonlinear screening, 다성분 mixing 또는 UV completion의 증명이 아니다.

### 9.7 $k^4$ 완성의 올바른 crossover 조건

식 (R2.17)에서 선형항과 $k^4$ 항이 같아지는 물리 파수는

$$
q_\times=\frac{c_s\sqrt A}{\bar M}.
\tag{R2.32}
$$

$k^4$ 항이 두-미분 strong coupling 전에 켜지려면

$$
q_\times\lesssim q_{\rm sc}
\quad\Longrightarrow\quad
\bar M\gtrsim\frac{c_s\sqrt A}{q_{\rm sc}}
\xrightarrow[\delta\to0]{}
(\kappa\rho_\infty)^{1/4}c_s^{1/4}.
\tag{R2.33}
$$

오늘의 최댓값은 $0.225\,{\rm eV}$보다 작고,
$a\in[10^{-4},1]$ 전체에서 요구되는 최댓값도 $7.31\,{\rm eV}$다. 따라서
$\bar M\sim\Lambda_3\sim80\,{\rm eV}$인 자연 규모 후보는 **crossover
조건만으로는** 배제되지 않는다. 이는 $\bar M$을 0차원 저장소에서 유도한
것도, full ADM의 부호와 자유도 수를 통과시킨 것도 아니다.

source가 꺼진 미래에 보존 current가 유한한 양의 값으로 남으면

$$
u\propto a^{-3},\qquad c_s\propto a^{-3/2},
\qquad \Lambda_E\propto a^{-21/8},
\qquad q_{\rm sc}\propto a^{-9/8}.
\tag{R2.34}
$$

두-미분 EFT만 무한 미래로 외삽하면 cutoff는 0으로 내려가므로 완료된
이론이 아니다. 그러나 유한 $\bar M>0$이면

$$
\frac{q_\times}{q_{\rm sc}}\propto c_s^{1/4}\propto a^{-3/8}\to0,
\tag{R2.35}
$$

이어서 $k^4$ 항이 그 강결합 전에 켜질 수 있다. 이것이 R2에서 확보한
구체적인 돌파 경로다. 다음 단계는 unitary-gauge 연산자의 정확한 조합과
부호를 고정하고, lapse·shift·bath·물질 섭동을 함께 제거한 reduced matrix와
$k^4$ 영역 자체의 새 cutoff를 계산하는 것이다.

### 9.8 R2 판정

식 (R2.6)은 분리형 potential의 곡률 no-go지만, 고정계량 $\pi$의 음의
질량을 물리적 타키온으로 승격하는 것은 게이지 반례에 막힌다. 대신 R2는
세 층을 분리한다.

1. 선택 초기조건의 fixed-metric 성장 진단은 $10^{-17}$ 이하이다.
2. 유한 관측구간의 clock+GR 단일-clock 부분계와 strong-coupling hierarchy는
   조건부로 통과한다.
3. 다성분 ADM, reservoir pole, $k^4$ 완성의 정확한 계수와 관측 likelihood는
   미완성이다.

따라서 R2는 암흑에너지를 증명한 단계가 아니라, 기존 배경을 폐기하지 않고
물리적 섭동과 UV 완성으로 전진할 수 있는 정량적 경로를 확보한 단계다.
