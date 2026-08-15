# 공변 전이 면에서의 물질 분율 고정 경로

Status: COMPLETE

범위: 현재 시각이나 관측된 밀도분율을 coupling에 넣지 않고, 공변 action 또는
유효 action이 정의하는 전이 초곡면 $\Sigma_*$에서만 $\Omega_m=1/D$를 만들 수
있는지 검토한다. 이 문서는 존재구성과 no-go를 분리하며 정본이나 제품 코드를
수정하지 않는다.

독립 검산:
`artifacts/verify_transient_transition_action.py`

## 1. 결론

조건부 존재구성은 가능하다. 그러나 현재 단계에서 제1원리 예측으로 승격할 수
있는 경로는 없다.

가장 비자명한 경로는 기존 무차원 함수

$$
v_D(y)=y\log y-y+D\left(y-\frac{y^2}{2}\right)+C
$$

를 단일 정적 scalar potential이 아니라 반응하는 두-current 혼합물의
composition free energy로 사용하는 것이다. 냉각 control $h(z)$를 넣은

$$
f_D(y,z)=v_D(y)-h(z)y
$$

에서는 stationary spinodal 조건 $f_y=f_{yy}=0$이

$$
y_* = \frac1D,
\qquad
h_* = D-1-\log D
$$

를 동시에 준다. $y$를 처음부터 $\Omega_m$라고 정의하지 않고, 두 공변 current의
number composition으로 정의한 뒤 cold, comoving, equal-partial-energy 및
two-sector saturation을 action의 master function에서 확인하면 그 면에서
$\rho_m/\rho_{\rm tot}=y_*=1/D$가 따라온다. 따라서 이 제한된 계에서는 단순한
기호 재명명보다 강한 current/stress 유도가 된다.

그렇지만 다음 조건이 빠지면 다시 이름만 바꾼 bridge가 된다.

1. $y$가 실제 matter-tag current와 complement-tag current의 비율이어야 한다.
2. spinodal에서 두 partial chemical energy가 같아야 한다.
3. entrainment, gradient energy, 별도 radiation/entropy energy가 분모에 남지 않아야
   한다.
4. complement tag를 radiation 또는 dark energy stress로 바꾸는 공변 decay/action이
   있어야 한다.
5. 유한 냉각에서 critical slowing을 이겨 정확히 spinodal에서 freeze하는 법칙이
   있어야 한다.

별도의 density spectator에는 정확성--유일성 tradeoff가 있다. full density
fraction을 trigger로 쓰면 radiation까지 포함해 $\Omega_b=q$가 정확하지만 표준
radiation--matter--vacuum 역사에서 보통 같은 면이 두 번 생긴다. matter--vacuum
subsystem fraction을 쓰면 면은 유일하지만
$\Omega_b=q(1-\Omega_r)$라는 정확한 radiation correction이 남는다. 두 coupling
모두 목표 관계를 action에 재표현한 target-engineered axiom이다. symmetron,
coupled scalar/CDM, first-order transition도 전이 면은 만들 수 있으나 각각
상수항 tuning, 유한시간 fixed-point 도달, stochastic nucleation 문제가 남는다.

따라서 최종 지위는 다음과 같다.

- 전이 면에서의 합성식: **[정리: 조건부]**
- $v_D$ spinodal multi-current: **[미완성: 유효 존재구성]**
- density spectator: **[미완성: target-engineered 존재구성]**
- 현재 관측 epoch가 $\Sigma_*$라는 주장: **[미완성]**
- 관측값 없는 제1원리 절대 scale과 전이 시각: **[미완성]**

## 2. 먼저 고정되는 정확한 합성식

$D>1$이고 $q$가 Poisson branching의 비자명한 소멸확률이라고 하자.

$$
q=e^{-D(1-q)}.
$$

소멸을 조건으로 한 offspring 평균은

$$
m=f'(q)=Dq<1
$$

이다. 이 통계량을 matter 내부의 baryon energy composition으로 구현해

$$
f_b^{(m)}:=\frac{\rho_b}{\rho_m}=m=Dq
$$

를 얻었다고 하자. 별도의 공변 전이 dynamics가 같은 면에서

$$
\left.\frac{\rho_m}{\rho_{\rm tot}}\right|_{\Sigma_*}=\frac1D
$$

를 주면 즉시

$$
\left.\frac{\rho_b}{\rho_{\rm tot}}\right|_{\Sigma_*}
=f_b^{(m)}\frac{\rho_m}{\rho_{\rm tot}}
=(Dq)\frac1D=q
$$

가 된다. 공간적으로 평탄한 GR Friedmann 제약

$$
3M_{\rm Pl}^2H_*^2=\rho_{\rm tot,*}
$$

까지 포함하면

$$
\boxed{\Omega_b|_{\Sigma_*}=q}
$$

이다. 곡률이나 modified-gravity 항이 critical-density 분모와 total material
stress의 차이를 만들면 마지막 등식은 자동으로 성립하지 않는다.

이 정리는 두 개의 독립된 물리 lemma를 요구한다. $m=Dq$는 extinction-conditioned
composition lemma이고, $1/D$는 transition-surface total-matter lemma다. 둘 중
하나를 다른 하나의 정의로 넣으면 유도가 아니다.

## 3. 후보 A: $v_D$를 쓰는 반응성 multi-fluid spinodal

### 3.1 공변 master action

서로 comoving인 matter-tag current와 complement-tag current를

$$
N_M^\mu=n_Mu^\mu,
\qquad
N_X^\mu=n_Xu^\mu,
\qquad
n=n_M+n_X,
\qquad
y=\frac{n_M}{n}
$$

로 둔다. $n_i=\sqrt{-g_{\mu\nu}N_i^\mu N_i^\nu}$이므로 $n_i$와 $y$는
공변 local scalar다. control clock $\vartheta$에는

$$
z:=\frac{\vartheta}{f_\vartheta}
$$

를 사용한다. $z$는 무차원이고, $\vartheta$의 단조 branch가 cooling 방향을
정한다. cold master energy 후보는

$$
\rho(n_M,n_X,z)
=n\left[m_*+\mu_*\{v_D(y)-h(z)y\}\right]
$$

이다. 보존적 부분의 action은 Brown/Schutz 또는 convective constraint를
사용하여

$$
S_A=\int d^4x\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\rho(n_M,n_X,z)
-\frac12(\nabla\vartheta)^2-U(\vartheta)
\right]+S_{\rm conv}
$$

로 쓸 수 있다. separate-current 보존을 강제하는 constraint를 쓰면 reaction이
금지되므로, 반응 구간에서는 total current만 보존하고 composition에는
Schwinger--Keldysh/Onsager completion이 추가되어야 한다. ordinary single-copy
action만으로 양의 entropy production까지 얻었다고 주장할 수 없다.

### 3.2 chemical EOM, current, stress

$e(y,z):=\rho/n$라 두면 partial chemical potentials는 정확히

$$
\mu_M=\frac{\partial\rho}{\partial n_M}
=e+(1-y)e_y,
$$

$$
\mu_X=\frac{\partial\rho}{\partial n_X}
=e-ye_y.
$$

따라서 affinity는

$$
\mathcal A:=\mu_M-\mu_X
=\mu_*\left[v_D'(y)-h(z)\right]
$$

이다. total number를 보존하는 dissipative closure 후보는

$$
\nabla_\mu N_M^\mu=\Gamma,
\qquad
\nabla_\mu N_X^\mu=-\Gamma,
$$

$$
\Gamma=-\kappa n\left[v_D'(y)-h(z)\right],
\qquad
\kappa>0
$$

이다. 균일 FLRW에서

$$
\dot n+3Hn=0,
\qquad
\dot y=-\kappa\left[v_D'(y)-h(z)\right].
$$

열 bath를 포함한 entropy current는

$$
T\nabla_\mu S^\mu=-\mathcal A\Gamma
=\kappa n\mu_*\left[v_D'(y)-h(z)\right]^2\ge0
$$

를 만족해야 한다. 이 heat term을 stress에서 빼면 total conservation이
깨진다.

master function의 Hilbert stress는 comoving, no-entrainment limit에서

$$
T^{\mu\nu}=(\rho+p)u^\mu u^\nu+pg^{\mu\nu},
\qquad
p=n_M\mu_M+n_X\mu_X-\rho
$$

이다. 위 cold energy는 $n$에 homogeneous degree one이므로 local equilibrium에서
$p=0$이다. control field의 방정식은 부호 관례를 고정한 균일계에서

$$
\ddot\vartheta+3H\dot\vartheta+U_{,\vartheta}
-\frac{n\mu_*y}{f_\vartheta}h'(z)=0
$$

형태다. 즉 $h$를 외부 시간함수로 넣지 않고 clock EOM과 함께 풀어야 한다.

### 3.3 $1/D$ spinodal의 정확한 유도

tilted free energy의 composition Hessian은

$$
f_{yy}=v_D''(y)=\frac1y-D
$$

이다. $v_D''=0$만 적는 것은 충분하지 않다. 실제 stationary branch의 stability
loss에는 $f_y=0$도 필요하다. 두 조건은

$$
f_y=\log y+D(1-y)-h(z)=0,
$$

$$
f_{yy}=\frac1y-D=0
$$

이고, 유일한 내부 해는

$$
y_*=\frac1D,
\qquad
h(z_*)=h_*:=D-1-\log D.
$$

또한

$$
f_{yyy}(y_*)=-D^2<0.
$$

$h<h_*$에서는 $y<1/D$인 locally stable stationary branch와 $y>1/D$인
unstable branch가 존재하며, $h\uparrow h_*$에서 둘이 합쳐진다. 따라서 이것은
단순 inflection 표시가 아니라 control을 포함한 saddle-node spinodal이다.

### 3.4 composition에서 energy fraction으로 가는 정리

spinodal stationary 조건 $e_y=0$에서는

$$
\mu_M=\mu_X=e,
\qquad
\rho=n_M\mu_M+n_X\mu_X=ne.
$$

따라서 cold, comoving, no-entrainment two-current stress의 partial-energy
decomposition은

$$
\frac{\rho_M}{\rho}
=\frac{n_M\mu_M}{n_M\mu_M+n_X\mu_X}
=\frac{n_M}{n}
=y.
$$

그 결과 spinodal에서

$$
\frac{\rho_M}{\rho}=\frac1D.
$$

여기서는 common partial energy $e$가 양수이고 유한하다고 가정했다. additive
constant $C$는 spinodal 위치와 fraction을 바꾸지 않지만 $e$와 total absolute
energy를 바꾼다. 따라서 $C$는 여전히 중력적으로 물리적인 자유도이며 absolute
abundance를 예측하지 않는다.

이 단계는 $y:=\Omega_m$이라는 정의가 아니다. current master function의
partial chemical energies가 같다는 EOM 결과를 사용한다. 따라서 다음 조건을
실제로 action에 유지할 때에는 genuine bridge다.

- 두 current가 total cold stress를 포화한다.
- spinodal에서 relative drift와 entrainment가 없다.
- gradient/wall energy가 무시 가능하다.
- entropy/radiation energy가 별도 분모로 남지 않는다.
- $M$ tag가 실제 post-transition matter stress로 energy-conserving하게 이어진다.

반대로 별도 $\rho_r$, vacuum potential, wall energy 또는 heat가 존재하면

$$
\Omega_m
=\frac{n_M\mu_M}{ne+\rho_r+\rho_{\rm vac}+\rho_{\rm wall}}
\ne y
$$

가 일반적이다. 이 항들을 complement current의 공변 stress로 실제 구현하지 않고
말로만 $X=$ non-matter라고 부르면 단순 재명명이다.

### 3.5 extinction composition과의 결합

matter-tag 내부에 다시 baryon과 cold-dark tag를 두고, extinction-conditioned
relic이 equal-energy inner composition

$$
\frac{\rho_b}{\rho_M}=m=Dq
$$

를 만든다고 하자. outer spinodal과 inner freeze가 같은 $\Sigma_*$에서 일어나면

$$
\frac{\rho_b}{\rho_{\rm tot}}
=\frac{\rho_b}{\rho_M}\frac{\rho_M}{\rho_{\rm tot}}
=(Dq)\frac1D=q.
$$

이것은 nested two-current 존재구성이다. 그러나 microscopic branching tag가
Brown/Schutz current가 되는 decay action, equal-energy lemma, total yield와
post-transition stress는 아직 추가 입력이다.

### 3.6 정확성을 죽이는 핵심 현상

stationary branch의 relaxation rate는

$$
\tau_{\rm rel}^{-1}=\kappa f_{yy}
$$

이고 spinodal에서 0으로 간다. 따라서 유한한 cooling rate에서는 critical
slowing 때문에 $y$가 $1/D$에 도달하기 전에 equilibrium tracking을 잃는다.
정확한 $y_*=1/D$ freeze는 다음 중 하나를 더 요구한다.

1. singular quasistatic limit,
2. 별도 constraint field가 $f_y=0$을 강제하는 존재구성,
3. Kibble--Zurek offset을 계산하고 상쇄하는 독립 dynamics,
4. $v_D''(y)$의 sign flip에 동기화된 freeze field.

1은 물리적 유한시간 예측이 아니고, 2와 4는 새 action 선택이며, 3은 noise와
초기조건에 민감하다. 이것이 후보 A의 가장 중요한 kill test다.

## 4. 후보 B: density spectator의 정확성--유일성 tradeoff

### 4.1 유일한 matter--vacuum subsystem spectator

Brown dust의 proper density $n$으로 $\rho_m(n)=m_dn$를 만들고, action의 vacuum
상수 $V_L>0$를 둔다. 다음 subsystem composition은 local covariant scalar다.

$$
y_L(n):=\frac{\rho_m(n)}{\rho_m(n)+V_L}.
$$

$\rho_m$을 이미 계산된 $T_{00}$로 넣으면 순환정의지만, metric variation 전에
정의된 Brown master function $\rho_m(n)$을 쓰면 ratio 자체는 local하다. 가장
작은 spectator action은

$$
S_{\chi L}=\int d^4x\sqrt{-g}
\left[
-\frac12(\nabla\chi)^2
-\frac{M_\chi^2}{2}\left(Dy_L-1\right)\chi^2
-\frac{\lambda_\chi}{4}\chi^4
\right].
$$

균일 EOM은

$$
\ddot\chi+3H\dot\chi
+M_\chi^2(Dy_L-1)\chi+\lambda_\chi\chi^3=0.
$$

early dust-dominated branch에서는 $y_L\to1$이므로 $D>1$에서 $\chi=0$은 stable이고,
late vacuum-dominated branch에서는 $y_L\to0$이므로 tachyonic하다. threshold는

$$
Dy_L-1=0
\iff
(D-1)\rho_m=V_L.
$$

별도 radiation은 $y_L$의 정의에 들어가지 않는다. conserved dust와 constant
vacuum이면

$$
y_L(a)=\frac{Aa^{-3}}{Aa^{-3}+V_L},
\qquad
\frac{dy_L}{d\log a}=-3y_L(1-y_L)<0
$$

이므로 각 expanding comoving worldline에서 crossing은 유일하다. threshold
위치는

$$
a_*^3=\frac{(D-1)A}{V_L}
$$

이고 $M_\chi$와 $\lambda_\chi$는 이 위치에 들어가지 않는다. 그러나 $V_L$과
dust current의 integration constant $A$가 absolute epoch를 정하므로 scale과
coincidence가 사라진 것은 아니다.

이 trigger는 기존 함수의 Hessian과 정확히 연결된다.

$$
Dy_L-1=-y_Lv_D''(y_L).
$$

즉 $v_D$의 유일한 curvature zero를 actual order field $\chi$의 mass sign flip으로
사용한 존재구성이다. 그렇지만 $y_L$는 current EOM에서 energy composition으로
나온 것이 아니라 matter와 vacuum energy의 ratio로 action 안에서 정의됐다.
따라서 후보 A의 equal-partial-energy 정리가 아니라 **target-aware spectator
axiom**이다. vacuum은 $w=-1$이고 chemical potential이 0인 반면 dust는 $w=0$이고
양의 chemical potential을 가지므로, 두 성분을 equal-energy current라고 부르는
것은 이전 stress-type no-go를 되살린다.

### 4.2 radiation correction과 phase-law scale

$\Sigma_*$에서 matter--vacuum subsystem은

$$
\rho_m+V_L=D\rho_m
$$

이지만 full total density는

$$
\rho_{\rm tot}=D\rho_m+\rho_r
$$

이다. 평탄한 배경에서

$$
\Omega_m=\frac{1-\Omega_r}{D},
$$

$$
\Omega_b=q(1-\Omega_r),
$$

$$
\Omega_c=\frac{1-Dq}{D}(1-\Omega_r),
$$

$$
\Omega_{\rm DE}=\left(1-\frac1D\right)(1-\Omega_r).
$$

네 성분의 합은 정확히 1이다. 따라서 subsystem spectator는 global uniqueness를
얻는 대신 direct $\Omega_b=q$를 잃고 calculable radiation correction을 준다.
$\Omega_r$를 0으로 놓는 근사는 exact cosmology closure가 아니다.

또한 de Sitter entropy나 phase law가 고르는 scale을 구분해야 한다. vacuum
상수의 asymptotic scale은

$$
H_L^2:=\frac{V_L}{3M_{\rm Pl}^2}
$$

이고 transition의 instantaneous Hubble scale은

$$
H_*^2
=\frac{D}{D-1}H_L^2
+\frac{\rho_{r,*}}{3M_{\rm Pl}^2}.
$$

따라서 de Sitter entropy identity가 직접 읽는 것은 $H_L$이지 $H_*$나 관측
$H_0$가 아니다. 세 값을 같다고 놓으면 apparent closure일 뿐이다. $H_0$를 얻으려면
전이 뒤 background를 풀고 observer epoch를 독립적으로 지정해야 한다.

$\chi=0$에서는 interaction과 그 $n$-variation이 모두 $\chi^2$에 비례하여
background stress와 dust force가 0이다. $\chi\ne0$ 뒤에는

$$
\frac{\partial y_L}{\partial n}
=\frac{(d\rho_m/dn)V_L}{(\rho_m+V_L)^2}
$$

가 matter chemical potential을 바꾸고 scalar--matter energy exchange를 만든다.
Brown number-potential shift symmetry를 유지하면 $\nabla_\mu J_m^\mu=0$은 보존할
수 있지만 개별 stress는 보존되지 않는다. total Hilbert stress에는 이 interaction
variation을 반드시 포함해야 한다.

local perturbation에서는 $y_L(x)$가 달라져 $\Sigma_*$가 wrinkled surface가 된다.
각 worldline의 monotonicity는 shell crossing, clustering된 dust와 dynamical DE에서
다시 증명해야 한다. loop-generated $\Delta m_\chi^2$, matter fifth force,
domain selection과 isocurvature가 kill test다.

### 4.3 full-density action과 exact surface

Brown/Schutz fluid scalar로 metric variation 전에 정의한 local densities를
$\rho_m$, $\rho_r$, $\rho_{\rm DE}$라 하자. 이미 계산된 stress component를 다시
action에 넣으면 순환정의가 되므로 반드시 underlying number/entropy scalar 또는
DE field potential을 사용해야 한다.

$$
\mathcal F
:=(D-1)\rho_m-\rho_{\rm DE}-\rho_r
$$

를 만들고 spectator action을

$$
S_\chi=\int d^4x\sqrt{-g}
\left[
-\frac12(\nabla\chi)^2
-\frac{g}{2M^2}\mathcal F\chi^2
-\frac\lambda4\chi^4
\right]
$$

로 둔다. 균일 EOM은

$$
\ddot\chi+3H\dot\chi
+\frac{g\mathcal F}{M^2}\chi+\lambda\chi^3=0
$$

이고

$$
m_{\rm eff}^2=\frac{g\mathcal F}{M^2}.
$$

$\chi=0$ branch에서는 interaction stress가 $\chi^2$에 비례하므로 고전적
background를 바꾸지 않는다. $\mathcal F=0$인 모든 점에서는

$$
\rho_{\rm DE}+\rho_r=(D-1)\rho_m,
$$

$$
\rho_{\rm tot}=D\rho_m,
\qquad
\Omega_m=\frac1D
$$

가 평탄한 Friedmann 제약에서 정확히 성립한다. radiation을 버리지 않고 얻는
장점이다. 여기에 $f_b^{(m)}=Dq$를 결합하면 같은 면에서 $\Omega_b=q$다.

### 4.4 왜 full-density 면은 일반적으로 유일하지 않은가

전이 전의 spectator limit에서

$$
\rho_m=Aa^{-3},
\qquad
\rho_r=Ba^{-4},
\qquad
\rho_{\rm DE}=C,
\qquad
A,B,C>0
$$

이면

$$
\mathcal F(a)=(D-1)Aa^{-3}-Ba^{-4}-C.
$$

$a\to0^+$와 $a\to\infty$에서 모두 $\mathcal F<0$이고

$$
\frac{d\mathcal F}{da}
=a^{-5}\left[4B-3(D-1)Aa\right]
$$

는 하나의 maximum을 갖는다. maximum이 양수면 root가 두 개, 0이면 tangent
root 하나, 음수면 root가 없다. 즉 표준 세 성분 역사에서는 matter fraction이
radiation 시대에 올라갔다가 vacuum 시대에 내려오므로 $1/D$를 보통 두 번 지난다.

late descending branch를

$$
\Sigma_*:\quad \mathcal F=0,
\qquad
u^\mu\nabla_\mu\mathcal F<0
$$

로 읽으면 수학적으로 한 면을 고를 수 있다. 그러나 inequality는 simple
$\chi^2\mathcal F$ action이 기억하지 못한다. 실제 action에서 early root를
무시하려면 monotonic clock, hysteresis 또는 memory field가 추가되어야 한다.
그 field의 초기상태와 activation scale이 새 자유도다.

radiation을 제거하면

$$
\mathcal F=(D-1)Aa^{-3}-C
$$

는 단조 감소하여 root가 하나지만, 이 경우 radiation을 포함한 exact relation을
주었다는 장점을 잃는다.

### 4.5 full-density current, stress, quantum stability

$\mathcal F$가 fluid proper densities에 의존하면 $\chi\ne0$ 뒤에는 각 sector의
chemical potential과 stress가 변한다. number-potential shift symmetry를 유지하면
각 particle number current는 보존할 수 있지만 energy stress는 교환된다.

$$
\nabla_\mu T_i^{\mu\nu}=Q_i^\nu,
\qquad
\nabla_\mu T_\chi^{\mu\nu}=Q_\chi^\nu,
\qquad
\sum_iQ_i^\nu+Q_\chi^\nu=0.
$$

마지막 등식은 전체 action의 diffeomorphism invariance가 보장한다. interaction
variation을 어느 한 sector에서 누락하면 보장되지 않는다.

$\chi=0$은 고전 EOM의 정확한 해지만 tachyonic side에서도 정확히 0에 놓인
고전 field는 스스로 움직이지 않는다. 실제 transition은 quantum/thermal
fluctuation이 시작한다. $Z_2$는 odd term을 막지만 $\chi^2$ counterterm은 막지
못하므로 radiative correction

$$
\Delta m_\chi^2\chi^2
$$

가 생기면 zero condition이 $\mathcal F=0$에서 이동한다. 보호 대칭 또는
renormalization condition 없이는 exact surface가 technically natural하지 않다.

이 후보는 관측 $\Omega_m$이나 현재 시간을 넣지는 않는다. 그러나
$\mathcal F=0$은 목표식 $D\rho_m=\rho_{\rm tot}$을 coupling에 그대로 재배열한
것이다. 따라서 **정확하지만 target-engineered existence axiom**이며 독립 예측이
아니다.

## 5. 후보 C: symmetron/chameleon density threshold

### 5.1 최소 action

보편적인 symmetron 후보는

$$
S_C=\int d^4x\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\nabla\phi)^2
-V_*+\frac{\mu^2}{2}\phi^2-\frac\lambda4\phi^4
\right]
+S_m[A^2(\phi)g_{\mu\nu},\Psi_m]
$$

이며

$$
A(\phi)=1+\frac{\phi^2}{2M^2}+O(\phi^4/M^4)
$$

로 둔다. nonrelativistic matter에서

$$
V_{\rm eff}(\phi)
=V_*+\frac12\left(-\mu^2+\frac{\widehat\rho_m}{M^2}\right)\phi^2
+\frac\lambda4\phi^4+cdots
$$

이고

$$
m_{\rm eff}^2=-\mu^2+\frac{\widehat\rho_m}{M^2}.
$$

따라서 monotonic dilution이면

$$
\widehat\rho_{m,*}=\mu^2M^2=: \rho_*
$$

가 action-defined density surface다. $\phi=\dot\phi=0$인 정확한 bifurcation
순간, 다른 성분이 없고

$$
V_*=(D-1)\rho_*
$$

를 **선택하면** $\Omega_m=1/D$다.

### 5.2 판정

이 결과는 $V_*$의 additive constant를 $D$와 threshold density에 맞춘 tuning이다.
별도 radiation 또는 wall energy $\rho_{\rm rest}$가 있으면

$$
\Omega_m|_*
=\frac{\rho_*}{D\rho_*+\rho_{\rm rest}}
=\frac1{D+\rho_{\rm rest}/\rho_*},
$$

이므로 exact relation이 깨진다. 후보 B처럼 모든 complement density를 mass에
넣으면 exact algebra를 회복하지만 그때는 후보 B의 target-engineered 구조로
돌아간다.

conformal matter coupling의 current/stress 관계는

$$
\nabla_\mu T_m^{\mu\nu}
=\alpha(\phi)T_m\nabla^\nu\phi,
\qquad
\alpha=\frac{d\log A}{d\phi},
$$

이고 scalar stress가 반대 항을 받아 total stress를 보존한다. particle-number
shift symmetry는 number current를 보존할 수 있다. baryon까지 보편적으로
couple하면 fifth force와 mass drift가 kill test이고, CDM에만 couple하면 threshold가
total matter density를 보지 못한다.

chameleon의 density-dependent minimum은 보통 연속적으로 이동하므로 유일한
transition surface가 없다. $m_{\rm eff}/H=1$ 같은 기준을 추가하면 expansion
clock과 새 scale을 넣으며 $1/D$ 관계는 따로 유도해야 한다.

자유 scale $\mu M$이 absolute transition density를 정한다. $D$는 그 scale을
고르지 않으므로 observer-time coincidence는 그대로 남는다.

## 6. 후보 D: coupled scalar/CDM scaling과 clock crossing

### 6.1 action, EOM, fixed point

CDM만 conformally couple하는 exponential model을

$$
S_D=\int d^4x\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\nabla\phi)^2
-V_0e^{-\lambda\phi/M_{\rm Pl}}
\right]
+S_c[e^{2\beta\phi/M_{\rm Pl}}g,\Psi_c]
+S_b[g,\Psi_b]
$$

로 둔다. exp의 인자 $\phi/M_{\rm Pl}$은 무차원이다. 배경 교환식은 부호
convention을 위 action에 맞추면

$$
\dot\rho_c+3H\rho_c
=\frac{\beta}{M_{\rm Pl}}\dot\phi\rho_c,
$$

$$
\ddot\phi+3H\dot\phi+V_{,\phi}
=-\frac{\beta}{M_{\rm Pl}}\rho_c,
$$

$$
\dot\rho_b+3H\rho_b=0.
$$

CDM과 scalar의 exchange를 합치면 total stress는 보존된다. autonomous variables

$$
X=\frac{\dot\phi}{\sqrt6HM_{\rm Pl}},
\qquad
Y=\frac{\sqrt V}{\sqrt3HM_{\rm Pl}},
\qquad
L=\lambda+\beta
$$

를 쓰면 baryon이 사라진 reduced scaling point는

$$
X_* = \frac{\sqrt{3/2}}{L},
\qquad
Y_*^2=\frac{3/2+\beta L}{L^2},
$$

$$
\Omega_{c,*}=\frac{\lambda L-3}{L^2},
\qquad
w_{\rm eff}=-\frac\beta L.
$$

$\Omega_{c,*}=1/D$를 만들려면

$$
D\left[\lambda(\lambda+\beta)-3\right]
=(\lambda+\beta)^2
$$

라는 parameter curve를 선택하면 된다. 예를 들어 $\beta$를 별도 CE 무차원
함수로 고르면 양의 branch의 $L$은

$$
L=\frac{D\beta+
\sqrt{D^2\beta^2+12D(D-1)}}{2(D-1)},
\qquad
\lambda=L-\beta
$$

다. verifier는 관측값이 아닌 여러 abstract $D,\beta$에서 autonomous residual과
reduced Jacobian stability를 확인한다.

### 6.2 유일한 면과 conserved-baryon 문제

scaling point는 면이 아니라 궤도의 asymptotic 상태다. monotonic $\phi$에 별도
trigger

$$
\Sigma_*:\quad \frac\phi{M_{\rm Pl}}=z_*(D,q)
$$

를 둘 수 있지만 $V_0$와 field-origin이 그 crossing의 absolute density를 정한다.
finite crossing에서 exact $\Omega_c=1/D$가 되려면 초기상태가 정확한 invariant
scaling solution 위에 있어야 한다. attraction만으로는 유한시간 exact equality를
보장하지 않는다.

더구나 $\beta\ne0$ fixed point에서 separately conserved baryon은

$$
\frac{d\log\Omega_b}{dN}=3w_{\rm eff}
=-\frac{3\beta}{\lambda+\beta}
$$

로 변한다. 따라서 nonzero constant baryon fraction을 가진 accelerating scaling
point는 아니다. 전이 순간 precursor를 baryon과 CDM으로 energy-conserving하게
재분배하면 그 한 면에서는 $f_b^{(m)}=Dq$를 만들 수 있지만, 이후 fraction은
다시 진화한다.

kill test는 CDM fifth force, modified growth, scalar/CDM isocurvature, trigger
field fluctuation, $V_0$와 $z_*$의 coincidence다. slope curve는 관측 $\Omega$를
literal로 넣지는 않지만 원하는 $1/D$를 만들도록 고른 target-aware model
selection이다.

## 7. 후보 E: first-order transition과 spinodal limit

구조적으로 다른 order field $\psi$의 유효 action을

$$
S_E=\int d^4x\sqrt{-g}
\left[
-\frac12(\nabla\psi)^2
-U(\psi,z)
\right],
$$

$$
U(\psi,z)=\frac{\lambda_\psi}{4}\psi^4
-\frac{A_\psi}{3}\psi^3
+\frac{B(z)}{2}\psi^2+U_0(z)
$$

로 둔다. $\psi=0$ false vacuum의 local curvature는 $B(z)$이고 $B=0$에서
false-vacuum spinodal이 생긴다. 다음 두 연결이 가능하다.

$$
B(z)=M_\psi^2v_D''(y(z))
$$

를 쓰면 spinodal은 $y=1/D$이고, 또는

$$
B(z)=\frac{g\mathcal F}{M^2}
$$

를 쓰면 후보 B의 density surface와 같다.

하지만 genuine first-order transition은 일반적으로 $B=0$까지 기다리지 않고
bounce nucleation rate가 충분히 커지는 시점에 bubble로 진행한다. 그러므로 실제
$\Sigma$는 확률적인 wall 집합이고, global constant-time hypersurface가 아니며,
supercooling 때문에 $y_{\rm nuc}\ne1/D$가 일반적이다. exact spinodal을 얻으려면
모든 earlier nucleation을 억제하고 classical instability까지 기다리게 해야 한다.
이는 별도 parameter tuning이다.

scalar, wall, released radiation과 matter를 모두 포함하면 diffeomorphism
invariance로 total stress는 보존되지만 각 sector에는 source가 생긴다.

$$
\nabla_\mu J_b^\mu=\Gamma_b,
\qquad
\nabla_\mu J_c^\mu=\Gamma_c,
$$

$$
\nabla_\mu T_{\rm wall}^{\mu\nu}
+\nabla_\mu T_{\rm heat}^{\mu\nu}
+\sum_i\nabla_\mu T_i^{\mu\nu}=0.
$$

벽을 지난 뒤 $\Gamma_b=\Gamma_c=0$이면 각 relic current가 보존된다. kill test는
latent-heat entropy dilution, bubble-induced isocurvature, domain wall, stochastic
offset, gravitational-wave/inhomogeneity bounds와 vacuum-remnant EOS다.

## 8. conserved-baryon no-go를 피하는 정확한 범위

freezeout 뒤

$$
\nabla_\mu J_b^\mu=0,
\qquad
\rho_b\propto a^{-3}
$$

이면

$$
\frac{d\log\Omega_b}{d\log a}=3w_{\rm tot}.
$$

따라서 $\Omega_b=q$를 모든 epoch의 attractor라고 주장하면 기존 no-go를 다시
맞는다. 여기서 허용되는 주장은 오직

$$
\Omega_b|_{\Sigma_*}=q
$$

이다. transition 동안에는 reacting current가 허용되고

$$
\sum_i\Gamma_i=0
$$

또는 conserved charge를 만족하도록 구성한다. freezeout 뒤 source를 0으로 만들어
baryon current를 보존한다. energy source도

$$
\sum_iQ_i^\nu=0
$$

가 되도록 scalar, wall과 heat를 포함한다. 이렇게 하면 no-go를 위반하지 않지만
등식이 현재에도 성립한다고 말하려면 현재 epoch가 $\Sigma_*$라는 독립 lemma가
필요하다.

## 9. 무차원·차원 원장

자연단위에서 mass dimension을 쓴다.

| core/항 | mass dimension | 무차원 조건 또는 정규화 |
|---|---:|---|
| $D,q,m,y,z,h,C,g,\lambda,\beta$ | 0 | 직접 core 사용 가능 |
| $\log y$ | 0 | $0<y<1$인 무차원 composition |
| $e^{-D(1-q)}$ | 0 | exponent $D(1-q)$가 무차원 |
| $e^{\beta\phi/M_{\rm Pl}}$ | 0 | exponent $\beta\phi/M_{\rm Pl}$가 무차원 |
| $\vartheta/f_\vartheta$, $\phi/M_{\rm Pl}$, $T/M_T$ | 0 | 차원 1 scale로 정규화 |
| $\chi,\phi,\vartheta,M,M_{\rm Pl},\mu_*,m_*$ | 1 | dimensionful field/scale |
| $n_i,s$ | 3 | proper current density |
| $\rho_i,p_i,\mathcal F$ | 4 | action density |
| $\mathcal F/M^4$ | 0 | density trigger의 무차원 core |
| $g\mathcal F\chi^2/M^2$ | 4 | spectator action term |
| $g\mathcal F/M^2$ | 2 | $m_{\rm eff}^2$ |
| $M_\chi^2(Dy_L-1)\chi^2$ | 4 | subsystem spectator action term |
| $n\mu_*v_D$ | 4 | multi-fluid master energy |
| $\kappa$ | 1 | $\Gamma=-\kappa n(\cdots)$의 $[\Gamma]=4$ |
| $u^\mu\nabla_\mu\mathcal F/M_*^5$ | 0 | oriented crossing을 쓸 때 필요한 정규화 |

모든 log, exp, fixed-point, composition core는 무차원이다. 차원 정합은 물리적
정당성이나 radiative stability를 보장하지 않는다.

## 10. 후보 비교

| 후보 | action-defined surface | $1/D$ exact성 | 추가 자유도/초기조건 | target-awareness | 주요 kill test | absolute scale·coincidence |
|---|---|---|---|---|---|---|
| A. $v_D$ multi-fluid spinodal | $f_y=f_{yy}=0$, $h(z)=h_*$ | cold two-current 포화에서 exact | 두 current, entropy, clock, $C,\mu_*$, $\kappa(z)$, freeze field | $1/D$는 Hessian에서 나오지만 $v_D$와 tag map은 선택 | critical slowing, heat, entrainment, post-transition EOS | $C,h(z)$의 microphysical scale와 clock IC가 남음 |
| B1. matter--vacuum spectator | $Dy_L-1=0$ | subsystem에서 exact, full $\Omega_b=q(1-\Omega_r)$ | $\chi,M_\chi,\lambda_\chi,V_L$, dust current | ratio와 zero를 coupling에 재기록 | loop mass, fifth force, wrinkled surface, radiation correction | $V_L$와 dust integration constant가 epoch를 정함 |
| B2. full-density spectator | $\mathcal F=0$ | radiation 포함 $\Omega_b=q$ exact | $\chi,g,M,\lambda$, fluid master functions, memory gate | 목표식을 coupling에 재기록 | 보통 두 roots, loop mass shift, fifth force | density integration constants와 gate가 epoch를 정함 |
| C. symmetron/chameleon | $m_{\rm eff}^2=0$ | two-sector와 tuned $V_*$에서만 exact | $\phi,\mu,M,\lambda,V_*$, field IC | $V_*=(D-1)\rho_*$ tuning | radiation correction, fifth force, domains | $\mu^2M^2$가 새 absolute density |
| D. coupled scalar/CDM | scaling point + clock crossing | exact invariant trajectory에서만 | $V_0,\lambda,\beta,z_*$, scalar IC | slope curve가 목표 fraction용 선택 | finite-time offset, growth, CDM force, baryon decay | $V_0,z_*$와 field origin이 epoch를 정함 |
| E. first order | bubble 또는 false-vacuum spinodal | spinodal limit에서만 | wall field, barrier, nucleation state, thermal history | $B(z)$ 연결을 선택 | stochastic nucleation, entropy, walls/isocurvature | barrier/temperature scale와 supercooling이 남음 |

## 11. 자유도와 관측 독립성에 대한 최종 판정

어느 후보도 현재 시각, 현재 $H$, 관측된 $\Omega_m$ 숫자를 coupling에 넣지 않아도
수학적 전이 면을 만들 수 있다. 그러나 이는 observer-time coincidence를 푼 것이
아니다.

- 후보 A의 $h(z_*)$는 $D$로 정해지지만 $z$의 absolute evolution은 $U(\vartheta)$,
  $f_\vartheta$와 초기조건이 정한다.
- 후보 B1의 unique ratio crossing은 $V_L$와 dust integration constant가 정하고,
  후보 B2의 late root 선택에는 추가 memory/clock까지 필요하다.
- 후보 C의 density는 $\mu^2M^2$가 정한다.
- 후보 D의 crossing은 $V_0$, field origin과 $z_*$가 정한다.
- 후보 E의 nucleation은 barrier와 thermal state가 정한다.

따라서 $\Sigma_*$가 어떤 물리 epoch인지 사전 고정하는 독립 observable이 필요하다.
그 observable을 보고 난 뒤 parameter를 맞추면 target leakage다. 허용되는 blind
절차는 action과 모든 dimensionless/dimensionful parameter를 먼저 고정하고,
$\Sigma_*$에서 함께 나오는 entropy injection, isocurvature, fifth force, growth,
wall signal 또는 post-transition EOS를 독립적으로 교차검정하는 것이다.

## 12. verifier가 확인한 것과 확인하지 않은 것

다음 명령은 exit code 0이다.

```powershell
python "_workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_transient_transition_action.py"
```

검산 항목은 다음과 같다.

1. abstract $D>1$ 여러 값에서 $m=Dq<1$과 $(Dq)(1/D)=q$,
2. $v_D$ tilted spinodal의 $y_*=1/D$, $h_*=D-1-\log D$와 양쪽 branch stability,
3. $\mathcal F=0\Rightarrow\Omega_m=1/D$ 및 radiation+vacuum 예제의 두 roots,
4. radiation이 없을 때 full-density trigger의 유일 root,
5. matter--vacuum $y_L$의 단조 unique crossing과
   $\Omega_b=q(1-\Omega_r)$ radiation correction,
6. $Dy_L-1=-y_Lv_D''(y_L)$의 정확한 algebra,
7. coupled scalar/CDM fixed-point residual과 reduced Jacobian local stability,
8. 모든 action term과 log/exp/fixed-point core의 차원.

verifier는 microphysical action의 존재, quantum naturalness, 관측 적합성,
entropy history, 현재 epoch와 $\Sigma_*$의 동일성 또는 exact freeze를 증명하지
않는다. 입력은 관측 central value가 아닌 abstract dimensionless test 값뿐이다.

## 13. 다음 승격 조건

이 경로를 [미완성]에서 올리려면 최소한 다음을 하나의 action/EFT에서 동시에
보여야 한다.

1. Brown/Carter 또는 Schwinger--Keldysh 변수로 outer current와 inner branching
   relic current를 함께 정의한다.
2. $v_D$와 monotonic $h(z)$가 microscopic detailed balance에서 나온다.
3. finite-rate cooling에서도 $y_f=1/D$가 exact하거나 계산된 오차범위 안에 든다.
4. heat, radiation, wall, scalar와 dark-energy stress를 포함해
   $\nabla_\mu T_{\rm tot}^{\mu\nu}=0$을 직접 검산한다.
5. transition 후 $\nabla_\mu J_b^\mu=0$과 baryon/CDM perturbation 안정성을 보인다.
6. $\Sigma_*$의 absolute scale와 관측 epoch 식별을 관측 target 없이 먼저 고정한다.
7. fifth force, entropy dilution, isocurvature, growth 또는 transition signal 중 하나를
   blind kill test로 사전 등록한다.

이 조건이 충족되기 전에는 목표 $\Omega_b=q$를 내릴 이유는 없지만, 살아 있는
정본 지위는 오직 **전이 면에서의 조건부 미완성 가설**이다.
