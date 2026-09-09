# 1. Navier--Stokes: 방정식, 에너지 항등식, 조건부 변분 목적함수

이 장은 Navier--Stokes 방정식을 CE에서 새로 유도하거나 3차원 정칙성
문제를 해결했다고 주장하지 않는다. 표준 비압축성 방정식의 정확한 에너지
구조와, 수치 최적화에 사용할 수 있는 잔차 functional을 분리한다.

독자는 벡터미적분, 약한 미분과 $L^2$ 내적의 기본 정의를 안다고 가정한다.
먼저 1.1절에서 해와 경계의 정의역을 고정하고, 1.2–1.3절에서 정확히
보존되는 에너지 구조와 3차원 열린 문제를 분리한다. 그 뒤 1.4–1.7절은
무차원 수치 목적함수와 CE source가 추가될 때 새로 필요한 가정을
차례로 밝힌다. 일반 변분 원리와 소산계의 경계는
[master action](06_Master_Action_Universal_Derivation.md)에서 보완한다.

## 1.1 정의역과 경계조건

에너지 항등식은 방정식의 모양만으로 성립하지 않고, 적분 경계에서 어떤
항이 사라지는지까지 지정해야 한다. 따라서 먼저 유체 branch와 해의
함수공간을 고정한다.

**[공리] 유체 branch:** $d=2$ 또는 $3$, $\Omega=\mathbb T^d$인
periodic domain 또는 매끄러운 bounded domain의 no-slip 조건
$\mathbf u|_{\partial\Omega}=0$을 택한다. 밀도 $\rho>0$와 동점성계수
$\nu>0$는 상수이고 $\mathbf f$는 단위질량당 외력이다. periodic
branch에서는 속도와 외력의 공간평균을 0으로 고정한다.

**[정의]** $H_\sigma$는 위 경계조건을 따르는 매끄러운
divergence-free vector field들의 $L^2$ closure이고, $V_\sigma$는
같은 집합의 $H^1$ closure다. bounded branch에서는
$V_\sigma\subset H_0^1(\Omega)^d$이고, periodic branch에서는
zero-mean periodic $H^1$ 공간이다. $V_\sigma$의 norm은
$\|\mathbf u\|_{V_\sigma}:=\|\nabla\mathbf u\|_2$로 둔다.

**[정의]** 비압축성 Navier--Stokes 방정식은
$$
\partial_t\mathbf u
+(\mathbf u\cdot\nabla)\mathbf u
=-\frac1\rho\nabla p+\nu\Delta\mathbf u+\mathbf f,
\qquad
\nabla\cdot\mathbf u=0.
$$
pressure는 divergence-free 제약을 집행하는 Lagrange multiplier다.

## 1.2 smooth solution의 운동에너지 항등식

이제 1.1절의 경계조건이 정확히 어디에 쓰이는지 확인한다. 아래 항등식은
충분히 매끄러운 해에만 먼저 성립하며, 약한 해로의 확장은 다음 절에서
별도로 다룬다.

**[정리]** 위 경계조건을 만족하는 충분히 매끄러운 해에 대해
$$
\frac12\frac{d}{dt}\|\mathbf u(t)\|_{L^2(\Omega)}^2
+\nu\|\nabla\mathbf u(t)\|_{L^2(\Omega)}^2
=(\mathbf f,\mathbf u)_{L^2}.
$$
물리적 kinetic energy에는 전체 식에 $\rho$를 곱한다.

**증명.** 방정식에 $\mathbf u$를 내적해 적분한다. 비선형항은
$$
\int_\Omega\mathbf u\cdot
(\mathbf u\cdot\nabla)\mathbf u\,dx
=\frac12\int_\Omega\nabla\cdot
(|\mathbf u|^2\mathbf u)\,dx=0,
$$
pressure 항은
$\int\mathbf u\cdot\nabla p=-\int p\nabla\cdot\mathbf u=0$이고,
부분적분으로
$\int\mathbf u\cdot\Delta\mathbf u=-\|\nabla\mathbf u\|_2^2$다.

**[산출]** $\mathbf f\in V_\sigma'$이면 Young 부등식으로
$$
\frac12\frac{d}{dt}\|\mathbf u\|_2^2
+\frac{\nu}{2}\|\nabla\mathbf u\|_2^2
\le\frac{1}{2\nu}\|\mathbf f\|_{V_\sigma'}^2.
$$
이는 에너지의 a priori bound이며 pointwise smoothness를 자동으로 주지
않는다.

## 1.3 weak solution의 지위

매끄러움이 사라지면 같은 계산을 등식으로 반복할 수 없으므로, 해의
함수공간과 보존되는 부등식을 먼저 구분해야 한다.

**[정의]** Leray--Hopf weak solution은
$$
\mathbf u\in
L^\infty_{\rm loc}([0,\infty);H_\sigma)
\cap
L^2_{\rm loc}([0,\infty);V_\sigma)
$$
이고 divergence-free test function에 대한 weak 방정식과 energy
inequality를 만족하는 해다.

**[정리]** $\mathbf u_0\in H_\sigma$와
$\mathbf f\in L^2_{\rm loc}([0,\infty);V_\sigma')$를 둔다.

1. $d=2$에서는 표준 유한에너지 초기자료와 위 경계조건 아래 global weak
   solution이 유일하고, 적절한 정칙 초기자료에는 global strong solution이
   존재한다.
2. $d=3$에서는 global Leray--Hopf weak solution의 존재가 알려져 있다.
3. $d=3$의 임의 smooth divergence-free 초기자료에 대한 global
   smoothness와 weak solution의 일반 유일성은 아직 증명되지 않았다.

**[미완성]** CE functional을 추가하는 것만으로 3차원 global regularity가
따른다는 증명은 없다.

## 1.4 무차원화

에너지 항등식의 적용 범위를 고정했으므로, 다음에는 수치 비교에 필요한
척도를 제거한다. 기준 길이와 속도는 물리 법칙이 아니라 좌표 선택이다.

**[정의]** 기준 length $L>0$, velocity $U>0$를 택해
$$
\mathbf x=L\hat{\mathbf x},\quad
t=\frac LU\hat t,\quad
\mathbf u=U\hat{\mathbf u},\quad
p=\rho U^2\hat p,\quad
\mathbf f=\frac{U^2}{L}\hat{\mathbf f}
$$
로 둔다.

**[산출]** 방정식은
$$
\partial_{\hat t}\hat{\mathbf u}
+(\hat{\mathbf u}\cdot\hat\nabla)\hat{\mathbf u}
=-\hat\nabla\hat p
+\frac1{\operatorname{Re}}\hat\Delta\hat{\mathbf u}
+\hat{\mathbf f},
\qquad
\hat\nabla\cdot\hat{\mathbf u}=0,
$$
$$
\operatorname{Re}:=\frac{UL}{\nu}
$$
가 된다. Reynolds number와 모든 hatted 변수는 무차원이다.

## 1.5 Taylor--Green exact branch

무차원 방정식의 각 항을 확인하려면 일반 해보다 직접 대입 가능한 기준
해가 유용하다. 다음 2차원 periodic branch는 그러한 검산 기준일 뿐,
3차원 정칙성의 증거는 아니다.

**[공리] 모델 선택:** 2차원 square torus
$[0,2\pi L]^2$, $\mathbf f=0$, 정수 $n\ge1$에 대한
wave number $k=n/L$와 amplitude $U_0$를 택한다.

**[정리]**
$$
u_x=U_0\sin(kx)\cos(ky)e^{-2\nu k^2t},
\qquad
u_y=-U_0\cos(kx)\sin(ky)e^{-2\nu k^2t},
$$
$$
p=p_0+\frac{\rho U_0^2}{4}
[\cos(2kx)+\cos(2ky)]e^{-4\nu k^2t}
$$
는 비압축성 Navier--Stokes의 정확한 smooth solution이다. 직접 미분하면
$\nabla\cdot\mathbf u=0$, viscous decay와 pressure가 각각 시간미분과
convective gradient를 상쇄한다.

이 해는 수치코드의 manufactured-solution 기준선으로 유용하지만
3차원 난류 정칙성을 검사하는 충분한 사례가 아니다.

## 1.6 잔차 최소화 functional

정확해의 검산과 달리 수치 근사에서는 방정식 위반을 하나의 무차원 양으로
측정해야 한다. 다음 functional은 그 측정 규칙이며 물리 작용이 아니다.

**[정의]** 1.4절의 무차원 변수로
$$
\mathcal R(\hat{\mathbf u},\hat p)
:=
\partial_{\hat t}\hat{\mathbf u}
+(\hat{\mathbf u}\cdot\hat\nabla)\hat{\mathbf u}
+\hat\nabla\hat p
-\operatorname{Re}^{-1}\hat\Delta\hat{\mathbf u}
-\hat{\mathbf f}
$$
를 정의한다.

**[공리] 수치모형:** dimensionless spacetime domain
$\hat I\times\hat\Omega$, $\kappa>0$와 경계·초기조건을 고정하고
$$
\mathcal J[\hat{\mathbf u},\hat p]
=\frac12\int_{\hat I\times\hat\Omega}
\left(
|\mathcal R|^2+\kappa|\hat\nabla\cdot\hat{\mathbf u}|^2
\right)d\hat t\,d^d\hat x
$$
를 least-squares objective로 사용한다.

**[정리]** $\mathcal J\ge0$이고, 허용 함수가 경계·초기조건을 만족할 때
$$
\mathcal J=0
\quad\Longleftrightarrow\quad
\mathcal R=0,\quad
\hat\nabla\cdot\hat{\mathbf u}=0
\quad\text{a.e.}
$$
다. 이는 residual norm의 직접 귀결이다.

**[미완성]** $\mathcal J$의 최소값이 0인지, minimizer가 존재·유일한지,
discretization이 수렴하는지는 함수공간, coercivity, inf--sup 조건과
resolution에 의존한다. dissipative Navier--Stokes를 이 objective 하나의
Lorentzian 물리 작용에서 유도했다고 부르지 않는다.

## 1.7 추가 CE source의 조건

기존 유체 방정식에 CE 자유도를 넣는 순간 에너지 수지는 새 stress의
구성법칙에 의존한다. 그러므로 효과를 주장하기 전에 가능한 결합 형태와
그것이 에너지 항등식에 남기는 항을 먼저 적는다.

**[공리] 별도 branch:** CE 자유도를 유체에 결합하려면 momentum equation에
$\rho^{-1}\nabla\cdot\tau_{\rm CE}$ 또는 명시적인 body force를 추가하고
그 구성방정식, causal relaxation time과 경계조건을 공급한다.

**[정리]** 수정된 energy balance에는
$$
\frac1\rho\int_\Omega
\mathbf u\cdot(\nabla\cdot\tau_{\rm CE})\,dx
=-\frac1\rho\int_\Omega\tau_{\rm CE}:\nabla\mathbf u\,dx
$$
가 단위질량당 항으로 추가된다. 물리적 energy balance 전체에 $\rho$를
곱하면 이에 대응하는 항은
$$
\int_\Omega
\mathbf u\cdot(\nabla\cdot\tau_{\rm CE})\,dx
=-\int_\Omega\tau_{\rm CE}:\nabla\mathbf u\,dx
$$
다. 따라서 $\tau_{\rm CE}$의 부호·대칭·동역학 없이 억제나 안정화를
결론낼 수 없다.

## 1.8 요약

앞 절의 결과는 표준 유체 정리, 수치 목적함수, 그리고 아직 비어 있는 CE
결합을 서로 바꾸어 부르지 않도록 구분해야 한다. 다음 표는 그 적용 범위와
형식 지위를 한곳에 모은다.

| 항목 | 지위 | 범위 |
|---|---|---|
| smooth energy identity | [정리] | periodic/no-slip, incompressible smooth 해 |
| Leray--Hopf energy inequality | [정리] | finite-energy weak solution |
| 2D Taylor--Green 해 | [정리] | 지정 periodic branch |
| dimensionless least-squares objective | [공리]·[정리] | 잔차 0과 PDE의 동치 |
| 3D global regularity | [미완성] | 표준 공개 문제 |
| CE stress의 안정화 효과 | [미완성] | constitutive law와 energy estimate 부재 |
