# 대칭·factorization·equipartition에 의한 밀도 사상 시도

Status: COMPLETE

## 0. 판정 요약

관측된 \(\Omega_b\)나 \(H_0\)를 식에 넣지 않고도 다음의 **좁은 정리**는
완전히 성립한다.

> 소멸 사건 \(E\)와 셀 에너지 \(W\)가 무상관이면, 에너지로 가중한
> 소멸 sector의 분율은 소멸확률 \(q=P(E)\)와 같다. 특히 label sector와
> energy sector의 상태·측도가 정확히 곱으로 분해되면 이 등식은 정확하다.

그러나 이 결과가 직접 주는 것은, 가장 강한 late-time dust 구성에서도

\[
f_b^{(m)}:=\frac{\rho_b}{\rho_b+\rho_{\rm cdm}}=q
\]

이다. 우주론의 critical-density fraction은

\[
\Omega_b:=\frac{\rho_b}{\rho_{\rm crit}}
=q\,\Omega_m,
\qquad
\Omega_m:=\frac{\rho_b+\rho_{\rm cdm}}{\rho_{\rm crit}},
\]

이므로 \(\Omega_b=q\)는 \(\Omega_m=1\)이라는 별도 조건 없이는 나오지
않는다. 암흑에너지가 공존하고 baryon current가 보존되면
\(f_b^{(m)}=q\)는 보존될 수 있지만 \(\Omega_b=q\)는 지속될 수 없다.

따라서 symmetry/factorization route는 **확률을 dust 내부 조성비로 바꾸는
조건부 존재구성**까지는 제공하지만, 확률을 현재 critical-density
fraction으로 바꾸는 자연 유도는 제공하지 않는다. 아래에는 이 긍정 결과와
정확한 실패 지점을 모두 증명한다.

## 1. 가중 사건 분율의 정확한 정리

확률공간 \((\mathcal X,\mathcal F,P)\), 사건 \(E\in\mathcal F\),
indicator \(X=\mathbf 1_E\), 그리고 셀 또는 미시상태의 비음 에너지
\(W\geq0\)를 잡는다. 다음을 가정한다.

\[
q:=P(E)\in(0,1),\qquad
0<\mu:=\mathbb E[W]<\infty.
\]

에너지 가중 사건 분율을

\[
\Omega_E^{(W)}
:=\frac{\mathbb E[XW]}{\mathbb E[W]}
\]

로 정의한다. 이 기호는 아직 cosmological \(\Omega_b\)가 아니다.

### 1.1 covariance 항등식 `[정리]`

\[
\boxed{
\Omega_E^{(W)}-q
=\frac{\operatorname{Cov}(\mathbf 1_E,W)}{\mathbb E[W]}
}
\]

이다. 실제로

\[
\operatorname{Cov}(X,W)
=\mathbb E[XW]-\mathbb E[X]\mathbb E[W]
=\mathbb E[XW]-q\mu
\]

를 \(\mu\)로 나누면 된다. 따라서 \(q\)만으로 energy fraction이 정해지는
것이 아니라, 정확히 하나의 추가 통계량인 \(\operatorname{Cov}(X,W)\)가
남는다.

### 1.2 equal conditional energy의 필요충분성 `[정리]`

\[
\mu_E:=\mathbb E[W\mid E],\qquad
\mu_{\bar E}:=\mathbb E[W\mid E^c]
\]

라 하면

\[
\boxed{
\Omega_E^{(W)}=q
\Longleftrightarrow
\mu_E=\mu_{\bar E}
\Longleftrightarrow
\mu_E=\mu_{\bar E}=\mu .
}
\]

증명은

\[
\Omega_E^{(W)}=\frac{q\mu_E}
{q\mu_E+(1-q)\mu_{\bar E}}
\]

에 \(0<q<1\)을 사용하는 직접 대수다. 따라서 다음의 관계를 구분해야
한다.

- \(X\)와 \(W\)의 독립성은 충분하지만 필요하지 않다.
- full distribution의 equipartition도 충분하지만 필요하지 않다.
- 정확히 필요한 것은 두 conditional **mean energy**의 equality 하나다.
- 단순한 label exchangeability나 동일한 셀 개수는 conditional mean의
  equality를 자동으로 보장하지 않는다.

### 1.3 \(q\) 단독 결정에 대한 구조적 no-go `[정리]`

같은 \(q\)를 고정하고

\[
W=a\quad\text{on }E,\qquad W=b\quad\text{on }E^c,
\qquad a,b>0
\]

를 택하면

\[
\Omega_E^{(W)}
=\frac{qa}{qa+(1-q)b}.
\]

\(a/b\)를 \(0\)에서 \(\infty\)까지 바꾸면 이 분율은 \(0\)에서 \(1\)까지
변한다. 모든 모형은 동일한 extinction probability \(q\)를 갖는다.
그러므로 **고정점 확률만으로 energy fraction을 정하는 정리는 존재할 수
없다.** 추가 대칭, 상태 factorization 또는 equal-energy 공리가 반드시
필요하다.

## 2. product Hilbert/probability space route

### 2.1 정확한 factorization 정리

Hilbert space와 상태가

\[
\mathcal H=\mathcal H_X\otimes\mathcal H_W,
\qquad
\varrho=\varrho_X\otimes\varrho_W
\]

로 분해되고, 소멸사건 projector와 에너지 관측량이 각각

\[
\Pi_E=\Pi_X\otimes I_W,
\qquad
\widehat W=I_X\otimes H_W
\]

라고 하자. 그러면

\[
\frac{\operatorname{Tr}(\varrho\Pi_E\widehat W)}
{\operatorname{Tr}(\varrho\widehat W)}
=\operatorname{Tr}(\varrho_X\Pi_X)=q.
\]

즉 **product state와 label-blind energy operator**는
\(\operatorname{Cov}(X,W)=0\)을 정확히 강제한다.

고전 확률 또는 Euclidean functional integral에서도 동일하다. 고정된
외부 metric \(g_0\) 위에서

\[
I[\xi,\psi;g_0]=I_X[\xi;g_0]+I_W[\psi;g_0],
\qquad
\mathcal D\mu=\mathcal D\mu_X\,\mathcal D\mu_W
\]

이고 boundary state도 곱이면

\[
Z=Z_XZ_W,qquad
\langle \mathbf1_E[\xi]W[\psi]\rangle
=\langle\mathbf1_E[\xi]\rangle\langle W[\psi]\rangle.
\]

이는 action factorization만이 아니라 **측도와 경계상태의 factorization**도
요구한다. 얽힌 초기상태 또는 상관된 boundary condition이 있으면 action이
분리되어도 covariance는 남을 수 있다.

### 2.2 대칭이 실제로 보장하는 것

label factor의 모든 unitary \(U_X\)에 대해 에너지 연산자가

\[
[\widehat W,U_X\otimes I_W]=0
\]

를 만족하면, 표준 tensor-factor commutant에서
\(\widehat W=I_X\otimes H_W\)다. 따라서 강한 label-blind symmetry는
에너지가 label을 직접 읽지 못하게 한다. 그러나 이것만으로 상태
\(\varrho\)가 product라는 결론은 나오지 않는다. 상관된 상태에서는
\(\Pi_E\)와 \(H_W\)가 서로 다른 factor에 있어도 covariance가 남는다.

반대로 상태까지 모든 \(U_X\) 아래 불변으로 두면 label state는 해당
irreducible block에서 maximally mixed가 되어 \(q\)가 projector rank의
비율로 제한된다. 일반적인 Poisson 고정점 \(q\)를 얻으려면 label state는
그 완전대칭을 이미 깨야 한다. 따라서 필요한 구조는 단순한 “대칭” 한
단어가 아니라 다음 두 독립 조건이다.

1. energy observable의 label blindness,
2. branching label state와 energy state의 무상관/product preparation.

### 2.3 gravity가 exact product를 깨는 이유

고정 배경에서는 \(I_X+I_W\)가 분리되어도 동적인 metric을 적분하면

\[
Z=\int\mathcal Dg\,e^{-I_g[g]}Z_X[g]Z_W[g]
\]

가 된다. 공통 metric과 Hamiltonian constraint를 통해 두 sector가 다시
상관된다. 고전적으로도 두 stress tensor가 같은 Friedmann 방정식의
\(H\)를 공유한다. 그러므로 exact factorization은 test-sector 또는
고정-background 근사에서는 정리지만, 공변 우주론에서는 별도 decoupling
정리나 작게 억제된 gravitational covariance의 오차경계가 필요하다.

## 3. many-cell/ergodic limit와 finite-\(N\) fluctuation

셀 \(i=1,\ldots,N\)에 대해 \(X_i\in\{0,1\}\), \(W_i>0\)를 두고

\[
\widehat\Omega_{E,N}
:=\frac{\sum_iX_iW_i}{\sum_iW_i}
\]

라 하자.

### 3.1 iid product 모형 `[정리]`

\(X_i\sim\operatorname{Bernoulli}(q)\)가 \(W_i\)들과 독립이고
\((X_i,W_i)\)가 iid이며 \(0<\mu=\mathbb E W<\infty\)이면

\[
\mathbb E[\widehat\Omega_{E,N}\mid W_1,\ldots,W_N]=q
\]

이므로 estimator는 유한 \(N\)에서도 정확히 unbiased다. strong law로

\[
\widehat\Omega_{E,N}\xrightarrow{\rm a.s.}q.
\]

또한 \(\mathbb E[W^2]<\infty\)이면 ratio delta method로

\[
\sqrt N(\widehat\Omega_{E,N}-q)
\Rightarrow
\mathcal N\!\left(
0,\frac{q(1-q)\mathbb E[W^2]}{\mu^2}
\right).
\]

따라서 leading finite-cell 표준편차는

\[
\boxed{
\sigma_N\simeq
\sqrt{\frac{q(1-q)}{N}
\left(1+\frac{\operatorname{Var}W}{\mu^2}\right)}.
}
\]

모든 셀 에너지가 같으면 binomial 결과
\(\sigma_N=\sqrt{q(1-q)/N}\)로 줄어든다. energy weight의 분산은
effective sample size를 줄인다.

### 3.2 stationary ergodic 모형 `[정리]`

iid 대신 \((X_i,W_i)\)가 stationary ergodic이고 적분 가능하면 ergodic
theorem으로

\[
\widehat\Omega_{E,N}\longrightarrow
\frac{\mathbb E[XW]}{\mathbb E[W]}
=q+\frac{\operatorname{Cov}(X,W)}{\mu}.
\]

즉 many-cell limit 자체는 covariance를 없애지 않는다. mixing과
summable correlation을 더 가정할 때만 central limit variance를

\[
\operatorname{Var}(\widehat\Omega_{E,N})
\simeq\frac1{N\mu^2}
\sum_{k}\operatorname{Cov}(Y_0,Y_k),
\qquad Y_i:=(X_i-q)W_i
\]

로 쓸 수 있다. 장거리 상관이 있으면 \(1/N\) scaling도 실패할 수 있다.

마지막으로 Galton--Watson의 “eventual extinction”은 모든 미래 세대를
참조하는 tail event다. 이를 한 spacetime cell의 국소 field label로 쓰려면
generation clock, coarse-graining map과 freeze-out hypersurface가 추가로
필요하다. ergodicity는 이 locality 문제를 해결하지 않는다.

## 4. 공변 dust 존재구성

이 절은 \(q\)를 관측에서 넣지 않고, 계산된 \(q_*\)를 conserved dust의
조성비로 구현할 수 있음을 보이는 **constrained variational existence
construction**이다. 자연 유도라고 주장하지 않는다.

### 4.1 작용

metric 부호를 \((-+++)\)로 두고 두 irrotational dust clock
\(T_A\), 에너지밀도 multiplier \(\rho_A\)를
\(A\in\{b,c\}\)에 대해 도입한다.

\[
\begin{aligned}
S={}&\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R-\Lambda
-\frac12\sum_{A=b,c}\rho_A
\left(g^{\mu\nu}\partial_\mu T_A\partial_\nu T_A+1\right)
\right]+S_{\Sigma_*},\\
S_{\Sigma_*}={}&\eta\left[(1-q_*)N_b-q_*N_c\right],\\
N_A:={}&-\int_{\Sigma_*}d\Sigma_\mu\,J_A^\mu,
\qquad
J_A^\mu=n_Au_A^\mu,
\qquad \rho_A=m_A n_A .
\end{aligned}
\]

여기서 \(q_*\)는 \(q_*=e^{-D(1-q_*)}\)의 최소근이며
\(\eta\)는 boundary constraint multiplier다. 이 작용에는 관측
\(\Omega_b\) 또는 \(H_0\)가 없다.

\(\rho_A\) 변분은

\[
g^{\mu\nu}\partial_\mu T_A\partial_\nu T_A=-1
\]

을, \(T_A\) 변분은

\[
\nabla_\mu(\rho_Au_A^\mu)=0,
\qquad u_{A\mu}:=-\partial_\mu T_A
\]

을 준다. metric 변분으로 얻는 on-shell stress tensor는

\[
T_A^{\mu\nu}=\rho_Au_A^\mu u_A^\nu,
\qquad p_A=0,qquad w_A=0.
\]

고정 질량 \(m_A\)이면 위 에너지 current 보존은 number-current 보존
\(\nabla_\mu J_A^\mu=0\)과 같다. \(\eta\) 변분은

\[
N_b=q_*(N_b+N_c)
\]

를 강제한다. \(m_b=m_c\)라는 equal-energy 조건과 균일 comoving FLRW
해를 택하면

\[
\rho_A\propto a^{-3},\qquad
\boxed{
\frac{\rho_b}{\rho_b+\rho_c}=q_*
}
\]

가 모든 후속 시각에 보존된다.

### 4.2 이 construction이 증명하는 것과 증명하지 않는 것

이것은 다음 명제의 존재 증명이다.

> q-derived boundary count, equal rest energy, 두 conserved dust current를
> 동시에 공리화하면 \(f_b^{(m)}=q\)인 공변 모형을 쓸 수 있다.

그러나 \(S_{\Sigma_*}\)가 바로 빠져 있던 bridge를 constraint로 넣는다.
이를 stochastic branching preparation으로 바꾸더라도 “extinction label을
baryon current에 붙인다”는 state map은 남는다. 또한 \(m_b=m_c\)는
label-exchange symmetry로 보호할 수 있지만, \(b\) sector에 Standard
Model gauge charge를 주고 \(c\) sector를 dark matter로 만들면 그
exchange symmetry가 깨지고 radiative·binding·thermal energy까지 같은지는
다시 증명해야 한다.

따라서 이 action은 nonexistence를 반박하지만 natural derivation을
완료하지 않는다.

### 4.3 scalar fixed-point action의 EOS no-go

계약의 scalar 후보

\[
S_x=\int\sqrt{-g}\left[-\frac{F^2}{2}(\partial x)^2
-M^4v_D(x)\right]
\]

에서 균일 정지점 \(x=q_*\)는

\[
T^{(x)}_{\mu\nu}=-M^4v_D(q_*)g_{\mu\nu},
\qquad w_x=-1\quad\bigl(v_D(q_*)\ne0\bigr)
\]

을 준다. 이는 baryon dust의 \(w_b\simeq0\)과 다르다. 더 강하게,
\(v_D\mapsto v_D+C\)는 정지점과 Hessian을 전혀 바꾸지 않으면서
에너지밀도를 \(M^4C\)만큼 임의로 이동시킨다. \(M\)도 고정점 위치를
바꾸지 않는 자유 scale이다. 따라서 scalar 고정점의 존재·안정성만으로
\(\rho_b\), \(\rho_{\rm crit}\) 또는 그 비를 정할 수 없다는 것이 정확한
degeneracy no-go다.

## 5. matter fraction과 critical-density fraction

두 dust sector에 대한 factorization/equipartition 정리는

\[
f_b^{(m)}=\frac{\rho_b}{\rho_b+\rho_c}=q
\]

만 정한다. 반면 cosmological parameter는

\[
\Omega_b=\frac{\rho_b}{3M_{\rm Pl}^2H^2},
\qquad
\Omega_m=\frac{\rho_b+\rho_c}{3M_{\rm Pl}^2H^2}
\]

이므로 항등적으로

\[
\boxed{\Omega_b=q\Omega_m.}
\]

\(\Omega_b=q\)를 얻는 방법은 세 가지뿐이다.

1. 해당 hypersurface에서 \(\Omega_m=1\)을 별도 공리화한다. 평탄 GR과
   비음이 아닌 다른 성분까지 가정하면 radiation과 dark energy가 그
   hypersurface에서 0이어야 한다. 비평탄 모형에서는 nonmatter density와
   curvature의 별도 상쇄를 허용할 수 있지만, 그 상쇄가 또 하나의 조건이다.
2. complement를 CDM이 아니라 **모든 critical energy**로 재정의하고 그
   전체에 equal conditional energy를 공리화한다. 그러면 서로 다른 EOS를
   가진 vacuum/radiation을 하나의 iid energy-cell ensemble로 묶어야 하며,
   그 공변 미시모형이 추가로 필요하다.
3. 특정 hypersurface에서 \(\rho_b=q\rho_{\rm crit}\)를 boundary constraint로
   직접 넣는다. 이것은 계산 가능한 모형이지만 유도가 아니라 equality의
   재공리화다.

평탄성만으로는 충분하지 않다. 평탄 GR은
\(\rho_{\rm tot}=\rho_{\rm crit}\)를 줄 뿐이며, dark energy가 있으면
\(\Omega_m<1\)일 수 있다.

## 6. dark energy 공존과 보존법칙

이 절에서는 flat GR, 즉 \(\rho_{\rm tot}=\rho_{\rm crit}\)를 명시적으로
가정한다. 각 성분이 별도 보존되고 \(b,c\)가 dust, \(\Lambda\)가 상수이면

\[
\dot\rho_b+3H\rho_b=0,
\qquad
\dot\rho_c+3H\rho_c=0,
\qquad
\dot\rho_\Lambda=0.
\]

따라서 \(f_b^{(m)}\)는 상수지만

\[
\frac{d\Omega_b}{d\ln a}
=3\Omega_b w_{\rm eff},
\qquad
w_{\rm eff}:=\frac{p_{\rm tot}}{\rho_{\rm tot}}
\]

이다. dust+\(\Lambda\)에서는 \(w_{\rm eff}=-\Omega_\Lambda\)이므로
\(\Omega_b\)는 dark-energy 시대에 감소한다. 상수인 \(q_*\)와의 equality는
최대 한 hypersurface에서만 성립할 수 있다.

상호작용을 허용하여

\[
\dot\rho_b+3H\rho_b=Q_b,
\qquad
\dot\rho_{\rm tot}+3H(\rho_{\rm tot}+p_{\rm tot})=0
\]

라 하면

\[
\frac{d\Omega_b}{d\ln a}
=\frac{Q_b}{H\rho_{\rm tot}}+3\Omega_bw_{\rm eff}.
\]

따라서 \(\Omega_b=q_*\)를 모든 시각에 강제하려면

\[
\boxed{Q_b=-3H\rho_b w_{\rm eff}}
\]

가 필요하다. \(\Lambda\) 시대에는 baryon sector로의 지속적인 energy
transfer다. 고정 baryon mass와 \(\nabla_\mu J_b^\mu=0\)을 함께 요구하면
\(Q_b=0\)이므로 이 조건과 양립하지 않는다. number current는 보존하면서
질량만 변하게 만들 수도 있지만, 그 경우 새 scalar coupling, fifth-force
및 equivalence-principle 검증이 추가된다.

결론적으로 다음 셋을 동시에 얻을 수 없다.

1. 상수 \(q_*\)와 모든 시각의 \(\Omega_b=q_*\),
2. 비영인 separately conserved dark energy,
3. 고정 질량의 conserved baryon current.

하나를 포기하거나 equality를 독립적으로 선택된 단일 hypersurface의
boundary condition으로 제한해야 한다.

## 7. equipartition의 정확한 역할

열평형의 고전 equipartition은 quadratic degree of freedom마다 평균
에너지가 같다는 조건부 정리다. 이를 현재 문제에 쓰려면 최소한

- 두 label sector가 같은 quadratic Hamiltonian과 같은 온도를 공유하고,
- 화학퍼텐셜·질량·내부 자유도·결합에 의한 차이가 없으며,
- freeze-out 전에 충분히 ergodic하고,
- label assignment가 energy와 독립

이어야 한다. 이 전제들은 곧
\(\mathbb E[W\mid E]=\mathbb E[W\mid E^c]\)를 구현하는 물리 공리들이다.
equipartition이라는 이름만으로 그 equality를 새로 유도하지는 않는다.
특히 baryon과 dark/vacuum sector는 gauge charge, EOS와 보존량이 다르므로
동일한 equilibrium fiber라는 설명이 별도로 필요하다.

## 8. 독립 공리와 target-awareness 회계

아래 회계는 Poisson fixed-point core와 \(D\)의 값이 이미 주어졌다고
가정하고, 그 밖에 필요한 독립 선택만 센다.

| 번호 | 추가 구조 | 역할 | 형식 지위 | target-awareness |
|---|---|---|---|---|
| A1 | extinction tail event를 국소 species label로 바꾸는 state map | \(E\leftrightarrow b\) | `[공리]` | 수치 입력은 없지만 관측된 수치 근접을 본 뒤 이 label을 고르면 hypothesis-level target-aware |
| A2 | label-blind energy Hamiltonian | \(W=I_X\otimes H_W\) | `[공리]` 또는 새 미시 대칭의 `[산출]` | 관측 수치 불사용 |
| A3 | product initial/boundary state 또는 정확한 zero covariance | equal conditional energy | `[공리]` | 관측 수치 불사용 |
| A4 | homogeneous stationary-ergodic many-cell preparation | ensemble fraction을 공간 평균으로 읽기 | `[공리]` | 관측 수치 불사용 |
| A5 | generation clock과 independently selected freeze-out \(\Sigma_*\) | tail event를 공변 초기자료로 만들기 | `[공리]` | \(t_0\)나 관측 epoch를 보고 고르면 target-aware |
| A6 | 두 label sector가 conserved dust이고 equal rest energy | \(f_b^{(m)}=q\)의 보존 | `[공리]`; 그 아래 결과는 `[정리]` | 관측 수치 불사용 |
| A7 | complement가 모든 matter라는 species 해석 | 분모를 \(\rho_b+\rho_c\)로 고정 | `[공리]` | 실제 species를 본 뒤 선택하면 model-selection aware |
| A8 | \(\Omega_m=1\), 또는 tuned interaction/boundary critical closure | \(f_b^{(m)}=q\)를 \(\Omega_b=q\)로 승격 | `[공리]` | 현재 epoch를 고르면 직접 target-aware; dark energy와 tension |

회계 결론은 다음과 같다.

- 순수 확률의 weighted-fraction equality에는 **zero covariance라는 하나의
  필요충분 조건**이 남는다.
- 이를 factorization으로 설명하면 최소 **두 조건**—label-blind dynamics와
  product state—이 필요하다.
- 위에 쓴 구체적인 공변 dust route의 \(f_b^{(m)}=q\)에는 core 밖의
  **A1--A7, 일곱 종류의 논리적으로 구별되는 물리 공리**가 들어간다.
  이것이 모든 가능한 UV completion에 대한 최소성 증명은 아니지만, 현재
  구성에서는 어느 역할도 core 정리에서 나오지 않는다. 일부는 더 깊은
  action에서 산출로 바뀔 수 있지만 현재는 독립이다.
- dark energy와 함께 \(\Omega_b=q\)까지 가려면 **A8이 추가**되며,
  separately conserved dark energy와 baryon current를 모두 유지하는
  지속적 equality route는 위 no-go에 걸린다.

식에 관측 \(\Omega_b\)나 \(H_0\)를 넣지 않았으므로 parameter leakage는
0개다. 그러나 이미 알려진 \(q\)-\(\Omega_b\) 수치 근접을 보고 A1과 A8을
선택했다면, 매개변수가 없어도 **가설 선택 자체는 target-aware**다.
독립 예측으로 승격하려면 이 state map, \(\Sigma_*\), denominator와
interaction law를 다음 자료를 보기 전에 동결해야 한다.

이번 탐색에서 실제로 비교한 후보는 (i) covariance/equal-mean,
(ii) product Hilbert/action, (iii) exchange symmetry/equipartition,
(iv) conserved two-dust action, (v) scalar-fixed-point stress의 다섯 경로다.
따라서 사후에 가장 가까운 경로 하나만 보고하는 것은 허용되지 않으며,
모두 이 artifact에 남긴다.

## 9. 최종 형식 지위

| 명제 | 판정 | 이유 |
|---|---|---|
| \(\Omega_E^{(W)}-q=\operatorname{Cov}(1_E,W)/\mathbb E W\) | `[정리]` | 정의에서 직접 증명 |
| \(\Omega_E^{(W)}=q\)와 equal conditional mean energy의 동치 | `[정리]` | \(0<q<1\)에서 필요충분 |
| product state+label-blind energy이면 equality | `[정리]` | tensor trace factorization |
| iid/ergodic many-cell limit와 finite-\(N\) fluctuation | `[정리]` | strong law, ergodic theorem, CLT 조건 명시 |
| constrained two-dust action의 \(f_b^{(m)}=q\) | `[산출]` | A1--A7 및 boundary constraint 아래 성립 |
| symmetry가 A1--A7을 현재 CE core에서 자동 생성 | `[미완성]` | local state map, product preparation, freeze-out 부재 |
| scalar minimum \(x=q\)가 baryon density를 준다 | **구조적으로 배제** | \(w=-1\), 자유 \(C\)와 \(M\) degeneracy |
| \(f_b^{(m)}=q\Rightarrow\Omega_b=q\) | **거짓** | 정확한 관계는 \(\Omega_b=q\Omega_m\) |
| conserved baryon+separate dark energy와 영구적 \(\Omega_b=q\) | **구조적으로 배제** | \(d\Omega_b/d\ln a=3\Omega_bw_{\rm eff}\) |
| 현재 \(\Omega_b=q\) | `[미완성]` | 독립 \(\Sigma_*\), critical closure와 blind 검증 부재 |
| 새 관측에 대한 \(\Omega_b\) prediction | 승격 불가 | 다섯 후보를 본 target-aware model selection 및 A1--A8 미유도 |

## 10. 다음에 유도할 가장 좁은 목표

긍정 경로를 보존하려면 다음 단계의 명제를 처음부터
\(\Omega_b=q\)로 잡아서는 안 된다. 가장 좁고 닫을 수 있는 목표는

\[
\boxed{
\text{local branching label current}
\;\Longrightarrow\;
\nabla_\mu J_b^\mu=0,
\quad
\mathbb E[W\mid E]=\mathbb E[W\mid E^c],
\quad
f_b^{(m)}=q
}
\]

이다. 이를 관측 독립적인 microscopic action에서 유도한 뒤에만
\(\Omega_m\)의 별도 dark-sector dynamics를 풀어
\(\Omega_b=q\Omega_m\)를 계산해야 한다. 이 순서를 지키면 가능한 좁은
정리는 살리면서, matter-composition fraction과 critical-density fraction을
혼동하지 않는다.
