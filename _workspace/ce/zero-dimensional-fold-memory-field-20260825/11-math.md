# 11-math — 0D 접힘 단일 환경장의 단계별 유도

Status: COMPLETE

## 1. 먼저 “0D가 남는다”의 타입을 분리한다

공간 절편 $\Sigma_t$에서 한 접힘의 support가 점 $\{\mathbf X_a(t)\}$이면 그
접힘은 공간적으로 0차원이다. 그러나 생성 시각 $t_a$ 뒤에도 접힘 자체가
존재하면 spacetime support는

$$
\mathcal W_a=\{(t,\mathbf X_a(t)):t\ge t_a\}
$$

인 worldline이므로 1차원이다. 반대로

$$
y_a=(t_a,\mathbf X_a)
$$

하나만 source로 삼으면 이것은 strict spacetime-0D 사건이다. 이 경우 남는 것은
사건 자체가 아니라 사건이 환경장에 만든 retarded response다. 이후 계산은
“strict 0D deposit + 남아 있는 환경 기억장”을 기본 경로로 삼는다.

## 2. 지속 공간-0D carrier와 하나의 장: 활성 중심 경로

사용자의 문장을 문자 그대로 읽으면 접힘은 한 번 생겼다가 사라지는 사건이 아니라,
각 공간 절편에서 점으로 남아 있는 carrier다. 다만 표준 조건부 양자역학은
비선택 outcome을 이런 carrier로 자동 변환하지 않는다. 그 연결을 가장 약한
형식으로 드러내기 위해, instrument 또는 decoherent coarse-graining이 정한
비선택 history 공간과 subprobability를 $\Gamma_{\rm ns}$,
$\nu_{\rm ns}$라 하자. CE가 별도 물리 사상으로 measurable carrier map
$F_t:\Gamma_{\rm ns}\to\Sigma_t$와 비음 가중치 $w(\gamma)$를 채택하면

$$
\boxed{
\mu_{F,t}(B)=
\int_{\Gamma_{\rm ns}}
w(\gamma)\mathbf1_B(F_t(\gamma))\nu_{\rm ns}(d\gamma)
}
\tag{P0}
$$

라는 weighted pushforward measure를 정의할 수 있다. 식 (P0)은 수학적
정의이지만, $F_t$와 $w$의 선택 및 이를 실제 stress로 읽는 일은 표준 양자역학의
정리가 아니라 **물리 사상 공리**다. 특히 $\mu_{F,t}(\Sigma_t)$는 확률 가중치일
수는 있어도 에너지나 $\Omega$가 아니다. instrument dependence, interference를
억제하는 decoherent history 조건, no-double-counting과 covariance도 별도
검증이 필요하다.

유한 coarse-graining에서 식 (P0)의 atomic image가 $n$개라고 하고 위치와
가중치를 $\mathbf X_j$, $w_j\ge0$라 두면

$$
\boxed{
\mu_F(d^3y)=\sum_{j=1}^n w_j\delta_{\mathbf X_j}(d^3y)
}
\tag{P1}
$$

로 둔다. $\mu_F$는 count를 반환하는 고정된 quenched measure다. 각
$\mathbf X_j$는 공간적으로 0차원이지만, 지속되는 접힘의 시공간 support는
$\{(t,\mathbf X_j):t\ge t_i\}$라는 worldline이다. 이 두 진술은 모순이 아니다.
즉 활성 최소 경로는 식 (P0)의 image와 weight가 comoving frame에서 고정되는
특수화를 택해 아래에서 시간 첨자를 생략한다.

여기서 **하나의 장**은 동역학적 상태장이 하나라는 뜻이다. $\mu_F$의 위치와
가중치는 이 최소 모형에서는 초기 배경 자료이며 운동방정식을 갖지 않는다. 만약
$\mathbf X_j(t)$나 $w_j(t)$를 $\psi$와 독립적으로 진화시키면 $\mu_F(t)$가 두 번째
상태가 되므로 one-state-field 주장은 깨진다.

기억 밀도 $\psi$를 saturation scale로 나눈 무차원 장과 응답함수를

$$
\chi(t,\mathbf x):=\frac{\psi(t,\mathbf x)}{\psi_s},
\qquad
\sigma(\chi)=\frac{\chi}{1+\chi}
\tag{P2}
$$

로 정의한다. 유한 초기시각 $t_i$와 초기 history를 지정하면, persistent carrier가
한 장을 통해 서로를 활성화하는 retarded Volterra 식은

$$
\boxed{
\chi(t,\mathbf x)=b(t,\mathbf x)+
A\int_{t_i}^{t}ds\int_{\Sigma_s}
K^{F}_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma\!\left(\chi(s,\mathbf y)\right)\mu_F(d^3y)
}
\tag{P3}
$$

이다. $A$, $b$, $\chi$, $\sigma$는 무차원이고 $[K^F_{\ell,R}]=T^{-1}$다.
차원 있는 $\psi$로 직접 쓰면 식 (P3)의 적분항 앞에 $\psi_s$가 필요하고
$\sigma$의 인자는 반드시 $\psi/\psi_s$여야 한다.

kernel에는 다음을 요구한다.

1. $K^F_{\ell,R}\ge0$이다.
2. $K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)=0$ unless
   $t>s$ and $d(\mathbf x,\mathbf y)\le c_\psi(t-s)$, where $0<c_\psi\le c$.
3. $\ell>0$에서 locally bounded이고 시간 적분 가능하다.
4. 무한 carrier 집합을 쓰면

$$
\sup_{\mathbf x}
\int_{t_i}^{t}ds\int
K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)\mu_F(d^3y)<\infty
\tag{P4}
$$

   같은 local summability를 추가한다.

$t_i=-\infty$인 stationary preparation은 kernel 적분 가능성과 bounded past
history를 별도로 증명한 뒤에만 허용한다. 식 (P3)에는 event counting process가
없다. 각 fold의 출력 $\sigma(\chi)$는 같은 장의 대수적 readout이며 독립된 두 번째
장이나 확률 상태가 아니다.

### 2.1 strict point와 유한 결합의 연결

식 (P1)은 carrier 위치의 수학적 support다. 물리적 field coupling을 strict
$\delta^{(3)}$로 그대로 계산하면 coincidence와 self-energy가 발산할 수 있으므로,
각 점에 정규화된 form factor $S_{\ell,j}$를 붙여

$$
\mu_{F,\ell}(\mathbf y)
=\sum_jw_jS_{\ell,j}(\mathbf y),
\qquad
\int S_{\ell,j}(\mathbf y)d^3y=1
\tag{P5}
$$

로 coupling을 coarse-grain한다. 유한 행렬 $W^{(\ell)}$는 이 smeared response를
carrier support에서 읽은 값이다. 예를 들어 bounded compact profile $f\ge0$로

$$
W^{(\ell)}_{ij}=\bar w\,
f\!\left(\frac{d(\mathbf X_i,\mathbf X_j)}{\ell}\right),
\qquad W_{ii}=0
\tag{P6}
$$

를 쓸 수 있다. $W_{ii}=0$은 “이웃이 실행한다”는 해석에서 직접 자기고리를
제거하는 선택이지 안정성 정리의 필요조건은 아니다. 유한 $W$를 쓴 사실만으로
미시 UV 문제가 해결되는 것은 아니며, 식 (P5) 또는 별도 renormalization이 그
연결을 담당해야 한다.

### 2.2 exponential memory의 정확한 finite-carrier 환원

$\chi_i(t):=\chi(t,\mathbf X_i)$라 하고

$$
h_\tau(u)=\frac1\tau e^{-u/\tau}\Theta(u),
\qquad \int_0^\infty h_\tau(u)du=1
$$

를 택한다. 고정된 $W\ge0$와 상수 seed $b_i\ge0$에서 식 (P3)의 유한 환원은

$$
\boxed{
\chi_i(t)=b_i+A\sum_jW_{ij}
\int_0^\infty h_\tau(u)\sigma(\chi_j(t-u))du
}
\tag{P7}
$$

다. $q_j=h_\tau*\sigma(\chi_j)$로 두면
$\tau\dot q_j=-q_j+\sigma(\chi_j)$이고
$\chi-b=AWq$이므로, $b$가 상수일 때

$$
\boxed{
\tau\dot{\boldsymbol\chi}
=-(\boldsymbol\chi-\mathbf b)
+AW\sigma(\boldsymbol\chi)
}
\tag{P8}
$$

을 정확히 얻는다. 시간 의존 $b$에는 우변에 $\tau\dot{\mathbf b}$가 추가된다.
식 (P8)은 지정한 exponential Volterra kernel의 exact Markov embedding이지
stochastic mean-field closure가 아니다. bounded Lipschitz $\sigma$, 유한 $W$와
연속 초기 history에서는 해가 전역적으로 유일하고, 비음 초기값은 비음 영역에
남는다. 또한 $0\le\sigma\le1$이므로 각 성분은 유한한 상계로 끌려간다.

거리 $d_{ij}/c_\psi$만큼의 전파 지연을 보존하면

$$
\boxed{
\tau\dot\chi_i(t)=-(\chi_i(t)-b_i)
+A\sum_jW_{ij}
\sigma\!\left(\chi_j(t-d_{ij})\right),
\qquad d_{ij}\ge\frac{d(\mathbf X_i,\mathbf X_j)}c
}
\tag{P9}
$$

라는 retarded delay equation을 얻는다. $d_{ij}=d(\mathbf X_i,\mathbf X_j)/c_\psi$,
$c_\psi\le c$를 택하면 광원뿔 밖 전파를 금지한다.

### 2.3 bootstrap threshold와 양의 안정 branch

$\mathbf b=0$, $\sigma(0)=0$, $\sigma'(0)=1$로 둔다. 지연이 없는 식 (P8)을
0에서 선형화하면

$$
J_0=\frac1\tau(-I+AW).
\tag{P10}
$$

$W\ge0$이면 Perron--Frobenius 고유값 $\rho(W)$가 있으므로 정확한 선형
bootstrap threshold는

$$
\boxed{\mathcal B:=A\rho(W)=1}
\tag{P11}
$$

이다. $\mathcal B<1$이면 0은 선형 안정이고 $\mathcal B>1$이면 Perron mode가
불안정하다. 이것은 offspring 수를 세는 Hawkes reproduction number가 아니라,
고정 carrier network를 통과하는 한 장의 선형 gain이다.

이제 모든 행합이 같은 $w>0$인 경우 $W\mathbf1=w\mathbf1$이고
$\rho(W)=w$다. $\mathcal B=Aw$라 하면 uniform sector의 정확한 식은

$$
\tau\dot\chi=-\chi+
\mathcal B\frac{\chi}{1+\chi}.
\tag{P12}
$$

따라서

$$
\chi_0=0,
\qquad
\boxed{\chi_*=\mathcal B-1\quad(\mathcal B>1)}.
\tag{P13}
$$

양의 branch에서 $\sigma'(\chi_*)=\mathcal B^{-2}$이고 전체 $n$차원
Jacobian spectrum은

$$
\operatorname{spec}J_*
=\left\{
\frac{-1+A\sigma'(\chi_*)\lambda_k(W)}\tau
\right\}_k.
\tag{P14}
$$

$|\lambda_k(W)|\le\rho(W)=w$이므로

$$
\operatorname{Re}\lambda(J_*)
\le-\frac{1-1/\mathcal B}{\tau}<0.
\tag{P15}
$$

즉 constant-row-sum network의 양의 uniform branch는 모든 선형 mode에 대해
국소 점근 안정하다. $W$가 reducible이면 별도 block의 비균일 평형 가능성까지
이 정리가 배제하지는 않는다.

지연식 (P9)의 평형 $\bar\chi$ 주위 characteristic equation은

$$
(1+\tau z)v_i
=A\sigma'(\bar\chi)
\sum_jW_{ij}e^{-zd_{ij}}v_j.
\tag{P16}
$$

$\operatorname{Re}z\ge0$이면 $|1+\tau z|\ge1$,
$|e^{-zd_{ij}}|\le1$이다. 따라서

$$
A\sigma'(\bar\chi)\rho(W)<1
\tag{P17}
$$

이면 우반평면 characteristic root가 없다는 modulus-bound 충분조건을 얻는다.
양의 uniform branch에서는 좌변 gain이 정확히 $1/\mathcal B<1$이므로 임의의
유한 비음 지연에서도 이 충분조건을 통과한다.

### 2.4 이 경로가 해결한 것과 해결하지 않은 것

$\mathbf b=0$이고 초기 history가 정확히 0이면 $\boldsymbol\chi\equiv0$은
초임계에서도 해다. 임계값은 seed를 증폭하는 조건이지 무에서 seed를 만드는
법칙이 아니다. 양의 초기 history, 외부 seed 또는 noise가 별도로 필요하다.

이 경로에서는 접힘 carrier $\mu_F$가 영구히 남아도 activation field $\chi$는
relaxation과 saturation을 가진다. 따라서 “접힘이 남는다”와 “각 activation의
양의 흔적이 영원히 additive하게 누적된다”는 서로 다른 명제다. 앞의 명제는
식 (P1)--(P17)과 양립하고, 뒤의 명제는 아래 §10의 stationary no-go에 걸린다.

식 (P3)은 하나의 동역학장을 가진 dissipative effective model이며 canonical
action이나 stress tensor를 아직 주지 않는다. fixed $\mu_F$도 translation 및
diffeomorphism invariance를 깨는 외부 구조이므로, carrier 형성 부문과 그 stress,
우주론적 통계 균질성·등방성, 비선택 양자경로에서 $\mu_F$로 가는 map은 모두
$[미완성]$이다. 따라서 이 branch만으로 암흑물질 또는 암흑에너지를 동일시할 수
없다.

## 3. 새 접힘이 생성되는 경우: 사건 기록과 하나의 기억장

접힘 사건을 locally finite adapted counting measure로

$$
N(d^4y)=\sum_a\delta_{y_a}(d^4y)
$$

라고 하자. $N(B)$는 영역 $B$ 안의 사건 수다. 예측 가능한 대상은 $N$ 자체가
아니라 뒤에서 정의할 compensator $\lambda(y\mid\mathcal F_{y^-})dV_y$다.
$K_{\ell,R}(x,y)$는 다음을
만족한다고 가정한다.

1. $y\notin J^-(x)$이면 $K_{\ell,R}(x,y)=0$이다.
2. $K_{\ell,R}\ge0$이다.
3. 유한 해상도 $\ell>0$에서 locally bounded이고 적분 가능하다.
4. 모든 source point $y$에 대해

$$
\int_{J^+(y)}K_{\ell,R}(x,y)dV_x=\tau(y)<\infty
$$

이다.

한 접힘이 무차원 크기 $A$를 deposit하면 환경 기억장은

$$
\boxed{
\psi(x)=A\int_{J^-(x)}K_{\ell,R}(x,y)N(d^4y)
}
\tag{1}
$$

이다. counting measure 적분이므로 식 (1)은 단순히 과거 사건들의 retarded
profile을 더한 것이다. 조건부 사건 강도는 strictly prior filtration만 사용해

$$
\boxed{
\lambda(x\mid\mathcal F_{x^-})
=\lambda_0(x)+F(\psi(x^-))
}
\tag{2}
$$

로 정의한다. $x^-$는 한 사건이 자기 자신을 같은 시각에 다시 발화시키는
algebraic loop를 금지한다.

$N$을 두 번째 독립 동역학장으로 두지 않으려면 그 법칙도 닫아야 한다.
$M\times\mathbb R_+$ 위의 primitive Poisson random measure
$\Pi(dV_y,dz)$를 두고 thinning으로

$$
\boxed{
N_\psi(dV_y)=
\int_0^\infty
\mathbf 1_{\{0<z\le\lambda(y\mid\psi_{y^-})\}}
\Pi(dV_y,dz)
}
\tag{2a}
$$

라고 정의한다. $z$는 intensity와 같은 단위를 가진 mark이므로
$dV_y\,dz$는 무차원 count intensity다. 식 (2a)를 식 (1)에 대입하면 유일한
동역학적 상태변수는 $\psi$이고 $N_\psi$는 $\psi$와 primitive noise에서 생성된
event record다. 다만 $\Pi$와 그 에너지 reservoir는 물리적으로 사라지는 것이
아니므로, 이는 one-state-field stochastic effective theory이지 닫힌
one-field Hamiltonian theory가 아니다.

## 4. 사건 경로의 차원 감사

독립 기저를 $(L,T)$로 두면

| 양 | 차원 | 이유 |
|---|---|---|
| $N(B)$, $A$ | $1$ | 사건 수와 event당 deposit |
| $\lambda$, $\lambda_0$ | $L^{-3}T^{-1}$ | 단위 공간·시간당 사건 수 |
| $\psi$, $\psi_s$, $K_{\ell,R}$ | $L^{-3}$ | 환경 기억 수밀도 |
| $\beta$ | $T^{-1}$ | $\beta\psi$가 강도여야 함 |
| $\tau$ | $T$ | kernel의 spacetime 적분 |
| $\ell$ | $L$ | 공간 smearing scale |

따라서

$$
\boxed{\mathcal R(y)=A\beta
\int_{J^+(y)}K_{\ell,R}(x,y)dV_x}
\tag{3}
$$

는 무차원이다. $\tau(y)=\tau$가 모든 $y$에서 같은 homogeneous limit에서만

$$
\boxed{\mathcal R=A\beta\tau}
\tag{4}
$$

라는 단일 상수를 쓸 수 있다. 팽창하거나 비정상적인 배경에서 식 (4)를 그대로
쓰는 것은 추가 균질성 가정이다.

## 5. 선형 자기흥분장의 정확한 평균 방정식

선형 경로를

$$
F(\psi)=\beta\psi
\tag{5}
$$

로 둔다. 조건부 compensator 정의에서

$$
\mathbb E[N(d^4y)]
=m(y)dV_y,\qquad
m(y):=\mathbb E[\lambda(y)]
$$

이므로 식 (1)의 기대값은

$$
\mathbb E[\psi(x)]
=A\int_{J^-(x)}K_{\ell,R}(x,y)m(y)dV_y
$$

이다. 식 (2)와 (5)에 대입하면

$$
\boxed{
m(x)=\lambda_0(x)+A\beta
\int_{J^-(x)}K_{\ell,R}(x,y)m(y)dV_y
}
\tag{6}
$$

를 얻는다. 식 (6)은 선형 모형에서는 mean-field 근사가 아니라 정확한
Volterra 평균 방정식이다.

엄밀한 light-cone support를 만드는 한 후보는 retarded Green 함수와 smeared
source를

$$
K_{\ell,R}(x,y)=
\int_M G_R(x,z)S_\ell(z,y)dV_z
$$

처럼 합성하는 것이다. 이때 정확히

$$
\operatorname{supp}S_\ell(\cdot,y)\subseteq J^+(y),
\qquad
\operatorname{supp}G_R(\cdot,z)\subseteq J^+(z)
$$

를 요구한다. 인과관계의 추이성 때문에 이 합성의 support는 $J^+(y)$ 안에
있다. 동일 시각의 공간 Gaussian smear는 이 조건을 만족하지 않으므로 causal
kernel 구성에 사용하지 않는다.

그러나 generic wave Green 함수는 부호가 바뀔 수 있으므로 위 합성만으로
$K_{\ell,R}\ge0$가 따라오지는 않는다. 따라서 positivity는 식 (2)의 intensity를
정의하기 위한 독립 kernel 공리로 유지한다. 실제 후보는 positivity-preserving
causal response/resolvent를 사용하거나, 합성된 raw kernel이
$K_{\ell,R}\ge0$임을 별도로 증명해야 한다. signed raw Green 함수를 intensity에
직접 넣는 경로는 허용하지 않는다. 일반적으로 허용된 kernel은 시간함수와
공간함수의 단순 곱으로 분리되지 않는다.

국소 Minkowski 또는 comoving rest frame에서 positivity와 causality를 동시에
만족하는 명시적 존재 witness도 만들 수 있다. $0<c_\psi\le c$,
$t_0=\ell/c_\psi>0$, $r=|\mathbf x-\mathbf y|$로 두고

$$
\boxed{
K_{\ell,R}^{\rm w}(t,r)=
\Theta(t-t_0)e^{-(t-t_0)/\tau}
\frac{3\,\Theta(c_\psi t-r)}
{4\pi(c_\psi t)^3}
}
\tag{6a}
$$

라고 하자. 이 함수는 비음이고, $t<t_0$ 또는 $r>c_\psi t$에서 0이며,
$t\downarrow t_0$에서의 우극한과 essential supremum이
$3/(4\pi\ell^3)$로 유한하다. $\Theta(0)$의 점값 convention은 이 적분 및
support 결론을 바꾸지 않는다. 고정된 $t>t_0$에서

$$
\int_{\mathbb R^3}
\frac{3\,\Theta(c_\psi t-r)}
{4\pi(c_\psi t)^3}d^3r=1
$$

이므로

$$
\int_0^\infty dt\int d^3r\,
K_{\ell,R}^{\rm w}(t,r)
=\int_{t_0}^\infty e^{-(t-t_0)/\tau}dt
=\tau.
\tag{6b}
$$

따라서 식 (6a)는 이 run의 kernel 공리들이 공집합이 아님을 보이는 정확한
구성이다. preferred rest frame을 사용하므로 일반 곡률 시공간의 미시적
covariant kernel을 유도한 것은 아니다.

homogeneous limit에서는 전체 공간을 적분한 response

$$
h(u)=\int_{\Sigma_{t_y+u}}K_{\ell,R}(x,y)d^3x,
\qquad
h(u)=0\quad(u\le0),
\qquad
\int_0^\infty h(u)du=\tau
$$

만 사용한다. 그러면 homogeneous mean은

$$
m(t)=\lambda_0(t)+A\beta
\int_0^\infty h(u)m(t-u)du.
\tag{7}
$$

정상상태 $\bar m$에서는

$$
\bar m=\bar\lambda_0+\mathcal R\bar m
$$

이므로

$$
\boxed{
\bar m=\frac{\bar\lambda_0}{1-\mathcal R},
\qquad
\bar\psi=A\tau\bar m
}
\tag{8}
$$

이다. 비음의 $\bar\lambda_0>0$에서 유한한 비음의 해가 있으려면
$\mathcal R<1$이어야 한다. $\mathcal R=1$이면 식 (8)의 원래 방정식이
$\bar m=\bar\lambda_0+\bar m$가 되어 모순이고, $\mathcal R>1$에서 나오는
음의 형식해는 intensity가 될 수 없다.

Poisson offspring cluster라는 추가 가정까지 둘 때, 한 사건의 평균 자손수가
$\mathcal R$이고 소멸확률은

$$
q=\exp[\mathcal R(q-1)]
\tag{9}
$$

의 최소해다.

- $\mathcal R<1$: $q=1$, 평균 총 cluster 크기는 $(1-\mathcal R)^{-1}$이다.
- $\mathcal R=1$: $q=1$이지만 평균 총 cluster 크기는 발산한다.
- $\mathcal R>1$: $q<1$이고 생존확률은 양수지만 유한 stationary mean은 없다.

따라서 선형 경로는 안정한 기억장과 무한 자기실행을 동시에 주지 않는다.
subcritical이면 정상상태는 가능하지만 유한 seed는 소멸하고, supercritical이면
생존 가능성이 생기지만 살아남은 경로의 선형 평균은 폭주한다.

## 6. Fourier--Laplace pole 계산

식 (6)을 정상 배경 주위에서 선형화하고

$$
\widetilde K(s,\mathbf k)
=\int_0^\infty dt\int d^3r\,
e^{-st-i\mathbf k\cdot\mathbf r}K(t,\mathbf r)
$$

를 정의한다. $[\widetilde K]=T$다. 자유 mode는

$$
\left[1-A\beta_{\rm eff}\widetilde K(s,\mathbf k)\right]
\delta\widetilde m(s,\mathbf k)=0
$$

를 만족하므로 pole 조건은

$$
\boxed{
1-A\beta_{\rm eff}\widetilde K(s,\mathbf k)=0
}
\tag{10}
$$

이다. $K\ge0$이고 $\operatorname{Re}s\ge0$이면

$$
\left|A\beta_{\rm eff}\widetilde K(s,\mathbf k)\right|
\le A\beta_{\rm eff}\tau.
$$

따라서 $A\beta_{\rm eff}\tau<1$은 우반평면 pole이 없다는 충분조건이다.

공간을 적분한 homogeneous response가 $h(t)=e^{-t/\tau}$이면

$$
\widetilde K(s,\mathbf 0)
=\frac1{s+\tau^{-1}}
$$

이므로

$$
\boxed{
s(\mathbf 0)=A\beta_{\rm eff}-\tau^{-1}
}
\tag{11}
$$

이다. 따라서 선형 진공 mode는

$$
s(0)=\frac{\mathcal R-1}{\tau}
$$

이므로 $\mathcal R=1$이 정확한 homogeneous bifurcation 경계다.

## 7. 포화된 기억장의 homogeneous closure

선형 폭주를 막기 위한 최소 포화 law를

$$
\lambda(\psi)=\lambda_0+
\frac{\beta\psi}{1+\psi/\psi_s}
\tag{12}
$$

로 둔다. exponential memory를 가진 deterministic continuum closure에서는

먼저 finite coarse-graining cell $V$에서 exact pathwise jump law를 분리한다.
$M_t=V\psi_t$, $M_s=V\psi_s$, $\Lambda_0=V\lambda_0$라 두면

$$
\boxed{
dM_t=-\frac{M_{t^-}}{\tau}dt+A\,dN_t^\psi,
\qquad
\Lambda_t=\Lambda_0+
\frac{\beta M_{t^-}}{1+M_{t^-}/M_s}
}
\tag{12a}
$$

가 stochastic 방정식이다. 정확한 평균식은

$$
\frac{d}{dt}\mathbb E[M_t]
=-\frac{\mathbb E[M_t]}{\tau}
+A\mathbb E[\Lambda(M_t)]
$$

이며 nonlinear expectation은 닫히지 않는다.

반면 $\mathbb E[\Lambda(M)]\approx\Lambda(\mathbb E[M])$를 채택하거나
사건률을 연속 source로 coarse-grain한 deterministic continuum closure에서는

$$
\boxed{
\dot\psi=-\frac{\psi}{\tau}
+A\lambda_0+
\frac{A\beta\psi}{1+\psi/\psi_s}
}
\tag{13}
$$

가 된다. 식 (13)은 stochastic 식 (12a)의 정리가 아니라 명시적인
deterministic mean-field 공리다.

무차원 변수

$$
x=\frac{\psi}{\psi_s},\qquad
u=\frac{t}{\tau},\qquad
\eta=\frac{A\lambda_0\tau}{\psi_s},\qquad
\mathcal R=A\beta\tau
$$

를 쓰면

$$
\boxed{
\frac{dx}{du}=-x+\eta+\frac{\mathcal R x}{1+x}
}
\tag{14}
$$

이다. 모든 항과 분모의 인자는 무차원이다.

### 7.1 baseline이 없는 경우

$\eta=0$에서

$$
0=x\left[-1+\frac{\mathcal R}{1+x}\right]
$$

이므로 고정점은

$$
x_0=0,\qquad
x_*=\mathcal R-1.
\tag{15}
$$

$x_*>0$는 $\mathcal R>1$일 때만 존재한다. 원래 변수에서는

$$
\boxed{\psi_*=\psi_s(\mathcal R-1)}.
\tag{16}
$$

deterministic closure 식 (13)의 정확한 Jacobian은

$$
J(\psi)=-\frac1\tau+
\frac{A\beta}{(1+\psi/\psi_s)^2}.
\tag{17}
$$

따라서

$$
J(0)=\frac{\mathcal R-1}{\tau},
$$

$$
\boxed{
J(\psi_*)=-\frac{\mathcal R-1}{\mathcal R\tau}<0
}
\tag{18}
$$

이다. $\mathcal R>1$에서 진공은 불안정하고 양의 고정점은 국소 점근
안정하다. $\mathcal R=1$에서는 $J(0)=0$이지만

$$
\frac{dx}{du}=-\frac{x^2}{1+x}<0\qquad(x>0)
$$

이므로 0은 비지수적으로 끌어당긴다.

양의 고정점에서의 미분 gain은

$$
\beta_{\rm eff}
=\left.\frac{d\lambda}{d\psi}\right|_{\psi_*}
=\frac{\beta}{\mathcal R^2}
$$

이고 effective reproduction은

$$
\boxed{
\mathcal R_{\rm eff}
=A\beta_{\rm eff}\tau
=\frac1{\mathcal R}<1
}
\tag{19}
$$

이다. 포화가 supercritical 진공을 subcritical 양의 배경으로 이동시킨다.

### 7.2 양의 baseline

$\eta>0$이면 고정점 방정식은

$$
x^2-(\mathcal R+\eta-1)x-\eta=0
$$

이고, 두 근의 곱이 $-\eta<0$이므로 양의 근은 정확히 하나다.

$$
\boxed{
x_+=\frac{\mathcal R+\eta-1+
\sqrt{(\mathcal R+\eta-1)^2+4\eta}}{2}
}
\tag{20}
$$

$f(x)=-x+\eta+\mathcal R x/(1+x)$는 $f(0)=\eta>0$이고
$f''(x)=-2\mathcal R/(1+x)^3<0$이므로 유일한 양의 crossing에서
$f'(x_+)<0$이다. 따라서 이 고정점은 국소 안정하다.

## 8. 평균장 고정점과 stochastic 생존은 다르다

유한 부피 $V$에서 총 기억을 $M=V\psi$, 포화 scale을
$M_s=V\psi_s$로 두자. baseline이 없을 때 총 사건률은

$$
\Lambda(M)=\frac{\beta M}{1+M/M_s}.
$$

새 사건이 하나도 없다면 $M(t)=M_0e^{-t/\tau}$이고, 앞으로의 누적 hazard는

$$
H_0=\int_0^\infty\Lambda(M_0e^{-t/\tau})dt
=\beta\tau M_s\log\left(1+\frac{M_0}{M_s}\right).
\tag{21}
$$

따라서

$$
\boxed{
\mathbb P(\text{no later event}\mid M_0)
=e^{-H_0}>0
}
\tag{22}
$$

이다. $\mathcal R>1$이고 식 (16)의 안정한 평균장 고정점이 있어도, 유한 seed의
개별 stochastic realization에는 즉시 소멸하는 양의 확률이 있다. 유한 상태
logistic birth--death chain은 deterministic positive fixed point를 가지면서도
0이 absorbing이면 결국 0에 흡수될 수 있는 더 강한 반례다.

그러므로 “안정한 양의 평균장”과 “거의 확실한 영구 자기실행”을 같은 명제로
쓸 수 없다. 후자는 지정된 stochastic law의 별도 생존 정리가 필요하다.

## 9. stochastic 존재·유일성의 범위

포화 함수

$$
F(\psi)=\frac{\beta\psi}{1+\psi/\psi_s}
$$

는 $\psi\ge0$에서

$$
0\le F(\psi)\le\beta\psi_s,
\qquad
0\le F'(\psi)=
\frac{\beta}{(1+\psi/\psi_s)^2}\le\beta
$$

를 만족한다. 따라서 유한 부피와 bounded baseline에서는 총 intensity가
bounded이고, 식 (2a)를 그 상계율의 Poisson measure에서 thinning하여
비폭발적인 경로를 시간순으로 유일하게 구성할 수 있다.

무한 시공간의 stationary version에는 더 강한 조건이 필요하다. nonlinear
Hawkes contraction 정리의 한 충분조건은

$$
A L_F\sup_y
\int_{J^+(y)}K_{\ell,R}(x,y)dV_x<1,
\qquad L_F=\beta,
$$

즉 homogeneous 경우 $\mathcal R<1$이다. 이 조건은 존재의 필요조건이 아니지만,
$\mathcal R>1$인 deterministic 양의 branch의 nontrivial stochastic stationary
measure를 보장하지도 않는다. $\lambda_0=0$에서는 영과정이 항상 stationary다.
따라서 stochastic global stationarity와 survival probability는
$[미완성]$으로 남고, 식 (16)--(20)은 deterministic closure의 정리로만 쓴다.

## 10. 영구 흔적의 additive no-go

각 deposit의 영향이 영원히 감쇠하지 않아

$$
\int_0^\infty h(t)dt=\infty
$$

이면 식 (3)의 $\mathcal R$도 양의 $A\beta$에서 무한대다. 선형 정상상태는
존재하지 않는다. 포화 intensity를 쓰더라도 relaxation을 완전히 제거하면

$$
\dot\psi=A\lambda(\psi)\ge0
$$

이고, 양의 seed에서 $\lambda(\psi)>0$이면 finite fixed point가 없다. 큰
$\psi$에서 식 (12)는 상수 $\beta\psi_s$로 포화하므로 $\psi$는 결국 선형으로
증가한다.

따라서 “모든 흔적이 additive하게 영원히 남고 계속 양의 발화를 만든다”와
“유한한 stationary one-field 상태”는 양립하지 않는다. 최소한 다음 중 하나가
필요하다.

1. 유한 기억시간 또는 relaxation,
2. fold annihilation,
3. resource depletion 또는 음의 feedback,
4. 누적 count가 아니라 bounded order parameter로의 비선형 사상.

이 네 항목은 편의적 선택이 아니라 유한 정상상태의 필요 구조다.

## 11. strict point limit의 UV 반례

공간 regulator로 정규화 Gaussian

$$
g_\ell(\mathbf r)
=\frac{e^{-r^2/(2\ell^2)}}{(2\pi\ell^2)^{3/2}}
$$

을 예로 들면

$$
g_\ell(0)\propto\ell^{-3},
$$

$$
\int g_\ell^2d^3x
=\frac1{8\pi^{3/2}\ell^3},
$$

$$
\int|\nabla g_\ell|^2d^3x
=\frac3{16\pi^{3/2}\ell^5}.
\tag{23}
$$

따라서 $\ell\to0$에서 coincidence field, quadratic local intensity와
gradient-energy형 양이 발산한다. Gaussian은 UV scaling을 보이기 위한
regulator이며 strict relativistic causality에는 compact spatial smear 또는
causal propagator가 따로 필요하다.

같은 문제는 canonical massive scalar의 정적 점원천에서 직접 보인다.

$$
(-\nabla^2+m^2)\phi=q\delta^{(3)}(\mathbf x),
\qquad
\phi(r)=\frac{qe^{-mr}}{4\pi r}.
$$

$r\to0$에서 $|\nabla\phi|^2\sim q^2/(16\pi^2r^4)$이므로 cutoff $a$ 바깥의
에너지는

$$
E_{r>a}\sim
\frac12\int_a4\pi r^2
\frac{q^2}{16\pi^2r^4}dr
=\frac{q^2}{8\pi a}
\tag{24}
$$

로 발산한다. 질량 $m$은 이 UV 발산을 제거하지 않는다. strict 0D는 source
distribution으로 사용할 수 있지만, self-coupling과 stress tensor를 계산하려면
$\ell>0$ 또는 명시적 renormalization이 필수다.

## 12. canonical real scalar lump의 Derrick 반례

닫힌 실수 스칼라 하나를

$$
S[\phi]=\int d^4x
\left[-\frac12\partial_\mu\phi\partial^\mu\phi-V(\phi)\right]
$$

로 두고 진공에서 뺀 유한 정적 에너지를

$$
E=T+U,
$$

$$
T=\frac12\int_{\mathbb R^3}|\nabla\varphi|^2d^3x,\qquad
U=\int_{\mathbb R^3}V(\phi_v+\varphi)d^3x
$$

라 하자. $\varphi_\lambda(\mathbf x)=\varphi(\lambda\mathbf x)$이면

$$
E(\lambda)=\lambda^{-1}T+\lambda^{-3}U.
\tag{25}
$$

정적 해의 scale variation은

$$
E'(1)=-T-3U=0
$$

을 요구한다. 따라서 $U=-T/3$이고

$$
\boxed{
E''(1)=2T+12U=-2T<0
}
\tag{26}
$$

이다. $T>0$인 비자명 정적 lump는 scale 방향에서 불안정하다. 특히
$V\ge0$이면 첫 variation 자체가 $T=U=0$인 진공만 허용한다.

따라서 “한 canonical real scalar의 안정한 정적 공간-0D fold”는 제거된다.
complex charged Q-ball, time-dependent oscillon, higher-derivative, gauge,
nonlocal 또는 open-system 경로는 식 (25)의 전제를 바꾸므로 별도 후보지만,
현재 모형에서 자동으로 유도되지 않는다.

## 13. source와 에너지 보존

source를 포함한 작용을

$$
S[\phi;J]=\int d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2-V(\phi)+J\phi\right]
$$

로 두면

$$
\Box_g\phi-V'(\phi)=-J
\tag{27}
$$

이고

$$
T_{\mu\nu}^{(\phi)}
=\nabla_\mu\phi\nabla_\nu\phi
-g_{\mu\nu}\left[\frac12(\nabla\phi)^2+V(\phi)\right].
$$

운동방정식을 사용하면

$$
\boxed{
\nabla_\mu T^\mu{}_\nu{}^{(\phi)}
=-J\nabla_\nu\phi
}
\tag{28}
$$

이다. stochastic deposit $J$가 장에 일을 하면 field stress만은 보존되지
않는다. 정확한 닫힘은

$$
\boxed{
\nabla_\mu
\left(T^{\mu\nu}_\phi+
T^{\mu\nu}_{\rm source/reservoir}\right)=0
}
\tag{29}
$$

을 요구한다.

$J=f(\phi)$인 보존적 국소 함수로 바꾸면
$V_{\rm eff}=V-\int f(\phi)d\phi$로 흡수할 수 있지만, 이때 독립적인 stochastic
deposit 법칙은 사라지고 식 (26)의 반례가 다시 적용된다. 따라서 R1/R2는
하나의 propagating memory field를 쓰지만 event/source와 reservoir를 가진
open effective theory다. “하나의 장”은 “아무 source degree도 없는 닫힌
한 자유도”와 같지 않다.

## 14. 수치 검산점

$\tau=2.5T$를 공통 검산 scale로 택한다.

먼저 persistent-carrier 경로에서는 8개 fold를 고리로 놓고 양쪽 이웃에 각각
$W_{i,i\pm1}=1/2$, 그 밖에는 0을 둔다. 그러면 모든 행합과 $\rho(W)$가 1이다.
$A=1.2$이면 $\mathcal B=1.2$, $\chi_*=0.2$이고 $W$의 고유값은
$\cos(2\pi k/8)$다. 식 (P14)의 8개 Jacobian 고유값은

$$
(-0.0666667,-0.164298,-0.4,-0.635702,
-0.733333,-0.635702,-0.4,-0.164298)T^{-1}
$$

로 모두 음수다. 지연 안정성 gain도 $1/\mathcal B=0.833333<1$이다. 반대로
$A=0.8$이면 0의 Perron mode가 $-0.08T^{-1}$이고, seed가 정확히 0이면
$A=1.2$에서도 우변은 정확히 0이다.

다음 표는 $A=1$, $\psi_s=1L^{-3}$인 새 fold 생성 보조 event 경로의
검산이다.

| $\mathcal R$ | $\psi_*$ | $J(0)$ 또는 $J(\psi_*)$ |
|---:|---:|---:|
| $0.8$ | 없음 | $-0.08T^{-1}$ |
| $1.0$ | $0$ | $0$ |
| $1.2$ | $0.2L^{-3}$ | $-0.0666666666667T^{-1}$ |

$\lambda_0=0.1L^{-3}T^{-1}$이면 $\eta=0.25$이고

$$
\psi_+=0.7732928049865L^{-3},
$$

$$
J(\psi_+)=-0.247355790739T^{-1}.
$$

## 15. 수학 판정

사용자의 “공간적으로 0차원인 접힘들이 환경에 남아 서로를 실행한다”를 문자
그대로 표현하는 활성 최소식은, 먼저 별도 물리 사상
$\nu_{\rm ns}\xrightarrow{(F,w)}\mu_F$를 채택한 뒤

$$
\boxed{
\begin{aligned}
\mu_F(d^3y)&=\sum_jw_j\delta_{\mathbf X_j}(d^3y),\\
\chi(t,\mathbf x)&=b(t,\mathbf x)+A
\int_{t_i}^{t}ds\int K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma(\chi(s,\mathbf y))\mu_F(d^3y),\\
\sigma(\chi)&=\frac{\chi}{1+\chi},
\qquad \chi=\psi/\psi_s
\end{aligned}
}
\tag{30P}
$$

이다. $\mu_F$는 지속하지만 고정된 공간-0D carrier이고 $\chi$만 동역학적
상태다. 유한 exponential-memory network에서는 그 식이 정확히 (P8)로 환원되고,
constant row sum에서는

$$
\mathcal B=A\rho(W),\qquad
\chi_*=\mathcal B-1\quad(\mathcal B>1),
\qquad
A\sigma'(\chi_*)\rho(W)=\frac1{\mathcal B}<1
$$

이다. $\mathcal B=1$은 0 상태의 bootstrap threshold이고 마지막 부등식은
포화된 양의 branch의 지연-강건한 선형 안정성 충분조건이다. seed는 별도로
필요하다.

반대로 “실행”이 기존 carrier의 재활성화가 아니라 **새 접힘 사건의 생성**을
뜻한다면 살아남는 보조 확률식은

$$
\boxed{
\psi(x)=A\int_{J^-(x)}
K_{\ell,R}(x,y)N_\psi(d^4y),\qquad
N_\psi=N[\psi,\Pi],\qquad
\lambda(x\mid\mathcal F_{x^-})
=\lambda_0(x)+
\frac{\beta\psi(x^-)}{1+\psi(x^-)/\psi_s}
}
\tag{30}
$$

이다. 단, $\ell>0$, exact causal support, finite memory 또는 동등한 negative
feedback, predictability, primitive noise와 source/reservoir stress를 함께
선언해야 한다.

따라서 식 (30P)은 persistent-carrier one-state-field deterministic/Volterra
경로이고, 식 (30)은 event-creation one-state-field open stochastic 경로다.
두 식을 같은 모형으로 섞으면 안 된다. 어느 쪽도 primitive reservoir까지 제거한
autonomous Hamiltonian 한 장은 아니며, quantum gravity, 암흑물질·암흑에너지,
절대 abundance는 아직 나오지 않는다.
