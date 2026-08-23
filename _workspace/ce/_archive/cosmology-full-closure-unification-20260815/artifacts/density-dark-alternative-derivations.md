# U2/U3 밀도·암흑부문 대안 유도

Status: COMPLETE

작성 기준일: 2026-08-15  
범위: U2(확률에서 바리온 abundance), U3(암흑물질·암흑에너지 분할)의
독립 route 연구. 이 문서는 수학·경로 lane의 보조 artifact이며 formal gate
closure를 선언하지 않는다.

## 0. 결론과 판정표

직접적인 `정지 scalar 값 = critical-density fraction` 경로는 사용하지 않았다.
대신 $q_{\rm ext}$가 **조성, 보존전하의 분배, 또는 반응 종점**을 고정하고,
별도 동역학이 총수율과 암흑부문 비율을 고정하는 다섯 경로를 구성했다.

다섯 경로 중 하나만으로 $\Omega_b$, $\Omega_c$, $\Omega_{\rm DE}$를 모두
절대적으로 정하는 경로는 아직 없다. 그러나 목표를 낮출 필요도 없다. 가장
짧은 완결 후보는 다음 계층형 결합이다.

$$
q_{\rm ext}
\xrightarrow{\text{반응 자유에너지}}
y_X
\xrightarrow{\text{비대칭/열적 동결}}
(Y_B,Y_\chi)
\xrightarrow{\text{dark scaling}}
\frac{\rho_c}{\rho_{\rm DE}}
\xrightarrow{\Sigma_*}
(\Omega_b,\Omega_c,\Omega_{\rm DE}).
$$

여기서 $\Sigma_*$는 관측 시각을 넣어 고르는 면이 아니라 action의 상전이,
반응률 또는 장의 값으로 정해지는 공변 hypersurface여야 한다. 이 마지막
조건이 특히 중요하다. 보존 바리온이 존재하는 가속 우주에서는 세
$\Omega_i$가 모두 상수인 late-time fixed point가 될 수 없기 때문이다.

| route | 구조 | $q_{\rm ext}$가 고정하는 것 | 절대 $\Omega_b$ | $\Omega_c/\Omega_{\rm DE}$ | 독립 산출 | 현재 판정 |
|---|---|---|---|---|---|---|
| A | 반응성 two-current + 자유에너지 | 동결 전구체의 number/charge composition | total yield가 미시적으로 계산되면 가능 | 없음 | 반응 relaxation, entropy/heat, 질량비 | **가장 직접적인 U2 후보** |
| E | extinction-conditioned branching tree | matter 내부의 descendant energy composition $Dq$ | 독립 $\Omega_m$ dynamics가 있으면 가능 | 없음 | conditioned multiplicity와 node statistics | **가장 강한 확률론적 U2 후보** |
| B | 보존전하 + asymmetric freeze-out | 총전하 중 visible/dark 분배 | 총 전하-엔트로피비가 계산되면 가능 | 질량과 전하 분배로 $\Omega_c/\Omega_b$만 | dark mass, isocurvature, annihilation 흔적 | **A의 보존전하 completion 후보** |
| C | topological/rolling chemical potential | 위상 전이 cascade의 종점 비율 또는 부호 | $\dot\theta/T$ attractor가 있으면 가능 | axion remnant를 넣어도 DE offset은 별도 | parity-odd 신호, anomaly, isocurvature | **총 비대칭 source 후보** |
| D | interacting dark fluid / conformal scalar | 직접 고정하지 않음; $D$가 coupling을 고정한다는 별도 lemma 필요 | 바리온을 만들지 못함 | 안정 scaling point로 가능 | $w_{\rm eff}$, 성장률, CDM 질량변화 | **가장 직접적인 U3 후보** |
| A/B/C + D | 계층형 결합 | 조성 + 총수율 + dark ratio | 조건 충족 시 가능 | 조건 충족 시 가능 | 모든 중간량에 교차관측 존재 | **full closure 재개 경로** |

`가능`은 관측 중심값을 coupling에 넣어 맞춘다는 뜻이 아니다. 아래에 적은
missing lemma가 동일 action과 초기 quantum state에서 독립적으로 유도되는
경우에만 가능하다는 뜻이다.

## 1. 공통 정의와 차원 원장

자연단위 $c=\hbar=k_B=1$, metric 부호 $(-+++)$를 쓴다. 질량차원은
$[d^4x]=-4$이고 action density는 4다. 이전 run에서 검증한

$$
v_D(y)=y\log y-y+D\left(y-\frac{y^2}{2}\right)+C,
$$

$$
v_D'(y)=\log y+D(1-y),qquad
v_D''(y)=\frac1y-D
$$

를 사용한다. $D>1$의 작은 근은

$$
q_{\rm ext}=\exp[-D(1-q_{\rm ext})],qquad
v_D''(q_{\rm ext})>0
$$

이다. 대표 입력 $D=3.1777584234099736$에서

$$
q_{\rm ext}=0.04864671964402821,qquad
v_D''(q_{\rm ext})=17.3786122284549.
$$

| 양 | 질량차원 | 비고 |
|---|---:|---|
| $q_{\rm ext},D,y,\theta,\lambda,\beta,\xi$ | 0 | log, exp, fixed-point core에 직접 허용 |
| $\phi,f_a,M_{\rm Pl},T,H,\kappa,\mu_*$ | 1 | $\theta=a/f_a$는 무차원 |
| $n_i,s,N_i^\mu,S^\mu$ | 3 | number/entropy current |
| $\Gamma_i=\nabla_\mu N_i^\mu$ | 4 | number 생성률 |
| $\rho,p,V,T^{\mu\nu}$ | 4 | energy density와 stress |
| $Q$ in $\dot\rho_i+3H(\rho_i+p_i)=Q_i$ | 5 | energy-transfer rate |
| $\langle\sigma v\rangle$ | $-2$ | Boltzmann collision term |
| $\partial_\mu\theta J_Q^\mu$ | 4 | 별도 질량 scale 없이 허용 |
| $g^2F\widetilde F$ | 4 | $\theta g^2F\widetilde F$도 차원 4 |

모든 exp/log 인자는 무차원이다. 아래 모델의 차원 정합성은 물리적 정당성이나
자연성을 대신하지 않는다.

## 2. Route A — 반응성 two-current/free-energy attractor

### 2.1 변분 원리와 물질 변수

반응하는 동안의 두 성분을 실제 저에너지 proton과 CDM으로 바로 부르지 않고,
동일한 공통 에너지를 가진 전구체 tag $X_b,X_c$로 둔다. 그래야 rest-mass
차이가 chemical affinity를 이동시켜 $q_{\rm ext}$를 깨뜨리는 문제를 피할 수
있다. 공통 four-velocity를 가진 current를

$$
N_b^\mu=n_bu^\mu,qquad N_c^\mu=n_cu^\mu,qquad
n=n_b+n_c,qquad y=\frac{n_b}{n}
$$

로 둔다. 이상 유체 부분은 Brown형 convective constraint를 포함한

$$
S_A=\int d^4x\sqrt{-g}
\left[\frac{M_{\rm Pl}^2}{2}R-\rho(n_b,n_c,s)\right]
+S_{\rm conv}
$$

로 만들 수 있다. 반응 구간의 master energy density 후보는

$$
\rho=m_Xn+\rho_{\rm th}(n,s)+n\mu_*v_D(y),
\qquad \mu_*>0
$$

다. 모든 항의 차원은 4다. ordinary single-copy ideal-fluid action만으로
비가역 반응은 나오지 않으므로, 반응은 covariant Onsager closure로 붙인다.
이는 action이라고 위장하지 않은 명시적 effective-fluid 원리다.

### 2.2 EOM, current와 entropy

화학 affinity는 label-blind thermal term에서

$$
\mathcal A:=\mu_b-\mu_c
=\left.\frac{\partial\rho}{\partial n_b}\right|_{n_c,s}
-\left.\frac{\partial\rho}{\partial n_c}\right|_{n_b,s}
=\mu_*v_D'(y)
$$

가 된다. total current를 보존하는 반응과 양의 mobility를

$$
\nabla_\mu N_b^\mu=\Gamma,qquad
\nabla_\mu N_c^\mu=-\Gamma,
$$

$$
\Gamma=-\frac{\kappa n}{\mu_*}\mathcal A
=-\kappa n v_D'(y),qquad \kappa>0
$$

로 두면, FLRW에서

$$
\dot n+3Hn=0,qquad
\dot y=-\kappa v_D'(y).
$$

따라서 $q_{\rm ext}$ 근처의 선형 mode는

$$
\delta\dot y=-\kappa v_D''(q_{\rm ext})\delta y
$$

이고 작은 근은 attractor다. 열을 포함한 Gibbs 관계를 사용하면 total stress
보존은

$$
T\nabla_\mu S^\mu=-\mathcal A\Gamma
=\kappa n\mu_*[v_D'(y)]^2\ge0
$$

와 정확히 양립한다. 반응으로 줄어든 free energy는 entropy/heat sector로
간다. 이를 생략하면 $T^{\mu\nu}$ 보존이 깨진다.

metric variation의 stress는

$$
T^{\mu\nu}=(\rho+p)u^\mu u^\nu+pg^{\mu\nu},
$$

$$
p=n_b\mu_b+n_c\mu_c+sT-\rho.
$$

$m_Xn+n\mu_*v_D(y)$는 고정 $y$에서 $n$에 선형이므로 그 냉각된 부분은
$p=0$이다. 즉 이전의 정지 scalar가 가진 $w=-1$ 문제가 이 route에는 없다.

### 2.3 동결, 정규화와 실제 baryon/CDM

반응 동결면은 관측 시각이 아니라

$$
\kappa(T_f)v_D''(q_{\rm ext})\simeq H(T_f)
$$

로 정의한다. 충분히 긴 반응 구간이면 $y_f\simeq q_{\rm ext}$다. 이후

$$
X_b\rightarrow \nu_b B+\cdots,qquad
X_c\rightarrow \nu_\chi\chi+\cdots
$$

가 각 tag 또는 그 보존전하를 전달한다고 하자. label-blind annihilation 또는
freeze-in이 total yield $Y_X=n/s$를 정하면

$$
\Omega_bh^2=
\frac{m_B\nu_bq_{\rm ext}Y_Xs_0}{\rho_{c,0}/h^2},
$$

$$
\Omega_ch^2=
\frac{m_\chi\nu_\chi(1-q_{\rm ext})Y_Xs_0}
{\rho_{c,0}/h^2}.
$$

여기서 $q_{\rm ext}$는 **number/charge composition**을 고정한다. $Y_X$,
$m_\chi$, decay stoichiometry와 entropy dilution은 고정하지 않는다. total
yield를 열적 동결로 계산하려면 별도로

$$
\frac{dY_X}{dx}=-\frac{s\langle\sigma v\rangle}{Hx}
\left(Y_X^2-Y_{X,{\rm eq}}^2\right),qquad x=\frac{m_X}{T}
$$

를 풀어야 한다.

### 2.4 자유량, 교차관측과 kill test

- 자유 매개변수/초기자료: $m_X,\mu_*,\kappa(T)$, annihilation 또는
  freeze-in coupling, $T_R$, $m_\chi$, $\nu_b,\nu_\chi$, 초기 total yield.
- 독립 교차관측: 반응 heat/entropy injection, BBN와 $\Delta N_{\rm eff}$,
  residual baryon--dark isocurvature, dark-particle mass, annihilation/decay 신호.
- kill A1: 실제 $b\leftrightarrow c$의 질량차와 chemical potentials를 넣었을
  때 $\mathcal A=\mu_*v_D'$가 유지되지 않으면 정확한 root가 이동한다.
- kill A2: microscopic detailed balance에서 $v_D$와 양의 mobility가 나오지
  않으면 constitutive law는 모델 공리로 남는다.
- kill A3: $T\nabla S\ge0$와 total stress 보존을 동시에 만족하는 heat sector가
  없으면 route는 폐기한다.
- kill A4: $Y_X$를 계산하지 않고 flatness 또는 관측 $\Omega_b$로 역산하면
  absolute-abundance 유도가 아니다.
- kill A5: 반응이 BBN 이후 실제 baryon number를 바꾸면 light-element와 CMB
  제약을 통과해야 한다.

**정확한 missing lemma A:** 관측 밀도를 사용하지 않은 Lorentz-covariant
microscopic theory가 (i) label-blind 전구체, (ii) 위 $v_D$의 chemical
affinity, (iii) $T_f$, (iv) $Y_X$, (v) decay stoichiometry와 $m_\chi$를
모두 고정해야 한다. 대안은 A의 total-yield 부분을 Route B/C로 교체하는 것이다.

## 2A. Route E — extinction-conditioned 국소 수축률을 matter composition으로

이 route는 $q_{\rm ext}$ 자체를 density로 읽지 않는다. 원래 Poisson
branching map

$$
f(s)=\exp[D(s-1)],\qquad q=f(q)
$$

의 **소멸 조건부 국소 offspring law**를 사용한다.

### 2A.1 조건부 Poisson 정리

조상의 자식 수를 $K$라 하고 전체 tree의 소멸 사건을 $E$라고 하자. $K=k$일
때 모든 $k$개 subtree가 소멸할 확률은 $q^k$이므로 Bayes 법칙으로

$$
\Pr(K=k\mid E)=\frac{\Pr(K=k)q^k}{q}.
$$

따라서 조건부 PGF는

$$
\mathbb E[s^K\mid E]
=\frac{f(qs)}q
=\exp[Dq(s-1)].
$$

즉 소멸 조건부 offspring law는 정확히

$$
K\mid E\sim{\rm Poisson}(m),qquad
m:=Dq=f'(q).
$$

작은 근은 $q<1/D$이므로 $m<1$이고 conditioned tree는 subcritical이다.
$m$은 단지 새 숫자가 아니라 original fixed-point iteration의 국소 수축률이자
소멸 조건부 평균 자식 수다. 대표 입력에서는

$$
\boxed{m=Dq=0.15458752312007407}.
$$

### 2A.2 total progeny에서 나오는 정확한 composition

소멸 조건부 tree의 root를 포함한 total node 수를 $T$라 하자. subcritical
Galton--Watson 재귀로

$$
\mathbb E[T]=1+m\mathbb E[T]
=\frac1{1-m},
$$

$$
\mathbb E[T-1]=\frac{m}{1-m}.
$$

따라서 많은 독립 tree를 합쳐 각 node가 같은 평균 rest energy를 가진다고
가정하면, 강한 대수의 법칙 아래 aggregate descendant energy fraction은

$$
f_b^{(m)}
=\frac{\mathbb E[T-1]}{\mathbb E[T]}
=m=Dq.
$$

여기서 `descendant tag $\leftrightarrow$ baryonic matter`,
`root tag $\leftrightarrow$ non-baryonic matter`를 microscopic species map으로
구현하면

$$
\boxed{f_b^{(m)}=\frac{\rho_b}{\rho_b+\rho_c}=Dq},
$$

$$
\boxed{\Omega_b=(Dq)\Omega_m}.
$$

가 된다. 이는 이전 direct-$q$ route의 additive vacuum-offset 반례와 scalar
$w=-1$ 반례를 피한다. 사용한 weight는 scalar potential energy가 아니라
동결 뒤 equal-energy dust node의 수이기 때문이다. 또한 출력은 처음부터
critical fraction이 아니라 **matter 내부 composition**이므로 개념적 층도
맞다.

### 2A.3 equal-energy/current realization의 최소 조건

동결면 $\Sigma_f$에서 root relic와 descendant relic current를

$$
J_r^\mu=n_ru^\mu,qquad J_b^\mu=n_bu^\mu
$$

로 두고, 동결 뒤

$$
\nabla_\mu J_r^\mu=0,qquad
\nabla_\mu J_b^\mu=0
$$

라 하자. 각 relic의 평균 에너지가 같으면

$$
T_r^{\mu\nu}=mn_ru^\mu u^\nu,qquad
T_b^{\mu\nu}=mn_bu^\mu u^\nu,qquad
\frac{n_b}{n_b+n_r}=Dq.
$$

이것이 current-level 존재구성이다. 그러나 branching tree의 `total progeny`는
역사적 event count다. parent가 자식으로 교체되면 모든 과거 node가 한 시각에
존재하지 않는다. 따라서 물리 realization은 다음 중 하나를 실제 action으로
구현해야 한다.

1. 각 branching event가 안정한 equal-energy memory/relic 하나를 남긴다.
2. reservoir가 event마다 relic rest energy를 공급하고 그 stress/entropy를
   함께 보존한다.
3. final particle 수가 total progeny와 같은 decay/cascade network를 만든다.

그렇지 않으면 수학적 node count를 입자 current로 바꾼 것이 아니다. 반응 중
energy-momentum conservation과 entropy production은 Route A의 reservoir/heat
sector 또는 Route B의 charge-conserving decay action으로 completion할 수 있다.

### 2A.4 readout 비유일성과 target-awareness

같은 conditioned Poisson law에는 여러 자연스러운 통계량이 있다.

$$
\Pr(K\ge1\mid E)=1-e^{-m}=0.14323149819200887,
$$

$$
\Pr(K=1\mid E)=me^{-m}=0.13244572058179405.
$$

또한 tree별 descendant fraction을 먼저 평균하면 aggregate ratio와 다르다.
Poisson subcritical total-progeny PGF를 적분하면

$$
\mathbb E\left[\frac{T-1}{T}\right]=\frac m2
=0.07729376156003704,
$$

반면 equal-energy node를 전체 ensemble에서 합친 ratio-of-expectations는 $m$이다.
따라서 $m$은 `모든 node가 실제 conserved relic이고 energy를 node 단위로 합산`
한다는 물리 측정연산자 아래에서만 선호된다. occupancy, exactly-one-child 또는
tree-uniform averaging보다 $m$을 고르는 이유가 단지 원하는 우주 수치와
가깝기 때문이면 target-aware 선택이다.

### 2A.5 $\Omega_m=1/D$ 결합과 재발하는 no-go

별도 우주 동역학이 지정된 면에서

$$
\Omega_m=\frac1D
$$

를 독립적으로 만들면

$$
\Omega_b=(Dq)\Omega_m=q
$$

가 정확히 성립한다. 대표값은

$$
\frac1D=0.3146872312990126.
$$

그러나 이를 **평탄 GR 가속기의 영구 attractor**로 만들면 conservation
no-go가 다시 발생한다. total matter에

$$
\dot\rho_m+3H\rho_m=Q_m
$$

를 허용하면

$$
\frac{d\ln\Omega_m}{d\ln a}
=3w_{\rm tot}+\frac{Q_m}{H\rho_m}.
$$

$\Omega_m=1/D$를 상수로 유지하려면

$$
\frac{Q_m}{H\rho_m}=-3w_{\rm tot}
$$

가 필요하다. 동시에 $f_b^{(m)}=m$을 상수로 두면
$Q_b=mQ_m$이어야 한다. fixed-mass conserved baryon은 $Q_b=0$이므로
$w_{\rm tot}\ne0$인 가속 구간에서는 양립하지 않는다. $Q_m$을 CDM에만 주면
baryon current는 보존되지만 $f_b^{(m)}$가 더 이상 $m$으로 유지되지 않는다.

따라서 허용 가능한 주장은 영구 attractor가 아니라 다음 둘 중 하나다.

- action-defined finite hypersurface $\Sigma_*$에서
  $\Omega_m(\Sigma_*)=1/D$인 transient crossing을 유도한다.
- BBN 이전에 interaction을 종료하고 이후 conserved fluids를 전방 적분해
  별도의 관측면에서 $\Omega_b$를 계산한다.

### 2A.6 자유량, 독립관측과 kill test

- 자유 매개변수/초기자료: covariant branching clock, event rate, reservoir,
  node-to-species map, relic mass/energy distribution, cascade 종료면, total tree
  density, $\Omega_m$ dynamics.
- $q$가 실제로 고정하는 것: conditioned offspring mean과 aggregate
  equal-node-energy matter composition $m=Dq$; total tree density와
  critical fraction은 고정하지 않는다.
- 독립 교차관측: baryon isocurvature와 higher factorial cumulants,
  Poisson-conditioned multiplicity, relic mass, reservoir entropy/decay products,
  $\Omega_m(a)$ transition과 growth.
- kill E1: microscopic cascade의 conditioned PGF가 $f(qs)/q$가 아니면 폐기.
- kill E2: historical node가 stable relic/current로 남지 않으면 total-progeny
  composition 해석을 폐기.
- kill E3: node energy가 tag 또는 generation에 의존하면 energy-weighted
  fraction을 다시 계산하며 $Dq$를 유지하지 않는다.
- kill E4: occupancy 또는 $P_1$도 가능한데 species detector가 왜 aggregate
  descendant/total을 측정하는지 action이 정하지 못하면 readout 미완성.
- kill E5: $\Omega_m=1/D$를 관측값에 맞춘 정의로 넣으면 유도가 아니다.
- kill E6: accelerated interval에서 $\Omega_m=1/D$와 conserved baryon,
  $f_b^{(m)}=Dq$를 동시에 상수로 주장하면 위 no-go로 폐기.

**정확한 missing lemma E:** covariant cascade action이 소멸 조건부 tree의 모든
node를 equal-energy conserved relic로 변환하고, 그 descendant tag가 baryon임을
독립 charge/symmetry로 고정하며, 별도 background action이 유일한
$\Sigma_*$에서 $\Omega_m=1/D$를 만든다는 명제다. 이 lemma가 성립하면 direct
$q\mapsto\Omega_b$를 쓰지 않고도 $\Omega_b=q$가 나온다. 실패하면 $Dq$는
Route A/B의 matter-composition target으로만 보존하고 absolute normalization은
동결수율로 계산한다.

## 3. Route B — 보존전하와 asymmetric freeze-out

### 3.1 action과 보존 current

복소 응축장 $\Phi=Re^{i\vartheta}/\sqrt2$, 안정한 Dirac dark field $\chi$,
Standard Model의 $B-L$ current를 포함하는 후보를

$$
S_B=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R-|\nabla\Phi|^2-V(\Phi)
+\bar\chi(i\!\not\!D-m_\chi)\chi
+\mathcal L_{\rm SM}
+\frac{\mathcal O_{B-L}\mathcal O_\chi}{\Lambda^{d-4}}
+{\rm h.c.}\right]
$$

로 둔다. transfer operator는 총 $U(1)_Q$를 보존하도록 charge를 배정한다.
응축장 torque가 끝난 뒤의 total current는

$$
J_Q^\mu=iQ_\Phi\Phi^*\!\stackrel{\leftrightarrow}{\nabla^\mu}\!\Phi
+J_{B-L}^\mu+Q_\chi\bar\chi\gamma^\mu\chi,
$$

$$
\nabla_\mu J_Q^\mu=0.
$$

초기 charge 생성에 Affleck--Dine형 작은 $U(1)$ breaking term을 쓰면 생성
구간에는

$$
\nabla_\mu J_Q^\mu=-\frac{\partial V_{\not Q}}{\partial\vartheta}
$$

이고, 그 항이 꺼진 뒤 comoving charge가 보존된다. scalar/fermion의 표준
Hilbert stress와 bath stress를 모두 합치면 diffeomorphism invariance로 total
$\nabla_\mu T^{\mu\nu}=0$이다.

### 3.2 chemical sharing과 Boltzmann EOM

transfer reaction이 평형일 때 각 반응 $a$에 대해

$$
\sum_i\nu_{ai}\mu_i=0,
$$

작은 chemical potential에서는

$$
n_i-\bar n_i=c_i(T)\mu_iT^2+O(\mu_i^3)
$$

가 된다. 이 선형계와 gauge neutrality, sphaleron constraint를 함께 풀면 총
charge가 visible과 dark sector에 분배된다. CE가 추가로 증명해야 할 것은

$$
Y_{\Delta B}=q_{\rm ext}Y_Q,qquad
Y_{\Delta\chi}=(1-q_{\rm ext})Y_Q
$$

또는 charge/stoichiometry를 포함한 동등식이다. finite relativistic species의
선형 chemical-equilibrium constraint만으로는 보통 정수 charge와 degeneracy의
유리함수만 나온다. Lambert-$W$형 $q_{\rm ext}$를 정확히 얻으려면 Boltzmann
suppression, cascade network 또는 별도의 nonlinear transfer dynamics가
필요하다.

비대칭과 symmetric component의 진화는 개략적으로

$$
\frac{dY_{\Delta i}}{dx}
=-\frac{\gamma_{\rm tr}}{sHx}\frac{\mathcal A_i}{T},
$$

$$
\frac{dY_{\Sigma\chi}}{dx}
=-\frac{s\langle\sigma v\rangle_\chi}{Hx}
\left(Y_\chi Y_{\bar\chi}-Y_{\chi,{\rm eq}}Y_{\bar\chi,{\rm eq}}\right)
$$

로 정해진다. transfer 동결은 $\Gamma_{\rm tr}(T_D)=H(T_D)$, symmetric
annihilation은 그보다 충분히 강해야 한다. detailed balance가 성립하면 collision
term은

$$
\nabla_\mu S^\mu=-\sum_i\frac{\mu_i}{T}C_i\ge0
$$

를 만족한다.

### 3.3 abundance와 $q$의 역할

symmetric component가 제거된 뒤

$$
\Omega_bh^2=
\frac{m_Bs_0|Y_{\Delta B}|}{\rho_{c,0}/h^2},qquad
\Omega_ch^2=
\frac{m_\chi s_0|Y_{\Delta\chi}|}{\rho_{c,0}/h^2}.
$$

따라서 단위 charge/stoichiometry 예에서는

$$
\frac{\Omega_c}{\Omega_b}
=\frac{m_\chi}{m_B}\frac{1-q_{\rm ext}}{q_{\rm ext}}.
$$

$q_{\rm ext}$는 **총 charge의 분배**를 정할 수 있지만 총
$Y_Q=n_Q/s$를 정하지 않는다. 반대로 $Y_Q$가 응축장 dynamics와 reheating에서
계산되면 absolute $\Omega_b$와 $\Omega_c$를 target insertion 없이 계산할 수
있다. dark energy는 이 route가 만들지 않는다.

### 3.4 자유량, 교차관측과 kill test

- 자유 매개변수/초기자료: $V(\Phi)$, 초기 $R,\dot\vartheta$, CP phase,
  $T_R$, transfer scale $\Lambda$, charge assignment, $m_\chi$,
  $\langle\sigma v\rangle_\chi$.
- 독립 교차관측: $m_\chi$와 $\Omega_c/\Omega_b$ 관계, baryon/dark
  isocurvature, symmetric annihilation 신호, mediator decay와 collider signature.
- kill B1: transfer network의 charge matrix가 $q_{\rm ext}$ eigenvector를
  생성하지 않으면 $q$ 분배는 폐기한다.
- kill B2: 초기 $Y_Q$를 관측 $\eta_B$에 맞추면 absolute 유도가 아니다.
- kill B3: symmetric remnant가 무시되지 않으면 위 질량비 식은 깨진다.
- kill B4: sphaleron, entropy dilution, phase transition을 생략하면 visible
  baryon yield가 아니다.
- kill B5: inflation fluctuation이 만드는 charge isocurvature가 허용치를
  넘으면 응축장 route를 폐기한다.

**정확한 missing lemma B:** action과 초기 quantum state가 유일한 attractor
$Y_Q=\mathcal Y(D,\text{measured micro couplings})$를 만들고, transfer matrix의
정규화된 charge-share eigenvector가 $q_{\rm ext}$임을 보여야 한다. 대안은
총 charge source를 Route C의 anomaly/rolling solution으로 바꾸는 것이다.

## 4. Route C — topological current와 chemical potential

### 4.1 공변 action, anomaly와 stress

compact angle $\theta=a/f_a$를 갖는 axion-like 후보는

$$
S_C=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac{f_a^2}{2}(\nabla\theta)^2-U(\theta)
+\mathcal L_{\rm plasma}
+c_Q\partial_\mu\theta J_Q^\mu
+\frac{c_A\theta g^2}{32\pi^2}F^a_{\mu\nu}\widetilde F^{a\mu\nu}
\right]
$$

로 쓸 수 있다. anomaly는

$$
\nabla_\mu J_Q^\mu
=c_{\rm an}\frac{g^2}{32\pi^2}F^a_{\mu\nu}\widetilde F^{a\mu\nu}
$$

를 준다. $\theta F\widetilde F$는 differential-form으로 metric 독립인
topological coupling이지만, axion과 gauge/plasma EOM을 통해 sector 사이에
에너지가 교환된다. total Hilbert stress는 보존된다. axion 부분은

$$
T_{\mu\nu}^{(\theta)}=f_a^2\partial_\mu\theta\partial_\nu\theta
-g_{\mu\nu}\left[\frac{f_a^2}{2}(\partial\theta)^2+U(\theta)\right]
$$

이고 EOM은

$$
f_a^2\Box\theta-U_{,\theta}
=-c_Q\nabla_\mu J_Q^\mu
-c_A\frac{g^2}{32\pi^2}F\widetilde F
$$

이다.

### 4.2 rolling chemical bias와 freeze-out

균일한 rolling background에서 derivative coupling은 반응 kinetics에

$$
\mu_Q=c_Q\dot\theta
$$

의 bias를 준다. 이 해석에는 Hamiltonian/kinetic subtlety가 있으므로 단순한
single-particle energy shift로 끝내지 않고 Boltzmann/linear-response 계산으로
검증해야 한다. thermal susceptibility를 $\chi_Q(T)=c_\chi T^2$라고 하면

$$
n_Q=\chi_Q\mu_Q+O(\mu_Q^3),
$$

$$
Y_Q(T_F)=\frac{45c_\chi c_Q}{2\pi^2g_{*s}(T_F)}
\frac{\dot\theta(T_F)}{T_F}
$$

가 된다. $Q$-violating topology change가
$\Gamma_Q(T_F)=H(T_F)$에서 동결되면 이 yield가 이후 보존된다. sphaleron과
charge-conversion matrix는 별도 계수 $c_{\rm sph}$로 포함해야 한다.

topological event의 연속 cascade가 평균 $D$인 Poisson branching을 실제로
이룬다면 $q_{\rm ext}$는 `baryonic sink에 끝나는 cascade의 비율` 같은 종점
확률을 고정할 수 있다. 그러나 절대 source는

$$
\frac{\Gamma_{\rm top}}{HT^3},\qquad
\frac{\mu_Q}{T},qquad
\text{washout eigenvalues}
$$

가 정한다. 확률과 energy/charge weight의 conditional mean이 다르면 종점
빈도 $q$와 생성 charge fraction도 다르다.

### 4.3 핵심 kill test: 정지 root는 baryogenesis를 만들지 않는다

이 route에서 가장 중요한 반례는 간단하다. $\theta=q_{\rm ext}$가 정지한
해라면

$$
\dot\theta=0\quad\Longrightarrow\quad
\mu_Q=0\quad\Longrightarrow\quad Y_Q^{\rm eq}=0.
$$

따라서 이전 scalar potential의 **정지값**을 chemical potential이라고 이름만
바꾸는 것은 작동하지 않는다. 필요한 것은 limit cycle, terminal-velocity
solution, moving wall 또는 anomaly backreaction을 포함한 rolling attractor다.

### 4.4 정규화, dark sector와 kill test

axion remnant를 dark matter로 쓰면 misalignment/fragmentation 계산이
$\Omega_c$를 줄 수 있지만 $f_a,U,\theta_i$에 의존한다. $U$의 상수 잔여를
dark energy로 쓰면 vacuum offset과 radiative stability가 다시 독립 문제다.
한 장이 세 역할을 할 수 있다는 것은 parameter-free closure의 증명이 아니다.

- 자유 매개변수/초기자료: $f_a,U,c_Q,c_A$, anomaly coefficient,
  $\theta_i,\dot\theta_i$, reheating, gauge conductivity와 sphaleron rate.
- 독립 교차관측: baryon isocurvature, parity-odd gauge field와 gravitational
  wave, axion coupling/mass, EDM, washout와 BBN.
- kill C1: 정지 $q$를 쓰면 $Y_B=0$이다.
- kill C2: rolling solution이 초기 misalignment에 민감하면 absolute yield는
  자유 초기조건이다.
- kill C3: anomaly/backreaction/diffusion을 생략한 chemical-potential 식은
  정량 예측이 아니다.
- kill C4: event count를 charge density로 바꾸는 weight theorem이 없으면
  $q$는 composition도 고정하지 못한다.
- kill C5: 같은 $U$의 additive constant로 $\Omega_{\rm DE}$를 정하면 이전
  vacuum-offset 반례가 그대로 남는다.

**정확한 missing lemma C:** action이 초기조건 독립인 rolling solution과
동적으로 선택된 $T_F$에서

$$
\left.\frac{\dot\theta}{T}\right|_{T_F}
=\mathcal F(D,c_A,c_Q,g)
$$

를 유일하게 정하고, topology-to-charge response와 washout까지 계산해야 한다.
대안은 C를 오직 $Y_Q$ source로 사용하고 charge 분배는 Route B에 맡기는 것이다.

## 5. Route D — unified dark-sector scaling attractor

### 5.1 D1: covariant effective interacting-vacuum fluid

먼저 가장 투명한 effective-fluid route를 둔다.

$$
T_b^{\mu\nu}=\rho_bu_b^\mu u_b^\nu,qquad
T_c^{\mu\nu}=\rho_cu_c^\mu u_c^\nu,qquad
T_{\rm de}^{\mu\nu}=-\rho_{\rm de}g^{\mu\nu}.
$$

baryon은 보존하고 dark sector만

$$
\nabla_\mu T_c^{\mu\nu}=Q^\nu,qquad
\nabla_\mu T_{\rm de}^{\mu\nu}=-Q^\nu,qquad
Q^\nu=\xi\Theta\rho_{\rm de}u^\nu
$$

로 상호작용시킨다. $\Theta=\nabla_\mu u^\mu$이고 FLRW에서 $\Theta=3H$다.
따라서

$$
\dot\rho_c+3H\rho_c=3H\xi\rho_{\rm de},
$$

$$
\dot\rho_{\rm de}=-3H\xi\rho_{\rm de}.
$$

$r=\rho_c/\rho_{\rm de}$는

$$
\frac{dr}{d\ln a}=3[\xi-(1-\xi)r]
$$

를 만족하므로

$$
r_*=\frac{\xi}{1-\xi},qquad
\left.\frac{d}{dr}\frac{dr}{d\ln a}\right|_{r_*}
=-3(1-\xi)<0
$$

인 안정한 scaling point가 있다. 이때

$$
w_{{\rm eff},D}=-(1-\xi),qquad
\rho_D\propto a^{-3\xi}.
$$

fixed CDM mass 해석에서는
$\nabla_\mu N_c^\mu=Q/m_c\ge0$이고 adiabatic particle creation entropy도
증가한다. CDM number를 엄밀히 보존하려면 대신 $m_c(\phi)$가 변하는 D2 같은
completion이 필요하다.

기존 CE의 관측 비율이 아닌 내부 후보

$$
R_D(D):=\alpha_sD
$$

를 dark coupling에서 유도할 수 있다는 lemma가 있다면

$$
\xi(D)=\frac{R_D(D)}{1+R_D(D)}
$$

로 두어 $r_*=R_D$를 얻는다. 이는 algebraically 정확하지만
$\xi(D)$ 자체는 아직 microscopic theorem이 아니다. 대표 입력에서는

$$
R_D=0.3746259405358018,qquad
\xi=0.2725293692550135,qquad
w_{{\rm eff},D}=-0.7274706307449865.
$$

즉 이 exact scaling branch는 $\Lambda$CDM이 아니라
$\rho_D\propto a^{-0.817588\ldots}$를 예측한다. distance와 growth에 대한
강한 독립 kill test다.

### 5.2 D2: conformally coupled canonical scalar completion

particle creation 대신 conserved CDM number와 variable mass를 쓰는 명시적
action 후보는

$$
S_{D2}=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R-\frac12(\nabla\phi)^2
-V_0e^{-\lambda\phi/M_{\rm Pl}}\right]
+S_c[A^2(\phi)g_{\mu\nu},\psi_c]+S_b[g_{\mu\nu},\psi_b],
$$

$$
A(\phi)=e^{\beta\phi/M_{\rm Pl}}
$$

다. convention을 $m_c(\phi)=m_{c0}A(\phi)$로 잡으면

$$
\nabla_\mu(n_cu_c^\mu)=0,qquad
\dot\rho_c+3H\rho_c=
\frac{\beta\dot\phi}{M_{\rm Pl}}\rho_c,
$$

$$
\ddot\phi+3H\dot\phi+V_{,\phi}
=-\frac{\beta}{M_{\rm Pl}}\rho_c.
$$

scalar stress는 canonical 식이고, CDM과 scalar stress의 exchange는 서로
상쇄되어 total stress가 보존된다. reversible variable-mass 해석에서는 cold
sector의 specific entropy가 일정하다.

바리온과 복사를 무시한 dark-only autonomous 변수

$$
x=\frac{\dot\phi}{\sqrt6HM_{\rm Pl}},qquad
z=\frac{\sqrt V}{\sqrt3HM_{\rm Pl}}
$$

에는 $L=\lambda+\beta$일 때 scaling point

$$
x_*=\frac{\sqrt{3/2}}L,qquad
z_*^2=\frac{\beta}{L}+\frac{3}{2L^2},
$$

$$
\Omega_{\phi,*}=\frac{\beta}{L}+\frac3{L^2},qquad
\Omega_{c,*}=\frac{\lambda L-3}{L^2},qquad
w_{\rm eff,*}=-\frac{\beta}{L}
$$

가 있다. $\lambda L\ge3$가 양의 CDM fraction에 필요하고
$\beta>\lambda/2$면 가속한다. $R_D$와 $-w_{\rm eff}=0.69$를 예시 이론
조건으로 동시에 주면

$$
\lambda=2.7738107016,qquad
\beta=6.1739657551
$$

이고 $\Omega_c/\Omega_\phi=R_D$다. autonomous Jacobian 고유값은

$$
-1.2675\pm6.69224i
$$

로 이 dark-only 예시는 국소 attractor다. 그러나 매우 큰 coupling은 dark
fifth force, CDM mass drift와 structure growth에서 즉시 시험된다.

### 5.3 정규화, freeze-out과 kill test

D1/D2의 scaling point는 **dark ratio**를 초기 amplitude와 무관하게 만들 수
있다. 그러나 $H_0$ 또는 전체 dark density scale은 고정하지 않는다. D2의
$V_0,m_{c0}$, 초기 trajectory가 dimensional normalization을 가진다. 상호작용을
종료하려면 $\phi=\phi_*$ 또는 다른 action-defined phase transition을 써야 하며,
`오늘`이라는 조건을 넣으면 안 된다.

- 자유 매개변수/초기자료: D1의 $\xi$와 interaction switch; D2의
  $\lambda,\beta,V_0,m_{c0},\phi_i,\dot\phi_i$와 radiation/baryon 초기밀도.
- 독립 교차관측: $H(z)$, $f\sigma_8$, CMB early dark energy, dark fifth
  force, CDM mass drift, ISW와 lensing.
- kill D1: $\xi=0.2725$ exact attractor의 $a^{-0.8176}$ expansion/growth가
  자료와 맞지 않으면 이 단순 interaction law를 폐기한다.
- kill D2: $\lambda,\beta$를 $R_D$에 맞춰 역산하면 유도가 아니다. 두 값은
  UV gauge/conformal sector에서 먼저 나와야 한다.
- kill D3: perturbation eigenmode에 ghost, gradient instability 또는 너무 큰
  fifth force가 있으면 background attractor와 무관하게 폐기한다.
- kill D4: $V_0$를 관측 $\rho_{\rm DE}$에 맞추면 absolute normalization은
  남는다.
- kill D5: conserved baryon을 같은 accelerated fixed point의 nonzero constant
  fraction이라고 주장하면 아래 no-go와 충돌한다.

**정확한 missing lemma D:** 단일 UV action의 beta function, mass generation과
potential이 관측 cosmology를 참조하지 않고 $\lambda(D),\beta(D)$ 또는
$\xi(D)$를 고정하고, 그 scaling point가 background와 perturbation 양쪽에서
안정해야 한다. 대안은 exact scaling을 영구 late-time 해로 쓰지 않고,
action-defined transition에서 상호작용이 꺼지는 transient tracker로 쓰는 것이다.

## 6. 다섯 경로를 동시에 연결할 때의 정확한 조건

### 6.1 비율만으로 세 fraction을 닫는 대수

한 hypersurface $\Sigma_*$에서

$$
A_*:=\frac{\rho_c}{\rho_b},qquad
R_*:=\frac{\rho_c}{\rho_{\rm DE}}
$$

가 action에서 계산되고 평탄하며 radiation이 무시 가능하면

$$
\Omega_b=\frac1{1+A_*+A_*/R_*},
$$

$$
\Omega_c=A_*\Omega_b,qquad
\Omega_{\rm DE}=\frac{A_*}{R_*}\Omega_b.
$$

Route A/B의 단위 stoichiometry에서는

$$
A_*=\frac{m_\chi}{m_B}\frac{1-q_{\rm ext}}{q_{\rm ext}}.
$$

따라서 $q$ 하나를 critical density와 직접 동일시하지 않고도, **독립적으로
유도된 질량비와 dark ratio**가 있으면 세 fraction을 정할 수 있다.

기존 CE conditional split

$$
R_D=\alpha_sD,qquad
\Omega_b=q,qquad
\Omega_c=(1-q)\frac{R_D}{1+R_D},qquad
\Omega_{\rm DE}=\frac{1-q}{1+R_D}
$$

와 동일한 대수를 위 물리 route로 얻기 위한 정확한 질량 lemma는

$$
\boxed{\frac{m_\chi\nu_\chi}{m_B\nu_b}
=\frac{R_D}{1+R_D}=\xi(D)}.
$$

대표 입력과 단위 stoichiometry에서는

$$
\frac{m_\chi}{m_p}=0.2725293693,qquad
m_\chi\simeq255.71\ {\rm MeV}.
$$

이는 관측을 맞춘 숫자로 숨길 것이 아니라 collider, cosmology와 dark-sector
실험이 죽일 수 있는 독립 예측 후보로 다뤄야 한다. 현재 action에서 이 mass
relation은 유도되지 않았다.

### 6.2 동시 fixed point no-go

위 대수는 한 면에서의 항등식이지 세 fraction의 영구 attractor 증명이 아니다.
고정질량 conserved baryon dust에는

$$
\frac{d\ln\Omega_b}{d\ln a}=3w_{\rm tot}.
$$

따라서 $\Omega_b>0$가 열린 시간구간에서 상수이면 $w_{\rm tot}=0$이다.
가속은 $w_{\rm tot}<-1/3$를 요구하므로 다음이 성립한다.

> **no-go:** fixed-mass conserved baryon이 nonzero인 가속 FLRW에서
> $(\Omega_b,\Omega_c,\Omega_{\rm DE})$ 모두를 상수로 만드는 late-time
> scaling attractor는 존재하지 않는다.

그러므로 A/B의 frozen baryon--CDM ratio와 D의 영구 CDM--DE scaling ratio를
동시에 오늘의 상수라고 둘 수 없다. 가능한 대안은 세 가지다.

1. dark interaction이 action-defined $\Sigma_*$에서 꺼지고, 그 뒤의 서로
   다른 redshift를 전방 적분한다.
2. 오늘은 attractor가 아니라 계산된 transient crossing이다. crossing 시각은
   관측 $t_0$가 아니라 미시 scale과 thermal clock에서 나와야 한다.
3. baryon mass/current도 변하게 한다. 이는 fifth force, mass drift와 baryon
   conservation의 강한 kill test를 감수해야 한다.

첫 번째가 가장 보수적인 full-closure 후보다.

### 6.3 절대 density와 fraction의 구별

두 비율과 flatness는 지정된 면의 $\Omega_i$를 정할 수 있지만
$\rho_{c,*}=3M_{\rm Pl}^2H_*^2$의 dimensionful scale은 정하지 않는다.
절대 energy density와 $H_0$까지 목표라면 추가로

$$
H_* = \mathcal H(D,\text{microphysical scales})
$$

가 필요하다. 반대로 thermal yield에서 $\rho_b=m_BY_Bs$를 계산해도
$\Omega_b$로 바꾸려면 같은 면의 $H_*$가 필요하다. 이 dimensional lemma를
ratio algebra로 대체할 수 없다.

## 7. 완결을 위한 missing-lemma 묶음과 우회 경로

목표는 세 fraction과 절대 scale 모두를 유지한다. 필요한 명제는 다음과 같이
분리된다.

| ID | 완결에 필요한 정확한 명제 | 1차 시도 | 실패 시 대안 |
|---|---|---|---|
| L-U2.1 | microscopic species/tag와 reaction network가 $v_D'(y)$ affinity를 만든다 | Route A dissipative EFT의 SK/convective completion | Route B의 charge-sharing cascade |
| L-U2.E | extinction-conditioned total progeny의 descendant node가 equal-energy conserved baryon relic가 된다 | covariant cascade + reservoir current | $Dq$를 Route A의 composition potential 최소로만 사용 |
| L-U2.2 | $q_{\rm ext}$가 charge-weighted 또는 energy-weighted 분배와 같다 | equal conditional energy/charge 정리 | 명시적 unequal weight를 계산해 readout 함수 도출 |
| L-U2.3 | $Y_Q$ 또는 $Y_X$가 초기조건 독립으로 계산된다 | thermal freeze-out/freeze-in | Route C rolling anomaly attractor |
| L-U2.4 | sphaleron, entropy와 decay 뒤 visible baryon current가 보존된다 | $B-L$ transfer action | late baryogenesis와 BBN 전 종료 |
| L-U3.1 | dark interaction coupling이 $R_D(D)$를 낸다 | conformal scalar/gauge dimensional transmutation | 다른 scalar-fluid 또는 transient tracker |
| L-U3.2 | background와 perturbation fixed point가 모두 안정하다 | Einstein--Boltzmann eigenmode 계산 | interaction 함수 $Q(\phi,\rho_i)$ 변경 |
| L-U23.1 | A/B freeze-out과 D interaction 종료를 잇는 공변 $\Sigma_*$가 유일하다 | scalar/thermal phase transition $F(\phi,T)=0$ | decay-clock field 또는 first-order transition |
| L-U23.2 | $m_\chi\nu_\chi/(m_B\nu_b)=R_D/(1+R_D)$가 symmetry에서 나온다 | 공통 confinement/portal mass generation | 질량비를 독립 예측으로 두고 다른 charge stoichiometry 탐색 |
| L-U23.3 | $H_*$와 reheating/entropy scale가 action에서 나온다 | inflation/reheating sector와 Friedmann constraint | primordial four-form/phase-transition scale route |

이 표의 어느 행도 관측 $\Omega_i$를 coupling에 넣어 해결할 수 없다. 실패한
행은 목표를 삭제하는 근거가 아니라 오른쪽 대안으로 route를 교체하는 기준이다.

## 8. 수치 scratch 검산

독립 script:
`artifacts/verify_density_dark_routes.py`

검산 항목은 다음과 같다.

1. $q_{\rm ext}$ residual과 positive Hessian.
2. extinction-conditioned PGF가 Poisson($Dq$)인지와
   descendant/total=$Dq$, $(Dq)/D=q$ 항등식.
3. $dy/dN=-(\kappa/H)v_D'(y)$의 $q$ 수렴.
4. entropy-production square의 비음수성.
5. $q+\Omega_c+\Omega_{\rm DE}=1$인 candidate algebra.
6. D1 ratio fixed point와 음의 안정 고유값.
7. D2 autonomous EOM residual, Jacobian 안정성과 dark ratio.
8. 질량 lemma를 썼을 때 ratio closure가 $\Omega_b=q$를 재생하는지.
9. 정지 topological field의 $\mu/T=0$ 반례.

이 검산은 다섯 route의 algebra를 확인할 뿐 physical bridge를 PASS시키지 않는다.

## 9. 1차 문헌 경계

아래 문헌은 route의 알려진 변분/동역학 구조가 실제로 존재한다는 배경만
제공한다. CE 고유의 $v_D$, $q_{\rm ext}$, mass lemma 또는 dark coupling
map을 증명하지 않는다. 접근일은 모두 2026-08-15다.

1. J. D. Brown, “Action functionals for relativistic perfect fluids,”
   *Class. Quantum Grav.* **10**, 1579 (1993),
   [arXiv:gr-qc/9304026](https://arxiv.org/abs/gr-qc/9304026),
   [DOI](https://doi.org/10.1088/0264-9381/10/8/017).
2. N. Andersson and G. L. Comer, “A covariant action principle for dissipative
   fluid dynamics,” *Class. Quantum Grav.* **32**, 075008 (2015),
   [arXiv:1306.3345](https://arxiv.org/abs/1306.3345).
3. D. E. Kaplan, M. A. Luty and K. M. Zurek, “Asymmetric Dark Matter,”
   *Phys. Rev. D* **79**, 115016 (2009),
   [arXiv:0901.4117](https://arxiv.org/abs/0901.4117).
4. I. Affleck and M. Dine, “A New Mechanism for Baryogenesis,”
   *Nucl. Phys. B* **249**, 361 (1985),
   [DOI](https://doi.org/10.1016/0550-3213(85)90021-5).
5. A. G. Cohen and D. B. Kaplan, “Spontaneous baryogenesis,”
   *Nucl. Phys. B* **308**, 913 (1988),
   [DOI](https://doi.org/10.1016/0550-3213(88)90134-4).
6. A. Dasgupta, R. K. Jain and R. Rangarajan, “Effective chemical potential in
   spontaneous baryogenesis,” [arXiv:1808.04027](https://arxiv.org/abs/1808.04027).
7. L. Amendola, “Coupled Quintessence,” *Phys. Rev. D* **62**, 043511 (2000),
   [arXiv:astro-ph/9908023](https://arxiv.org/abs/astro-ph/9908023).
8. W. Zimdahl, D. Pavón and L. P. Chimento, “Interacting Quintessence,”
   *Phys. Lett. B* **521**, 133 (2001),
   [arXiv:astro-ph/0105479](https://arxiv.org/abs/astro-ph/0105479),
   [DOI](https://doi.org/10.1016/S0370-2693(01)01174-1).

## 10. 최종 route 판정

U2에는 Route A를 기본 골격으로 두되, total conserved charge와 symmetric
relic 제거는 Route B, absolute charge source는 Route C에서 가져오는 것이
가장 구조적으로 완전하다. U3에는 Route D가 ratio attractor를 실제 EOM으로
만든다. 다만 단순 exact-scaling law는 강한 expansion/growth 예측을 내므로
그대로 정본화하기 전에 kill test를 먼저 통과해야 한다.

full success의 가장 구체적인 다음 action은 `equal-energy charged precursor +
rolling charge source + conformally coupled dark scalar + action-defined transition`
이다. 여기서 우선 증명해야 할 한 문장은 다음이다.

> 동일 microscopic action이 $q_{\rm ext}$ charge share,
> $m_\chi/m_B=\xi(D)$, $r_{c/{\rm DE}}=\alpha_sD$, 그리고 유일한
> transition surface $\Sigma_*$를 관측 밀도 없이 동시에 유도한다.

이 문장이 성립하면 두 ratio와 flatness로 세 fraction이 닫히고, 별도의
$H_*$ lemma가 절대 density와 Hubble scale을 닫는다. 성립하지 않는 구성요소는
위 표의 대체 route로 교환해야 하며, 기존 정본의 목표를 성급히 내릴 이유는
없다.
