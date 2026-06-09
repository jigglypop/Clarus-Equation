# 05i. CE Physical Path Prior Package

## 0. 목표

[05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md)까지의 결과는 다음이었다.

$$
W^{1,p}/C^0
+\text{ good-rate }W
+\text{ recovery prior}
+\text{ finite-to-continuum scale}
\Longrightarrow
\text{manifest path concentration}.
$$

남은 병목은 recovery prior 자리에 들어갈 **물리적 continuum prior**의 실제 선택이었다. 이 문서는 후보를 Brownian bridge, Gaussian/Sobolev, dense atomic으로 분류하고, 각 후보가 \(W^{1,p}/C^0\) 경로공간에서 어떤 support와 recovery를 주는지 닫는다.

핵심 결론:

> raw kinetic action \(S_E/\hbar\)를 density로 두고 Brownian bridge를 prior로 쓰면 recovery mass는 **실패**한다. Brownian path는 a.s. \(W^{1,1}\) 바깥이라 \(W=\infty\) a.s.이기 때문이다. 닫히는 길은 두 개다. kinetic 항을 prior 안으로 흡수하는 Wiener packaging(Route W), 그리고 \(\sqrt\hbar\)-scaled bridge의 LDP rate로 kinetic 항을 받는 Schilder packaging(Route S)이다. Route S에서는 LDP rate가 정확히 \(S_E-\min S_E\), 즉 CE의 \(E_{\mathrm{fold}}\) kinetic part로 나온다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| Brownian bridge의 \(C^0\) full support | `Exact under assumptions` | Gaussian support 정리 import, 정리 3.1 |
| raw kinetic density + Brownian prior recovery | `False` | 장애물 정리 2.1, 따름정리 2.2 |
| Route W: Wiener packaging recovery | `Exact under assumptions` | 정리 3.2 |
| Route W만으로 zero-temperature 농축 | `False in general` | 주의 3.3 |
| Route S: scaled bridge LDP rate \(=S_E-\min S_E\) | `Exact under assumptions` | 정리 4.2 |
| bounded \(S_{\mathrm{supp}}\)는 manifest set을 못 바꿈 | `Exact` | 정리 5.1 |
| \(\beta\)-coupled \(S_{\mathrm{supp}}\) 농축 | `Exact under assumptions` | 정리 5.2 |
| manifold target, unbounded potential | `Open` | 8절 |

## 1. 후보 prior 분류

경로공간은 [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)를 따라

$$
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,\mathbb R^d),
\qquad
I=[0,1],
\qquad
\text{readout topology}=C^0
$$

로 두고, 이 문서에서는 target을 \(\mathbb R^d\)로 제한한다. readout 공간은

$$
\Gamma
=
C^0_{x_i,x_f}
=
\{\gamma\in C^0(I,\mathbb R^d):\gamma(0)=x_i,\ \gamma(1)=x_f\}
$$

이다. kinetic action은

$$
S_E[\gamma]
=
\frac12\int_0^1|\dot\gamma(t)|^2dt
\qquad(\gamma\in H^1),
\qquad
S_E[\gamma]=\infty
\quad(\gamma\in C^0\setminus H^1).
$$

후보 prior:

| 후보 | endpoint 조건 | support | 비고 |
|---|---|---|---|
| Brownian bridge \(\mathbb B_{x_i,x_f}\) | 자동 | \(C^0_{x_i,x_f}\) 전체 | 정리 3.1 |
| Gaussian on \(H^1\) (Cameron-Martin \(H^1_{0,0}\)) | 자동 | \(C^0_{x_i,x_f}\)의 닫힘 | CM 공간 자체는 null set |
| dense atomic \(\sum2^{-n}\delta_{\gamma_n}\) | 선택 가능 | full support 가능 | 05g 정리 5.1, `Tooling` |
| finite mesh pushforward \((\iota_N)_*\mu_N\) | 자동 | finite | 05h package |

dense atomic prior는 수학적 proof device이지 물리적 \(\mathcal D\gamma\) 후보가 아니다. 따라서 물리 후보는 Brownian bridge 계열로 좁혀진다.

## 2. 장애물 정리: kinetic density는 Brownian prior에서 a.s. 무한

CE 문서의 형식적 표기

$$
d\mu_{\mathrm{CE}}
\propto
e^{-S_E[\gamma]/\hbar-S_{\mathrm{supp}}[\gamma]}\,\mathcal D\gamma
$$

를 \(\mathcal D\gamma=\mathbb B_{x_i,x_f}\)로 읽고 \(S_E/\hbar\)를 density에 남겨두면 어떻게 되는지 먼저 확인한다.

### 정리 2.1: \(\mathbb B_{x_i,x_f}(W^{1,p})=0\)

\(\mathbb B=\mathbb B_{x_i,x_f}\)를 \([0,1]\) 위 Brownian bridge의 law라고 하자. 모든 \(p\ge1\)에 대해

$$
\mathbb B\big(W^{1,p}(I,\mathbb R^d)\big)=0.
$$

증명:

bounded interval에서 \(W^{1,p}\subset W^{1,1}\)이므로 \(p=1\)만 보이면 된다.

(i) \(\gamma\in W^{1,1}\)이면 \(\gamma\)는 absolutely continuous이고 total variation \(V(\gamma)=\int_0^1|\dot\gamma|dt<\infty\)이다. partition \(0=t_0<\dots<t_n=1\)에 대해

$$
\sum_k|\gamma(t_{k+1})-\gamma(t_k)|^2
\le
\max_k|\gamma(t_{k+1})-\gamma(t_k)|
\cdot
\sum_k|\gamma(t_{k+1})-\gamma(t_k)|
\le
\omega_\gamma(\mathrm{mesh})\cdot V(\gamma).
$$

\(\gamma\)는 \([0,1]\)에서 uniformly continuous이므로 mesh가 0으로 가면 우변이 0으로 간다. 따라서 \(W^{1,1}\) path의 quadratic variation은 0이다.

(ii) Brownian bridge는 dyadic partition 열을 따라 a.s.

$$
\sum_k|\gamma(t_{k+1})-\gamma(t_k)|^2
\to d
$$

를 만족한다. 이는 Brownian motion의 quadratic variation 정리의 bridge 버전으로 표준 결과다(외부 import).

(i)과 (ii)는 양립할 수 없으므로 \(\mathbb B(W^{1,1})=0\)이다. 끝.

### 따름정리 2.2: raw kinetic recovery 실패

\(\mu_{\mathrm{base}}=\mathbb B_{x_i,x_f}\)이고

$$
W[\gamma]\ge S_E[\gamma]/\hbar
$$

라고 하자(\(S_{\mathrm{supp}}\ge0\)). 그러면

1. \(\{W<\infty\}\subset H^1\)이므로 \(\mu_{\mathrm{base}}\{W<\infty\}=0\).
2. \(Z_\beta=\int e^{-\beta W}d\mu_{\mathrm{base}}=0\)이라 Gibbs 측도 자체가 정의되지 않는다.
3. \(\operatorname{supp}\mu_{\mathrm{base}}=C^0_{x_i,x_f}\) 안에 \(H^1\) path가 있어 \(W_{\min}<\infty\)인데도

$$
\mu_{\mathrm{base}}\{\gamma:W[\gamma]<W_{\min}+\eta\}
\le
\mu_{\mathrm{base}}(H^1)
=0.
$$

즉 [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md)의 recovery mass가 실패한다.

해석:

> 이는 05g 반례 2.1의 물리 버전이다. full support는 성립하지만 near-minimum set이 prior null set이다. Feynman의 \(e^{-S_E/\hbar}\mathcal D\gamma\)가 측도 곱 density로 읽히지 않는 고전적 이유가 recovery mass 언어로 정확히 재현된다. kinetic 항은 density가 아니라 **측도 안**에 살아야 한다.

## 3. Route W: Wiener packaging

kinetic 항을 prior로 흡수한다.

$$
\mu_{\mathrm{base}}:=\mathbb B_{x_i,x_f},
\qquad
W:=S_{\mathrm{supp}}.
$$

### 정리 3.1: Brownian bridge full support

\(\mathbb B_{x_i,x_f}\)는 \(C^0_{x_i,x_f}\)에서 full support다.

증명:

\(\mathbb B_{x_i,x_f}\)는 \(C^0_{x_i,x_f}\) 위 Gaussian measure로, 평균은 직선 경로

$$
\gamma_{\mathrm{lin}}(t)=x_i+t(x_f-x_i)
$$

이고 Cameron-Martin 공간은

$$
H^1_{0,0}
=
\{h\in H^1(I,\mathbb R^d):h(0)=h(1)=0\},
\qquad
\|h\|_{\mathrm{CM}}^2=\int_0^1|\dot h|^2dt
$$

이다. Gaussian measure의 support는 평균 더하기 Cameron-Martin 공간의 닫힘이다(외부 import). \(H^1_{0,0}\)은 \(C^0_{0,0}\)에서 dense이므로

$$
\operatorname{supp}\mathbb B_{x_i,x_f}
=
\gamma_{\mathrm{lin}}+\overline{H^1_{0,0}}^{\,C^0}
=
C^0_{x_i,x_f}.
$$

끝.

### 정리 3.2: Route W recovery와 fixed-\(\beta\) CE 측도

\(S_{\mathrm{supp}}:C^0_{x_i,x_f}\to[0,\infty]\)가 어떤 minimizer에서 continuous이면(예: \(S_{\mathrm{supp}}[\gamma]=\int_0^1V(\gamma(t))dt\), \(V\ge0\) continuous) recovery mass가 성립하고, fixed \(\beta\)에서

$$
d\mu_{\mathrm{CE}}
=
\frac{e^{-\beta S_{\mathrm{supp}}}}{Z_\beta}
d\mathbb B_{x_i,x_f}
$$

는 well-defined probability다.

증명:

정리 3.1의 full support와 minimizer continuity에 05g 정리 3.3을 적용하면 recovery mass가 나온다. recovery mass는 \(Z_\beta>0\)을 주고, \(S_{\mathrm{supp}}\ge0\)이므로 \(Z_\beta\le1<\infty\)다. 끝.

이는 Euclidean Feynman-Kac packaging과 동일하다. 05d에서 요구한 \(0<\int e^{-W}d\mu_{\mathrm{ref}}<\infty\)가 이 route에서 닫힌다.

### 주의 3.3: Route W만으로는 zero-temperature 농축이 안 닫힌다

\(E=S_{\mathrm{supp}}\)는 일반적으로 \(C^0\)에서 good rate function이 아니다.

반례: \(V\equiv0\)이면 \(S_{\mathrm{supp}}\equiv0\)이고 sublevel set이 \(C^0_{x_i,x_f}\) 전체라 compact가 아니다. \(V\)가 bounded여도 sublevel set은 진폭이 큰 진동 경로를 모두 포함해 compact가 아니다.

따라서 [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 정리를 \(E=S_{\mathrm{supp}}\)에 적용할 수 없다. 물리적으로도 fixed Brownian prior에서 \(\beta\to\infty\)는 potential만 조이는 극한이지 semiclassical 극한이 아니다. kinetic 선택 정보가 빠진다. 이를 닫는 것이 Route S다.

## 4. Route S: scaled bridge와 Schilder packaging

semiclassical 파라미터를 prior에 직접 넣는다.

$$
\mu^\varepsilon
:=
\text{law of }\ \gamma_{\mathrm{lin}}+\sqrt\varepsilon\,B^0,
\qquad
\varepsilon=\hbar,
$$

여기서 \(B^0\)는 \(0\)에서 \(0\)으로 가는 standard Brownian bridge다. 각 \(\mu^\varepsilon\)은 endpoint 조건을 자동으로 만족하고 정리 3.1과 같은 이유로 \(C^0_{x_i,x_f}\) full support다.

### 보조정리 4.1: kinetic sublevel의 compactness와 l.s.c.

\(J_0:C^0_{x_i,x_f}\to[0,\infty]\)를

$$
J_0[\gamma]=S_E[\gamma]
$$

로 두면 \(J_0\)는 \(C^0\)에서 l.s.c.이고 모든 sublevel \(\{J_0\le c\}\)는 compact다.

증명:

(compactness) \(\gamma\in\{J_0\le c\}\)이면 Cauchy-Schwarz로

$$
|\gamma(t)-\gamma(s)|
\le
\Big(\int_s^t|\dot\gamma|^2\Big)^{1/2}|t-s|^{1/2}
\le
\sqrt{2c}\,|t-s|^{1/2}.
$$

equi-Hölder이고 \(|\gamma(t)|\le|x_i|+\sqrt{2c}\)로 균등유계이므로 Arzelà-Ascoli에 의해 \(C^0\) precompact다.

(l.s.c.) \(\gamma_n\to\gamma\) uniformly이고 \(\liminf J_0[\gamma_n]=c<\infty\)라 하자. 부분열에서 \(\dot\gamma_n\)이 \(L^2\) bounded이므로 weak limit \(v\in L^2\)를 갖는다(Banach-Alaoglu, 외부 import). \(\gamma_n(t)=x_i+\int_0^t\dot\gamma_n\)에서 극한을 취하면 \(\gamma(t)=x_i+\int_0^tv\), 즉 \(\gamma\in H^1\)이고 \(\dot\gamma=v\)다. norm의 weak l.s.c.로

$$
J_0[\gamma]=\tfrac12\|v\|_{L^2}^2
\le
\liminf_n\tfrac12\|\dot\gamma_n\|_{L^2}^2
=c.
$$

따라서 sublevel은 닫혀 있고, precompact와 합쳐 compact다. 끝.

### 정리 4.2: scaled bridge LDP와 rate의 \(E_{\mathrm{fold}}\) 형태

\(\{\mu^\varepsilon\}\)은 \(C^0_{x_i,x_f}\)에서 good rate function

$$
J[\gamma]
=
\frac12\int_0^1|\dot\gamma-\dot\gamma_{\mathrm{lin}}|^2dt
\quad(\gamma\in H^1_{x_i,x_f}),
\qquad
J=\infty\ \text{otherwise}
$$

를 갖는 large deviation principle을 만족한다. 또한

$$
J[\gamma]
=
S_E[\gamma]-S_E[\gamma_{\mathrm{lin}}]
=
S_E[\gamma]-\min_{H^1_{x_i,x_f}}S_E.
$$

증명:

(LDP) centered Gaussian measure의 \(\sqrt\varepsilon\)-scaling은 Cameron-Martin norm 제곱의 절반을 rate로 갖는 LDP를 만족한다(generalized Schilder, 외부 import). \(B^0\)의 CM norm은 \(\|\dot h\|_{L^2}\)이므로 \(\sqrt\varepsilon B^0\)의 rate는 \(\frac12\int|\dot h|^2\) (\(h\in H^1_{0,0}\))이고, 상수 shift \(\gamma_{\mathrm{lin}}\)는 contraction principle로 rate를 \(J[\gamma]=\frac12\int|\dot\gamma-\dot\gamma_{\mathrm{lin}}|^2\)로 옮긴다. goodness는 보조정리 4.1과 같은 Arzelà-Ascoli 논리로 성립한다.

(rate 항등식) \(\gamma\in H^1_{x_i,x_f}\)이면 \(\dot\gamma_{\mathrm{lin}}\equiv v:=x_f-x_i\)이고 \(\int_0^1\dot\gamma=x_f-x_i=v\)이므로

$$
J[\gamma]
=
\frac12\int|\dot\gamma|^2
-\int\dot\gamma\cdot v
+\frac12|v|^2
=
S_E[\gamma]-\frac12|v|^2.
$$

Jensen 부등식으로 \(S_E[\gamma]\ge\frac12|\int\dot\gamma|^2=\frac12|v|^2=S_E[\gamma_{\mathrm{lin}}]\)이고 등호는 \(\dot\gamma\) 상수, 즉 \(\gamma=\gamma_{\mathrm{lin}}\)일 때다. 따라서 \(\frac12|v|^2=\min S_E\)이고 항등식이 성립한다. 끝.

해석:

> prior 자체의 LDP rate가 정확히 \(S_E-\min S_E\)다. 이는 CE 문서의 \(E_{\mathrm{fold}}=W-W_{\min}\) 정규화가 kinetic part에서 인위적 선택이 아니라 scaled Brownian prior의 표준 rate로 자동으로 나온다는 뜻이다.

또한 LDP lower bound는 open ball에 대해

$$
\liminf_{\varepsilon\to0}
\varepsilon\log\mu^\varepsilon\big(B(\gamma,\delta)\big)
\ge
-\inf_{B(\gamma,\delta)}J
$$

를 주므로, [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md)의 scaled recovery mass 조건

$$
\frac1{\beta_\varepsilon}\log\frac1{\mu^\varepsilon(B_\eta)}\to0
\qquad(\beta_\varepsilon=1/\varepsilon)
$$

이 minimizer 근방에서 **정리로** 닫힌다. 05g에서 공리로 남겨야 했던 recovery가 Route S에서는 prior의 구조에서 나온다.

## 5. \(S_{\mathrm{supp}}\)의 scaling 선택

CE 문서의 \(W=S_E/\hbar+S_{\mathrm{supp}}\)에서 \(S_{\mathrm{supp}}\)에 \(1/\hbar\)가 붙는지 여부가 manifest set을 바꾼다. Route S에서 이 갈림길이 정리로 분리된다.

### 정리 5.1: bounded tilt는 manifest set을 바꾸지 못한다

\(S_{\mathrm{supp}}:C^0_{x_i,x_f}\to[0,C]\)가 bounded measurable이고

$$
d\nu^\varepsilon
=
\frac{e^{-S_{\mathrm{supp}}}}{Z^\varepsilon}d\mu^\varepsilon
$$

라고 하자. 그러면 \(\{\nu^\varepsilon\}\)은 \(\{\mu^\varepsilon\}\)과 **같은** rate \(J\)의 LDP를 만족하고, 따라서 \(\varepsilon\to0\)에서 \(\operatorname*{argmin}J=\{\gamma_{\mathrm{lin}}\}\) 근방으로 농축한다.

증명:

\(e^{-C}\le e^{-S_{\mathrm{supp}}}\le1\)이므로 \(Z^\varepsilon\in[e^{-C},1]\)이고 모든 Borel \(A\)에 대해

$$
e^{-C}\mu^\varepsilon(A)
\le
\nu^\varepsilon(A)
\le
e^{C}\mu^\varepsilon(A).
$$

양변에 \(\varepsilon\log\)를 취하면 상수항이 사라지므로 \(\nu^\varepsilon\)의 \(\varepsilon\log\) 점근은 \(\mu^\varepsilon\)과 동일하다. LDP rate가 같으므로 good rate \(J\)의 유일 minimizer \(\gamma_{\mathrm{lin}}\)(정리 4.2의 Jensen 등호 조건) 근방으로 농축한다. 끝.

해석:

> \(\hbar\)-bounded suppression은 finite-\(\varepsilon\) weight만 바꾸고 manifest 극한에는 흔적을 남기지 못한다. CE가 \(S_{\mathrm{supp}}\)로 선택 결과를 바꾸려면 \(S_{\mathrm{supp}}\)가 \(1/\hbar\) scale로 결합하거나(정리 5.2) \(\{0,\infty\}\)값 hard constraint여야 한다.

### 정리 5.2: \(\beta\)-coupled tilt 농축

\(S_{\mathrm{supp}}:C^0_{x_i,x_f}\to[0,\infty)\)가 continuous bounded라고 하자.

$$
d\nu^\varepsilon
=
\frac{e^{-S_{\mathrm{supp}}/\varepsilon}}{Z^\varepsilon}d\mu^\varepsilon
$$

로 두면 \(\{\nu^\varepsilon\}\)은 good rate

$$
K[\gamma]=J[\gamma]+S_{\mathrm{supp}}[\gamma]-m_*,
\qquad
m_*=\inf_{C^0_{x_i,x_f}}\big(J+S_{\mathrm{supp}}\big)
$$

의 LDP를 만족하고, 모든 open \(U\supset M_*:=\operatorname*{argmin}(J+S_{\mathrm{supp}})\)에 대해

$$
\nu^\varepsilon(U)\to1
\qquad(\varepsilon\to0).
$$

증명:

bounded continuous tilt에 대한 tilted LDP(Varadhan, 외부 import)로 \(\nu^\varepsilon\)은 rate \(K\)의 LDP를 만족한다. \(K\)의 goodness: \(S_{\mathrm{supp}}\ge0\)이므로 \(\{J+S_{\mathrm{supp}}\le c\}\subset\{J\le c\}\)이고 후자는 compact(정리 4.2), 전자는 l.s.c. 합의 sublevel이라 닫혀 있어 compact다. minimizer 존재는 05e 정리 1.1의 첫 단락과 같다.

농축: \(F=C^0_{x_i,x_f}\setminus U\)는 닫힌집합이고 05e의 gap 논리로

$$
\delta_U:=\inf_F\big(J+S_{\mathrm{supp}}\big)-m_*>0
$$

이다. LDP upper bound로

$$
\limsup_{\varepsilon\to0}\varepsilon\log\nu^\varepsilon(F)
\le
-\inf_FK
\le
-\delta_U<0,
$$

따라서 \(\nu^\varepsilon(F)\to0\)이다. 끝.

해석:

> 이것이 CE \(W=S_E/\hbar+S_{\mathrm{supp}}\)의 올바른 Route S 독법이다. 농축 대상은
>
> $$
> \operatorname*{argmin}\big(S_E+S_{\mathrm{supp}}^{\mathrm{phys}}\big),
> \qquad
> S_{\mathrm{supp}}^{\mathrm{phys}}:=\varepsilon S_{\mathrm{supp}}/\varepsilon
> $$
>
> 즉 kinetic과 suppression이 **같은 \(1/\hbar\) scale**로 경쟁할 때만 둘 다 선택에 참여한다.

## 6. 05e/05g/05h 조건과의 매핑

Route S package가 이전 문서의 가정을 어디까지 정리로 바꾸는지 정리한다.

| 이전 문서의 가정 | Route S에서의 지위 |
|---|---|
| 05e: \(W\) good rate | 정리 4.2 + 정리 5.2의 \(K\) goodness로 닫힘 |
| 05e: recovery mass | LDP lower bound로 닫힘 (공리에서 정리로 강등) |
| 05f: \(W^{1,p}/C^0\) topology | rate의 유효 domain이 \(H^1_{x_i,x_f}\), readout이 \(C^0\)로 일치 |
| 05g A2'' prior/support axiom | 정리 3.1 full support + LDP lower bound로 닫힘 |
| 05h A3 scaled recovery mass | \(\beta_\varepsilon=1/\varepsilon\)에서 LDP lower bound와 동일 |
| 05h finite mesh consistency | random walk bridge의 invariance principle(Donsker, 외부 import)로 후보 존재, 세부 검증은 `Open` |

## 7. 권장 A2''' physical prior 공리

05f의 A2'(action/topology), 05g의 A2''(prior/support)에 이어 다음을 둔다.

> **A2''' physical path prior axiom.**
> CE continuum prior는 scaled Brownian bridge family
> \[
> \mu^\hbar=\text{law of }\gamma_{\mathrm{lin}}+\sqrt\hbar\,B^0
> \]
> 로 둔다. 이때
>
> 1. endpoint 조건 \(\gamma(0)=x_i,\gamma(1)=x_f\)는 construction으로 성립한다.
> 2. \(\mu^\hbar\)는 \(C^0_{x_i,x_f}\) full support이고 good rate \(J=S_E-\min S_E\)의 LDP를 만족한다.
> 3. 선택에 참여하는 suppression은 \(e^{-S_{\mathrm{supp}}/\hbar}\)로 결합한다. \(\hbar\)-bounded suppression은 finite-\(\hbar\) reweighting으로만 쓴다.
> 4. fixed-\(\hbar\) CE probability는 Route W packaging \(d\mu_{\mathrm{CE}}\propto e^{-S_{\mathrm{supp}}/\hbar}d\mu^\hbar\)로 정의한다.

이 공리 아래에서 manifest 극한은

$$
\nu^\hbar
\Longrightarrow
\delta_{\gamma_*},
\qquad
\gamma_*
=
\operatorname*{argmin}\big(S_E+S_{\mathrm{supp}}\big)
$$

(유일 minimizer일 때)로 닫힌다.

## 8. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| \(\mathbb B(W^{1,p})=0\) 장애물 | 정리 2.1 |
| raw kinetic density recovery 실패 | 따름정리 2.2 |
| Brownian bridge full support | 정리 3.1 |
| Route W fixed-\(\beta\) CE 측도 존재 | 정리 3.2 |
| \(S_{\mathrm{supp}}\) 단독 good-rate 실패 | 주의 3.3 |
| kinetic sublevel compactness/l.s.c. | 보조정리 4.1 |
| scaled bridge LDP, rate \(=S_E-\min S_E\) | 정리 4.2 |
| bounded tilt의 manifest 불변 | 정리 5.1 |
| \(\beta\)-coupled tilt 농축 | 정리 5.2 |

외부 import로 쓴 표준 정리:

| import | 사용 위치 |
|---|---|
| Brownian quadratic variation | 정리 2.1 (ii) |
| Gaussian support 정리 | 정리 3.1 |
| Banach-Alaoglu와 norm weak l.s.c. | 보조정리 4.1 |
| generalized Schilder LDP | 정리 4.2 |
| Varadhan tilted LDP | 정리 5.2 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| CE 문서군의 \(S_{\mathrm{supp}}\) scaling 규약 | 실제 CE 문서에서 \(S_{\mathrm{supp}}\)가 bounded tilt인지 \(\beta\)-coupled인지 감사 (05d 후속) |
| unbounded potential | \(V\) unbounded일 때 Varadhan moment 조건 또는 exponential tightness 확인 |
| hard constraint suppression | \(S_{\mathrm{supp}}\in\{0,\infty\}\) 조건부 bridge의 recovery 확인 |
| manifold target \(M\) | \(\mathbb R^d\)를 Riemannian manifold로 올리는 Brownian bridge 구성 |
| finite mesh consistency 세부 | random walk bridge가 05h A3의 outer gap/lower consistency를 만족하는지 |

## 9. 결론

CE physical path prior는 다음으로 고정한다.

$$
\boxed{
\mu^\hbar=\text{law of }\gamma_{\mathrm{lin}}+\sqrt\hbar\,B^0,
\qquad
d\nu^\hbar\propto e^{-S_{\mathrm{supp}}/\hbar}d\mu^\hbar.
}
$$

이 선택 아래에서

$$
\boxed{
\text{full support}
+\text{LDP rate }S_E-\min S_E
+\beta\text{-coupled }S_{\mathrm{supp}}
\Longrightarrow
\nu^\hbar\ \text{concentrates on}\
\operatorname*{argmin}(S_E+S_{\mathrm{supp}}).
}
$$

05g에서 공리였던 recovery mass와 05h에서 공리였던 scaled recovery mass가 이 prior에서는 LDP lower bound 정리로 내려온다. 남은 병목은 실제 CE 문서가 \(S_{\mathrm{supp}}\)를 bounded tilt로 쓰는지 \(\beta\)-coupled로 쓰는지의 규약 감사다.
