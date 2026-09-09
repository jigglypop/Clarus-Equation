# 05f. CE Action/Topology Package

이 문서는 CE action의 good-rate 성질을 얻기 위한 topology·lower-semicontinuity·compact sublevel package를 비교한다. 닫힌 것은 명시한 함수공간과 coercivity 가정 아래의 수학이며, 실제 CE action과 물리 경로공간 선택은 별도 공리·미완성 입력이다.

독자는 05e의 추상 good-rate 정리를 먼저 읽는다. 수학·계산·$C^1$ route, no-go, continuum 및 mesh package, suppression과 남은 물리 선택을 순서대로 확인한다.

## 0. 목표

농축 정리를 CE에 적용하려면 action의 이름만으로는 부족하고 topology에 상대적인 compactness가 필요하다. 이 절은 충분조건과 실패 조건을 분리한다.

[05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)는 다음 조건부 정리를 닫았다.

$$
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
\text{ is l.s.c. good-rate}
\quad+\quad
\mu_{\mathrm{base}}\text{ has recovery mass}
\Longrightarrow
\mu_\beta\text{ concentrates on }\operatorname{argmin}W.
$$

남은 선택은 실제 CE가 어떤 경로공간 위상과 action package를 채택할지다. 기존 CE 문서의 A2는 $C^1$ 위상을 쓰지만, kinetic action만으로는 $C^1$-compactness가 나오지 않는다. 따라서 이 문서는 CE 9_등호이전 수학 코어의 action/topology package를 고정한다.

따라서

> continuum 증명 코어는 $W^{1,p}/C^0$ Tonelli package로 둔다. finite mesh는 유한차원 계산·검증층으로 둔다. $C^1$ pathspace는 acceleration/curvature penalty가 $S_E$ 또는 $S_{\mathrm{supp}}$에 들어간 강화판에서만 쓴다.

형식 출처:

| 항목 | 출처 | 용도 |
|---|---|---|
| finite mesh $\mathcal P_{I,N}$ | `[정리]` | 코드, toy, 수치 검증 |
| $W^{1,p}/C^0$ Tonelli | `[공리: 모델 선택]`; compactness는 `[정리]` | CE 농축 정리의 기본 해석학 |
| $C^1+\ddot\gamma$ penalty | `[공리: 모델 선택]`; 충분조건은 `[정리]` | $C^1$ 정칙성을 요구하는 강화판 |
| kinetic bound의 위상학적 함의 | `[정리]` kinetic bound만으로 $C^1$-compactness가 따르지 않음 | 정리 2.1 |

## 1. 선택 규칙

서로 다른 topology package는 동일한 결론을 자동으로 공유하지 않는다. 다음 선택 규칙은 정리 적용의 정의역을 고정하며 물리 prior를 결정하지 않는다.

CE 경로공간은 두 층으로 분리한다.

### 1.1 수학 코어

수학 코어는 최소한의 l.s.c.와 compact sublevel 조건을 사용한다. 이는 계산 해상도나 $C^1$ regularity를 보장하지 않는다.

continuum 농축 정리를 적용할 때는

$$
\mathcal P_I
=
W^{1,p}_{x_i,x_f}(I,M),
\qquad p>1
$$

를 기본 후보공간으로 둔다. topology/readout은 $C^0$ 수렴을 기준으로 둔다.

직관:

- $\gamma(t)$의 위치 경로는 $C^0$로 읽는다.
- $\dot\gamma$ 정보는 topology가 아니라 action $S_E$ 안에 들어간다.
- endpoint, occupation, residual readout $K_\phi(x,\gamma)$는 보통 위치 경로에 의존하므로 $C^0$가 자연스럽다.

### 1.2 계산 코어

계산 코어는 finite representation에서 검산 가능한 contract를 택한다. continuum limit에는 별도 mesh·temperature scaling이 필요하다.

finite mesh에서는

$$
\mathcal P_{I,N}\simeq M^{N-1}
$$

를 쓴다. 이 층은 이미 finite PreEq 코드와 잘 맞는다.

### 1.3 $C^1$ 강화판

$C^1$ 강화판은 더 강한 topology와 추가 coercivity를 요구한다. 약한 kinetic bound를 이 결론의 근거로 사용할 수 없다.

기존 CE 문서의 $C^1$ pathspace를 유지하려면

$$
S_{\mathrm{supp}}
\quad\text{또는}\quad
S_E
$$

가 acceleration/curvature를 제어해야 한다. 예:

$$
S_{\mathrm{reg}}[\gamma]
=
\lambda\int_I\|\nabla_t\dot\gamma(t)\|^pdt,
\qquad \lambda>0,\ p>1.
$$

이 항 없이 $C^1$-compactness를 말하면 안 된다.

## 2. kinetic bound의 $C^1$-compactness no-go

kinetic bound만으로 derivative의 균등 제어가 되지 않으면 $C^1$ precompactness는 실패한다. 다음 no-go는 stronger route의 필요성을 보이는 반례 경계다.

기존 CE 문서의 A2는 거리

$$
d_{C^1}(\gamma_1,\gamma_2)
=
\sup_t|\gamma_1(t)-\gamma_2(t)|
+\sup_t|\dot\gamma_1(t)-\dot\gamma_2(t)|
$$

를 사용한다. 그러나 kinetic action

$$
S_E[\gamma]=\int_I|\dot\gamma(t)|^2dt
$$

만으로는 $\dot\gamma$의 균등수렴 subsequence를 보장하지 않는다.

### 정리 2.1 (kinetic bound의 $C^1$-compactness no-go)

정리는 제시한 topology와 bound 아래에서만 적용된다. 이 결과는 $C^0$ 또는 Sobolev compactness를 부정하는 것이 아니다.

`[정리]` endpoint를 고정해도 kinetic action의 유계성만으로
그 sublevel은 $C^1$에서 일반적으로 compact하지 않다.

$I=[0,2\pi]$, $M=\mathbb R$, endpoint $0$을 고정하고

$$
\gamma_n(t)=\frac{\sin(nt)}{n}
$$

라 두자. 그러면

$$
\gamma_n(0)=\gamma_n(2\pi)=0
$$

이고

$$
\dot\gamma_n(t)=\cos(nt).
$$

kinetic action은

$$
\int_0^{2\pi}|\dot\gamma_n(t)|^2dt
=
\int_0^{2\pi}\cos^2(nt)dt
=
\pi
$$

로 균일하게 bounded다. 하지만 $\{\dot\gamma_n\}$는 $C^0$에서 equicontinuous하지 않다. 실제로

$$
|\dot\gamma_n(t)-\dot\gamma_n(s)|
=
|\cos(nt)-\cos(ns)|
$$

는 $|t-s|$가 작아도 $n$이 커지면 크게 진동할 수 있다. 따라서 Arzela-Ascoli 조건이 실패하고 $C^1$ 수렴 부분열을 보장할 수 없다.

결론:

$$
\{S_E\le c\}
\quad\text{is not compact in }C^1\text{ in general}.
$$

kinetic action만 둔 $C^1$ pathspace는 good-rate compactness 가정을
충족하지 않는다. $C^1$ 위상을 채택하려면 acceleration/curvature 제어를
`[공리: 모델 선택]`으로 추가하고 그 강화 조건 아래의 정리를 적용해야 한다.

## 3. Canonical continuum package: $W^{1,p}/C^0$

continuum route는 $W^{1,p}$ bound와 compact embedding을 통해 $C^0$ topology에서 sublevel을 포획한다. prior support와 boundary 조건은 별도로 맞춰야 한다.

### 세팅

세팅은 후보 경로의 정의역, endpoint 조건, action의 regularity를 고정한다. 실제 CE의 gauge·boundary 자료가 이 세팅을 만족한다는 주장은 포함하지 않는다.

우선 $M=\mathbb R^d$이고 endpoint $x_i,x_f$가 고정되어 있다고 하자.

$$
\mathcal P_I
=
W^{1,p}_{x_i,x_f}(I,\mathbb R^d),
\qquad p>1.
$$

action은

$$
S_E[\gamma]
=
\int_I L(t,\gamma(t),\dot\gamma(t))dt
$$

이고 $L$이 다음 Tonelli-type 조건을 만족한다고 하자.

1. $L$은 $(t,q,v)$에 대해 Borel measurable이고 $q,v$에 대해 lower semicontinuous다.
2. $v\mapsto L(t,q,v)$는 convex다.
3. coercive growth:

$$
L(t,q,v)\ge a\|v\|^p-b,
\qquad a>0,\ b\ge0.
$$

또한 $S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]$는 $C^0$-l.s.c.라고 하자.

### 정리 3.1: $C^0$-compact sublevel

정리는 명시한 coercivity와 embedding 가정으로 compactness를 얻는다. nonproper target이나 약한 성장 조건은 반례가 될 수 있다.

위 가정 아래에서 $S_E$의 sublevel은 $C^0$ topology에서 precompact다. $S_E$가 $C^0$-l.s.c.이면 sublevel은 compact다.

### 증명

증명은 sublevel bound를 함수공간 bound로 바꾸고 compact embedding을 적용한다. 이 단계의 topology를 바꾸면 같은 논증을 재사용할 수 없다.

$S_E[\gamma]\le C$이면 coercive growth로

$$
a\int_I\|\dot\gamma(t)\|^pdt-b|I|
\le C
$$

이므로

$$
\|\dot\gamma\|_{L^p}^p
\le
\frac{C+b|I|}{a}.
$$

endpoint가 고정되어 있으므로 임의의 $s,t\in I$에 대해 Holder 부등식으로

$$
\|\gamma(t)-\gamma(s)\|
\le
\int_s^t\|\dot\gamma(r)\|dr
\le
\|\dot\gamma\|_{L^p}|t-s|^{1-1/p}.
$$

따라서 sublevel 안의 경로들은 균일한 Holder modulus를 가진다. 또한 endpoint 고정과 위 부등식으로 $\|\gamma\|_{C^0}$도 균일하게 bounded다. Arzela-Ascoli 정리에 의해 $C^0$-precompact하다.

추가로 $S_E$가 $C^0$-l.s.c.이면 $\{S_E\le C\}$는 $C^0$에서 닫혀 있다. precompact set의 닫힌 부분집합이므로 compact다. 끝.

### 정리 3.2: CE $W$의 good-rate

good-rate 결론은 앞 compactness와 lower-semicontinuity를 결합한 형식 결과다. physical action의 정당화와는 구별된다.

정리 3.1의 가정에 더해 $S_E$와 $S_{\mathrm{supp}}$가 $C^0$-l.s.c.이고 $S_{\mathrm{supp}}\ge0$이라고 하자. 그러면

$$
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
$$

는 $C^0$ topology에서 l.s.c.이고 compact sublevel을 가진다.

증명:

l.s.c. 함수의 양의 상수배와 합은 l.s.c.다. 또한 $S_{\mathrm{supp}}\ge0$이므로

$$
W[\gamma]\le c
\Longrightarrow
S_E[\gamma]\le\hbar c.
$$

따라서 $\{W\le c\}\subset\{S_E\le\hbar c\}$이고, 오른쪽은 compact다. 왼쪽은 $W$의 l.s.c.로 닫혀 있으므로 compact다. 끝.

### 따름정리 3.3: CE 선택 농축

따름정리는 recovery prior가 추가될 때만 Gibbs 농축으로 넘어간다. prior 질량은 action regularity에서 자동으로 나오지 않는다.

$\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)$가 모든 $\eta>0$에 대해

$$
\mu_{\mathrm{base}}\big(\{\gamma:W[\gamma]<W_{\min}+\eta\}\big)>0
$$

를 만족하면

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta(W[\gamma]-W_{\min})}}{Z_\beta}
\mu_{\mathrm{base}}(d\gamma)
$$

는

$$
\operatorname*{argmin}_{\operatorname{supp}\mu_{\mathrm{base}}}W
$$

로 농축한다.

증명:

[05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 정리 1.1을 적용한다. 끝.

## 4. $S_{\mathrm{supp}}$의 허용 형태

suppression은 비음성·l.s.c. 같은 조건을 지킬 때 기존 compactness를 보존한다. 임의의 비국소·음의 항은 이 package의 범위 밖이다.

이 문서에서 $S_{\mathrm{supp}}$는 "클라루스장/비선택 경로 효과를 통합한 추가 비용"이다. 수학 코어에서는 아래 조건만 요구한다.

$$
S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]
\quad\text{is }C^0\text{-l.s.c.}
$$

허용되는 예:

| 형태 | 예 | 조건 |
|---|---|---|
| endpoint penalty | $G(\gamma(t_f))$ | $G$ l.s.c. |
| occupation penalty | $\int_I V(t,\gamma(t))dt$ | $V\ge0$, l.s.c. |
| obstacle/fold penalty | $\int_I \psi(d(\gamma(t),\mathcal O))dt$ | $\psi$ l.s.c. |
| residual readout penalty | $\int_I \rho(\phi(\gamma(t)))dt$ | $\phi,\rho$ l.s.c., nonnegative |
| curvature/acceleration penalty | $\lambda\int_I\|\nabla_t\dot\gamma\|^pdt$ | 강화판, $W^{2,p}$ class 필요 |

주의:

- $\phi$ readout 자체는 05a-05b의 pushforward 결과다.
- $S_{\mathrm{supp}}$ 안에 $\phi$를 넣으면 CE residual이 다시 action에 feed back된다.
- feedback functional의 채택은 `[공리: 모델 선택]`이고, residual과 물리 작용의 동일시는 `[공리: 물리 사상]`이다. 구체식이 없으면 `[미완성]`이며 수학 정리는 $S_{\mathrm{supp}}$가 주어졌을 때만 작동한다.

## 5. finite mesh package

finite mesh package는 유한차원 coercivity와 양의 prior로 직접 검산할 수 있다. continuum 물리 해석은 mesh 독립성 검증이 남는다.

실험과 코드는 finite mesh를 기본으로 둔다.

시간격자

$$
t_0<t_1<\dots<t_N
$$

와 endpoint $x_0=x_i$, $x_N=x_f$를 고정하면

$$
\mathcal P_{I,N}=M^{N-1}
$$

이다. discrete action은 예를 들어

$$
S_{E,N}(x_1,\dots,x_{N-1})
=
\sum_{k=0}^{N-1}
\Delta t_k\,
L\left(t_k,x_k,\frac{x_{k+1}-x_k}{\Delta t_k}\right)
$$

이다.

닫힘:

| $M$ | 조건 | 결론 |
|---|---|---|
| compact | $W_N$ continuous | compact Gibbs 농축 |
| $\mathbb R^d$ | $W_N$ continuous/coercive | 02a noncompact 농축 |
| finite candidate subset | $W_N$ arbitrary finite energy | 01 finite 농축 |

이 package는 `reality_stone.clarus.pre_eq`와 가장 직접적으로 연결된다.

## 6. $C^1$ 강화 package

강화 package는 derivative equicontinuity를 제공하는 별도 조건을 요구한다. $W^{1,p}/C^0$ 정리보다 강한 주장을 추가 근거 없이 얻지 않는다.

기존 CE A2의 $C^1$ 거리

$$
d_I(\gamma_1,\gamma_2)
=
\sup_t d(\gamma_1(t),\gamma_2(t))
+\sup_t\|\dot\gamma_1(t)-\dot\gamma_2(t)\|
$$

를 유지하려면 action이 $\dot\gamma$의 equicontinuity까지 제어해야 한다.

충분조건:

1. admissible class $\mathcal P_I\subset W^{2,p}(I,M)$, $p>1$.
2. endpoint와 initial/final velocity 또는 velocity bound가 고정/제어된다.
3. action에 다음 하한이 있다.

$$
W[\gamma]\ge
a\|\dot\gamma\|_{L^\infty}^p
+b\int_I\|\nabla_t\dot\gamma(t)\|^pdt
-c.
$$

그러면 $\dot\gamma$는 $W^{1,p}$ bounded family가 되고, 1차원 Sobolev compact embedding으로 $\dot\gamma$는 $C^0$에서 precompact하다. $\gamma$도 endpoint와 velocity bound로 $C^0$-precompact하므로 $\gamma$는 $C^1$-precompact하다.

출처:

강화된 action 가정 아래 `[정리]`

이 package를 택하면 $S_{\mathrm{supp}}$는 단순 비용이 아니라 고주파 경로/급격한 fold를 억제하는 regularizer 역할을 한다.

## 7. A2 공리 수정 제안

수정 제안은 형식 package의 요구를 CE 공리 문장에 반영하는 모델 선택이다. 공리 채택이 empirical validation이나 action 유도를 대체하지 않는다.

기존 A2:

> $\mathcal P_I$는 $C^1$-수렴에 해당하는 자연스러운 거리와 Borel 구조를 가진다.

05f 이후 권장 A2':

> $\mathcal P_I$는 기본적으로 $W^{1,p}_{x_i,x_f}(I,M)$의 admissible path class이며, manifest/readout topology는 $C^0$ Borel 구조를 사용한다. 속도와 접힘 정보는 topology가 아니라 $S_E$, $S_{\mathrm{supp}}$, $K_\phi$ 안에서 읽는다. $C^1$ topology는 acceleration/curvature suppression이 action에 포함된 regularized variant에서만 사용한다.

이 수정은 이론을 약하게 만드는 것이 아니라, compactness 주장을 실제 정리 조건과 맞추는 것이다.

## 8. 닫힌 것과 남은 것

닫힌 compactness 정리와 실제 CE pathspace·action·prior의 미완성 선택을 분리한다. 표의 충분조건은 필요충분 분류가 아니다.

닫힌 것:

| 항목 | 상태 |
|---|---|
| kinetic bound만으로 $C^1$-compactness가 따르지 않음 | `[정리]` 2.1 |
| $W^{1,p}/C^0$ sublevel precompactness | 정리 3.1 |
| $W=S_E/\hbar+S_{\mathrm{supp}}$ good-rate | 정리 3.2 |
| CE 선택 농축 | 따름정리 3.3 |
| finite mesh package | 5절 |
| $C^1$ regularized sufficient condition | 6절 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| $M$이 일반 Riemannian/Lorentzian manifold일 때 | chart/localization 또는 auxiliary Riemannian metric으로 $W^{1,p}$ package 확장 |
| $\mu_{\mathrm{base}}$의 실제 구성 | reference/CE probability가 recovery mass를 주는지 정의 |
| $S_{\mathrm{supp}}$ 물리 형태 | residual readout, curvature penalty, obstacle/fold penalty 중 선택 |
| continuum limit | finite mesh $W_N$이 continuum $W$로 Gamma 수렴하는지 검증 |

## 9. 결론

결론적으로 topology와 coercivity를 명시해야 good-rate 농축의 정의역이 정해진다. 물리적 CE 적용은 이 package를 만족하는 실제 입력을 추가로 제시해야 한다.

CE pathspace bridge의 기본 수학 package는 다음으로 고정한다.

$$
\boxed{
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,M),
\quad p>1,
\quad
\text{readout topology}=C^0.
}
$$

이 위에서 Tonelli-type $S_E$와 nonnegative l.s.c. $S_{\mathrm{supp}}$를 쓰면

$$
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
$$

는 good-rate가 되고, [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 농축 정리가 적용된다.

따라서 CE의 선택/비선택 경로 bridge는 이제 다음 모양으로 닫힌다.

$$
\boxed{
W^{1,p}/C^0
+\text{ Tonelli action}
+S_{\mathrm{supp}}\ge0\text{ l.s.c.}
\Longrightarrow
\text{manifest path concentration}.
}
$$

다음 병목은 $\mu_{\mathrm{base}}$가 minimizer 근방에 실제로 positive mass를 주는지, 즉 reference/CE prior의 support를 닫는 일이다. 이 prior/support 조건은 [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md)에서 recovery mass package로 분리해 닫는다.
