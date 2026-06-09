# 05f. CE Action/Topology Package

## 0. 목표

[05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)는 다음 조건부 정리를 닫았다.

$$
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
\text{ is l.s.c. good-rate}
\quad+\quad
\mu_{\mathrm{base}}\text{ has recovery mass}
\Longrightarrow
\mu_\beta\text{ concentrates on }\operatorname{argmin}W.
$$

남은 선택은 실제 CE가 어떤 경로공간 위상과 action package를 채택할지다. 기존 CE 문서의 A2는 \(C^1\) 위상을 쓰지만, kinetic action만으로는 \(C^1\)-compactness가 나오지 않는다. 따라서 이 문서는 CE 9_등호이전 수학 코어의 action/topology package를 고정한다.

결론:

> continuum 증명 코어는 \(W^{1,p}/C^0\) Tonelli package로 둔다. finite mesh는 계산/검증용 exact approximation으로 둔다. \(C^1\) pathspace는 acceleration/curvature penalty가 \(S_E\) 또는 \(S_{\mathrm{supp}}\)에 들어간 강화판에서만 쓴다.

현재 판정:

| package | 지위 | 용도 |
|---|---|---|
| finite mesh \(\mathcal P_{I,N}\) | `Exact under assumptions` | 코드, toy, 수치 검증 |
| \(W^{1,p}/C^0\) Tonelli | `Canonical continuum package` | CE 농축 정리의 기본 해석학 |
| \(C^1+\ddot\gamma\) penalty | `Stronger regularized package` | 기존 \(C^1\) 문구를 살리는 강화판 |
| bare \(C^1\)+kinetic action | `Rejected for compactness` | kinetic energy만으로 compact sublevel 실패 |

## 1. 선택 규칙

CE 경로공간은 두 층으로 분리한다.

### 1.1 수학 코어

continuum 농축 정리를 적용할 때는

$$
\mathcal P_I
=
W^{1,p}_{x_i,x_f}(I,M),
\qquad p>1
$$

를 기본 후보공간으로 둔다. topology/readout은 \(C^0\) 수렴을 기준으로 둔다.

직관:

- \(\gamma(t)\)의 위치 경로는 \(C^0\)로 읽는다.
- \(\dot\gamma\) 정보는 topology가 아니라 action \(S_E\) 안에 들어간다.
- endpoint, occupation, residual readout \(K_\phi(x,\gamma)\)는 보통 위치 경로에 의존하므로 \(C^0\)가 자연스럽다.

### 1.2 계산 코어

finite mesh에서는

$$
\mathcal P_{I,N}\simeq M^{N-1}
$$

를 쓴다. 이 층은 이미 finite PreEq 코드와 잘 맞는다.

### 1.3 \(C^1\) 강화판

기존 CE 문서의 \(C^1\) pathspace를 유지하려면

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

이 항 없이 \(C^1\)-compactness를 말하면 안 된다.

## 2. 왜 bare \(C^1\)은 실패하는가

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

만으로는 \(\dot\gamma\)의 균등수렴 subsequence를 보장하지 않는다.

### 반례 2.1

\(I=[0,2\pi]\), \(M=\mathbb R\), endpoint \(0\)을 고정하고

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

로 균일하게 bounded다. 하지만 \(\{\dot\gamma_n\}\)는 \(C^0\)에서 equicontinuous하지 않다. 실제로

$$
|\dot\gamma_n(t)-\dot\gamma_n(s)|
=
|\cos(nt)-\cos(ns)|
$$

는 \(|t-s|\)가 작아도 \(n\)이 커지면 크게 진동할 수 있다. 따라서 Arzela-Ascoli 조건이 실패하고 \(C^1\) 수렴 부분열을 보장할 수 없다.

결론:

$$
\{S_E\le c\}
\quad\text{is not compact in }C^1\text{ in general}.
$$

따라서 기존 \(C^1\) pathspace 문장은 그냥 쓰면 `Bridge`다.

## 3. Canonical continuum package: \(W^{1,p}/C^0\)

### 세팅

우선 \(M=\mathbb R^d\)이고 endpoint \(x_i,x_f\)가 고정되어 있다고 하자.

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

이고 \(L\)이 다음 Tonelli-type 조건을 만족한다고 하자.

1. \(L\)은 \((t,q,v)\)에 대해 Borel measurable이고 \(q,v\)에 대해 lower semicontinuous다.
2. \(v\mapsto L(t,q,v)\)는 convex다.
3. coercive growth:

$$
L(t,q,v)\ge a\|v\|^p-b,
\qquad a>0,\ b\ge0.
$$

또한 \(S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]\)는 \(C^0\)-l.s.c.라고 하자.

### 정리 3.1: \(C^0\)-compact sublevel

위 가정 아래에서 \(S_E\)의 sublevel은 \(C^0\) topology에서 precompact다. \(S_E\)가 \(C^0\)-l.s.c.이면 sublevel은 compact다.

### 증명

\(S_E[\gamma]\le C\)이면 coercive growth로

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

endpoint가 고정되어 있으므로 임의의 \(s,t\in I\)에 대해 Holder 부등식으로

$$
\|\gamma(t)-\gamma(s)\|
\le
\int_s^t\|\dot\gamma(r)\|dr
\le
\|\dot\gamma\|_{L^p}|t-s|^{1-1/p}.
$$

따라서 sublevel 안의 경로들은 균일한 Holder modulus를 가진다. 또한 endpoint 고정과 위 부등식으로 \(\|\gamma\|_{C^0}\)도 균일하게 bounded다. Arzela-Ascoli 정리에 의해 \(C^0\)-precompact하다.

추가로 \(S_E\)가 \(C^0\)-l.s.c.이면 \(\{S_E\le C\}\)는 \(C^0\)에서 닫혀 있다. precompact set의 닫힌 부분집합이므로 compact다. 끝.

### 정리 3.2: CE \(W\)의 good-rate

정리 3.1의 가정에 더해 \(S_E\)와 \(S_{\mathrm{supp}}\)가 \(C^0\)-l.s.c.이고 \(S_{\mathrm{supp}}\ge0\)이라고 하자. 그러면

$$
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
$$

는 \(C^0\) topology에서 l.s.c.이고 compact sublevel을 가진다.

증명:

l.s.c. 함수의 양의 상수배와 합은 l.s.c.다. 또한 \(S_{\mathrm{supp}}\ge0\)이므로

$$
W[\gamma]\le c
\Longrightarrow
S_E[\gamma]\le\hbar c.
$$

따라서 \(\{W\le c\}\subset\{S_E\le\hbar c\}\)이고, 오른쪽은 compact다. 왼쪽은 \(W\)의 l.s.c.로 닫혀 있으므로 compact다. 끝.

### 따름정리 3.3: CE 선택 농축

\(\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)\)가 모든 \(\eta>0\)에 대해

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

## 4. \(S_{\mathrm{supp}}\)의 허용 형태

이 문서에서 \(S_{\mathrm{supp}}\)는 "클라루스장/비선택 경로 효과를 통합한 추가 비용"이다. 수학 코어에서는 아래 조건만 요구한다.

$$
S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]
\quad\text{is }C^0\text{-l.s.c.}
$$

허용되는 예:

| 형태 | 예 | 조건 |
|---|---|---|
| endpoint penalty | \(G(\gamma(t_f))\) | \(G\) l.s.c. |
| occupation penalty | \(\int_I V(t,\gamma(t))dt\) | \(V\ge0\), l.s.c. |
| obstacle/fold penalty | \(\int_I \psi(d(\gamma(t),\mathcal O))dt\) | \(\psi\) l.s.c. |
| residual readout penalty | \(\int_I \rho(\phi(\gamma(t)))dt\) | \(\phi,\rho\) l.s.c., nonnegative |
| curvature/acceleration penalty | \(\lambda\int_I\|\nabla_t\dot\gamma\|^pdt\) | 강화판, \(W^{2,p}\) class 필요 |

주의:

- \(\phi\) readout 자체는 05a-05b의 pushforward 결과다.
- \(S_{\mathrm{supp}}\) 안에 \(\phi\)를 넣으면 CE residual이 다시 action에 feed back된다.
- 이 feed-back은 `Selection/Bridge`이고, 수학 정리는 \(S_{\mathrm{supp}}\)가 주어졌을 때만 작동한다.

## 5. finite mesh package

실험과 코드는 finite mesh를 기본으로 둔다.

시간격자

$$
t_0<t_1<\dots<t_N
$$

와 endpoint \(x_0=x_i\), \(x_N=x_f\)를 고정하면

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

| \(M\) | 조건 | 결론 |
|---|---|---|
| compact | \(W_N\) continuous | compact Gibbs 농축 |
| \(\mathbb R^d\) | \(W_N\) continuous/coercive | 02a noncompact 농축 |
| finite candidate subset | \(W_N\) arbitrary finite energy | 01 finite 농축 |

이 package는 `reality_stone.clarus.pre_eq`와 가장 직접적으로 연결된다.

## 6. \(C^1\) 강화 package

기존 CE A2의 \(C^1\) 거리

$$
d_I(\gamma_1,\gamma_2)
=
\sup_t d(\gamma_1(t),\gamma_2(t))
+\sup_t\|\dot\gamma_1(t)-\dot\gamma_2(t)\|
$$

를 유지하려면 action이 \(\dot\gamma\)의 equicontinuity까지 제어해야 한다.

충분조건:

1. admissible class \(\mathcal P_I\subset W^{2,p}(I,M)\), \(p>1\).
2. endpoint와 initial/final velocity 또는 velocity bound가 고정/제어된다.
3. action에 다음 하한이 있다.

$$
W[\gamma]\ge
a\|\dot\gamma\|_{L^\infty}^p
+b\int_I\|\nabla_t\dot\gamma(t)\|^pdt
-c.
$$

그러면 \(\dot\gamma\)는 \(W^{1,p}\) bounded family가 되고, 1차원 Sobolev compact embedding으로 \(\dot\gamma\)는 \(C^0\)에서 precompact하다. \(\gamma\)도 endpoint와 velocity bound로 \(C^0\)-precompact하므로 \(\gamma\)는 \(C^1\)-precompact하다.

판정:

`Exact under stronger action assumptions`

이 package를 택하면 \(S_{\mathrm{supp}}\)는 단순 비용이 아니라 고주파 경로/급격한 fold를 억제하는 regularizer 역할을 한다.

## 7. A2 공리 수정 제안

기존 A2:

> \(\mathcal P_I\)는 \(C^1\)-수렴에 해당하는 자연스러운 거리와 Borel 구조를 가진다.

05f 이후 권장 A2':

> \(\mathcal P_I\)는 기본적으로 \(W^{1,p}_{x_i,x_f}(I,M)\)의 admissible path class이며, manifest/readout topology는 \(C^0\) Borel 구조를 사용한다. 속도와 접힘 정보는 topology가 아니라 \(S_E\), \(S_{\mathrm{supp}}\), \(K_\phi\) 안에서 읽는다. \(C^1\) topology는 acceleration/curvature suppression이 action에 포함된 regularized variant에서만 사용한다.

이 수정은 이론을 약하게 만드는 것이 아니라, compactness 주장을 실제 정리 조건과 맞추는 것이다.

## 8. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| bare \(C^1\)+kinetic action의 compactness 실패 | 반례 2.1 |
| \(W^{1,p}/C^0\) sublevel precompactness | 정리 3.1 |
| \(W=S_E/\hbar+S_{\mathrm{supp}}\) good-rate | 정리 3.2 |
| CE 선택 농축 | 따름정리 3.3 |
| finite mesh package | 5절 |
| \(C^1\) regularized sufficient condition | 6절 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| \(M\)이 일반 Riemannian/Lorentzian manifold일 때 | chart/localization 또는 auxiliary Riemannian metric으로 \(W^{1,p}\) package 확장 |
| \(\mu_{\mathrm{base}}\)의 실제 구성 | reference/CE probability가 recovery mass를 주는지 정의 |
| \(S_{\mathrm{supp}}\) 물리 형태 | residual readout, curvature penalty, obstacle/fold penalty 중 선택 |
| continuum limit | finite mesh \(W_N\)이 continuum \(W\)로 Gamma 수렴하는지 검증 |

## 9. 결론

CE pathspace bridge의 기본 수학 package는 다음으로 고정한다.

$$
\boxed{
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,M),
\quad p>1,
\quad
\text{readout topology}=C^0.
}
$$

이 위에서 Tonelli-type \(S_E\)와 nonnegative l.s.c. \(S_{\mathrm{supp}}\)를 쓰면

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

다음 병목은 \(\mu_{\mathrm{base}}\)가 minimizer 근방에 실제로 positive mass를 주는지, 즉 reference/CE prior의 support를 닫는 일이다. 이 prior/support 조건은 [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md)에서 recovery mass package로 분리해 닫는다.
