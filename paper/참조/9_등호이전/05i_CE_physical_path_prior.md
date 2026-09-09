# 05i. CE Physical Path Prior Package

이 문서는 CE path integral에 쓸 수 있는 수학적 reference prior를 Brownian bridge와 Sobolev–Gaussian route로 구분하고 support·normalization 조건을 정리한다. 닫힌 것은 지정한 path space와 sigma-algebra 위의 확률측도 성질이며, 실제 물리 ensemble·gauge·causality를 선택하는 일은 별도 공리·미완성 bridge다.

독자는 05f의 topology와 05g의 recovery package를 먼저 읽는다. 서로 다른 후보공간의 비혼용, Brownian과 Sobolev prior의 정리, recovery route, finite approximation과 CE A4 선택 경계를 순서대로 읽는다.

## 0. 목표

formal $\mathcal D\gamma$는 countably additive measure나 full support를 제공하지 않는다. 이 절은 수학적으로 정의된 prior와 물리적 경로 ensemble의 차이를 먼저 고정한다.

[05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)는 CE continuum 코어를

$$
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,M),
\qquad
\text{readout topology}=C^0
$$

로 고정했다. [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md)는 recovery mass 조건을, [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md)는 finite-to-continuum scale 조건을 닫았다.

이제 남은 질문은 실제 physical prior다.

핵심 결론:

> Brownian bridge는 $C^0$ path prior로는 full support를 갖지만, $W^{1,p}$ kinetic/Tonelli action과 그대로 결합하면 맞지 않는다. Brownian path는 거의 surely $W^{1,p}$가 아니기 때문이다. 현재 $W^{1,p}/C^0$ 코어와 가장 직접적으로 맞는 continuum prior는 Sobolev-Gaussian prior다.

형식 출처:

| prior route | 판정 | 이유 |
|---|---|---|
| Sobolev-Gaussian on $H^1$ | `[정리]` | $H^1$ 안에 살고 full support를 가짐 |
| Brownian bridge의 support 경계 | `[정리]` | $C^0$ full support지만 $W^{1,p}$ 질량은 0; Tonelli prior로 쓸 수 없음 |
| dense atomic prior | `[정리]` | separable pathspace에서 항상 만들 수 있음 |
| finite mesh prior | `[정리]` | 05h의 scaled recovery 조건과 결합 |
| physical $\mathcal D\gamma$ | `[미완성]` | 어떤 route를 물리 원리로 채택할지 선택 필요 |

## 1. 두 후보공간을 섞으면 안 된다

$C^0$와 $W^{1,p}$ 또는 $H^1$은 topology·Borel sigma-algebra·support가 다른 공간이다. 한 공간의 full support와 다른 공간의 kinetic action을 결합하면 recovery와 normalization이 무정의가 될 수 있다.

CE에는 서로 다른 두 층이 있다.

| 층 | 공간 | prior 예 |
|---|---|---|
| variational/Tonelli route | $W^{1,p}_{x_i,x_f}$ with $C^0$ readout | Sobolev-Gaussian, dense atomic, finite mesh limit |
| stochastic/Brownian route | $C^0_{x_i,x_f}$ | Brownian bridge, diffusion bridge |

둘은 같은 것이 아니다.

Tonelli action

$$
S_E[\gamma]=\int_I L(t,\gamma(t),\dot\gamma(t))dt
$$

은 $\dot\gamma$가 있는 absolutely continuous path에서 자연스럽다. 반면 Brownian bridge path는 연속이지만 almost surely absolutely continuous가 아니다.

따라서 Brownian prior를 쓰려면 다음 둘 중 하나를 택해야 한다.

1. kinetic term을 Brownian reference measure 안으로 흡수하고, 남은 potential/suppression action만 $C^0$ 위에서 쓴다.
2. Brownian을 continuum prior로 쓰지 않고 finite mesh approximation 또는 Sobolev prior로 바꾼다.

## 2. Brownian bridge의 정확한 지위

Brownian bridge는 연속 경로 공간의 확률측도라는 장점이 있지만 일반적으로 Sobolev kinetic action의 prior가 아니다. 다음 정리는 이 두 지위를 구별해 route 혼용의 반례를 막는다.

여기서는 $I=[0,1]$, $M=\mathbb R^d$로 둔다. endpoint $x_i,x_f$를 잇는 Brownian bridge를

$$
B^{x_i,x_f}_t
=
(1-t)x_i+tx_f+\sqrt{\sigma}\,\widetilde B_t
$$

라 쓰자. 여기서 $\widetilde B$는 $0$에서 시작해 $0$에서 끝나는 표준 Brownian bridge다.

### 정리 2.1: Brownian bridge는 $C^0$ full support

정리는 고정 endpoint와 $C^0$ topology에서의 support 결론이다. 이는 미분가능성·gauge invariance·실제 CE dynamics를 보장하지 않는다.

$\mu_{\mathrm{BB}}$를 $C^0_{x_i,x_f}([0,1],\mathbb R^d)$ 위의 Brownian bridge law라고 하자. 그러면 임의의 $h\in C^0_{x_i,x_f}$와 $\varepsilon>0$에 대해

$$
\mu_{\mathrm{BB}}
\{\gamma:\|\gamma-h\|_\infty<\varepsilon\}>0.
$$

즉 $\mu_{\mathrm{BB}}$는 $C^0$ topology에서 full support다.

증명:

polygonal path는 $C^0_{x_i,x_f}$에서 조밀하다. 따라서 먼저 $h$가 polygonal이고 $h(0)=x_i,h(1)=x_f$라고 하자. endpoint 선형 경로를 뺀

$$
k(t)=h(t)-((1-t)x_i+tx_f)
$$

는 $H^1_0$에 속한다. Brownian bridge의 Cameron-Martin space는 $H^1_0$이고, Cameron-Martin theorem에 의해 $k$만큼 shift한 bridge law는 원래 bridge law와 서로 absolutely continuous다. 따라서

$$
\mu_{\mathrm{BB}}\{\|\gamma-h\|_\infty<\varepsilon\}
>0
$$

는

$$
\mu_{\mathrm{BB}}\{\|\gamma-\ell\|_\infty<\varepsilon\}>0,
\qquad
\ell(t)=(1-t)x_i+tx_f
$$

와 동치다. 마지막 확률은 Brownian bridge가 작은 sup-norm tube 안에 머무는 사건이며 양수다.

일반 $h\in C^0_{x_i,x_f}$는 polygonal $h_m$으로 $\|h_m-h\|_\infty<\varepsilon/2$가 되게 근사한다. 그러면

$$
\{\|\gamma-h_m\|_\infty<\varepsilon/2\}
\subset
\{\|\gamma-h\|_\infty<\varepsilon\}
$$

이므로 양의 질량이 따라온다. 끝.

### 정리 2.2: Brownian bridge는 $W^{1,p}$ prior가 아니다

이 정리는 Brownian 경로의 regularity 한계를 밝힌다. 따라서 kinetic functional을 그대로 density로 재가중하는 구성은 정의역·적분 가능성을 별도로 해결해야 한다.

$p\ge1$라고 하자. Brownian bridge law는

$$
\mu_{\mathrm{BB}}(W^{1,p}_{x_i,x_f})=0.
$$

증명:

$W^{1,p}\subset W^{1,1}$이고 $W^{1,1}$ path는 absolutely continuous이다. 따라서 각 path는 finite variation을 갖는다.

finite variation continuous path $f$는 임의의 mesh size가 0으로 가는 partition $\Pi_n$에 대해 quadratic variation이 0이다.

$$
\sum_{[u,v]\in\Pi_n}|f(v)-f(u)|^2
\le
\max_{[u,v]\in\Pi_n}|f(v)-f(u)|
\operatorname{Var}(f)
\to0.
$$

반면 Brownian bridge는 Brownian motion과 같은 quadratic variation을 갖는다. 균등 partition에 대해 각 성분 $j=1,\dots,d$는

$$
\sum_i
\left|
B^{x_i,x_f,j}_{t_{i+1}}-B^{x_i,x_f,j}_{t_i}
\right|^2
\to \sigma
$$

almost surely이고, Euclidean norm을 사용한 전체 합은

$$
\sum_i |B^{x_i,x_f}_{t_{i+1}}-B^{x_i,x_f}_{t_i}|^2
\to d\sigma
$$

almost surely다. 따라서 $\sigma>0$인 Brownian bridge sample path는 finite variation일 수 없고, $W^{1,1}$, 따라서 $W^{1,p}$에도 속하지 않는다. 끝.

결론:

$$
S_E[\gamma]=\int |\dot\gamma|^pdt
$$

를 finite-energy action으로 두고 $S_E=\infty$ outside $W^{1,p}$로 확장하면

$$
\mu_{\mathrm{BB}}\{S_E<\infty\}=0.
$$

즉 Brownian bridge를 $\mu_{\mathrm{base}}$로 두고 kinetic action을 다시 Gibbs reweight하면 분모가 살아나지 않는다.

## 3. Sobolev-Gaussian prior

Sobolev–Gaussian route는 kinetic regularity와 support를 같은 함수공간에서 다루기 위한 수학적 구성이다. covariance·boundary 조건·reference scale은 입력으로 선택해야 하며 물리 ensemble에서 유도되지 않는다.

현재 CE 코어와 가장 잘 맞는 physical-looking continuum prior는 Sobolev 공간 안에 사는 Gaussian prior다. 여기서는 가장 깨끗한 $p=2$ package를 쓴다.

### 세팅

세팅은 path space, Borel sigma-algebra, Gaussian covariance와 endpoint 처리의 정의역을 고정한다. gauge quotient나 Lorentzian causality는 이 measure 정의만으로 처리되지 않는다.

endpoint 선형 경로를

$$
\ell(t)=(1-t)x_i+tx_f
$$

라 두고

$$
H=H^1_0([0,1],\mathbb R^d)
$$

라 하자. 그러면 endpoint 고정 경로공간은 affine Hilbert space

$$
\ell+H=H^1_{x_i,x_f}
$$

다.

$\{e_n\}_{n\ge1}$을 $H$의 orthonormal basis라 하고, 양수열 $\lambda_n>0$이

$$
\sum_{n=1}^\infty \lambda_n<\infty
$$

를 만족한다고 하자. 독립 표준정규 $\xi_n$으로

$$
X=\sum_{n=1}^\infty \sqrt{\lambda_n}\xi_ne_n
$$

를 정의한다.

### 정리 3.1: Sobolev-Gaussian은 $H^1$ 안에 산다

정리는 명시한 covariance와 계수 합 조건 아래의 almost-sure regularity 결과다. 다른 covariance 또는 무한차원 target에는 그대로 적용되지 않는다.

위 조건 아래에서 $X\in H$ almost surely이고

$$
\mu_{\mathrm{SG}}:=\operatorname{Law}(\ell+X)
$$

는 $H^1_{x_i,x_f}$ 위의 Borel probability measure다.

증명:

Hilbert norm에 대해

$$
\mathbb E\|X\|_H^2
=
\sum_{n=1}^\infty \lambda_n
<\infty.
$$

따라서 $\|X\|_H<\infty$ almost surely이고 $X\in H$ almost surely다. 끝.

### 정리 3.2: Sobolev-Gaussian은 $H^1$ full support

full support는 $H^1$ 열린 근방마다 양의 질량이 있다는 뜻이다. 이는 energy near-minimum set이 그 근방을 포함한다는 continuity 조건과 결합되어야 recovery가 된다.

$\lambda_n>0$ for all $n$이면 $\mu_{\mathrm{SG}}$는 $H^1_{x_i,x_f}$에서 full support다. 즉 임의의 $g\in H^1_{x_i,x_f}$, $r>0$에 대해

$$
\mu_{\mathrm{SG}}\{\gamma:\|\gamma-g\|_{H^1}<r\}>0.
$$

증명:

$g=\ell+h$, $h\in H$라 하자. finite span이 $H$에서 조밀하므로 $h^{(k)}=\sum_{n=1}^k a_ne_n$을 골라

$$
\|h-h^{(k)}\|_H<r/4
$$

로 만든다.

이제

$$
X=X_{\le k}+X_{>k}
$$

로 나눈다. $X_{\le k}$는 $\mathbb R^{kd}$의 nondegenerate Gaussian이므로

$$
\mathbb P(\|X_{\le k}-h^{(k)}\|_H<r/4)>0.
$$

또한 $\mathbb E\|X_{>k}\|_H^2=\sum_{n>k}\lambda_n$이므로 $k$를 더 키우면

$$
\mathbb P(\|X_{>k}\|_H<r/2)>0
$$

가 된다. 두 사건은 독립이므로 동시에 일어날 확률도 양수다. 그 사건 위에서

$$
\|X-h\|_H
\le
\|X_{\le k}-h^{(k)}\|_H
+\|X_{>k}\|_H
+\|h^{(k)}-h\|_H
<r.
$$

따라서 $H^1$-ball은 양의 질량을 갖는다. 끝.

### 정리 3.3: $C^0$ readout full support

$C^0$ readout support는 embedding과 topology를 통해 얻는 결론이다. $C^0$ support를 $H^1$에서의 mass statement로 역으로 해석할 수는 없다.

1차원 Sobolev embedding에 의해

$$
H^1_{x_i,x_f}\hookrightarrow C^0_{x_i,x_f}
$$

는 continuous다. 따라서 $\mu_{\mathrm{SG}}$를 $C^0$ Borel measure로 보아도, 모든 $g\in H^1_{x_i,x_f}$와 $\varepsilon>0$에 대해

$$
\mu_{\mathrm{SG}}\{\gamma:\|\gamma-g\|_\infty<\varepsilon\}>0.
$$

또한 $H^1_{x_i,x_f}$는 $C^0_{x_i,x_f}$에서 조밀하므로, $\mu_{\mathrm{SG}}$의 $C^0$-support는 전체 $C^0_{x_i,x_f}$다.

증명:

embedding continuity로 어떤 $r>0$가 존재해서

$$
\|\gamma-g\|_{H^1}<r
\quad\Longrightarrow\quad
\|\gamma-g\|_\infty<\varepsilon
$$

이다. 정리 3.2로 왼쪽 $H^1$-ball은 양의 질량을 갖는다. 따라서 $C^0$-ball도 양의 질량을 갖는다.

일반 $c\in C^0_{x_i,x_f}$에 대해서는 $H^1$ path $g$를 $\|g-c\|_\infty<\varepsilon/2$로 잡고 위 논리를 적용한다. 끝.

## 4. Sobolev-Gaussian recovery

recovery는 action의 continuity 또는 near-minimum tube가 prior support와 만나는 추가 조건이다. 다음 정리는 support 자체와 농축 분모 하한을 구별한다.

### 정리 4.1: continuity route

continuity route는 minimizer 주변의 열린 energy sublevel을 만들고 full support를 사용한다. l.s.c.만 있을 때는 이 논증이 실패할 수 있다.

$\mu_{\mathrm{base}}=\mu_{\mathrm{SG}}$라고 하자. $W:H^1_{x_i,x_f}\to[0,\infty]$가 어떤 minimizer $\gamma_*\in H^1_{x_i,x_f}$에서 $H^1$-continuous이고

$$
W(\gamma_*)=W_{\min}<\infty
$$

이면 recovery mass가 성립한다.

$$
\mu_{\mathrm{SG}}\{\gamma:W[\gamma]<W_{\min}+\eta\}>0
\qquad(\eta>0).
$$

증명:

$W$가 $\gamma_*$에서 $H^1$-continuous이므로 임의의 $\eta>0$에 대해 어떤 $H^1$-open ball $B_{H^1}(\gamma_*,r)$가 존재해서

$$
B_{H^1}(\gamma_*,r)
\subset
\{\gamma:W[\gamma]<W_{\min}+\eta\}
$$

이다. 정리 3.2로 이 ball은 양의 $\mu_{\mathrm{SG}}$-질량을 갖는다. 끝.

### 정리 4.2: $C^0$-continuity route

$C^0$ route는 action과 interpolation이 해당 topology에서 연속이라는 별도 가정을 쓴다. 더 강한 Sobolev 정칙성을 자동으로 보장하지 않는다.

만약 $W$가 $\gamma_*$에서 $C^0$-continuous이면 같은 결론이 성립한다.

증명:

$C^0$-continuity로 어떤 $C^0$-ball이 near-minimum set 안에 들어간다. 정리 3.3이 그 $C^0$-ball의 양질량을 준다. 끝.

주의:

- $W$가 l.s.c.일 뿐이면 05g의 반례 때문에 recovery mass가 자동이 아니다.
- 이 경우에는 positive tube recovery를 별도 공리로 두어야 한다.

## 5. Brownian route를 살리는 방법

Brownian prior를 사용하려면 kinetic cost를 prior law에 흡수하는 등 정의역을 일관되게 바꾸는 package가 필요하다. 이는 다른 route의 action을 그대로 옮기는 조작이 아니다.

Brownian bridge를 쓰고 싶다면 kinetic action을 다시 penalty로 곱하면 안 된다. 대신 다음처럼 읽어야 한다.

### Package B: kinetic term absorbed into prior

Package B는 reference measure와 reweighting functional의 책임을 재배치하는 모델 선택이다. normalization, gauge fixing, causality 및 continuum physical meaning은 여전히 확인해야 한다.

경로공간:

$$
\Gamma=C^0_{x_i,x_f}.
$$

base prior:

$$
\mu_{\mathrm{base}}=\mu_{\mathrm{BB}}.
$$

energy:

$$
W_{\mathrm{B}}[\gamma]
=
S_{\mathrm{pot}}[\gamma]/\hbar
+S_{\mathrm{supp}}[\gamma],
$$

where $S_{\mathrm{pot}}$ and $S_{\mathrm{supp}}$ are defined on $C^0$. In this route the Brownian covariance already carries the kinetic reference.

필요 조건:

| 조건 | 이유 |
|---|---|
| $W_{\mathrm{B}}$ is l.s.c. good-rate on $C^0$ 또는 tightness 대체 정리 | Brownian route의 compactness 확보 |
| Brownian full support | open tube recovery |
| $W_{\mathrm{B}}$ continuous at minimizer 또는 positive tube recovery | 05g recovery mass |
| finite mesh approximation consistency | 05h scale condition |

출처:

`[미완성]`

이 route는 물리적으로 자연스러울 수 있지만, 05f의 Tonelli proof를 그대로 쓰는 route가 아니다. 새로운 $C^0$/Brownian good-rate 또는 large-deviation package가 필요하다.

## 6. CE 권장 A4 공리

A4는 실제 CE 계산에서 선택할 path prior·sigma-algebra·finite approximation contract를 명시하는 공리다. 공리 채택은 physical ensemble의 실증적 정당화나 유일성을 증명하지 않는다.

05f-05h까지의 선택을 유지하려면 현재 가장 안전한 physical path prior 공리는 다음이다.

> **A4 physical path prior axiom.**  
> CE continuum prior는 다음 중 하나로 명시한다.
>
> **S-route.** $H^1_{x_i,x_f}$ 또는 더 강한 Sobolev pathspace 위의 trace-class Gaussian prior $\mu_{\mathrm{SG}}$를 쓴다. 이 prior는 chosen Sobolev topology에서 full support를 갖고 $C^0$ readout support도 충분하다. $W$는 minimizer에서 continuous이거나 positive tube recovery를 만족해야 한다.
>
> **B-route.** Brownian bridge prior $\mu_{\mathrm{BB}}$를 $C^0_{x_i,x_f}$ 위에 둔다. 이때 kinetic action은 prior에 흡수된 것으로 보고, 별도의 $W^{1,p}$ kinetic Gibbs penalty를 다시 곱하지 않는다.
>
> **F-route.** finite mesh prior를 쓰고 [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md)의 scaled recovery condition을 확인한다.

현재 문서군의 canonical continuum proof는 S-route다.

## 7. 닫힌 것과 남은 것

수학적으로 닫힌 support/recovery 결과와 gauge·causality·reference normalization·finite-to-continuum 물리 해석의 남은 병목을 구분한다. 표의 package는 채택 후보이지 승격 판정이 아니다.

닫힌 것:

| 항목 | 상태 |
|---|---|
| Brownian bridge $C^0$ full support | 정리 2.1 |
| Brownian bridge not $W^{1,p}$ | 정리 2.2 |
| Sobolev-Gaussian $H^1$ support | 정리 3.1, 3.2 |
| Sobolev-Gaussian $C^0$ readout support | 정리 3.3 |
| Sobolev-Gaussian recovery under continuity | 정리 4.1, 4.2 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| CE physical choice | S-route, B-route, F-route 중 하나를 CE 본문 공리로 채택 |
| $S_{\mathrm{supp}}$ physical form | residual/curvature/obstacle/fold penalty 중 무엇을 실제 action으로 둘지 결정 |
| Brownian route good-rate | $C^0$ Brownian package에서 compactness 또는 tightness 기반 농축 정리 작성 |
| manifold support theorem | $M$이 일반 manifold일 때 chart/localization 또는 embedded manifold prior로 확장 |

## 8. 결론

결론적으로 CE prior는 path space와 topology를 먼저 고른 뒤 reference measure·support·recovery를 같은 정의역에서 검증해야 한다. 실제 물리 ensemble은 이 수학 package 밖의 외부 선택으로 남는다.

CE가 현재의 $W^{1,p}/C^0$ Tonelli proof를 유지하려면 physical continuum prior는

$$
\boxed{
\mu_{\mathrm{base}}=\mu_{\mathrm{SG}}
\quad\text{on}\quad
H^1_{x_i,x_f}
}
$$

처럼 Sobolev 안에 사는 full-support probability로 두는 것이 가장 깨끗하다.

그러면

$$
\boxed{
H^1/C^0
+\text{ Sobolev-Gaussian full support}
+\text{ good-rate }W
+\text{ continuity/tube recovery}
\Longrightarrow
\text{manifest path concentration}.
}
$$

반대로 Brownian bridge를 쓰려면 결론은 이렇게 바뀐다.

$$
\boxed{
\text{Brownian bridge is a }C^0\text{ prior, not a }W^{1,p}\text{ prior}.
}
$$

따라서 다음 병목은 $S_{\mathrm{supp}}$의 실제 물리 형태다. prior/topology/action의 뼈대는 이제 닫혔고, 남은 것은 어떤 suppression/fold/residual penalty가 CE의 물리 내용을 담는지 정하는 일이다.
