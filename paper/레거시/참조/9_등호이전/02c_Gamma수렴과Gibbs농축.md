# 02c. Gamma 수렴과 Gibbs 농축

이 문서는 scale에 따라 변하는 에너지열의 변분적 극한과 Gibbs 측도의 확률적 농축을 구별해 연결한다. 핵심 결과는 Gamma 수렴만으로는 Gibbs 정규화에 필요한 기준측도의 양질량을 보장하지 않으며, 이를 보충하는 recovery 조건 또는 더 강한 국소균등수렴이 필요하다는 것이다.

독자는 02a의 noncompact 농축과 약수렴을 먼저 알고, Gamma 수렴의 liminf·recovery 정의는 이 문서에서 처음 보아도 된다. 세팅과 내부 선택 반례 뒤에 국소균등 버전, Gamma 버전, 움직이는 recovery의 속도 조건을 읽고, 마지막에 jet 및 CE 적용의 조건부 범위를 확인하는 순서다.

## 0. 목표

02a는 고정된 에너지의 noncompact 농축을, 02b는 명시적으로 움직이는 jet 중심을 다루었다. 여기서는 에너지 자체가 변할 때 최소화의 변분적 안정성과 확률적 정규화가 서로 다른 가정을 요구함을 보이며, 둘을 같은 극한 명제로 합쳐 부르지 않는다.

$$
E_n \longrightarrow E_0
$$

핵심 질문:

> $E_n$으로 만든 Gibbs 측도 $\mu_n$는 $E_0$의 minimizer로 농축하는가?

정직한 답:

- $E_n\to E_0$가 국소균등이고 minimizer가 유일하면, $\beta_n\to\infty$만으로도 농축이 닫힌다.
- Gamma 수렴만으로는 분모의 양질량 하한이 자동으로 나오지 않는다. 고정 기준측도 $\mu_0$와 맞는 recovery 조건이 필요하다.
- $E_0$의 최소집합 내부 선택은 $\beta_n$과 $E_n-E_0$의 상대 scale이 결정할 수 있다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| 국소균등수렴 + equicoercive + 유일 minimizer | `[정리]` | 본문 정리 4.1 |
| Gamma 수렴 + positive-mass recovery | `[정리]` | 본문 정리 5.2 |
| 최소집합 내부 선택 | `[미완성]` | $\beta_n(E_n-E_0)$ scale에 의존 |
| jet 농축과의 연결 | `[정리]` | 02b의 움직이는 중심 정리 |

## 1. 세팅

정리의 정의역은 유클리드 topology를 갖는 $\mathbb R^d$와 고정 기준측도 $\mu_0$로 정한다. $n$은 에너지와 온도를 함께 바꾸는 지수이고, $\beta_n$의 발산과 $E_n-E_0$의 상대 속도는 최소집합 내부 선택에서 별도의 정보를 낼 수 있다.

후보공간은

$$
A=\mathbb R^d
$$

이다. 초기 모호함 상태는 Borel 확률측도

$$
\mu_0\in\mathcal P(\mathbb R^d)
$$

이고

$$
S=\operatorname{supp}\mu_0
$$

라 둔다.

에너지열은

$$
E_n:\mathbb R^d\to\mathbb R_{\ge0}
$$

이고 극한 에너지는

$$
E_0:\mathbb R^d\to\mathbb R_{\ge0}
$$

이다. 강도열은

$$
\beta_n\to\infty
$$

이다.

Gibbs 측도는

$$
\mu_n(dx)
=
\frac{e^{-\beta_nE_n(x)}}{Z_n}\mu_0(dx),
\qquad
Z_n=\int_{\mathbb R^d}e^{-\beta_nE_n(x)}\mu_0(dx)
$$

이다.

$E_0$의 support 위 최소값과 최소집합은

$$
m=\inf_{x\in S}E_0(x),
\qquad
A_*=\operatorname*{argmin}_{x\in S}E_0(x)
$$

이다.

## 2. 왜 02a로 부족한가

고정 에너지의 gap 추정은 에너지가 변하면 그대로 유지되지 않는다. 이 절은 minimizer·gap·기준측도 질량이 각각 무엇을 통제하는지 나누며, $E_n\to E_0$라는 기호만으로 Gibbs 분모의 하한이나 약수렴을 결론내릴 수 없음을 설명한다.

$$
\beta\to\infty
$$

를 보낸다. 이때 gap은 한 에너지 $E$ 위에서 고정된다.

하지만 $E_n$이 변하면 세 가지가 새로 생긴다.

| 문제 | 의미 |
|---|---|
| minimizer 이동 | $\operatorname*{argmin}E_n$이 $n$에 따라 움직임 |
| gap 이동 | $E_n$의 바깥 gap이 $n$에 따라 줄어들 수 있음 |
| 기준측도 문제 | $\Gamma$-recovery point 주변에 $\mu_0$-질량이 충분하지 않을 수 있음 |

따라서 에너지 수렴만 말하면 부족하고, Gibbs 분모를 지탱할 positive-mass recovery가 필요하다.

## 3. 내부 선택 반례

극한 에너지의 최소집합이 여럿이면 Gamma 수렴이나 점별 수렴은 그 내부의 상대 가중치를 일반적으로 정하지 못한다. 다음 유한 공간 반례는 작아지는 에너지 차이가 발산하는 역온도에 의해 증폭되는 순서-의존성을 보이며, 연속 공간의 topology나 CE의 물리 선택을 증명하는 예는 아니다.

두 점 공간

$$
A=\{0,1\}
$$

위의 균등측도 $\mu_0$를 잡자.

$$
E_n(0)=\frac1{\sqrt n},
\qquad
E_n(1)=0
$$

라 두면

$$
E_n\to E_0,\qquad E_0(0)=E_0(1)=0
$$

이다. 따라서 $E_0$의 최소집합은 $\{0,1\}$ 전체다.

하지만 $\beta_n=n$이면

$$
\frac{\mu_n(0)}{\mu_n(1)}
=
e^{-n\cdot n^{-1/2}}
=
e^{-\sqrt n}
\to0
$$

이므로

$$
\mu_n\Rightarrow\delta_1.
$$

해석:

- $E_0$만 보면 $0$과 $1$은 구분되지 않는다.
- $E_n$의 작은 차이가 $\beta_n$에 증폭되어 내부 선택을 만든다.
- 따라서 최소집합 내부 분포를 고정하려면 별도 `[공리: 모델 선택]`이 필요하다.

## 4. 국소균등수렴 버전

가장 직접적인 충분조건은 compact 집합마다 에너지 오차를 균등하게 통제하는 것이다. 다음 가정과 정리는 유일 최소점, equicoercivity, 고정 기준측도의 양의 근방 질량을 함께 쓰며, 유한-$n$ 오차율이나 여러 최소점의 내부 선택까지 주장하지 않는다.

### 가정 U

가정 U는 변분적 최소점 안정성뿐 아니라 Gibbs 분자와 분모를 같은 compact 포획집합에서 비교할 수 있게 한다. 특히 equicoercivity는 $n$에 따라 sublevel 집합이 무한대로 도피하는 반례를 배제하는 가정이지, 국소균등수렴의 결과가 아니다.

1. $E_0$는 continuous/coercive다.
2. $E_0|_S$의 minimizer는 유일하다.

$$
A_*=\{x_*\}
$$

3. $E_n\to E_0$는 모든 compact $K\subset\mathbb R^d$에서 균등이다.
4. $\{E_n\}$은 equicoercive다. 즉 모든 $c<\infty$에 대해 어떤 compact $K_c$가 존재해서 충분히 큰 $n$에 대해

$$
\{x:E_n(x)\le c\}\subset K_c
$$

이다.

**정리 4.1**  
가정 U 아래에서

$$
\mu_n\Rightarrow\delta_{x_*}
$$

이다.

**증명.**

$x_*$를 포함하는 임의의 열린집합 $U$를 잡는다. $S\setminus U=\varnothing$이면 자명하다.

$S\setminus U\ne\varnothing$라 하자. 02a 보조정리 3.1과 coercivity 논법에 의해, 충분히 큰 compact $K$ 안에서 $E_0$는 $S\setminus U$ 위에 양의 gap을 갖는다. 더 구체적으로, 작은 $\eta>0$를 택하고 $V=\{x:E_0(x)<m+\eta\}\cap U$를 잡으면 $\mu_0(V)>0$이다.

equicoercivity로 $c>m+2\eta$에 대해 $\{E_n\le c\}\subset K_c$인 compact $K_c$를 잡는다. compact $K_c\cap(S\setminus U)$ 위에서는 어떤 $\delta>0$에 대해

$$
E_0(x)\ge m+\delta
$$

이다. $\eta<\delta/4$로 잡는다.

국소균등수렴에 의해 충분히 큰 $n$에 대해 $K_c$ 위에서

$$
|E_n(x)-E_0(x)|<\eta
$$

이다. 따라서

$$
E_n(x)\ge m+\delta-\eta,
\qquad x\in K_c\cap(S\setminus U)
$$

이고

$$
E_n(x)\le m+2\eta,
\qquad x\in V
$$

이다.

분모는

$$
Z_n
\ge
\int_Ve^{-\beta_nE_n(x)}\mu_0(dx)
\ge
e^{-\beta_n(m+2\eta)}\mu_0(V)
$$

이다.

compact 안의 바깥 질량은

$$
\int_{K_c\cap(S\setminus U)}e^{-\beta_nE_n(x)}\mu_0(dx)
\le
e^{-\beta_n(m+\delta-\eta)}
$$

이므로

$$
\frac{
\int_{K_c\cap(S\setminus U)}e^{-\beta_nE_n(x)}\mu_0(dx)
}{Z_n}
\le
\frac1{\mu_0(V)}
e^{-\beta_n(\delta-3\eta)}
\to0
$$

이다.

compact 바깥에서는 $E_n>c$이므로

$$
\frac{
\int_{\mathbb R^d\setminus K_c}e^{-\beta_nE_n(x)}\mu_0(dx)
}{Z_n}
\le
\frac1{\mu_0(V)}
e^{-\beta_n(c-m-2\eta)}
\to0
$$

이다. 따라서 $\mu_n(\mathbb R^d\setminus U)\to0$, 즉 $\mu_n(U)\to1$이다. bounded continuous test function에 대한 근방 분해로

$$
\mu_n\Rightarrow\delta_{x_*}
$$

가 따른다. $\square$

**주의.**  
이 정리에는 $\beta_n\sup_K|E_n-E_0|\to0$ 같은 joint scaling이 필요하지 않다. 유일 minimizer 바깥의 gap $\delta$가 고정되어 있고, 균등오차가 결국 $\delta$보다 작아지면 $\beta_n\to\infty$가 gap을 증폭한다.

## 5. Gamma 수렴 버전

국소균등수렴은 많은 변분 문제에서 불필요하게 강하므로, 최소화 문제의 안정성에는 Gamma 수렴을 사용할 수 있다. 그러나 이 약한 convergence가 확률적 정규화까지 통제하려면, recovery가 단일 점열이 아니라 고정 기준측도에서 양의 질량을 가진 집합으로 실현되어야 한다.

### 정의 5.1: Gamma 수렴

이 정의는 선택한 topology에서 에너지열의 하반연속성과 근사 가능성을 표현한다. liminf와 recovery는 minimizer 값과 위치의 안정성에 관한 조건이며, Gibbs 온도나 분모의 크기를 그 자체로 제한하지 않는다.

$E_n\xrightarrow{\Gamma}E_0$란 다음 두 조건을 뜻한다.

1. liminf: $x_n\to x$이면

$$
E_0(x)\le\liminf_nE_n(x_n).
$$

2. recovery: 각 $x$에 대해 $x_n\to x$이고

$$
\limsup_nE_n(x_n)\le E_0(x)
$$

인 열이 존재한다.

Gamma 수렴은 minimizer의 위치 안정성에는 강하지만, $\mu_0$-양질량 분모 하한을 자동으로 주지 않는다.

### 가정 G

가정 G는 Gamma 수렴의 변분적 정보를 Gibbs 측도로 옮기는 데 필요한 추가 전제를 명시한다. 네 번째 항의 positive-mass recovery는 $x_*$로 가까워지는 한 점의 recovery sequence보다 강하며, 바로 그 차이가 정규화 상수의 하한을 만든다.

1. $E_n\xrightarrow{\Gamma}E_0$.
2. $\{E_n\}$은 equicoercive다.
3. $E_0|_S$의 minimizer는 유일하다.

$$
A_*=\{x_*\}
$$

4. positive-mass recovery: $x_*$의 임의의 열린근방 $U$와 임의의 $\eta>0$에 대해, 어떤 Borel 집합 $V\subset U$와 $n_0$가 존재해서

$$
\mu_0(V)>0
$$

이고 모든 $n\ge n_0$에 대해

$$
\sup_{x\in V}E_n(x)\le m+\eta
$$

이다.

**정리 5.2**  
가정 G 아래에서

$$
\mu_n\Rightarrow\delta_{x_*}
$$

이다.

**증명.**

$x_*$를 포함하는 열린집합 $U$를 잡는다. equicoercivity로 compact 바깥 질량은 정리 4.1과 같은 방식으로 제어할 수 있다. 따라서 충분히 큰 compact $K$ 안에서 $K\cap(S\setminus U)$만 보면 된다.

Gamma liminf와 유일 minimizer를 이용하면 $K\cap(S\setminus U)$ 위에 uniform gap이 생긴다. 만약 그렇지 않다면, 어떤 부분열과 $x_n\in K\cap(S\setminus U)$가 존재해서

$$
E_n(x_n)\to m
$$

이다. compactness로 $x_n\to x\in K\cap(S\setminus U)$인 부분열을 잡을 수 있다. Gamma liminf에 의해

$$
E_0(x)\le\liminf_nE_n(x_n)=m
$$

이므로 $x$는 $E_0$의 minimizer다. 유일성 때문에 $x=x_*$이어야 한다. 그러나 $x\in S\setminus U$이고 $x_*\in U$이므로 모순이다.

따라서 어떤 $\delta>0$가 존재해서 충분히 큰 $n$에 대해

$$
E_n(x)\ge m+\delta,
\qquad x\in K\cap(S\setminus U)
$$

이다.

positive-mass recovery에서 $\eta<\delta/4$를 택하면 $V\subset U$, $\mu_0(V)>0$, 그리고 충분히 큰 $n$에 대해

$$
\sup_VE_n\le m+\eta
$$

이다. 따라서

$$
Z_n\ge e^{-\beta_n(m+\eta)}\mu_0(V)
$$

이고

$$
\frac{
\int_{K\cap(S\setminus U)}e^{-\beta_nE_n(x)}\mu_0(dx)
}{Z_n}
\le
\frac1{\mu_0(V)}e^{-\beta_n(\delta-\eta)}
\to0.
$$

compact 바깥 질량도 equicoercivity와 같은 분모 하한으로 0이 된다. 따라서 $\mu_n(U)\to1$이고 약수렴이 따른다. $\square$

## 6. Moving recovery와 속도 조건

가정 G에서 회복집합의 양의 질량이 $n$에 독립적으로 유지되면 gap이 역온도에 의해 충분히 증폭된다. 반면 recovery가 움직이며 기준측도 질량을 잃으면 정규화 하한도 함께 작아지므로, 다음 조건은 에너지 오차·온도·질량 손실의 상대 scale을 명시하는 충분조건일 뿐 필요충분 조건이나 실측 법칙은 아니다.

충분조건:

$$
\sup_{x\in V_n}E_n(x)\le m+\eta_n,
\qquad
\mu_0(V_n)>0
$$

이고, 바깥 gap이 $\delta>0$일 때

$$
\beta_n(\delta-\eta_n)+\log\mu_0(V_n)\to+\infty
$$

이면 바깥 질량은 0으로 간다.

특히 $\mu_0(V_n)$이 $n$에 무관하게 양의 하한을 갖고 $\eta_n<\delta/2$가 결국 성립하면 충분하다.

## 7. 02b jet 농축과의 관계

jet 에너지는 움직이는 중심이 명시되어 있어 Gamma 수렴의 일반 틀 안에서도 더 강한 직접 비교가 가능하다. 이 관계는 02b의 수학적 정리를 재해석하는 것이며, finite-difference 관측을 CE의 물리적 변수로 승격하거나 임의의 scale-dependent 에너지에 동일한 속도 조건이 불필요하다고 말하지 않는다.

$$
E_h(s)=\|s-D_h(x)\|^2,
\qquad
E_0(s)=\|s-J(x)\|^2
$$

형태다. $D_h(x)\to J(x)$이면 compact $K$ 위에서

$$
\sup_{s\in K}|E_h(s)-E_0(s)|
\le
C_K\|D_h(x)-J(x)\|
\to0.
$$

따라서 02b는 정리 4.1의 특수한 예로도 읽을 수 있다. 다만 02b의 직접 증명은 더 강하다. 움직이는 중심 $D_h(x)\to J(x)$와 $\beta_h\to\infty$만으로

$$
\mu_h\Rightarrow\delta_{J(x)}
$$

가 따라온다. 즉 jet 농축에는 $\beta_h\|D_h-J\|\to0$ 같은 조건이 필요하지 않다.

## 8. 닫힌 것과 열린 것

지금까지의 결과는 각 convergence 가정과 기준측도 조건을 보존할 때만 닫힌다. 다음 구분은 수학적으로 증명된 농축 범위와, minimizer 내부 선택 및 CE 경로공간처럼 추가 topology·측도 검증이 필요한 미완성 다리를 혼동하지 않도록 한다.

닫힌 것:

| 항목 | 상태 |
|---|---|
| 고정 $E$, $\beta\to\infty$ | [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md) |
| 국소균등 $E_n\to E_0$, 유일 minimizer | 정리 4.1 |
| Gamma 수렴 + positive-mass recovery | 정리 5.2 |
| moving recovery 속도 조건 | 6절 |
| jet 농축 | [02b_미분과Jet농축.md](02b_미분과Jet농축.md) |

열린 것:

| 항목 | 이유 |
|---|---|
| 순수 Gamma만으로 Gibbs 분포 전체 결정 | $\mu_0$-분모 하한과 내부 선택이 빠짐 |
| 최소집합 $A_*$ 내부 분포 | $\beta_n(E_n-E_0)$ scale에 의존 |
| CE 경로공간 Gamma 버전 | l.s.c., equicoercivity, positive-mass recovery 확인 필요 |

## 9. 결론

결론적으로 Gamma 수렴은 에너지 최소화의 극한을, Gibbs/Laplace 농축은 온도와 정규화까지 포함한 측도 극한을 다룬다. CE에 이 구조를 적용하려면 경로공간의 topology, lower semicontinuity, equicoercivity, 기준측도와 양질량 recovery를 외부 가정 또는 검증 입력으로 명시해야 하며, 이 문서는 그 다리가 충족되었다고 주장하지 않는다.

에너지가 변할 때의 핵심은 다음이다.

$$
\text{gap}
\quad\text{vs}\quad
\text{recovery mass}
\quad\text{vs}\quad
\beta_n\text{-amplified residual}.
$$

유일 minimizer 바깥으로 빠지는 질량은 gap이 이긴다. 그러나 minimizer 근방을 지탱할 $\mu_0$-양질량 recovery가 없으면 Gibbs 분모가 닫히지 않는다.
