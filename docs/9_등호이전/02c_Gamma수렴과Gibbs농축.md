# 02c. Gamma 수렴과 Gibbs 농축

## 0. 목표

02a는 고정된 에너지 \(E\)에서 non-compact 농축을 닫았다. 02b는 미분/jet에서 scale \(h\)에 따라 에너지 중심이 움직이는 경우를 다뤘다. 이 문서는 더 일반적으로 에너지 자체가 변하는 경우를 다룬다.

$$
E_n \longrightarrow E_0
$$

핵심 질문:

> \(E_n\)으로 만든 Gibbs 측도 \(\mu_n\)는 \(E_0\)의 minimizer로 농축하는가?

정직한 답:

- \(E_n\to E_0\)가 국소균등이고 minimizer가 유일하면, \(\beta_n\to\infty\)만으로도 농축이 닫힌다.
- Gamma 수렴만으로는 분모의 양질량 하한이 자동으로 나오지 않는다. 고정 기준측도 \(\mu_0\)와 맞는 recovery 조건이 필요하다.
- \(E_0\)의 최소집합 내부 선택은 \(\beta_n\)과 \(E_n-E_0\)의 상대 scale이 결정할 수 있다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| 국소균등수렴 + equicoercive + 유일 minimizer | `Exact under assumptions` | 본문 정리 4.1 |
| Gamma 수렴 + positive-mass recovery | `Exact under assumptions` | 본문 정리 5.2 |
| 최소집합 내부 선택 | `Open/Selection` | \(\beta_n(E_n-E_0)\) scale에 의존 |
| jet 농축과의 연결 | `Exact under assumptions` | 02b의 움직이는 중심 정리 |

## 1. 세팅

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

\(E_0\)의 support 위 최소값과 최소집합은

$$
m=\inf_{x\in S}E_0(x),
\qquad
A_*=\operatorname*{argmin}_{x\in S}E_0(x)
$$

이다.

## 2. 왜 02a로 부족한가

02a 정리 6.1은 하나의 고정된 \(E\)에 대해

$$
\beta\to\infty
$$

를 보낸다. 이때 gap은 한 에너지 \(E\) 위에서 고정된다.

하지만 \(E_n\)이 변하면 세 가지가 새로 생긴다.

| 문제 | 의미 |
|---|---|
| minimizer 이동 | \(\operatorname*{argmin}E_n\)이 \(n\)에 따라 움직임 |
| gap 이동 | \(E_n\)의 바깥 gap이 \(n\)에 따라 줄어들 수 있음 |
| 기준측도 문제 | \(\Gamma\)-recovery point 주변에 \(\mu_0\)-질량이 충분하지 않을 수 있음 |

따라서 에너지 수렴만 말하면 부족하고, Gibbs 분모를 지탱할 positive-mass recovery가 필요하다.

## 3. 내부 선택 반례

순수한 극한 에너지 \(E_0\)는 최소집합 내부 선택을 결정하지 못한다.

두 점 공간

$$
A=\{0,1\}
$$

위의 균등측도 \(\mu_0\)를 잡자.

$$
E_n(0)=\frac1{\sqrt n},
\qquad
E_n(1)=0
$$

라 두면

$$
E_n\to E_0,\qquad E_0(0)=E_0(1)=0
$$

이다. 따라서 \(E_0\)의 최소집합은 \(\{0,1\}\) 전체다.

하지만 \(\beta_n=n\)이면

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

- \(E_0\)만 보면 \(0\)과 \(1\)은 구분되지 않는다.
- \(E_n\)의 작은 차이가 \(\beta_n\)에 증폭되어 내부 선택을 만든다.
- 따라서 최소집합 내부 분포는 별도 `Selection` 층이다.

## 4. 국소균등수렴 버전

가장 깨끗한 닫힘은 \(E_n\to E_0\)가 compact 위에서 균등인 경우다.

### 가정 U

1. \(E_0\)는 continuous/coercive다.
2. \(E_0|_S\)의 minimizer는 유일하다.

$$
A_*=\{x_*\}
$$

3. \(E_n\to E_0\)는 모든 compact \(K\subset\mathbb R^d\)에서 균등이다.
4. \(\{E_n\}\)은 equicoercive다. 즉 모든 \(c<\infty\)에 대해 어떤 compact \(K_c\)가 존재해서 충분히 큰 \(n\)에 대해

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

\(x_*\)를 포함하는 임의의 열린집합 \(U\)를 잡는다. \(S\setminus U=\varnothing\)이면 자명하다.

\(S\setminus U\ne\varnothing\)라 하자. 02a 보조정리 3.1과 coercivity 논법에 의해, 충분히 큰 compact \(K\) 안에서 \(E_0\)는 \(S\setminus U\) 위에 양의 gap을 갖는다. 더 구체적으로, 작은 \(\eta>0\)를 택하고 \(V=\{x:E_0(x)<m+\eta\}\cap U\)를 잡으면 \(\mu_0(V)>0\)이다.

equicoercivity로 \(c>m+2\eta\)에 대해 \(\{E_n\le c\}\subset K_c\)인 compact \(K_c\)를 잡는다. compact \(K_c\cap(S\setminus U)\) 위에서는 어떤 \(\delta>0\)에 대해

$$
E_0(x)\ge m+\delta
$$

이다. \(\eta<\delta/4\)로 잡는다.

국소균등수렴에 의해 충분히 큰 \(n\)에 대해 \(K_c\) 위에서

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

compact 바깥에서는 \(E_n>c\)이므로

$$
\frac{
\int_{\mathbb R^d\setminus K_c}e^{-\beta_nE_n(x)}\mu_0(dx)
}{Z_n}
\le
\frac1{\mu_0(V)}
e^{-\beta_n(c-m-2\eta)}
\to0
$$

이다. 따라서 \(\mu_n(\mathbb R^d\setminus U)\to0\), 즉 \(\mu_n(U)\to1\)이다. bounded continuous test function에 대한 근방 분해로

$$
\mu_n\Rightarrow\delta_{x_*}
$$

가 따른다. \(\square\)

**주의.**  
이 정리에는 \(\beta_n\sup_K|E_n-E_0|\to0\) 같은 joint scaling이 필요하지 않다. 유일 minimizer 바깥의 gap \(\delta\)가 고정되어 있고, 균등오차가 결국 \(\delta\)보다 작아지면 \(\beta_n\to\infty\)가 gap을 증폭한다.

## 5. Gamma 수렴 버전

국소균등수렴은 강한 가정이다. 더 약한 표준 도구는 Gamma 수렴이다.

### 정의 5.1: Gamma 수렴

\(E_n\xrightarrow{\Gamma}E_0\)란 다음 두 조건을 뜻한다.

1. liminf: \(x_n\to x\)이면

$$
E_0(x)\le\liminf_nE_n(x_n).
$$

2. recovery: 각 \(x\)에 대해 \(x_n\to x\)이고

$$
\limsup_nE_n(x_n)\le E_0(x)
$$

인 열이 존재한다.

Gamma 수렴은 minimizer의 위치 안정성에는 강하지만, \(\mu_0\)-양질량 분모 하한을 자동으로 주지 않는다.

### 가정 G

1. \(E_n\xrightarrow{\Gamma}E_0\).
2. \(\{E_n\}\)은 equicoercive다.
3. \(E_0|_S\)의 minimizer는 유일하다.

$$
A_*=\{x_*\}
$$

4. positive-mass recovery: \(x_*\)의 임의의 열린근방 \(U\)와 임의의 \(\eta>0\)에 대해, 어떤 Borel 집합 \(V\subset U\)와 \(n_0\)가 존재해서

$$
\mu_0(V)>0
$$

이고 모든 \(n\ge n_0\)에 대해

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

\(x_*\)를 포함하는 열린집합 \(U\)를 잡는다. equicoercivity로 compact 바깥 질량은 정리 4.1과 같은 방식으로 제어할 수 있다. 따라서 충분히 큰 compact \(K\) 안에서 \(K\cap(S\setminus U)\)만 보면 된다.

Gamma liminf와 유일 minimizer를 이용하면 \(K\cap(S\setminus U)\) 위에 uniform gap이 생긴다. 만약 그렇지 않다면, 어떤 부분열과 \(x_n\in K\cap(S\setminus U)\)가 존재해서

$$
E_n(x_n)\to m
$$

이다. compactness로 \(x_n\to x\in K\cap(S\setminus U)\)인 부분열을 잡을 수 있다. Gamma liminf에 의해

$$
E_0(x)\le\liminf_nE_n(x_n)=m
$$

이므로 \(x\)는 \(E_0\)의 minimizer다. 유일성 때문에 \(x=x_*\)이어야 한다. 그러나 \(x\in S\setminus U\)이고 \(x_*\in U\)이므로 모순이다.

따라서 어떤 \(\delta>0\)가 존재해서 충분히 큰 \(n\)에 대해

$$
E_n(x)\ge m+\delta,
\qquad x\in K\cap(S\setminus U)
$$

이다.

positive-mass recovery에서 \(\eta<\delta/4\)를 택하면 \(V\subset U\), \(\mu_0(V)>0\), 그리고 충분히 큰 \(n\)에 대해

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

compact 바깥 질량도 equicoercivity와 같은 분모 하한으로 0이 된다. 따라서 \(\mu_n(U)\to1\)이고 약수렴이 따른다. \(\square\)

## 6. Moving recovery와 속도 조건

가정 G의 recovery 집합 \(V\)가 고정된 양질량 집합이면 속도 조건이 필요 없다. 하지만 recovery가 \(V_n\)처럼 움직이고 \(\mu_0(V_n)\)가 줄어들 수 있으면 속도 조건이 다시 나타난다.

충분조건:

$$
\sup_{x\in V_n}E_n(x)\le m+\eta_n,
\qquad
\mu_0(V_n)>0
$$

이고, 바깥 gap이 \(\delta>0\)일 때

$$
\beta_n(\delta-\eta_n)+\log\mu_0(V_n)\to+\infty
$$

이면 바깥 질량은 0으로 간다.

특히 \(\mu_0(V_n)\)이 \(n\)에 무관하게 양의 하한을 갖고 \(\eta_n<\delta/2\)가 결국 성립하면 충분하다.

## 7. 02b jet 농축과의 관계

02b의 jet 에너지는

$$
E_h(s)=\|s-D_h(x)\|^2,
\qquad
E_0(s)=\|s-J(x)\|^2
$$

형태다. \(D_h(x)\to J(x)\)이면 compact \(K\) 위에서

$$
\sup_{s\in K}|E_h(s)-E_0(s)|
\le
C_K\|D_h(x)-J(x)\|
\to0.
$$

따라서 02b는 정리 4.1의 특수한 예로도 읽을 수 있다. 다만 02b의 직접 증명은 더 강하다. 움직이는 중심 \(D_h(x)\to J(x)\)와 \(\beta_h\to\infty\)만으로

$$
\mu_h\Rightarrow\delta_{J(x)}
$$

가 따라온다. 즉 jet 농축에는 \(\beta_h\|D_h-J\|\to0\) 같은 조건이 필요하지 않다.

## 8. 닫힌 것과 열린 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| 고정 \(E\), \(\beta\to\infty\) | [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md) |
| 국소균등 \(E_n\to E_0\), 유일 minimizer | 정리 4.1 |
| Gamma 수렴 + positive-mass recovery | 정리 5.2 |
| moving recovery 속도 조건 | 6절 |
| jet 농축 | [02b_미분과Jet농축.md](02b_미분과Jet농축.md) |

열린 것:

| 항목 | 이유 |
|---|---|
| 순수 Gamma만으로 Gibbs 분포 전체 결정 | \(\mu_0\)-분모 하한과 내부 선택이 빠짐 |
| 최소집합 \(A_*\) 내부 분포 | \(\beta_n(E_n-E_0)\) scale에 의존 |
| CE 경로공간 Gamma 버전 | l.s.c., equicoercivity, positive-mass recovery 확인 필요 |

## 9. 결론

에너지가 변할 때의 핵심은 다음이다.

$$
\text{gap}
\quad\text{vs}\quad
\text{recovery mass}
\quad\text{vs}\quad
\beta_n\text{-amplified residual}.
$$

유일 minimizer 바깥으로 빠지는 질량은 gap이 이긴다. 그러나 minimizer 근방을 지탱할 \(\mu_0\)-양질량 recovery가 없으면 Gibbs 분모가 닫히지 않는다.
