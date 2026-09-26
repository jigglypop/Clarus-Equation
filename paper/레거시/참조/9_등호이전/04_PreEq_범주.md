# 04. `PreEq` 범주 도구

이 문서는 등호 이전 후보 상태의 변환을 유한 비음수 커널의 범주로 적고, 그 합성과 zero-temperature 극한을 형식적으로 분리한다. 새 범주론을 주장하는 것이 아니라 표준 weighted kernel 구조를 CE의 조건 작동으로 읽을 때 정의·정리와 해석의 경계를 밝히는 것이 목적이다.

독자는 유한집합, 행렬곱, Gibbs 가중을 알고 있으면 된다. 대상과 사상·합성·항등을 정의한 뒤 범주 법칙과 상태 작용을 보이고, tropical 극한 및 Markov/Kleisli와의 관계는 04a에서 읽는다; 가측 공간 확장은 여기서 닫지 않는다.

## 0. 목표

앞 장의 후보 상태와 에너지 작동은 여러 단계로 이어질 수 있으므로, 그 연결을 합성 가능한 수학적 객체로 적을 필요가 있다. 이 절은 유한 커널에 한정되며 CE 동역학·물리 시간·측정 인과를 범주 구조만으로 유도하지 않는다.

이 장은 등호 이전 동역학을 범주론적으로 포장하는 최소 도구를 만든다.

정직한 판정:

> `PreEq_fin`은 완전히 새로운 범주론이 아니라 비음수 커널 범주를 등호 이전 후보 상태의 변환으로 읽은 것이다. 새로운 것은 범주의 존재가 아니라 그 해석과 CE bridge 위치다.

## 1. `PreEq_fin`

범주를 이루려면 대상의 범위와 사상의 정의역·공역, 합성과 항등을 고정해야 한다. 유한집합에서는 모든 커널과 유한합이 자동으로 가측·정의되지만, 일반 가측공간에서는 kernel measurability와 적분 가능성을 별도로 가정해야 한다.

### 정의 1.1: 대상

대상은 후보 상태의 라벨을 담는 공집합 아닌 유한집합으로 정의한다. 이는 유한합 증명을 위한 정의역 제한이며 연속 상태·무한 support를 포괄하지 않는다.

`PreEq_fin`의 대상은 공집합이 아닌 유한집합이다.

$$
\operatorname{Ob}(\mathbf{PreEq}_{\mathrm{fin}})
=
\{A:\;0<|A|<\infty\}
$$

### 정의 1.2: 사상

사상은 입력 후보마다 영이 아닌 비음수 가중치를 출력 후보에 배정하는 kernel이다. row의 양의 질량 조건은 정규화를 위한 모델 선택이며 row 합 1의 확률 kernel과는 구별된다.

$A\to B$의 사상은 비음수 커널이다.

$$
K:A\times B\to\mathbb R_{\ge0}
$$

단, 각 $a\in A$에 대해

$$
\sum_{b\in B}K(a,b)>0
$$

이어야 한다.

조건 에너지 $E:A\times B\to\mathbb R_{\ge0}$와 강도 $\beta$가 주어지면

$$
K_E(a,b)=e^{-\beta E(a,b)}
$$

는 사상의 예다.

### 정의 1.3: 합성

두 단계 조건 작동은 중간 후보를 합산해 합성한다. 유한성 덕분에 수렴 문제가 없으며, 일반 kernel 합성에는 measurable kernel과 Tonelli/Fubini 조건이 필요하다.

$K:A\to B$, $L:B\to C$의 합성은

$$
(L\circ K)(a,c)
=
\sum_{b\in B}K(a,b)L(b,c)
$$

로 둔다.

### 정의 1.4: 항등사상

항등사상은 후보를 바꾸지 않는 대각 kernel이다. 이 정의는 유한 라벨 공간의 항등이며 정규화 연산 자체가 항등이라는 뜻은 아니다.

항등사상은 Kronecker delta 커널이다.

$$
I_A(a,a')
=
\begin{cases}
1, & a=a'\\
0, & a\ne a'
\end{cases}
$$

## 2. 범주 법칙

앞 절의 정의가 실제 범주가 되는지 확인하려면 합성의 닫힘과 항등·결합 법칙을 검증해야 한다. 다음 정리는 비음수성과 각 row의 양의 합을 쓰며 energy kernel의 물리적 의미나 최적 경로의 유일성은 말하지 않는다.

**정리 2.1**  
위 정의에서 `PreEq_fin`은 범주를 이룬다.

**증명.**

합성의 닫힘: $K,L$이 비음수이므로

$$
(L\circ K)(a,c)=\sum_{b\in B}K(a,b)L(b,c)\ge0
$$

이다. 또한 각 $a$에 대해 어떤 $b$는 $K(a,b)>0$이고, 그 $b$에 대해 어떤 $c$는 $L(b,c)>0$이므로 어떤 $c$에 대해 $(L\circ K)(a,c)>0$이다.

항등법칙:

$$
(K\circ I_A)(a,b)
=
\sum_{a'\in A}I_A(a,a')K(a',b)
=K(a,b)
$$

이고

$$
(I_B\circ K)(a,b)
=
\sum_{b'\in B}K(a,b')I_B(b',b)
=K(a,b)
$$

이다.

결합법칙:

$$
\big(M\circ(L\circ K)\big)(a,d)
=
\sum_{c\in C}\sum_{b\in B}K(a,b)L(b,c)M(c,d)
$$

이고

$$
\big((M\circ L)\circ K\big)(a,d)
=
\sum_{b\in B}\sum_{c\in C}K(a,b)L(b,c)M(c,d)
$$

이므로 두 값이 같다. $\square$

## 3. 상태 작용

커널 합성과 확률 상태 갱신은 구별해야 한다. 정규화 전 작용은 선형이고 합성과 양립하지만, 정규화 후 작용은 양의 전체 스칼라를 잊는 사영 확률단체에서만 같은 구조로 읽힌다.

커널 $K:A\to B$는 $A$ 위의 모호함 상태 $\mu$를 $B$ 위의 모호함 상태로 보낸다.

정규화 전:

$$
\widetilde K_*\mu(b)=\sum_{a\in A}\mu(a)K(a,b)
$$

정규화 후:

$$
K_*\mu(b)
=
\frac{\widetilde K_*\mu(b)}
{\sum_{b'\in B}\widetilde K_*\mu(b')}
$$

이다.

**정리 3.1**  
정규화 전 상태 작용은 합성과 양립한다.

$$
\widetilde{(L\circ K)}_*\mu
=
\widetilde L_*(\widetilde K_*\mu)
$$

**증명.**

$$
\widetilde{(L\circ K)}_*\mu(c)
=
\sum_{a\in A}\mu(a)(L\circ K)(a,c)
$$

$$
=
\sum_{a\in A}\mu(a)\sum_{b\in B}K(a,b)L(b,c)
$$

$$
=
\sum_{b\in B}\left(\sum_{a\in A}\mu(a)K(a,b)\right)L(b,c)
=
\widetilde L_*(\widetilde K_*\mu)(c)
$$

이다. $\square$

정규화 후에는 전체 양의 스칼라 배를 같은 상태로 보는 사영 확률단체 위에서 합성과 양립한다.

## 4. Zero-temperature와 tropical 극한

에너지 kernel을 합성하면 중간 경로 에너지가 합해진 유한합이 남는다. 아래 극한은 유한 중간집합과 고정 에너지에서 닫히며, 유한 온도에서 하나의 경로가 선택되거나 연속 경로에서도 같다는 주장은 아니다.

조건 에너지에서 만든 커널

$$
K_\beta(a,b)=e^{-\beta E_1(a,b)}
$$

과

$$
L_\beta(b,c)=e^{-\beta E_2(b,c)}
$$

의 합성은

$$
(L_\beta\circ K_\beta)(a,c)
=
\sum_{b\in B}e^{-\beta(E_1(a,b)+E_2(b,c))}
$$

이다. $\beta\to\infty$에서 지배항은 최소 에너지 경로다.

$$
-\frac1\beta\log (L_\beta\circ K_\beta)(a,c)
\to
\min_{b\in B}\left(E_1(a,b)+E_2(b,c)\right)
$$

## 5. Log-sum-exp 정리

앞 절의 tropical 표현은 유한 개 지수항의 점근식에 의존한다. 무한 합·적분에서는 Laplace 원리 등 추가 정규성 없이는 그대로 사용할 수 없다.

**정리 5.1**  
실수 $u_1,\dots,u_m$에 대해

$$
\lim_{\beta\to\infty}
-\frac1\beta\log\sum_{j=1}^m e^{-\beta u_j}
=
\min_j u_j
$$

이다.

**증명.**

$u_*=\min_j u_j$라 두면

$$
\sum_j e^{-\beta u_j}
=
e^{-\beta u_*}\sum_j e^{-\beta(u_j-u_*)}
$$

이다. 따라서

$$
-\frac1\beta\log\sum_j e^{-\beta u_j}
=
u_*-\frac1\beta\log\sum_j e^{-\beta(u_j-u_*)}
$$

이다. 합 안의 각 항은 $0\le e^{-\beta(u_j-u_*)}\le1$이고 최소항에 대해서는 1이다. 따라서 합은 $1$ 이상 $m$ 이하이다.

$$
0\le
\frac1\beta\log\sum_j e^{-\beta(u_j-u_*)}
\le
\frac{\log m}{\beta}
\to0
$$

이므로 극한은 $u_*$다. $\square$

## 6. 도구 해석

이제 닫힌 유한 kernel 결과를 등호 이전 언어로 해석할 수 있지만, 대응표는 수학적 동형이나 물리 실재의 증명이 아니다. Markov·Kleisli·tropical 관계와 CE bridge의 추가 조건은 연결 문서에서 분리하며, 필요한 상태공간·측도·동역학 입력 전에는 미완성이다.

`PreEq_fin`의 세 층:

| 층 | 수학 | 등호 이전 해석 |
|---|---|---|
| 커널 $K$ | 비음수 행렬 | 후보 상태 전이 |
| Gibbs 커널 $e^{-\beta E}$ | 에너지 기반 커널 | 조건 작동 |
| tropical 극한 | min-plus 합성 | 완전 manifest 경로 |

이 구조는 Markov category, Kleisli category, tropical category와 친화적이다. 다만 이 폴더에서는 우선 정의와 정리가 닫힌 유한 커널 범주만 사용한다.

표준 범주론과의 정확한 위치는 [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md)에서 분리해 정리한다. Gibbs kernel 합성이 zero-temperature에서 tropical/min-plus 합성으로 내려가는 functorial 극한은 [04b_Tropical_Functor.md](04b_Tropical_Functor.md)에 둔다.
