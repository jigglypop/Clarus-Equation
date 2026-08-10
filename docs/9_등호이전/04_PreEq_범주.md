# 04. `PreEq` 범주 도구

## 0. 목표

이 장은 등호 이전 동역학을 범주론적으로 포장하는 최소 도구를 만든다.

정직한 판정:

> `PreEq_fin`은 완전히 새로운 범주론이 아니라 비음수 커널 범주를 등호 이전 후보 상태의 변환으로 읽은 것이다. 새로운 것은 범주의 존재가 아니라 그 해석과 CE bridge 위치다.

## 1. `PreEq_fin`

### 정의 1.1: 대상

`PreEq_fin`의 대상은 공집합이 아닌 유한집합이다.

$$
\operatorname{Ob}(\mathbf{PreEq}_{\mathrm{fin}})
=
\{A:\;0<|A|<\infty\}
$$

### 정의 1.2: 사상

\(A\to B\)의 사상은 비음수 커널이다.

$$
K:A\times B\to\mathbb R_{\ge0}
$$

단, 각 \(a\in A\)에 대해

$$
\sum_{b\in B}K(a,b)>0
$$

이어야 한다.

조건 에너지 \(E:A\times B\to\mathbb R_{\ge0}\)와 강도 \(\beta\)가 주어지면

$$
K_E(a,b)=e^{-\beta E(a,b)}
$$

는 사상의 예다.

### 정의 1.3: 합성

\(K:A\to B\), \(L:B\to C\)의 합성은

$$
(L\circ K)(a,c)
=
\sum_{b\in B}K(a,b)L(b,c)
$$

로 둔다.

### 정의 1.4: 항등사상

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

**정리 2.1**  
위 정의에서 `PreEq_fin`은 범주를 이룬다.

**증명.**

합성의 닫힘: \(K,L\)이 비음수이므로

$$
(L\circ K)(a,c)=\sum_{b\in B}K(a,b)L(b,c)\ge0
$$

이다. 또한 각 \(a\)에 대해 어떤 \(b\)는 \(K(a,b)>0\)이고, 그 \(b\)에 대해 어떤 \(c\)는 \(L(b,c)>0\)이므로 어떤 \(c\)에 대해 \((L\circ K)(a,c)>0\)이다.

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

이므로 두 값이 같다. \(\square\)

## 3. 상태 작용

커널 \(K:A\to B\)는 \(A\) 위의 모호함 상태 \(\mu\)를 \(B\) 위의 모호함 상태로 보낸다.

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

이다. \(\square\)

정규화 후에는 전체 양의 스칼라 배를 같은 상태로 보는 사영 확률단체 위에서 합성과 양립한다.

## 4. Zero-temperature와 tropical 극한

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

이다. \(\beta\to\infty\)에서 지배항은 최소 에너지 경로다.

$$
-\frac1\beta\log (L_\beta\circ K_\beta)(a,c)
\to
\min_{b\in B}\left(E_1(a,b)+E_2(b,c)\right)
$$

## 5. Log-sum-exp 정리

**정리 5.1**  
실수 \(u_1,\dots,u_m\)에 대해

$$
\lim_{\beta\to\infty}
-\frac1\beta\log\sum_{j=1}^m e^{-\beta u_j}
=
\min_j u_j
$$

이다.

**증명.**

\(u_*=\min_j u_j\)라 두면

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

이다. 합 안의 각 항은 \(0\le e^{-\beta(u_j-u_*)}\le1\)이고 최소항에 대해서는 1이다. 따라서 합은 \(1\) 이상 \(m\) 이하이다.

$$
0\le
\frac1\beta\log\sum_j e^{-\beta(u_j-u_*)}
\le
\frac{\log m}{\beta}
\to0
$$

이므로 극한은 \(u_*\)다. \(\square\)

## 6. 도구 해석

`PreEq_fin`의 세 층:

| 층 | 수학 | 등호 이전 해석 |
|---|---|---|
| 커널 \(K\) | 비음수 행렬 | 후보 상태 전이 |
| Gibbs 커널 \(e^{-\beta E}\) | 에너지 기반 커널 | 조건 작동 |
| tropical 극한 | min-plus 합성 | 완전 manifest 경로 |

이 구조는 Markov category, Kleisli category, tropical category와 친화적이다. 다만 이 폴더에서는 우선 정의와 정리가 닫힌 유한 커널 범주만 사용한다.

표준 범주론과의 정확한 위치는 [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md)에서 분리해 정리한다. Gibbs kernel 합성이 zero-temperature에서 tropical/min-plus 합성으로 내려가는 functorial 극한은 [04b_Tropical_Functor.md](04b_Tropical_Functor.md)에 둔다.
