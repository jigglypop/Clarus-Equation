# 04a. Markov/Kleisli 위치 정리

## 0. 목표

04장은 `PreEq_fin`을 비음수 커널 범주로 정의했다. 이 문서는 그 구조가 표준 범주론의 어디에 놓이는지 정리한다.

핵심 판정:

> `PreEq_fin`은 표준 Markov category 자체가 아니라, 정규화되지 않은 finite nonnegative kernel의 범주다. Markov category는 그 안의 row-stochastic 부분범주로 들어온다.

## 1. 세 범주

### 1.1 Weighted kernel category

대상은 공집합이 아닌 유한집합이다. 사상 \(A\to B\)은 각 row가 영이 아닌 비음수 커널이다.

$$
K(a,b)\ge0,\qquad \sum_b K(a,b)>0
$$

이것이 04장의 \(\mathbf{PreEq}_{\mathrm{fin}}\)이다.

### 1.2 Stochastic subcategory

row 합이 1인 커널만 모으면 확률 커널 범주가 된다.

$$
\sum_b K(a,b)=1
$$

이 부분범주를

$$
\mathbf{Stoch}_{\mathrm{fin}}
\subset
\mathbf{PreEq}_{\mathrm{fin}}
$$

라 둔다.

### 1.3 Kleisli 관점

유한집합 \(A\)에 대해

$$
W(A)=\{w:A\to\mathbb R_{\ge0}\}
$$

를 finite weight monad로 보면, 커널 \(K:A\to B\)는 함수

$$
A\to W(B)
$$

다. 따라서 비음수 커널 범주는 weight monad의 Kleisli 범주로 읽을 수 있다.

다만 `PreEq_fin`은 각 row가 영이 아니어야 한다. 이 제한은 정규화와 manifest readout을 위해 추가한 조건이다.

## 2. Stochastic 부분범주 정리

**정리 2.1**  
\(\mathbf{Stoch}_{\mathrm{fin}}\)은 \(\mathbf{PreEq}_{\mathrm{fin}}\)의 wide subcategory다.

**증명.**

대상은 같다. 항등 커널 \(I_A\)는 각 row 합이 1이다.

\(K:A\to B\), \(L:B\to C\)가 stochastic이면

$$
\sum_c(L\circ K)(a,c)
=
\sum_c\sum_b K(a,b)L(b,c)
=
\sum_b K(a,b)\sum_cL(b,c)
=
\sum_b K(a,b)=1
$$

이다. 따라서 합성도 stochastic이다. \(\square\)

## 3. Row mass 분해

`PreEq_fin`의 커널은 stochastic kernel과 row mass로 분해된다.

각 \(a\in A\)에 대해

$$
r_K(a)=\sum_bK(a,b)>0
$$

라 두고

$$
\bar K(a,b)=\frac{K(a,b)}{r_K(a)}
$$

라 두면 \(\bar K\)는 stochastic이다.

따라서

$$
K(a,b)=r_K(a)\bar K(a,b)
$$

이다.

해석:

| 항 | 의미 |
|---|---|
| \(\bar K\) | 조건을 통과한 뒤의 상대적 후보 전이 |
| \(r_K(a)\) | 후보 \(a\)의 생존량, evidence, 억압되지 않은 질량 |

이 row mass 때문에 `PreEq_fin`은 단순한 Markov transition보다 더 많은 정보를 갖는다.

## 4. 왜 row mass를 버리면 안 되는가

상태 \(\mu\)에 대한 정규화 전 작용은

$$
\widetilde K_*\mu(b)=\sum_a\mu(a)K(a,b)
$$

이다. 전체 partition factor는

$$
Z_K(\mu)=\sum_b\widetilde K_*\mu(b)
=
\sum_a\mu(a)r_K(a)
$$

이다.

즉 \(r_K(a)\)는 후보 \(a\)가 조건 아래에서 얼마나 살아남는지 나타내는 에너지 정보다. 이 값이 사라지면 Gibbs 재가중의 핵심이 사라진다.

## 5. Energy kernel과 tropical화

양의 커널이 주어지면 에너지 표현을 둘 수 있다.

$$
E_K(a,b)=-\frac1\beta\log K(a,b)
$$

Gibbs 커널은 반대로

$$
K_\beta(a,b)=e^{-\beta E(a,b)}
$$

다.

두 커널의 합성은

$$
(L_\beta\circ K_\beta)(a,c)
=
\sum_b e^{-\beta(E_1(a,b)+E_2(b,c))}
$$

이고, 따라서

$$
-\frac1\beta\log(L_\beta\circ K_\beta)(a,c)
\to
\min_b(E_1(a,b)+E_2(b,c)).
$$

이것이 `PreEq_fin`에서 tropical/min-plus 구조가 나타나는 정확한 이유다.

## 6. 판정표

| 구조 | 표준 이름 | 이 폴더에서의 지위 |
|---|---|---|
| row-stochastic kernel | finite Markov category | `[정의]`, `[정리]` |
| nonnegative weighted kernel | weight Kleisli category | `[정의]`, `[정리]` |
| nonzero row 제한 | manifest 정규화 조건 | `[공리: 모델 선택]`; 정규화는 `[정리]` |
| Gibbs kernel | energy 기반 weighted kernel | `[정의]` |
| \(\beta\to\infty\) | tropical/min-plus limit | `[정리]` |

## 7. 결론

새 범주를 새로 발명했다고 주장하면 과하다. 정확한 주장은 다음이다.

> 등호 이전 수학은 finite weighted kernel/Kleisli 구조를 후보 상태의 조건 작동으로 해석하고, zero-temperature 극한을 manifest 연산으로 읽는다.
