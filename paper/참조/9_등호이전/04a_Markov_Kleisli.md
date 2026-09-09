# 04a. Markov/Kleisli 위치 정리

이 문서는 `PreEq_fin`을 finite weighted kernel 범주와 weight monad의 Kleisli 관점 안에 위치시킨다. row-stochastic Markov 부분범주, 정규화되지 않은 가중 kernel, CE의 manifest 해석을 같은 것으로 부르지 않고 포함 관계와 추가 선택을 구분한다.

독자는 04장의 대상·사상·합성과 유한 Gibbs kernel을 먼저 읽으면 된다. 세 범주의 정의와 Kleisli 대응, stochastic 부분범주·row mass·tropical 표현을 거쳐 형식 결과와 CE·확률 해석의 경계를 읽는 순서다.

## 0. 목표

04장의 비음수 kernel은 확률 kernel보다 넓고 그 차이는 각 입력 row의 전체 질량에 있다. 여기서는 표준 범주론에서의 위치만 정리하며 Kleisli 표현이 실제 데이터 생성 인과나 CE 물리 시간축을 제공한다고 주장하지 않는다.

04장은 `PreEq_fin`을 비음수 커널 범주로 정의했다. 이 문서는 그 구조가 표준 범주론의 어디에 놓이는지 정리한다.

핵심 판정:

> `PreEq_fin`은 표준 Markov category 자체가 아니라, 정규화되지 않은 finite nonnegative kernel의 범주다. Markov category는 그 안의 row-stochastic 부분범주로 들어온다.

## 1. 세 범주

row 합 조건에 따라 같은 유한 kernel도 서로 다른 범주적 역할을 갖는다. 모든 함수가 가측이고 합이 유한한 전제를 쓰며, 일반 가측공간의 Markov kernel은 확률측도성과 measurability를 별도로 요구한다.

### 1.1 Weighted kernel category

weighted kernel은 입력마다 양의 총 가중치를 남기는 비음수 사상이다. nonzero-row 제한은 weight monad 전체의 성질이 아니라 manifest 정규화를 위해 택한 부분범주 조건이다.

대상은 공집합이 아닌 유한집합이다. 사상 $A\to B$은 각 row가 영이 아닌 비음수 커널이다.

$$
K(a,b)\ge0,\qquad \sum_b K(a,b)>0
$$

이것이 04장의 $\mathbf{PreEq}_{\mathrm{fin}}$이다.

### 1.2 Stochastic subcategory

row 합이 1이면 kernel은 정규화 없이 확률분포를 출력한다. 이는 finite Markov kernel의 형식 모델이지만 독립성·인과 방향·시간 균질성은 stochastic성만으로 나오지 않는다.

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

Kleisli 관점에서는 각 입력 원소가 출력집합 위 비음수 weight 함수 하나를 내보내므로 kernel과 같은 자료가 된다. 전체 질량을 확률·evidence·생존량 중 무엇으로 읽는지는 추가 모델 선택이다.

유한집합 $A$에 대해

$$
W(A)=\{w:A\to\mathbb R_{\ge0}\}
$$

를 finite weight monad로 보면, 커널 $K:A\to B$는 함수

$$
A\to W(B)
$$

다. 따라서 비음수 커널 범주는 weight monad의 Kleisli 범주로 읽을 수 있다.

다만 `PreEq_fin`은 각 row가 영이 아니어야 한다. 이 제한은 정규화와 manifest readout을 위해 추가한 조건이다.

## 2. Stochastic 부분범주 정리

stochastic kernel이 `PreEq_fin` 안에서 합성과 항등에 대해 닫힌다는 것이 포함 관계의 형식 내용이다. 다음 정리는 유한합과 row 합 1만 쓰며 weighted kernel의 정규화가 선형 functor라는 주장은 하지 않는다.

**정리 2.1**  
$\mathbf{Stoch}_{\mathrm{fin}}$은 $\mathbf{PreEq}_{\mathrm{fin}}$의 wide subcategory다.

**증명.**

대상은 같다. 항등 커널 $I_A$는 각 row 합이 1이다.

$K:A\to B$, $L:B\to C$가 stochastic이면

$$
\sum_c(L\circ K)(a,c)
=
\sum_c\sum_b K(a,b)L(b,c)
=
\sum_b K(a,b)\sum_cL(b,c)
=
\sum_b K(a,b)=1
$$

이다. 따라서 합성도 stochastic이다. $\square$

## 3. Row mass 분해

각 row의 양의 전체 질량을 떼면 normalized transition과 input-dependent mass factor를 분리할 수 있다. 분해는 유한 대수 결과이며 row mass의 물리적 의미는 에너지·prior·관측 모델이 추가될 때만 조건부다.

`PreEq_fin`의 커널은 stochastic kernel과 row mass로 분해된다.

각 $a\in A$에 대해

$$
r_K(a)=\sum_bK(a,b)>0
$$

라 두고

$$
\bar K(a,b)=\frac{K(a,b)}{r_K(a)}
$$

라 두면 $\bar K$는 stochastic이다.

따라서

$$
K(a,b)=r_K(a)\bar K(a,b)
$$

이다.

해석:

| 항 | 의미 |
|---|---|
| $\bar K$ | 조건을 통과한 뒤의 상대적 후보 전이 |
| $r_K(a)$ | 후보 $a$의 생존량, evidence, 억압되지 않은 질량 |

이 row mass 때문에 `PreEq_fin`은 단순한 Markov transition보다 더 많은 정보를 갖는다.

## 4. 왜 row mass를 버리면 안 되는가

row mass는 정규화 전 상태의 전체 scale을 바꾸므로 이를 버리면 Gibbs 재가중의 상대 evidence가 사라질 수 있다. 이 보존 이유는 row mass가 관측 가능한 에너지나 실제 원인이라는 결론과 다르다.

상태 $\mu$에 대한 정규화 전 작용은

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

즉 $r_K(a)$는 후보 $a$가 조건 아래에서 얼마나 살아남는지 나타내는 에너지 정보다. 이 값이 사라지면 Gibbs 재가중의 핵심이 사라진다.

## 5. Energy kernel과 tropical화

로그 에너지 표현은 성분이 양수일 때만 유한하게 정의된다. 영 성분·무한 에너지·연속 kernel 적분에서는 확장값, measurability, 적분 조건이 필요하며 아래 극한은 유한합의 형식 결과다.

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

아래 표의 닫힌 범위는 유한 weighted/stochastic kernel의 대수 구조다. 일반 Markov category, 측도론적 Kleisli kernel, CE 물리 모델로의 functor가 이미 구축되었다는 뜻은 아니다.

| 구조 | 표준 이름 | 이 폴더에서의 지위 |
|---|---|---|
| row-stochastic kernel | finite Markov category | `[정의]`, `[정리]` |
| nonnegative weighted kernel | weight Kleisli category | `[정의]`, `[정리]` |
| nonzero row 제한 | manifest 정규화 조건 | `[공리: 모델 선택]`; 정규화는 `[정리]` |
| Gibbs kernel | energy 기반 weighted kernel | `[정의]` |
| $\beta\to\infty$ | tropical/min-plus limit | `[정리]` |

## 7. 결론

닫힌 형식 결과는 standard finite weighted kernel/Kleisli 구조와 stochastic 부분범주의 관계다. CE의 조건 작동·manifest·물리 확률 해석, 일반 가측공간 합성, 측정 인과·실증 식별성은 이 문서가 해결하지 않은 범위다.

새 범주를 새로 발명했다고 주장하면 과하다. 정확한 주장은 다음이다.

> 등호 이전 수학은 finite weighted kernel/Kleisli 구조를 후보 상태의 조건 작동으로 해석하고, zero-temperature 극한을 manifest 연산으로 읽는다.
