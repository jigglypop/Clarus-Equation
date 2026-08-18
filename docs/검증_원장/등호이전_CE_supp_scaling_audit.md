# 05j. CE Suppression Scaling

## 0. 목표

같은 suppression 기호라도 지수에서 $\beta$와 결합하는 방식에 따라
zero-temperature 극한이 달라진다. 이 문서는 외부 물리 식을 판정하지 않고,
세 scaling의 정확한 측도론만 고정한다.

| label | 가중치 | zero-temperature 결론 |
|---|---|---|
| supp:stat | $e^{-\beta\mathcal I_0-B}$ | bounded $B$는 $\mathcal I_0$의 최소집합을 바꾸지 않음 |
| supp:dyn | $e^{-\beta(\mathcal I_0+\mathcal J)}$ | $\mathcal I_0+\mathcal J$의 최소집합으로 농축 |
| supp:hard | $\mathbf1_Ae^{-\beta\mathcal I_0}$ | $A$ 안의 최소집합으로 농축; 조건부 측도의 존재 필요 |

$\mathcal I_0,\mathcal J,B$와 $\beta$는 모두 무차원이다.

## 1. 공통 세팅

**[정의]** $X$를 metric space, $\mu_0\in\mathcal P(X)$를 Borel
probability라 하자. 무차원 functional
$\mathcal I_0:X\to\mathbb R\cup\{+\infty\}$가
[../9_등호이전/05_CE_브리지.md](../9_등호이전/05_CE_브리지.md) 1절의 good-rate와 recovery mass
조건을 만족한다고 하자. 기준 Gibbs measure를

$$
\mu_\beta^0(dx)
=
\frac{e^{-\beta\mathcal I_0(x)}}{Z_\beta^0}\,\mu_0(dx)
$$

로 둔다.

## 2. Bounded tilt

**[정리]** $B:X\to\mathbb R$가 measurable이고
$m\leq B\leq M$이라 하자.

$$
\mu_\beta^B(dx)
=
\frac{e^{-\beta\mathcal I_0(x)-B(x)}}{Z_\beta^B}\,\mu_0(dx)
$$

이면 $\mathcal I_0$의 최소집합 $M_0$의 모든 열린 이웃
$U\supset M_0$에 대해

$$
\mu_\beta^B(X\setminus U)\to0.
$$

**증명.** 모든 measurable $A\subset X$에 대해
$$
\mu_\beta^B(A)
\leq e^{M-m}\mu_\beta^0(A).
$$
실제로 분자는 $e^{-m}$배 이하이고 partition function은
$e^{-M}Z_\beta^0$ 이상이다. 기준 measure의 농축 정리를 적용한다.
$\square$

따라서 $\beta$와 함께 커지지 않는 bounded tilt는 유한 $\beta$의
상대 가중치는 바꾸지만 zero-temperature 최소집합은 바꾸지 않는다.
비유계 $B$에는 이 정리를 적용할 수 없다.

## 3. $\beta$-coupled suppression

**[정의]** 무차원 lower-semicontinuous
$\mathcal J:X\to\mathbb R\cup\{+\infty\}$에 대해

$$
\mu_\beta^{\rm dyn}(dx)
=
\frac{e^{-\beta(\mathcal I_0(x)+\mathcal J(x))}}
{Z_\beta^{\rm dyn}}\,\mu_0(dx)
$$

로 둔다.

**[정리]** $\mathcal I_0+\mathcal J$가 good-rate이고 그 최소 높이
근방에 $\mu_0$-recovery mass가 있으면

$$
\mu_\beta^{\rm dyn}
$$

은
$$
M_{\rm dyn}
=
\operatorname*{argmin}_X(\mathcal I_0+\mathcal J)
$$
의 모든 열린 이웃에 농축한다.

**증명.** 05 브리지 1절의 정리에
$\mathcal I=\mathcal I_0+\mathcal J$를 대입한다. $\square$

물리적 suppression action $S_{\rm supp}$를 사용할 때에는
$\mathcal J=S_{\rm supp}/\hbar$ 또는 $S_{\rm supp}/S_*$처럼
무차원화한다. $\mathcal J$의 실제 형태는 **[공리: 물리 모형]**이다.

## 4. Hard constraint

**[정의]** $A\subset X$가 measurable이고
$$
0<
\int_Ae^{-\beta\mathcal I_0}\,d\mu_0
<\infty
$$
라 하자. 조건부 measure를
$$
\mu_\beta^A(dx)
=
\frac{\mathbf1_A(x)e^{-\beta\mathcal I_0(x)}}
{\int_Ae^{-\beta\mathcal I_0}\,d\mu_0}\,\mu_0(dx)
$$
로 둔다.

**[정리]** Restricted space
$$
X_A:=\operatorname{supp}(\mu_0|_A)
$$
위에서 $\mathcal I_0$가 good-rate이고 recovery mass를 가지면
$\mu_\beta^A$는
$$
\operatorname*{argmin}_{X_A}\mathcal I_0
$$
의 모든 열린 이웃에 농축한다.

**증명.** 정규화한 $\mu_0|_A$를 새 prior로 보고 05 브리지 1절을
적용한다. $\square$

$\mu_0(A)=0$이면 이 조건화는 정의되지 않는다. 특히 Brownian
$C^0$ prior를 $W^{1,p}$ finite-kinetic-action 집합으로 조건화하는
경우에는 [../9_등호이전/05i_CE_physical_path_prior.md](../9_등호이전/05i_CE_physical_path_prior.md)
정리 2.2에 의해 분모가 0이다.

## 5. Scale을 바꾸면 생기는 차이

**[산출]**

- supp:stat은 $M_0$를 보존한다.
- supp:dyn은 good-rate·recovery 조건 아래 $M_{\rm dyn}$을 선택한다.
- supp:hard는 원래 최소자가 $X_A$에 없으면 restricted minimizer로
  선택을 바꾼다.

이 세 결론은 서로 대체할 수 없다. 같은 $S_{\rm supp}$ 기호를 쓰더라도
$$
e^{-B},\qquad
e^{-S_{\rm supp}/\hbar},\qquad
\mathbf1_A
$$
중 어느 연산인지 먼저 선언해야 한다.

## 6. CE에 남은 선택

다음은 **[미완성]**이다.

- 실제 CE suppression이 stat, dyn, hard 중 어느 층에 속하는지
- 비유계 suppression의 partition function과 good-rate 조건
- hard event의 양의 prior 질량과 continuum recovery
- finite-$\beta$ 분율을 실험 outcome으로 옮기는 instrument

수치 분율의 일치만으로 이 scale 선택을 정리로 승격하지 않는다.
