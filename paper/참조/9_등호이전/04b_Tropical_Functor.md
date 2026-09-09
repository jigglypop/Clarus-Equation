# 04b. Tropical Functor 극한

이 문서는 유한 energy kernel의 min-plus 합성과 Gibbs kernel의 합성이 유한 온도에서는 다르지만, tropicalization 뒤 zero-temperature에서 점별로 양립함을 보인다. 여기서 functor는 유한 $\beta$의 엄밀한 functor가 아니라 명시한 오차 경계를 가진 점근적 합성 보존이라는 제한된 뜻이다.

독자는 04장의 finite kernel 범주와 log-sum-exp 정리를 먼저 읽으면 된다. energy semiring의 대상·사상·합성을 정의하고 Gibbs deformation 및 tropicalization을 거쳐, 극한 정리와 CE 해석의 미완성 경계를 읽는 순서다.

## 0. 목표

04장의 점별 log-sum-exp 결과를 두 사상의 합성에 적용하려면 유한 중간집합과 같은 $\beta$ 스케일을 고정해야 한다. 이 절은 그 조건 아래의 수학적 극한을 말하며, finite temperature에서 tropical category와 weighted kernel category 사이의 엄밀한 functor를 주장하지 않는다.

04장은 `PreEq_fin`을 비음수 커널 범주로 정의했고, log-sum-exp 극한을 증명했다. 이 문서는 그 결과를 functorial statement로 올린다.

정직한 판정:

> $E\mapsto e^{-\beta E}$는 유한 $\beta$에서 tropical category에서 `PreEq_fin`으로 가는 엄밀한 functor가 아니다. 합성에서 $\min$이 아니라 $\sum$이 생기기 때문이다. 하지만 $-\frac1\beta\log$로 tropicalization한 뒤 $\beta\to\infty$를 보내면 합성과 양립한다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| min-plus energy category | `[정의]`, `[정리]` | 유한집합과 extended energy |
| Gibbs deformation | `[정의]` | $e^{-\beta E}$ 커널 |
| tropicalized composition limit | `[정리]` | log-sum-exp 오차 $\le\log|B|/\beta$ |

## 1. Tropical energy category

min-plus semiring에서 energy의 덧셈은 경로 연결, 최소는 대안 경로 선택에 해당한다. 아래 범주는 유한집합과 extended nonnegative energy에 한정되며, 무한 에너지·연속 경로·측도론적 합성은 추가 가정 없이는 포함되지 않는다.

### 정의 1.1: 대상

대상은 유한 후보 라벨 집합으로 정한다. 이는 모든 최소와 합성이 유한한 선택지에서 정의되게 하는 범위 제한이다.

대상은 공집합이 아닌 유한집합이다.

$$
\operatorname{Ob}(\mathbf{TropPreEq}_{\mathrm{fin}})
=
\{A:0<|A|<\infty\}
$$

### 정의 1.2: 사상

사상은 extended energy kernel이며 각 입력에 적어도 하나의 유한 비용 출력을 요구한다. 이 조건은 Gibbs 변환의 row가 영이 되는 퇴화 반례를 배제하는 정의역 가정이다.

$A\to B$의 사상은 extended energy kernel이다.

$$
E:A\times B\to[0,\infty]
$$

단, 각 $a\in A$에 대해 어떤 $b\in B$가 존재해서

$$
E(a,b)<\infty
$$

이어야 한다.

이 조건은 Gibbs kernel $e^{-\beta E}$의 각 row가 영이 되지 않게 하는 조건이다.

### 정의 1.3: 합성

합성은 중간 후보를 거친 총비용의 최소로 정의한다. 이는 선택한 min-plus semiring의 연산이며 확률적 marginalization과는 유한 온도에서 다르다.

$E:A\to B$, $F:B\to C$의 tropical 합성은

$$
(F\odot E)(a,c)
=
\min_{b\in B}\big(E(a,b)+F(b,c)\big)
$$

이다.

### 정의 1.4: 항등

항등 energy는 같은 후보에 비용 0, 다른 후보에 불가능 비용을 부여한다. 이는 범주 항등의 형식 정의이지 관측 과정의 무변화를 뜻하지 않는다.

항등 energy는

$$
I_A^{\mathrm{trop}}(a,a')
=
\begin{cases}
0, & a=a'\\
\infty, & a\ne a'
\end{cases}
$$

이다.

## 2. 범주 법칙

정의한 tropical 합성이 범주 법칙을 만족하는지는 extended 값의 최소와 유한 중간집합을 써서 확인한다. 다음 정리는 CE 경로나 물리적 최소작용 원리를 증명하지 않는다.

**정리 2.1**  
$\mathbf{TropPreEq}_{\mathrm{fin}}$은 범주를 이룬다.

**증명.**

닫힘: 각 $a$에 대해 $E(a,b)<\infty$인 $b$가 있고, 그 $b$에 대해 $F(b,c)<\infty$인 $c$가 있으므로 $(F\odot E)(a,c)<\infty$인 $c$가 존재한다.

항등법칙:

$$
(E\odot I_A^{\mathrm{trop}})(a,b)
=
\min_{a'\in A}\big(I_A^{\mathrm{trop}}(a,a')+E(a',b)\big)
=
E(a,b)
$$

이고

$$
(I_B^{\mathrm{trop}}\odot E)(a,b)
=
\min_{b'\in B}\big(E(a,b')+I_B^{\mathrm{trop}}(b',b)\big)
=
E(a,b)
$$

이다.

결합법칙:

$$
\big(G\odot(F\odot E)\big)(a,d)
=
\min_{c\in C}
\left[
\min_{b\in B}(E(a,b)+F(b,c))+G(c,d)
\right]
$$

$$
=
\min_{b\in B,c\in C}
\big(E(a,b)+F(b,c)+G(c,d)\big)
$$

이고

$$
\big((G\odot F)\odot E\big)(a,d)
=
\min_{b\in B}
\left[
E(a,b)+\min_{c\in C}(F(b,c)+G(c,d))
\right]
$$

$$
=
\min_{b\in B,c\in C}
\big(E(a,b)+F(b,c)+G(c,d)\big)
$$

이므로 두 값이 같다. $\square$

## 3. Gibbs deformation

Gibbs deformation은 energy를 양의 가중 kernel로 보내는 구성이다. 유한 $\beta$에서 이 구성은 min 합성을 sum 합성으로 정확히 보존하지 않으므로 functor라는 명칭을 그대로 쓸 수 없다는 점이 핵심 경계다.

energy kernel $E:A\to B$와 $\beta>0$에 대해 Gibbs kernel을

$$
G_\beta(E)(a,b)
=
e^{-\beta E(a,b)}
$$

로 둔다. 여기서 $e^{-\beta\infty}=0$이다.

각 row에 finite energy 항이 하나 이상 있으므로 $G_\beta(E)$는 `PreEq_fin`의 사상이다.

하지만 일반적으로

$$
G_\beta(F\odot E)
\ne
G_\beta(F)\circ G_\beta(E)
$$

이다. 왼쪽은 최소 경로만 보고, 오른쪽은 모든 중간 후보 $b$를 합산한다.

## 4. Tropicalization

역방향 tropicalization은 kernel 성분의 로그를 에너지 스케일로 읽는다. 영 성분은 $\infty$로 보내며, 이 정의는 고정된 양의 $\beta$와 성분별 연산에 대한 것이다.

비음수 커널 $K:A\to B$에 대해 $\beta$-tropicalization을

$$
\operatorname{Trop}_\beta(K)(a,b)
=
-\frac1\beta\log K(a,b)
$$

로 둔다. $K(a,b)=0$이면 값은 $\infty$다.

Gibbs kernel에는 정확히

$$
\operatorname{Trop}_\beta(G_\beta(E))=E
$$

가 성립한다.

## 5. Functorial 극한 정리

이제 유한 중간집합에서 log-sum-exp 오차를 정량화하면 합성의 극한이 tropical 합성과 일치한다. 정리는 점별 극한과 $N_{a,c}$에 의존한 경계만 보장하며, 무한 객체·$\beta$ 의존 에너지·교환 불가능한 극한 순서는 포함하지 않는다.

**정리 5.1**  
$E:A\to B$, $F:B\to C$가 $\mathbf{TropPreEq}_{\mathrm{fin}}$의 사상이라고 하자. 그러면 모든 $a\in A,c\in C$에 대해

$$
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)(a,c)
\to
(F\odot E)(a,c)
$$

이다. 더 정확히,

$$
0
\le
(F\odot E)(a,c)
-
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)(a,c)
\le
\frac{\log N_{a,c}}{\beta}
$$

이다. 여기서

$$
N_{a,c}
=
\#\{b\in B:E(a,b)+F(b,c)<\infty\}
$$

이고 $N_{a,c}=0$이면 양쪽은 모두 $\infty$로 읽는다.

**증명.**

$u_b=E(a,b)+F(b,c)$라 두자. 그러면

$$
\big(G_\beta(F)\circ G_\beta(E)\big)(a,c)
=
\sum_{b\in B}e^{-\beta u_b}
$$

이다. finite $u_b$가 없으면 합은 0이고 tropicalized 값은 $\infty$, 또한 $(F\odot E)(a,c)=\infty$다.

이제 finite $u_b$가 있다고 하자. $u_*=\min_bu_b$라 두면

$$
\sum_be^{-\beta u_b}
=
e^{-\beta u_*}
\sum_be^{-\beta(u_b-u_*)}
$$

이다. finite 항의 수를 $N_{a,c}$라 하면

$$
1
\le
\sum_be^{-\beta(u_b-u_*)}
\le
N_{a,c}
$$

이다. 따라서

$$
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)(a,c)
=
u_*-\frac1\beta
\log\sum_be^{-\beta(u_b-u_*)}
$$

이고

$$
0
\le
u_*-
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)(a,c)
\le
\frac{\log N_{a,c}}{\beta}
$$

이다. $u_*=(F\odot E)(a,c)$이므로 정리가 따른다. $\square$

## 6. Asymptotic functoriality

도식의 교환은 각 성분의 $\beta\to\infty$ 극한이라는 의미에서만 성립한다. 따라서 이는 엄밀한 범주 functor의 등식이 아니라 앞 정리의 점근적 합성 호환성을 압축한 표현이다.

정리 5.1은 다음 도식이 $\beta\to\infty$에서 commute한다는 뜻이다.

$$
\begin{array}{ccc}
E,F & \xrightarrow{\quad G_\beta\quad} & G_\beta(E),G_\beta(F)\\
\downarrow \odot && \downarrow \circ\\
F\odot E & \xleftarrow{\quad \operatorname{Trop}_\beta\quad} &
G_\beta(F)\circ G_\beta(E)
\end{array}
$$

즉

$$
\lim_{\beta\to\infty}
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)
=
F\odot E.
$$

이것이 이 문서에서 말하는 tropical functor 극한이다.

## 7. PreEq 해석

soft composition과 min-plus composition의 대응은 계산적 비유로 유용하지만, 최소 경로가 실제로 물리적으로 manifest된다는 결론은 에너지·prior·readout 공리가 더해져야 한다. 표는 형식 변환과 CE 해석을 구별한다.

| finite temperature | zero temperature |
|---|---|
| 모든 중간 후보 $b$가 합산됨 | 최소 에너지 중간 후보만 남음 |
| $\sum_b e^{-\beta(E+F)}$ | $\min_b(E+F)$ |
| `PreEq_fin` kernel 합성 | tropical/min-plus 합성 |

따라서 PreEq의 조건 작동은 $\beta<\infty$에서는 후보 전체의 soft composition이고, $\beta\to\infty$에서는 manifest path의 min-plus composition이다.

## 8. 결론

따라서 닫힌 결론은 유한 energy kernel에서의 tropicalized 합성 극한이다. 일반 CE 경로공간의 functorial 극한, 측도 정규화, 물리적 선택 해석은 이 문서가 완성하지 않은 다리다.

04장의 log-sum-exp 정리는 점별 계산이었다. 이 문서는 그 계산을 범주적 statement로 올린다.

$$
\boxed{
\operatorname{Trop}_\beta
\big(G_\beta(F)\circ G_\beta(E)\big)
\xrightarrow{\beta\to\infty}
F\odot E
}
$$

이로써 `PreEq_fin`은 zero-temperature에서 tropical/min-plus 범주로 내려가는 정확한 극한 도구를 가진다.
