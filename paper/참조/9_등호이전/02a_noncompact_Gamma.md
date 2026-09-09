# 02a. Non-compact 농축 정리

이 문서는 compact성에 기대지 않고 Gibbs 재가중이 언제 하나의 최소점으로 농축하는지를 증명한다. 목적은 후보공간이 무한히 뻗을 때도 질량의 도피를 막는 정확한 가정을 분리하고, 그 결과를 CE 경로공간에 적용하려면 무엇이 아직 비어 있는지를 드러내는 데 있다.

독자는 확률측도, 약수렴, 연속함수의 최소화에 익숙하되 Gamma 수렴은 처음 접해도 된다. 먼저 고정 에너지의 non-compact 정리를 닫고, 다음으로 여러 최소점과 예제를 거쳐, 마지막에 scale 의존 에너지와 CE 적용이 별도의 미완성 다리임을 읽는 순서다.

## 0. 목표

02장은 compact metric space에서 농축을 증명했지만, 실제 후보공간은 흔히 무한히 멀리 갈 수 있다. 여기서는 연속성과 coercivity가 그 도피를 어떻게 차단하는지 고정 에너지 문제에서 먼저 보이며, Gamma 수렴은 그 뒤에 필요한 별도 도구로 남긴다.

핵심 정리:

> $A=\mathbb R^n$, $E$가 continuous/coercive이고 $\operatorname{supp}\mu_0$ 위의 minimizer가 유일하면, Gibbs 재가중 $\mu_\beta$는 그 minimizer의 Dirac 측도로 약수렴한다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| $\mathbb R^n$ coercive 농축 | `[정리]` | 직접 증명 가능 |
| 여러 minimizer로의 농축 | `[정리]` | 열린 근방 농축으로 닫힘 |
| Gamma-convergence | 조건부 `[정리]`; 일반 scale-dependent Gibbs는 `[미완성]` | 02c에서 분리 |

## 1. 세팅

목표의 정리가 어떤 정의역에서 성립하는지 먼저 고정한다. 이 절의 $\mathbb R^n$ 및 Borel 확률측도는 선택한 수학적 세팅이며, 더 일반적인 Polish 공간으로의 확장은 자동으로 따라오지 않는다.

후보공간은

$$
A=\mathbb R^n
$$

이다. 초기 모호함 상태는 Borel 확률측도

$$
\mu_0\in\mathcal P(\mathbb R^n)
$$

이다. 그 support를

$$
S=\operatorname{supp}\mu_0
$$

라 둔다.

조건 에너지는 연속함수

$$
E:\mathbb R^n\to\mathbb R_{\ge0}
$$

이고 coercive라고 가정한다.

$$
E(x)\to\infty\qquad \|x\|\to\infty
$$

조건 강도 $\beta\ge0$에서 Gibbs 재가중은

$$
\mu_\beta(dx)
=
\frac{e^{-\beta E(x)}}{Z_\beta}\mu_0(dx),
\qquad
Z_\beta=\int_{\mathbb R^n}e^{-\beta E(x)}\mu_0(dx)
$$

이다.

$E\ge0$이고 $\mu_0$가 확률측도이므로

$$
0<Z_\beta\le1
$$

이다.

## 2. 최소점 존재

이제 coercivity가 단순한 성장 조건을 넘어 최소점의 존재를 보장함을 확인한다. 다음 보조정리는 연속성, 닫힌 support, 유계 sublevel set을 함께 쓰는 정리이며, 이 중 하나라도 빠지면 최소화열이 공간 밖으로 달아날 수 있다.

최소값을

$$
m=\inf_{x\in S}E(x)
$$

라 둔다.

**보조정리 2.1**  
$E$가 continuous/coercive이면 $E|_S$는 최소값을 갖는다. 즉 어떤 $x_0\in S$에 대해

$$
E(x_0)=m
$$

이다.

**증명.**

정의에 의해 $E(x_k)\to m$인 열 $x_k\in S$를 잡을 수 있다. $E$가 coercive이므로 어떤 $R>0$가 존재해서

$$
\|x\|>R\quad\Rightarrow\quad E(x)>m+1
$$

이다. 따라서 충분히 큰 $k$에 대해 $x_k\in S\cap\overline B_R$이다.

$S$는 닫힌집합이고 $\overline B_R$는 compact이므로 $S\cap\overline B_R$는 compact이다. 부분열을 잡아

$$
x_{k_j}\to x_0\in S\cap\overline B_R
$$

라 할 수 있다. $E$의 연속성으로

$$
E(x_0)=\lim_jE(x_{k_j})=m
$$

이다. $\square$

## 3. Gap 보조정리

최소점이 존재한다고 해서 아직 농축 속도를 말할 수는 없다. 유일 최소점이라는 추가 가정 아래에서는 그 점을 피한 닫힌 부분에서 양의 에너지 간격이 생기며, 다음 보조정리가 그 간격의 정리적 근거가 된다.

$$
A_*=\operatorname*{argmin}_{x\in S}E(x)=\{x_*\}
$$

**보조정리 3.1**  
$U\subset\mathbb R^n$가 $x_*$를 포함하는 열린집합이면, $S\setminus U$가 비어 있지 않은 경우 어떤 $\delta_U>0$가 존재해서

$$
E(x)\ge m+\delta_U,
\qquad x\in S\setminus U
$$

이다.

**증명.**

$F=S\setminus U$라 두자. $S$는 닫혀 있고 $U$는 열려 있으므로 $F$는 닫혀 있다.

반대로

$$
\inf_{x\in F}E(x)=m
$$

이라고 가정하자. 그러면 $y_k\in F$이고 $E(y_k)\to m$인 열을 잡을 수 있다. coercivity 때문에 충분히 큰 $k$에 대해 $y_k$는 어떤 compact ball 안에 갇힌다. 따라서 부분열을 잡아

$$
y_{k_j}\to y_\infty
$$

라 할 수 있다.

$F$가 닫혀 있으므로 $y_\infty\in F$이다. $E$의 연속성으로

$$
E(y_\infty)=m
$$

이다. 따라서 $y_\infty$는 minimizer다. minimizer는 유일하므로 $y_\infty=x_*$이다. 하지만 $x_*\in U$이고 $F=S\setminus U$이므로 $x_*\notin F$다. 모순이다.

따라서

$$
\delta_U=\inf_{x\in F}E(x)-m>0
$$

이다. $\square$

## 4. Partition function 하한

분자에서 얻은 gap을 확률측도 비율로 바꾸려면 Gibbs 분모가 지나치게 작아지지 않아야 한다. support의 정의와 에너지 연속성이 최소점 근방에 양의 초기 질량을 준다는 사실이 이 하한의 유일한 입력이다.

**보조정리 4.1**  
임의의 $\eta>0$에 대해

$$
V_\eta=\{x:E(x)<m+\eta\}
$$

는 양의 $\mu_0$-질량을 가진다.

$$
\mu_0(V_\eta)>0
$$

따라서

$$
Z_\beta
\ge
e^{-\beta(m+\eta)}\mu_0(V_\eta)
$$

이다.

**증명.**

$E(x_*)=m$이고 $E$는 연속이므로 $V_\eta$는 $x_*$를 포함하는 열린집합이다. $x_*\in S=\operatorname{supp}\mu_0$이므로 $x_*$의 모든 열린 근방은 양의 $\mu_0$-질량을 가진다. 따라서 $\mu_0(V_\eta)>0$이다.

또한 $V_\eta$ 위에서는 $E(x)<m+\eta$이므로

$$
e^{-\beta E(x)}
\ge
e^{-\beta(m+\eta)}
$$

이다. 따라서

$$
Z_\beta
=
\int e^{-\beta E(x)}\mu_0(dx)
\ge
\int_{V_\eta}e^{-\beta E(x)}\mu_0(dx)
\ge
e^{-\beta(m+\eta)}\mu_0(V_\eta)
$$

이다. $\square$

## 5. Tightness 보조정리

non-compact 공간에서는 최소점 근방의 수렴 주장과 질량이 무한대로 새지 않는다는 주장을 구별해야 한다. 다음 정리는 임의의 양의 하한 $\beta_0$ 이후의 측도족에 compact한 포획집합이 있음을 보이는 형식적 결과이며, 수치 실험의 안정성 주장은 아니다.

**정리 5.1**  
$E$가 continuous/coercive이면, 임의의 $\beta_0>0$에 대해 측도열

$$
\{\mu_\beta:\beta\ge\beta_0\}
$$

은 tight하다.

**증명.**

$\epsilon>0$을 잡는다. $\eta>0$를 하나 고정하고

$$
V_\eta=\{x:E(x)<m+\eta\}
$$

라 둔다. 보조정리 4.1에 의해 $\mu_0(V_\eta)>0$이다.

$c>m+\eta$를 잡고

$$
K_c=\{x:E(x)\le c\}
$$

라 두자. $E$가 continuous/coercive이므로 $K_c$는 compact이다.

$K_c^c$에서는 $E(x)>c$이므로

$$
\int_{K_c^c}e^{-\beta E(x)}\mu_0(dx)
\le
e^{-\beta c}
$$

이다. 한편 보조정리 4.1에 의해

$$
Z_\beta
\ge
e^{-\beta(m+\eta)}\mu_0(V_\eta)
$$

이다. 따라서

$$
\mu_\beta(K_c^c)
\le
\frac{e^{-\beta(c-m-\eta)}}{\mu_0(V_\eta)}
$$

이다. $\beta\ge\beta_0$에서 오른쪽을 균일하게 작게 만들기 위해 $c$를 충분히 크게 잡으면

$$
\mu_\beta(K_c)\ge1-\epsilon,
\qquad \beta\ge\beta_0
$$

가 된다. 따라서 $\{\mu_\beta:\beta\ge\beta_0\}$는 tight하다. $\square$

## 6. Non-compact 유일 manifest 정리

앞의 최소점, gap, 분모 하한을 결합하면 고정된 에너지에서의 핵심 결론에 도달한다. 이 정리는 support 위의 유일 최소점과 coercivity를 전제로 한 약수렴 정리이며, CE의 물리적 선택 메커니즘 자체를 증명하는 명제는 아니다.

**정리 6.1**  
$\mu_0\in\mathcal P(\mathbb R^n)$, $E:\mathbb R^n\to\mathbb R_{\ge0}$가 continuous/coercive라고 하자. 또한

$$
\operatorname*{argmin}_{x\in\operatorname{supp}\mu_0}E(x)=\{x_*\}
$$

라고 하자. 그러면

$$
\mu_\beta\Rightarrow\delta_{x_*}
\qquad(\beta\to\infty)
$$

이다.

**증명.**

먼저 $x_*$를 포함하는 임의의 열린집합 $U$를 잡는다. $S\setminus U=\varnothing$이면 $\mu_0(\mathbb R^n\setminus U)=0$이므로 모든 $\beta$에 대해 $\mu_\beta(U)=1$이다.

이제 $S\setminus U\ne\varnothing$라 하자. 보조정리 3.1에 의해 어떤 $\delta_U>0$가 존재해서

$$
E(x)\ge m+\delta_U,
\qquad x\in S\setminus U
$$

이다.

$\eta=\delta_U/2$로 둔다. 그러면 분자는

$$
\int_{\mathbb R^n\setminus U}e^{-\beta E(x)}\mu_0(dx)
=
\int_{S\setminus U}e^{-\beta E(x)}\mu_0(dx)
\le
e^{-\beta(m+\delta_U)}\mu_0(S\setminus U)
$$

이고, 보조정리 4.1에 의해 분모는

$$
Z_\beta
\ge
e^{-\beta(m+\delta_U/2)}\mu_0(V_{\delta_U/2})
$$

이다. 따라서

$$
\mu_\beta(\mathbb R^n\setminus U)
\le
\frac{\mu_0(S\setminus U)}{\mu_0(V_{\delta_U/2})}
e^{-\beta\delta_U/2}
\to0
$$

이다. 즉

$$
\mu_\beta(U)\to1
$$

이다.

이제 약수렴을 보인다. $f:\mathbb R^n\to\mathbb R$를 bounded continuous 함수라 하자. $\epsilon>0$을 잡는다. 연속성으로 어떤 열린 근방 $U\ni x_*$가 존재해서

$$
|f(x)-f(x_*)|<\epsilon,
\qquad x\in U
$$

이다. 그러면

$$
\left|\int f\,d\mu_\beta-f(x_*)\right|
\le
\epsilon
+2\|f\|_\infty\,\mu_\beta(\mathbb R^n\setminus U)
$$

이고, 오른쪽은 $\limsup_{\beta\to\infty}$에서 $\epsilon$ 이하가 된다. $\epsilon$은 임의이므로

$$
\int f\,d\mu_\beta\to f(x_*)
$$

이다. 따라서 $\mu_\beta\Rightarrow\delta_{x_*}$이다. $\square$

## 7. 여러 minimizer 버전

유일성 가정이 사라지면 어느 하나의 Dirac 측도로 수렴한다는 결론은 일반적으로 거짓이다. 이때 형식적으로 닫히는 결론은 최소집합을 포함하는 열린 근방으로의 농축뿐이며, 최소집합 내부의 가중치는 초기 측도와 더 높은 차수 구조에 의존한다.

$$
A_*=\operatorname*{argmin}_{x\in S}E(x)
$$

**정리 7.1**  
위 세팅에서 $A_*$가 여러 점을 가질 수 있다고 하자. $U$가 $A_*$를 포함하는 열린집합이면

$$
\mu_\beta(U)\to1
$$

이다.

**증명.**

coercivity와 연속성 때문에 $A_*$는 공집합이 아닌 compact 집합이다. $F=S\setminus U$에서 $\inf_FE=m$이라고 가정하면, 보조정리 3.1의 compactness 논리와 같은 방식으로 $F$ 안에 minimizer가 존재한다. 이는 $U\supset A_*$와 모순이다. 따라서 $F$ 위에는 양의 gap이 있고, 같은 분자/분모 평가로 $\mu_\beta(\mathbb R^n\setminus U)\to0$이다. $\square$

## 8. 예제

가장 단순한 이차 에너지는 정리의 각 가정이 어떤 역할을 하는지 확인하게 한다. 이 예제는 정리의 적용 사례일 뿐, 일반 non-compact 시스템의 rate나 유한-$\beta$ 오차를 산출하지는 않는다.

실수 전체를 후보공간으로 둔다.

$$
A=\mathbb R,\qquad
E(x)=(x-1)^2
$$

임의의 Borel 확률측도 $\mu_0$가 $1$을 support에 포함하고, $1$이 support 위의 유일한 minimizer라고 하자. 예를 들어 $\mu_0$가 모든 열린구간에 양의 질량을 주는 정규분포라면 조건이 성립한다.

$E$는 continuous/coercive이고

$$
\operatorname*{argmin}_{x\in\operatorname{supp}\mu_0}E(x)=\{1\}
$$

이다. 따라서

$$
\mu_\beta\Rightarrow\delta_1
$$

이다.

## 9. Gamma-convergence의 위치

여기까지의 증명은 하나의 고정된 $E$만 다루므로, scale과 함께 에너지가 변하는 문제에 그대로 이식할 수 없다. Gamma 수렴은 변하는 함수열의 변분적 극한을 통제하는 조건부 도구이고, Gibbs 분포의 분모 및 초기 질량을 통제하는 가정은 그 정의만으로 제공하지 않는다.

$$
E_\beta,\quad E_n,\quad E_\ell
$$

처럼 scale에 따라 변하면 다른 문제가 된다. 그때 필요한 도구가 Gamma-convergence다.

확인 항목:

| 확인 항목 | 의미 |
|---|---|
| liminf 부등식 | $x_n\to x$이면 $E_\infty(x)\le\liminf E_n(x_n)$ |
| recovery sequence | 각 $x$에 대해 $x_n\to x$, $E_n(x_n)\to E_\infty(x)$인 열 존재 |
| equicoercivity | $E_n$의 sublevel set들이 함께 compact하게 갇힘 |

scale에 따라 변하는 Gibbs 농축은 [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)에서 분리해 다룬다. 핵심은 순수 Gamma 수렴만이 아니라, Gibbs 분모를 지탱하는 $\mu_0$-양질량 recovery 조건까지 함께 확인해야 한다는 점이다.

02a의 첫 닫힘은 정리 6.1, 즉 고정된 $E$에 대한 non-compact 농축이다.

## 10. CE bridge 판정표

CE에서 $A=\mathcal P_I$ 같은 경로공간을 쓰려면 유클리드 공간에 대한 정리 6.1을 이름만 바꾸어 적용할 수 없다. 다음 표는 필요한 공간·측도·에너지 조건을 점검하는 미완성 다리의 명세이며, 표의 조건이 충족되었다는 증거나 경로공간 정리를 이 문서는 제공하지 않는다.

| 항목 | 질문 | 판정 |
|---|---|---|
| 공간 | 경로공간이 $\mathbb R^n$ 또는 충분히 좋은 Polish space인가 | `필수` |
| support | 초기 측도 support가 정의되어 있는가 | `필수` |
| 에너지 | $E_{\mathrm{fold}}$가 continuous 또는 l.s.c.인가 | `필수` |
| escape | coercive/equicoercive 조건이 있는가 | `필수` |
| minimizer | 선택 경로가 유일한가, 아니면 최소집합인가 | `필수` |

이 표가 비면 CE 경로공간으로 가는 물리 사상은 `[미완성]`이다. 표가 채워지면 그때 non-compact 농축 정리를 CE 경로공간 위로 옮길 수 있다.

## 11. 결론

따라서 이 문서에서 닫힌 범위는 고정된 연속 coercive 에너지에 한정된다. 이 결과는 명시한 가정 아래의 해석학적 `[정리]`이고, noncompact CE 경로공간의 모델화·Gamma 극한·물리적 해석은 별도로 반증 가능 조건을 채워야 하는 `[미완성]` 다리다.

$$
E\ \mathrm{coercive}
\quad+\quad
\operatorname*{argmin}_{\operatorname{supp}\mu_0}E=\{x_*\}
\quad\Longrightarrow\quad
\mu_\beta\Rightarrow\delta_{x_*}
$$

이것은 새 물리 사상이 아니라 표준 해석학 위에서 가정이 명시된 `[정리]`다.
