# 02a. Non-compact 농축 정리

## 0. 목표

02장은 compact metric space에서 농축을 증명했다. 이 문서는 가장 먼저 외부 검증 가능한 non-compact 정리를 닫는다.

핵심 정리:

> \(A=\mathbb R^n\), \(E\)가 continuous/coercive이고 \(\operatorname{supp}\mu_0\) 위의 minimizer가 유일하면, Gibbs 재가중 \(\mu_\beta\)는 그 minimizer의 Dirac 측도로 약수렴한다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| \(\mathbb R^n\) coercive 농축 | `Exact under assumptions` | 직접 증명 가능 |
| 여러 minimizer로의 농축 | `Exact under assumptions` | 열린 근방 농축으로 닫힘 |
| Gamma-convergence | `Exact under assumptions`/`Open` | scale-dependent Gibbs는 02c에서 분리 |

## 1. 세팅

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

조건 강도 \(\beta\ge0\)에서 Gibbs 재가중은

$$
\mu_\beta(dx)
=
\frac{e^{-\beta E(x)}}{Z_\beta}\mu_0(dx),
\qquad
Z_\beta=\int_{\mathbb R^n}e^{-\beta E(x)}\mu_0(dx)
$$

이다.

\(E\ge0\)이고 \(\mu_0\)가 확률측도이므로

$$
0<Z_\beta\le1
$$

이다.

## 2. 최소점 존재

최소값을

$$
m=\inf_{x\in S}E(x)
$$

라 둔다.

**보조정리 2.1**  
\(E\)가 continuous/coercive이면 \(E|_S\)는 최소값을 갖는다. 즉 어떤 \(x_0\in S\)에 대해

$$
E(x_0)=m
$$

이다.

**증명.**

정의에 의해 \(E(x_k)\to m\)인 열 \(x_k\in S\)를 잡을 수 있다. \(E\)가 coercive이므로 어떤 \(R>0\)가 존재해서

$$
\|x\|>R\quad\Rightarrow\quad E(x)>m+1
$$

이다. 따라서 충분히 큰 \(k\)에 대해 \(x_k\in S\cap\overline B_R\)이다.

\(S\)는 닫힌집합이고 \(\overline B_R\)는 compact이므로 \(S\cap\overline B_R\)는 compact이다. 부분열을 잡아

$$
x_{k_j}\to x_0\in S\cap\overline B_R
$$

라 할 수 있다. \(E\)의 연속성으로

$$
E(x_0)=\lim_jE(x_{k_j})=m
$$

이다. \(\square\)

## 3. Gap 보조정리

이제 minimizer가 유일하다고 하자.

$$
A_*=\operatorname*{argmin}_{x\in S}E(x)=\{x_*\}
$$

**보조정리 3.1**  
\(U\subset\mathbb R^n\)가 \(x_*\)를 포함하는 열린집합이면, \(S\setminus U\)가 비어 있지 않은 경우 어떤 \(\delta_U>0\)가 존재해서

$$
E(x)\ge m+\delta_U,
\qquad x\in S\setminus U
$$

이다.

**증명.**

\(F=S\setminus U\)라 두자. \(S\)는 닫혀 있고 \(U\)는 열려 있으므로 \(F\)는 닫혀 있다.

반대로

$$
\inf_{x\in F}E(x)=m
$$

이라고 가정하자. 그러면 \(y_k\in F\)이고 \(E(y_k)\to m\)인 열을 잡을 수 있다. coercivity 때문에 충분히 큰 \(k\)에 대해 \(y_k\)는 어떤 compact ball 안에 갇힌다. 따라서 부분열을 잡아

$$
y_{k_j}\to y_\infty
$$

라 할 수 있다.

\(F\)가 닫혀 있으므로 \(y_\infty\in F\)이다. \(E\)의 연속성으로

$$
E(y_\infty)=m
$$

이다. 따라서 \(y_\infty\)는 minimizer다. minimizer는 유일하므로 \(y_\infty=x_*\)이다. 하지만 \(x_*\in U\)이고 \(F=S\setminus U\)이므로 \(x_*\notin F\)다. 모순이다.

따라서

$$
\delta_U=\inf_{x\in F}E(x)-m>0
$$

이다. \(\square\)

## 4. Partition function 하한

**보조정리 4.1**  
임의의 \(\eta>0\)에 대해

$$
V_\eta=\{x:E(x)<m+\eta\}
$$

는 양의 \(\mu_0\)-질량을 가진다.

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

\(E(x_*)=m\)이고 \(E\)는 연속이므로 \(V_\eta\)는 \(x_*\)를 포함하는 열린집합이다. \(x_*\in S=\operatorname{supp}\mu_0\)이므로 \(x_*\)의 모든 열린 근방은 양의 \(\mu_0\)-질량을 가진다. 따라서 \(\mu_0(V_\eta)>0\)이다.

또한 \(V_\eta\) 위에서는 \(E(x)<m+\eta\)이므로

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

이다. \(\square\)

## 5. Tightness 보조정리

**정리 5.1**  
\(E\)가 continuous/coercive이면, 임의의 \(\beta_0>0\)에 대해 측도열

$$
\{\mu_\beta:\beta\ge\beta_0\}
$$

은 tight하다.

**증명.**

\(\epsilon>0\)을 잡는다. \(\eta>0\)를 하나 고정하고

$$
V_\eta=\{x:E(x)<m+\eta\}
$$

라 둔다. 보조정리 4.1에 의해 \(\mu_0(V_\eta)>0\)이다.

\(c>m+\eta\)를 잡고

$$
K_c=\{x:E(x)\le c\}
$$

라 두자. \(E\)가 continuous/coercive이므로 \(K_c\)는 compact이다.

\(K_c^c\)에서는 \(E(x)>c\)이므로

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

이다. \(\beta\ge\beta_0\)에서 오른쪽을 균일하게 작게 만들기 위해 \(c\)를 충분히 크게 잡으면

$$
\mu_\beta(K_c)\ge1-\epsilon,
\qquad \beta\ge\beta_0
$$

가 된다. 따라서 \(\{\mu_\beta:\beta\ge\beta_0\}\)는 tight하다. \(\square\)

## 6. Non-compact 유일 manifest 정리

**정리 6.1**  
\(\mu_0\in\mathcal P(\mathbb R^n)\), \(E:\mathbb R^n\to\mathbb R_{\ge0}\)가 continuous/coercive라고 하자. 또한

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

먼저 \(x_*\)를 포함하는 임의의 열린집합 \(U\)를 잡는다. \(S\setminus U=\varnothing\)이면 \(\mu_0(\mathbb R^n\setminus U)=0\)이므로 모든 \(\beta\)에 대해 \(\mu_\beta(U)=1\)이다.

이제 \(S\setminus U\ne\varnothing\)라 하자. 보조정리 3.1에 의해 어떤 \(\delta_U>0\)가 존재해서

$$
E(x)\ge m+\delta_U,
\qquad x\in S\setminus U
$$

이다.

\(\eta=\delta_U/2\)로 둔다. 그러면 분자는

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

이제 약수렴을 보인다. \(f:\mathbb R^n\to\mathbb R\)를 bounded continuous 함수라 하자. \(\epsilon>0\)을 잡는다. 연속성으로 어떤 열린 근방 \(U\ni x_*\)가 존재해서

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

이고, 오른쪽은 \(\limsup_{\beta\to\infty}\)에서 \(\epsilon\) 이하가 된다. \(\epsilon\)은 임의이므로

$$
\int f\,d\mu_\beta\to f(x_*)
$$

이다. 따라서 \(\mu_\beta\Rightarrow\delta_{x_*}\)이다. \(\square\)

## 7. 여러 minimizer 버전

최소집합이 하나가 아닐 때는 Dirac 수렴이 아니라 최소집합 근방으로의 농축이 닫힌 statement다.

$$
A_*=\operatorname*{argmin}_{x\in S}E(x)
$$

**정리 7.1**  
위 세팅에서 \(A_*\)가 여러 점을 가질 수 있다고 하자. \(U\)가 \(A_*\)를 포함하는 열린집합이면

$$
\mu_\beta(U)\to1
$$

이다.

**증명.**

coercivity와 연속성 때문에 \(A_*\)는 공집합이 아닌 compact 집합이다. \(F=S\setminus U\)에서 \(\inf_FE=m\)이라고 가정하면, 보조정리 3.1의 compactness 논리와 같은 방식으로 \(F\) 안에 minimizer가 존재한다. 이는 \(U\supset A_*\)와 모순이다. 따라서 \(F\) 위에는 양의 gap이 있고, 같은 분자/분모 평가로 \(\mu_\beta(\mathbb R^n\setminus U)\to0\)이다. \(\square\)

## 8. 예제

실수 전체를 후보공간으로 둔다.

$$
A=\mathbb R,\qquad
E(x)=(x-1)^2
$$

임의의 Borel 확률측도 \(\mu_0\)가 \(1\)을 support에 포함하고, \(1\)이 support 위의 유일한 minimizer라고 하자. 예를 들어 \(\mu_0\)가 모든 열린구간에 양의 질량을 주는 정규분포라면 조건이 성립한다.

\(E\)는 continuous/coercive이고

$$
\operatorname*{argmin}_{x\in\operatorname{supp}\mu_0}E(x)=\{1\}
$$

이다. 따라서

$$
\mu_\beta\Rightarrow\delta_1
$$

이다.

## 9. Gamma-convergence의 위치

이 문서에서 닫은 것은 고정된 하나의 \(E\)에 대한 non-compact 농축이다. 조건 에너지가

$$
E_\beta,\quad E_n,\quad E_\ell
$$

처럼 scale에 따라 변하면 다른 문제가 된다. 그때 필요한 도구가 Gamma-convergence다.

확인 항목:

| 확인 항목 | 의미 |
|---|---|
| liminf 부등식 | \(x_n\to x\)이면 \(E_\infty(x)\le\liminf E_n(x_n)\) |
| recovery sequence | 각 \(x\)에 대해 \(x_n\to x\), \(E_n(x_n)\to E_\infty(x)\)인 열 존재 |
| equicoercivity | \(E_n\)의 sublevel set들이 함께 compact하게 갇힘 |

scale에 따라 변하는 Gibbs 농축은 [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)에서 분리해 다룬다. 핵심은 순수 Gamma 수렴만이 아니라, Gibbs 분모를 지탱하는 \(\mu_0\)-양질량 recovery 조건까지 함께 확인해야 한다는 점이다.

02a의 첫 닫힘은 정리 6.1, 즉 고정된 \(E\)에 대한 non-compact 농축이다.

## 10. CE bridge 판정표

CE bridge에서 \(A=\mathcal P_I\) 같은 경로공간을 쓰려면, 정리 6.1을 그대로 쓸 수 없다. 그 대신 아래 항목을 채워야 한다.

| 항목 | 질문 | 판정 |
|---|---|---|
| 공간 | 경로공간이 \(\mathbb R^n\) 또는 충분히 좋은 Polish space인가 | `필수` |
| support | 초기 측도 support가 정의되어 있는가 | `필수` |
| 에너지 | \(E_{\mathrm{fold}}\)가 continuous 또는 l.s.c.인가 | `필수` |
| escape | coercive/equicoercive 조건이 있는가 | `필수` |
| minimizer | 선택 경로가 유일한가, 아니면 최소집합인가 | `필수` |

이 표가 비면 05장은 여전히 `Bridge`다. 표가 채워지면 그때 non-compact 농축 정리를 CE 경로공간 위로 옮길 수 있다.

## 11. 결론

02장의 non-compact 핵심은 이제 닫힌다.

$$
E\ \mathrm{coercive}
\quad+\quad
\operatorname*{argmin}_{\operatorname{supp}\mu_0}E=\{x_*\}
\quad\Longrightarrow\quad
\mu_\beta\Rightarrow\delta_{x_*}
$$

이것은 새 물리 bridge가 아니라 표준 해석학 위의 `Exact under assumptions` 정리다.
