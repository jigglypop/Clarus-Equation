# 02b. 미분과 Jet 농축

이 문서는 finite difference 후보가 미분값과 고차 jet로 모이는 과정을 Gibbs 농축의 언어로 정리한다. 핵심은 $h\to0$가 후보 에너지의 중심을 옮기는 극한이고, 한 점의 manifest readout에는 별도로 발산하는 readout 강도 $\beta_h$가 필요하다는 구분이다.

독자는 일변수 미분, Taylor 전개, 확률측도의 약수렴을 알고 있으면 된다. 먼저 jet의 정의와 유한차분 수렴을 고정한 뒤 fixed-$\beta$의 한계를 보이고, 이어 농축 정리·잔류·Gamma 도구 및 CE 해석의 적용 경계를 순서대로 읽는다.

## 0. 목표

02a의 고정 에너지 농축을 미분 문제에 연결하려면, scale $h$가 바뀔 때 어떤 것이 수렴하고 어떤 것이 농축을 일으키는지를 분리해야 한다. 이 절은 그 목표와 형식 지위를 먼저 고정하며, 아래의 비유가 표준 미분 정리를 대체하지 않음을 밝힌다.

핵심 관점:

> 미분은 여러 scale의 secant/finite-difference 후보가 한 tangent/jet 값으로 manifest 되는 극한이다.

단, 첫 정직한 correction이 있다.

> $h\to0$만으로 Gibbs 분포가 Dirac로 collapse하지는 않는다. $h$는 에너지 중심을 jet로 보내고, 실제 농축은 $\beta_h\to\infty$ 같은 readout 강도가 함께 필요하다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| finite-difference jet 수렴 | `[정리]` | Taylor 정리 |
| fixed-$\beta$ posterior 한계 | `[정리]` | 중심만 jet로 이동 |
| $\beta_h\to\infty$ jet 농축 | `[정리]` | 02a형 Gibbs 농축 |
| 비선택 잔류 = 고차 jet 정보 | residual은 `[정리]`; readout은 `[공리: 모델 선택]` | deterministic residual과 잔류 readout을 분리 |

## 1. Finite difference jet

이제 유한한 차수 $k$에서 비교할 후보공간과 중심을 정의한다. 이 정의는 $f$가 필요한 점들에서 값이 존재한다는 것만 사용하며, 실제 jet 수렴에는 다음 절의 $C^k$ regularity가 추가로 필요하다.

$f:\mathbb R\to\mathbb R$라 하자. $j\ge1$에 대해 forward difference를

$$
\Delta_h^j f(x)
=
\sum_{i=0}^j(-1)^{j-i}\binom ji f(x+ih)
$$

로 둔다.

정규화된 finite difference를

$$
D_h^0f(x)=f(x),
\qquad
D_h^jf(x)=\frac{\Delta_h^j f(x)}{h^j}\quad(j\ge1)
$$

로 둔다.

$k$-jet 후보공간은

$$
A_k=\mathbb R^{k+1}
$$

이고, finite-difference jet 중심은

$$
a_h^{(k)}(x)
=
\big(D_h^0f(x),D_h^1f(x),\dots,D_h^kf(x)\big)
\in A_k
$$

이다.

진짜 $k$-jet은

$$
J_x^kf
=
\big(f(x),f'(x),\dots,f^{(k)}(x)\big)
$$

이다.

## 2. Jet 수렴 정리

앞 절의 후보 중심이 진짜 jet으로 간다는 말은 미분 가능성의 정량적 내용이다. 다음 정리는 $x$ 근방의 $C^k$ 가정 아래 Taylor 나머지가 소거 항등식 뒤에 충분히 작다는 사실을 쓰며, 비매끄러운 함수나 경계에서 한쪽 차분으로 얻는 결론은 포함하지 않는다.

**정리 2.1**  
$f\in C^k$이고 $x$ 근방에서 정의되어 있으면

$$
a_h^{(k)}(x)\to J_x^kf
\qquad(h\to0)
$$

이다.

**증명.**

$j=0$은 정의상 자명하다. $j\ge1$에 대해 Taylor 전개를 쓰면

$$
f(x+ih)
=
\sum_{r=0}^j\frac{(ih)^r}{r!}f^{(r)}(x)
+o(h^j)
$$

이다. 이를 $\Delta_h^j$에 대입하면

$$
\Delta_h^j f(x)
=
\sum_{r=0}^j
\frac{h^r f^{(r)}(x)}{r!}
\sum_{i=0}^j(-1)^{j-i}\binom ji i^r
+o(h^j).
$$

finite difference 항등식에 의해

$$
\sum_{i=0}^j(-1)^{j-i}\binom ji i^r=0\quad(r<j),
\qquad
\sum_{i=0}^j(-1)^{j-i}\binom ji i^j=j!
$$

이다. 따라서

$$
\Delta_h^j f(x)=h^j f^{(j)}(x)+o(h^j)
$$

이고

$$
D_h^jf(x)\to f^{(j)}(x)
$$

이다. 모든 $j=0,\dots,k$에 대해 성립하므로 벡터 수렴이 따른다. $\square$

## 3. PreEq jet 에너지

수렴하는 중심만으로는 확률적 readout이 정해지지 않으므로, 중심에서의 이차 거리를 에너지로 선택한다. 양의 가중치와 유클리드 topology는 이 문서의 채택된 모델 세팅이며, 서로 다른 단위의 jet 좌표에서는 가중치가 정규화 역할을 하므로 그 선택을 물리적 산출로 해석할 수 없다.

후보 $s=(s_0,\dots,s_k)\in A_k$에 대해 scale $h$의 jet 에너지를

$$
E_h(s)
=
\sum_{j=0}^kw_j\left(s_j-D_h^jf(x)\right)^2
$$

로 둔다. 여기서 $w_j>0$이다.

초기 모호함 상태는 Borel 확률측도

$$
\mu_0\in\mathcal P(A_k)
$$

이고, $J_x^kf\in\operatorname{supp}\mu_0$라고 가정한다.

readout 강도 $\beta_h>0$에서

$$
\mu_h(ds)
=
\frac{e^{-\beta_hE_h(s)}}{Z_h}\mu_0(ds),
\qquad
Z_h=\int_{A_k}e^{-\beta_hE_h(s)}\mu_0(ds)
$$

로 둔다.

## 4. Fixed beta는 collapse가 아니다

에너지 중심이 $J$로 움직여도 온도에 해당하는 $\beta$가 고정되어 있으면 측도의 폭은 사라지지 않는다. 다음 명제는 지배수렴정리로 닫히는 형식 결과이며, prior의 support·질량 분포가 남기는 posterior를 Dirac 농축으로 부르는 오해를 막는다.

**명제 4.1**  
$J=J_x^kf$라 하자. $\beta_h=\beta$가 고정되어 있고 $a_h^{(k)}(x)\to J$이면, $E_h(s)$는 점별로

$$
E_0(s)=\sum_{j=0}^kw_j(s_j-J_j)^2
$$

에 수렴한다. 그러면

$$
\mu_h\Rightarrow
\mu_0^{\beta,J}
$$

이고,

$$
\mu_0^{\beta,J}(ds)
=
\frac{e^{-\beta E_0(s)}}{\int e^{-\beta E_0(u)}\mu_0(du)}
\mu_0(ds)
$$

이다.

따라서 fixed $\beta$에서는 한 점 Dirac가 아니라 finite-temperature posterior가 남는다.

**증명.**

bounded continuous $g:A_k\to\mathbb R$를 잡는다. 그러면

$$
\int g(s)\,\mu_h(ds)
=
\frac{\int g(s)e^{-\beta E_h(s)}\mu_0(ds)}
{\int e^{-\beta E_h(s)}\mu_0(ds)}
$$

이다. $a_h^{(k)}\to J$이므로 $E_h(s)\to E_0(s)$가 모든 $s$에서 성립한다. 또한

$$
|g(s)e^{-\beta E_h(s)}|\le\|g\|_\infty,
\qquad
0\le e^{-\beta E_h(s)}\le1
$$

이다. 지배수렴정리에 의해 분자와 분모가 각각

$$
\int g(s)e^{-\beta E_0(s)}\mu_0(ds),
\qquad
\int e^{-\beta E_0(s)}\mu_0(ds)
$$

로 수렴한다. 분모는 양수다. 따라서 $\mu_h\Rightarrow\mu_0^{\beta,J}$이다. $\square$

**해석.**

$h\to0$은 secant/finite-difference 중심을 tangent/jet 중심으로 보낼 뿐이다. 분포의 폭을 0으로 만들려면 $\beta_h\to\infty$가 필요하다.

## 5. Jet 농축 정리

Dirac readout을 얻으려면 중심 수렴에 더해 $\beta_h\to\infty$가 필요하다. 다음 정리는 유한차원 후보공간, 양의 유한 가중치, 그리고 진짜 jet 주변의 양의 prior 질량을 전제로 한 정리이며, 유한 표본의 추정 오차나 수치 최적화의 성공률을 보장하지 않는다.

**정리 5.1**  
$f\in C^k$, $J=J_x^kf\in\operatorname{supp}\mu_0$, $\beta_h\to\infty$라고 하자. 그러면

$$
\mu_h\Rightarrow\delta_J
\qquad(h\to0)
$$

이다.

**증명.**

$J$를 포함하는 열린집합 $U\subset A_k$를 잡는다. 어떤 $r>0$에 대해

$$
B(J,r)\subset U
$$

이다.

가중치 행렬을 $W=\operatorname{diag}(w_0,\dots,w_k)$라 두고

$$
\lambda_-=\min_jw_j,\qquad
\lambda_+=\max_jw_j
$$

라 하자.

정리 2.1에 의해 $h$가 충분히 작으면

$$
\|a_h^{(k)}(x)-J\|<r/4
$$

이다. 밖에서는 $s\notin U$이므로 $s\notin B(J,r)$이고 $\|s-J\|\ge r$이다. 따라서

$$
\|s-a_h^{(k)}\|
\ge
\|s-J\|-\|a_h^{(k)}-J\|
\ge
3r/4
$$

이고

$$
E_h(s)\ge \lambda_-\left(\frac{3r}{4}\right)^2
\qquad(s\notin U)
$$

이다.

이제 $0<\rho<r/4$를 충분히 작게 잡아

$$
\lambda_+(2\rho)^2
<
\lambda_-\left(\frac{3r}{4}\right)^2
$$

가 되게 한다. 안쪽에는

$$
V=B(J,\rho)
$$

를 잡는다. $J\in\operatorname{supp}\mu_0$이므로 $\mu_0(V)>0$이다. $h$가 더 작아서 $\|a_h^{(k)}-J\|<\rho$이면, $s\in V$에 대해

$$
\|s-a_h^{(k)}\|
\le
\|s-J\|+\|J-a_h^{(k)}\|
<
2\rho
$$

이고

$$
E_h(s)\le \lambda_+(2\rho)^2
\qquad(s\in V)
$$

이다.

따라서

$$
g_U
=
\lambda_-\left(\frac{3r}{4}\right)^2
-
\lambda_+(2\rho)^2
$$

는 양수다.

그러면

$$
\mu_h(A_k\setminus U)
\le
\frac{
e^{-\beta_h\lambda_-(3r/4)^2}\mu_0(A_k\setminus U)
}{
e^{-\beta_h\lambda_+(2\rho)^2}\mu_0(V)
}
\le
\frac1{\mu_0(V)}e^{-\beta_hg_U}
\to0
$$

이다. 따라서 $\mu_h(U)\to1$이다.

마지막으로 bounded continuous $g:A_k\to\mathbb R$에 대해 02a 정리 6.1과 같은 근방 분해를 쓰면

$$
\int g\,d\mu_h\to g(J)
$$

이다. 따라서 $\mu_h\Rightarrow\delta_J$이다. $\square$

## 6. Taylor 잔류와 고차 도함수

농축 뒤에도 유한한 $h$에서는 중심이 진짜 jet과 일치하지 않으며, 그 차이가 고차 정보의 정량적 흔적을 남긴다. 여기서는 $C^{k+1}$ regularity와 forward difference라는 선택 아래의 1차 점근식만 증명하며, 잡음이 있는 데이터에서 고차 도함수를 안정적으로 추정한다는 주장은 하지 않는다.

jet 중심의 잔류를

$$
R_h^{(k)}(x)=a_h^{(k)}(x)-J_x^kf
$$

라 두자.

**정리 6.1**  
$f\in C^{k+1}$이면 $j=1,\dots,k$에 대해

$$
D_h^jf(x)
=
f^{(j)}(x)
+
\frac j2h f^{(j+1)}(x)
+o(h)
$$

이다. 따라서

$$
\frac1hR_h^{(k)}(x)
\to
\left(0,\frac12f''(x),1\cdot f'''(x),\dots,\frac k2f^{(k+1)}(x)\right)
$$

이다. 첫 좌표는 $D_h^0f(x)=f(x)$라서 잔류가 0이다.

**증명.**

Taylor 전개를 한 차수 더 쓰면

$$
\Delta_h^j f(x)
=
h^jf^{(j)}(x)
+
h^{j+1}
\frac{j!S(j+1,j)}{(j+1)!}
f^{(j+1)}(x)
+o(h^{j+1})
$$

이다. 여기서 $S(\cdot,\cdot)$는 Stirling number of the second kind이고

$$
S(j+1,j)=\binom{j+1}{2}
$$

이다. 따라서 계수는

$$
\frac{j!\binom{j+1}{2}}{(j+1)!}
=
\frac j2
$$

이다. $h^j$로 나누면 원하는 식이 나온다. $\square$

## 7. 1차 미분의 경우

차수 하나의 경우는 앞의 일반 정리가 친숙한 secant와 tangent의 구별로 환원됨을 보여 준다. 이 예는 smooth 함수의 국소 점근식이며, 불연속·뾰족점·관측 노이즈가 있는 신호에서 같은 잔류식을 그대로 적용하는 근거는 아니다.

$k=1$이면 후보공간은

$$
A_1=\mathbb R^2
$$

이고

$$
a_h^{(1)}(x)
=
\left(f(x),\frac{f(x+h)-f(x)}h\right)
$$

이다. $f\in C^1$이면

$$
a_h^{(1)}(x)\to(f(x),f'(x)).
$$

$\beta_h\to\infty$이면

$$
\mu_h\Rightarrow\delta_{(f(x),f'(x))}.
$$

$f\in C^2$이면 slope 잔류는

$$
\frac{f(x+h)-f(x)}h-f'(x)
=
\frac h2f''(x)+o(h)
$$

이다. 즉 tangent slope가 manifest 되고, finite-scale secant slope의 잔류 속도는 $f''(x)$를 읽는다.

## 8. 비선택 잔류의 정확한 지위

지금까지의 증명은 중심과 측도의 극한 및 결정론적 Taylor 잔류까지다. 비선택 잔류를 CE의 관측량으로 읽는 단계는 수학 정리에서 따라오지 않으므로, 다음 표와 식은 증명된 부분과 모델 선택·미완성 부분을 분리해 둔다.

이 문서에서 `[정리]`로 닫힌 것은 다음이다.

| 대상 | 판정 |
|---|---|
| finite difference jet가 $J_x^kf$로 수렴 | `[정리]` |
| $\beta_h\to\infty$에서 $\mu_h\Rightarrow\delta_{J_x^kf}$ | `[정리]` |
| finite-scale 중심 잔류 $R_h^{(k)}$가 고차 도함수를 담음 | `[정리]` |

하지만

$$
\mu_{\mathrm{ns},h}
=
\mu_h|_{A_k\setminus\{J_x^kf\}}
$$

를 어떤 moment/readout으로 읽을지는 아직 `[미완성]`이다. 가장 단순한 readout은 중심 잔류 또는 평균 잔류지만, 채택하려면 `[공리: 모델 선택]`으로 고정해야 한다.

$$
\mathcal R_h
=
\int_{A_k}(s-J_x^kf)\,\mu_h(ds)
$$

이 값이 Taylor 잔류 $R_h^{(k)}$와 정확히 같으려면 prior의 대칭성, 충분히 큰 $\beta_h$, 또는 별도 보정 조건이 필요하다.

## 9. Gamma 도구와의 관계

jet 에너지는 $h$마다 달라지지만 중심이 명시적으로 수렴하므로 이 문서에서는 직접적인 근방 추정으로 충분하다. 일반 scale-dependent 에너지에서는 topology, liminf, recovery, equicoercivity 및 분모 질량이 별도로 필요하며, 그 일반화는 연결 문서의 조건부 결과이지 이 절의 자동 따름정리가 아니다.

이 문서의 jet 에너지는 scale $h$에 따라 변한다.

$$
E_h(s)=\sum_{j=0}^kw_j(s_j-D_h^jf(x))^2
$$

따라서 일반적인 scale-dependent Gibbs 농축의 특수한 경우로도 볼 수 있다. 그 일반 도구는 [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)에 둔다.

다만 jet 농축은 움직이는 중심이 명시적으로 주어지므로 더 직접적으로 닫힌다. 즉

$$
D_h(x)\to J_x^kf,\qquad \beta_h\to\infty
$$

이면 $\beta_h\|D_h-J\|\to0$ 같은 joint scaling 없이도

$$
\mu_h\Rightarrow\delta_{J_x^kf}
$$

가 성립한다.

## 10. 결론

따라서 이 문서가 정리로 닫는 것은 매끄러운 함수의 finite-difference 중심과 발산 readout 아래의 Gibbs 농축이다. 이를 미분의 PreEq 해석 또는 CE의 비선택 readout으로 확장하는 문장은 채택 공리와 미완성 검증 다리를 필요로 하며, 아래 도식은 그 범위를 압축해 표시한다.

미분은 PreEq 관점에서 다음처럼 읽힌다.

$$
\text{finite-scale slope/jet candidates}
\xrightarrow[h\to0]{\text{center convergence}}
J_x^kf
\xrightarrow[\beta_h\to\infty]{\text{readout}}
\delta_{J_x^kf}.
$$

즉 미분의 1차 값은 manifest readout이고, finite-scale 잔류의 수렴 속도는 고차 도함수, 즉 jet hierarchy를 담는다.
