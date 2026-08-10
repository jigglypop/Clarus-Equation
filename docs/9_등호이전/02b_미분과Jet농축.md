# 02b. 미분과 Jet 농축

## 0. 목표

이 문서는 미분을 등호 이전 문법으로 다시 쓴다.

핵심 관점:

> 미분은 여러 scale의 secant/finite-difference 후보가 한 tangent/jet 값으로 manifest 되는 극한이다.

단, 첫 정직한 correction이 있다.

> \(h\to0\)만으로 Gibbs 분포가 Dirac로 collapse하지는 않는다. \(h\)는 에너지 중심을 jet로 보내고, 실제 농축은 \(\beta_h\to\infty\) 같은 readout 강도가 함께 필요하다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| finite-difference jet 수렴 | `[정리]` | Taylor 정리 |
| fixed-\(\beta\) posterior 한계 | `[정리]` | 중심만 jet로 이동 |
| \(\beta_h\to\infty\) jet 농축 | `[정리]` | 02a형 Gibbs 농축 |
| 비선택 잔류 = 고차 jet 정보 | residual은 `[정리]`; readout은 `[공리: 모델 선택]` | deterministic residual과 잔류 readout을 분리 |

## 1. Finite difference jet

\(f:\mathbb R\to\mathbb R\)라 하자. \(j\ge1\)에 대해 forward difference를

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

\(k\)-jet 후보공간은

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

진짜 \(k\)-jet은

$$
J_x^kf
=
\big(f(x),f'(x),\dots,f^{(k)}(x)\big)
$$

이다.

## 2. Jet 수렴 정리

**정리 2.1**  
\(f\in C^k\)이고 \(x\) 근방에서 정의되어 있으면

$$
a_h^{(k)}(x)\to J_x^kf
\qquad(h\to0)
$$

이다.

**증명.**

\(j=0\)은 정의상 자명하다. \(j\ge1\)에 대해 Taylor 전개를 쓰면

$$
f(x+ih)
=
\sum_{r=0}^j\frac{(ih)^r}{r!}f^{(r)}(x)
+o(h^j)
$$

이다. 이를 \(\Delta_h^j\)에 대입하면

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

이다. 모든 \(j=0,\dots,k\)에 대해 성립하므로 벡터 수렴이 따른다. \(\square\)

## 3. PreEq jet 에너지

후보 \(s=(s_0,\dots,s_k)\in A_k\)에 대해 scale \(h\)의 jet 에너지를

$$
E_h(s)
=
\sum_{j=0}^kw_j\left(s_j-D_h^jf(x)\right)^2
$$

로 둔다. 여기서 \(w_j>0\)이다.

초기 모호함 상태는 Borel 확률측도

$$
\mu_0\in\mathcal P(A_k)
$$

이고, \(J_x^kf\in\operatorname{supp}\mu_0\)라고 가정한다.

readout 강도 \(\beta_h>0\)에서

$$
\mu_h(ds)
=
\frac{e^{-\beta_hE_h(s)}}{Z_h}\mu_0(ds),
\qquad
Z_h=\int_{A_k}e^{-\beta_hE_h(s)}\mu_0(ds)
$$

로 둔다.

## 4. Fixed beta는 collapse가 아니다

**명제 4.1**  
\(J=J_x^kf\)라 하자. \(\beta_h=\beta\)가 고정되어 있고 \(a_h^{(k)}(x)\to J\)이면, \(E_h(s)\)는 점별로

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

따라서 fixed \(\beta\)에서는 한 점 Dirac가 아니라 finite-temperature posterior가 남는다.

**증명.**

bounded continuous \(g:A_k\to\mathbb R\)를 잡는다. 그러면

$$
\int g(s)\,\mu_h(ds)
=
\frac{\int g(s)e^{-\beta E_h(s)}\mu_0(ds)}
{\int e^{-\beta E_h(s)}\mu_0(ds)}
$$

이다. \(a_h^{(k)}\to J\)이므로 \(E_h(s)\to E_0(s)\)가 모든 \(s\)에서 성립한다. 또한

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

로 수렴한다. 분모는 양수다. 따라서 \(\mu_h\Rightarrow\mu_0^{\beta,J}\)이다. \(\square\)

**해석.**

\(h\to0\)은 secant/finite-difference 중심을 tangent/jet 중심으로 보낼 뿐이다. 분포의 폭을 0으로 만들려면 \(\beta_h\to\infty\)가 필요하다.

## 5. Jet 농축 정리

**정리 5.1**  
\(f\in C^k\), \(J=J_x^kf\in\operatorname{supp}\mu_0\), \(\beta_h\to\infty\)라고 하자. 그러면

$$
\mu_h\Rightarrow\delta_J
\qquad(h\to0)
$$

이다.

**증명.**

\(J\)를 포함하는 열린집합 \(U\subset A_k\)를 잡는다. 어떤 \(r>0\)에 대해

$$
B(J,r)\subset U
$$

이다.

가중치 행렬을 \(W=\operatorname{diag}(w_0,\dots,w_k)\)라 두고

$$
\lambda_-=\min_jw_j,\qquad
\lambda_+=\max_jw_j
$$

라 하자.

정리 2.1에 의해 \(h\)가 충분히 작으면

$$
\|a_h^{(k)}(x)-J\|<r/4
$$

이다. 밖에서는 \(s\notin U\)이므로 \(s\notin B(J,r)\)이고 \(\|s-J\|\ge r\)이다. 따라서

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

이제 \(0<\rho<r/4\)를 충분히 작게 잡아

$$
\lambda_+(2\rho)^2
<
\lambda_-\left(\frac{3r}{4}\right)^2
$$

가 되게 한다. 안쪽에는

$$
V=B(J,\rho)
$$

를 잡는다. \(J\in\operatorname{supp}\mu_0\)이므로 \(\mu_0(V)>0\)이다. \(h\)가 더 작아서 \(\|a_h^{(k)}-J\|<\rho\)이면, \(s\in V\)에 대해

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

이다. 따라서 \(\mu_h(U)\to1\)이다.

마지막으로 bounded continuous \(g:A_k\to\mathbb R\)에 대해 02a 정리 6.1과 같은 근방 분해를 쓰면

$$
\int g\,d\mu_h\to g(J)
$$

이다. 따라서 \(\mu_h\Rightarrow\delta_J\)이다. \(\square\)

## 6. Taylor 잔류와 고차 도함수

jet 중심의 잔류를

$$
R_h^{(k)}(x)=a_h^{(k)}(x)-J_x^kf
$$

라 두자.

**정리 6.1**  
\(f\in C^{k+1}\)이면 \(j=1,\dots,k\)에 대해

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

이다. 첫 좌표는 \(D_h^0f(x)=f(x)\)라서 잔류가 0이다.

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

이다. 여기서 \(S(\cdot,\cdot)\)는 Stirling number of the second kind이고

$$
S(j+1,j)=\binom{j+1}{2}
$$

이다. 따라서 계수는

$$
\frac{j!\binom{j+1}{2}}{(j+1)!}
=
\frac j2
$$

이다. \(h^j\)로 나누면 원하는 식이 나온다. \(\square\)

## 7. 1차 미분의 경우

\(k=1\)이면 후보공간은

$$
A_1=\mathbb R^2
$$

이고

$$
a_h^{(1)}(x)
=
\left(f(x),\frac{f(x+h)-f(x)}h\right)
$$

이다. \(f\in C^1\)이면

$$
a_h^{(1)}(x)\to(f(x),f'(x)).
$$

\(\beta_h\to\infty\)이면

$$
\mu_h\Rightarrow\delta_{(f(x),f'(x))}.
$$

\(f\in C^2\)이면 slope 잔류는

$$
\frac{f(x+h)-f(x)}h-f'(x)
=
\frac h2f''(x)+o(h)
$$

이다. 즉 tangent slope가 manifest 되고, finite-scale secant slope의 잔류 속도는 \(f''(x)\)를 읽는다.

## 8. 비선택 잔류의 정확한 지위

이 문서에서 `[정리]`로 닫힌 것은 다음이다.

| 대상 | 판정 |
|---|---|
| finite difference jet가 \(J_x^kf\)로 수렴 | `[정리]` |
| \(\beta_h\to\infty\)에서 \(\mu_h\Rightarrow\delta_{J_x^kf}\) | `[정리]` |
| finite-scale 중심 잔류 \(R_h^{(k)}\)가 고차 도함수를 담음 | `[정리]` |

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

이 값이 Taylor 잔류 \(R_h^{(k)}\)와 정확히 같으려면 prior의 대칭성, 충분히 큰 \(\beta_h\), 또는 별도 보정 조건이 필요하다.

## 9. Gamma 도구와의 관계

이 문서의 jet 에너지는 scale \(h\)에 따라 변한다.

$$
E_h(s)=\sum_{j=0}^kw_j(s_j-D_h^jf(x))^2
$$

따라서 일반적인 scale-dependent Gibbs 농축의 특수한 경우로도 볼 수 있다. 그 일반 도구는 [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)에 둔다.

다만 jet 농축은 움직이는 중심이 명시적으로 주어지므로 더 직접적으로 닫힌다. 즉

$$
D_h(x)\to J_x^kf,\qquad \beta_h\to\infty
$$

이면 \(\beta_h\|D_h-J\|\to0\) 같은 joint scaling 없이도

$$
\mu_h\Rightarrow\delta_{J_x^kf}
$$

가 성립한다.

## 10. 결론

미분은 PreEq 관점에서 다음처럼 읽힌다.

$$
\text{finite-scale slope/jet candidates}
\xrightarrow[h\to0]{\text{center convergence}}
J_x^kf
\xrightarrow[\beta_h\to\infty]{\text{readout}}
\delta_{J_x^kf}.
$$

즉 미분의 1차 값은 manifest readout이고, finite-scale 잔류의 수렴 속도는 고차 도함수, 즉 jet hierarchy를 담는다.
