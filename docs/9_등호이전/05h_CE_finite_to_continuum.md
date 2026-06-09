# 05h. CE Finite-to-Continuum Package

## 0. 목표

[05g_CE_prior_support_package.md](05g_CE_prior_support_package.md)는 CE 농축에 필요한 prior/support 조건을

$$
\mu_{\mathrm{base}}\{\gamma:W[\gamma]<W_{\min}+\eta\}>0
\qquad(\eta>0)
$$

로 고정했다.

남은 질문은 finite mesh package가 continuum package와 같은 manifest limit을 주는지다.

finite mesh:

$$
(\mathcal P_{I,N},W_N,\mu_N)
$$

continuum CE:

$$
(\mathcal P_I,W,\mu_{\mathrm{base}})
$$

핵심 결론:

> fixed \(N\)에서는 finite positive prior만으로 충분하다. 그러나 \(N\to\infty\)와 \(\beta_N\to\infty\)를 함께 보내면 positive prior만으로는 부족하다. near-minimizer recovery mass가 \(\beta_N\) scale에서 너무 빨리 사라지지 않아야 한다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| fixed mesh 농축 | `Exact` | finite positive prior + \(\beta\to\infty\) |
| positive prior의 joint-limit 충분성 | `False in general` | 반례 2.1 |
| recovery mass scale 조건 | `Exact under assumptions` | 정리 3.1 |
| fixed-\(\beta\) Gibbs convergence | `Exact under assumptions` | compact/continuous/weak convergence package |
| full physical continuum prior | `Bridge/Open` | Brownian/Sobolev/Gaussian support 선택 필요 |

## 1. 세팅

\(\Gamma=\mathcal P_I\)를 continuum pathspace라고 하자. finite mesh 후보공간은 finite set

$$
A_N=\mathcal P_{I,N}
$$

이고, mesh path를 continuum path로 보는 embedding 또는 reconstruction map을

$$
\iota_N:A_N\to\Gamma
$$

라 둔다.

finite prior와 energy:

$$
\mu_N\in\mathcal P(A_N),
\qquad
W_N:A_N\to[0,\infty].
$$

항상 \(\mu_N(a)>0\) for all \(a\in A_N\)라고 하자.

finite Gibbs measure:

$$
\nu_{N,\beta}(a)
=
\frac{e^{-\beta W_N(a)}\mu_N(a)}
{Z_{N,\beta}},
\qquad
Z_{N,\beta}
=
\sum_{a\in A_N}e^{-\beta W_N(a)}\mu_N(a).
$$

continuum에서 읽으려면 pushforward를 쓴다.

$$
\bar\nu_{N,\beta}
:=
(\iota_N)_*\nu_{N,\beta}.
$$

continuum energy의 support 위 최소값과 최소집합은

$$
m=\inf_{\gamma\in\operatorname{supp}\mu_{\mathrm{base}}}W[\gamma],
\qquad
M=\operatorname*{argmin}_{\gamma\in\operatorname{supp}\mu_{\mathrm{base}}}W[\gamma]
$$

이다.

## 2. fixed mesh와 joint mesh는 다르다

### 정리 2.1: fixed \(N\) finite concentration

\(A_N\)이 finite set이고 \(\mu_N(a)>0\) for all \(a\in A_N\)라고 하자. \(W_N\)의 minimizer set을

$$
M_N=\operatorname*{argmin}_{a\in A_N}W_N(a)
$$

라 두면 모든 \(a\notin M_N\)에 대해

$$
\nu_{N,\beta}(a)\to0
\qquad(\beta\to\infty)
$$

이다.

특히 \(M_N=\{a_N^*\}\)이면

$$
\nu_{N,\beta}\Rightarrow\delta_{a_N^*}.
$$

증명:

\(a\notin M_N\)이면

$$
\Delta_a=W_N(a)-\min W_N>0.
$$

어떤 \(a_*\in M_N\)에 대해

$$
\frac{\nu_{N,\beta}(a)}{\nu_{N,\beta}(a_*)}
=
\frac{\mu_N(a)}{\mu_N(a_*)}e^{-\beta\Delta_a}
\to0.
$$

finite set이므로 모든 비최소점의 질량이 0으로 간다. 끝.

### 반례 2.1: positive weight만으로는 joint limit이 깨진다

각 \(N\)에서

$$
A_N=\{a,b\},
\qquad
W_N(a)=0,
\qquad
W_N(b)=1
$$

라고 하자. 모든 \(N\)에서 \(a\)가 유일 minimizer다.

하지만 inverse temperature를

$$
\beta_N=N
$$

으로 두고 prior를

$$
\mu_N(a)=\frac{e^{-N^2}}{1+e^{-N^2}},
\qquad
\mu_N(b)=\frac1{1+e^{-N^2}}
$$

로 잡으면 각 \(N\)에서 prior weight는 양수지만

$$
\nu_{N,\beta_N}(a)
=
\frac{e^{-N^2}}
{e^{-N^2}+e^{-N}}
\to0.
$$

즉 finite mesh의 유일 minimizer가 joint limit에서는 사라진다.

해석:

> energy gap \(1\)은 \(\beta_N=N\)에 의해 \(e^{-N}\)으로 보상된다. 그런데 recovery weight가 \(e^{-N^2}\)로 더 빨리 죽으면 Gibbs 분모가 minimizer를 붙잡지 못한다.

따라서 finite-to-continuum bridge에는 다음 scale 조건이 필요하다.

$$
\frac1{\beta_N}\log\frac1{\text{recovery mass}_N}\to0.
$$

## 3. joint finite-to-continuum concentration

### 가정 J

다음 네 조건을 둔다.

1. \(\beta_N\to\infty\).
2. \(M\subset\Gamma\)는 nonempty compact set이다.
3. outer gap consistency: 모든 open \(U\supset M\)에 대해 어떤 \(\delta_U>0\)와 \(N_U\)가 존재해서 \(N\ge N_U\)이면

$$
\iota_N(a)\notin U
\quad\Longrightarrow\quad
W_N(a)\ge m+\delta_U.
$$

4. scaled recovery mass: 모든 \(\eta>0\)에 대해 \(B_{N,\eta}\subset A_N\)가 존재해서 충분히 큰 \(N\)에서

$$
W_N(a)\le m+\eta
\qquad(a\in B_{N,\eta})
$$

이고

$$
r_{N,\eta}:=\mu_N(B_{N,\eta})>0,
\qquad
\frac1{\beta_N}\log\frac1{r_{N,\eta}}\to0.
$$

### 정리 3.1: scaled recovery concentration

가정 J 아래에서 모든 open \(U\supset M\)에 대해

$$
\bar\nu_{N,\beta_N}(U)\to1.
$$

특히 \(M=\{\gamma_*\}\)이면

$$
\bar\nu_{N,\beta_N}\Rightarrow\delta_{\gamma_*}.
$$

증명:

open \(U\supset M\)을 잡고 outer gap \(\delta_U>0\)를 택한다. \(\eta=\delta_U/2\)로 둔다.

분모는 recovery set에서

$$
Z_{N,\beta_N}
\ge
\sum_{a\in B_{N,\eta}}e^{-\beta_NW_N(a)}\mu_N(a)
\ge
e^{-\beta_N(m+\eta)}r_{N,\eta}.
$$

반면 \(\iota_N(a)\notin U\)이면 \(W_N(a)\ge m+\delta_U\)이므로

$$
\sum_{\iota_N(a)\notin U}
e^{-\beta_NW_N(a)}\mu_N(a)
\le
e^{-\beta_N(m+\delta_U)}.
$$

따라서

$$
\bar\nu_{N,\beta_N}(\Gamma\setminus U)
\le
\frac{e^{-\beta_N(m+\delta_U)}}
{e^{-\beta_N(m+\eta)}r_{N,\eta}}
=
\exp[-\beta_N(\delta_U-\eta)]
\frac1{r_{N,\eta}}.
$$

로그를 취하면

$$
\log \bar\nu_{N,\beta_N}(\Gamma\setminus U)
\le
-\beta_N\delta_U/2
+\log\frac1{r_{N,\delta_U/2}}.
$$

scaled recovery mass 때문에 오른쪽은 \(-\infty\)로 간다. 따라서 바깥 질량은 0으로 간다.

만약 \(M=\{\gamma_*\}\)이면 임의의 열린근방 \(U\ni\gamma_*\)에 대해 \(\bar\nu_{N,\beta_N}(U)\to1\)이다. metric space에서 이는 \(\delta_{\gamma_*}\)로의 약수렴을 준다. 끝.

## 4. outer gap consistency를 얻는 충분조건

가정 J의 outer gap은 CE continuum energy \(W\)에서 보통 나온다.

### 정리 4.1: continuum gap + lower energy consistency

\(\Gamma\)가 metric space이고 \(W:\Gamma\to[0,\infty]\)가 l.s.c.이며 compact sublevel을 갖는다고 하자. \(M=\operatorname*{argmin}W\)가 nonempty compact이고, \(m=\min W\)라 하자.

임의의 open \(U\supset M\)에 대해 어떤 \(c_U>0\)가 존재해서

$$
\gamma\notin U
\quad\Longrightarrow\quad
W(\gamma)\ge m+c_U
$$

이다.

또한 어떤 \(N_U\) 이후로

$$
W_N(a)\ge W(\iota_N(a))-c_U/2
\qquad(a\in A_N)
$$

이면 가정 J의 outer gap consistency가

$$
\delta_U=c_U/2
$$

로 성립한다.

증명:

첫 문장은 [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 gap 논리와 같다. 만약 \(c_U=0\)이면 \(\gamma_n\notin U\)이고 \(W(\gamma_n)\to m\)인 열이 있다. compact sublevel에서 부분열을 잡으면 \(\gamma_n\to\gamma\notin U\)이고 l.s.c.에 의해 \(W(\gamma)\le m\)이다. 따라서 \(\gamma\in M\subset U\), 모순이다.

이제 \(\iota_N(a)\notin U\)이면 \(W(\iota_N(a))\ge m+c_U\)이고

$$
W_N(a)\ge W(\iota_N(a))-c_U/2\ge m+c_U/2.
$$

끝.

주의:

- locally uniform \(W_N\to W\)와 equicoercivity가 있으면 이 lower consistency를 보통 얻는다.
- Gamma 수렴은 minimizer 안정성에는 강하지만, Gibbs 분모에는 scaled recovery mass가 별도로 필요하다.

## 5. fixed-\(\beta\) thermal convergence

joint zero-temperature 농축과 별개로, finite Gibbs ensemble이 continuum Gibbs ensemble을 근사하는지도 확인해야 한다.

### 가정 T

1. \(\Gamma\)는 compact metric space다.
2. \(W:\Gamma\to\mathbb R\)는 continuous다.
3. pushed prior

$$
\bar\mu_N=(\iota_N)_*\mu_N
$$

가

$$
\bar\mu_N\Rightarrow\mu_{\mathrm{base}}
$$

로 약수렴한다.
4. energy reconstruction error가 균등하게 사라진다.

$$
\varepsilon_N:=\sup_{a\in A_N}|W_N(a)-W(\iota_N(a))|\to0.
$$

continuum Gibbs measure는

$$
\nu_\beta(d\gamma)
=
\frac{e^{-\beta W(\gamma)}}{Z_\beta}
\mu_{\mathrm{base}}(d\gamma).
$$

### 정리 5.1: fixed-\(\beta\) Gibbs convergence

가정 T 아래에서 고정된 \(\beta<\infty\)마다

$$
\bar\nu_{N,\beta}\Rightarrow\nu_\beta.
$$

증명:

bounded continuous \(f\)를 잡는다. numerator는

$$
\int f\,d\bar\nu_{N,\beta}\cdot Z_{N,\beta}
=
\sum_{a\in A_N}
f(\iota_N(a))e^{-\beta W_N(a)}\mu_N(a).
$$

균등오차 \(\varepsilon_N\to0\) 때문에

$$
\sup_a
\left|
f(\iota_N(a))e^{-\beta W_N(a)}
-f(\iota_N(a))e^{-\beta W(\iota_N(a))}
\right|
\to0.
$$

따라서 numerator는

$$
\int f(\gamma)e^{-\beta W(\gamma)}\bar\mu_N(d\gamma)
$$

와 같은 극한을 갖는다. \(\bar\mu_N\Rightarrow\mu_{\mathrm{base}}\)이고 \(f e^{-\beta W}\)는 continuous bounded이므로

$$
\int f e^{-\beta W}d\bar\mu_N
\to
\int f e^{-\beta W}d\mu_{\mathrm{base}}.
$$

\(f=1\)을 적용하면 \(Z_{N,\beta}\to Z_\beta\)도 얻는다. \(Z_\beta>0\)이므로 비율의 극한이 곧 \(\bar\nu_{N,\beta}\Rightarrow\nu_\beta\)다. 끝.

## 6. 두 극한을 합치는 방법

두 종류의 닫힘이 있다.

| 목적 | 필요한 정리 |
|---|---|
| finite mesh zero-temperature가 continuum minimizer로 바로 가는가 | 정리 3.1 |
| fixed thermal ensemble이 continuum CE Gibbs measure를 근사하는가 | 정리 5.1 |
| \(\beta\to\infty\) continuum 농축과 finite approximation을 함께 쓰는가 | diagonal argument |

### 정리 6.1: fixed-\(\beta\) convergence + continuum concentration의 diagonal

정리 5.1이 모든 \(\beta=k\in\mathbb N\)에 대해 성립하고, continuum Gibbs measure가

$$
\nu_k\Rightarrow\delta_{\gamma_*}
\qquad(k\to\infty)
$$

라고 하자. 그러면 어떤 증가열 \(N(k)\to\infty\)가 존재해서

$$
\bar\nu_{N(k),k}\Rightarrow\delta_{\gamma_*}.
$$

증명:

weak convergence를 metrize하는 거리 \(d\)를 하나 택한다. 정리 5.1에 의해 각 \(k\)마다 충분히 큰 \(N(k)\)를 잡아

$$
d(\bar\nu_{N(k),k},\nu_k)<1/k
$$

가 되게 할 수 있다. 그러면

$$
d(\bar\nu_{N(k),k},\delta_{\gamma_*})
\le
1/k+d(\nu_k,\delta_{\gamma_*})
\to0.
$$

끝.

주의:

이 정리는 schedule의 존재를 준다. 임의의 \(\beta_N\)에 대한 명시적 보장은 정리 3.1의 scaled recovery 조건이 담당한다.

## 7. Gamma 수렴만으로 부족한 이유

[02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)는 이미 Gamma 수렴이 positive-mass recovery 없이는 Gibbs 분모를 닫지 못한다고 정리했다.

05h에서 더 강하게 확인한 것은 다음이다.

> finite mesh에서 모든 후보 weight가 양수여도, 그 양수성이 \(\beta_N\) scale에서 충분하지 않으면 minimizer가 joint limit에서 사라진다.

따라서 finite-to-continuum CE bridge의 올바른 조건은

$$
\text{energy convergence}
+\text{outer gap}
+\text{scaled recovery mass}
$$

다.

## 8. CE 권장 A3 공리

05f의 A2'가 action/topology를 정하고, 05g의 A2''가 prior/support를 정했다면, finite-to-continuum bridge에는 다음 A3를 둔다.

> **A3 finite-to-continuum axiom.**  
> finite mesh package \((A_N,W_N,\mu_N,\iota_N)\)와 inverse-temperature schedule \(\beta_N\to\infty\)는 다음을 만족한다.
> 
> 1. continuum minimizer set \(M\) 바깥에는 uniform outer gap이 있다.
> 2. finite energy \(W_N\)은 이 outer gap을 깨지 않는 lower consistency를 갖는다.
> 3. near-minimizer mesh set \(B_{N,\eta}\)가 있고
> \[
> \frac1{\beta_N}\log\frac1{\mu_N(B_{N,\eta})}\to0
> \]
> 를 만족한다.
> 4. fixed thermal CE ensemble까지 근사하려면 \((\iota_N)_*\mu_N\Rightarrow\mu_{\mathrm{base}}\)와 fixed-\(\beta\) energy consistency를 추가한다.

이 공리 아래에서

$$
\bar\nu_{N,\beta_N}(U)\to1
\qquad(U\supset M)
$$

이다.

## 9. 결론

CE finite mesh package와 continuum package의 일관성은 다음으로 닫힌다.

$$
\boxed{
\begin{gathered}
W_N\ \text{preserves outer gaps},\\
\mu_N(B_{N,\eta})\ge e^{-o(\beta_N)},\\
\beta_N\to\infty
\end{gathered}
\Longrightarrow
\bar\nu_{N,\beta_N}\ \text{concentrates on}\ M.
}
$$

따라서 지금까지의 CE pathspace bridge는

$$
\boxed{
W^{1,p}/C^0
+\text{ good-rate }W
+\text{ recovery prior}
+\text{ finite-to-continuum scale}
\Longrightarrow
\text{manifest path concentration}.
}
$$

다음 병목은 물리적 continuum prior의 실제 선택이다. 즉 Brownian/Sobolev/Gaussian path prior가 CE가 선택한 \(W^{1,p}/C^0\) 경로공간에서 어떤 support를 갖는지, 그리고 \(S_{\mathrm{supp}}\)의 물리 형태가 어떤 recovery set을 만드는지 확인해야 한다.
