# 05e. CE Good-Rate Theorem

## 0. 목표

[05d_pathspace_audit.md](05d_pathspace_audit.md)는 CE 문서 안에 이미 \(\mathcal P_I\), \(\mu_{\mathrm{ref}}\), \(W\), \(F\), \(\mathcal P_{\mathrm{ns}}\), \(K_\phi\)가 들어 있음을 확인했다. 남은 병목은 하나였다.

$$
W[\gamma]
=
\frac{S_E[\gamma]}{\hbar}+S_{\mathrm{supp}}[\gamma]
$$

가 실제 경로공간에서 good rate function인가?

이 문서는 그 병목을 두 층으로 닫는다.

1. **추상 정리**: \(E\)가 l.s.c.이고 compact sublevel을 가지며 recovery mass가 있으면 Gibbs 측도는 minimizer set으로 농축한다.
2. **CE 충분조건**: \(S_E\)가 compact sublevel을 만들고 \(S_{\mathrm{supp}}\ge0\)가 l.s.c.이면 \(W\)도 good rate function이 된다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| 추상 good-rate 농축 | `[정리]` | 아래 정리 1.1에서 직접 증명 |
| \(W\to E_{\mathrm{fold}}\) shift | `[정리]` | 상수 이동은 minimizer와 Gibbs 비율을 보존 |
| \(S_E+S_{\mathrm{supp}}\) sufficient package | `[정리]` | l.s.c.와 compact sublevel 보존 |
| 실제 CE \(S_E\), \(S_{\mathrm{supp}}\) 검증 | `[미완성]` | 경로공간 위상과 suppress action의 해석학적 정의가 더 필요 |

## 1. 추상 good-rate 농축 정리

### 세팅

\(\Gamma\)를 Polish space라 하자. \(\mu\in\mathcal P(\Gamma)\)는 Borel probability measure이고

$$
S=\operatorname{supp}\mu
$$

라 둔다. 에너지

$$
E:\Gamma\to[0,\infty]
$$

가 다음을 만족한다고 하자.

1. \(E\)는 lower semicontinuous다.
2. 모든 \(c<\infty\)에 대해 \(\{\gamma:E(\gamma)\le c\}\)는 compact다.
3. \(m:=\inf_{\gamma\in S}E(\gamma)<\infty\).
4. recovery mass:

$$
R_\eta:=\{\gamma\in\Gamma:E(\gamma)<m+\eta\}
$$

가 모든 \(\eta>0\)에 대해

$$
\mu(R_\eta)>0
$$

를 만족한다.

Gibbs 재가중은

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta E(\gamma)}}{Z_\beta}\,\mu(d\gamma),
\qquad
Z_\beta=\int_\Gamma e^{-\beta E(\gamma)}\mu(d\gamma)
$$

이다.

최소집합은 support 위에서 잡는다.

$$
\Gamma_*=
\operatorname*{argmin}_{\gamma\in S}E(\gamma)
=
\{\gamma\in S:E(\gamma)=m\}.
$$

### 정리 1.1: good-rate 농축

위 가정 아래에서 \(\Gamma_*\)는 공집합이 아닌 compact set이다. 또한 모든 open \(U\supset\Gamma_*\)에 대해

$$
\mu_\beta(U)\to1
\qquad(\beta\to\infty).
$$

특히 \(\Gamma_*=\{\gamma_*\}\)이면

$$
\mu_\beta\Rightarrow\delta_{\gamma_*}.
$$

### 증명

먼저 minimizer 존재를 보인다. \(E(\gamma_n)\to m\)인 \(\gamma_n\in S\)를 잡는다. 충분히 큰 \(n\)에 대해 \(\gamma_n\in\{E\le m+1\}\)이고, 이 sublevel은 compact다. 부분열을 잡아 \(\gamma_{n_j}\to\gamma_*\)라 하자. \(S\)는 닫힌집합이므로 \(\gamma_*\in S\)이다. l.s.c.에 의해

$$
E(\gamma_*)\le\liminf_jE(\gamma_{n_j})=m.
$$

따라서 \(E(\gamma_*)=m\), 즉 \(\Gamma_*\ne\varnothing\)다. \(\Gamma_*=\{E\le m\}\cap S\)는 compact sublevel의 닫힌 부분집합이므로 compact다.

이제 \(U\supset\Gamma_*\)를 open set으로 잡는다. \(F=S\setminus U\)라 두자. \(F=\varnothing\)이면 자명하다. \(F\ne\varnothing\)라 하자. \(F\)는 닫혀 있다.

claim:

$$
\delta_U:=\inf_{\gamma\in F}E(\gamma)-m>0.
$$

만약 \(\delta_U=0\)이면 \(E(\eta_n)\to m\)인 \(\eta_n\in F\)를 잡을 수 있다. 위와 같은 compact sublevel 논리로 부분열 \(\eta_{n_j}\to\eta\in F\)를 얻는다. l.s.c.에 의해 \(E(\eta)\le m\), 따라서 \(\eta\in\Gamma_*\subset U\)다. 그러나 \(\eta\in F=S\setminus U\)이므로 모순이다.

이제 \(\eta=\delta_U/2\)를 택한다. 분모는 recovery mass로부터

$$
Z_\beta
\ge
\int_{R_\eta}e^{-\beta E(\gamma)}\mu(d\gamma)
\ge
e^{-\beta(m+\eta)}\mu(R_\eta)
$$

이다. 분자는

$$
\int_{\Gamma\setminus U}e^{-\beta E(\gamma)}\mu(d\gamma)
=
\int_{F}e^{-\beta E(\gamma)}\mu(d\gamma)
\le
e^{-\beta(m+\delta_U)}.
$$

따라서

$$
\mu_\beta(\Gamma\setminus U)
\le
\frac{1}{\mu(R_{\delta_U/2})}
e^{-\beta\delta_U/2}
\to0.
$$

즉 \(\mu_\beta(U)\to1\)이다.

마지막으로 \(\Gamma_*=\{\gamma_*\}\)라고 하자. bounded continuous \(f\)와 \(\varepsilon>0\)를 잡는다. 연속성으로 어떤 open \(U\ni\gamma_*\)가 존재해서 \(U\) 위에서 \(|f(\gamma)-f(\gamma_*)|<\varepsilon\)이다. 그러면

$$
\left|\int f\,d\mu_\beta-f(\gamma_*)\right|
\le
\varepsilon
+2\|f\|_\infty\mu_\beta(\Gamma\setminus U)
$$

이고 오른쪽의 \(\limsup\)는 \(\varepsilon\) 이하이다. \(\varepsilon\)은 임의이므로 약수렴이 따른다. 끝.

## 2. CE \(W\)에서 \(E_{\mathrm{fold}}\)로

CE 선택함수는

$$
F[\gamma]=W[\gamma]+c
$$

로 읽을 수 있다. \(W_{\min}:=\inf_{\gamma\in S}W[\gamma]\)가 유한하고 \(W\)가 good rate function이면

$$
E_{\mathrm{fold}}(\gamma)
=
W[\gamma]-W_{\min}
$$

를 둔다.

그러면

$$
E_{\mathrm{fold}}\ge0,
\qquad
\operatorname{argmin}E_{\mathrm{fold}}
=
\operatorname{argmin}W
=
\operatorname{argmin}F.
$$

또한 \(W\)의 sublevel과 \(E_{\mathrm{fold}}\)의 sublevel은 상수만큼 이동한 같은 집합이다.

$$
\{E_{\mathrm{fold}}\le c\}
=
\{W\le W_{\min}+c\}.
$$

따라서 \(W\)가 l.s.c.이고 compact sublevel을 가지면 \(E_{\mathrm{fold}}\)도 그렇다. 정리 1.1을 적용하면

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta(W[\gamma]-W_{\min})}}{Z_\beta}\,\mu_{\mathrm{base}}(d\gamma)
$$

는 \(\operatorname{argmin}W\)로 농축한다.

## 3. \(S_E+S_{\mathrm{supp}}\) sufficient package

이제 CE action 형태를 넣는다.

$$
W[\gamma]
=
\frac{S_E[\gamma]}{\hbar}
+S_{\mathrm{supp}}[\gamma],
\qquad \hbar>0.
$$

### 정리 3.1: suppress cost가 good-rate를 보존하는 조건

가정:

1. \(S_E:\mathcal P_I\to[0,\infty]\)는 l.s.c.다.
2. \(S_E\)의 sublevel set \(\{S_E\le c\}\)는 모든 \(c<\infty\)에서 compact다.
3. \(S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]\)는 l.s.c.다.

그러면 \(W=S_E/\hbar+S_{\mathrm{supp}}\)는 l.s.c.이고 compact sublevel을 가진다.

### 증명

l.s.c. 함수의 양의 상수배와 합은 l.s.c.다. 따라서 \(W\)는 l.s.c.다.

또한 \(S_{\mathrm{supp}}\ge0\)이므로

$$
W[\gamma]\le c
\quad\Longrightarrow\quad
S_E[\gamma]\le \hbar c.
$$

따라서

$$
\{W\le c\}
\subset
\{S_E\le\hbar c\}.
$$

오른쪽은 compact이고, 왼쪽은 \(W\)의 l.s.c. 때문에 닫힌집합이다. compact set의 닫힌 부분집합은 compact이므로 \(\{W\le c\}\)는 compact다. 끝.

### 따름정리 3.2: CE 선택 농축

위 정리 3.1의 가정에 더해 \(\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)\)가 recovery mass

$$
\mu_{\mathrm{base}}\big(\{\gamma:W[\gamma]<W_{\min}+\eta\}\big)>0
\qquad(\eta>0)
$$

를 만족한다고 하자. 그러면

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta(W[\gamma]-W_{\min})}}{Z_\beta}\,
\mu_{\mathrm{base}}(d\gamma)
$$

는

$$
\operatorname*{argmin}_{\operatorname{supp}\mu_{\mathrm{base}}}W
$$

로 농축한다.

증명은 정리 1.1에 \(E= W-W_{\min}\)을 적용하면 끝난다.

## 4. 실제 pathspace에서 compact sublevel을 얻는 방법

여기부터는 실제 CE bridge가 선택해야 하는 해석학 패키지다.

### 4.1 finite mesh package

경로를 \(N\)개의 시간격자로 자르면

$$
\mathcal P_{I,N}\simeq M^{N-1}
$$

이다. \(M\)이 compact이면 \(\mathcal P_{I,N}\)도 compact다. \(M=\mathbb R^d\)이면 \(W_N\)이 continuous/coercive일 때 [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md)를 그대로 쓴다.

출처:

`[정리]`

### 4.2 \(H^1/C^0\) Tonelli package

표준 변분법은 보통 \(C^1\)보다 약한 공간에서 닫힌다. 예를 들어

$$
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,\mathbb R^d),
\qquad p>1
$$

로 잡고 \(C^0\) topology로 readout한다고 하자. Euclidean action이

$$
S_E[\gamma]
=
\int_I L(t,\gamma(t),\dot\gamma(t))\,dt
$$

꼴이고 \(L\)이 다음을 만족한다고 하자.

1. \(L\)은 lower semicontinuous다.
2. \(v\mapsto L(t,q,v)\)는 convex다.
3. 어떤 \(a>0,b\ge0\)에 대해

$$
L(t,q,v)\ge a\|v\|^p-b.
$$

그러면 \(S_E\)는 weak \(W^{1,p}\) topology에서 l.s.c.이고, sublevel은 \(W^{1,p}\)에서 bounded다. \(p>1\)이면 \(W^{1,p}\)는 reflexive이고, \(I\)가 compact이므로 bounded sequence는 \(C^0\)에서 equicontinuous/bounded subsequence를 갖는다. 따라서 \(C^0\) readout topology에서는 compactness를 얻을 수 있다.

요지:

$$
\int_I\|\dot\gamma\|^pdt\le C
\quad\Longrightarrow\quad
\gamma\text{들이 }C^0\text{에서 precompact}.
$$

출처:

표준 변분 가정 아래 `[정리]`

주의: 이 패키지는 \(C^1\)-compactness가 아니라 \(C^0\)-compactness를 준다.

### 4.3 \(C^1\) pathspace를 유지할 때

기존 CE 문서는 \(C^1\) 수렴에 해당하는 pathspace를 말한다. 그러나 kinetic action

$$
\int_I\|\dot\gamma(t)\|^2dt
$$

만으로는 \(\dot\gamma\)의 균등수렴이나 equicontinuity가 보장되지 않는다. 즉 \(C^1\) topology에서는 compact sublevel이 자동으로 나오지 않는다.

\(C^1\)을 유지하려면 추가 제어가 필요하다. 예를 들어 admissible path class를 \(W^{2,p}\) 안의 닫힌 class로 두고 다음 하한을 요구할 수 있다.

$$
S_E[\gamma]\ge
a\|\dot\gamma\|_{L^\infty}^p
+b\int_I\|\ddot\gamma(t)\|^pdt
-c,
\qquad p>1.
$$

또는 \(S_{\mathrm{supp}}\)가 acceleration/curvature penalty를 포함해

$$
\int_I\|\ddot\gamma(t)\|^pdt
$$

를 제어해야 한다. 그러면 \(\gamma\)와 \(\dot\gamma\) 모두 Arzela-Ascoli 조건을 만족하고 \(C^1\) precompactness가 나온다.

출처:

강화된 action 가정 아래 `[정리]`

## 5. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| good-rate Gibbs 농축 | 정리 1.1 |
| \(W\to E_{\mathrm{fold}}\) 상수 shift | 2절 |
| \(S_E\) compact sublevel + \(S_{\mathrm{supp}}\ge0\) l.s.c.이면 \(W\) good-rate | 정리 3.1 |
| finite mesh 경로공간 | 4.1 |
| \(H^1/C^0\) Tonelli package | 4.2 |
| \(C^1\) compactness sufficient condition | 4.3 |

남은 것:

| 병목 | 의미 |
|---|---|
| CE가 실제로 어떤 path topology를 쓰는가 | \(C^1\)인지, \(H^1/C^0\)인지, finite mesh인지 선택 필요 |
| \(S_{\mathrm{supp}}\)의 해석학적 형태 | l.s.c.인지, nonnegative인지, curvature/acceleration을 제어하는지 필요 |
| \(\mu_{\mathrm{base}}\) support | minimizer 근방에 positive mass를 주는지 필요 |
| 최소집합 내부 선택 | 여러 minimizer가 있을 때 어느 branch가 선택되는지는 별도 scale/selection 문제 |

## 6. 결론

CE bridge의 농축 문장은 이제 다음 조건부 정리로 닫힌다.

$$
\boxed{
\begin{gathered}
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}
\text{ is l.s.c. good-rate}
\\
\text{and}\quad
\mu_{\mathrm{base}}\text{ has recovery mass}
\\
\Longrightarrow
\mu_\beta
\text{ concentrates on }
\operatorname{argmin}W.
\end{gathered}
}
$$

하지만 실제 CE action을 완전히 닫으려면 다음 선택이 필요하다.

$$
\boxed{
\text{path topology}
\quad+\quad
\text{analytic form of }S_{\mathrm{supp}}.
}
$$

이 선택 없이는 \(C^1\) 경로공간에서 compact sublevel을 주장하면 안 된다. 운동에너지형 action만 쓰려면 \(H^1/C^0\) 쪽이 자연스럽고, \(C^1\)을 유지하려면 acceleration/curvature suppression이 action 안에 들어가야 한다.

이 action/topology 선택은 [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)에서 \(W^{1,p}/C^0\) 기본 package와 \(C^1\) 강화 package로 분리해 닫는다.
