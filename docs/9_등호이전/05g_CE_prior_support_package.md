# 05g. CE Prior/Support Package

## 0. 목표

[05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)는 CE continuum 경로공간의 기본 package를

$$
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,M),
\qquad
\text{readout topology}=C^0
$$

로 고정했다. [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 농축 정리를 적용하려면 마지막으로 prior/support 조건이 필요하다.

핵심 조건:

$$
\mu_{\mathrm{base}}
\big(\{\gamma:W[\gamma]<W_{\min}+\eta\}\big)>0
\qquad(\eta>0).
$$

이 문서는 이 조건이 언제 자동으로 닫히고, 언제 별도 공리로 남는지 정리한다.

핵심 결론:

> full support만으로는 충분하지 않다. \(W\)가 l.s.c.일 뿐이면 near-minimum set이 열린집합을 포함하지 않을 수 있다. 자동 recovery를 얻으려면 finite positive prior, minimizer atom, \(W\)의 minimizer 근방 연속성, 또는 positive tube recovery를 요구해야 한다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| finite positive prior | `[정리]` | minimizer 후보가 양의 weight를 가짐 |
| full support + \(W\) continuous at minimizer | `[정리]` | near-minimum set이 열린 근방을 포함 |
| l.s.c.와 full support의 recovery 함의 | `[정리]` 둘만으로 recovery가 따르지 않음 | 정리 2.1 |
| \(\mu_{\mathrm{CE}}\) recovery | `[정리]` reference recovery와 동치 | \(e^{-W}\) reweighting은 null set을 새로 살리지 않음 |
| physical reference measure construction | `[미완성]` | 형식적 \(\mathcal D\gamma\)의 실제 support가 필요 |

## 1. 표기 정리

기존 CE 문서의 \(\mu_0\)는 \(\sigma\)-finite reference measure다. PreEq 문서의 \(\mu_0\)는 probability prior다. 따라서 이 문서에서는 다음 표기를 쓴다.

| 기호 | 의미 |
|---|---|
| \(\mu_{\mathrm{ref}}\) | CE reference measure, possibly \(\sigma\)-finite |
| \(\mu_{\mathrm{CE}}\) | \(e^{-W}\mu_{\mathrm{ref}}\)를 정규화한 CE probability |
| \(\mu_{\mathrm{base}}\) | Gibbs/readout 농축을 시작하는 probability prior |

CE 확률측도는

$$
d\mu_{\mathrm{CE}}(\gamma)
=
\frac{e^{-W[\gamma]}}{Z_W}\,d\mu_{\mathrm{ref}}(\gamma),
\qquad
Z_W=\int e^{-W[\gamma]}d\mu_{\mathrm{ref}}(\gamma).
$$

\(\mu_{\mathrm{ref}}\)가 \(\sigma\)-finite일 뿐이면 그 자체는 PreEq prior가 아니다. raw reference prior를 쓰려면 먼저 finite normalization이 있어야 한다. 일반적으로 안전한 선택은

$$
\mu_{\mathrm{base}}:=\mu_{\mathrm{CE}}
$$

이다.

## 2. Recovery mass는 자동이 아니다

### 정의 2.1: recovery mass

\(\Gamma\)를 topological space, \(\mu\in\mathcal P(\Gamma)\), \(W:\Gamma\to[0,\infty]\)라 하자. \(S=\operatorname{supp}\mu\) 위의 최소값을

$$
W_{\min}=\inf_{\gamma\in S}W[\gamma]
$$

라 둔다. recovery mass 조건은 모든 \(\eta>0\)에 대해

$$
R_\eta(W)
:=
\{\gamma:W[\gamma]<W_{\min}+\eta\}
$$

가 양의 \(\mu\)-질량을 갖는다는 뜻이다.

$$
\mu(R_\eta(W))>0.
$$

이 조건은 Gibbs 분모의 하한을 만든다.

$$
Z_\beta
\ge
e^{-\beta(W_{\min}+\eta)}
\mu(R_\eta(W)).
$$

### 정리 2.1 (full support와 l.s.c.의 recovery no-go)

`[정리]` full support와 lower semicontinuity만으로 positive recovery mass는 일반적으로 따르지 않는다.

\(\Gamma=[0,1]\), \(\mu\)를 Lebesgue measure라고 하자. \(\mu\)는 full support다. 에너지를

$$
W(0)=0,
\qquad
W(x)=1\quad(x>0)
$$

로 둔다. 그러면 \(W\)는 l.s.c.이고 compact sublevel을 가진다. 최소점 \(0\)도 \(\operatorname{supp}\mu\) 안에 있다.

하지만 \(0<\eta<1\)이면

$$
R_\eta(W)=\{0\}
$$

이고

$$
\mu(R_\eta(W))=0.
$$

따라서 full support와 l.s.c.만으로는 recovery mass가 나오지 않는다.

해석:

> support는 열린 근방의 질량을 보장하지만, l.s.c. 에너지의 strict near-minimum set은 열린 근방을 포함하지 않을 수 있다.

## 3. Recovery가 자동으로 닫히는 경우

### 정리 3.1: finite positive prior

\(A\)가 finite set이고 \(\mu(a)>0\) for all \(a\in A\)라고 하자. \(W:A\to\mathbb R\)가 finite-valued이면 모든 \(\eta>0\)에 대해

$$
\mu(R_\eta(W))>0.
$$

증명:

최소점 \(a_*\in A\)가 존재하고 \(W(a_*)=W_{\min}\)이다. 모든 \(\eta>0\)에 대해 \(a_*\in R_\eta(W)\)이므로

$$
\mu(R_\eta(W))\ge\mu(a_*)>0.
$$

끝.

### 정리 3.2: atom at a minimizer

\(\Gamma\)가 임의의 measurable space이고 \(\gamma_*\)가 \(W\)의 support 위 minimizer라고 하자. 만약

$$
\mu(\{\gamma_*\})>0
$$

이면 recovery mass가 성립한다.

증명:

모든 \(\eta>0\)에 대해 \(\gamma_*\in R_\eta(W)\)이므로

$$
\mu(R_\eta(W))\ge\mu(\{\gamma_*\})>0.
$$

끝.

### 정리 3.3: full support + minimizer continuity

\(\Gamma\)가 topological space이고 \(\mu\)가 full support라고 하자. 즉 모든 nonempty open \(U\subset\Gamma\)에 대해 \(\mu(U)>0\)이다. \(W\)가 어떤 minimizer \(\gamma_*\in\operatorname{supp}\mu\)에서 continuous이면 recovery mass가 성립한다.

증명:

\(W(\gamma_*)=W_{\min}\)이다. 임의의 \(\eta>0\)에 대해 \(W\)가 \(\gamma_*\)에서 continuous이므로 어떤 열린 근방 \(U_\eta\ni\gamma_*\)가 존재해서

$$
W[\gamma]<W_{\min}+\eta,
\qquad \gamma\in U_\eta
$$

이다. 따라서

$$
U_\eta\subset R_\eta(W).
$$

\(\mu\)가 full support이므로 \(\mu(U_\eta)>0\), 따라서 \(\mu(R_\eta(W))>0\)이다. 끝.

### 정리 3.4: positive tube recovery

다음 조건을 positive tube recovery라고 부른다.

> 모든 \(\eta>0\)에 대해 어떤 Borel set \(V_\eta\)가 존재해서
> \[
> \mu(V_\eta)>0,
> \qquad
> \sup_{\gamma\in V_\eta}W[\gamma]\le W_{\min}+\eta.
> \]

그러면 recovery mass가 성립한다.

증명:

\(V_\eta\subset\{\gamma:W[\gamma]\le W_{\min}+\eta\}\)이다. strict inequality가 필요하면 \(\eta\) 대신 \(\eta/2\)를 적용하면 된다.

$$
V_{\eta/2}\subset R_\eta(W)
$$

이고 \(\mu(V_{\eta/2})>0\)이므로 \(\mu(R_\eta(W))>0\)이다. 끝.

## 4. CE reweighting은 support를 새로 만들지 않는다

CE base prior를

$$
d\mu_{\mathrm{CE}}
=
Z_W^{-1}e^{-W}d\mu_{\mathrm{ref}}
$$

로 둘 때, recovery mass는 사실상 \(\mu_{\mathrm{ref}}\)의 recovery mass와 동치다.

### 정리 4.1: \(\mu_{\mathrm{CE}}\)와 \(\mu_{\mathrm{ref}}\)의 near-minimum 질량

\(W_{\min}>-\infty\), \(Z_W<\infty\)라고 하자. 그러면 모든 \(\eta>0\)에 대해

$$
\mu_{\mathrm{CE}}(R_\eta(W))>0
\quad\Longleftrightarrow\quad
\mu_{\mathrm{ref}}(R_\eta(W))>0.
$$

증명:

한 방향은 절대연속성으로 즉시 따른다.

$$
\mu_{\mathrm{ref}}(R_\eta(W))=0
\Rightarrow
\mu_{\mathrm{CE}}(R_\eta(W))=0.
$$

반대로 \(\mu_{\mathrm{ref}}(R_\eta(W))>0\)라고 하자. \(R_\eta(W)\) 위에서는 \(W<W_{\min}+\eta\)이므로

$$
e^{-W}>e^{-(W_{\min}+\eta)}.
$$

따라서

$$
\mu_{\mathrm{CE}}(R_\eta(W))
=
\frac1{Z_W}\int_{R_\eta(W)}e^{-W}d\mu_{\mathrm{ref}}
\ge
\frac{e^{-(W_{\min}+\eta)}}{Z_W}
\mu_{\mathrm{ref}}(R_\eta(W))
>0.
$$

끝.

결론:

> \(\mu_{\mathrm{CE}}\)는 \(\mu_{\mathrm{ref}}\)가 보지 못하는 near-minimum set을 새로 보게 만들지 않는다. CE reweighting은 질량을 재분배하지만 support 문제를 해결하지는 않는다.

## 5. Full-support prior의 존재

\(\Gamma\)가 separable metric space이면 full-support Borel probability는 항상 만들 수 있다.

### 정리 5.1: dense atomic full-support prior

\(\Gamma\)가 separable metric space라고 하자. 조밀한 열 \(\{\gamma_n\}_{n\ge1}\)를 잡고

$$
\mu_{\mathrm{dense}}
=
\sum_{n=1}^\infty 2^{-n}\delta_{\gamma_n}
$$

라 두면 \(\mu_{\mathrm{dense}}\in\mathcal P(\Gamma)\)이고 full support다.

증명:

전체 질량은 \(\sum 2^{-n}=1\)이다. 임의의 nonempty open \(U\)를 잡으면 dense property 때문에 어떤 \(n\)에 대해 \(\gamma_n\in U\)이다. 따라서

$$
\mu_{\mathrm{dense}}(U)\ge2^{-n}>0.
$$

끝.

주의:

- 이 prior는 수학적으로는 full support를 보장한다.
- 그러나 \(W\)가 l.s.c.일 뿐이면 full support만으로 recovery가 안 된다.
- dense atomic prior의 존재는 `[정리]`지만, 이를 물리적 \(\mathcal D\gamma\)와 동일시할 근거는 없어 `[미완성]`이다.

## 6. CE 권장 prior package

CE 9_등호이전 문서군에서는 다음 package를 권장한다.

### Package F: finite mesh

$$
\mathcal P_{I,N}=\{\gamma_1,\dots,\gamma_N\},
\qquad
\mu_N(\gamma_i)>0.
$$

그러면 recovery mass는 정리 3.1로 자동이다.

출처:

`[정리]`

### Package C: continuous energy, full-support prior

\(\mathcal P_I\)에 \(C^0\) topology를 쓰고 \(\mu_{\mathrm{base}}\)가 full support라고 하자. \(W\)가 적어도 하나의 minimizer \(\gamma_*\)에서 continuous이면 recovery mass는 정리 3.3으로 자동이다.

출처:

`[정리]`

### Package T: tube recovery axiom

\(W\)가 l.s.c.일 뿐인 Tonelli package에서는 아래를 명시적 support axiom으로 둔다.

$$
\forall\eta>0,\quad
\exists V_\eta\in\mathcal B(\mathcal P_I):
\mu_{\mathrm{base}}(V_\eta)>0,\quad
\sup_{V_\eta}W\le W_{\min}+\eta.
$$

출처:

`[정리]`

### Package A: atomized minimizer recovery

minimizer를 알고 있거나 finite approximation에서 선택 후보를 명시할 수 있다면 minimizer 또는 near-minimizer sequence에 atom을 둘 수 있다.

예:

$$
\mu_{\mathrm{rec}}
=
\sum_{n=1}^\infty 2^{-n}\delta_{\gamma_n},
\qquad
W[\gamma_n]<W_{\min}+1/n.
$$

그러면 recovery mass는 자동이다. 하지만 이 prior는 \(W\)를 보고 만든 것이므로 물리적 prior라기보다 proof device 또는 algorithmic search prior다.

출처:

search prior 채택은 `[공리: 모델 선택]`; recovery는 `[정리]`

## 7. 권장 A2'' prior/support 공리

05f의 A2'가 action/topology를 정했다면, 이 문서는 다음 A2''를 제안한다.

> **A2'' prior/support axiom.**  
> \(\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)\)는 \(C^0\) Borel probability이고, \(W\)의 support 위 최소값 \(W_{\min}\)에 대해 모든 \(\eta>0\)에서
> \[
> \mu_{\mathrm{base}}\{\gamma:W[\gamma]<W_{\min}+\eta\}>0
> \]
> 를 만족한다.

이 공리는 다음 중 하나로 검증될 수 있다.

| 검증 방식 | 충분조건 |
|---|---|
| finite mesh | 모든 후보 weight 양수 |
| continuity route | \(\mu_{\mathrm{base}}\) full support, \(W\) continuous at a minimizer |
| tube route | positive tube recovery |
| atom route | minimizer/near-minimizer atom |
| CE route | \(\mu_{\mathrm{ref}}\)가 recovery mass를 갖고 \(d\mu_{\mathrm{CE}}\propto e^{-W}d\mu_{\mathrm{ref}}\) |

## 8. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| full support와 l.s.c.만으로 recovery가 따르지 않음 | `[정리]` 2.1 |
| finite positive prior recovery | 정리 3.1 |
| atom recovery | 정리 3.2 |
| full support + minimizer continuity recovery | 정리 3.3 |
| positive tube recovery | 정리 3.4 |
| \(\mu_{\mathrm{CE}}\) recovery iff \(\mu_{\mathrm{ref}}\) recovery | 정리 4.1 |
| separable space full-support probability existence | 정리 5.1 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| physical \(\mu_{\mathrm{ref}}\) construction | 형식적 \(\mathcal D\gamma\)를 실제 measure/prior로 고정 |
| Brownian/Sobolev/Gaussian support theorem | chosen prior가 \(W^{1,p}/C^0\) 경로공간에서 full support인지 확인 |
| finite-to-continuum consistency | finite mesh priors \(\mu_N\)이 continuum prior/action으로 수렴하는지 확인 |
| \(S_{\mathrm{supp}}\) physical form | residual/curvature/obstacle penalty 중 선택 |

## 9. 결론

CE pathspace bridge의 prior/support 조건은 다음으로 고정한다.

$$
\boxed{
\mu_{\mathrm{base}}
\big(\{\gamma:W[\gamma]<W_{\min}+\eta\}\big)>0
\quad(\eta>0).
}
$$

이 조건이 있으면 [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)와 [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)를 결합해

$$
\boxed{
W^{1,p}/C^0
+\text{ good-rate }W
+\text{ recovery prior}
\Longrightarrow
\text{manifest path concentration}.
}
$$

가 된다.

다음 병목은 finite mesh package와 continuum package의 일관성이다. 즉 \(W_N\to W\), \(\mu_N\to\mu_{\mathrm{base}}\)일 때 finite PreEq 농축이 continuum CE 농축으로 수렴하는지 확인해야 한다. 이 joint-limit 조건은 [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md)에서 scaled recovery mass package로 분리해 닫는다.
