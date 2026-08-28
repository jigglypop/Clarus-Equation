# 05e. CE Good-Rate Theorem

이 문서는 good-rate functional과 recovery prior 아래 Gibbs 측도가 최소집합 근방으로 농축하는 추상 정리 및 CE 적용의 충분조건을 분리한다. CE action과 pathspace의 실제 선택은 정리가 아니라 물리 모형의 입력이며, compact sublevel은 topology에 의존한다.

독자는 02a와 05의 Gibbs package를 먼저 읽는다. 추상 정리, CE functional, suppression package, 각 topology route와 반례·미완성 범위를 순서대로 확인한다.

## 0. 목표

농축의 핵심은 최소값 존재뿐 아니라 escape를 막는 compact sublevel과 최소점 근방의 prior 질량이다. 이 절은 그 형식 가정과 CE 물리 해석의 경계를 고정한다.

[../검증_원장/등호이전_pathspace_audit.md](../검증_원장/등호이전_pathspace_audit.md)는 CE 문서 안에 이미 $\mathcal P_I$, $\mu_{\mathrm{ref}}$, $W$, $F$, $\mathcal P_{\mathrm{ns}}$, $K_\phi$가 들어 있음을 확인했다. 남은 병목은 하나였다.

$$
W[\gamma]
=
\frac{S_E[\gamma]}{\hbar}+S_{\mathrm{supp}}[\gamma]
$$

가 실제 경로공간에서 good rate function인가?

이 문서는 그 병목을 두 층으로 닫는다.

1. **추상 정리**: $E$가 l.s.c.이고 compact sublevel을 가지며 recovery mass가 있으면 Gibbs 측도는 minimizer set으로 농축한다.
2. **CE 충분조건**: $S_E$가 compact sublevel을 만들고 $S_{\mathrm{supp}}\ge0$가 l.s.c.이면 $W$도 good rate function이 된다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| 추상 good-rate 농축 | `[정리]` | 아래 정리 1.1에서 직접 증명 |
| $W\to E_{\mathrm{fold}}$ shift | `[정리]` | 상수 이동은 minimizer와 Gibbs 비율을 보존 |
| $S_E+S_{\mathrm{supp}}$ sufficient package | `[정리]` | l.s.c.와 compact sublevel 보존 |
| 실제 CE $S_E$, $S_{\mathrm{supp}}$ 검증 | `[미완성]` | 경로공간 위상과 suppress action의 해석학적 정의가 더 필요 |

## 1. 추상 good-rate 농축 정리

다음 정리는 metric topology, lower-semicontinuity, coercivity에 해당하는 compact sublevel, recovery mass를 함께 전제한다. 어느 하나가 빠지면 minimizer 부재·질량 도피·분모 퇴화 반례가 가능하다.

### 세팅

세팅은 확률측도와 무차원 functional의 정의역을 고정한다. 실제 path integral measure나 CE prior가 이 조건을 만족한다는 것은 별도 검증 문제다.

$\Gamma$를 Polish space라 하자. $\mu\in\mathcal P(\Gamma)$는 Borel probability measure이고

$$
S=\operatorname{supp}\mu
$$

라 둔다. 에너지

$$
E:\Gamma\to[0,\infty]
$$

가 다음을 만족한다고 하자.

1. $E$는 lower semicontinuous다.
2. 모든 $c<\infty$에 대해 $\{\gamma:E(\gamma)\le c\}$는 compact다.
3. $m:=\inf_{\gamma\in S}E(\gamma)<\infty$.
4. recovery mass:

$$
R_\eta:=\{\gamma\in\Gamma:E(\gamma)<m+\eta\}
$$

가 모든 $\eta>0$에 대해

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

정리의 결론은 최소집합을 포함하는 열린 근방으로의 질량 집중이다. 유일 최소점, rate, 독립 물리 에너지 보존은 추가 가정 없이는 따라오지 않는다.

위 가정 아래에서 $\Gamma_*$는 공집합이 아닌 compact set이다. 또한 모든 open $U\supset\Gamma_*$에 대해

$$
\mu_\beta(U)\to1
\qquad(\beta\to\infty).
$$

특히 $\Gamma_*=\{\gamma_*\}$이면

$$
\mu_\beta\Rightarrow\delta_{\gamma_*}.
$$

### 증명

증명은 compact sublevel에서의 양의 gap과 recovery mass의 분모 하한을 결합한다. 이 두 단계가 없으면 단순 lower-semicontinuity만으로는 결론을 얻지 못한다.

먼저 minimizer 존재를 보인다. $E(\gamma_n)\to m$인 $\gamma_n\in S$를 잡는다. 충분히 큰 $n$에 대해 $\gamma_n\in\{E\le m+1\}$이고, 이 sublevel은 compact다. 부분열을 잡아 $\gamma_{n_j}\to\gamma_*$라 하자. $S$는 닫힌집합이므로 $\gamma_*\in S$이다. l.s.c.에 의해

$$
E(\gamma_*)\le\liminf_jE(\gamma_{n_j})=m.
$$

따라서 $E(\gamma_*)=m$, 즉 $\Gamma_*\ne\varnothing$다. $\Gamma_*=\{E\le m\}\cap S$는 compact sublevel의 닫힌 부분집합이므로 compact다.

이제 $U\supset\Gamma_*$를 open set으로 잡는다. $F=S\setminus U$라 두자. $F=\varnothing$이면 자명하다. $F\ne\varnothing$라 하자. $F$는 닫혀 있다.

claim:

$$
\delta_U:=\inf_{\gamma\in F}E(\gamma)-m>0.
$$

만약 $\delta_U=0$이면 $E(\eta_n)\to m$인 $\eta_n\in F$를 잡을 수 있다. 위와 같은 compact sublevel 논리로 부분열 $\eta_{n_j}\to\eta\in F$를 얻는다. l.s.c.에 의해 $E(\eta)\le m$, 따라서 $\eta\in\Gamma_*\subset U$다. 그러나 $\eta\in F=S\setminus U$이므로 모순이다.

이제 $\eta=\delta_U/2$를 택한다. 분모는 recovery mass로부터

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

즉 $\mu_\beta(U)\to1$이다.

마지막으로 $\Gamma_*=\{\gamma_*\}$라고 하자. bounded continuous $f$와 $\varepsilon>0$를 잡는다. 연속성으로 어떤 open $U\ni\gamma_*$가 존재해서 $U$ 위에서 $|f(\gamma)-f(\gamma_*)|<\varepsilon$이다. 그러면

$$
\left|\int f\,d\mu_\beta-f(\gamma_*)\right|
\le
\varepsilon
+2\|f\|_\infty\mu_\beta(\Gamma\setminus U)
$$

이고 오른쪽의 $\limsup$는 $\varepsilon$ 이하이다. $\varepsilon$은 임의이므로 약수렴이 따른다. 끝.

## 2. CE $W$에서 $E_{\mathrm{fold}}$로

추상 객체를 CE notation으로 옮기는 것은 정의역·단위·prior를 지정한 조건부 mapping이다. 이 대응이 실제 우주론적 action을 유도한다는 물리 주장은 미완성이다.

CE 선택함수는

$$
F[\gamma]=W[\gamma]+c
$$

로 읽을 수 있다. $W_{\min}:=\inf_{\gamma\in S}W[\gamma]$가 유한하고 $W$가 good rate function이면

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

또한 $W$의 sublevel과 $E_{\mathrm{fold}}$의 sublevel은 상수만큼 이동한 같은 집합이다.

$$
\{E_{\mathrm{fold}}\le c\}
=
\{W\le W_{\min}+c\}.
$$

따라서 $W$가 l.s.c.이고 compact sublevel을 가지면 $E_{\mathrm{fold}}$도 그렇다. 정리 1.1을 적용하면

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta(W[\gamma]-W_{\min})}}{Z_\beta}\,\mu_{\mathrm{base}}(d\gamma)
$$

는 $\operatorname{argmin}W$로 농축한다.

## 3. $S_E+S_{\mathrm{supp}}$ sufficient package

suppression을 더해도 good-rate 성질이 유지되려면 lower-semicontinuity와 비음성 등 명시한 충분조건이 필요하다. 이는 필요한 조건의 완전한 분류가 아니다.

이제 CE action 형태를 넣는다.

$$
W[\gamma]
=
\frac{S_E[\gamma]}{\hbar}
+S_{\mathrm{supp}}[\gamma],
\qquad \hbar>0.
$$

### 정리 3.1: suppress cost가 good-rate를 보존하는 조건

정리는 원 action의 compact sublevel과 suppression의 정규성을 사용한다. 음의 또는 비가측 suppression은 이 결론의 반례가 될 수 있다.

가정:

1. $S_E:\mathcal P_I\to[0,\infty]$는 l.s.c.다.
2. $S_E$의 sublevel set $\{S_E\le c\}$는 모든 $c<\infty$에서 compact다.
3. $S_{\mathrm{supp}}:\mathcal P_I\to[0,\infty]$는 l.s.c.다.

그러면 $W=S_E/\hbar+S_{\mathrm{supp}}$는 l.s.c.이고 compact sublevel을 가진다.

### 증명

증명은 새 sublevel을 기존 compact sublevel의 닫힌 부분으로 가두는 단계다. 실제 CE boundary term과 gauge 처리는 이 간단한 포함관계 밖의 문제다.

l.s.c. 함수의 양의 상수배와 합은 l.s.c.다. 따라서 $W$는 l.s.c.다.

또한 $S_{\mathrm{supp}}\ge0$이므로

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

오른쪽은 compact이고, 왼쪽은 $W$의 l.s.c. 때문에 닫힌집합이다. compact set의 닫힌 부분집합은 compact이므로 $\{W\le c\}$는 compact다. 끝.

### 따름정리 3.2: CE 선택 농축

따름정리는 앞 sufficient package가 충족될 때만 추상 농축을 CE 표기로 다시 쓴다. 이는 채택 action의 실증적 정당화나 단일 물리 경로 선택을 말하지 않는다.

위 정리 3.1의 가정에 더해 $\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)$가 recovery mass

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

증명은 정리 1.1에 $E= W-W_{\min}$을 적용하면 끝난다.

## 4. 실제 pathspace에서 compact sublevel을 얻는 방법

compactness는 선택한 topology에 상대적이므로 mesh, Sobolev, $C^1$ route를 섞을 수 없다. 다음 항목은 충분조건과 실패 경계를 비교한다.

여기부터는 실제 CE bridge가 선택해야 하는 해석학 패키지다.

### 4.1 finite mesh package

finite mesh에서는 유한차원 coercivity가 compactness를 제공하지만 continuum joint limit에는 mesh와 temperature의 추가 scaling이 필요하다.

경로를 $N$개의 시간격자로 자르면

$$
\mathcal P_{I,N}\simeq M^{N-1}
$$

이다. $M$이 compact이면 $\mathcal P_{I,N}$도 compact다. $M=\mathbb R^d$이면 $W_N$이 continuous/coercive일 때 [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md)를 그대로 쓴다.

출처:

`[정리]`

### 4.2 $H^1/C^0$ Tonelli package

Tonelli형 조건은 Sobolev bound를 통해 $C^0$ compactness를 얻는 route다. prior support와 boundary 조건이 맞지 않으면 recovery가 실패할 수 있다.

표준 변분법은 보통 $C^1$보다 약한 공간에서 닫힌다. 예를 들어

$$
\mathcal P_I=W^{1,p}_{x_i,x_f}(I,\mathbb R^d),
\qquad p>1
$$

로 잡고 $C^0$ topology로 readout한다고 하자. Euclidean action이

$$
S_E[\gamma]
=
\int_I L(t,\gamma(t),\dot\gamma(t))\,dt
$$

꼴이고 $L$이 다음을 만족한다고 하자.

1. $L$은 lower semicontinuous다.
2. $v\mapsto L(t,q,v)$는 convex다.
3. 어떤 $a>0,b\ge0$에 대해

$$
L(t,q,v)\ge a\|v\|^p-b.
$$

그러면 $S_E$는 weak $W^{1,p}$ topology에서 l.s.c.이고, sublevel은 $W^{1,p}$에서 bounded다. $p>1$이면 $W^{1,p}$는 reflexive이고, $I$가 compact이므로 bounded sequence는 $C^0$에서 equicontinuous/bounded subsequence를 갖는다. 따라서 $C^0$ readout topology에서는 compactness를 얻을 수 있다.

요지:

$$
\int_I\|\dot\gamma\|^pdt\le C
\quad\Longrightarrow\quad
\gamma\text{들이 }C^0\text{에서 precompact}.
$$

출처:

표준 변분 가정 아래 `[정리]`

주의: 이 패키지는 $C^1$-compactness가 아니라 $C^0$-compactness를 준다.

### 4.3 $C^1$ pathspace를 유지할 때

$C^1$ topology는 더 강한 compactness 자료를 요구한다. $H^1$ coercivity만으로 $C^1$ precompactness를 주장하는 것은 반례 위험이 있다.

기존 CE 문서는 $C^1$ 수렴에 해당하는 pathspace를 말한다. 그러나 kinetic action

$$
\int_I\|\dot\gamma(t)\|^2dt
$$

만으로는 $\dot\gamma$의 균등수렴이나 equicontinuity가 보장되지 않는다. 즉 $C^1$ topology에서는 compact sublevel이 자동으로 나오지 않는다.

$C^1$을 유지하려면 추가 제어가 필요하다. 예를 들어 admissible path class를 $W^{2,p}$ 안의 닫힌 class로 두고 다음 하한을 요구할 수 있다.

$$
S_E[\gamma]\ge
a\|\dot\gamma\|_{L^\infty}^p
+b\int_I\|\ddot\gamma(t)\|^pdt
-c,
\qquad p>1.
$$

또는 $S_{\mathrm{supp}}$가 acceleration/curvature penalty를 포함해

$$
\int_I\|\ddot\gamma(t)\|^pdt
$$

를 제어해야 한다. 그러면 $\gamma$와 $\dot\gamma$ 모두 Arzela-Ascoli 조건을 만족하고 $C^1$ precompactness가 나온다.

출처:

강화된 action 가정 아래 `[정리]`

## 5. 닫힌 것과 남은 것

여기서 닫힌 정리와 실제 CE action·prior·topology의 미완성 선택을 분리한다. 표의 후보 package는 물리 채택 판정이 아니라 검증해야 할 가정의 목록이다.

닫힌 것:

| 항목 | 상태 |
|---|---|
| good-rate Gibbs 농축 | 정리 1.1 |
| $W\to E_{\mathrm{fold}}$ 상수 shift | 2절 |
| $S_E$ compact sublevel + $S_{\mathrm{supp}}\ge0$ l.s.c.이면 $W$ good-rate | 정리 3.1 |
| finite mesh 경로공간 | 4.1 |
| $H^1/C^0$ Tonelli package | 4.2 |
| $C^1$ compactness sufficient condition | 4.3 |

남은 것:

| 병목 | 의미 |
|---|---|
| CE가 실제로 어떤 path topology를 쓰는가 | $C^1$인지, $H^1/C^0$인지, finite mesh인지 선택 필요 |
| $S_{\mathrm{supp}}$의 해석학적 형태 | l.s.c.인지, nonnegative인지, curvature/acceleration을 제어하는지 필요 |
| $\mu_{\mathrm{base}}$ support | minimizer 근방에 positive mass를 주는지 필요 |
| 최소집합 내부 선택 | 여러 minimizer가 있을 때 어느 branch가 선택되는지는 별도 scale/selection 문제 |

## 6. 결론

결론적으로 good-rate 농축은 명시한 topology·coercivity·compact sublevel·recovery mass의 결과다. CE bridge는 이 입력을 실제 모형에서 충족시키는 추가 작업이 남아 있다.

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

이 선택 없이는 $C^1$ 경로공간에서 compact sublevel을 주장하면 안 된다. 운동에너지형 action만 쓰려면 $H^1/C^0$ 쪽이 자연스럽고, $C^1$을 유지하려면 acceleration/curvature suppression이 action 안에 들어가야 한다.

이 action/topology 선택은 [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md)에서 $W^{1,p}/C^0$ 기본 package와 $C^1$ 강화 package로 분리해 닫는다.
