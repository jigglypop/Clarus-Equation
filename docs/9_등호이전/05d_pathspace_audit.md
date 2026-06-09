# 05d. CE Pathspace Audit

## 0. 목표

[05c_pathspace_closure_checklist.md](05c_pathspace_closure_checklist.md)는 CE 경로공간을 PreEq 농축 정리에 올리기 위한 일반 조건표다. 이 문서는 그 조건표를 실제 CE 문서와 대조한다.

핵심 결론:

> CE 문서에는 이미 \(\mathcal P_I\), 참조측도, CE 가중치 \(W\), 선택함수 \(F\), 비선택 경로공간, 잔류장 커널이 들어 있다. 따라서 확률공간 구성과 비선택 pushforward는 조건부로 닫힌다. 남은 병목은 \(W/F\)가 좋은 rate function인지, 즉 lower semicontinuity, compact sublevel/coercivity, recovery mass를 만족하는지다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| \((\mathcal P_I,\mathcal B)\) | `Exact under CE assumptions` | CE 문서가 \(C^1\) 경로공간, Polish/Borel 구조를 가정 |
| CE 확률측도 \(\mu_{\mathrm{CE}}\) | `Exact under CE assumptions` | \(W\) 가측성과 \(\int e^{-W}d\mu_{\mathrm{ref}}<\infty\)로 정규화 가능 |
| 선택함수 \(F\)와 \(W\)의 동일성 | `Exact under CE assumptions` | \(F=W+c\)이면 argmin과 Gibbs 가중치가 상수배만 다름 |
| \(\Gamma_{\mathrm{sel}}\), \(\mathcal P_{\mathrm{ns}}\)의 가측성 | `Exact under assumptions` | \(F\) 가측/l.s.c. 정의를 쓰면 닫힘 |
| 비선택 잔류장 \(\phi(x)\) | `Exact under kernel assumptions` | \(K(x,\cdot)\in L^1(\mu_{\mathrm{ns}})\)이면 적분 정의 가능 |
| \(\beta\to\infty\) 농축 | `Still bridge` | l.s.c., coercivity/good rate, positive-mass recovery가 아직 실제 \(W\)에서 증명되지 않음 |

## 1. 기존 CE 문서에서 이미 채워진 것

주요 출처는 [../참조/epsilon_제1원리_유도.md](../참조/epsilon_제1원리_유도.md)의 12-13장이다.

| 05c 필요 데이터 | 기존 CE 대응 | 상태 |
|---|---|---|
| 후보공간 \(\Gamma\) | \(\Gamma:=\mathcal P_I\), finite interval \(I\)의 허용 세계선/경로공간 | 있음 |
| 위상/가측구조 | \(C^1\) 수렴에 해당하는 거리 \(d_I\), Borel \(\sigma\)-algebra | 있음, Polish는 가정 |
| 참조측도 | \(\mu_{\mathrm{ref}}\) 또는 기존 표기 \(\mu_0\), \(\sigma\)-finite reference measure | 있음 |
| CE 가중치 | \(W[\gamma]=S_E[\gamma]/\hbar+S_{\mathrm{supp}}[\gamma]\) | 있음 |
| CE 확률측도 | \(d\mu_{\mathrm{CE}}=Z_W^{-1}e^{-W}d\mu_{\mathrm{ref}}\) | 있음 |
| 선택함수 | \(F:\mathcal P_I\to\mathbb R\cup\{+\infty\}\), \(F=W+c\) | 있음 |
| 선택 경로 | \(\Gamma_{\mathrm{sel}}=\operatorname{argmin}F\) 또는 near-minimizer 극한 | 있음 |
| 비선택 경로공간 | \(\mathcal P_{\mathrm{ns}}=\mathcal P_I\setminus\Gamma_{\mathrm{sel}}\) | 있음 |
| 잔류장 커널 | \(K:M\times\mathcal P_{\mathrm{ns}}\to\mathbb R\) | 있음 |
| 잔류장 | 기존 문서의 \(S(x)=\int_{\mathcal P_{\mathrm{ns}}}K(x,\gamma)d\mu_{\mathrm{ns}}\) | 있음, 09 기준으로 \(\phi(x)\)로 읽는 것이 안전 |

따라서 05c에서 비어 있던 \(\Gamma\), \(\mu_0\), \(E_{\mathrm{fold}}\) 자리는 다음처럼 채울 수 있다.

$$
\Gamma=\mathcal P_I.
$$

다만 측도 표기는 충돌을 피하기 위해 바꾸는 편이 낫다.

$$
\mu_{\mathrm{ref}}
\quad\text{CE reference measure}
$$

$$
d\mu_{\mathrm{CE}}(\gamma)
=
\frac{e^{-W[\gamma]}}{Z_W}\,d\mu_{\mathrm{ref}}(\gamma).
$$

PreEq의 기본 prior는 두 가지 선택지가 있다.

| 선택 | 정의 | 해석 |
|---|---|---|
| raw reference prior | \(\mu_0:=\mu_{\mathrm{ref}}\) | CE weight \(W\)를 PreEq energy로 한 번만 적용 |
| CE base prior | \(\mu_0:=\mu_{\mathrm{CE}}\) | CE가 이미 만든 확률공간 위에 추가 readout energy를 적용 |

이 문서에서는 중복 가중치를 피하려고 다음 표기를 권장한다.

$$
\mu_{\mathrm{base}}\in\{\mu_{\mathrm{ref}},\mu_{\mathrm{CE}}\}.
$$

추가 PreEq/readout 농축은

$$
d\mu_\beta(\gamma)
=
\frac{e^{-\beta E_{\mathrm{fold}}(\gamma)}}{Z_\beta}\,d\mu_{\mathrm{base}}(\gamma)
$$

로 쓴다.

## 2. 바로 닫히는 정리

### 정리 2.1: CE 가중 확률공간

가정:

1. \((\mathcal P_I,\mathcal B(\mathcal P_I))\)가 measurable space다.
2. \(\mu_{\mathrm{ref}}\)가 \(\sigma\)-finite measure다.
3. \(W:\mathcal P_I\to\mathbb R\cup\{+\infty\}\)가 가측이다.
4. \(0<Z_W:=\int_{\mathcal P_I}e^{-W[\gamma]}d\mu_{\mathrm{ref}}(\gamma)<\infty\).

그러면

$$
\mu_{\mathrm{CE}}(A)
=
\frac1{Z_W}\int_Ae^{-W[\gamma]}d\mu_{\mathrm{ref}}(\gamma)
$$

는 \((\mathcal P_I,\mathcal B(\mathcal P_I))\) 위의 확률측도다.

증명:

\(e^{-W}\)는 \(W\)의 가측성 때문에 가측이고 음이 아니다. 따라서
\(\nu(A)=\int_Ae^{-W}d\mu_{\mathrm{ref}}\)는 measure다. \(Z_W=\nu(\mathcal P_I)\)가 양의 유한값이므로 \(\mu_{\mathrm{CE}}=\nu/Z_W\)는 total mass 1인 measure다. 끝.

### 정리 2.2: \(F=W+c\)의 선택 동치

가정:

1. \(F[\gamma]=W[\gamma]+c\) for constant \(c\).
2. \(\operatorname{argmin}W\) 또는 \(\operatorname{argmin}F\)가 정의된다.

그러면

$$
\operatorname{argmin}F=\operatorname{argmin}W
$$

이고

$$
\frac{e^{-F[\gamma]}}{\int e^{-F}d\mu_{\mathrm{ref}}}
=
\frac{e^{-W[\gamma]}}{\int e^{-W}d\mu_{\mathrm{ref}}}.
$$

증명:

상수 \(c\)를 더해도 모든 경로 사이의 순서가 변하지 않는다. 또한 \(e^{-F}=e^{-c}e^{-W}\)이고 정규화상수에도 같은 상수 \(e^{-c}\)가 곱해지므로 소거된다. 끝.

### 정리 2.3: near-minimizer 선택집합의 가측성

가정:

1. \(F:\mathcal P_I\to\mathbb R\cup\{+\infty\}\)가 가측이다.
2. \(F_{\min}=\inf_{\gamma\in\mathcal P_I}F[\gamma]\)가 실수 또는 확장실수로 정의된다.

그러면 모든 \(\delta>0\)에 대해

$$
\Gamma_{\mathrm{sel}}^{(\delta)}
=
\{\gamma:F[\gamma]\le F_{\min}+\delta\}
$$

는 \(\mathcal B(\mathcal P_I)\)-measurable이다.

추가로 \(F\)가 lower semicontinuous이고 \(\mathcal P_I\)가 topological space이면 \(\Gamma_{\mathrm{sel}}^{(\delta)}\)는 closed sublevel set이다.

증명:

가측함수의 sublevel set은 가측이다. lower semicontinuous 함수의 sublevel set은 닫혀 있다. 끝.

### 정리 2.4: 비선택 경로공간

가정:

1. \(\Gamma_{\mathrm{sel}}\in\mathcal B(\mathcal P_I)\).
2. \(\mathcal P_{\mathrm{ns}}=\mathcal P_I\setminus\Gamma_{\mathrm{sel}}\).

그러면 \(\mathcal P_{\mathrm{ns}}\)는 measurable subset이고

$$
\mathcal B(\mathcal P_{\mathrm{ns}})
=
\{A\cap\mathcal P_{\mathrm{ns}}:A\in\mathcal B(\mathcal P_I)\}
$$

가 자연스러운 부분 Borel 구조다. 또한

$$
\mu_{\mathrm{ns}}(B)=\mu_{\mathrm{base}}(B)
$$

로 제한하면 finite measure가 된다. 조건부 잔류분포를 원하면 \(\mu_{\mathrm{base}}(\mathcal P_{\mathrm{ns}})>0\)일 때

$$
\widehat\mu_{\mathrm{ns}}(B)
=
\frac{\mu_{\mathrm{base}}(B\cap\mathcal P_{\mathrm{ns}})}
{\mu_{\mathrm{base}}(\mathcal P_{\mathrm{ns}})}
$$

로 정규화한다.

증명:

가측집합의 여집합은 가측이고, 부분공간 \(\sigma\)-algebra는 측도의 제한을 보존한다. 전체질량은 \(\mu_{\mathrm{base}}(\mathcal P_{\mathrm{ns}})\le1\)이므로 finite다. 끝.

### 정리 2.5: 잔류장 pushforward/readout

가정:

1. \(\mu_{\mathrm{ns}}\)가 \(\mathcal P_{\mathrm{ns}}\) 위의 finite measure다.
2. 각 \(x\in M\)에 대해 \(K(x,\cdot)\)가 measurable이다.
3. 각 \(x\in M\)에 대해 \(K(x,\cdot)\in L^1(\mu_{\mathrm{ns}})\).

그러면

$$
\phi(x)
=
\int_{\mathcal P_{\mathrm{ns}}}K(x,\gamma)d\mu_{\mathrm{ns}}(\gamma)
$$

는 well-defined real-valued readout이다.

증명:

가측성과 \(L^1\) 조건으로 Lebesgue integral이 존재하고 절대값 적분이 유한하므로 값이 유한하다. 끝.

## 3. 농축 정리로 승격되는 조건

CE 선택을 실제 PreEq 농축으로 닫으려면 \(E_{\mathrm{fold}}\)를 정해야 한다. 가장 자연스러운 선택은 다음이다.

$$
E_{\mathrm{fold}}(\gamma)
=
F[\gamma]-F_{\min}
$$

또는 \(F=W+c\)를 쓰면

$$
E_{\mathrm{fold}}(\gamma)
=
W[\gamma]-\inf_{\eta\in\mathcal P_I}W[\eta].
$$

그러면 \(E_{\mathrm{fold}}\ge0\)이고

$$
\Gamma_*=\operatorname{argmin}E_{\mathrm{fold}}
=
\operatorname{argmin}F
=
\operatorname{argmin}W.
$$

### 정리 3.1: compact pathspace 농축

가정:

1. \(\mathcal P_I\)가 compact metric space다.
2. \(E_{\mathrm{fold}}\)가 continuous다.
3. \(\mu_{\mathrm{base}}\in\mathcal P(\mathcal P_I)\)다.
4. \(\Gamma_*=\operatorname{argmin}_{\operatorname{supp}\mu_{\mathrm{base}}}E_{\mathrm{fold}}\ne\varnothing\).

그러면 모든 open \(U\supset\Gamma_*\)에 대해

$$
\mu_\beta(U)\to1
\quad(\beta\to\infty).
$$

증명:

[02_연속공간과측도.md](02_연속공간과측도.md)의 compact Gibbs 농축 정리를 \(\Gamma=\mathcal P_I\), \(E=E_{\mathrm{fold}}\)에 적용한다. compactness와 continuity가 minimizer 존재와 outside-\(U\) energy gap을 보장하고, Gibbs factor가 gap 밖 질량을 지수적으로 누른다. 끝.

### 정리 3.2: Polish/noncompact pathspace 농축

가정:

1. \(\mathcal P_I\)가 Polish space다.
2. \(E_{\mathrm{fold}}\)가 l.s.c.이고 coercive/good rate function이다.
3. \(\mu_{\mathrm{base}}\)가 positive-mass recovery를 만족한다.
4. \(\Gamma_*\ne\varnothing\).

그러면 모든 open \(U\supset\Gamma_*\)에 대해

$$
\mu_\beta(U)\to1.
$$

특히 \(\Gamma_*=\{\gamma_*\}\)이면

$$
\mu_\beta\Rightarrow\delta_{\gamma_*}.
$$

증명:

[02a_noncompact_Gamma.md](02a_noncompact_Gamma.md)와 [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)의 noncompact/Gamma 농축 정리를 \(\Gamma=\mathcal P_I\), \(E=E_{\mathrm{fold}}\)에 적용한다. coercivity가 mass escape를 막고, l.s.c.가 sublevel 안정성을 주며, recovery mass가 minimizer 근방의 분모 기여를 보장한다. 끝.

## 4. 발견된 표기/논리 충돌

### 4.1 \(\mu_0\) 충돌

기존 CE 문서의 \(\mu_0\)는 \(\sigma\)-finite reference measure다. PreEq 문서의 \(\mu_0\)는 보통 probability prior다. 둘을 같은 기호로 쓰면

$$
e^{-\beta E_{\mathrm{fold}}}\,e^{-W}\,d\mu_0
$$

처럼 CE 가중치가 두 번 들어갔는지 한 번 들어갔는지 모호해진다.

권장 표기:

| 기호 | 의미 |
|---|---|
| \(\mu_{\mathrm{ref}}\) | CE reference measure |
| \(\mu_{\mathrm{CE}}\) | \(e^{-W}\)로 정규화한 CE probability |
| \(\mu_{\mathrm{base}}\) | PreEq reweighting을 시작할 base prior |
| \(\mu_\beta\) | \(E_{\mathrm{fold}}\)로 다시 reweight한 posterior |

### 4.2 \(\Gamma_{\mathrm{sel}}=\Gamma(\Omega)\) 문제

기존 CE 문서에는 random worldline \(\Gamma:\Omega\to\mathcal P_I\)의 image를 \(\Gamma_{\mathrm{sel}}\)로 보는 표현이 있다. 이 표현은 물리적 직관에는 좋지만 수학적으로는 조심해야 한다.

문제:

1. measurable map의 image는 일반적으로 Borel일 필요가 없다.
2. random realization의 image와 variational minimizer set은 같은 개념이 아니다.

안전한 정의:

$$
\Gamma_{\mathrm{sel}}:=\operatorname{argmin}F
$$

또는

$$
\Gamma_{\mathrm{sel}}:=\bigcap_{n=1}^{\infty}
\{\gamma:F[\gamma]\le F_{\min}+1/n\}.
$$

이 정의는 \(F\)의 가측성/l.s.c. 조건 아래에서 \(\mathcal P_{\mathrm{ns}}\)를 안정적으로 만든다.

### 4.3 \(S(x)\), \(\Phi\), \(\phi\) 충돌

기존 CE 문서 12.6은 비선택 경로공간에서 나온 장을 \(S(x)\)로 쓴다. 다른 문서에서는 \(\Phi\)가 물리적 Clarus field, path Hessian, suppressive degree 등으로 쓰인다.

09의 용어 규칙에 맞추면 이 문서군에서는 다음이 안전하다.

| 기호 | 권장 의미 |
|---|---|
| \(\Phi\) | 큰 물리/형이상학적 Clarus field 또는 CE bridge 대상 |
| \(\phi\) | 비선택 잔류측도의 구체적 readout |
| \(K_\phi\) | \(\phi\)를 만드는 커널 |
| \(S_E\) | Euclidean action |
| \(S_{\mathrm{supp}}\) | suppression cost |

따라서 12.6의 \(S(x)\)는 9_등호이전 문맥에서는

$$
\phi(x)=\int K_\phi(x,\gamma)d\mu_{\mathrm{ns}}(\gamma)
$$

로 읽는 것이 좋다.

## 5. 아직 증명되지 않은 것

다음은 지금 문서만으로는 닫히지 않는다.

| 미해결 | 왜 필요한가 | 닫는 방법 |
|---|---|---|
| \(W\) 또는 \(F\)의 l.s.c. | minimizer와 closed sublevel 안정성 | \(S_E\), \(S_{\mathrm{supp}}\)의 해석학적 정의 필요 |
| compact sublevel/coercivity | pathspace 바깥으로 질량이 새는 것을 방지 | 경로 길이, 작용, boundary condition으로 tightness 증명 |
| \(0<Z_W<\infty\)의 실제 증명 | CE 확률측도 존재 | reference measure와 action bound 필요 |
| positive-mass recovery | Gibbs 분모가 minimizer 근방을 실제로 본다는 보장 | \(\mu_{\mathrm{base}}\) support 조건 필요 |
| \(\Gamma_{\mathrm{sel}}\)의 물리적 유일성 | set concentration인지 Dirac인지 결정 | degeneracy/symmetry breaking 조건 필요 |
| \(K_\phi\)의 물리 커널 선택 | 잔류장이 무엇을 의미하는지 결정 | endpoint/occupation/curvature 중 실험 또는 물리 원리로 선택 |
| large-deviation principle | CE가 실제로 \(F\) minimizer로 수렴하는 해석 | \(F\)를 good rate function으로 세우고 LDP를 증명 |

## 6. 다음에 닫을 정리 후보

가장 값이 큰 다음 정리는 이것이다.

**정리 후보 6.1: CE good-rate theorem**

이 후보는 [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)에서 조건부 정리로 닫혔다. 아래 문장은 05d 시점의 목표 형태로 남겨 둔다.

가정:

1. \(I=[t_i,t_f]\)가 compact interval이다.
2. target manifold \(M\)이 complete Riemannian manifold이고 endpoint/boundary condition이 고정된다.
3. \(\mathcal P_I\subset C^1(I,M)\)는 closed admissible path class다.
4. \(S_E\)가 coercive하고 l.s.c.다.
5. \(S_{\mathrm{supp}}\ge0\)가 l.s.c.다.
6. \(W=S_E/\hbar+S_{\mathrm{supp}}\)의 sublevel set이 compact하다.
7. \(\mu_{\mathrm{base}}\)가 모든 minimizer 근방에 positive mass를 준다.

결론:

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta(W[\gamma]-W_{\min})}}{Z_\beta}
\,d\mu_{\mathrm{base}}(\gamma)
$$

는

$$
\operatorname{argmin}W
$$

로 농축한다.

이 정리가 닫히면 05 CE bridge의 핵심 선택 문장은 `Bridge`에서 `Exact under assumptions`로 승격된다.

## 7. 이 이론이 나중에 주는 것

이 축이 닫히면 나중에 얻는 것은 세 가지다.

1. **선택의 수학적 엔진**: 등호 이후의 manifest state를 임의 선언이 아니라 prior, energy, temperature/readout의 극한으로 표현한다.
2. **비선택 정보의 보존 법칙**: 선택되지 않은 후보들이 그냥 사라지는 것이 아니라 finite measure와 pushforward readout \(\phi\)로 남는다.
3. **실험 가능한 AGI/물리 bridge**: \(\phi\)를 재주입했을 때 성능이나 안정성이 올라가는지 ablation으로 물을 수 있다. 실패하면 bridge가 틀린 것이고, 성공하면 어떤 커널 \(K_\phi\)가 유효한지 좁힐 수 있다.

따라서 현재 이론의 가장 강한 형태는 다음 한 줄이다.

$$
\boxed{
\text{ambiguity prior}
\xrightarrow{\;E_{\mathrm{fold}},\beta\;}
\text{manifest concentration}
\quad+\quad
\text{nonselected residual readout}
}
$$

아직 철학이 아니라 수학으로 남겨야 하는 병목은 \(W/F\)가 실제 CE 경로공간에서 good rate function인지다.
