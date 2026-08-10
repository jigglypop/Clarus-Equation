# 05c. Pathspace Closure Checklist

## 0. 목표

05장은 CE 경로공간을 PreEq 후보공간으로 읽었다.

$$
A:=\Gamma=\mathcal P_I
$$

하지만 이 대응만으로 02a/02c의 농축 정리를 바로 적용할 수는 없다. 이 문서는 CE 경로공간 bridge를 조건부 정리로 내리기 위해 필요한 수학 가정을 고정한다.

핵심 질문:

> 어떤 조건을 채우면 \(\mu_\beta(d\gamma)\propto e^{-\beta E_{\mathrm{fold}}(\gamma)}\mu_0(d\gamma)\)가 선택 경로 집합으로 농축한다고 말할 수 있는가?

실제 CE 문서의 \(\mathcal P_I\), \(\mu_{\mathrm{ref}}\), \(W/F\), \(\mathcal P_{\mathrm{ns}}\)와 대조한 결과는 [05d_pathspace_audit.md](05d_pathspace_audit.md)에 둔다. 여기서는 일반 정리 적용 조건만 유지한다.

형식 출처:

| 항목 | 판정 |
|---|---|
| finite path approximation | `[정리]` |
| compact pathspace 농축 | `[정리]` |
| Polish/non-compact pathspace 농축 | `[정리]` |
| 실제 CE \(\Gamma\), \(S[\gamma]\), \(\Phi\) 식별 | `[미완성]` |

## 1. 최소 데이터

CE pathspace 농축에 필요한 데이터는 다음 여섯 개다.

| 기호 | 의미 | 필요 조건 |
|---|---|---|
| \(\Gamma\) | 경로 후보공간 | measurable/topological space |
| \(\mu_0\) | 초기 경로 prior | Borel probability |
| \(E_{\mathrm{fold}}\) | 접힘/억압 에너지 | measurable, preferably l.s.c. |
| \(\beta\) | readout strength | \(\beta\to\infty\) 또는 관측 scale |
| \(\Gamma_*\) | manifest 경로 집합 | argmin set |
| \(\Gamma_{\mathrm{ns}}\) | 비선택 경로 집합 | measurable complement |

이 중 하나라도 비어 있으면 05장의 물리 사상은 `[미완성]`이다.

## 2. 세 가지 닫힘 수준

### Level F: finite path approximation

경로 후보가 유한집합이면

$$
\Gamma_N=\{\gamma_1,\dots,\gamma_N\}
$$

01장의 정리를 그대로 쓴다.

$$
\mu_{\beta,N}(\gamma_i)
=
\frac{e^{-\beta E_i}\mu_0(\gamma_i)}
\sum_j e^{-\beta E_j}\mu_0(\gamma_j)
$$

결론:

> \(\beta\to\infty\)에서 최소 에너지 경로 집합으로 농축한다.

출처: `[정리]`

### Level C: compact pathspace

\(\Gamma\)가 compact metric space이고 \(E_{\mathrm{fold}}\)가 continuous라면 02장의 compact 농축 정리를 쓴다.

필요 조건:

| 조건 | 의미 |
|---|---|
| compactness | escape 없음 |
| continuity | 최소집합 바깥 gap 확보 |
| \(\mu_0\) Borel probability | Gibbs 분모 정의 |
| \(A_*=\arg\min_{\operatorname{supp}\mu_0}E\ne\varnothing\) | manifest set 존재 |

결론:

$$
\mu_\beta(U)\to1
$$

for every open \(U\supset\Gamma_*\).

출처: `[정리]`

### Level P: Polish/non-compact pathspace

\(\Gamma\)가 Polish space이고 \(E_{\mathrm{fold}}\)가 lower semicontinuous이며 coercive/equicoercive 조건을 갖추면 02a/02c류 정리를 목표로 삼는다.

필요 조건:

| 조건 | 의미 |
|---|---|
| Polish space | 약수렴과 tightness 도구 사용 |
| \(E\) l.s.c. | minimizer 안정성 |
| coercive/tightness | 경로가 무한대로 escape하지 않음 |
| positive-mass recovery | 최소점 근방 Gibbs 분모 하한 |
| unique minimizer 또는 최소집합 | Dirac 또는 set-concentration 구분 |

결론 후보:

$$
\mu_\beta\Rightarrow\delta_{\gamma_*}
$$

또는

$$
\mu_\beta(U)\to1
\quad(U\supset\Gamma_*).
$$

출처: `[정리]`

## 3. Energy 선택 후보

경로 에너지는 자동으로 정해지지 않는다. 후보는 다음과 같다.

| 후보 | 식 | 판정 |
|---|---|---|
| action energy | \(E(\gamma)=S_E[\gamma]\) | `[미완성]`; 채택 시 `[공리: 물리 사상]` |
| suppress action | \(E(\gamma)=S_{\mathrm{supp}}[\gamma]\) | `[미완성]`; 채택 시 `[공리: 물리 사상]` |
| fold functional | \(E(\gamma)=\Phi[\gamma]\) | `[미완성]`; 채택 시 `[공리: 물리 사상]` |
| hybrid | \(E(\gamma)=aS_E[\gamma]+bS_{\mathrm{supp}}[\gamma]\) | `[미완성]`; 계수는 `[공리: 모델 선택]` |

수학 정리는 \(E\)가 주어졌을 때만 작동한다. \(E\)의 물리적 정체성은 별도 문제다.

## 4. Coercivity / escape 조건

non-compact pathspace에서 가장 위험한 실패는 질량 escape다.

필요한 형태:

$$
E_{\mathrm{fold}}(\gamma)\to\infty
\quad
\text{as }\gamma\text{ escapes compact sets}.
$$

또는 에너지열 \(E_n\)이면 equicoercivity:

$$
\{E_n\le c\}
\subset K_c
$$

가 충분히 큰 \(n\)에 대해 같은 compact \(K_c\) 안에 들어야 한다.

이 조건이 없으면 Gibbs 분포가 최소경로가 아니라 무한한 경로공간 바깥으로 질량을 흘릴 수 있다.

## 5. Positive-mass recovery

Gamma 수렴만으로는 Gibbs 분모가 닫히지 않는다. 따라서 최소 경로 \(\gamma_*\) 근방에 초기 prior가 충분한 질량을 줘야 한다.

고정 recovery:

$$
\forall U\ni\gamma_*,\ \forall\eta>0,\ 
\exists V\subset U:
\mu_0(V)>0,\quad
\sup_VE_n\le m+\eta.
$$

moving recovery:

$$
\beta_n(\delta-\eta_n)+\log\mu_0(V_n)\to+\infty.
$$

이 조건이 없으면 최소경로가 형식적으로 있어도 \(\mu_0\)가 그 주변을 충분히 보지 못해 농축 결론이 깨진다.

## 6. 선택집합과 잔류집합

최소집합:

$$
\Gamma_*
=
\operatorname*{argmin}_{\gamma\in\operatorname{supp}\mu_0}
E_{\mathrm{fold}}(\gamma).
$$

비선택집합:

$$
\Gamma_{\mathrm{ns}}
=
\Gamma\setminus\Gamma_*.
$$

필요:

| 항목 | 조건 |
|---|---|
| \(\Gamma_*\) | measurable 또는 closed |
| \(\Gamma_{\mathrm{ns}}\) | measurable |
| raw residual | \(\nu_{\mathrm{ns},\beta}=\mathbf1_{\Gamma_{\mathrm{ns}}}\mu_\beta\) finite |
| \(\phi\) pushforward | \(K_\phi\in L^1(\nu_{\mathrm{ns},\beta})\) |

## 7. 적용 가능한 정리 문장

아래 문장이 05장의 목표 형태다.

**정리 후보 7.1**  
\(\Gamma\)가 Polish space이고 \(\mu_0\in\mathcal P(\Gamma)\)라고 하자. \(E:\Gamma\to[0,\infty]\)가 l.s.c.이고 coercive이며, \(\operatorname*{argmin}_{\operatorname{supp}\mu_0}E=\{\gamma_*\}\)라고 하자. 또한 \(\gamma_*\in\operatorname{supp}\mu_0\)이고 모든 근방이 positive mass recovery를 만족한다고 하자. 그러면

$$
\mu_\beta(d\gamma)
=
\frac{e^{-\beta E(\gamma)}}{Z_\beta}\mu_0(d\gamma)
\Rightarrow
\delta_{\gamma_*}.
$$

이 정리 자체는 표준 Gibbs 농축 정리의 pathspace 버전이다. 남은 것은 CE의 실제 \(\Gamma\), \(\mu_0\), \(E_{\mathrm{fold}}\)가 가정을 만족하는지다.

## 8. 실패 조건

| 실패 | 결과 |
|---|---|
| \(\Gamma\) measurable 구조 없음 | 확률측도 자체가 정의되지 않음 |
| \(\mu_0\) support 불명 | manifest set 정의 불가 |
| \(E_{\mathrm{fold}}\) l.s.c. 실패 | minimizer 안정성 약화 |
| coercivity 실패 | mass escape 가능 |
| recovery mass 실패 | Gibbs 분모 하한 실패 |
| minimizer 다중 | Dirac가 아니라 set-concentration만 가능 |
| \(K_\phi\) integrability 실패 | 잔류장 readout 불가 |

## 9. 결론

CE bridge를 수학적으로 닫는 다음 문은 이것이다.

$$
\boxed{
\Gamma,\ \mu_0,\ E_{\mathrm{fold}}
\text{가 02a/02c의 가정을 만족하는가?}
}
$$

그 답이 예이면 pathspace 선택/비선택 농축은 가정이 명시된 `[정리]`가 된다. 아니면 05장의 물리 사상은 `[미완성]`으로 남는다.
