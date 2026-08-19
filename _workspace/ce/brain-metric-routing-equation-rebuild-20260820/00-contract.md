# 실제 신경 계량–라우팅 식 재정비 계약

Status: COMPLETE

PREDECESSOR: `docs/6_뇌/11_리만계량_라우팅_논문.md`

## 질문과 최대 주장

실제 신경 기록에서 좌표변환에 공변이고 무차원 선요소를 가지며, held-out 자료로 식별 가능한 “기능 계량”과 “라우팅”을 어떻게 최소한으로 정의할 것인가?

이 run이 낼 수 있는 최대 결과는 다음 세 가지다.

1. 미래의 사전 지정 출력 분포에 대한 조건부 Fisher pullback을 **과업·출력 상대적 operational metric**으로 정의한다.
2. source history가 target future의 조건부 log score를 얼마나 개선하는지를 **lagged conditional predictive routing**으로 정의한다.
3. 두 양은 같은 기록에서 함께 추정할 수 있지만 서로를 결정하거나 매개하지 않는다는 비식별 경계를 증명한다.

이 run은 실제 자료를 실행하지 않으며, 뇌의 보편 계량, 물리적 피질 곡률, 구조 연결의 생산 법칙, 인과 라우팅, 기억 또는 의식을 발견했다고 주장하지 않는다.

## 재시작 이유

| 이전 항목 | 보존되는 결과 | 이번에 제거하는 승격 |
|---|---|---|
| 공식 PFC 처리 cache | 고정된 3차원 selectivity chart에서 stage별 covariance/precision SPD 차이가 기술적으로 관측됨 | `C^{-1}`이 비선형 chart에서도 성립하는 뇌의 국소 Riemann tensor라는 주장 |
| G1/G2/G3D | SPD 요약을 계산하고 개입·예측 대조를 실행한 simulator 결과 | SPD 요약이 dynamics를 매개하거나 raw response보다 고유한 정보를 가진다는 주장 |
| C1 | predictor가 persistence보다 정확했으나 planner advantage는 `STOP` | 예측 정확도가 곧 제어 알고리즘 또는 메타인지라는 주장 |
| SCC/BrainRuntime memory | simulator 내부 구조와 zero-store recall 산출 | SCC가 생물학적 기억·의식의 실체라는 주장 |

`g=C^{-1}`은 삭제하지 않고 **고정 선형 selectivity chart의 상대 정밀도 경험식**으로 강등한다. 새 primary metric의 정의로 사용하지 않는다.

## 공간과 관측량

세션 `j`, 문맥 `c`, source 영역 `A`, target 영역 `B`를 고정한다.

$$
x^A_{\le t}\xrightarrow{\psi_j}z^A_t\in Z_j,
\qquad
o_{t+H}\in O,
\qquad
x^B_{t+\delta}\in X_B.
$$

- $x^A$: 실제 source neural observation.
- $z^A$: training fold에서만 적합한 source latent chart.
- $h_t$: stimulus, 행동, target/source history, global-state proxy, block 위치, drift 등 사전 지정 nuisance history.
- $o$: metric을 정의하는 미래 출력. primary protocol에서는 routing target $x^B$와 구별되는 행동·과업 출력을 사전 지정한다. 같은 target-neural 출력을 쓰는 분석은 독립 결합의 증거가 아닌 secondary sensitivity로 격리한다.
- $x^B$: routing이 예측하는 target-area future observation.
- $H,\ell,\delta$: 결과를 보기 전에 고정하는 future horizon, source history window, routing delay.

물리적 cortical-ribbon metric $h_{ab}=\partial_a r\cdot\partial_b r$와 아래의 latent functional metric은 정의역과 단위가 다르다. 명시적 map과 pullback 없이는 동일시하거나 빼지 않는다.

## 동결할 식

### 1. 조건부 output-Fisher metric

**[적용조건]** $p_j(o\mid z,h,c)$는 chart 내부에서 $z$에 대해 미분 가능하고 score의 second moment가 유한해야 한다. 기대값과 $z$ 미분의 교환이 허용되어야 하며, Gaussian 특수형에서는 $\mu,\Sigma\in C^1$이고 $\Sigma=\Sigma^T\succ0$여야 한다. 이 조건을 만족하지 않는 discrete/singular submodel은 별도 정의 없이는 아래 식에 넣지 않는다.

$$
G^{o\leftarrow A}_{j,c;ab}(z)
=
\mathbb E_{h\mid z,c}\mathbb E_{o\mid z,h,c}
\left[
\partial_a\log p_j(o\mid z,h,c)\,
\partial_b\log p_j(o\mid z,h,c)
\right].
\tag{M1}
$$

$G\succeq0$이며 output이 구별하지 못하는 방향이 있으면 pseudometric이다. SPD가 필요한 경우 calibration-only reference tensor를 사용한다.

$$
G^{(\lambda)}_{j,c}(z)=G_{j,c}(z)+\lambda G_{j,\mathrm{ref}}(z),
\qquad
G_{j,\mathrm{ref}}\succ0,
\quad \lambda>0.
\tag{M2}
$$

$G_{\rm ref}$는 같은 $(0,2)$ tensor law와 같은 단위를 가져야 하고 $\lambda$는 무차원이다. 모든 chart에 동일한 숫자 행렬 $I$를 더하는 규칙은 금지한다.

Gaussian 출력에서는 평균항과 covariance항을 모두 포함한다.

$$
\begin{aligned}
o\mid z,h,c&\sim\mathcal N(\mu,\Sigma),\\
I_{ab}(z,h,c)
&=(\partial_a\mu)^T\Sigma^{-1}(\partial_b\mu)
+\frac12\operatorname{tr}
\left(\Sigma^{-1}(\partial_a\Sigma)
\Sigma^{-1}(\partial_b\Sigma)\right),\\
G_{ab}(z,c)&=\mathbb E_{h\mid z,c}[I_{ab}(z,h,c)].
\end{aligned}
\tag{M3}
$$

mean-only 식은 $\partial_a\Sigma=0$을 독립적으로 고정했을 때만 허용한다.

### 2. 조건부 예측 라우팅

$$
\begin{aligned}
p_0&=p_j(x^B_{t+\delta}\mid\mathcal H_t^B,c),\\
p_1&=p_j(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c),\\
R_{j,c}^{A\to B}(\ell,\delta)
&=\frac1{N_{\rm test}}\sum_{t\in\rm test}
\left[\log p_1-\log p_0\right],
\end{aligned}
\tag{R1}
$$

여기서 $\mathcal H_t^B$는 target 자신의 과거와 $h_{\le t}$를 포함한다. $R$의 단위는 nat/sample이며, 허용되는 해석은 “추가 source history가 held-out target future 예측을 개선했다”까지다.

문맥 의존성은 route의 존재와 분리한다. 두 model 모두 context baseline $b_c$는 허용하고 source-history effect만 공유 $f$와 context-specific $f_c$로 다르게 둔다.

$$
\begin{aligned}
p_{\rm static}&=p(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c;\ b_c+f(z^A_{t-\ell:t})),\\
p_{\rm interaction}&=p(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c;\ b_c+f_c(z^A_{t-\ell:t})).
\end{aligned}
$$

$$
\Delta R^{A\to B}_{j,\mathrm{ctx}}
=\frac1{N_{\rm test}}\left[
\operatorname{ELPD}(p_{\rm interaction})
-\operatorname{ELPD}(p_{\rm static})
\right].
\tag{R2}
$$

### 3. 정직한 결합

$$
\boxed{
\mathcal B_{j,c}^{A\to B}(z)
=\left(G_{j,c}^{o\leftarrow A}(z),R_{j,c}^{A\to B}\right)
}
\tag{J1}
$$

문맥별 field summary는 모든 context에 공통인 calibration measure $q_j(z)$와 공통 chart/trivialization 또는 사전 지정 transport 아래

$$
\bar G_{j,c}
=\arg\min_{Q\succ0}
\mathbb E_{z\sim q_j}
d_{\rm AI}^2\!\left(Q,G^{(\lambda)}_{j,c}(z)\right)
$$

로 정의한다. 문맥 대비의 보고값은 단일 scalar가 아니라

$$
\Xi_j(c_0,c_1;A\to B)
=\left(
d_{\rm AI}(\bar G_{j,c_0},\bar G_{j,c_1}),
\Delta R^{A\to B}_{j,\rm ctx}
\right)
\tag{J2}
$$

라는 ordered pair다. 두 성분을 더하거나 곱하는 가중치는 별도 공리 없이는 정의하지 않는다.

## 필수 수학 게이트

1. **좌표 공변성:** $z'=\phi(z)$, $J_\phi=\partial z'/\partial z$에서 $G'=J_\phi^{-T}GJ_\phi^{-1}$과 $dz'^TG'dz'=dz^TGdz$를 증명한다.
2. **무차원성:** $[G_{ab}]=[z_a]^{-1}[z_b]^{-1}$, $ds^2=dz^TGdz$는 무차원, log-score 차이와 AIRM도 무차원이어야 한다.
3. **SPD 조건:** Fisher가 PSD일 수 있음을 숨기지 않고, reference tensor가 calibration-only·공변·SPD임을 요구한다.
4. **비교 가능성:** AIRM/Karcher mean은 같은 tangent space, 공통 chart 또는 사전 지정 transport가 있을 때만 사용한다. 서로 다른 animal/session chart의 raw SPD 평균을 금지한다.
5. **비식별 반례:** 같은 $G$와 다른 $R$, 같은 $R$과 다른 $G$, common-input이 만드는 $R>0$, 관측적으로 동치인 mediated/non-mediated 구조를 각각 제시한다.
6. **곡률 경계:** 단일 또는 상수 SPD 행렬로 Christoffel, Riemann curvature, geodesic dynamics를 주장하지 않는다.

## 필수 경험적 대조 계약

- metric: shared-context, gain-only, diagonal, reference-dominated, nuisance-only 대조.
- routing: target-history-only, source time/block shift, reverse $B\to A$, global-state augmented 대조. negative lag/time reverse는 시간 대칭·leakage·공통구동 진단이며 단독 strict zero null로 쓰지 않는다.
- leakage: chart, normalizer, nuisance model, hyperparameter는 training block에서만 적합한다.
- 표본 단위: trial/bin을 독립 동물처럼 세지 않고 session/animal 계층을 유지한다.
- 좌표 감사: 사전 지정 orthogonal, diagonal rescale, bounded-condition affine transform에서 tensor/scalar 결과를 재현한다.

## 금지된 승격

- $C^{-1}$ 또는 $G$를 뇌의 유일하거나 물리적인 계량이라고 부르기.
- $R>0$을 causal/structural/synaptic routing이라고 부르기.
- $G\Rightarrow R$, $R\Rightarrow G$, $W\to G\to x$를 관측자료로 매개 식별하기.
- 한 점의 SPD, constant metric, 비선형 좌표의 Christoffel만으로 곡률을 주장하기.
- simulator 결과를 실제 기억, SCC, metacognition 또는 의식으로 승격하기.

## 완료 조건

- `11-math.md`: (M1)--(J2)의 유도, 좌표/단위 감사, 완전 반례.
- `12-routes.md`: 실제자료 route, 개입 route, 구조-longitudinal route와 각 claim ceiling.
- `artifacts/dimensionless-audit.md`: 무차원성의 식별·회귀 감사를 기록한다.
- `20-audit.md`: 독립 수학·형식지위 감사에서 P0가 없어야 한다.
- 감사 PASS 후에만 canonical brain paper를 갱신한다. 실제자료 분석과 새 seed는 그 다음 별도 run으로 연다.
