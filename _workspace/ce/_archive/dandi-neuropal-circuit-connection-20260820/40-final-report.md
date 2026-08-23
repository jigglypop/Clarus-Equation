# NeuroPAL 회로-계량 실자료 루프 최종 보고

Status: COMPLETE

Verdict: COMPLETE_NEGATIVE_RESULT

## 결론

DANDI 000541의 서로 겹치지 않는 실제 *C. elegans* development worm 아홉 개에서 A3, A4, A5를 순서대로 시험했다. 세 식은 모두 사전등록된 matched-control gate에서 실패했다. 각 단계의 confirmation 다섯 개체는 개발 실패 뒤 열지 않았다. 따라서 회로가 기능적 manifold를 끌어당긴다는 예측, 물리적 뇌 주름 변형, 인과 회로, AGI 기억 메커니즘을 지지하지 않는다.

성립한 것은 더 좁다. NeuroPAL identity·좌표·calcium을 같은 row로 결합하고, 뉴런별 calibration 역치와 edge별 관측 강도로 대칭 PSD graph Laplacian을 구성할 수 있다. 그러나 이 graph feature는 fixed geometry나 matched surrogate보다 held-out 미래 칼슘상태를 안정적으로 더 잘 예측하지 못했다.

## 기존 식과 새 식의 차이

원래 식 (21)--(26)은 parent responsibility, edge delay, STP와 synaptic magnitude를 요구한다. DANDI 000541에는 이 receipt가 없으므로 원래 인과식은 실행하지 않았다. 상관이나 lag로 parent receipt를 꾸며내지 않고 관측 대체식만 시험했다.

| 단계 | 바꾼 핵심 | 실제 반영된 이질성 | 개발 판정 |
|---|---|---|---|
| A3 | hard-threshold residual로 정적 $L_c,\Omega$ 구성 | 뉴런별 $\mu_i,s_i,\theta_i$; edge별 $c_{ij},\Omega_{ij}$ | 실패: mean $0.005010$, 여러 null보다 낮음 |
| A4 | $a_i=\sigma(z_i-2.5)$와 $m_{ij}(t)=\ell_{ij}^2e^{-h_{ij}(t)}$ | soft 뉴런 역치; edge별 $\widetilde\alpha_{ij}$; 순간 길이비 | 실패: $0/3$, mean $-0.034226$ |
| A5 | $L_t=L_0+\Delta L_t$로 발달 기하와 순간 변형 분리 | A4 이질성에 $\beta_s,\beta_\Delta\ge0$ joint fit 추가 | 실패: $1/3$, mean $-38.270912$ |

물리적 edge delay $d_{ij}$는 어떤 단계에도 들어갔다고 주장하지 않는다. 자료에 독립 receipt가 없기 때문이다. A3의 한 sample은 4 Hz에서 0.25초인 관측 lag이고, 17·31·47 sample은 shift-null이다.

## “공간이 얼마나 끌리는가”의 계산

A4와 A5의 구성에서는

$$
h_{ij}(t)=\operatorname{clip}
\left(\widetilde\alpha_{ij}a_i(t)a_j(t),0,4\right)
$$

이고

$$
\frac{\ell^{eff}_{ij}(t)}{\ell_{ij}}=e^{-h_{ij}(t)/2},
\qquad
\frac{w_{ij}(t)}{w_{0,ij}}=e^{h_{ij}(t)}.
$$

따라서 한 edge의 길이는 최대 $e^{-2}\simeq0.1353$까지 줄고 conductance는 최대 $e^4\simeq54.60$까지 늘도록 수학적으로 bounded했다. 실제 A4 development에서 mean $h$는 $0.00189$--$0.00244$였고 mean 길이비는 $0.99880$--$0.99910$이었다. 평균 수축은 약 $0.09$--$0.12\%$뿐이었다.

A5의 첫 worm에서는 deformation RMS가 construction에서 test로 $5.37$배 커져 held-out score가 $-114.821759$까지 무너졌다. 둘째는 $3.52$배, 셋째는 $0.17$배였다. 순간 변형 방향이 worm 내부의 시간 block 사이에서도 안정적이지 않았다.

## 오일러 회로와 주름

Euler circuit의 존재는 graph의 연결성과 차수 조건이다. edge 비용을 바꿔도 조합적 회로의 존재는 바뀌지 않고 경로 비용만 달라진다. A4--A5는 이 비용 변화를 계산한 weighted-graph 모형이다. 점이 매우 많다는 사실만으로 smooth Riemann manifold가 되지는 않는다. graph energy의 continuum convergence와 chart compatibility가 별도로 필요하다.

인간 피질의 해부학적 주름은 embedding $X:M\to\mathbb R^3$, 제1기본형 $g^{anat}_{ab}=\partial_aX\cdot\partial_bX$, 제2기본형 $b_{ab}=n\cdot\partial_a\partial_bX$의 문제다. 주요 주름은 태아기·주산기에 형성되고 출생 뒤와 청소년기에도 형태 지표가 변한다. 유전적 제약과 성장 역학을 함께 봐야 하므로 “전부 유전”이나 “청소년기에 처음 완성” 중 하나로 환원할 수 없다. 이번 worm 자료는 사람 피질 주름을 측정하지 않는다.

관련 근거는 [DANDI 000541](https://doi.org/10.48324/dandi.000541/0.241009.1457), [통합 whole-brain imaging corpus 논문](https://doi.org/10.1016/j.crmeth.2024.100964), [주산기 gyrification 연구](https://doi.org/10.1038/s42003-025-08155-z), [청소년기 gyrification 연구](https://doi.org/10.1371/journal.pone.0084914)에 있다.

## 차원·형식 지위

| 코어 인자 | 차원 벡터 $(M,L,T,\Theta)$ | 판정 | 정규화 |
|---|---|---|---|
| $z_i-2.5$ | $(0,0,0,0)$ | 무차원 | raw activity를 $s_i$로 나눔 |
| $h_{ij}=\widetilde\alpha_{ij}a_ia_j$ | $(0,0,0,0)$ | 무차원 | correlation strength를 positive median으로 나눔 |
| $\ell_{ij}=d_{ij}/\ell_{ref}$ | $(0,0,0,0)$ | 무차원 | worm별 median 6-NN 거리 |
| $L_0z/s_0$, $\Delta Lz/s_\Delta$ | $(0,0,0,0)$ | 무차원 | construction-only RMS |

무차원성은 차원 정합만 보장하며 물리적 정당성이나 예측 성공을 보장하지 않는다. $L_t\succeq0$은 양의 대칭 conductance에서 따르는 graph-Laplacian 정리다. $L_t$를 실제 cortical Riemann metric 또는 curvature와 동일시하는 다리는 `[미완성]`이다. A3--A5의 predictive claims는 `[경험식: 기각]`이다.

## 재현·봉인 영수증

- A3 result: `artifacts/a3-development-result.json`; confirmation 없음; script hash 일치.
- A4 result: `artifacts/a4-development-result.json`; confirmation 없음; script hash 일치.
- A5 result: `artifacts/a5-development-result.json`; confirmation 없음; script hash 일치.
- A5 failure diagnostic: `artifacts/diagnose_a5_development_shift.py`.
- focused self-tests: A3, A4, A5 모두 통과.
- dimensionless regression: `16 passed`.
- canonical paper: `docs/6_뇌/11_리만계량_라우팅_논문.md`의 5.7--5.8절.

