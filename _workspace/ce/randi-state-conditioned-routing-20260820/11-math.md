# 수학·식별 레인: 상태-조건부 유효 라우팅

Status: COMPLETE

## 정의

**[정의]** 동물·세션 $i$의 event $e$에서 $H_{ie}$는 event order, 이전 source·dose,
지난 자극 이후 시간과 session/block을 포함한 자극 전 history다. $Z_{ie}$는 정본
source identity, $q$는 dose·timing·geometry 정책, $X_{ie}^{\mathrm{pre}}$는 onset 전
공통 timestamp window의 population state다. 고정 target $B$의 outcome은 사전 고정
window 함수 $\psi$로 정의한다.

$$
Y_{ie}^B=\psi\!\left(\operatorname{trace}_{ie}^B
[t_0+\Delta_1,t_0+\Delta_2]\right).
$$

**[정의]** $g_{ie}=g(X_{ie}^{\mathrm{pre}})$는 global gain 요약이고,
$r_{ie}=r(X_{ie}^{\mathrm{pre}})$는 $g$를 제외한 population configuration이다.
모든 encoder는 train fold의 pre-state만 사용해 동결한다.

## 예측 모델과 필수 대조

**[정의]** 공통 전처리, animal/session split, likelihood와 용량 규칙 아래 다음 네
모델을 비교한다.

$$
M_0:p_0(Y^B\mid Z,q,H),
$$

$$
M_1:p_1(Y^B\mid Z,q,H,g),
$$

$$
M_{2,\mathrm{add}}:p_{2,\mathrm{add}}(Y^B\mid Z,q,H,g,r),
$$

$$
M_{2,\mathrm{int}}:p_{2,\mathrm{int}}
(Y^B\mid Z,q,H,g,r,Z\mathbin{\otimes}r).
$$

**[산출]** 상태-조건부 라우팅에 필요한 primary contrast는 held-out proper log score의
다음 차이다.

$$
\Delta_{\mathrm{config}}
=\ell(M_{2,\mathrm{int}})-\ell(M_{2,\mathrm{add}}).
$$

$M_{2,\mathrm{add}}$는 pre-state의 일반적 예측력을 흡수한다. 따라서 단순한
$M_2>M_1>M_0$만으로는 source별 라우팅을 식별하지 못한다. time bin, neuron pair와
반복 event는 독립 표본이 아니므로 score와 불확도는 animal/session 단위로 집계한다.

## 좁은 인과 estimand

**[정의]** 같은 배정 층의 active control source 집합 $C$와 사전 고정 가중치 $w_c$가
있을 때 source-choice 정책의 상태별 총효과를 다음과 같이 둔다.

$$
\tau_{A:C\to B}(x,h)
=\mathbb E\!\left[Y^B(Z=A;q)-\sum_{c\in C}w_cY^B(Z=c;q)
\mid X^{\mathrm{pre}}=x,H=h\right].
$$

두 상태에서의 effect modification은 다음 차이다.

$$
D_B(x_1,x_2;h)
=\tau_{A:C\to B}(x_1,h)-\tau_{A:C\to B}(x_2,h).
$$

**[미완성]** 이 estimand를 식별하려면 explicit canonical source join, 실제 event-level
assignment strata, sequential exchangeability, within-stratum positivity, 고정 정책,
pre-treatment state, treatment-independent missingness와 carryover history가 모두
필요하다. 논문의 “mostly random”이라는 서술만으로는 충분하지 않다.

## 완전한 반례와 주장 상한

**[정리]** $M_{2,\mathrm{int}}$가 아닌 일반 full-state model의 예측 이득은
state-conditioned routing의 충분조건이 아니다.

증명. 실제 자료가 $Y^B=h(X^{\mathrm{pre}})+\epsilon$을 따르고 $Z$의 효과가 전혀
없다고 하자. $M_0$와 global-gain-only $M_1$이 $h$를 표현하지 못하지만 full-state
모델이 표현하면 $M_2>M_1>M_0$가 성립한다. 그러나 source별 효과는 모든 상태에서
0이다. 따라서 일반 예측 이득만으로 source-by-state routing을 결론 내릴 수 없다. □

**[산출]** source/state-dependent 광학 artifact, session drift와 이전 stimulation의
잔류 효과도 같은 겉모양을 만들 수 있다. active-source control은 common light
artifact를 일부 줄이지만 source별 targeting geometry·expression·detection quality를
자동 제거하지 않는다.

**[미완성]** 현재 schema에서는 source join, assignment strata와 positivity를 확정하지
못했으므로 $\tau$, $D_B$, $\Delta_{\mathrm{config}}$를 계산하지 않는다. 가능한 최대
문장은 “명시적 source identity와 event-level randomization이 확인된 경우, 자극 직전
population state가 randomized source-targeting policy의 fixed-target calcium response
contrast를 수정하는지 시험할 수 있다”이다.

## 후속 falsifier

다음 중 하나면 causal 상태-조건부 라우팅을 중지하고 observational prediction으로
격하한다.

1. explicit source join, complete assignment 또는 missingness table이 없다.
2. 실제 randomization strata나 within-stratum positivity가 없다.
3. held-out animal/session에서 $M_{2,\mathrm{int}}$가 $M_{2,\mathrm{add}}$를 이기지 못한다.
4. aligned-state permutation, dose·geometry, gain, event history와 session 대조 뒤
   source-by-state interaction이 남지 않는다.
5. optical/ROI adverse control에서 같은 interaction이 재현된다.

