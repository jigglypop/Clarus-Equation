# A4 상태 의존 그래프 계량 사전등록

Status: FROZEN_BEFORE_A4_RESPONSE

## 실패에서 바꾸는 단 하나의 메커니즘

A3의 hard threshold는 회로장을 거의 영행렬로 만들었고 $\Omega\mapsto-\Omega$가 endpoint를 바꾸지 않았다. A4는 방향항과 정적 회로장을 폐기하고, 뉴런별 soft threshold와 대칭 회로강도가 매 순간 edge 비용을 바꾸는 상태 의존 그래프 계량만 시험한다. 이는 새 모델이며 A3의 성공으로 재분류하지 않는다.

새 source는 `source-manifest-v3.json`의 여덟 개체다. 공식 content size 순서에서 A3 cohort 다음 여덟 개를 결과와 무관하게 고정했다. 앞의 3개체만 development, 뒤의 5개체만 confirmation이다.

## 고정 분할과 뉴런별 역치

A3와 같은 $K=[0,0.2T)$, $B=[0.2T+8,0.7T-8)$, $T_{test}=[0.7T+8,T-1)$ 및 stimulus exclusion을 사용한다. 모든 normalizer와 top-3 projector는 $K$에서만 적합해 고정한다.

$$
\vartheta_i=\mu_i+2.5s_i,\quad s_i=1.4826\,\operatorname{MAD}_K(x_i),\qquad
z_i(t)=\frac{x_i(t)-\mu_i}{\max(s_i,10^{-6})},\qquad
a_i(t)=\sigma\!\left(\frac{x_i(t)-\vartheta_i}{s_i}\right)=\sigma(z_i(t)-2.5).
\tag{A4.1}
$$

$a_i\in(0,1)$는 hard event count가 아니라 뉴런별 역치에 대한 bounded excitability다. $K$에서 적합한 중심화와 projector $P_3$를 고정하고 $r(t)=(I-P_3)(z(t)-\bar z_K)$로 둔다. A4에서는 분석에 쓰는 전체 기록에서 finite인 뉴런만 outcome 값과 무관한 사전 규칙으로 남긴다. 결측값을 0이나 평균으로 대치하지 않으며, 남은 뉴런이 20개 미만이면 apparatus failure다.

## 회로별 강도와 상태 의존 edge 계량

NeuroPAL 좌표에서 deterministic 6-NN union graph를 만들고, 그 edge 거리의 median을 $\ell_{ref}$로 둔다. $\ell_{ij}=d_{ij}/\ell_{ref}$는 무차원이다. construction $B$에서 zero-lag residual correlation과 non-wrapping shift $\tau\in\{17,31,47\}$의 edge별 평균을 구한다.

$$
\alpha_{ij}=\left[\operatorname{corr}_B(r_i(t),r_j(t))
-\frac13\sum_{\tau}\operatorname{corr}_B(r_i(t),r_j(t-\tau))_{sym}\right]_+,
\quad
\widetilde\alpha_{ij}=\frac{\alpha_{ij}}{\operatorname{median}_{\alpha>0}\alpha}.
\tag{A4.2}
$$

양의 median이 없으면 `NO_DYNAMIC_EDGE`로 중단한다. 물리적 parent, synapse, 방향 또는 cycle을 뜻하지 않고 대칭 기능연결 강도만 뜻한다.

$$
h_{ij}(t)=\operatorname{clip}(\widetilde\alpha_{ij}a_i(t)a_j(t),0,4),
$$

$$
m_{ij}(t)=\ell_{ij}^2e^{-h_{ij}(t)},\qquad
w_{ij}(t)=m_{ij}(t)^{-1}=\ell_{ij}^{-2}e^{h_{ij}(t)}.
\tag{A4.3}
$$

따라서 해당 edge의 유효 길이비는 $\ell^{eff}_{ij}/\ell_{ij}=e^{-h_{ij}/2}$, conductance 비는 $e^{h_{ij}}$다. clip 때문에 각각 $[e^{-2},1]$와 $[1,e^4]$에 있다. 정점 좌표를 움직였다는 뜻이 아니라 고정된 점 사이의 기능적 이동비용이 변한다는 뜻이다. $w(t)>0$에서 graph Laplacian $L_t=D_t-w_t$는 대칭 positive semidefinite다.

## 동일 용량 예측식

construction에서만 $L_tz(t)$의 RMS $s_L>0$를 고정한다.

$$
\widehat z(t+1)=b+\beta_0z(t)-\beta_g\frac{L_tz(t)}{s_L},
\qquad \beta_g\ge0.
\tag{A4.4}
$$

$b=(b_1,\ldots,b_N)$는 뉴런별 절편 벡터이고, 그 외에는 worm별 scalar 두 개뿐이며 ridge는 $10^{-2}$로 고정한다. unconstrained 해가 $\beta_g<0$이면 $\beta_g=0$으로 두고 $\beta_0$만 재적합한다. fixed-geometry와 모든 control도 같은 용량·제약·construction/test 행을 사용한다.

## 식별성·대조·판정

candidate는 positive $\alpha$, nonzero $h$ variance, nonzero graph-feature RMS, 그리고 construction의 중심화된 두 열 $[z,-L_tz/s_L]$ rank 2 및 singular-value ratio $\ge0.05$를 요구한다. 실패는 `UNIDENTIFIABLE_GRAPH_TERM`이며 threshold나 graph를 낮춰 수리하지 않는다. clipping fraction을 보고한다.

필수 대조는 fixed geometry $(h=0)$, edge-strength shuffle within length tercile, construction time-shift $\alpha$, 31-sample state-time shift, identity-coordinate permutation 후 6-NN 재구성, construction-block independent-neuron phase randomization이다. time-shift control은 candidate의 차감식을 반복하지 않고

$$
\alpha^{shift}_{ij}=\left[\frac13\sum_{\tau\in\{17,31,47\}}
\operatorname{corr}_B(r_i(t),r_j(t-\tau))_{sym}\right]_+
$$

를 같은 방식으로 positive-median 정규화한다. 이는 강도 multiset을 보존하는 null이 아니라 먼 시차 상관만으로 만든 adverse operator이며, 별도의 edge-shuffle이 강도 재배치 null을 담당한다. 모든 arm은 동일한 31-sample guard를 사용한다.

각 LINDI index의 `generationMetadata` asset ID·path·size를 manifest와 대조한 뒤에만 청크를 읽는다. manifest의 SHA-256은 DANDI가 선언한 immutable full-asset digest다. range-read한 청크만으로 전체 blob SHA를 재계산하지 않으므로 provenance 지위는 `PARTIAL_RANGE_PROVENANCE`이며 full-byte verification으로 보고하지 않는다.

primary $\Delta_w$는 fixed geometry 대비 held-out Gaussian log-score 차이다. development는 3/3 admissible, 2/3 이상 $\Delta_w>0$, mean $\Delta>0$, candidate mean이 모든 control mean보다 커야 한다. 통과 전 confirmation은 열지 않는다. confirmation은 5/5 positive, exact one-sided sign-flip $p<0.05$, candidate mean이 모든 control보다 커야 한다. 단위는 worm이며 frame·neuron·seed는 복제수가 아니다.

development 실패 뒤 A4의 threshold, top-3, 6-NN, shifts, clip, ridge, horizon, controls를 같은 자료에 맞춰 바꾸지 않는다. 확인 실패 뒤에도 추가 적합하지 않는다. 통과하더라도 지위는 `OBSERVATIONAL_STATE_DEPENDENT_GRAPH_PREDICTOR`이며 해부학적 주름 변형이나 리만 곡률의 관측 증명이 아니다.
