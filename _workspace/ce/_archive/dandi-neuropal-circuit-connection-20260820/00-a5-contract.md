# A5 고정 기하와 회로 변형 분해 사전등록

Status: FROZEN_BEFORE_A5_RESPONSE

## A4 실패와 새 추정량

A4는 $L_t$ 전체를 하나의 RMS와 하나의 coefficient에 넣어 inherited spatial geometry와 activity-induced deformation을 분리하지 못했다. A5는 정확한 항등식 $L_t=L_0+\Delta L_t$를 사용한다. A4 결과를 다시 맞추는 수정이 아니라 `source-manifest-v4.json`의 한 번도 response-open되지 않은 여덟 자산에서 시험하는 새 추정량이다. 앞 3개는 development, 뒤 5개는 confirmation이다.

K/B/T split, stimulus exclusion, complete-case neuron, $s_i=1.4826\,MAD_i$, $a_i=\sigma(z_i-2.5)$, calibration-only $P_3$, 6-NN, $\ell_{ref}$, $\widetilde\alpha$, $h=\operatorname{clip}(\widetilde\alpha a_ia_j,0,4)$는 A4 계약 그대로 고정한다. LINDI provenance도 ID·path·size를 먼저 대조하는 `PARTIAL_RANGE_PROVENANCE`다.

## 식

$$
W_{0,ij}=\mathbf1[(i,j)\in E_{6NN}]\ell_{ij}^{-2},
\qquad L_0=D_0-W_0,
\tag{A5.1}
$$

$$
\Delta W_{ij}(t)=W_{0,ij}\left(e^{h_{ij}(t)}-1\right)\ge0,
\qquad
\Delta L_t=\Delta D_t-\Delta W_t\succeq0.
\tag{A5.2}
$$

construction $B$에서만

$$
s_0=\operatorname{RMS}_B(L_0z_t)>0,\qquad
s_\Delta=\operatorname{RMS}_B(\Delta L_tz_t)>0
\tag{A5.3}
$$

를 고정하고

$$
\widehat z(t+1)=b+\beta_0z(t)
-\beta_s\frac{L_0z(t)}{s_0}
-\beta_\Delta\frac{\Delta L_tz(t)}{s_\Delta},
\quad b\in\mathbb R^N,\quad \beta_s,\beta_\Delta\ge0
\tag{A5.4}
$$

로 예측한다. $\ell$이 $\ell_{ref}$로 무차원화됐으므로 모든 항과 coefficient가 무차원이다. ridge $10^{-2}$는 고정한다. $\beta_s,\beta_\Delta$는 unconstrained 해를 순차 절단하지 않고 가능한 active set을 전부 열거해 joint nonnegative ridge optimum을 택한다.

no-deformation은 $\beta_\Delta=0$인 정확한 nested model이다. $\beta_\Delta$는 B-RMS 한 단위의 incremental predictive coefficient이지 물리적 변형량이 아니다. 실제 변형 크기는 별도로 $h$, $e^{-h/2}$, $s_\Delta/s_0$를 보고한다.

## 식별성과 대조군

candidate는 A4 admission gate에 더해 $s_0,s_\Delta>10^{-10}$, $\operatorname{var}_B(h)>10^{-10}$, 중심화된 flattened design $[z,-L_0z/s_0,-\Delta L_tz/s_\Delta]$ rank 3, $\sigma_{min}/\sigma_{max}\ge0.05$를 요구한다. no-deformation design은 rank 2여야 한다.

same-capacity controls는 edge-strength shuffle, adverse time-shift $\alpha^{shift}$, 31-sample state shift, identity-coordinate permutation, independent-neuron phase randomization이다. 각 control은 $\Delta L$만 교체하고 original $L_0$, $z$, split, $b$, $\beta_0$, $\beta_s$, ridge를 유지한다. 각 control도 자체 $s_{\Delta,k}$와 rank-3 gate를 통과해야 한다. 하나라도 degenerate이면 `CONTROL_DEGENERATE`이며 “모든 control을 이겼다”는 pass를 허용하지 않는다.

## 판정

각 worm에서 no-deformation 대비 proper Gaussian held-out log-score $\delta_w$와 각 same-capacity control 대비 paired 차이를 계산한다. development는 3/3 admissible, 2/3 이상 $\delta_w>0$, mean $\delta>0$, A5 mean이 모든 control mean보다 커야 한다. 통과 전 confirmation은 봉인한다.

confirmation은 5/5 admissible positive $\delta_w$, exact one-sided animal sign-flip $p<0.05$, 모든 control보다 큰 mean, 그리고 동일 animal sign을 함께 뒤집는 exact max-statistic family에서 모든 adjusted $p<0.05$를 요구한다. frame·neuron·seed는 복제수가 아니다.

A5가 실패하면 같은 자산에서 threshold, $P_3$, 6-NN, shifts, clip, RMS, ridge, horizon, constraint, controls 또는 자산 subset을 바꾸지 않는다. 성공해도 지위는 `OBSERVATIONAL_INCREMENTAL_GRAPH_FEATURE`이며 해부학적 주름 변형, 시냅스 parent, 방향, cycle, 인과 또는 실제 리만 곡률을 증명하지 않는다.

