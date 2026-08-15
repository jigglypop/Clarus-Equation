# Phase A estimator·benchmark 대안 경로

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-connectome-geometric-memory-20260816/12-routes.md`

## 1. 공통 목표와 평가 규칙

모든 경로의 목표는 training의 $(x_t,u_t,z_t,x_{t+1})$만 사용해 held-out intervention 다음 상태의 graph-seed별 Gaussian NLL을 낮추는 것이다. primary 통계단위는 frame이 아니라 fresh graph seed이며, 공통 manifest $\sigma$와 같은 split을 사용한다. exact coefficient/support는 known-identity와 full-rank certificate가 참일 때만 secondary로 연다.

모든 estimator 경로는 primary endpoint를 위해 선택되므로 `target-aware`다. 따라서 후보 수, ridge grid, context basis, rank와 STOP 결과를 selection ledger에 기록해야 한다. confirmation은 이 경로 선택에 사용하지 않는다.

## 2. R1 — 문맥별 $A_z$와 공유 $B$의 joint fit

$$
\widehat x_{t+1}=\widehat A_{z_t}x_t+\widehat Bu_t.
$$

- **dof:** nominal $n(Kn+m)$; ridge를 쓰면 `11-math.md` 식 (9)의 effective dof도 보고.
- **식별 gate:** stacked design rank $Kn+m$, 모든 $X_z$ full rank, residualized-input Gram 식 (3) 양의 정부호.
- **장점:** 계약의 생성 가정과 정확히 맞고, 문맥 전이를 분리하면서 intervention mechanism $B$의 sample을 모든 문맥에서 공유한다.
- **target-aware/look-elsewhere:** 사전에 V1 candidate로 고정한다. ridge 하나를 고정하거나 development 내부 nested grid 횟수를 공개한다.
- **교차예측:** training에 없는 intervention vector·dose에서 context별 next-state를 예측하고, 가능하면 held-out trajectory seed와 graph seed를 동시에 바꾼다.
- **kill test:** rank certificate false, unknown mix exact-edge request, input/time shuffle이 intact와 동률, 또는 graph-seed paired $\Delta_s$가 0 이하이면 해당 claim을 STOP한다.

**V1 권고안은 R1이다.** 가장 작은 모형으로 계약의 공유-$B$ 가정을 직접 시험하며 pooled와의 nominal dof 차이가 정확히 $(K-1)n^2$다.

## 3. R2 — 문맥별 $A_z,B_z$ 완전 분리 fit

$$
\widehat x_{t+1}=\widehat A_{z_t}x_t+\widehat B_{z_t}u_t.
$$

- **dof:** $nK(n+m)$; R1보다 $(K-1)nm$ 많다.
- **식별 gate:** 각 문맥별 $[X_z;U_z]$가 rank $n+m$이어야 한다.
- **용도:** shared-$B$ misspecification stress test. training SSE가 더 작다는 사실만으로 선택하지 않는다.
- **target-aware/look-elsewhere:** primary NLL을 본 뒤 R1 대신 채택하면 model-selection 자유도가 하나 늘어난다. 사전 지정 secondary로만 둔다.
- **교차예측:** seen context의 unseen intervention에는 사용할 수 있지만, 표본이 적은 문맥과 unseen context에는 약하다.
- **kill test:** parameter/FLOP 차이를 숨기거나, 문맥별 design rank가 부족하거나, R1 대비 OOD 이득이 development graph 전반에서 재현되지 않으면 공유기전 반박으로 승격하지 않는다.

## 4. R3 — pooled $A$와 공유 $B$ baseline

$$
\widehat x_{t+1}=\widehat Ax_t+\widehat Bu_t.
$$

- **dof:** $n(n+m)$.
- **식별 gate:** pooled $[X;U]$ rank $n+m$.
- **용도:** 문맥 label을 쓰는 추가 $(K-1)n^2$ parameter가 OOD에서 값을 주는지 묻는 필수 baseline.
- **target-aware/look-elsewhere:** endpoint에 맞춰 선택된 baseline이지만 하나로 고정한다. 여러 pooling 규칙 중 사후 최선을 고르지 않는다.
- **교차예측:** 같은 held-out intervention과 graph seeds를 R1과 paired 평가한다.
- **kill test:** 모든 $A_z$가 같거나 context가 사실상 상수인 negative fixture에서 R1과 예측이 동률이어야 한다. 이때 R1 strict 우위를 보고하면 harness 실패다.

## 5. R4 — 관측 context feature의 bilinear transition

관측 context feature $h(z)\in\mathbb R^q$를 사전에 정의하고

$$
A(z)=A_0+\sum_{j=1}^{q}h_j(z)M_j,
\qquad
\widehat x_{t+1}=A(z_t)x_t+\widehat Bu_t
$$

로 fit한다.

- **dof:** $n((q+1)n+m)$.
- **식별 gate:** $[x_t;h_1(z_t)x_t;\ldots;h_q(z_t)x_t;u_t]$ stacked design이 full row rank.
- **장점:** context가 순서·연속 covariate를 가질 때 $q<K-1$로 구조를 공유하고 unseen context feature에 보간할 수 있다.
- **경계:** categorical full contrast $q=K-1$이면 R1의 재매개화이므로 독립적인 세 번째 성공 증거가 아니다.
- **target-aware/look-elsewhere:** context basis, $q$, interaction을 endpoint 전에 고정한다. 여러 basis를 시험하면 그 수만큼 selection ledger와 nested validation이 필요하다.
- **교차예측:** 학습하지 않은 $h(z)$ 값 또는 held-out context level에서 next-state를 예측한다.
- **kill test:** context feature가 outcome/test에서 만들어졌거나, basis rank가 부족하거나, unseen-context 보간이 pooled보다 나쁘면 이 경로를 STOP한다.

## 6. V1 판정표

| 경로 | 역할 | nominal dof | V1 지위 | 주 kill condition |
|---|---|---:|---|---|
| R1 shared-$B$ context fit | candidate | $n(Kn+m)$ | **권고** | full-rank/refusal/shuffle/NLL gate 실패 |
| R2 fully separated fit | misspecification stress | $nK(n+m)$ | secondary | extra dof 회계 또는 per-context rank 실패 |
| R3 pooled fit | 필수 baseline | $n(n+m)$ | **필수** | equal-context negative fixture에서 허위 격차 |
| R4 bilinear observed-context | structured-context alternative | $n((q+1)n+m)$ | V1 보류 | basis 사후선택 또는 unseen-context 실패 |

V1은 R1 대 R3의 paired development 비교로 고정한다. R2는 shared-$B$ 가정을 공격하는 secondary이고, R4는 one-hot 재매개화가 아닌 외생 context structure가 사전등록되는 후속 버전에서만 연다. 어느 경로도 성공 자체를 SCC·기억·생물학·의식 또는 AGI 증거로 해석하지 않는다.
