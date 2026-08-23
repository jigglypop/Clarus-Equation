# 최종 보고서

Status: COMPLETE

## 결과

기존의 correlation-to-spatial-contraction 계열 A3--A5를 다시 조정하지 않고, “회로 하나가 생기면 고정된 뉴런 상태공간이 얼마나 끌리는가”를 두 개의 서로 다른 수학량으로 재정의했다.

수동적 변화는

$$
g_{\rm pass}(T)=J_T^\top G_TJ_T
$$

로 측정한다. 이 식은 flow derivative $J_T$가 full rank일 때만 genuine pullback Riemann metric이다. 회로 $\Gamma$를 $W(\varepsilon)=W_0+\varepsilon C_\Gamma$로 켜면 $\dot g_\Gamma$가 국소 응답이고, 전후 generalized eigenvalue의 제곱근이 방향별 finite stretch ratio다.

능동적 도달 난이도는

$$
E_T^*(v)=
\begin{cases}
v^\top\mathcal W_c(T)^\dagger v,&v\in\operatorname{Im}\mathcal W_c(T),\\
+\infty,&\text{otherwise}
\end{cases}
$$

로 측정한다. 이는 actuator $B$, cost $R$, horizon $T$를 포함한 terminal minimum-energy value이며 local Riemann/sub-Riemannian metric이라고 부르지 않는다.

## 뉴런 이질성 반영

기본 dynamics에는 signed edge strength $W_{ij}$, efficacy $p_{ij}^n$, integer delay $d_{ij}$, update fraction $\lambda_i$, neuron-specific net offset $b_i-\theta_i$가 각각 들어간다. 다만 independent calibration이 없으면 $b_i$와 $\theta_i$는 별도 식별되지 않는다. activity-dependent $p$를 사용할 때는 $p,e$를 augmented state에 넣어 도함수를 다시 계산해야 한다.

## 실제 주름과의 경계

실제 cortical surface의 계량은 measured embedding $X(\sigma,t)$에서

$$
g^{\rm anat}_{ab}=\partial_aX\cdot\partial_bX
$$

로 정의된다. activation-state pullback과 cortical material metric은 domain이 다르다. longitudinal embedding, tissue thickness, growth/material law, boundary condition과 observation map이 없으므로 둘의 bridge는 `BLOCKED_INPUT`이다. 주요 주름에는 태아기·초기 발달과 유전 영향이 있으나 영아기와 청소년기에도 morphology가 계속 변한다. 이것이 순간 회로가 물리 표면을 직접 잡아당긴다는 뜻은 아니다.

## 감사와 검증

- 상태 감사: PASS, 남은 P0/P1 없음.
- 무차원 checker: `17 passed`.
- deterministic A6 witness: PASS.
- analytic/finite-difference $\dot g$: `0.2797950506157409` / `0.2797950505034619`.
- analytic/finite-difference $\dot E$: `-0.148720999405116` / `-0.14872099945995032`.
- 새 empirical response와 confirmation: unopened.

## 형식 지위

- delayed tangent dynamics: 정의와 조건부 산출,
- full-rank passive pullback: 조건부 정리,
- finite-horizon endpoint minimum energy: 조건부 정리,
- eligibility plasticity: 모델 선택 공리,
- anatomy-to-state bridge와 실제 folding mechanism: 미완성/입력 차단,
- 실제 뇌 또는 AGI 개선: 아직 경험 검증 전.

## 다음 재개 조건

실제 뇌에 적용하려면 sender/receiver identity, signed strength, efficacy, per-edge delay, independently calibrated threshold 또는 net offset, actuator map과 fixed cost가 같은 session/event frame으로 결합되어야 한다. 이를 correlation, 거리 또는 graph support로 대체 생성하면 안 된다. 그 receipt가 생기기 전에는 synthetic tangent 검증 이상으로 주장을 올리지 않는다.
