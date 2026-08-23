# 경로 레인

Status: COMPLETE

## R1 — A6-P passive flow-pullback

선택 경로다. 고정된 history·input path·reference metric 아래

$$
g_{\rm pass}=J_T^\top G_TJ_T
$$

를 계산한다.

- 추가 자유도: $G_T$, horizon $T$, state normalization. 모두 endpoint와 독립적으로 고정해야 한다.
- target-aware 여부: 아니다. supervised target을 metric construction에 넣지 않는다.
- 관측·검산: tangent $J_T$와 symmetric finite difference의 일치, singular values, rank, condition number.
- 회로 한 개의 readout: $\dot g_\Gamma$, generalized eigenvalue $\Lambda_m$, $\Delta_\Gamma\log V_g$.
- killing falsifier: finite-difference Jacobian 불일치, undeclared state-dependent $p$, $J_T$ rank loss를 숨기는 ridge, outcome으로 고른 $G_T/T$.
- 주장 상한: fixed coordinate·trajectory 주변의 local state-space deformation.

## R2 — A6-C finite-horizon reachability-energy

선택된 보조 경로다.

$$
E_T^*(v)=v^\top\mathcal W_c(T)^\dagger v
$$

를 reachable subspace에서만 사용한다.

- 추가 자유도: actuator $B_n$, cost $R_n$, horizon $T$, pseudoinverse tolerance.
- target-aware 여부: $v_m$과 모든 자유도를 결과 전에 고정할 때만 아니다.
- 관측·검산: Gramian symmetry/PSD, rank, smallest positive eigenvalue, condition number, residual $\|H_TU^*-v\|$.
- 회로 한 개의 readout: full-rank 영역에서 $dE_T^*/d\varepsilon$, 그 밖에서는 finite pre/post energy와 subspace status.
- killing falsifier: $v\notin\operatorname{Im}\mathcal W_c$, ill-conditioned inverse를 안정한 값처럼 보고, actuator/cost 변경을 회로 효과로 오인.
- 주장 상한: specified actuator와 quadratic cost 아래의 linearized endpoint control energy.

## R3 — hybrid threshold/saltation

hard threshold나 projection을 유지하려면 trajectory를 smooth segment로 나누고 switching surface마다 saltation matrix를 삽입한다.

- 추가 자유도: event predicate, crossing order, reset map, simultaneous-event tie break.
- target-aware 여부: runtime semantics에서 사전 고정될 때만 아니다.
- 검산: switching 양쪽 one-sided finite difference.
- killing falsifier: grazing crossing, simultaneous ambiguous event, event 순서에 따른 불안정한 Jacobian.
- 상태: OPEN. A6-P/C의 smooth 기본식과 섞지 않는다.

## R4 — plasticity-augmented state

$p=p(a,e)$를 실제 dynamics로 만들려면

$$
z_n=(\xi_n,e_n,p_n)
$$

를 전 상태로 두고 $D_zF$를 계산한다.

- 추가 자유도: eligibility decay, modulator, update clock, boundary projection.
- target-aware 여부: modulator가 held-out answer를 읽지 않고 사전 고정될 때만 아니다.
- 검산: 모든 $\partial p/\partial a$, $\partial p/\partial e$ 항을 포함한 augmented finite difference.
- killing falsifier: omitted derivative, projection event 미기록, 서로 다른 tick/epoch clock 혼합.
- 상태: OPEN EMPIRICAL MODEL. frozen-$p$ 정리의 성공이나 실패를 소급 변경하지 않는다.

## R5 — anatomical cortical-fold mechanics

실제 주름을 계산하려면 longitudinal embedding $X(\sigma,t)$와 성장 tensor, thickness, constitutive law, boundary condition이 필요하다. 최소 구조는

$$
g^{\rm anat}_{ab}=\partial_aX\cdot\partial_bX,
\qquad
b_{ab}=n\cdot\partial_a\partial_bX
$$

이며, 여기에 독립적인 elastic/growth energy를 붙여야 한다.

- 추가 자유도: material parameters와 growth law가 매우 많다.
- target-aware 여부: anatomy endpoint로 조정하면 target-aware다.
- killing falsifier: longitudinal geometry·material receipt 부재, A6-P activation tensor와 anatomy tensor의 좌표 혼동.
- 상태: `BLOCKED_INPUT`. 현재 뇌 알고리즘 경로의 구현 후보가 아니다.

## R6 — directed Finsler/travel-time

비대칭 edge delay를 distance로 읽고 싶다면 undirected Riemann metric 대신 directed path cost 또는 Finsler 후보가 필요하다.

- 추가 자유도: path composition law, waiting cost, cycle convention, disconnected pair policy.
- killing falsifier: triangle inequality 실패를 숨기거나 negative/signed synaptic weight를 양의 travel time으로 임의 변환.
- 상태: DEFERRED. A6-C보다 추가 공리가 많다.

## 선택 순서와 중단 규칙

1. 먼저 작은 synthetic smooth network에서 A6-P tangent/finite-difference와 A6-C minimum-energy identity를 검산한다.
2. 그 뒤에만 실제 runtime의 sender/receiver orientation, per-edge delay, threshold semantics, $p$ receipt를 확인한다.
3. receipt가 없으면 값을 correlation·거리·graph support에서 대체 생성하지 않고 `BLOCKED_INPUT`으로 둔다.
4. A3--A5의 threshold·clip·RMS·ridge·horizon을 다시 조정하지 않는다.
5. 이번 run에서는 새 empirical response와 confirmation을 열지 않는다.

## 선택 결론

“회로가 manifold를 끈다”의 정식 표현은 하나가 아니다.

$$
\boxed{g_{\rm pass}=J_T^\top G_TJ_T}
$$

가 회로가 자유 흐름의 infinitesimal 길이를 어떻게 변형하는지 답하고,

$$
\boxed{E_T^*(v)=v^\top\mathcal W_c(T)^\dagger v}
$$

가 외부 입력으로 목표에 도달하기 얼마나 어려운지 답한다. 실제 피질 주름은 별도 anatomical surface mechanics 문제다.
