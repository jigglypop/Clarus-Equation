# Final report

Status: COMPLETE

Final verdict: MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED

## 무엇을 확인했는가

이번 루프는 “회로 하나가 생겼을 때 상태공간 매니폴드가 얼마나 끌리는가”를 나타내는
A6 식을 작은 예제 한 개가 아니라 여덟 개의 이질적 지연 네트워크에서 반증하려고
했다. 각 네트워크에는 서로 다른 update fraction, net threshold offset, signed
connection strength, efficacy, edge delay와 actuator cost가 들어갔다.

수동적 deformation은

\[
g_{\rm pass}=J_T^TG_TJ_T
\]

로, 회로의 국소 영향은

\[
\dot g_\Gamma=\dot J_T^TG_TJ_T+J_T^TG_T\dot J_T
\]

로 계산했다. analytic tangent와 total circuit derivative는 모두 nonlinear central
finite difference와 약 `1e-12` 수준에서 일치했다. delay history 전체와 metric,
injection, projection을 함께 옮긴 좌표변환에서는 길이ㆍgeneralized stretchㆍvolume
ratio가 보존되었다. terminal metric을 옮기지 않은 adverse chart는 최소
`1.54e-3`만큼 깨져, 단순한 좌표 숫자 바꾸기를 기하학적 불변성으로 오인하지 않았다.

능동적 제어 쪽은

\[
W_c=H\bar R^{-1}H^T,
\qquad E^*(v)=v^TW_c^\dagger v
\]

를 reachable augmented terminal state에만 적용했다. weighted least-norm control과
Gramian energy는 모든 seed에서 일치했고, full-rank domain의 `dot E`도 finite
difference와 일치했다. 반대로 zero/rank-one actuator의 unreachable target에는
pseudoinverse 숫자를 energy로 읽지 않고 `Infinity`를 반환했다.

## 이번에 실제로 고친 문제

수학식은 첫 property 실행에서 실패하지 않았다. 대신 사후 감사가 두 구현 P1을
찾았다. full-rank와 weight-domain 값을 JSON에만 기록하고 Boolean PASS에 연결하지
않은 문제였다. 최초 source/result를 보존한 뒤 같은 식ㆍseedㆍtolerance로 Revision 1을
재실행했고, 강화된 gate도 8/8 PASS했다. 이는 threshold tuning이나 formula rescue가
아니다.

또한 state-dependent efficacy에는

\[
\partial(p_{ij}a_j)=p_{ij}\partial a_j+a_j\partial p_{ij},
\qquad \dot p_{ij}=\nabla p_{ij}\cdot\dot\xi
\]

가 반드시 필요함을 수치로 확인했다. full 식의 오차는 약 `1e-12`였지만 이 항들을
뺀 식은 `6.06e-2`, `2.21e-3`으로 명확히 실패했다. 따라서 frozen-`p` A6를 plastic
synapse 식으로 그대로 확대할 수 없다.

## 무엇을 아직 말할 수 없는가

- 이 결과는 actual BrainRuntime의 hard threshold, active-mask selection, clamp,
  STP, refractory/adaptation 또는 Torch/Rust delay parity를 검증하지 않았다.
- `b_i-theta_i`만 나타나는 현재 식에서 별도 calibration 없이 개별 bias와 threshold를
  분리 식별할 수 없다.
- 상태공간 pullback은 cortical tissue의 물리적 Riemann metric, 응력, 곡률 또는
  실제 뇌 주름이 아니다.
- 해부학적 주름 다리는 longitudinal embedding, thickness, growth/material law,
  boundary condition과 circuit-to-tissue observation receipt가 없으므로 계속
  `BLOCKED_INPUT`이다.
- synthetic seed는 empirical replicate나 AGI 성능 증거가 아니다.

## 다음 재귀 루프

다음 식-테스트 루프는 A6를 actual runtime에 바로 덮어씌우지 않고 별도 A7-H
hybrid branch로 시작해야 한다. 먼저 한 backend의 exact step semantics를 고정하고,
hard threshold/selection surface의 양쪽 one-sided derivative와 saltation event,
delay-buffer state, STP/refractory/adaptation의 augmented Jacobian을 세운다. 이후
Torch/Rust가 동일 event receipt를 내는지 검사한다. 이 gate가 없으면 smooth A6의
PASS를 AGI brain runtime의 manifold deformation PASS로 승격할 수 없다.

현재 A6-P/C 식은 조건부 수학 구조로 유지한다. 다음 loop에서 실패하면 smooth 식의
tolerance를 바꾸지 않고 hybrid event equation 또는 runtime semantics를 수정한다.
