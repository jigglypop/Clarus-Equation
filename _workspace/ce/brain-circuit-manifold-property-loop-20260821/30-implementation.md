# Implementation

Status: COMPLETE

## 구현 범위

`artifacts/a6_property_witness.py`는 NumPy만 사용하는 smooth delayed-state A6
property harness다. empirical response, actual BrainRuntime state 또는 anatomy asset을
열지 않는다.

구현한 검사는 다음과 같다.

1. heterogeneous `lambda_i`, signed `W_ij`, frozen `p_ij`, neuron-specific net
   offset `b_i-theta_i`, edge delay `d_ij`, time-varying actuator/cost의 nonlinear
   rollout.
2. delay-augmented analytic tangent와 initial-current finite difference.
3. `W(epsilon)` trajectory를 따라 `dot a`, total `dot A`, `dot B`, `dot Phi`,
   `dot J`, `dot g`와 중앙차분 비교.
4. delay lift 전체의 non-orthogonal coordinate rechart, transformed terminal metric,
   injection/projection과 quadratic/geodesic-local invariants.
5. augmented terminal control map, Gramian, weighted least-norm solution, reachable
   energy, chart covariance와 full-rank energy derivative.
6. zero actuator, rank-one unreachable target, near-singular operational rank와 exact
   passive rank-loss killing controls.
7. exact state-dependent efficacy fixture의 full `partial p/partial xi`, `dot p`와
   두 omitted-term controls.

## Revision 1

최초 실행의 source/result는 `*.initial.*`로 보존했다. 사후 감사의 두 P1에 따라
다음만 강화했다.

- randomized passive full-rank와 circuit/weight domain receipt를 Boolean PASS에 연결.
- strict JSON, source/contract/runtime provenance, status 문자열 추가.
- zero-response 진단을 실제 direct-edge-only partial derivative로 정정.

seed, equation, fixture, finite-difference step와 tolerance는 바꾸지 않았다. formula
revision은 0회다.

## 최종 구현 식별자

- source SHA-256:
  `c67f1f790c291f622db6362f31eeb58b9b7e4bc147d7c8d99d08f48fe511074d`
- contract SHA-256:
  `6322e6045b87f4fc7cd5d2bee29f1ceb800808de9dc1d2b5fcc0604926434fb4`
- Python `3.11.9`, NumPy `2.4.6` via `.codex/hooks/python.cmd`.

이 구현은 actual BrainRuntime의 hard/hybrid dynamics를 검증하지 않는다.
