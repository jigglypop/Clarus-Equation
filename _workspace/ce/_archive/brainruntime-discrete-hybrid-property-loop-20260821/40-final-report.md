# Final report

Status: COMPLETE

Gate: PASS

## 1. 결론

A7-H는 A6의 매끄러운 activation-delay 모형을 실제 `BrainRuntime`의 한 tick에
가까운 이산 하이브리드 식으로 확장했고, 고정 synthetic fixture에서 그 식을
검증했다. 최종 지위는 다음과 같다.

```text
DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED /
HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED / EMPIRICAL_UNTESTED
```

`PASS`는 fixed-mode branch 식과 그 경계 처리의 수학/구현 일치를 뜻한다. Rust가
delay를 구현했다는 뜻은 아니다. 오히려 delay-on mismatch를 정확히 재현했기 때문에
그 부분은 `BLOCKED`로 남는다.

## 2. 다시 세운 식에서 달라진 점

기존 A6 식은 smooth delayed activation만을 상태로 삼았으므로 실제 runtime의 STP,
refractory, memory trace, adaptation, delay ring, previous lifecycle mask, bit, TopK를
미분할 수 없었다. A7-H는 continuous state를

$$
z=(a,r,m,w,u,x,d^{(0)},d^{(1)})\in\mathbb R^{24}
$$

로 확장하고, discrete state를 별도로 뒀다. strict branch에서는 24 by 24
Jacobian을, clip 면에서는 방향에 따른 one-sided derivative를, bit/TopK/lifecycle
변화에서는 미분 대신 transition receipt를 사용했다. continuous-time saltation은
이 ticked map에 필요한 event-time 가정이 없으므로 폐기했다.

지연은 누적 counter $\kappa$와 read slot $k=\kappa\bmod2$를 분리했다. 각 호출은
slot을 먼저 읽고 old activation을 대입한 뒤 $\kappa$를 하나 올린다. 따라서 zero
ring에서 첫 도착은 세 번째 호출 $t=2$다.

## 3. 무엇이 검증됐는가

- Torch/mirror one-step continuous error: `4.8676e-8`;
- full 24-dimensional Jacobian error: `2.6270e-12`;
- reachable clip-face worst final error: `6.5854e-8`;
- ring arrival at `t=2`: recurrent norm `3.5435e-3`;
- lifecycle mask's next-tick effect: `2.3413e-2`;
- no-delay Torch/Rust error: `7.4506e-9`;
- delay-on Torch/Rust activation mismatch: `3.3332e-2`.

Neuron permutation은 수학 mirror와 actual Torch에서 통과했다. arbitrary dense chart
covariance는 componentwise tanh, axis-aligned clamp, absolute salience와 TopK가 보존되지
않으므로 주장하지 않는다.

## 4. 사용자가 지적한 뉴런별 차이는 어디까지 반영됐는가

회로 강도 $W_{ij}$는 이미 heterogeneous signed matrix로 반영되고 이번 fixture도
서로 다른 양/음 강도를 사용했다. 그러나 bit lower/upper와 active threshold는 현재
코드에서 global scalar다. 수학 envelope에 $\theta_i^\pm,\vartheta_i$를 쓰는 것과
actual runtime이 이를 구현했다는 것은 다른 주장이다. 따라서 뉴런별 역치는 아직
`HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED`다.

또한 이 상태공간 미분은 cortical folding의 물리적 곡률이나 발달기 주름을 측정하지
않는다. 해부 표면 metric/curvature와 runtime state tangent 사이의 관측 bridge는
여전히 별도 입력과 검증이 필요하다.

## 5. 실패와 수정의 보존

첫 실행은 잘못 쓴 Rust source path에서 계산 전 중단돼 `P2_APPARATUS`로 보존됐다.
첫 passing result는 package version receipt가 비어 있어 다시 보존됐다. 최종본은
source-tree package의 `__version__=0.2.10`을 기록했으며, 모든 수치가 정확히
재현됐다. 수식, fixture, 방향, step, tolerance 또는 pass gate는 실행 후 바뀌지 않았다.

## 6. 다음 재귀 경로

다음 식-시험 루프의 최우선 순위는 둘을 섞지 않고 분리하는 것이다.

1. **A8-T:** 실제 config/state에 neuronwise
   $\theta_i^-,\theta_i^+,\vartheta_i$를 도입하고 scalar broadcast가 기존 동작과
   완전히 호환되는지, permutation과 guard receipt가 유지되는지 시험한다.
2. **A8-D:** Rust API에 Torch와 같은 ring buffer/counter를 넣는 별도 구현 계약을
   세우고, 현재 H-G 반례를 그대로 회귀 시험으로 사용한다.

A8-T의 threshold 구현은 지연 kernel 수리와 독립적이므로 먼저 진행할 수 있다.
어느 route도 brain biology, AGI, 주름 geometry를 자동으로 승격하지 않는다.
