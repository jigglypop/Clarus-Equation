# 제한 기억ㆍreplayㆍ계층 계획 G7 사전등록

> 상태: **[산출] G7 V4 PASS / 후속 G7-M V1 validation FAILㆍV2 locked test PASS**
>
> 계약: `../../experiments/preregistration/memory_replay_planning_v1.json`

## 0. 비용과 주장

NumPy 합성계수와 짧은 물리 rollout만 사용한다. 저장하는 기억은 5차원 chart prototype뿐이며 궤적ㆍcheckpointㆍ외부 자료는 금지한다. 통과는 제한 기억에서 희귀 인과 regime의 보존과 모델기반 macro-action 효능만 뜻한다.

## 1. 기억 수식

episode 계수 (hat\theta_e)의 novelty와 rarity를

\[
n_e=\min_k\lVert D^{-1}(\hat\theta_e-\mu_k)\rVert,
\qquad r_e=(1+c_k)^{-1}
\]

로 두고 priority를 (p_e=n_e+\lambda_r r_e)로 정한다. 같은 chart이면 online 평균을 갱신하고, 새 chart인데 용량이 찼으면 priority가 가장 낮은 항목을 교체한다. replay 시에는 희귀 chart의 priority를 다시 올리되 원본 궤적은 저장하지 않는다.

## 2. stream과 비교군

- regime AㆍBㆍC 세 개, 기억 용량 3
- B는 초기에 두 번만 등장하는 희귀 regime
- 후반에는 C가 반복되어 recency FIFO에서 A/B가 밀려남
- 비교: `recency_fifo`, `reservoir`, `priority_replay`
- 독립 test seed에서 A/B/C recall과 40-step 제어 평가

## 3. 계층 계획

상위 계획기는 9개 macro-action 후보를 5 step 동안 내부 rollout하고 목표 원점까지의 비용이 가장 작은 후보를 선택한다. 하위 실행기는 선택된 action을 5 step 적용하고 다시 계획한다. zero-action과 recency 기억 기반 MPC를 기준선으로 둔다.

## 4. 통과 조건

1. priority replay의 세 regime recall 정확도 100%.
2. 희귀 B 계수오차가 recency보다 50% 이상 낮음.
3. 40-step 계획비용이 zero-action보다 25% 이상 낮음.
4. recency-MPC보다 20% 이상 낮음.
5. 기억 항목이 용량 3 이하이며 저장 prototype 총량 1KB 이하.
6. 외부 다운로드ㆍtrajectory 저장 0, wall time 15초 이하.

실패하면 test를 열기 전에 priority 수식, merge 거리, planning horizon을 별도 버전으로 바꾼다. 실제 로봇/대규모 replay buffer는 `SKIPPED_COST`다.

## 5. 루프 1 결과와 V2

V1은 120-byte 기억으로 recall 100%, 희귀 B 오차 96.8% 감소를 달성했지만 5-stepㆍaction 0.22에서 모든 planner가 zero action을 골라 계획 gate에 실패했다. 또한 등록 seed 대신 고정 seed 하나만 쓴 구현 결함을 발견했다.

V2는 모든 validation/test seed를 독립 실행하고 macro horizon을 10, action 후보를 `[-1,0,1]`로 바꾼다. 기억식과 통과 기준은 유지한다.

V2도 모든 정책이 zero action을 골랐다. 원점 목표에서는 B와 C가 모두 원점 복원력을 가져 기억 차이가 행동 차이로 이어지지 않았다. V3는 새 seed에서 초기 위치 유지 과제를 사용하고, B와 후반 C의 회전장 부호를 반대로 둔다. 등록 action cost는 `0.001`이다.

V3는 zero-action 대비 68.5% 개선했지만 correct/recency planner가 같은 coarse action을 골라 실패했다. V4는 후반 C의 actuator gain을 `-1`로 바꾸어 같은 명령의 물리적 의미가 반전되는 문맥을 사용한다. 새 seed를 사용한다.

## 6. V4 최종 결과: G7 PASS

locked test 9개에서 recall 정확도는 100%, 기억은 3개 prototypeㆍ120 bytes였다. 희귀 B 오차는 recency보다 97.4% 낮았다. priority-memory MPC 비용 `0.0710`은 zero-action `0.2258`보다 68.5%, recency-MPC `0.9710`보다 92.7% 낮았다. 실행은 0.18초, 외부 비용은 0이다.

## 7. 범위 경계와 다음 기억 실험

**[산출]** 이 prototype은 제한된 regime prototype의 recall과 macro-action
선택을 시험했다. partial cue에서 실제 episode trajectory를 복원하는
attractor, replay로 느린 world model을 갱신하는 consolidation, forgetting과
schema distortion, world-validity projection은 구현하지 않았다.

이 항목 중 episodic identity retrieval과 제한된 offline recombination은
[G7-M episodic LTM과 constrained imagination](42_Episodic_Attractor_Constrained_Imagination_G7M.md)의
2×2 factorial V1/V2로 사전등록해 평가했다. G7 V4나 G7-M V2의 PASS를 인간
기억ㆍ꿈 또는 해마-피질 통합의 증거로 확장하지 않는다.

## 8. 후속 G7-M V1 결과

> 사전등록 raw SHA-256:
> `6487156371e4c42877fa0813dd170fb000ce11fe05e51f34bceb74653159fac0`
>
> commits: 최초 등록 `48be628`, pre-implementation clarification
> `d2deae4`, 구현 `81c7ce6`, validation FAIL 보존 `0cc20f2`

**[산출]** G7-M V1은 validation check 73개 중 67개를 통과하고 6개를
실패했다. all-of 계약에 따라 전체는 **FAIL**이며 locked test는 미개봉이다.

| 분리한 경로 | validation 산출 | 지위 |
|---|---|---|
| persistent LTM identity | accuracy 0.993229, positive coverage 0.996354 | **[산출] PASS** |
| persistent LTM trajectory 이득 | relative reduction -0.036422; paired absolute gain -0.004759, 95% interval [-0.007192, -0.002326] | **[산출] V1 예측 FAIL** |
| LTM attractor 안정성 | convergence 0.418229, extra-step stability 0.540104; 각 gate 0.99 | **[미완성] FAIL** |
| constrained offline recombination | novel coverage 1.0, hidden NRMSE 0.089741; no-dream 0.654725 | **[산출] PASS**, known-slot 합성 family 한정 |

실패 check는 `L_main.hidden_reduction`, `L_main.hidden_ci`,
`M10.convergence`, `M10.extra_stability`, `M11.convergence`,
`M11.extra_stability`다. synthetic-to-LTM insert, observed overwrite,
heldout-target read와 constraint violation은 모두 0이었다.

따라서 G7-M V1의 LTM 추가 효능은 identity retrieval에 한정된다. trajectory
reinstatement와 episodic attractor는 해결되지 않았다. dream-like 경로의
성공도 observed 조각으로 known missing slot을 채운
`synthetic/hypothetical` schema-table augmentation일 뿐이다. 인간의 꿈,
수면, 장기기억 일반론, planning 또는 AGI의 증거로 확장하지 않는다.

## 9. 후속 G7-M V2 결과

> 사전등록 raw SHA-256:
> `973e90111ee98862a5c9ffc3f86509b46ee4e263b5a977e7e1504e00109092b9`
>
> 산출물:
> [`validation`](../../artifacts/agi/episodic_ltm_dream_factorial_validation_v2.json),
> [`locked test`](../../artifacts/agi/episodic_ltm_dream_factorial_test_v2.json)

**[예측: 사전등록 당시]** V1의 soft recurrent trace mixing만 cue-anchored
single-exemplar completion으로 바꾸면 fresh seed에서 trajectory 복원이 좋아지고
identityㆍcoverageㆍfalse-memory guard는 유지된다.

**[산출]** validation 40 seed가 84개 performance check와 7개 resource check를
모두 통과한 뒤 test 60 seed를 열었고, locked test도 84+7 check를 모두 통과했다.
test에서 `M10/M11`의 identity와 coverage는 0.995486, hidden NRMSE는 0.004074,
accepted-wrong rate는 0, lure false-recall rate는 0.027431이었다. 같은 fresh
seed의 frozen V1 soft hidden NRMSE 0.134971 대비 paired 이득은 0.130897,
95% interval [0.128358, 0.133437], strict win 60/60이다.

dream-like cell `M01/M11`의 novel coverage는 1.0, hidden NRMSE는 0.089235였고
no-dream은 0.656863이었다. synthetic-to-LTM insert와 observed overwrite는 0이다.
factorial interaction은 사실상 0이므로 두 경로의 **공존**만 확인했으며 synergy를
주장하지 않는다.

**[미완성]** V2 연산은 deterministic content-addressed exemplar/template
completion이다. recurrent attractor, replay consolidation, 인간의 장기기억이나
꿈ㆍ수면, 일반 세계모형, 이 장의 macro-action planning 또는 AGI를 입증하지 않는다.
V1 recurrent convergenceㆍstability FAIL과 V1 locked-test 미개봉 상태도 그대로
보존한다. 세부 계약ㆍhashㆍ경계는 [G7-M 42장](42_Episodic_Attractor_Constrained_Imagination_G7M.md)에 둔다.
