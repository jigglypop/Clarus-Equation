# 물리 독립 AGI 코어 V0 구현 기록

Status: COMPLETE

## 1. 구현 결과

승인된 여섯 파일에만 `PHYSICS_INDEPENDENT_CORE_SCAFFOLD`를 구현했다.

1. `records.py`: immutable canonical records, strict int64/finite-number/NFC·UTF-8 grammar, public/private serialization 경계, hash-chain ledger records.
2. `contracts.py`: `WorldAdapter`, `WorldModel`, `MemoryStore`, `CorePlanner`, `SafetyBoundary`, `ActionExecutor`를 분리한 protocol surface.
3. `orchestrator.py`: genesis, belief inference, bounded planning, safety authorization, verified execution, experience commit, public ledger append를 조립하는 family-independent orchestrator.
4. `procedural_world.py`: 정확한 XOR/SET 두 규칙 family, authenticated permit, immutable session nonce ledger, 안전 fallback, tabular memory/model과 finite counterfactual rollout.
5. `__init__.py`: 격리된 명시 API와 AGI·의식·뇌 알고리즘 비주장 표지.
6. `test_agi_lab_core.py`: 계약 불변식과 반례 회귀를 다루는 focused certificate.

루트 `reality_stone.clarus`에서는 재수출하지 않았고 Phase A sealed module·runner·manifest·result 및 기존 SCC/언어/기억 구현은 수정하거나 실행하지 않았다.

## 2. 실제 조립 경계

오케스트레이터는 구체 world family를 import하거나 검사하지 않는다. 생성은 `WorldAdapter`, 전이는 `ActionExecutor`, 기억의 초기화·추론·갱신은 `MemoryStore`, 예측과 action-sequence rollout은 `WorldModel`에만 의존한다. 참조 구성에서는 한 `ProceduralWorldAdapter`가 adapter/executor 두 역할을, 한 `TabularWorldModel`이 memory/model 두 역할을 구조적으로 구현하지만 생성자 경계는 독립되어 있다.

동일한 오케스트레이터 source가 다음 비동형 전이 규칙을 실행한다.

```text
XOR: T(s,a) = s xor a   -> trace (1, 1, 0)
SET: T(s,a) = a         -> trace (1, 0, 1)
```

planner는 hidden rule이나 evaluator truth를 받지 않는다. paired hidden worlds의 visible prefix가 같은 동안 proposal과 learner-visible public ledger bytes가 동일함을 직접 검사한다.

## 3. permit과 재현성 경계

permit은 schema, episode, tick, world commitment, full session digest, action-space, safety policy, proposal, action, nonce에 결합되고 HMAC으로 인증된다. executor는 permit/session의 world commitment가 자기 world commitment와도 같은지 확인한다. raw action, 위조·stale·cross-world·terminal permit은 reducer에 들어가지 않는다.

single-use의 정확한 범위는 반환된 descendant session을 따라 앞으로 진행하는 동일 linear history다. `WorldSession.used_nonces`가 단조 증가하므로 그 history에서는 재사용이 거부된다. 그러나 과거 immutable snapshot 자체를 다시 넣으면 동일한 deterministic fork가 재현된다. 이것까지 전역 차단하려면 외부 durable executor/custody가 필요하다. 숨은 mutable consumed-registry를 넣는 방식은 같은 canonical state의 replay 결과를 실행 이력에 따라 바꾸므로 채택하지 않았다.

public ledger는 secret key와 authentication tag를 포함하지 않고 permit public-claim digest만 기록한다. 따라서 같은 기능 상태를 다른 session key로 실행해도 public ledger bytes는 같다.

## 4. 독립 검토에서 닫은 결함

구현 중 독립 검토로 다음을 발견하고 focused 회귀로 고정했다.

- adapter 자신의 world commitment 비교 누락으로 가능한 원본 session+permit의 cross-world 치환;
- hidden-world paired test에서 learner update 이후 public ledger equality 직접 증거 누락;
- raw action 무전이 fixture 누락;
- 중간 event 재계산 뒤 downstream hash propagation 직접 증거 누락;
- lone Unicode surrogate가 frozen record를 통과한 뒤 UTF-8 serialization에서 실패하는 canonical grammar 공백;
- `MemoryStore`, `ActionExecutor`, counterfactual rollout의 암묵적 역할/API 불일치;
- 외부 protocol component가 의미상 모순된 session/start/step/receipt를 반환해도 ledger에 들어가던 postcondition 공백.

최종 오케스트레이터는 genesis의 session/observation/action-space/policy, transition의 session/WorldStep/nonce/commitment, update의 belief/receipt/experience를 교차검증한다. 결함 있는 genesis adapter, executor, memory fixture는 모두 fail closed한다.

## 5. 범위 제한

이 구현은 pure finite software-world reference다. postcondition 검증은 외부 물리 actuator의 이미 발생한 side effect를 되돌리는 transaction이 아니다. 실제 장치 연결에는 prepare/commit, idempotency, durable custody와 물리적 containment가 별도로 필요하다.

또한 이 구현은 planner 성능, unseen-family 일반화, AGI, 의식, 인간 뇌의 근본 알고리즘, 언어의 필연성, SCC 또는 리만 metric의 필요성을 증명하지 않는다. PIC-H1--PIC-H5는 계속 `[미완성]`이다.
