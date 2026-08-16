# 물리 독립 AGI 코어 V0 최종 보고서

Status: COMPLETE

Formal scope: finite pure software-world mechanism certificate

AGI status: NOT ESTABLISHED

## 초록

현실 물리법칙이나 로봇 하드웨어를 코어 전제에 넣지 않고도 `관측 → 세계모델/기억 → 제한된 반사실 rollout → 계획 → 안전 허가 → 검증된 실행 → 경험 갱신 → 재현 가능한 ledger`의 최소 폐루프를 실행하는 V0 scaffold를 구현했다. 같은 protocol-only orchestrator는 비동형 XOR/SET 규칙 세계를 교체 실행하고, hidden evaluator truth가 learner 경로로 새지 않으며, post-genesis agent transition은 authenticated permit을 받은 pure reducer에서만 일어난다. 전체 canonical state를 고정하면 public ledger가 byte-identical하게 replay된다.

이 결과의 정확한 등급은 **기초 메커니즘 구현 성공**이다. unseen law 학습 성능, 언어적 의미 구성, SCC 기억 압축, 리만/제어기하, 수정 가능한 자기 가설, 의식 또는 AGI 전체는 아직 검증하지 않았다.

## 1. 무엇이 성공했는가

| 단계 | 현재 판정 | 의미 |
|---|---|---|
| 형식 계약 | PASS | 반례가 있는 강화 주장을 좁히고 V0의 정확한 정의역을 고정 |
| 수학·무차원성 | PASS | 두 비동형 family, canonical replay, linear-history nonce 정리와 snapshot-fork no-go 확인 |
| V0 구현 | PASS | 명시 protocol 조립과 pure event-sourced reference loop 구현 |
| focused 검증 | PASS | `16 passed`; 독립 검토 `ACCEPT`, 열린 P0/P1 0 |
| unseen-family 성능 | 미실행 | V1 preregistered procedural universe가 필요 |
| AGI·뇌·의식 | 미입증 | 상위 경험 주장을 만들 자료와 시험이 없음 |

PIC-I1--PIC-I5는 이 exact 유한 구현에 한해 `[산출]`이다.

- PIC-I1: concrete family dispatch 없는 동일 orchestrator가 XOR/SET 두 family를 실행한다.
- PIC-I2: genesis 뒤 agent transition은 permit 검증과 pure reducer 경로만 통과한다.
- PIC-I3: 같은 visible history에서 hidden truth가 달라도 proposal과 learner public-ledger bytes가 같다.
- PIC-I4: 전체 canonical initial state/history/code/schema가 같으면 public ledger가 같다.
- PIC-I5: forbidden action은 planner 선호와 무관하게 차단되고 permit nonce는 descendant linear history에서 재사용되지 않는다.

## 2. 핵심 구조

```text
WorldAdapter ──genesis──┐
MemoryStore ─infer/update├─> CoreOrchestrator ─> public hash-chain ledger
WorldModel ─predict/rollout┤          │
Planner ─rank──────────────┤          v
SafetyBoundary ─permit─────┤   ActionExecutor ─> pure world transition
```

참조 구현은 구성 편의를 위해 같은 객체가 두 protocol 역할을 수행할 수 있지만 오케스트레이터 생성자에서는 경계를 분리했다. 따라서 후속 연구에서 기억, 모델, planner, 안전정책, world adapter와 executor를 독립 교체할 수 있다.

## 3. 중요한 no-go와 남은 경계

순수 deterministic reducer와 전역 anti-rollback을 외부 상태 없이 동시에 얻을 수는 없다. descendant session은 nonce ledger 때문에 같은 permit을 거부하지만 과거 immutable snapshot을 다시 실행하면 같은 deterministic branch가 재현된다. 숨은 process-global consumed registry는 이를 막는 대신 동일 canonical input의 결과를 과거 호출 이력에 의존하게 만들어 PIC-I4를 깨므로 제외했다. 전역 single-use는 durable custody/executor가 담당해야 한다.

또한 protocol postcondition 검증은 pure software reducer에서는 fail-closed지만, 실제 actuator가 먼저 외부 side effect를 낸 뒤 모순된 receipt를 반환하는 경우 그 효과를 되돌리지 못한다. 현실 장치 단계에는 prepare/commit, idempotency key, durable journal, timeout/interrupt와 containment가 필요하다. 이는 물리법칙을 지능 코어에 하드코딩하는 문제가 아니라 외부 실행 adapter의 책임 경계다.

## 4. 아직 성공하지 않은 것

PIC-H1--PIC-H5는 모두 `[미완성]`이다.

- V0 tabular model은 규칙을 일반적으로 발견하는 learner가 아니라 폐루프 wiring을 검증하는 참조 구성이다.
- unseen procedural family, hidden context, partial observation, 연속 상태, 장기 계획과 distribution shift를 시험하지 않았다.
- SCC가 기억·계획에 유리하다는 결과가 없다.
- 리만 metric이나 제어기하가 planning을 개선한다는 결과가 없다.
- 언어가 실행 가능한 의미·의도·행동 공간을 만든다는 가설을 구현하지 않았다.
- `PrincipalIdentity`, 수정 가능한 `SelfHypothesis`, 불교적 무아에서 착안한 집착 완화/책임 경계 분리를 구현하지 않았다.
- 인간 뇌의 예측·신체보호 알고리즘, 생물학적 기억, 의식 또는 아라한 상태에 관한 증거가 아니다.

## 5. 다음 진행: Procedural Universe V1

다음 run은 V0 파일과 결과를 불변 predecessor로 두고 `artifacts/procedural-universe-v1-blueprint.md`를 사전등록 문서로 전환한다. 목표는 실제 물리 상수를 외우는 능력이 아니라, 제한된 이산 세계 문법 안에서 처음 보는 법칙 family를 능동 탐색으로 추론하고 새 목표에 적용하는 능력이다.

진행 순서는 다음과 같다.

1. typed entity/state/relation/action과 guarded rewrite AST interpreter를 독립 구현한다.
2. learner-facing token과 evaluator-only law AST를 분리하고 두 번째 독립 interpreter로 semantics를 대조한다.
3. family skeleton을 통계 단위로 한 development split에서 generator acceptance와 난이도를 고정한다.
4. random/reactive/model-free/memory-planner matched baseline과 oracle ceiling을 같은 budget으로 비교한다.
5. encoding isomorphism, wrong-law support, no-support, observational-twin intervention, metadata/reward leakage, duplication, brute-force, query contamination, canary, independent interpreter, retry 금지의 12개 kill test를 통과시킨다.
6. model·budget·threshold·generator·evaluator hash를 freeze한 뒤 별도 custody의 one-shot confirmation만 실행한다.

V1이 통과해도 허용되는 최강 결론은 `PROCEDURAL_UNIVERSE_V1_GO`다. 이는 고정된 procedural grammar 안의 unseen-law 적응 증거이지 인간 수준 AGI, 현실 물리 grounding, 의식 또는 불교적 성취가 아니다.

## 6. 최종 판정

V0의 목표였던 **물리 독립적 인터페이스와 안전·정보·재현 경계를 가진 최소 실행 코어**는 성공했다. 이로써 다음 연구를 기존 대형 AGI 코드의 직관적 결합이 아니라 교체 가능한 protocol과 반증 가능한 단계로 진행할 기반이 생겼다.

현재 전체 연구의 위치는 다음 한 문장으로 요약된다.

> AGI를 만들었다는 결과가 아니라, AGI 후보가 처음 보는 세계 법칙을 배울 수 있는지 정직하게 시험할 수 있도록 가장 아래 실행·권한·기억·재현 층을 닫았다.
