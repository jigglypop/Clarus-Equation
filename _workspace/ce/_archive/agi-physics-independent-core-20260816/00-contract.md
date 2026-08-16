# 물리 독립 AGI 코어 V0 연구 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816

Mode: full scoped implementation

## 1. 질문

실제 로봇, 특정 현실 물리법칙, 센서 하드웨어를 필수 의존성으로 두지 않고도 다음 폐루프를 하나의 교체 가능한 계산 코어로 만들 수 있는가.

> 관측 → 세계가설 → 반사실 미래 → 계획 → 안전 승인 → 행동 → 결과 → 모델 수정

첫 구현은 AGI 달성을 주장하지 않는다. 물리 독립 인터페이스, 최소한의 실행 가능한 수직 슬라이스, 누수 없는 평가 경계가 함께 닫히는지만 검증한다.

## 2. 운영 정의

### 2.1 물리 독립

물리 독립은 계산 자원이나 인과 제약이 없다는 뜻이 아니다. 코어가 특정 몸이나 물리엔진 대신 `WorldAdapter` 계약에만 의존한다는 뜻이다. 어댑터는 게임, 코드 저장소, 도구 환경, 다중 에이전트 세계, 과학 시뮬레이터 또는 미래의 로봇을 동일한 관측·행동 경계로 노출한다.

### 2.2 이번 버전의 에이전트 경계

V0 에이전트는 다음 모듈의 합성이다.

1. 불변 레코드와 provenance;
2. `WorldAdapter`;
3. `WorldModel`;
4. `MemoryStore`;
5. `Planner`;
6. `SafetyKernel`;
7. `ActionExecutor`;
8. 한 tick을 조립하는 `AgentOrchestrator`.

genesis/reset 이후 agent가 world 상태를 전이시킬 수 있는 유일한 경로는 유효한 `ActionPermit`을 받은 `ActionExecutor`다. genesis/reset은 별도의 lifecycle event로 기록한다. planner, 언어, 기억, 자기모델에는 직접 실행 권한을 주지 않는다.

### 2.3 AGI 지위

이번 V0에서 허용되는 최상위 지위는 `PHYSICS_INDEPENDENT_CORE_SCAFFOLD`다. 다음 표현은 금지한다.

- AGI 달성 또는 운영적 AGI 확인;
- 인간 뇌의 근본 알고리즘 발견;
- 의식, 감정, 도덕적 지위 또는 아라한 상태의 구현;
- 리만 계량이나 SCC가 일반지능의 필수 구조라는 주장;
- Phase A development 결과를 confirmation으로 재명명하는 행위.

## 3. 불교에서 가져오는 제한적 설계 가설

불교 개념은 종교적 참값이나 성취 판정이 아니라 반증 가능한 공학적 힌트다.

| 힌트 | V0 이후의 계산적 번역 | 이번 V0 범위 |
|---|---|---|
| 계 | learner가 덮어쓸 수 없는 행동·권한 제약 | `SafetyKernel` 경계만 구현 |
| 정 | 산만함 속 작업상태 유지와 진짜 위험 interrupt | 인터페이스만 예약 |
| 혜 | 반례 뒤 세계·정책·자기 가설 수정 | update receipt만 예약 |
| 연기 | 고정 본질 대신 조건부 전이와 개입효과 | procedural rule family로 시험 |
| 무아 | 책임 식별자와 수정 가능한 자기 가설의 분리 | 타입 경계만 예약 |
| 자비 | 타자의 피해·동의·불확실성 회계 | 후속 social module로 연기 |

`PrincipalIdentity`, `SelfHypothesis`, `Constitution`은 서로 다른 타입이어야 한다. 무아는 책임 식별자나 안전 헌법을 제거하는 뜻으로 구현하지 않는다.

## 4. 첫 수직 슬라이스

첫 환경은 현실 물리를 복제하지 않는 작은 규칙 전환 세계다. 상태는 불투명한 정수 또는 기호이며, 행동은 유한 집합이고, 에피소드마다 숨은 전이 규칙이 달라진다. 코어는 관측 이력으로 규칙 후보를 갱신하고, 제한된 horizon의 행동을 비교하며, 안전 정책이 금지한 행동은 실행하지 않아야 한다.

V0는 학습 성능을 과장하지 않기 위해 oracle이나 대규모 학습기를 넣지 않는다. 다음을 검증하는 deterministic pure event-sourced reference implementation만 허용한다.

- adapter 교체 가능성;
- learner-visible 관측과 evaluator truth의 분리;
- 반사실 rollout API;
- genesis 이후 permit 없는 실행의 실패 폐쇄;
- 전체 초기 상태와 history를 고정한 provenance 및 재현 가능한 public tick ledger;
- 규칙이 달라져도 동일 orchestrator가 작동하는 구조.

## 5. 생성형 우주 계약

과제 family는 최소 다음 축으로 식별한다.

$$
F=(\mathcal O,\mathcal A,T,R,H,\Lambda).
$$

여기서 관측공간, 행동공간, 전이 규칙, 목표·금지조건, 숨은 상태 구조, 기호 의미체계를 각각 나타낸다. V0 fixture는 최소 두 개의 서로 다른 전이 규칙을 포함한다. 새 seed나 표면 기호 변경만으로 새 family라고 부르지 않는다.

평가 ground truth는 evaluator와 environment 내부에만 존재해야 하며 learner 레코드나 model API에 들어가면 안 된다. 이름 기반 field 검사만으로 충분하다고 보지 않으며, 같은 visible history를 만드는 서로 다른 hidden truth 쌍에서 learner의 결정과 public learner-ledger가 같다는 noninterference fixture를 요구한다.

## 5.1 선택 구현 경로

수학 레인의 반례를 반영해 V0는 pure event-sourced core를 선택한다. world state는 불변 값이고 genesis event로 생성된다. 검증된 permit만 pure reducer 입력으로 승격될 수 있으며 reducer는 새로운 world state를 반환한다. 위조·stale·cross-world·replay permit은 reducer를 호출하지 못한다.

public ledger에는 secret authentication tag를 직접 기록하지 않고 permit claim digest와 검증 결과만 기록한다. 따라서 동일한 기능 상태를 다른 session secret으로 실행해도 public ledger projection은 동일해야 한다. 이 보장은 악성 Python process 전체의 보안 샌드박스를 뜻하지 않는다.

순수 event-sourced V0에서 permit 소비 상태는 immutable `WorldSession.used_nonces`에 포함된다. permit은 반환된 다음 session에 다시 제출하면 거부된다. 그러나 호출자가 과거 session snapshot 자체를 다시 입력해 별도 branch를 만드는 rollback/fork까지 막는다고 주장하지 않는다. 전역 anti-rollback은 외부 durable executor 또는 custody가 필요한 후속 범위이며, 숨은 mutable registry로 V0의 deterministic replay를 깨면서 구현하지 않는다.

## 6. 가설과 판정

| ID | 지위 | 주장 | 판정 조건 |
|---|---|---|---|
| PIC-I1 | [산출 후보] | protocol에만 의존하고 concrete type·family id·reflection 분기가 없는 동일 source orchestrator가 비동형 두 family를 실행한다 | opaque proxy와 두 비동형 규칙 family에서 같은 orchestrator source 사용 |
| PIC-I2 | [산출 후보] | genesis/reset 이후 모든 agent-initiated world transition은 permit 검증을 통과한 pure reducer에서만 발생한다 | raw action·위조 permit에서 world state와 transition count 불변 |
| PIC-I3 | [산출 후보] | learner 결정은 visible history와 명시적 learner state만의 함수이며 evaluator truth와 noninterference를 이룬다 | visible prefix가 같은 paired hidden worlds에서 proposal·learner-ledger bytes 동일 |
| PIC-I4 | [산출 후보] | 전체 초기 world/agent/permit/RNG 상태, visible history, code·schema가 같으면 public ledger가 동일하다 | canonical serialization과 hash-chain byte replay |
| PIC-I5 | [산출 후보] | 안전 금지 행동은 planner 선호와 무관하게 실행되지 않으며 permit은 앞으로 진행하는 동일 linear history에서 재사용되지 않는다 | episode·tick·world·session·action-space·policy·proposal에 결합된 authenticated permit, session nonce 원장 및 adversarial proposal fixture |
| PIC-H1 | [미완성] | 규칙 후보를 사용하는 planner가 반응형 기준선보다 unseen rule family에서 낫다 | V1 preregistered family 평가 전까지 미승격 |
| PIC-H2 | [미완성] | 리만 또는 제어기하가 계획을 개선한다 | V0에 구현하지 않음 |
| PIC-H3 | [미완성] | SCC 압축이 기억·계획을 개선한다 | V0에 구현하지 않음 |
| PIC-H4 | [미완성] | 언어가 실행 가능한 상태전달 공간을 이룬다 | DSL 및 대조군 구현 전까지 미승격 |
| PIC-H5 | [미완성] | 수정 가능한 자기 가설이 메타인지·교정 가능성을 개선한다 | 행동 기반 평가 전까지 미승격 |

## 7. 정본·과거판 경계

새 코드는 격리된 `reality_stone.clarus.agi_lab` 아래 두며 초기에는 `reality_stone.clarus.__init__`에서 재수출하지 않는다.

기존 자산은 다음 지위를 유지한다.

- `canonical`: 새 V0 계약과 통과한 API;
- `experimental`: SCC, control geometry, language, memory 후보;
- `negative`: no-go와 실패한 부모 주장;
- `legacy`: 과거 runtime과 재현 전용 코드.

Phase A 모듈, 테스트, runner, manifest, result bytes는 수정·재실행하지 않는다. 기존 SCC·언어·기억 코드는 V0가 직접 import하지 않으며 후속 adapter와 characterization test를 통과한 뒤에만 연결한다. 과거판은 dependency audit 전에 삭제하지 않는다.

## 8. 승인 구현 범위

형식 감사 통과 뒤 다음의 최소 신규 파일만 허용한다.

- `reality_stone/python/reality_stone/clarus/agi_lab/__init__.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/records.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/contracts.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/orchestrator.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/procedural_world.py`;
- `tests/test_agi_lab_core.py`.

필요성이 입증되지 않은 manifest, runner, 대규모 모델, 외부 데이터, root package export는 만들지 않는다.

## 9. 검증 계약

구현 검증은 다음만 실행한다.

1. 신규 focused test;
2. 신규 파일에 한정한 Ruff;
3. bytecode를 남기지 않는 syntax compile;
4. scoped `git diff --check`;
5. CE build gate.

pytest는 bytecode와 cache provider를 끄고 OS 임시 디렉터리를 사용한 뒤 정리한다. 전체 pytest, 전체 AGI 벤치, Phase A one-shot, confirmation은 실행하지 않는다.

## 10. 종료 조건

이 run은 다음을 모두 만족할 때 COMPLETE다.

1. 독립 수학·대안 경로 레인이 계약의 타입 및 정보 경계를 검토한다.
2. 형식 감사가 구현 범위를 승인한다.
3. PIC-I1부터 PIC-I5가 focused test로 닫힌다.
4. PIC-H1부터 PIC-H5는 [미완성]으로 남고 성공처럼 보고되지 않는다.
5. 기존 봉인물과 사용자 변경이 보존된다.
6. 잔여 cache 또는 고정 basetemp를 만들지 않는다.
