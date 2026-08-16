# AGI Core V0 구현 후 독립 API 계약 리뷰

Status: COMPLETE

Review decision: ACCEPT

Open severity: P0 = 0, P1 = 0

## 1. 결론

최종 고정된 여섯 파일을 독립적으로 다시 읽고 승인된 focused test만 재실행했다. 이 exact snapshot에서 PIC-I1--PIC-I5는 번들 V0 reference composition에 한해 구현 certificate를 충족한다. 따라서 다섯 항목은 `[산출 후보]`에서 좁은 구현 `[산출]`로 승격할 근거가 충분하다.

이 판정은 두 procedural family, 현재 protocol과 현재 reference component에 한정된다. AGI 달성, 인간 뇌의 근본 알고리즘, 의식, SCC 또는 리만 구조의 필요성, unseen-family confirmation은 증명하지 않는다. `__init__.py:56-59`의 scaffold 지위와 세 상위 claim lock은 그대로 타당하다.

## 2. 검토 snapshot

| 파일 | SHA-256 |
|---|---|
| `reality_stone/python/reality_stone/clarus/agi_lab/__init__.py` | `1BD80EE8AE37E107665A44A968C7F53C2655A02D0DF45435AC8A089241D1FD98` |
| `reality_stone/python/reality_stone/clarus/agi_lab/records.py` | `A178B82A79B1D73B1CA03D9B5C0F89B3C469352D13B7DA12C0A5C2B259D1DBE4` |
| `reality_stone/python/reality_stone/clarus/agi_lab/contracts.py` | `18ED414F2A4BE0388C7504321CFD49A81C1AA567272B46E5A6A33282225F7B2E` |
| `reality_stone/python/reality_stone/clarus/agi_lab/orchestrator.py` | `2E37349784F105C8527BB0F0FB7397B75668ABE514EB5A4283CDCA074171F3C5` |
| `reality_stone/python/reality_stone/clarus/agi_lab/procedural_world.py` | `4B2CCDAACA0A836137E41CD8F2858A02030C22EBC8370F7A25E3B77D228B050B` |
| `tests/test_agi_lab_core.py` | `5FE57F67C8A76436C265AC65F9E5DBAC23DE27FDDF2EF035C02465BF5856910F` |

검토는 제품 코드를 수정하지 않고 수행했다. 최종 검증 명령의 의미상 형태는 다음과 같다.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m pytest -q -p no:cacheprovider tests/test_agi_lab_core.py --basetemp <unique-OS-temp>
```

결과는 `16 passed in 0.32s`였다. 고유 OS 임시 디렉터리는 경로를 검증한 뒤 제거했고 repository cache나 bytecode를 남기지 않았다. 전체 pytest, 전체 AGI benchmark와 confirmation run은 실행하지 않았다.

## 3. 계약 2.2/4 API 표면

| 계약 역할 | 구현 매핑 | 판정 |
|---|---|---|
| immutable records/provenance | `records.py:22-129`, frozen record군 `132-569`, ledger `orchestrator.py:114-179` | 충족 |
| `WorldAdapter` | genesis-only protocol `contracts.py:94-95` | 충족 |
| `WorldModel` | `WorldModel(DynamicsPredictor)` `contracts.py:34-51`; `TabularWorldModel.predict/rollout` `procedural_world.py:398-435` | 충족 |
| `MemoryStore` | `BeliefEngine + OnlineLearner` protocol `contracts.py:26-31,54-61`; orchestrator의 단일 memory dependency `orchestrator.py:190-199,217,242,291` | 충족 |
| `Planner` | `CorePlanner` `contracts.py:64-73`; reference planner가 rollout을 호출 `procedural_world.py:459-492` | 충족 |
| `SafetyKernel` | `SafetyBoundary` `contracts.py:76-85`; authenticated reference kernel `procedural_world.py:137-190` | 충족 |
| `ActionExecutor` | WorldAdapter와 분리된 protocol `contracts.py:88-95`; orchestrator는 `_executor.execute`만 호출 `orchestrator.py:275-283` | 충족 |
| `AgentOrchestrator` | protocol-only `CoreOrchestrator` `orchestrator.py:7-34,182-324` | 충족 |
| counterfactual rollout | nonempty action sequence API `contracts.py:42-47`; 순수 multi-step reference rollout `procedural_world.py:414-435`; two-step fixture `tests/test_agi_lab_core.py:204-232` | 충족 |

`WorldAdapter`와 `ActionExecutor`는 nominal inheritance로 합치지 않았다. 같은 pure reference adapter가 두 protocol을 구조적으로 구현하지만 orchestrator dependency와 호출 권한은 분리되어 있다. `MemoryStore`와 `WorldModel`도 protocol 역할은 분리되어 있고 reference slice에서는 같은 stateless `TabularWorldModel` 객체를 각 역할로 주입한다.

## 4. PIC-I1--PIC-I5 매핑

| Claim | 코드 근거 | focused test 근거 | 독립 판정 |
|---|---|---|---|
| PIC-I1 | orchestrator는 `.contracts`와 immutable record만 가져오며 concrete family import가 없다(`orchestrator.py:7-34`). XOR와 SET reducer는 adapter 내부에만 있다(`procedural_world.py:193-201,350-380`). | 두 비동형 trace, 동일 orchestrator, opaque/poison proxy, forbidden source token 검사(`tests/test_agi_lab_core.py:235-253`) | 충족. 두 family에 대한 유한 구성 증인이지 모든 미래 adapter에 대한 보편 정리는 아니다. |
| PIC-I2 | genesis는 별도 lifecycle path이고(`orchestrator.py:208-235`), agent transition은 permit을 받는 executor만 호출한다(`268-324`). executor는 검증 뒤 새 session을 반환하며 입력을 바꾸지 않는다(`procedural_world.py:265-347`). | raw action·위조·stale permit 무전이(`307-353`), genesis 경계(`380-390`), reducer input byte 불변(`439-448`) | 충족. post-genesis agent transition 정의역으로 제한된다. |
| PIC-I3 | learner/planner 입력은 `CoreObservation`, `BeliefState`, action space와 model뿐이며 evaluator rule은 adapter private field다. agent record에는 truth/family/callback escape가 없다. | 동일 visible prefix에서 proposal과 갱신 후 public ledger bytes를 모두 비교(`255-280`); public field audit(`526-535`) | 충족. paired XOR/SET fixture의 visible-history noninterference 증인이다. |
| PIC-I4 | exact int64, finite float, NFC + strict UTF-8, tuple-only recursive grammar와 deterministic JSON을 사용한다(`records.py:22-129`). ledger digest는 schema/index/type/payload/previous digest를 포함하고 전 체인을 재검산한다(`orchestrator.py:114-179`). | Unicode/order/signed-zero canonicalization(`174-202`), 서로 다른 secret의 public ledger equality(`392-405`), full-state byte replay와 downstream rehash propagation(`406-438`) | 충족. 동일 full initial state/history/code/schema라는 조건부 replay 주장이다. |
| PIC-I5 | permit claim은 episode/tick/world/session/action-space/policy/proposal/action/nonce에 결합되고 HMAC 인증된다(`records.py:367-418`, `procedural_world.py:76-134`). executor는 permit·session·자기 world commitment를 함께 검사하고 nonce를 다음 immutable session에 기록한다(`procedural_world.py:265-347`). | adversarial forbidden/fallback(`282-305`), 모든 binding 변형·linear-history replay·두 cross-world substitution(`307-353`), terminal authenticated permit(`354-379`) | 충족. 현재 linear history의 single-use이며 old-snapshot anti-rollback은 미주장이다. |

## 5. 불변 record와 deterministic serialization

모든 공개 runtime record는 `@dataclass(frozen=True)`이고 nested field는 exact type과 tuple grammar로 다시 검증된다. `frozen=True`만 믿지 않고 list, mutable nested value, custom numeric coercion, bool-as-int, nonfinite float와 duplicate key를 거부한다. 문자열은 NFC로 정규화한 뒤 strict UTF-8 encoding 가능성을 검사하므로 lone surrogate가 record 안에 들어가 뒤늦게 serialization을 깨뜨릴 수 없다(`records.py:45-71`; `tests/test_agi_lab_core.py:174-192`).

Canonical map은 unique NFC key를 UTF-8 byte order로 정렬하고, belief transition과 used nonce도 canonical order와 uniqueness를 강제한다(`records.py:77-98,237-244,298-318`). `canonical_bytes`는 NaN 금지, compact separator, sorted object keys, UTF-8을 고정한다(`116-129`). 따라서 현재 grammar 안에서 mutable alias, process hash order와 signed-zero 차이가 public bytes에 스며들지 않는다.

`ActionPermit.authentication_tag`는 private metadata로 public canonical serialization에서 제외된다(`records.py:379`; `_jsonable` `101-110`). 실행 ledger에는 permit 전체나 secret을 넣지 않고 public claim digest와 verification result만 넣는다(`orchestrator.py:298-312`). 이것이 서로 다른 session key의 기능 trace가 같은 public ledger를 만드는 이유이며 fixture가 직접 확인한다.

## 6. protocol 결과 fail-closed 검증

교체 가능한 protocol의 signature만 맞고 의미가 모순되는 결과를 그대로 ledger에 기록하지 않는다.

- Genesis validator는 exact record type, request-observation, session state/goal/action space, policy, empty nonce, live flag와 zero transition count를 교차검증한다(`orchestrator.py:40-63`).
- Transition validator는 episode/tick/state/goal, immutable world/action-space/policy commitment, 정확히 추가된 permit nonce, transition count, terminal/goal flag를 session과 `WorldStep` 사이에서 교차검증한다(`66-93`).
- Memory validator는 previous/new model version과 exact experience digest를 belief/experience에 대조한다(`96-111`). Planner의 proposal model version도 current inferred belief와 같아야 한다(`251-256`).
- `CoreRuntimeState` 자체도 session과 observation의 episode/tick/state/goal 일치를 강제한다(`records.py:505-530`).

Broken genesis, executor와 memory fixture는 이 세 경계가 모두 fail-closed이고 입력 runtime bytes가 보존됨을 확인한다(`tests/test_agi_lab_core.py:450-523`). 이 검증은 bundled pure executor를 전제로 한다. 임의의 악성 Python 객체가 protocol 호출 전에 외부 I/O side effect를 내는 것까지 되돌리는 process sandbox 주장은 아니다.

## 7. 검토 중 발견되어 최종 snapshot에서 해소된 항목

| 최초 심각도 | 발견 | 최종 해소 근거 |
|---|---|---|
| P0 | permit/session의 world commitment를 현재 executor world와 대조하지 않아 cross-world reducer substitution 가능 | 삼중 equality `permit == session == self world` (`procedural_world.py:270-280`)와 원본 foreign session regression (`tests/test_agi_lab_core.py:345-352`) |
| P1 | 동일 visible prefix의 learner-update 후 ledger equality assert 부재 | `tests/test_agi_lab_core.py:268-273` |
| P1 | accepted record에 UTF-8 비직렬화 surrogate가 들어갈 수 있음 | strict UTF-8 grammar와 두 rejection fixture (`records.py:45-71`; test `189-192`) |
| P1 | tamper detection만 있고 변경 지점 이후 rehash propagation 증거 부재 | 두 event sequence를 재구성해 이후 digest 전부 불일치 확인(`tests/test_agi_lab_core.py:421-436`) |
| P1 | raw action 무전이 fixture 부재 | exact-type rejection과 state bytes/count 불변(`tests/test_agi_lab_core.py:324-333`) |
| P1 | `MemoryStore`, `ActionExecutor`, counterfactual rollout의 명시 API 공백 | 분리 protocol과 sequence rollout(`contracts.py:34-95`; `tests/test_agi_lab_core.py:204-232`) |
| P1 | 외부 protocol이 상호 모순된 session/step/receipt를 반환해도 수용 | validators와 adversarial fail-closed fixture(`orchestrator.py:40-111`; test `450-523`) |

최종 hash 기준으로 위 항목은 모두 닫혔다. 열린 P0/P1은 없다.

## 8. 주장 경계와 남는 제한

1. Permit single-use는 반환된 다음 `WorldSession`으로 전진하는 동일 linear history에서 성립한다. 과거 immutable snapshot을 다시 넣어 별도 deterministic branch를 만드는 rollback/fork 저항성은 계약이 명시적으로 제외한다. 전역 anti-rollback에는 외부 durable custody가 필요하다.
2. `world_instance_id`, authority key와 protocol object wiring은 trusted provisioning 입력이다. 이 certificate는 악성 Python process 전체의 reflection, object graph 탈취나 외부 side effect를 막는 보안 sandbox가 아니다.
3. Protocol parametricity는 현재 two-family witness와 bundled reference composition에서 검사됐다. 임의의 미래 protocol implementation의 purity나 noninterference를 자동으로 증명하지 않는다. 새 adapter/model/memory/executor는 같은 adversarial boundary suite를 다시 통과해야 한다.
4. PIC-H1--PIC-H5, unseen-family confirmation, 리만/SCC/언어/self-model 연결, AGI·의식·뇌 알고리즘 주장은 이번 구현 산출에 포함되지 않는다.

## 9. 최종 판정

고정된 여섯 파일은 계약이 승인한 V0 pure event-sourced vertical slice를 구현한다. PIC-I1--PIC-I5의 좁은 구현 certificate와 계약 2.2/4의 명시 API 역할은 서로 일치하며, focused test와 정적 API 감사에서 열린 P0/P1은 없다. 후속 문서는 이 결과를 `PHYSICS_INDEPENDENT_CORE_SCAFFOLD`의 `[산출]`로만 인용해야 하며 AGI 또는 상위 가설의 confirmation으로 합산해서는 안 된다.
