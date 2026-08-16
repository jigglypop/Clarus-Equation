# 물리 독립 AGI 코어 V0 형식 지위 감사

Status: COMPLETE

Gate: PASS

## 1. 감사 범위와 입력 완결성

감사 대상은 업데이트된 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`, `artifacts/verify_core_contract.py`와 `artifacts/procedural-universe-v1-blueprint.md`다. `00-contract.md`, `11-math.md`, `12-routes.md`는 모두 `Status: COMPLETE`이고, 외부 관측값을 사용하지 않는 이번 계약에서 physics-sourcer를 열지 않은 `10-sources.md:1-3`의 skip 사유는 적합하다.

이 감사의 PASS는 V0의 좁은 구현을 시작해도 된다는 뜻이다. 아직 제품 구현이 없으므로 PIC-I1--PIC-I5가 성립했다는 판정, 성능 통과, AGI 증거 또는 V1 confirmation 승인은 아니다. 수학 레인도 이를 명시한다(`11-math.md:7-9`).

## 2. 최종 판정 요약

`11-math.md`가 찾은 다섯 후보 반례 가운데 네 개는 업데이트된 계약에서 주장 정의역과 판정조건을 좁혀 해소됐다. plain frozen permit 반례는 authenticated single-use permit과 pure reducer 경로를 구현 조건으로 선택하여 배제했다. 따라서 현재 활성 문서에는 열린 P0가 없다.

선택된 R1 pure event-sourced route는 가능한 세 경로 중 최소 authority surface를 택한 **[공리: 구현 경로 선택]**이다. numerical dof가 0이라는 사실은 R1의 유일성이나 우월성을 정리로 만들지 않는다. `12-routes.md:27`, `12-routes.md:144-146`에 따라 R1은 형식적으로 구성 가능한 1순위일 뿐 아직 구현되지 않았다.

PIC-I1--PIC-I5는 모두 **[산출 후보]**로 정확히 낮춰져 있다(`00-contract.md:98-102`). 이들은 아래 승인 파일에서 구현되고 focused test가 통과한 뒤에만 해당 구현의 **[산출]**로 기록할 수 있다. PIC-H1--PIC-H5는 모두 **[미완성]**이며 V0 구현으로 승격할 수 없다(`00-contract.md:103-107`, `00-contract.md:153-154`).

## 3. 반례 폐쇄 원장

| 반례 | 기존 넓은 문구의 실제 지위 | 업데이트된 좁은 문구 | 감사 판정 |
|---|---|---|---|
| 같은 orchestrator를 두 family에 사용해도 concrete-type branch 가능 | 완전 반례 | protocol-only, concrete type·family id·reflection 분기 금지, 동일 source와 opaque proxy를 요구함(`00-contract.md:98`) | P0 해소. 구현에서 import/AST·poison proxy kill test 필수 |
| permit 없는 reset도 world를 변경함 | 완전 반례 | genesis/reset을 별도 lifecycle event로 분리하고 결론을 그 이후 agent-initiated transition으로 한정함(`00-contract.md:36`, `00-contract.md:90`, `00-contract.md:99`) | P0 해소. reset은 같은 episode의 숨은 mutation일 수 없음 |
| `truth`라는 field가 없어도 alias·closure로 truth 누수 가능 | 완전 반례 | field-name 검사를 충분조건으로 쓰지 않고 paired hidden-world dynamic noninterference를 요구함(`00-contract.md:86`, `00-contract.md:100`) | P0 해소. 구현에서 동일 visible prefix의 proposal·ledger byte equality 필수 |
| 같은 관측·seed·config라도 memory가 다르면 ledger가 달라짐 | 완전 반례 | 초기 world/agent/permit/RNG state, visible history, code·schema를 replay domain에 포함함(`00-contract.md:73`, `00-contract.md:101`) | P0 해소. full-state replay로만 주장 가능 |
| frozen permit은 생성·replace·replay 가능 | 완전한 잘못된 구현 후보 | episode·tick·world·action-space·policy·proposal 결합, authentication과 single-use를 요구하고 verified permit만 reducer에 전달함(`00-contract.md:90`, `00-contract.md:102`) | P0 해소. nominal dataclass만으로 구현하면 즉시 기각 |

`verify_core_contract.py:77-103`은 reset과 naive permit 반례를, `verify_core_contract.py:193-218`은 제한된 authenticated permit 구성 증인을 제공한다. 이 scratch verifier는 제품 모듈을 import하지 않으며 proposal/action-space binding, terminal liveness와 pure event-sourced 제품 구현 전체를 검사하지 않는다. 따라서 그 출력은 반례와 유한 구성 가능성의 **[산출]**이지 PIC-I1--PIC-I5의 구현 certificate가 아니다.

## 4. Claim별 형식 지위

| Claim ID | 계약 지위 | 감사 후 실제 지위 | 근거와 승격 조건 |
|---|---|---|---|
| PIC-I1 | [산출 후보] | [산출 후보], 지위 정합 | 비동형 두 family의 존재 증인은 `11-math.md:146-176`에 있으나 universal adapter 독립 정리는 아니다. 동일 source, protocol-only import boundary, opaque wrapper와 poison proxy test가 모두 필요하다 |
| PIC-I2 | [산출 후보] | [산출 후보], 지위 정합 | post-genesis agent transition만 정의역이다. Genesis/reset과 autonomous evaluator event는 별도 lifecycle event여야 하며 raw/invalid event에서 world state가 불변이어야 한다 |
| PIC-I3 | [산출 후보] | [산출 후보], 지위 정합 | 이름 검사가 아니라 식 (4)의 noninterference가 기준이다(`11-math.md:71-80`). Paired hidden truth에서 visible history와 learner state가 같으면 proposal과 public learner-ledger bytes가 같아야 한다 |
| PIC-I4 | [산출 후보] | [산출 후보], 지위 정합 | full initial state와 결정적 tick function 아래의 ledger equality만 조건부 정리다(`11-math.md:124-139`). session secret은 public ledger projection에서 제외한다(`00-contract.md:92`) |
| PIC-I5 | [산출 후보] | [산출 후보], 지위 정합 | SafetyKernel 선호나 nominal permit type이 아니라 bound authenticated single-use permit과 verified-event constructor 경계가 필요하다. terminal 이후 permit도 거부해야 한다(`11-math.md:231`) |
| PIC-H1 | [미완성] | [미완성], 지위 정합 | V1 unseen-family preregistration, matched baseline과 fresh family-level confirmation이 없음 |
| PIC-H2 | [미완성] | [미완성], 지위 정합 | 리만·제어기하 구현, 대조군과 metric-null kill test가 없음 |
| PIC-H3 | [미완성] | [미완성], 지위 정합 | SCC adapter, 압축 대조군과 기억·계획 endpoint가 없음 |
| PIC-H4 | [미완성] | [미완성], 지위 정합 | 언어/DSL의 인과적 상태전달, negative control과 행동 endpoint가 없음 |
| PIC-H5 | [미완성] | [미완성], 지위 정합 | `SelfHypothesis`의 행동 기반 calibration·corrigibility 평가가 없음 |

PIC-H1--PIC-H5가 미완성이라는 사실은 V0 gate의 결함이 아니라 의도적으로 제외한 주장 범위다. R1--R3 어느 route도 이 가설을 올려주지 않는다는 `12-routes.md:140-146`의 판정과 일치한다.

## 5. R1 pure event-sourced route의 승인 조건

R1의 새 공리는 다음 문장으로 고정한다.

> 모든 post-genesis world/agent 변화는 명시적인 이전 immutable state와 authorized event의 pure function이다.

이 공리 아래 V0 구현은 다음 불변식을 모두 만족해야 한다.

1. World state와 learner-visible record는 재귀적으로 immutable하고 canonicalizable한 값만 가진다. `frozen=True`는 충분조건이 아니다.
2. `Genesis`는 episode의 최초 lifecycle event다. Reset이 필요하면 새 episode identity와 새 genesis를 만들거나 명시적인 lifecycle event로 기록해야 하며 같은 episode state를 몰래 덮어쓸 수 없다.
3. Post-genesis agent action은 raw action이 아니라 검증에 성공한 event만 pure reducer에 도달한다.
4. Forged, stale, cross-world, wrong-action-space, wrong-policy, wrong-proposal, replay와 post-terminal permit은 reducer 호출 횟수와 world state를 바꾸지 않는다.
5. Rejection을 public ledger에 남기기로 선택했다면 world state는 불변이고 rejection entry만 결정적으로 추가돼야 한다. 남기지 않기로 선택했다면 ledger tail도 불변이어야 한다. 둘을 실행 중 혼용할 수 없다.
6. Orchestrator는 procedural family module, concrete type, class name, family id, extra attribute와 hidden transition callback을 읽지 않는다.
7. Evaluator truth는 observation projection 외의 alias, callback, closure, global 또는 shared mutable reference로 learner에 도달하지 않는다.
8. Public ledger에는 authentication secret/tag를 기록하지 않고 permit claim digest와 verification result만 기록한다. 서로 다른 session key에서 기능 state와 event sequence가 같으면 public ledger projection도 같아야 한다.
9. Reducer와 tick function은 wall clock, global RNG, process-randomized iteration, object address, nondeterministic concurrency와 외부 mutable state를 읽지 않는다.
10. Canonical hash-chain에서 중간 entry 한 byte를 바꾸면 그 지점 이후 digest가 모두 달라져야 한다.

위 항목은 구현 후 검증할 조건이며 현재 이미 성립했다고 기록하지 않는다.

## 6. Genesis/reset 예외의 정확한 범위

PIC-I2의 허용 결론은 다음뿐이다.

$$
\text{post-genesis agent-initiated transition}
\Longrightarrow
\text{verified authorized event through the pure reducer}.
$$

Genesis는 이 함의의 정의역 밖에 있는 최초 lifecycle event다. Reset은 다음 둘 중 하나로만 허용한다.

- 이전 episode tail을 닫고 새로운 episode id와 `Genesis`를 시작한다.
- 별도 lifecycle authority가 발행한 reset event를 event log에 명시한다.

Reset, evaluator intervention 또는 autonomous event를 일반 agent action처럼 세어 PIC-I2의 결론을 넓히거나, 반대로 lifecycle이라는 이름으로 같은 episode의 숨은 mutation을 허용해서는 안 된다. `12-routes.md:37-46`의 R1 정의가 이 경계를 충족한다.

## 7. V1 blueprint 판정

`artifacts/procedural-universe-v1-blueprint.md`는 문서 자체가 `V1 PREREGISTRATION BLUEPRINT ONLY`, `V0 implementation scope: OUTSIDE`라고 명시한다(`procedural-universe-v1-blueprint.md:5-9`). 따라서 V0 구현 승인 파일에 포함되지 않고 PIC-H1을 승격하지 않는다.

Blueprint의 family schema, split, baseline, primary metric, confirmation threshold와 kill test는 후속 V1에서 고정할 **[공리: 평가 설계]** 및 조건부 **[예측 초안]**이다. 현재 empirical 결과가 아니다. Exact confirmation gate가 적혀 있어도 아직 generator, evaluator, checkpoint, custody와 one-shot 실행이 없으므로 `PROCEDURAL_UNIVERSE_V1_GO` 지위는 발생하지 않는다(`procedural-universe-v1-blueprint.md:403-419`).

V1 전용 P2가 하나 남는다. V0 계약의 $F=(\mathcal O,\mathcal A,T,R,H,\Lambda)$ 설명에서는 $H$를 숨은 상태 구조로 쓰지만 blueprint의 $U=(E,X,R,A,\mathcal T,\Omega,G,H)$에서는 $H$를 query horizon으로 쓴다. 서로 다른 튜플 안의 국소 기호라 논리 모순은 아니지만 후속 V1 계약에서는 `hidden_state_structure`와 `query_horizon`으로 풀어 써야 한다. 이는 V0 구현 gate를 막지 않는다.

## 8. AGI·의식·뇌 알고리즘 비승격

V0에서 허용되는 최상위 명칭은 `PHYSICS_INDEPENDENT_CORE_SCAFFOLD`다(`00-contract.md:40-46`). 다음 결론은 모두 활성 주장 수 0이다.

- AGI 달성 또는 운영적 AGI 확인;
- 인간 뇌의 근본 알고리즘 발견;
- 의식, 감정, 도덕적 지위 또는 아라한 상태 구현;
- 리만 계량 또는 SCC가 일반지능의 필수 구조라는 결론;
- Phase A development를 confirmation으로 재명명하는 결론.

불교의 계·정·혜·연기·무아·자비는 타입과 안전 설계의 **[가설적 공학 힌트]**일 뿐 종교적 참값이나 성취 판정이 아니다. `PrincipalIdentity`, `SelfHypothesis`, `Constitution`의 nominal type 분리는 V0 placeholder boundary일 뿐 의미적 무아, 자기의식 또는 corrigibility의 증거가 아니다.

V1 blueprint도 현실 물리, 로봇 grounding, 인간 뇌 알고리즘, AGI 전체와 의식에는 별도 계약과 confirmation이 필요하다고 명시한다(`procedural-universe-v1-blueprint.md:486-490`). 지위 정합이다.

## 9. 심각도와 계수 원장

| 항목 | 수 | 판정 |
|---|---:|---|
| 검사한 Claim ID | 10 | PIC-I1--PIC-I5, PIC-H1--PIC-H5 |
| 조건부 형식 정리/구성 명제 | 3 | full-state ledger replay, invalid-event 무전이, 두 비동형 family의 공통 protocol 구성 가능성 |
| 명시해야 할 핵심 구현 공리 | 8 | recursive immutability, pure reducer, lifecycle boundary, protocol parametricity, dynamic noninterference, authenticated single-use authority, deterministic state, public ledger projection |
| [산출 후보] | 5 | PIC-I1--PIC-I5 |
| [미완성] | 5 | PIC-H1--PIC-H5 |
| 열린 P0 | 0 | 수학 레인의 부모 문구 반례는 업데이트 계약에서 범위 축소로 해소 |
| 열린 P1 문서 결함 | 0 | 구현 부재는 후보 지위와 일치하며 build 단계의 검증 대상 |
| 열린 P2 | 1 | V1 전용 $H$ 기호 충돌; 후속 계약에서 해소 |
| 활성 부모 주장 삭제 | 0 | 거짓인 넓은 문구는 현재 계약에 활성화되어 있지 않음 |
| 반례로 좁힌 부모 문구 | 5 | adapter 독립, reset 전칭, field-name 경계, replay domain, nominal permit |

P0/P1이 없고 모든 활성 주장의 지위가 근거와 일치하므로 Gate는 PASS다. P2는 V1 후속 계약의 표기 정리이며 V0 build를 차단하지 않는다.

## 10. 승인 구현 파일 범위

형식 gate가 승인하는 제품 변경은 `00-contract.md:122-133`의 다음 여섯 파일로만 제한한다.

- `reality_stone/python/reality_stone/clarus/agi_lab/__init__.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/records.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/contracts.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/orchestrator.py`;
- `reality_stone/python/reality_stone/clarus/agi_lab/procedural_world.py`;
- `tests/test_agi_lab_core.py`.

이 승인은 다음을 포함하지 않는다.

- V1 blueprint의 generator, manifest, runner, baseline 또는 confirmation 구현;
- 대규모 모델, 외부 데이터 또는 root package export;
- 기존 Phase A 코드, 테스트, runner, manifest와 result byte의 수정·재실행;
- SCC, 리만/제어기하, 언어, self-model 또는 social module 연결;
- 전체 pytest, 전체 AGI benchmark 또는 confirmation 실행.

구현자는 위 여섯 파일 안에서 R1과 섹션 5의 kill test를 닫아야 한다. 범위 밖 파일이 필요해지면 구현을 확대하지 말고 계약과 형식 감사를 다시 열어야 한다.

## 11. Gate 결론

업데이트 계약은 수학 레인의 완전 반례를 숨기지 않고 좁은 구현 가능 명제로 교체했다. R1은 선택 공리로 정확히 표시됐고, reset/genesis 예외와 public ledger projection이 명시됐으며, PIC-H1--PIC-H5와 AGI·의식·뇌 알고리즘 주장은 미승격 상태다. 따라서 승인된 여섯 파일의 V0 pure event-sourced reference implementation으로 진행할 수 있다.

Gate: PASS
