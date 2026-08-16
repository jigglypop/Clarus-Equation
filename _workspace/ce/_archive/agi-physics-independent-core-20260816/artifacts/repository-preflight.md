Status: COMPLETE

# 물리 독립 AGI 코어 V0 저장소 preflight

조사 기준일: 2026-08-16  
조사 범위: 저장소 지침, 현재 작업 트리, 승인된 신규 파일 경로, 기존 AGI·Phase A·SCC·기하·언어·기억·runtime 자산의 의존 경계  
변경 범위: 이 문서 한 개만 신규 작성했다. 제품 파일, `_workspace/ce/.active-run`, 기존 run, 봉인물, pytest 잔여물은 수정·삭제·재실행하지 않았다.

## 1. 최종 판정

계약이 승인한 `reality_stone.clarus.agi_lab` 신규 6파일 경로는 현재 checkout에서 모두 다음 조건을 만족한다.

1. 동일 경로의 파일이나 디렉터리가 존재하지 않는다.
2. Git tracked 경로와 대소문자 비구분 충돌이 없다.
3. repository ignore 규칙에 걸리지 않는다.
4. 현재 run 밖의 제품·테스트·문서에는 `agi_lab` 또는 `test_agi_lab_core` 참조가 없다.
5. `reality_stone.clarus.__init__`에는 `agi_lab` 재수출이 없다.

따라서 **경로·추적·명명 관점에서는 충돌 없이 신규 구현을 시작할 수 있다.** 형식 감사 `20-audit.md`는 `Status: COMPLETE`, `Gate: PASS`이고 hook도 `OK gate`를 반환했다. 그러므로 정확히 승인된 6파일의 구현을 시작할 수 있지만, 이 preflight와 Gate는 PIC-I1--PIC-I5 구현 통과를 뜻하지 않는다.

## 2. 최신 run 상태

현재 active pointer는 `_workspace/ce/agi-physics-independent-core-20260816`을 가리킨다(`_workspace/ce/.active-run:1`). 최신 계약은 `Status: COMPLETE`, Phase A predecessor, `full scoped implementation`을 명시한다(`00-contract.md:3-7`).

`run.cmd status`와 `check lanes`로 확인한 최신 상태는 다음과 같다.

| 단계 | 상태 |
|---|---|
| `00-contract.md` | COMPLETE |
| `10-sources.md` | SKIPPED |
| `11-math.md` | COMPLETE |
| `12-routes.md` | COMPLETE |
| `check lanes` | OK |
| `20-audit.md` | COMPLETE, Gate PASS |
| `check gate` | OK |
| 구현·검증·최종 보고 | MISSING |

수학 레인은 좁은 코어가 구성 가능하지만 초기 판정문에 대한 반례와 필요한 조건을 식별했다(`11-math.md:11-20`). 계약은 이를 반영해 pure event-sourced route, genesis 경계, authenticated single-use permit, full-state replay, public-ledger/secret 분리를 채택했다(`00-contract.md:88-102`). routes 레인도 R1 pure event-sourced core를 1순위로 판정했고 아직 구현은 없다고 명시한다(`12-routes.md:23-28`, `12-routes.md:142-146`).

## 3. 적용 저장소 지침

루트에는 `AGENTS.md`가 없으며, 관련 지침은 `.codex/AGENTS.md`와 `ex-codex/AGENTS.md`다. 새 제품 경로는 `ex-codex/` 밖이므로 직접 적용할 하위 전용 지침은 없다. CE 작업에는 다음 전역 규율을 적용한다.

- 신규 주장·승격·논문화는 CE research run으로 처리한다(`.codex/AGENTS.md:17-18`).
- predecessor가 감사한 결론은 재유도하지 않는다(`.codex/AGENTS.md:42-45`).
- 기본 검사는 변경 파일에 직접 연결된 최소 검사 한 개다(`.codex/AGENTS.md:49-58`).
- Python은 bytecode와 pytest cache를 끄고 고유 OS basetemp를 정리한다(`.codex/AGENTS.md:54-57`).
- 실패한 부모 주장을 통과로 표현하거나 기계 PASS를 이론 지위로 확대하지 않는다(`.codex/AGENTS.md:67-72`).

계약의 검증 범위도 focused test, 신규 파일 Ruff, 메모리 compile, scoped `git diff --check`, CE build gate로 한정한다(`00-contract.md:135-145`). 전체 pytest, 전체 AGI benchmark, Phase A one-shot과 confirmation은 이번 구현 범위가 아니다.

## 4. 작업 트리 보존 경계

조사 시점 `git status --short --branch`에는 다음 사용자·선행 작업 상태가 있었다.

- tracked 수정: `_workspace/ce/.active-run`;
- untracked: 현재 run 디렉터리;
- untracked: 기존 repository audit와 frontier comparison 아래 pytest basetemp 5개 디렉터리.

제품 경로인 `reality_stone/`, `tests/`, `docs/`, `experiments/`의 기존 tracked 파일 수정은 표시되지 않았다. 위 dirty/untracked 항목은 이 preflight의 소유물이 아니며 수정·stage·삭제하지 않았다. 특히 잔여 pytest 경로는 별도 정리 작업의 대상일 수 있으나 이번 구현이 그것을 재사용하거나 광역 glob으로 제거해서는 안 된다.

`.gitignore`는 `__pycache__`, `.pytest_cache`, `.pytest_tmp_*`, 루트 `.tmp`를 제외한다(`.gitignore:1-12`). 현재 잔여물의 이름은 이 규칙과 일치하지 않아 untracked로 보인다. 신규 focused test는 저장소 안에 고정 basetemp를 만들지 않고 OS 임시 경로를 소유·정리해야 한다.

## 5. 승인 6경로 충돌 감사

계약이 허용한 경로는 `00-contract.md:122-133`의 정확한 여섯 개뿐이다. 각 경로에 대해 `Test-Path`, exact `git ls-files`, tracked 경로의 case-insensitive equality, `git check-ignore`를 확인했다.

| 승인 경로 | 존재 | tracked | ignored | 대소문자 tracked 충돌 | 판정 |
|---|---:|---:|---:|---:|---|
| `reality_stone/python/reality_stone/clarus/agi_lab/__init__.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |
| `reality_stone/python/reality_stone/clarus/agi_lab/records.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |
| `reality_stone/python/reality_stone/clarus/agi_lab/contracts.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |
| `reality_stone/python/reality_stone/clarus/agi_lab/orchestrator.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |
| `reality_stone/python/reality_stone/clarus/agi_lab/procedural_world.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |
| `tests/test_agi_lab_core.py` | 아니오 | 아니오 | 아니오 | 아니오 | 신규 생성 가능 |

부모 디렉터리 `reality_stone/python/reality_stone/clarus/`와 `tests/`는 존재하는 디렉터리다. 현재 run을 제외한 저장소 검색에서 `agi_lab`과 `test_agi_lab_core` 참조는 0건이었고, `reality_stone/python/reality_stone/clarus/__init__.py`에도 `agi_lab` export가 없다.

### 5.1 Python 부모 package 주의점

경로 충돌은 없지만 Python의 정상 submodule import는 먼저 `reality_stone.clarus.__init__`을 실행한다. 현재 initializer는 `agent`, `nested_scc_tower`, `unified_metric` 등 여러 기존 모듈을 optional import하고 `ImportError`를 숨긴다(`reality_stone/python/reality_stone/clarus/__init__.py:280-321`, `:378-419`).

따라서 이번 계약의 “V0가 기존 runtime을 직접 import하지 않는다”는 보장은 다음처럼 좁게 시험해야 한다.

- `agi_lab` 네 구현 파일 안에서 기존 `runtime`, `agent`, SCC, metric, language, memory module을 import하지 않는다.
- root `clarus.__init__`에 신규 import나 `__all__` 항목을 추가하지 않는다.
- `agi_lab/__init__.py`는 local V0 surface만 노출하며 root package side effect를 확대하지 않는다.
- full-suite import isolation이나 기존 initializer 정리는 별도 범위다.

이는 구현을 막는 경로 충돌은 아니지만, “새 코어가 기존 runtime과 완전히 process-isolated하다”는 더 강한 주장을 허용하지 않는다.

## 6. 기존 자산 registry 초기안

분류 축은 lifecycle과 허용 주장이다. 구현되어 있다는 사실과 과학적 효능은 분리한다.

| 자산 | lifecycle | 이번 V0 허용 용도 | 금지·경계 근거 |
|---|---|---|---|
| `00-contract.md`, `11-math.md`, `12-routes.md`, `20-audit.md` | `canonical` | V0 타입·권한·정보·재현 계약과 구현 승인 범위 | 구현 전 PIC-I1--PIC-I5 통과로 표현 금지 |
| Phase A 최종 보고서·봉인 bytes | `canonical predecessor` | known-identity synthetic development의 지위와 no-go 인용 | confirmation·기억·AGI로 확대 금지; Phase A 보고서 `:6-12`, `:168-182` |
| `causal_recurrent_geometry_benchmark.py`와 runner/test/manifest | `experimental predecessor` | 후속 characterization 설계 참고 | V0 직접 import·수정·재실행 금지; `00-contract.md:109-120` |
| `scc_atlas.py`, `brain_scc_study.py`, nested SCC 계열 | `experimental` | 후속 SCC adapter 후보 | finite SCC와 생물학 identity를 분리; `scc_atlas.py:1-11`, `brain_scc_study.py:1-6` |
| `unified_metric.py`, `covariant_metric_flow.py`, `brain_geometry_benchmark.py` | `experimental` | 후속 control/geometry arm과 baseline 후보 | world direction·AGI·biology 증거 아님; `unified_metric.py:1-17`, `brain_geometry_benchmark.py:1-5` |
| `neural_language_gate.py`, `neural_language_blind_gate.py` | `experimental method-control` | 후속 DSL·composition 음성대조 설계 | biological/full-brain claim은 false lock; `neural_language_gate.py:16-31`, `neural_language_blind_gate.py:29-33` |
| `episodic_memory.py`, `local_memory.py`, memory benchmarks | `experimental` | 후속 memory adapter와 abstention baseline | AGI·인간 기억기전으로 확대 금지; `episodic_memory.py:1-46`, `local_memory.py:1-7` |
| 같은 graph에 동일 edge semantics로 반복 SCC를 적용해 비자명 hierarchy를 얻는 부모 주장 | `negative` | no-go 회귀 fixture | 첫 condensation 뒤 DAG; CRGM 보고서 `:70-90` |
| 정적 SPD metric만으로 방향·부호·유일 goal을 얻는 부모 주장 | `negative` | metric baseline kill test | metric 대칭성 no-go; CRGM 보고서 `:47-53`, `:92-103` |
| V9 실패 architecture와 sealed seed 재사용 | `negative` | 실패 이력·회귀 근거 | 재사용 금지; CRGM 보고서 `:235-244` |
| `agent.py`의 `ConsciousnessMonitor`를 의식 깊이로 해석하는 주장 | `negative` | 이름과 해석을 제거한 안정성 baseline만 후속 검토 | 구현은 active-ratio deviation의 지수변환; `agent.py:173-215` |
| `runtime.py`, `agent.py`, `sleep.py`, 기존 `docs/7_AGI` 설계 서사와 `_archive/agi-*` | `legacy` | provenance·historical comparison | V0 dependency 금지; 계약 `:109-120` |

기존 CRGM 12주 계획은 폐기 대상이 아니라 후속 연구 트랙이다. 이 계획은 Phase A→matched SCC compression→Gramian memory를 순서대로 시험하고, 의식·자아를 endpoint에서 제외한다(`agi-connectome-geometric-memory-20260816/artifacts/twelve-week-roadmap.md:5-29`). V0는 이 트랙보다 먼저 권한·정보·재현 scaffold를 닫는다.

## 7. 첫 구현 patch의 정확한 경계

Gate가 통과했으므로 첫 patch는 승인된 6파일만 추가하고 기존 byte는 수정하지 않는다.

### 7.1 권고 의존 DAG

```text
records.py
   ↓
contracts.py
   ├──────────────┐
   ↓              ↓
procedural_world.py   orchestrator.py
          \          /
           test_agi_lab_core.py

agi_lab/__init__.py → local V0 public names only
```

- `records.py`: standard library만 사용한다. recursive immutable value grammar, canonical serialization, digest, genesis/action/proposal/permit/ledger record를 둔다.
- `contracts.py`: `typing.Protocol`과 `records`만 사용한다. adapter, model, memory, planner, safety, executor 경계를 정의하고 concrete family를 import하지 않는다.
- `procedural_world.py`: 두 구조적으로 다른 finite rule family와 opaque adapter를 둔다. evaluator truth projection과 learner observation을 다른 타입으로 유지한다.
- `orchestrator.py`: pure tick assembly, permit issue/verify/consume, reducer 진입, public ledger projection을 둔다. family id, concrete `isinstance`, `type`, `hasattr`, reflection dispatch를 금지한다.
- `agi_lab/__init__.py`: root package 수정 없이 local surface만 제한적으로 노출한다.
- `test_agi_lab_core.py`: PIC-I1--PIC-I5와 수학 레인의 kill test를 한 파일에 고정한다.

### 7.2 구현 필수 불변식

수학·routes 레인에 따라 다음은 선택 사항이 아니다.

1. canonical value grammar는 callback, arbitrary object, mutable alias, NaN/Inf를 거부하고 `bool`을 `int`로 받지 않는다(`11-math.md:30-49`).
2. learner/evaluator 경계는 field-name 검사보다 강한 visible-history noninterference로 시험한다(`11-math.md:51-80`).
3. permit은 episode, tick, world, action space, policy, proposal, nonce와 authentication에 결합하고 single-use여야 한다(`11-math.md:82-113`).
4. replay domain에는 전체 world/agent/permit/RNG 상태, history, code/schema commitment가 포함돼야 한다(`11-math.md:115-141`).
5. secret authentication tag는 public ledger에 기록하지 않고 claim digest와 검증 결과만 투영한다(`00-contract.md:90-92`, `12-routes.md:78-86`).
6. invalid permit은 reducer 호출과 world transition count를 모두 바꾸지 않는다.
7. genesis/reset은 agent-initiated post-genesis action theorem과 분리한다.

## 8. focused 검증 범위

첫 patch 뒤 허용되는 검증은 계약에 고정된 최소 범위뿐이다.

1. `tests/test_agi_lab_core.py` 단독 pytest;
2. 승인된 신규 Python 5파일과 신규 test에 한정한 Ruff `--no-cache`;
3. source를 메모리에서 `compile()`하는 syntax 검사;
4. 승인 6경로에 대한 `git diff --check`;
5. 이후 CE build gate.

pytest는 `python -B`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, OS 임시 basetemp와 `finally` 정리를 사용한다. 기존 잔여 basetemp를 재사용하거나 지우지 않는다. 전체 pytest·AGI benchmark·Phase A 실행은 생략하며, 생략은 실패가 아니다.

## 9. 현재 gate와 구현 상태

| 항목 | 현재 상태 | 재개·승격 조건 |
|---|---|---|
| 저장소·경로 preflight | COMPLETE | 이 문서로 충족 |
| contract와 lanes | COMPLETE / OK | 현재 bytes 유지 |
| 형식 감사 | COMPLETE, Gate PASS | 승인 범위는 계약의 신규 6파일로 고정 |
| hook gate | OK | 현재 감사·계약 bytes 유지 |
| 구현 | 미착수 | 승인 6파일만 추가 가능 |
| PIC-I1--PIC-I5 | 산출 후보 | focused adversarial test와 독립 검증 |
| PIC-H1--PIC-H5 | 미완성 | V0에서는 승격하지 않음 |

최종 결론은 **“신규 `agi_lab` 6경로는 충돌 없이 사용 가능하고 형식 감사 Gate도 통과했으므로, 승인된 최소 patch를 시작할 수 있다”**이다. 이는 AGI, 의식, 뇌 알고리즘, SCC 또는 리만기하의 효능 판정이 아니다.
