---
name: ce-research
description: Clarus-Equation 연구를 출처·수학·대안 경로의 독립 레인, 형식 지위 감사, 구현·수치 검증, 최종 집필로 수행한다. CE/Clarus-EQ 식·가설·논문·코드의 유도, 반례, 재현, 승격 판단이나 병렬 연구를 요청할 때 사용한다. 단순 계산과 guard 제품 벤치만 필요한 요청에는 사용하지 않는다.
---

# CE Research

역할 카드는 4장(ce-physics-sourcer, ce-math-verifier, ce-status-auditor, ce-impl-engineer)이다. 해당 레인을 실행하는 순간에만 그 카드를 읽는다 — 모든 카드를 선로딩하지 않는다. 집필은 역할 카드 없이 아래 집필 규칙으로 직접 한다.

## 코어 명령

    <codex-home>/hooks/run.sh init|status|check|revise|gc ...        # POSIX
    <codex-home>/hooks/run.ps1 init|status|check|revise|gc ...       # Windows

- `init <run-dir>`: run 생성. `REUSE?` 목록이 나오면 같은 주제의 미완성 run을 새로 파지 말고 이어받는다.
- `status <run-dir>`: 단계별 상태·Gate·수정 카운트 한 화면. 현황 파악에 stage 파일을 재독하지 말고 이것을 쓴다.
- `check <run-dir> <contract|lanes|gate|build|final>`: 해당 단계까지의 전체 체인을 검사한다. gate는 20-audit.md의 `Gate: PASS`를 요구한다.
- `revise <run-dir> <role>`: 수정 루프 기록. 역할당 2회를 넘으면 거부되므로 주장 범위를 좁혀 살릴 것은 살리고, 남는 결함은 BLOCKED로 보고서에 남긴다.
- `gc <workspace>`: 완결 run을 `_archive/`로 이동하고 미완성 run을 나열한다. 연구 세션을 마칠 때 실행한다.

모든 산출물은 _workspace/ce/<run-id>/에 쓴다. run-id는 `<주제>-<YYYYMMDD>`. 원본은 역할 권한 밖에서 수정하지 않는다.

`.codex` 하네스·스킬·역할 카드의 문구, 실행 범위 또는 cache 정책을 고치는 요청은 이 연구 파이프라인의 대상이 아니다. 이 경우 run과 레인을 만들지 말고 대상 설정 파일만 spot 수정한다.

## 산출물 위생 (모든 역할 공통)

- run 루트에는 번호 stage 파일 8개만 둔다. 사전등록, 루프별 노트, 검증 스크래치 등 보조 문서는 전부 artifacts/에 만들고 stage 파일에서 경로로 인용한다. `check final`이 루트의 잡파일을 차단한다.
- stage 파일 이름을 바꾸거나 변형본(10-provenance.md, 30-build-validation.md 등)을 만들지 않는다. 내용이 다르면 절을 나눠 같은 파일 안에 쓴다.
- 수식은 블록 `$$ ... $$`(별도 줄), 인라인 `$...$`만 쓴다. `\[ \]`·`\( \)`는 렌더링되지 않으므로 금지. 수학 기호는 코드스팬이 아니라 수식 모드로 쓴다.
- stage check는 단계 전환마다 필요한 것만 한 번 실행한다. 마지막 `check final`이 선행 체인을 포함하므로 같은 byte에서 모든 check를 연속 반복하지 않는다.
- 구현 검증은 `$ce-validate`의 최소 범위 규율을 따른다. CE full research run이라는 이유만으로 전체 pytest·전체 bench를 실행하지 않는다.

## 모드 선택

- **full**: 신규 주장, 승격 판단, 논문화. 아래 6단계 전부.
- **light**: 완결된 선행 run의 후속·반복(v8→v9 등). 계약에 `PREDECESSOR: _workspace/ce/<선행-run>`을 적고, 레인은 math-verifier만 실행하며 감사(gate)와 최종 보고서만 유지한다. 선행 run이 이미 감사한 결론은 재유도하지 않고 경로만 인용한다.

## 뇌 알고리즘 new-run admission gate

### 실제 뇌 식 기반 발견 최우선 규약

뇌·기억·의식·AGI 연구는 `.codex/harnesses/real_brain_equation_discovery_loop.md`를 최우선 경로로 적용한다. 실제 뇌에서 확립된 기전식과 측정모형을 출발점으로 잠그고, CE 가설은 $F_{\mathrm{bio}}$에 대한 명시적 추가항 $\Delta F_{\mathrm{CE}}$, 새 상태 또는 경계조건으로 분리한다. 실제 뇌 데이터에서 생물 기준식 재현 → CE 식의 분리 적합 → held-out/개입 반증 → 잔차 분류 → 새 판본의 순서로 찾는다. L0 simulator 탐색은 실제 데이터 경로의 식별 가능성이나 장치를 진단할 때만 후순위로 연다.

뇌/AGI `00-contract.md`는 결과를 보기 전에 `BIO_STARTING_MECHANISM`, `CE_DELTA`, `MEASUREMENT_MODEL`, `DATA_PROVENANCE`, `DATA_SPLIT`, `OBSERVABLES`, `RESIDUAL_RULE`, `FALSIFIER`, `MATCHED_CONTROLS`, `MODEL_SELECTION`, `REVISION_TRIGGER`, `CLAIM_CEILING`을 고정한다. 핵심 기전식·분모·측정모형·데이터 provenance가 `UNVERIFIED`이면 채점 구현으로 진행하지 않는다. 데이터에 맞춘 식 개정은 `empirical_calibration_loop.md`에서 T 잔차가 확정된 뒤 한 판본 한 구조 변경으로만 수행하고, 구 식·반례·residual을 원장에 보존하며 새 독립 confirmation을 요구한다.

뇌·기억·의식·connectome 연구는 새 run을 열기 전에 다음 순서를 지킨다.

1. `_workspace/ce/brain-algorithm-route-ledger.md`와 직접 선행 run의 `12-routes.md`, `31-validation.md`, 존재하면 `40-final-report.md`만 먼저 읽는다. 40이 없으면 가장 늦은 numbered audit와 원장 행을 사용하고 closure 부재를 계약에 기록한다. 세부 artifact는 특정 수치나 hash를 검증할 때만 연다.
2. 00-contract에 `PREDECESSOR_EVIDENCE` 표를 만들고 각 결과를 artifact/hash, `PASS|STOP|APPARATUS_INVALID|BLOCKED`, 보존 가능한 좁은 주장, 재시도 금지 조건으로 고정한다.
3. 후보 집합을 적고 실제 기전식·데이터 연결 가능성, 사다리 승급 가능성, 인과 식별 가능성, capability dependency, 이전 STOP이 남긴 정보량, 독립 falsifier와 matched control 순으로 정렬한다. 양성처럼 보이는 수치나 구현 편의성만으로 고르지 않는다.
4. threshold·seed·endpoint·decoder만 바꾼 반복은 같은 실패 경로다. 새 mechanism, 새 개입 seam, 또는 실패를 가르는 새 adverse control이 없으면 후보에서 퇴역시킨다.
5. 선택 후보와 기각 후보를 결과 확인 전에 계약과 원장에 기록하고 math-verifier가 cross-run 일관성을 확인한 뒤 구현한다.

이 gate의 목적은 simulator에서 뇌의 알고리즘 후보를 더 잘 식별하는 것이다. simulator PASS를 실제 뇌, 기억 또는 의식의 생물학적 증거로 승격하지 않는다.

## 6단계 (full)

1. **계약**: 질문·정의역·주장·기호·허용 오차·PREDECESSOR를 00-contract.md에 고정하고 `Status: COMPLETE`로 닫는다.
2. **레인(조건부 팬아웃)**: 해당하는 레인만 실행한다.
   - physics-sourcer → 10-sources.md: 계약이 관측값·외부 데이터를 인용할 때만.
   - math-verifier → 11-math.md + 12-routes.md: 항상. 검산과 우회 경로 탐색을 한 레인에서 수행한다. routes는 계약에 Open/Hypothesis가 있거나 검산이 P0/막힘을 냈을 때 실행하고, 아니면 12에 `Status: SKIPPED (사유 한 줄)`.
   - 스킵한 레인은 파일에 `Status: SKIPPED (사유 한 줄)`만 쓴다. 각 레인 파일은 판정·표 중심으로 유지하고 상세 계산·로그는 artifacts/에 두고 경로만 인용한다.
3. **감사**: status-auditor가 20-audit.md에 형식 출처·반례를 기록하고 `Gate: PASS|REVISE|BLOCKED`를 판정한다.
4. **구현**: `check <run-dir> gate` 통과 후 승인된 범위만 impl-engineer가 구현하고 $ce-validate로 검증한다(30/31). 코드 변경이 없으면 `Status: SKIPPED (사유)`.
5. **수정 루프**: P0/P1은 `revise <run-dir> <role>`로 기록한 뒤 지목된 역할만 고친다. 한도 초과 시 주장 범위를 좁혀 살릴 수 있는 좁은 정리를 분리하고, 남는 결함만 BLOCKED로 보고서에 남긴다.
6. **집필**: 40-final-report.md를 직접 작성한다. $ce-doc-write 규약(지위 태그·완전 반례 처리·문체·수식 표기)을 따르고 아래 집필 규칙을 지킨다. `check <run-dir> final` 통과를 확인한다.

run이 끝나면 오케스트레이터는 원장의 해당 후보 상태·증거 경로·다음 falsifier를 먼저 갱신하고, root/main agent에게 정확한 변경 경로와 검증 결과를 인계한다. subagent와 역할 agent는 stage/commit/push하지 않는다. 사용자가 발행을 지시한 경우에만 main이 루트 Git 인계 규율에 따라 commit/push한다.

## 집필 규칙 (40-final-report.md)

- 어떤 판정도 강화하지 않는다 — 레인이 경험식이라 한 것을 산출로, BLOCKED를 "사실상 해결"로 바꿔 쓰지 않는다. 새 증거가 필요하면 쓰지 말고 담당 레인으로 돌려보낸다.
- 구성: 초록(6문장 이내: 배경·문제·방법·핵심 수치·형식 지위·한계) → 서론 → 정의·표기 → 공리(번호+하위분류) → 정리·증명 → 산출 → 관측 비교(중립 서술) → 미완성 과제와 한계(미해결 P0/P1·BLOCKED 그대로) → 재현성(명령·경로) → 참조(식별자+접근 날짜, 2차 인용 금지).
- 본문은 완결된 문단 산문으로 쓴다. 자기완결적으로 — 독자가 레인 파일을 열지 않고도 논증을 따라오게 핵심 유도를 본문에 서술한다. 분량 규율은 중간 레인 산출물용이고 최종 보고서는 완결성이 우선한다.
- 금지: 돌파구·혁신적 류 과장어, 자명하다 류 증명 회피어, 검증 기록 없는 유도됨·제1원리·닫힘, 기계 검사 상태 문자열(PASS 등)의 본문 복사.

## 끈질김 (진취성 규율)

- **BLOCKED는 최후 수단이다.** 주장이 막히면 먼저 (a) routes 후보를 소진했는지, (b) 주장 범위를 좁혀 살릴 좁은 정리가 있는지, (c) 공리 1개 추가로 명시하면 열리는지를 순서대로 시도하고, 각 시도의 기각 근거를 기록한 뒤에만 BLOCKED를 쓴다.
- BLOCKED에는 **재개 조건**(무엇이 오면 다시 열리는지)을 반드시 적는다. 재개 조건 없는 BLOCKED는 미완성 감사다.
- ABANDONED는 구조적 불가능(반례 확정, no-go, 필요한 증거의 원천 부재)의 근거가 있을 때만 쓴다. 분량·피로·세션 길이는 사유가 아니다.
- 부정 결과도 완결한다: 반례·기각으로 끝나는 run도 40-final-report.md까지 간다. "죽은 경로 확인"은 실패가 아니라 산출이다.
- run은 시작했으면 반드시 COMPLETE 또는 근거 있는 ABANDONED로 닫는다. 중간 방치 금지 (stop 훅이 잡는다).

닫힘 판단에는 $ce-closure-gate, 무차원 식에는 $ce-dimensionless, 정본 문서 반영에는 $ce-doc-write를 적용한다.

마지막 상태 메시지에 다음을 한 줄로 남긴다.

    CE_RUN=_workspace/ce/<run-id>
