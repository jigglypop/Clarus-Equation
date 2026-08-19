---
name: ce-research
description: Clarus-Equation 연구를 출처·수학·대안 경로의 독립 레인, 형식 지위 감사, 구현·수치 검증, 최종 집필로 수행한다. CE/Clarus-EQ 식·가설·논문·코드의 유도, 반례, 재현, 승격 판단이나 병렬 연구를 요청할 때 사용한다. 단순 계산과 guard 제품 벤치만 필요한 요청에는 사용하지 않는다.
---

# CE Research

레인·감사·구현은 Agent 도구로 해당 역할 subagent(ce-physics-sourcer, ce-math-verifier, ce-status-auditor, ce-impl-engineer)에 위임한다. 역할 카드를 오케스트레이터 문맥에 로드하지 않고, 독립 레인은 한 메시지에서 병렬로 띄운다. 집필은 subagent 없이 오케스트레이터가 직접 한다 — 판정 문맥을 이미 갖고 있으므로 재독이 낭비다.

## 코어 명령

    sh .claude/hooks/run.sh init|status|check|revise|gc ...

- `init <run-dir>`: run 생성. `REUSE?` 목록이 나오면 같은 주제의 미완성 run을 새로 파지 말고 이어받는다.
- `status <run-dir>`: 단계별 상태·Gate·수정 카운트 한 화면. 현황 파악에 stage 파일을 재독하지 말고 이것을 쓴다.
- `check <run-dir> <contract|lanes|gate|build|final>`: 해당 단계까지의 전체 체인을 검사한다. gate는 20-audit.md의 `Gate: PASS`를 요구한다.
- `revise <run-dir> <role>`: 수정 루프 기록. 역할당 2회를 넘으면 거부되므로 주장 범위를 좁혀 살릴 것은 살리고, 남는 결함은 BLOCKED로 보고서에 남긴다.
- `gc <workspace>`: 완결 run을 `_archive/`로 이동하고 미완성 run을 나열한다. 연구 세션을 마칠 때 실행한다.

모든 산출물은 _workspace/ce/<run-id>/에 쓴다. run-id는 `<주제>-<YYYYMMDD>`. 원본은 역할 권한 밖에서 수정하지 않는다.

## 산출물 위생 (모든 역할 공통)

- run 루트에는 번호 stage 파일 8개만 둔다. 사전등록, 루프별 노트, 검증 스크래치 등 보조 문서는 전부 artifacts/에 만들고 stage 파일에서 경로로 인용한다. `check final`이 루트의 잡파일을 차단한다.
- stage 파일 이름을 바꾸거나 변형본(10-provenance.md, 30-build-validation.md 등)을 만들지 않는다. 내용이 다르면 절을 나눠 같은 파일 안에 쓴다.
- 수식은 블록 `$$ ... $$`(별도 줄), 인라인 `$...$`만 쓴다. `\[ \]`·`\( \)`는 렌더링되지 않으므로 금지. 수학 기호는 코드스팬이 아니라 수식 모드로 쓴다.

## 모드 선택

- **full**: 신규 주장, 승격 판단, 논문화. 아래 6단계 전부.
- **light**: 완결된 선행 run의 후속·반복(v8→v9 등). 계약에 `PREDECESSOR: _workspace/ce/<선행-run>`을 적고, 레인은 math-verifier만 실행하며 감사(gate)와 최종 보고서만 유지한다. 선행 run이 이미 감사한 결론은 재유도하지 않고 경로만 인용한다.

## 6단계 (full)

1. **계약**: 질문·정의역·주장·기호·허용 오차·PREDECESSOR를 00-contract.md에 고정하고 `Status: COMPLETE`로 닫는다.
2. **레인(조건부 팬아웃)**: 해당하는 레인만 독립 문맥에서 병렬 실행한다.
   - physics-sourcer → 10-sources.md: 계약이 관측값·외부 데이터를 인용할 때만.
   - math-verifier → 11-math.md + 12-routes.md: 항상. 검산과 우회 경로 탐색을 한 스폰에서 수행한다. routes는 계약에 Open/Hypothesis가 있거나 검산이 P0/막힘을 냈을 때 실행하고, 아니면 12에 `Status: SKIPPED (사유 한 줄)`.
   - 스킵한 레인은 파일에 `Status: SKIPPED (사유 한 줄)`만 쓴다. 각 레인 파일은 판정·표 중심으로 유지하고 상세 계산·로그는 artifacts/에 두고 경로만 인용한다.
3. **감사**: status-auditor가 20-audit.md에 형식 출처·반례를 기록하고 `Gate: PASS|REVISE|BLOCKED`를 판정한다.
4. **구현**: `check <run-dir> gate` 통과 후 승인된 범위만 impl-engineer가 구현하고 /ce-validate로 검증한다(30/31). 코드 변경이 없으면 `Status: SKIPPED (사유)`.
5. **수정 루프**: P0/P1은 `revise <run-dir> <role>`로 기록한 뒤 지목된 역할만 고친다. 한도 초과 시 주장 범위를 좁혀 살릴 수 있는 좁은 정리를 분리하고, 남는 결함만 BLOCKED로 보고서에 남긴다.
6. **집필**: 오케스트레이터가 40-final-report.md를 직접 작성한다. /ce-doc-write 규약(지위 태그·완전 반례 처리·문체·수식 표기)을 따르고 아래 집필 규칙을 지킨다. `check <run-dir> final` 통과를 확인한다.

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

닫힘 판단에는 /ce-closure-gate, 무차원 식에는 /ce-dimensionless, 정본 문서 반영에는 /ce-doc-write를 적용한다.

마지막 상태 메시지에 다음을 한 줄로 남긴다.

    CE_RUN=_workspace/ce/<run-id>
