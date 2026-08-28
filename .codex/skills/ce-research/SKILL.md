---
name: ce-research
description: Clarus-Equation 연구를 출처·수학·대안 경로의 독립 레인, 형식 지위 감사, 구현·수치 검증, 최종 집필로 수행한다. CE/Clarus-EQ 식·가설·논문·코드의 유도, 반례, 재현, 승격 판단이나 병렬 연구를 요청할 때 사용한다. 단순 계산만 필요한 요청에는 사용하지 않는다.
---

# CE Research

> 현재 저장 정책: 새 `CE_RUN`과 `_workspace/`를 만들지 않는다. 기존 run은 읽기 전용 근거로만 사용하며, 최신 연구 결과·감사표·재현 명령은 루트 `docs/`의 해당 정본에 직접 반영한다. 아래의 run 생성·갱신 명령은 레거시이며 실행하지 않는다.

역할 카드는 4장(ce-physics-sourcer, ce-math-verifier, ce-status-auditor, ce-impl-engineer)이다. 해당 레인을 실행하는 순간에만 그 카드를 읽는다 — 모든 카드를 선로딩하지 않는다. 집필은 역할 카드 없이 아래 집필 규칙으로 직접 한다.

## 코어 명령

    <codex-home>/hooks/run.sh init|status|check|revise|counterexample|pivot|gc ...
    <codex-home>/hooks/run.ps1 init|status|check|revise|counterexample|pivot|gc ...

- `init <run-dir>`: 기존 경로면 그 run을 재개한다. 다른 미완성 run이 있으면 디렉터리를 만들기 전에 `REUSE_REQUIRED`로 중단한다.
- `init --new-contract <run-dir>`: 기존 목표와 계보를 공유하지 않는 독립 연구 프로그램일 때만 새 최상위 run을 연다. 같은 프로그램의 새 가설·판본·독립 확인·복구는 아래 epoch/pivot을 쓴다.
- `status <run-dir>`: 단계별 상태·Gate·수정 카운트 한 화면. 현황 파악에 stage 파일을 재독하지 말고 이것을 쓴다.
- `check <run-dir> <contract|lanes|gate|build|final>`: 해당 단계까지의 전체 체인을 검사한다. gate는 20-audit.md의 `Gate: PASS`를 요구한다.
- `revise <run-dir> <role>`: 구현·측정·정밀도 같은 국소 수리 기록. 역할당 3회까지 허용한다. 이 한도는 연구 목표를 축소하거나 `BLOCKED`로 닫는 근거가 아니다. 이론 잔차나 반례라면 같은 run에서 `counterexample`과 `pivot`으로 전환한다.
- `counterexample <run-dir> <cex-id>`: 반례 또는 실증 모순을 같은 run의 `artifacts/epochs/<cex-id>/`에 연다. 실패한 부모 주장과 witness를 잠그되 새 workspace를 만들지 않는다.
- `pivot <run-dir> <cex-id> <route-id>`: 최소 3개의 구조적으로 다른 기전 포트폴리오를 검증하고 선택 route를 같은 epoch 안에 연다. threshold·seed·endpoint·decoder만 바꾼 후보는 거부한다.
- `gc <workspace>`: 완결 run 중 `.pin`, 활성 epoch, 외부 문서·코드의 참조가 없는 것만 `_archive/`로 옮기고 `INDEX.tsv`에 기록한다. 참조된 경로와 미완성 run은 이동하지 않는다.

계약·레인·원장·검증 receipt·계산 산출물은 `_workspace/ce/<program-id>/`에 쓴다. 상세 논문 정본은 예외로 주제에 맞는 `docs/<분야>/<논문>/00_논문목차.md`와 그 목차가 순서대로 연결한 장별 Markdown 묶음을 제자리 갱신한다. 긴 강의·유도도 같은 입구+장 구조를 쓴다. `_workspace/`에는 논문 사본·초안·v2·final 사본을 만들지 않는다. 새 프로그램 ID에는 retry·recheck·recovery·날짜 판본을 연쇄적으로 붙이지 않는다. 기존 dated run은 그대로 보존하되 후속 가설·반례·재계산은 그 run의 `artifacts/epochs/`에 둔다.

동일한 연구 목표와 증거 계보를 유지하는 동안에는 하나의 `CE_RUN`만 사용한다. 계산 실패 복구, 새 식 판본, 독립 확인, 전수 재계산, 민감도 분석, 검증기, 감사, 표·그림과 원고 수정은 새 작업공간의 근거가 아니며 기존 run의 단계 파일과 `artifacts/epochs/`에 누적한다. 새 작업을 시작하기 전에 `.active-run`과 해당 run의 `status`를 먼저 확인한다.

`.codex` 하네스·스킬·역할 카드의 문구, 실행 범위 또는 cache 정책을 고치는 요청은 이 연구 파이프라인의 대상이 아니다. 이 경우 run과 레인을 만들지 말고 대상 설정 파일만 spot 수정한다.

## 산출물 위생 (모든 역할 공통)

- run 루트에는 번호 core stage 파일 8개와, 실증 수치 권위가 필요할 때 동결한 `35-result-ledger.md` 하나만 둔다. 사전등록, 루프별 노트, 검증 스크래치 등 보조 문서는 전부 artifacts/에 만들고 stage 파일에서 경로로 인용한다. `check final`이 그 밖의 루트 잡파일을 차단한다.
- `40-final-report.md`는 논문이 아니라 짧은 인계 기록이다. `DOCS_PAPER: docs/<분야>/<논문>/00_논문목차.md`, run 결론, 증거 경로와 epoch 통합 여부만 적는다. 논문 서사는 그 목차가 연결한 장 파일에만 둔다.
- stage 파일 이름을 바꾸거나 변형본(10-provenance.md, 30-build-validation.md 등)을 만들지 않는다. 내용이 다르면 절을 나눠 같은 파일 안에 쓴다.
- 수식은 블록 `$$ ... $$`(별도 줄), 인라인 `$...$`만 쓴다. `\[ \]`·`\( \)`는 렌더링되지 않으므로 금지. 수학 기호는 코드스팬이 아니라 수식 모드로 쓴다.
- stage check는 단계 전환마다 필요한 것만 한 번 실행한다. 마지막 `check final`이 선행 체인을 포함하므로 같은 byte에서 모든 check를 연속 반복하지 않는다.
- 구현 검증은 `$ce-validate`의 최소 범위 규율을 따른다. CE full research run이라는 이유만으로 전체 pytest·전체 bench를 실행하지 않는다.

## 모드 선택

- **full**: 신규 주장, 승격 판단, 논문화. 아래 6단계 전부.
- **light**: 같은 프로그램의 후속·반복. 새 top-level run 대신 epoch를 열고, 기존 감사 결론은 재유도하지 않으며 변경된 기전·falsifier·split만 검산한다.

## 6단계 (full)

1. **계약**: 질문·정의역·주장·기호·허용 오차·PREDECESSOR를 00-contract.md에 고정하고 `Status: COMPLETE`로 닫는다.
2. **레인(조건부 팬아웃)**: 해당하는 레인만 실행한다.
   - physics-sourcer → 10-sources.md: 계약이 관측값·외부 데이터를 인용할 때만.
   - math-verifier → 11-math.md + 12-routes.md: 항상. 검산과 우회 경로 탐색을 한 레인에서 수행한다. routes는 계약에 Open/Hypothesis가 있거나 검산이 P0/막힘을 냈을 때 실행하고, 아니면 12에 `Status: SKIPPED (사유 한 줄)`.
   - 스킵한 레인은 파일에 `Status: SKIPPED (사유 한 줄)`만 쓴다. 각 레인 파일은 판정·표 중심으로 유지하고 상세 계산·로그는 artifacts/에 두고 경로만 인용한다.
3. **감사**: status-auditor가 20-audit.md에 형식 출처·반례를 기록하고 `Gate: PASS|REVISE|BLOCKED`를 판정한다.
4. **구현**: `check <run-dir> gate` 통과 후 승인된 범위만 impl-engineer가 구현하고 $ce-validate로 검증한다(30/31). 코드 변경이 없으면 `Status: SKIPPED (사유)`.
5. **수정·돌파구 루프**: D/I/P/C/B 구현·측정 결함은 `revise`로 최소 수정한다. T 잔차·완전 반례·핵심 실증 모순은 `counterexample`로 부모 주장과 witness를 잠그고, math-verifier가 구조적으로 다른 기전 route를 최소 3개 만든 뒤 하나를 outcome-blind로 선택해 `pivot`한다. 좁은 참 명제는 보존할 수 있지만 그것을 반례 해결이나 돌파구로 세지 않는다. `BLOCKED`는 명시적 no-go·외부 자료 부재·세 route 소진 뒤에만 허용한다.
6. **집필**: 안정화된 원장과 검증 산출물을 읽기 전용 근거로 삼아 기존 `docs/<분야>/<논문>/00_논문목차.md`와 관련 장을 제자리 갱신한다. 새 장을 만들기 전에 기존 장과 역할이 겹치지 않는지 확인한다. `$ce-paper-write` 규약을 따르고 아래 집필 규칙을 지킨다. `40-final-report.md`에는 목차 경로와 짧은 run 인계만 남긴 뒤 `check <run-dir> final`을 통과한다.

run이 끝나면 오케스트레이터는 원장의 해당 후보 상태·증거 경로·다음 falsifier를 먼저 갱신하고, root/main agent에게 정확한 변경 경로와 검증 결과를 인계한다. subagent와 역할 agent는 stage/commit/push하지 않는다. 사용자가 발행을 지시한 경우에만 main이 루트 Git 인계 규율에 따라 commit/push한다.

## 집필 규칙 (`docs/<분야>/<논문>/00_논문목차.md` + 장 파일)

- 어떤 판정도 강화하지 않는다 — 레인이 경험식이라 한 것을 산출로, BLOCKED를 "사실상 해결"로 바꿔 쓰지 않는다. 새 증거가 필요하면 쓰지 말고 담당 레인으로 돌려보낸다.
- 이론·수학 원고 구성: 제목 → 초록(배경·문제·방법·핵심 결과·형식 지위·한계) → 서론 → 정의·표기 → 공리(번호+하위분류) → 정리·증명 → 산출 → 관측 비교(중립 서술) → 미완성 과제와 한계 → 재현성(명령·경로) → 참조(식별자+접근 날짜, 2차 인용 금지).
- 실증·뇌 원고 구성: 제목 → 초록 → 서론과 연구 질문 → 자료·코호트·provenance → 측정모형·전처리·QC → 사전 고정 endpoint와 통계 → 주 결과 → 사전 고정한 모든 control·sensitivity·음성 결과 → 논의와 대안 설명 → 증거 사다리·주장 상한 → 한계와 다음 반증 조건 → 재현성·자료/코드 → 1차 출처. 핵심 표와 해석에 필요한 그림을 본문에 넣는다.
- 상세성: 독자가 원장이나 레인 로그를 열지 않아도 표본, 단위, estimand, 분모, 제외/QC, 불확도, 주 수치, sensitivity와 실패를 재구성할 수 있어야 한다. 전수 계산을 했다면 전체 사전 고정 결과를 표로 보존하고 주 결과의 개별 기여도·강건성 진단을 산문으로 해석한다.
- 본문은 완결된 문단 산문으로 쓴다. 자기완결적으로 — 독자가 레인 파일을 열지 않고도 논증을 따라오게 핵심 유도를 본문에 서술한다. 분량 규율은 중간 레인 산출물용이고 최종 보고서는 완결성이 우선한다.
- 금지: 한 줄 상태 보고서, 원장 표·검산 로그만 나열한 문서, 유리한 endpoint만 고른 요약, 돌파구·혁신적 류 과장어, 자명하다 류 증명 회피어, 검증 기록 없는 유도됨·제1원리·닫힘, 기계 검사 상태 문자열(PASS 등)의 본문 복사.
- 목차는 `## 논문 조립 순서`에서 같은 논문 폴더의 장 파일을 독자가 읽을 순서대로 연결한다. 각 장은 한 역할만 맡고 다른 장의 표·유도·결론을 복제하지 않는다. `_workspace/`나 run의 `artifacts/`에 논문을 쓰거나 같은 내용을 복제하면 완결 실패다. `docs/README.md`와 분야 읽기 지도에는 목차만 연결한다.

## 끈질김 (진취성 규율)

- **BLOCKED는 최후 수단이다.** 주장이 막히면 먼저 반례를 잠그고, 실패한 가정을 서로 다르게 바꾸는 기전 route를 최소 3개 구성해 판별 실험을 시도한다. 구조적으로 가능한 route가 없다는 no-go 또는 세 route의 kill condition 충족을 기록한 뒤에만 BLOCKED를 쓴다.
- BLOCKED에는 **재개 조건**(무엇이 오면 다시 열리는지)을 반드시 적는다. 재개 조건 없는 BLOCKED는 미완성 감사다.
- ABANDONED는 구조적 불가능(반례 확정, no-go, 필요한 증거의 원천 부재)의 근거가 있을 때만 쓴다. 분량·피로·세션 길이는 사유가 아니다.
- 부정 결과도 완결한다: 반례·기각은 실패한 식의 음성대조군이자 다음 pivot의 입력이다. 부모 주장은 철회하되 연구 목표는 모델 클래스 전체의 no-go가 나오기 전까지 자동 축소하지 않는다.
- run은 시작했으면 반드시 COMPLETE 또는 근거 있는 ABANDONED로 닫는다. 중간 방치 금지 (stop 훅이 잡는다).

닫힘 판단에는 `$ce-closure-gate`, 무차원 식에는 `$ce-dimensionless`, 원장에는 `$ce-ledger-write`, 논문과 강의에는 `$ce-paper-write`를 적용한다.

마지막 상태 메시지에 다음을 한 줄로 남긴다.

    CE_RUN=_workspace/ce/<run-id>
