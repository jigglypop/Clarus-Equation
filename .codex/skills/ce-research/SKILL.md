---
name: ce-research
description: Clarus-Equation 연구를 출처·수학·대안 경로의 독립 레인, 형식 지위 감사, 구현·수치 검증, 최종 집필로 수행한다. CE/Clarus-EQ 식·가설·논문·코드의 유도, 반례, 재현, 승격 판단이나 병렬 연구를 요청할 때 사용한다. 단순 계산과 guard 제품 벤치만 필요한 요청에는 사용하지 않는다.
---

# CE Research

먼저 ../../agents/ce-status-auditor.md와 필요한 역할 카드만 읽는다. 모든 카드를 선로딩하지 않는다.

## 코어 명령

    <codex-home>/hooks/run.sh init|check|revise|gc ...        # POSIX
    <codex-home>/hooks/run.ps1 init|check|revise|gc ...       # Windows

- `init <run-dir>`: run 생성. `REUSE?` 목록이 나오면 같은 주제의 미완성 run을 새로 파지 말고 이어받는다.
- `check <run-dir> <contract|lanes|gate|build|final>`: 해당 단계까지의 전체 체인을 검사한다. gate는 20-audit.md의 `Gate: PASS`를 요구한다.
- `revise <run-dir> <role>`: 수정 루프 기록. 역할당 2회를 넘으면 거부되므로 BLOCKED로 강등하고 보고서에 남긴다.
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
   - math-verifier → 11-math.md: 항상.
   - route-explorer → 12-routes.md: 계약에 Open/Hypothesis 항목이 있을 때만.
   - 스킵한 레인은 파일에 `Status: SKIPPED (사유 한 줄)`만 쓴다. 각 레인 파일은 판정·표 중심으로 유지하고 상세 계산·로그는 artifacts/에 두고 경로만 인용한다.
3. **감사**: status-auditor가 20-audit.md에 형식 출처·반례를 기록하고 `Gate: PASS|REVISE|BLOCKED`를 판정한다.
4. **구현**: `check <run-dir> gate` 통과 후 승인된 범위만 impl-engineer가 구현하고 $ce-validate로 검증한다(30/31). 코드 변경이 없으면 `Status: SKIPPED (사유)`.
5. **수정 루프**: P0/P1은 `revise <run-dir> <role>`로 기록한 뒤 지목된 역할만 고친다. 한도 초과 시 BLOCKED로 보고서에 남긴다.
6. **집필**: paper-writer가 판정을 강화하지 않고 40-final-report.md를 작성한다. `check <run-dir> final` 통과를 확인한다.

run을 중단하려면 40-final-report.md에 `Status: ABANDONED (사유)`를 쓴다. 방치하지 않는다.

닫힘 판단에는 $ce-closure-gate, 무차원 식에는 $ce-dimensionless, 문서 반영에는 $ce-doc-write를 적용한다.

마지막 상태 메시지에 다음을 한 줄로 남긴다.

    CE_RUN=_workspace/ce/<run-id>
