# CE Claude 하네스

Clarus-Equation 연구·구현을 위한 Claude Code 지침.

## 라우팅

요청을 받으면 먼저 규모를 판단한다. **작은 요청에 run을 파지 않는 것이 속도 규율의 핵심이다.**

| 요청 | 행동 |
|---|---|
| 단순 질문·계산·파일 위치·한 줄 수정 | run 없이 바로 답한다 (spot). 스킬을 로드하지 않는다 |
| 특정 주장 지위·증명 판정 | /ce-closure-gate 스킬 |
| 식의 차원·무차원 검사 | /ce-dimensionless 스킬 |
| 테스트·회귀·수치 검증 | /ce-validate 스킬 |
| 정본 문서 작성·수정 | /ce-doc-write 스킬 |
| guard 벤치·ASR·회귀 | /clarus-guard-bench 스킬 |
| 신규 주장 검증·승격 판단·논문화·병렬 연구 | /ce-research 스킬 (full) |
| 완결 run의 후속·반복 (v8→v9 등) | /ce-research 스킬 (light, PREDECESSOR 지정) |

선택한 스킬 하나만 로드한다. 레인은 Agent 도구로 역할 subagent 4종(ce-physics-sourcer, ce-math-verifier, ce-status-auditor, ce-impl-engineer)을 병렬 실행한다 — 역할 카드를 오케스트레이터 문맥에 로드하지 않는다. 우회 경로 탐색(routes)은 math-verifier가 같은 스폰에서 수행하고, 최종 집필은 오케스트레이터가 직접 한다.

## 코어 명령

    sh .claude/hooks/run.sh <cmd> ...       # 플랫폼 공통 (프리빌드 바이너리 직행)

- `init <run-dir>` — run 생성, `.active-run` 포인터 설정. `REUSE?` 목록이 나오면 새로 파지 말고 이어받는다.
- `status <run-dir>` — 단계별 상태·Gate·수정 카운트를 한 화면에 출력. **run 현황 파악에 stage 파일을 다시 읽지 말고 이것을 쓴다.**
- `check <run-dir> <contract|lanes|gate|build|final>` — 해당 단계까지 전체 체인 검사.
- `revise <run-dir> <role>` — 수정 루프 기록. 역할당 2회 상한.
- `gc <workspace>` — 완결 run을 `_archive/`로 이동, 미완성 run 나열. 세션 종료 시 실행.

## 심각도 어휘 (모든 역할 공통)

- **P0**: 주장을 무너뜨리는 결함 — 완전한 반례, 증명 오류, 수치 불일치가 결론을 바꿈. 해소 전 Gate PASS 불가.
- **P1**: 결론은 유지되나 방치 불가 — 증명 공백, 숨은 공리, 기준선 불일치. revise로 처리하거나 BLOCKED로 기록.
- **P2**: 표기·분량·정리 문제. 최종 보고서에 목록만 남기고 수정 루프를 소모하지 않는다.

## 속도 규율

- 카드·스킬을 선로딩하지 않는다. 레인 작업은 subagent에 위임하고 결과 판정만 회수한다.
- run 현황은 `status`로 본다. stage 파일 재독은 내용을 인용할 때만.
- PREDECESSOR가 있으면 선행 run이 검증한 결론을 재유도·재대조하지 않고 경로만 인용한다.
- 상세 계산·로그·발췌는 artifacts/에 쓰고 stage 파일에는 판정·표·경로만 남긴다.

## 끈질김 (진취성 규율)

- **BLOCKED는 최후 수단이다.** 막히면 (a) 우회 경로 후보 소진, (b) 주장 범위 축소로 좁은 정리 salvage, (c) 공리 1개 명시 추가 순으로 시도하고 기각 근거를 기록한 뒤에만 쓴다. BLOCKED에는 재개 조건이 필수다.
- ABANDONED는 구조적 불가능(반례 확정·no-go·증거 원천 부재)의 근거가 있을 때만. 분량·피로·세션 길이는 사유가 아니다.
- 부정 결과도 완결한다: 반례·기각으로 끝나는 run도 40-final-report.md까지 간다. 죽은 경로 확인은 실패가 아니라 산출이다.
- revise 한도 소진은 run 포기 사유가 아니다 — 살릴 것을 좁혀 살리고 남는 결함만 BLOCKED로 보고서에 남긴다.

## 데이터 반출 게이트

- git commit/push는 PreToolUse 훅 `check-large-data.sh`(.claude/hooks = .codex/hooks 미러)가 검사한다: 95MB 초과 파일과 `_workspace/` 아래 데이터 확장자(zip·mat·pkl·npy·npz·h5·hdf5·pt·onnx·parquet·bin·exe)는 차단된다.
- 연구 데이터 원본은 커밋하지 않는다. 매니페스트·코드·요약 산출만 커밋하고 원본은 문서화된 DOI에서 재취득한다 (.gitignore가 정본).
- 게이트에 걸리면 우회(-f, --no-verify)하지 말고 `git rm --cached`로 언트래킹한 뒤 .gitignore 패턴을 보강한다.

## 정직성 (협상 불가)

- 실패를 통과로 표현하지 않는다. 실행하지 않은 검증을 실행했다고 쓰지 않는다.
- 기계 검사 상태 문자열(PASS 등)을 이론 지위처럼 문서에 복사하지 않는다.
- ce-closure-gate 기록 없이 "유도됨·제1원리·닫힘"을 쓰지 않는다.
- 관측 근접은 증명이 아니다. 형식 지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐이다.
