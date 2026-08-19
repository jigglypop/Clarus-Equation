# CE Codex 하네스 상세 규칙

루트 `../AGENTS.md`가 기본 속도·라우팅 규칙의 정본이다. 이 파일은 `.codex` 내부 작업이나 명시적 CE 연구 run에만 상세 규칙을 보충한다.

## 라우팅

요청을 받으면 먼저 규모를 판단한다. **작은 요청에 run을 파지 않는 것이 속도 규율의 핵심이다.**

| 요청 | 행동 |
|---|---|
| 단순 질문·계산·파일 위치·한 줄 수정 | run 없이 바로 답한다 (spot). 스킬을 로드하지 않는다 |
| 특정 주장 지위·증명 판정 | $ce-closure-gate |
| 식의 차원·무차원 검사 | $ce-dimensionless |
| 테스트·회귀·수치 검증 | $ce-validate |
| 주장·상수·판본·진리값 원장 작성·수정 | $ce-ledger-write + ce-ledger-writer |
| 강의·유도·독자 가이드·논문 원고 작성·수정 | $ce-paper-write + ce-paper-writer |
| 문서 유형 판별·공통 지위 규약 확인 | $ce-doc-write (라우터) |
| guard 벤치·ASR·회귀 | $clarus-guard-bench |
| 신규 주장 검증·승격 판단·논문화·병렬 연구 | $ce-research (full) |
| 완결 run의 후속·반복 (v8→v9 등) | $ce-research (light, PREDECESSOR 지정) |
| `.codex` 하네스·스킬·역할 카드·실행 정책 수정 | run 없이 대상 파일만 최소 수정 (spot) |

선택한 스킬 하나만 로드한다. 역할 카드는 4장(sourcer, math-verifier, status-auditor, impl-engineer)이며 해당 레인을 실행하는 순간에만 읽는다. 우회 경로 탐색(routes)은 math-verifier 레인에 포함되고, 최종 집필은 역할 카드 없이 $ce-research의 집필 규칙으로 직접 한다.

## 코어 명령

    .codex/hooks/run.sh <cmd> ...     # POSIX
    .codex/hooks/run.cmd <cmd> ...    # Windows (프리빌드 바이너리 직행)

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

- 카드·스킬을 선로딩하지 않는다. 필요한 시점에 필요한 것만 읽는다.
- run 현황은 `status`로 본다. stage 파일 재독은 내용을 인용할 때만.
- PREDECESSOR가 있으면 선행 run이 검증한 결론을 재유도·재대조하지 않고 경로만 인용한다.
- 상세 계산·로그·발췌는 artifacts/에 쓰고 stage 파일에는 판정·표·경로만 남긴다.
- 레인 파일은 3–4만 자 재생성이 아니라 판정 중심의 짧은 문서다.
- `check final`은 앞 단계를 포함한다. 같은 상태에서 contract→lanes→gate→build→final을 연속 재실행하지 말고, 단계 전환에 필요한 check와 마지막 final을 각각 한 번만 실행한다.

## 실행·캐시 규율

- “검증해줘”, “고쳐줘”, “하네스 개선”은 전체 테스트 실행 권한이 아니다. 기본은 변경 파일에 직접 연결된 가장 작은 검사 **한 개**이며, green이면 멈춘다.
- 관련 회귀 확대는 작은 검사가 실패했거나 공용 API·의존성 경계를 바꿨을 때만 한다. 저장소 전체 pytest·전체 bench·release 검증은 사용자가 `전체`, `full`, `release`, `CI 재현`을 명시한 경우에만 실행한다.
- 코드가 바뀌지 않았는데 같은 green 명령을 다시 실행하지 않는다. 다른 역할의 성공 로그가 현재 byte와 명령을 고정했다면 인용하고 중복 실행하지 않는다.
- Python은 기존 interpreter를 `-B`로 사용하고 child 환경에 `PYTHONDONTWRITEBYTECODE=1`을 둔다. pytest는 `-p no:cacheprovider`와 실행별 고유 `--basetemp`를 사용한다.
- basetemp는 하네스가 만든 고유 소유 경로만 쓰고 성공·실패·중단 모두 `finally`에서 제거한다. 고정 이름 재사용, repo root 임시물, 광역 glob cleanup은 금지한다.
- Ruff는 `--no-cache`를 쓴다. 문법 확인은 source를 메모리에서 `compile()`하고 `compileall`로 pyc를 만들지 않는다.
- 현재 interpreter로 가능한 작업에 새 venv나 `uv run`을 만들지 않는다. dependency resolution이 실제로 필요할 때만 기존 프로젝트 실행기를 사용한다.
- Windows에서는 `.codex/hooks/python.cmd doctor`로 실행기를 먼저 확인하고, focused Python/pytest는 같은 래퍼의 `python`/`pytest` 모드로 실행한다. 프롬프트 입력을 기다리지 않으며, Application Control이 차단한 `.venv`나 uv-managed Python을 우회하지 않는다.
- full 검증을 생략한 것은 실패가 아니다. 최종 보고에는 실행한 최소 검사와 생략한 확대 검증을 각각 명시한다.

## main Git 인계 규율

- Git 상태 변경과 발행은 root/main agent 한 명만 맡는다. 모든 subagent는 `status`·`diff`·hash 같은 읽기 전용 확인만 하며 `add`·`commit`·`fetch`·`pull`·`rebase`·`push`·branch/worktree 변경을 하지 않는다.
- main은 종료 전에 루트·현재 branch·upstream·HEAD·remote tip·변경 경로를 확인하고, 작업 범위가 아닌 dirty path가 하나라도 있으면 자동 정리하거나 함께 stage하지 않는다.
- 발행할 때는 승인된 경로 manifest만 `git add -- <paths>`로 stage하고 staged diff·관련 검증을 확인한 뒤 `.codex/hooks/check-large-data.cmd --commit`을 명시적으로 실행한다. push 직전에는 같은 실행기의 `--push`도 실행한다. 이 gate는 연구 binary 확장자와 95MB 초과 Git blob을 막으며 secret scanner를 대신하지 않는다. `git add .`, `git add -A`, force push, 자동 stash/reset/clean/rebase는 금지한다.
- push 직전 다시 fetch하여 `origin/main`이 preflight base와 같은지 확인하고 fast-forward 일반 push만 사용한다. 완료 후 remote `refs/heads/main`과 local HEAD의 일치, commit SHA, 정확한 경로와 남은 dirt를 보고한다.

## 뇌 알고리즘 경로 선택

- 새 뇌/기억/의식 run은 직전 실행의 `12-routes.md`, `31-validation.md`, 존재하면 `40-final-report.md`, 그리고 `_workspace/ce/brain-algorithm-route-ledger.md`를 먼저 읽는다. 40이 없으면 가장 늦은 numbered audit와 원장 행을 사용하고 closure 부재를 계약에 기록한다. 전체 artifact를 재독하거나 실패한 실험을 endpoint·threshold·seed만 바꿔 재시도하지 않는다.
- 오케스트레이터는 양성처럼 보이는 수치가 아니라 인과 식별 가능성, 기존 STOP이 남긴 정보, 독립 반증 대조군, capability dependency 순서로 다음 후보를 고른다. 선택 근거와 기각·퇴역 경로를 00-contract에 고정한 뒤 레인을 연다.
- simulator 결과는 실제 뇌의 증거로 자동 승격하지 않는다. 전역 원장은 후보·의존성·증거 경로·상태·다음 falsifier만 관리하고, 서사나 의식 동일시를 기록하지 않는다.

## 끈질김 (진취성 규율)

- **BLOCKED는 최후 수단이다.** 막히면 (a) 우회 경로 후보 소진, (b) 주장 범위 축소로 좁은 정리 salvage, (c) 공리 1개 명시 추가 순으로 시도하고 기각 근거를 기록한 뒤에만 쓴다. BLOCKED에는 재개 조건이 필수다.
- ABANDONED는 구조적 불가능(반례 확정·no-go·증거 원천 부재)의 근거가 있을 때만. 분량·피로·세션 길이는 사유가 아니다.
- 부정 결과도 완결한다: 반례·기각으로 끝나는 run도 40-final-report.md까지 간다. 죽은 경로 확인은 실패가 아니라 산출이다.
- revise 한도 소진은 run 포기 사유가 아니다 — 살릴 것을 좁혀 살리고 남는 결함만 BLOCKED로 보고서에 남긴다.

## 데이터 반출 게이트

- Claude의 PreToolUse와 main의 필수 수동 preflight는 Windows 네이티브 `check-large-data.cmd`를 사용한다. POSIX `check-large-data.sh`는 같은 좁은 정책의 미러다. 95MB 초과 Git blob과 `_workspace/` 아래 지정 연구 binary 확장자를 차단하며, 일반 secret/source scanner는 아니다.
- 연구 데이터 원본은 커밋하지 않는다. 매니페스트·코드·요약 산출만 커밋하고 원본은 문서화된 DOI에서 재취득한다 (.gitignore가 정본).
- 게이트에 걸리면 우회(-f, --no-verify)하지 말고 `git rm --cached`로 언트래킹한 뒤 .gitignore 패턴을 보강한다.

## 정직성 (협상 불가)

- 실패를 통과로 표현하지 않는다. 실행하지 않은 검증을 실행했다고 쓰지 않는다.
- 기계 검사 상태 문자열(PASS 등)을 이론 지위처럼 문서에 복사하지 않는다.
- ce-closure-gate 기록 없이 "유도됨·제1원리·닫힘"을 쓰지 않는다.
- 관측 근접은 증명이 아니다. 형식 지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐이다.
