# CE Codex 하네스 규칙

## 현재 저장 정책 (최우선)

- `reality_stone`은 제거되었으므로 패키지·경로·import를 요구하지 않는다.
- 새 `CE_RUN`이나 `_workspace/` 디렉터리를 만들지 않는다. 기존 `_workspace/`는 과거 기록으로 읽기만 한다.
- 연구·검증의 최신 결론과 재현 경로는 루트 `docs/`의 해당 정본을 제자리 갱신한다.
- 임시 계산은 OS 임시 디렉터리를 사용하고 종료 시 이번 실행이 만든 정확한 경로만 제거한다.
- 아래에 남은 run 설명은 과거 호환 문서이며 이 절과 충돌하면 적용하지 않는다.

루트 `../AGENTS.md`가 기본 지침의 정본이다. 이 파일은 `.codex/` 자체를 고치거나 명시적인 CE 연구 run을 수행할 때만 적용한다.

## 라우팅

작은 요청에는 run과 subagent를 만들지 않는다. 아래에서 가장 좁은 경로 하나만 고른다.

| 요청 | 행동 |
|---|---|
| 질문·계산·파일 탐색·작은 수정 | direct |
| 주장 지위·증명 완결성·반례 | `$ce-closure-gate` |
| 차원 또는 무차원성 | `$ce-dimensionless` |
| 테스트·회귀·수치 무결성 | `$ce-validate` |
| 주장·상수·판본 원장 | `$ce-ledger-write` |
| 강의·유도·논문 원고 | `$ce-paper-write` |
| 신규 CE 주장 전체 연구 | `$ce-research` full |
| 기존 run의 후속 검증 | `$ce-research` light |
| `.codex/` 설정과 하네스 수정 | direct |

전문 스킬은 하나만 먼저 읽는다. full 연구에서만 필요한 레인을 병렬화한다. 역할 에이전트는 `ce-physics-sourcer`, `ce-math-verifier`, `ce-status-auditor`, `ce-impl-engineer`, `ce-ledger-writer`, `ce-paper-writer` 여섯 개뿐이다. 원장 문서는 `ce-ledger-writer`가 먼저 안정화하고, 강의·유도·독자 가이드·논문 원고는 `ce-paper-writer`가 그 원장을 읽기 전용으로 사용한다.

## Run 명령

    .codex/hooks/run.sh <cmd> ...     # POSIX
    .codex/hooks/run.cmd <cmd> ...    # Windows (프리빌드 바이너리 직행)

- `init <run-dir>`: 기존 run을 재개하거나 새 run을 만든다. 활성 미완성 run이 있으면 `REUSE_REQUIRED`로 멈춘다.
- `init --new-contract <run-dir>`: 목표와 증거 계보가 독립적인 연구에만 쓴다.
- `status <run-dir>`: 단계, Gate, 수정 횟수를 한 화면에 본다. 현황 확인을 위해 stage 파일을 반복해 읽지 않는다.
- `check <run-dir> <contract|lanes|gate|build|final>`: 해당 단계까지 검사한다.
- `revise`, `counterexample`, `pivot`: 상세 계약은 `$ce-research`에서 필요할 때만 읽는다.
- `gc <workspace>`: 완료되고 참조되지 않는 run만 보관한다.

## 판정

- **P0**: 결론을 무너뜨리는 반례, 증명 오류, 수치 불일치. 해소 전 PASS 금지.
- **P1**: 결론은 유지하지만 방치할 수 없는 증명 공백, 단위 괴리, 기준선 불일치.
- **P2**: 표기와 정리 문제. 연구 수정 횟수를 쓰지 않는다.

## 속도와 산출물

- 카드와 스킬을 선로드하지 않는다. 필요한 시점에 하나만 읽는다.
- 선행 run이 검증한 결론은 재유도하지 않고 경로를 인용한다.
- stage 파일은 판정과 경로만 담고 상세 로그는 `artifacts/`에 둔다.
- 논문 정본은 `docs/<분야>/<논문>/00_논문목차.md`와 연결된 장에 둔다. `_workspace/`에 논문 사본을 만들지 않는다.
- `40-final-report.md`는 `DOCS_PAPER`, 결론, 근거 경로만 담는 인계 기록이다.
- `check final`은 앞 단계를 포함하므로 같은 상태에서 모든 check를 연속 반복하지 않는다.

## 검증

- 변경과 직접 연결된 가장 작은 검사 하나부터 실행하고 green이면 멈춘다.
- 공용 API, 수치 계약, 경계가 바뀌었거나 작은 검사가 실패할 때만 관련 묶음으로 넓힌다.
- 전체 pytest, bench, release 검증은 사용자가 `전체`, `full`, `release`, `CI 재현`을 명시한 경우에만 실행한다.
- Windows Python은 `.codex/hooks/python.cmd doctor` 후 같은 래퍼를 쓴다. 차단된 venv나 대화형 설치를 우회하지 않는다.
- 같은 byte와 명령의 green 결과를 반복 실행하지 않는다.

## 전문 계약

일반 요청에서 아래 문서를 선로드하지 않는다. 해당 연구가 실제로 필요할 때만 읽는다.

- 실측 교정: `harnesses/empirical_calibration_loop.md`
- Rust/CUDA parity: `harnesses/curvature_backend_parity.md`
- 뇌/AGI 연구: `harnesses/real_brain_equation_discovery_loop.md`
- 뇌 증거 수준: `harnesses/brain_evidence_ladder.md`

## main Git 인계 규율

- Git 상태 변경과 발행은 root/main agent 한 명만 맡는다. 모든 subagent는 `status`·`diff`·hash 같은 읽기 전용 확인만 하며 `add`·`commit`·`fetch`·`pull`·`rebase`·`push`·branch/worktree 변경을 하지 않는다.
- main은 종료 전에 루트·현재 branch·upstream·HEAD·remote tip·변경 경로를 확인하고, 작업 범위가 아닌 dirty path가 하나라도 있으면 자동 정리하거나 함께 stage하지 않는다.
- 발행할 때는 승인된 경로 manifest만 `git add -- <paths>`로 stage하고 staged diff·관련 검증을 확인한 뒤 `.codex/hooks/check-large-data.cmd --commit`을 명시적으로 실행한다. push 직전에는 같은 실행기의 `--push`도 실행한다. 이 gate는 연구 binary 확장자와 95MB 초과 Git blob을 막으며 secret scanner를 대신하지 않는다. `git add .`, `git add -A`, force push, 자동 stash/reset/clean/rebase는 금지한다.
- push 직전 다시 fetch하여 `origin/main`이 preflight base와 같은지 확인하고 fast-forward 일반 push만 사용한다. 완료 후 remote `refs/heads/main`과 local HEAD의 일치, commit SHA, 정확한 경로와 남은 dirt를 보고한다.

## 끈질김 (진취성 규율)

- **BLOCKED는 최후 수단이다.** 한 후보식의 반례나 역할별 수리 한도만으로 쓰지 않는다. 반례를 잠근 뒤 서로 다른 상태·상호작용·측정·개입 구조의 경로를 최소 3개 만들고, 각 경로의 판별 예측·음성대조·봉인 split·중단 조건을 시험한다. 명시적 모델 클래스 no-go, 필요한 외부 자료의 부재, 또는 세 경로의 중단 조건이 모두 충족된 경우에만 재개 조건과 함께 쓴다.
- ABANDONED는 명시한 모델 클래스의 구조적 불가능이나 증거 원천의 확정적 부재가 있을 때만 쓴다. 한 식의 반례, 분량·피로·세션 길이는 사유가 아니다.
- 부정 결과도 완결한다. 반례·기각은 `docs/` 조립 논문의 해당 장과 짧은 `40-final-report.md` 인계 기록에 반영한다. 죽은 경로 확인은 실패가 아니라 산출이다.
- revise 한도 소진은 run 포기나 목표 축소 사유가 아니다. 실패식을 음성대조군으로 고정하고 `counterexample → 세 기전 경로 → pivot`으로 전환한다. 좁혀서 참이 된 명제는 보존 결과일 뿐 돌파구 성공으로 세지 않는다.

## 데이터 반출 게이트

- Claude의 PreToolUse와 main의 필수 수동 preflight는 Windows 네이티브 `check-large-data.cmd`를 사용한다. POSIX `check-large-data.sh`는 같은 좁은 정책의 미러다. 95MB 초과 Git blob과 `_workspace/` 아래 지정 연구 binary 확장자를 차단하며, 일반 secret/source scanner는 아니다.
- 연구 데이터 원본은 커밋하지 않는다. 매니페스트·코드·요약 산출만 커밋하고 원본은 문서화된 DOI에서 재취득한다 (.gitignore가 정본).
- 게이트에 걸리면 우회(-f, --no-verify)하지 말고 `git rm --cached`로 언트래킹한 뒤 .gitignore 패턴을 보강한다.

## 정직성 (협상 불가)

- 실패를 통과로 표현하지 않는다. 실행하지 않은 검증을 실행했다고 쓰지 않는다.
- 기계 검사 상태 문자열(PASS 등)을 이론 지위처럼 문서에 복사하지 않는다.
- ce-closure-gate 기록 없이 "유도됨·제1원리·닫힘"을 쓰지 않는다.
- 관측 근접은 증명이 아니다. 형식 지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐이다.
