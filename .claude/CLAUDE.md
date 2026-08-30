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
| 주장·상수·판본·진리값 원장 작성·수정 | /ce-ledger-write + ce-ledger-writer |
| 강의·유도·독자 가이드·논문 원고 작성·수정 | /ce-paper-write + ce-paper-writer |
| 문서 유형 판별·공통 지위 규약 확인 | /ce-doc-write 스킬 |
| guard 벤치·ASR·회귀 | /clarus-guard-bench 스킬 |
| 신규 주장 검증·승격 판단·논문화·병렬 연구 | /ce-research 스킬 (full) |
| 완결 run의 후속·반복 (v8→v9 등) | /ce-research 스킬 (light, PREDECESSOR 지정) |
| `.claude`/`.codex` 하네스·역할·실행 정책 수정 | run 없이 대상 파일만 최소 수정 (spot) |

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
- `check final`은 앞 단계를 포함한다. 같은 byte에서 모든 stage check를 연속 반복하지 않는다.

## 실행·캐시 규율

- 기본 검증은 변경 파일에 직접 연결된 가장 작은 검사 한 개다. 전체 pytest·전체 bench·release 검증은 사용자가 `전체`, `full`, `release`, `CI 재현`을 명시한 경우에만 실행한다.
- Windows에서는 `.claude/hooks/python.cmd doctor|python|pytest`를 사용한다. 이 래퍼는 공유 Codex 하네스로 위임하며, 정책 차단된 `.venv`/uv Python이나 대화형 선택 프롬프트를 사용하지 않는다.
- Python은 `-B`와 `PYTHONDONTWRITEBYTECODE=1`, pytest는 `-p no:cacheprovider`와 실행별 고유 basetemp를 사용한다. Ruff는 `--no-cache`, 문법 검사는 in-memory `compile()`을 우선한다.
- 현재 interpreter로 가능한 작업에 새 venv나 `uv run`을 만들지 않는다. dependency resolution이 실제로 필요하면 먼저 캐시/정책 전제와 승인 범위를 명시한다.
- 같은 byte와 명령의 green 검증을 반복하지 않는다. 확대 검증을 생략했다면 실패로 쓰지 말고 실행하지 않았다고 명시한다.

## 실측 교정 루프와 식 개정

- 수치 검증이 어긋나면(backend parity FAIL, 기준선 잔차 $|z|>3$, 차원 불일치) 같은 명령을 반복하지 말고 `.claude/harnesses/empirical_calibration_loop.md`(`.codex` 미러와 동기)의 루프를 따른다: 최소 재현 → 첫 분기 중간값 → 원인 분류(차원 D·구현 I·정밀도 P·convention C·기준선 B·이론 T) → 교정 1건 → 최소 재검. 불일치당 3사이클 상한.
- 교정 가능한 것은 구현 코드, 결과를 보기 전에 선언된 자유 파라미터, sourcer가 출처 검증한 기준선 갱신뿐이다. tolerance·fixture·cutoff·acceptance·seed는 결과를 본 뒤 바꾸지 않는다.
- **식 개정은 계약 수준 행위다.** T(이론) 분류가 확정된 잔차만 식 개정의 근거가 되며, 개정은 (a) 구 판본을 반례·기각 근거와 함께 원장에 보존하고, (b) 새 판본을 새 계약·새 동결 fixture로 세우며, (c) 어떤 관측 기준선(출처 포함)이 개정을 강제했는지 명기한다. 조용한 식 수정, 코드로 이론 잔차 흡수, 판본 기록 없는 덮어쓰기는 금지.
- 실험 결과로 식을 세울 때는 관측에서 무차원 비율들을 falsifier로 먼저 동결하고, 자유 파라미터 수가 재현할 비율 수보다 적어야 하며, 비율별 개별 재조정을 금지한다. 관측 근접은 증명이 아니다.

## main Git 인계 규율

- root/main agent만 Git 상태 변경과 발행을 맡는다. subagent는 읽기 전용 상태·diff·hash만 확인하고 add/commit/fetch/pull/rebase/push/branch/worktree 변경을 하지 않는다.
- main은 루트·branch·upstream·HEAD·remote tip·승인된 변경 경로 manifest를 확인한다. 범위 밖 dirty path는 자동 stash/reset/clean하거나 함께 stage하지 않는다.
- 발행 시 승인 경로만 stage하고 staged diff·관련 검증을 확인한 뒤 `.codex/hooks/check-large-data.cmd --commit`을 명시적으로 실행한다. push 직전에는 같은 실행기의 `--push`도 실행한다. 이 gate는 연구 binary 확장자와 95MB 초과 Git blob을 막으며 secret scanner를 대신하지 않는다. 일반 fast-forward push만 허용하며 force push와 자동 rebase는 금지한다.
- push 후 remote main SHA와 local HEAD를 대조하고 commit SHA, 정확한 발행 경로, 검증, 남은 dirt를 보고한다.

## 뇌 알고리즘 경로 선택

- **최우선 과제는 실제 뇌 식 기반 발견 루프다.** 실제 뇌에서 확립된 기전식($F_{\rm bio}$)과 측정모형($\mathcal H$)을 출발점으로 고정하고, CE 가설을 명시적 추가항($\Delta F_{\rm CE}$)으로 분리한 뒤 실제 뇌 데이터의 held-out 잔차와 개입 falsifier로 판정한다. 정본: `.claude/harnesses/real_brain_equation_discovery_loop.md`(`.codex` 미러와 동기) — 계약 필수 필드 12종(`BIO_STARTING_MECHANISM`…`CLAIM_CEILING`)을 결과 확인 전에 고정하고, 핵심 입력이 `UNVERIFIED`이면 채점 구현으로 진행하지 않는다. 식 개정은 사이클당 구조 변경 1건·새 판본 계약으로만.
- **주축은 두 질문이다: (1) 실제 뇌가 그 연산을 쓰는가(기전 실재성), (2) 실제 뇌 데이터로 반증을 시도했는가(데이터 검증).** 시뮬레이터 성립은 보조 증거다. 정본 규정은 `.claude/harnesses/brain_evidence_ladder.md`(`.codex` 미러와 동기) — 증거 사다리 L0(합성)→L1(관측 비율)→L2(창발 통계)→L3(실데이터 예측)→L4(개입 동일성)와 원시 연산 허용 목록(스파이크·STDP·3-인자·측방억제·Dale·항상성·2시간척도·수면 스케일링)을 따른다.
- **L4 이전에는 "뇌가 이렇게 동작한다"를 쓰지 않는다.** L1–L3 통과는 "정합"이다. 비허용 원시 연산(부호 자유 W, 알고리즘적 WTA, 전역 열거)을 쓴 run은 추상 알고리즘 트랙으로 강등하고 뇌 주장을 금지한다. 모든 뇌 run 기계 상태에 사다리 위치(`BIO_EVIDENCE_Lx`)를 명기한다.
- 새 뇌/기억/의식 run 전에 선행 run의 `12-routes.md`, `31-validation.md`, 존재하면 `40-final-report.md`, 그리고 `_workspace/ce/brain-algorithm-route-ledger.md`를 읽는다. 40이 없으면 가장 늦은 numbered audit와 원장 행을 사용하고 closure 부재를 계약에 기록한다.
- 오케스트레이터는 결과 수치가 양성처럼 보이는 순서가 아니라 사다리 승급 가능성, 인과 식별 가능성, 이전 STOP이 남긴 정보, 독립 falsifier와 대조군, capability dependency로 후보를 정렬한다. endpoint·threshold·seed만 바꾼 재시도는 같은 경로로 보고 퇴역시킨다.
- simulator 결과를 실제 뇌·기억·의식의 동일성 주장으로 승격하지 않는다. 후보 선택과 퇴역 근거를 계약과 원장에 먼저 고정한 뒤 구현한다. L1 비율·L3 데이터는 sourcer 검증 기준선만 쓰고 UNVERIFIED 수치는 게이트에 넣지 않는다.

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
