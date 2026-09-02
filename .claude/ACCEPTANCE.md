# `.claude` 연구 하네스 수용 보고 (2026-09-02)

명세: 사용자 제공 HARNESS_SPEC v1 (첨부 문서. 저장소에는 없음). 구현 순서는 명세 §12대로
`lib/ledger.py` → `lib/verify_derivation.py` → 훅 4 → `settings.json` → 스킬 6 → 에이전트 8 →
`CLAUDE.md` → 드라이버 → 테스트.

## 1. 테스트 결과

`.claude\hooks\python.cmd pytest tests\test_harness.py -q` → **14 passed** (3.4초).
`tests\test_repository_harness.py`와 함께 17 passed. `python.cmd harness` PASS, `source` PASS(8 파일),
`links` 깨진 링크 0 (고아 22는 기존 Phase 2-5 보류 사항).

| # | 시나리오 | 결과 | 비고 |
|---|---|---|---|
| A1 | `src/x.py` Write 시도 | PASS | write-gate exit 2, 허용 목록 메시지. 추가: `ledger/`를 prover가 쓰면 2, ledger-writer/메인은 0, 저장소 밖은 0 |
| A2 | verify 블록 있는 derivation 저장 | PASS | `hook_result.json` 생성, additionalContext에 결과. numeric pass |
| A3 | verify 블록 없는 derivation | PASS | `symbolic: skipped`, reason "no verify block", 차단 없음 |
| A4 | 거짓 항등식 | PASS(조건부) | numeric fail, 차단 없음. **symbolic은 sympy 부재로 skipped** — sympy가 있으면 fail을 단언하도록 테스트가 분기 |
| A5 | active 질문 + 항목 없음 + Stop | PASS | exit 2. prover의 SubagentStop은 통과, active 없으면 통과 |
| A6 | 3회 차단 후 4회째 | PASS | `[2,2,2,0]`, `INCOMPLETE-Q-TEST-1-attempt-2.yaml`(L0/continue) 생성, 이후 Stop 통과 |
| A7 | L3인데 counterexamples 비어 있지 않음 | PASS | `validate` exit 1 "mismatch … L2", Stop exit 2 |
| A8 | `--dry-run` | PASS | priority 최소 Q 출력, `questions.yaml` byte 동일, `logs/` 미생성 |
| A9 | Q-TEST-1 실제 `--max-iters 1` | **미실행** | 수동. `scripts\research-loop.cmd --question Q-TEST-1 --max-iters 1` |
| A10 | adversary 정의 | PASS | tools에 Write/Edit/MultiEdit 없음, Bash 있음. 8 카드 frontmatter 모두 검사 |
| A11 | SKILL.md 6종 | PASS | name/description, 트리거 문장, 본문 ≤150줄. **명세 스킬 6종만** 대상(레거시 ce-* 제외) |
| A12 | 필수 키 빠진 yaml | PASS | exit 1 + "missing keys: attempt, timestamp, claim, level, …" |
| 추가 | after-attempt 상태 전이 | PASS | L1→consecutive_low 1·active, promote→resolved + 하위 질문 open 생성 |

## 2. 명세와 달리 구현한 지점과 이유

| 항목 | 명세 | 구현 | 이유 |
|---|---|---|---|
| 훅 언어 | bash 4종 | `.cmd` 래퍼 → `python.cmd` → `lib/*.py` (`.sh` 쌍둥이 동봉) | 이 저장소는 WSL bash·`.venv`를 피하는 native `.cmd` 위임 구조다. 훅 로직은 어차피 Python이라 셸 층만 바꿨다 |
| 문서 루트 | 명세의 문서 루트(은퇴 경로 이름) | `paper/<분야>/<논문>/` | `문서루트/`는 `repository_harness.py`가 은퇴 경로로 FAIL 처리한다. 코드·문서에 그 문자열을 쓰지 않았다 |
| 쓰기 허용 경로 | 6개 | 6개 + `paper/ .codex/ _workspace/ tests/ examples/ experiments/ benchmarks/ artifacts/` + 루트 설정 파일 | 기존 direct 모드(impl-engineer가 tests/examples 수정)와 `_workspace/` 규율이 살아 있어야 한다. 저장소 밖 경로는 게이트 대상이 아니다 |
| 서브에이전트 식별 | `CLAUDE_AGENT_NAME` 환경변수 | stdin JSON `agent_type` 우선, 환경변수는 보조 | 공식 문서에 환경변수는 없고 `agent_type` 필드가 있다(가이드 에이전트 확인) |
| PostToolUse 출력 | stdout 그대로 | `hookSpecificOutput.additionalContext` JSON | PostToolUse의 plain stdout은 문맥에 들어가지 않는다. additionalContext만 들어간다 |
| `git push` deny | `Bash(git push *)` | `git push --force*`, `git push -f*`만 deny | 루트 AGENTS.md가 사용자 명시 지시 하의 일반 push를 허용한다. force만 금지 |
| Stop/SubagentStop | 동일 스크립트 | SubagentStop은 `agent_type == ledger-writer`일 때만 검사, 식별 불가면 통과 | prover 등 다른 서브에이전트 종료를 막으면 루프가 꼬인다(명세 §8.3.1의 취지) |
| L2 정의 | numeric pass ∧ symbolic ≠ pass | + symbolic pass인데 반례 있음/survived<3도 L2 | 명세 표에 빈 칸이 있어 단조성을 위해 L2로 고정. evidence-ladder 스킬에 명시 |
| `sourcer 실행됨` | 미정의 | `sourcer` 키가 mapping이면 실행, `null`이면 미실행 | 기계가 판정할 수 있는 표현이 필요했다 |
| Lean 면제 | "사람이 명시적으로 면제" | `verification.lean_waived: true` (사람만 적음) | 필드로 고정해야 검증 가능 |
| CLAUDE.md 분량 | 30줄 | 약 45줄 | 이 저장소의 정직성·Git·한국어·닫힘 예산 규율은 Claude에서 CLAUDE.md로만 로드된다. 명세 §10 내용은 모두 포함 |
| 드라이버 | `research-loop.sh` | `scripts/research_loop.py` + `.cmd`/`.sh` 래퍼 | Windows 우선. `claude -p … --max-turns --output-format json`은 명세 그대로 |
| 원장 큐 씨앗 | 없음 | `Q-0001`(진전 원장 §2 표적, priority 1), `Q-TEST-1`(A9용, priority 5) | 드라이버가 실제 표적부터 잡도록. 표적 연결은 `origin` 필드 |
| 커맨드 | `/paper`만 | `/paper` + `/attempt` | 대화형에서 한 attempt를 돌리는 진입점. 에이전트·훅·게이트 추가가 아니다 |
| `.gitignore` | 언급 없음 | `!.claude/ACCEPTANCE.md`, `!verify/**/*.json`, `logs/` | `.claude/*`와 `*.json`이 전역 무시라 증거 파일이 사라진다 |

## 3. 환경 제약 (실측)

- 정책 허용 Python `C:\Python314\python.exe`에 **sympy가 없다**(pyyaml·numpy·pytest는 있음).
  하네스는 설치하지 않는다. 그 결과 `symbolic`은 항상 `skipped`이고 **최고 증거 등급이 L2**다.
  관리자가 `requirements-harness.txt`로 sympy를 설치하면 같은 derivation을 재검증해 L3로 올릴 수 있다.
- `claude` CLI가 PATH에 없으면 드라이버는 exit 127로 멈춘다(dry-run은 영향 없음).

## 4. 기존 구조와의 관계 (삭제·이동 없음)

- `ce-*` 역할 카드 7종, 스킬 8종, 커맨드 9종은 `.codex` 미러로 그대로 두었다. 이 하네스의
  루프 8 역할과 병존한다. 겹치는 이름은 없다(`ledger-writer` ≠ `ce-ledger-writer`: 전자는
  `ledger/` yaml, 후자는 `paper/검증_원장` Markdown).
- `settings.json`의 기존 훅(`goal-reminder`, `check-large-data`)은 유지하고 4종을 추가했다.
- `.codex/`는 건드리지 않았다.

## 5. 제안 (구현하지 않음, 사용자 결정)

1. `ce-status-auditor`·`ce-closure-gate`의 역할은 명세 부록 A대로 verify 훅 + judge + pivot-playbook이
   대체한다. Claude 쪽 `ce-*` 미러 중 `ce-research`·`ce-status-auditor`·`ce-closure-gate`를 은퇴시킬지
   결정이 필요하다(`.codex` 정본과의 동기화 규칙도 함께).
2. `paper/검증_원장`의 Markdown 원장과 `ledger/entries` yaml 원장의 관계: L3 이상 항목을
   Markdown 원장에 옮기는 절차를 `/paper` 또는 ce-ledger-write에 두어야 한다.
3. `repository_harness.py` 필수 경로에 `.claude/hooks/lib/ledger.py`·`verify_derivation.py`를 추가하면
   `harness` 한 번으로 이 하네스의 존재도 검사된다(`.codex` 수정이라 보류).
4. A9 실행 뒤 Q-TEST-1은 `park`(known result) 또는 삭제.

## 6. v2 재설계 (2026-09-02, 사용자 지시)

지시: 최신 모델 사용, 토큰 절약, 에이전트 축소, `ex-claude/`를 보고 `.claude` 재설계, 옛 것 삭제.

- 에이전트 15 → 5. prover(=conjecturer+numerics+ce-math-verifier), adversary(+ce-status-auditor), judge(+ledger-writer), sourcer(+ce-physics-sourcer), paper-writer(+ce-paper-writer+ce-ledger-writer). prover·adversary만 `model: inherit`, judge·paper-writer는 sonnet, sourcer는 haiku.
- `ledger/` 소유자는 judge. `write_gate.py`·`ledger_or_block.py`·`ledger.py` 메시지·`tests/test_harness.py`(A1·A5·A10b)·`scripts/research_loop.py` 프롬프트를 그에 맞게 수정.
- 스킬 14+1 → 10: research-loop·ledger-format·ko-academic-prose 재작성(ce-paper-write 본문 흡수), closure-gate·dimensionless·validate·explain-plan은 ce- 접두사를 뗀 채 유지. ce-research·ce-doc-write·ce-ledger-write·ce-paper-write·clarus-guard-bench 삭제.
- 커맨드 11 → 8: /attempt /paper /explain-plan /audit /dim /validate /status /gc. /ce-research /ce-light /guard-bench 삭제.
- 훅 `.sh` 쌍둥이와 `run.*` 래퍼 삭제. `.codex/`는 손대지 않았고 미러 관계를 끊었다.
- 테스트 결과는 아래 §7에 기록.

## 7. v2 검증 결과 (2026-09-02)

| 명령 | 결과 |
|---|---|
| `python.cmd pytest tests\test_harness.py -q` | 14 passed (A10b는 5 에이전트·`inherit` 허용으로 갱신) |
| `python.cmd harness` | PASS (지침 예산 7683/8192) |
| `python.cmd source .claude\hooks\lib scripts tests\test_harness.py` | PASS (8 파일) |
| `python.cmd links --json` | 깨진 링크 0, 앵커 0, 고아 22(Phase 2 보류) |
| `goal-reminder.cmd`, `session-start.cmd`, `scripts\research-loop.cmd --dry-run` | 정상 출력 |
| `rm -rf ex-claude` | settings.json deny 규칙이 차단 → `rmdir /s /q`로 삭제 (게이트 동작 확인) |
