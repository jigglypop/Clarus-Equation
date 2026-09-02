# CE Claude 하네스 (v2, 2026-09-02)

Claude Code용 Clarus-Equation 연구 하네스. 가설→유도→기계검증→반증→판정으로 도는
**내부 루프**를 게이트 없이 빠르게 돌리고, 통과한 결과(L3 이상)만 원장(`ledger/`)에
쌓아 논문 단계의 **외부 루프**(/paper)로 넘긴다. v1(HARNESS_SPEC, `ACCEPTANCE.md`)의
루프·훅·원장 코드를 그대로 쓰되 에이전트를 15종에서 5종으로 줄이고, 옛 `ce-*` 미러와
제거된 제품의 잔존물을 지웠다. `.codex/`는 Codex 전용이며 더 이상 미러 관계가 아니다.

## 설계 원칙 (하네스 설계론)

| 층 | 원칙 | 여기서는 |
|---|---|---|
| 지침 | 항상 로드되는 `CLAUDE.md`는 얇게. 규칙은 "무엇을"이 아니라 "왜"까지 한 줄 | 약 50줄 |
| 지식 | 스킬은 필요할 때만 로드(점진 공개). 본문 ≤150줄, 상세는 `references/` | 10종 |
| 강제 | 사람이 지키라고 쓰지 말고 훅(결정적 코드)이 막는다 | 훅 6종, `ledger.py`·`verify_derivation.py` |
| 격리 | 서브에이전트는 문맥 오염 방지·최소 도구·역할별 모델 목적으로만 | 5종. `inherit`는 결과를 바꾸는 prover·adversary 둘만, 나머지는 sonnet·haiku |
| 상태 | 기억은 파일에만. 세션은 새 문맥으로 시작 | `ledger/`, `paper/진전_원장.md`, `_workspace/` |
| 게이트 | 탐색은 fail-open, 논문은 fail-closed | L3 미만 인용 금지 |
| 권한 | allowlist로 프롬프트 최소화, deny로 되돌릴 수 없는 것만 막기 | `settings.json` |

## 에이전트 5종 (15 → 5)

| 이름 | 흡수한 옛 역할 | 모델 | 이유 |
|---|---|---|---|
| prover | conjecturer, numerics, ce-math-verifier(routes) | inherit | 유도 품질이 루프의 상한. 후보·유도·수치 스크립트를 한 문맥에서 |
| adversary | ce-status-auditor(형식 지위·P0/P1 감사) | inherit | 부수는 쪽이 약하면 L3가 오염된다 |
| judge | ledger-writer | sonnet | 등급은 표 대조, 기록은 `ledger.py validate`가 검증. 강한 모델이 필요 없다 |
| sourcer | ce-physics-sourcer | haiku | 검색·분류만 |
| paper-writer | ce-paper-writer, ce-ledger-writer(md 원장) | sonnet | 규칙(ko-academic-prose)에 따른 문체 작업. 원장→원고 두 단계 |

삭제: ce-impl-engineer(메인 세션이 직접 코드 수정), ce-explanation-planner(스킬 `explain-plan`으로만 유지), clarus-guard-bench(제거된 제품), `ce-research` 레인 파이프라인(research-loop이 대체).

## 구조

```
CLAUDE.md            프로젝트 지침
settings.json        permissions + 훅 6종
agents/              prover adversary sourcer judge paper-writer
skills/              research-loop evidence-ladder pivot-playbook derivation-style ledger-format ko-academic-prose
                     closure-gate dimensionless explain-plan validate
commands/            /attempt /paper /explain-plan /audit /dim /validate /status /gc
hooks/
  session-start.cmd  SessionStart → lib/session_start.py (원장 요약 40줄)
  goal-reminder.cmd  UserPromptSubmit → .codex/hooks/goal_reminder.py (진전 원장 §2 주입)
  check-large-data.cmd PreToolUse Bash|PowerShell → 데이터 반출 게이트
  write-gate.cmd     PreToolUse Write|Edit → lib/write_gate.py (허용 경로, ledger/는 judge만)
  verify-on-save.cmd PostToolUse Write|Edit → lib/verify_on_save.py (derivation 자동 검증, fail-open)
  ledger-or-block.cmd Stop·SubagentStop → lib/ledger_or_block.py (원장 항목 없으면 exit 2, 3회 후 INCOMPLETE)
  python.cmd         .codex/hooks/python.cmd 위임 (doctor harness source pytest links lint)
  lib/ledger.py      원장 읽기·쓰기·검증·상태 전이
  lib/verify_derivation.py  verify 블록 기계검증 (sympy 선택, numpy 필수)
../ledger/  ../derivations/  ../verify/  ../scripts/research_loop.py  ../tests/test_harness.py
```

## 루프 한 바퀴

prover(후보 3~5, 반증조건) → 오케스트레이터가 1개 선택 → prover(derivation + verify 블록,
저장 즉시 훅 검증, 필요 시 수치 스크립트) → adversary(반례·survived·숨은 가정·지위) →
[반례 없을 때 sourcer] → judge(L0~L4, verdict, 원장 항목 + questions.yaml). Stop 훅이 항목
존재·스키마·등급 일치를 검사한다. 상태 전이는 `ledger.py after-attempt`.

## 설치·검증

```
.claude\hooks\python.cmd doctor
.claude\hooks\python.cmd harness
.claude\hooks\python.cmd pytest tests\test_harness.py -q
scripts\research-loop.cmd --dry-run
```

의존성은 `requirements-harness.txt`(pyyaml, sympy, numpy, pytest). 하네스는 설치하지
않는다. **sympy가 없으면 기호검증이 항상 skipped라 최고 등급이 L2다.**

## 주의

- 훅은 빌드하지 않는다. `.cmd` → `python.cmd` → `lib/*.py` 경로만 쓴다(WSL bash·`.venv`·
  uv Python 회피). `.sh` 쌍둥이는 제거했다.
- `verify/**/*.json`은 `.gitignore` 예외로 추적된다. `logs/`는 추적하지 않는다.
- 역할 카드·스킬 본문은 `.claude`가 정본이다. `.codex`와 동기화하지 않는다.
