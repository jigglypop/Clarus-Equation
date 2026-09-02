---
name: ledger-format
description: ledger/ 아래 파일(questions.yaml, entries/*.yaml, index.md)을 읽거나 쓸 때, questions.yaml을 갱신할 때, 원장 항목을 새로 만들 때 반드시 참조. judge 전용. paper/검증_원장의 Markdown 원장 규칙은 마지막 절에 있다(paper-writer 1단계).
---

# 연구 원장 형식

원장은 **항목식**이다. 산문 금지. 원장은 기계(`ledger.py`)가 읽는다. 쓰는 주체는
judge 하나이며 write-gate 훅이 다른 서브에이전트의 `ledger/` 쓰기를 막는다. 메인
세션(사람)은 직접 고칠 수 있다.

## 1. 열린 문제 큐 — `ledger/questions.yaml`

```yaml
questions:
  - id: Q-0007
    title: "비콤팩트 집중 정리 6.1의 경계항 소거 조건"
    status: open            # open | active | resolved | parked | escalated
    priority: 2             # 1(높음)~5
    origin: E-20260901-003  # 이 질문을 낳은 항목. 없으면 human 또는 paper 경로
    attempts: 3             # bump-attempt가 증가
    consecutive_low: 1      # 연속 L1 이하. 3이면 escalated
    pivots_tried: [partial, alt_derivation]
    notes: ""               # prover 후보 모드 잔여 후보, park 사유
```

- `active`는 동시에 최대 1개. `ledger.py next-question`이 open 중 priority 최소(동률이면
  id 오름차순)를 active로 올린다.
- 상태 전이는 `ledger.py after-attempt <Q> <N>`만 한다. 손으로 status를 바꾸지 않는다.
- 새 질문은 `ledger.py add-question --id Q-xxxx --title "..." --priority 3 --origin E-...`.

## 2. 항목 — `ledger/entries/YYYYMMDD-NNN-<slug>.yaml`

```yaml
id: E-20260902-004          # judge가 부여. 같은 날 NNN 증가
question: Q-0007
attempt: 4
timestamp: 2026-09-02T14:03:11+09:00
claim: "경계항은 ρ(x)=O(|x|^{-n-1}) 가정 하에서 소거된다."
level: L3                   # evidence-ladder. 근거 필드의 함수
verdict: promote            # continue | pivot | promote | park
pivot_step: null            # verdict가 pivot이면 필수: partial | alt_derivation | reformulate | weaken
derivation: derivations/Q-0007/attempt-04.derivation.md   # 존재해야 함. L0면 null
verification:
  symbolic: pass            # pass | fail | skipped  (훅·스크립트 결과 그대로)
  numeric: pass
  lean: skipped
  artifacts: verify/Q-0007/attempt-04/
adversary:
  counterexamples: []       # 각 {input, expected, observed, note}
  survived_checks: [dimension, limit_n_to_1, boundary_rho_zero]
sourcer:                    # 실행하지 않았으면 null. 실행했으면 mapping (빈 prior_art라도)
  prior_art: []             # 각 {ref, relation: identical|special_case|generalizes|unrelated, note}
assumptions:
  - "ρ ∈ C^1 이며 ρ(x)=O(|x|^{-n-1})"
open_questions_spawned: [Q-0009]   # 또는 [{id, title, priority}]
next_action: "논문 6.1절 초안에 반영"
```

- 필수 키: id, question, attempt, timestamp, claim, level, verdict, derivation, verification,
  adversary, assumptions, next_action. 나머지는 선택.
- `level`은 `ledger.py`가 근거 필드로 재계산해 불일치면 FAIL. FAIL이면 근거 필드가
  사실과 다른지 먼저 본다.
- 항목은 수정하지 않고 새 attempt로 새 항목을 만든다. 예외: 사람의 명시적 정정.

## 3. 명령

```
.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py validate ledger/entries/<file>
.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py check-current
.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py reindex
.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py summary
```

쓰기 전 `validate` PASS, 쓴 뒤 `reindex`. `index.md`는 자동 생성물이며 손으로 고치지 않는다.

## 4. INCOMPLETE 항목

Stop 훅이 3회 차단 후 `INCOMPLETE-<Q>-attempt-N.yaml`(L0, continue)을 자동 생성한다.
다음 attempt는 N+1로 간다. 이 항목은 "기록 없이 끝난 attempt"라는 사실 자체를 남긴다.

## 5. paper/검증_원장 Markdown 원장 (paper-writer 1단계)

- 대상: `paper/검증_원장/*.md`, `paper/진전_원장.md`. 입력은 `ledger/entries/`의 L3 이상
  항목뿐이며 각주 `[E-YYYYMMDD-NNN]`을 남긴다.
- 지위 태그는 정의·정리·공리·산출·경험식·미완성·예측 7종. 표와 짧은 항목만 쓰고
  산문·서사를 넣지 않는다. 관측 근접을 정리·산출로 승격하지 않는다.
- 완전한 반례가 있는 부모 주장은 삭제하고 살아남는 좁은 명제와 반례 범위를 남긴다.
- 원장을 먼저 안정화하고 편집을 멈춘 뒤에 원고(2단계)를 쓴다. 같은 세션에서 원장과
  원고를 오가며 고치지 않는다.
- 항목 앵커는 `<a id="qnb-eNN-a"></a>` 형식으로 두고 `python.cmd links`로 검사한다.
