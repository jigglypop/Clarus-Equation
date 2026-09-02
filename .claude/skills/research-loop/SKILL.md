---
name: research-loop
description: 연구 질문(ledger/questions.yaml의 Q-id)을 잡고 attempt를 돌릴 때, 다음 에이전트를 누구로 할지 정할 때, 루프를 시작·재개·종료할 때 반드시 참조. 탐색 단계에는 게이트를 두지 않고 논문 단계에만 둔다는 원칙을 포함한다. /attempt, research-loop 드라이버, "attempt N을 돌려라" 요청이 트리거다.
---

# 연구 루프 (내부 루프 상태기계)

한 세션 = 한 attempt. 오케스트레이터는 직접 유도·판정하지 않고 위임만 한다.
메인 문맥을 얇게 유지해야 편향 없이 다음 세션을 새 문맥으로 시작할 수 있다.

## 0. 시작

1. `ledger/questions.yaml`에서 `status: active`인 Q를 읽는다. 없으면
   `.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py next-question`으로 올린다.
2. attempt 번호 N은 `bump-attempt <Q>`가 올린 값이다(드라이버 또는 대화형에서 직접).
   Stop 훅이 `attempts` 값의 항목을 찾는다.
3. 진전 원장 §2(훅이 주입)와 Q의 연결을 한 줄로 적는다. 연결이 없으면 `[범위변경]`으로
   사용자에게 묻고 멈춘다.
4. `_workspace/<YYYYMMDD>-<트랙>-<주제>.md` 노트를 만들거나 이어 쓴다(주차장 포함).

## 1. 상태기계 (에이전트 5종)

```
[1] prover(모드: 후보) → 후보 3~5 {claim, falsifier, cost, assumptions_added, target_link}
[2] 오케스트레이터가 1개 선택: 반증조건이 가장 싸게 검사되는 것 우선
[3] prover(모드: 유도) → derivations/<Q>/attempt-NN.derivation.md (verify 블록 필수)
     ↳ 저장 즉시 PostToolUse 훅이 verify_derivation.py 실행, 결과가 문맥에 주입
     ↳ symbolic fail/skipped이면 같은 prover가 verify/<Q>/attempt-NN/check_*.py로 수치 증거
[4] adversary → {counterexamples[], survived_checks[], hidden_assumptions[], status, severity}
[5] sourcer   → adversary가 반례를 못 찾았을 때만 (L3 후보)
[6] judge     → {level, verdict, pivot_step, next_action, open_questions_spawned}
                 + ledger/entries/ 항목 작성 + questions.yaml 갱신 (validate PASS, check-current 0)
     ↳ Stop 훅 ledger-or-block이 항목 존재·스키마·등급 일치를 검사
```

각 에이전트는 Agent 도구로 `subagent_type=<이름>`으로 띄우고, 출력 마지막 fenced
`json` 블록만 다음 단계 입력으로 넘긴다. 역할 카드를 메인 문맥에 읽어 들이지 않는다.
모델은 카드가 정한다(prover·adversary만 inherit, judge·paper-writer는 sonnet, sourcer는 haiku).

## 2. 호출 생략 규칙 (속도·토큰)

- attempt 2+에서는 prover 후보 모드를 생략하고 `questions.yaml`의 `notes`에 저장된
  잔여 후보를 쓴다. pivot 뒤에는 다시 부른다.
- 훅 symbolic pass면 수치 스크립트 생략. adversary가 반례를 찾으면 sourcer 생략.
- 같은 byte의 green 검증을 반복하지 않는다.
- 각 에이전트의 `parking` 목록은 workspace 노트 주차장에 옮기고 실행하지 않는다.

## 3. verdict별 다음 반복

| verdict | 질문 상태 | 다음 attempt |
|---|---|---|
| continue | active | 같은 경로에서 유도 보강 |
| pivot | active, `pivots_tried` += pivot_step | prover 후보 모드에 pivot_step 주입 (pivot-playbook) |
| promote | resolved | 하위 질문을 open으로 큐에 추가 |
| park | parked | notes에 사유 |

상태 전이는 `ledger.py after-attempt <Q> <N>`이 결정적으로 수행한다.
`consecutive_low`가 3이면 escalated로 바뀌어 사람에게 넘어간다.

## 4. 종료

1. judge가 `validate` PASS와 `check-current` 0을 보고했는지 본다.
2. `after-attempt <Q> <N>`을 실행한다. 진전 원장 §7에 한 행을 남긴다.
3. 마지막 메시지: 진전 종류(level·verdict) → 항목 id → 유도·검증 경로 → 반례 수 →
   주차장 항목 수 → 다음 attempt의 첫 단계 한 줄.

## 5. 원칙

- 탐색은 fail-open: 검증 실패는 문맥에 들어오되 차단하지 않는다. 논문은
  fail-closed: L3 미만은 인용 금지(evidence-ladder). 게이트는 논문 단계에만 있다.
- 반례는 정지 신호가 아니라 경로 전환 신호다(pivot-playbook).
- 어려우면 adversary를 먼저 부른다. 부수는 것이 세우는 것보다 싸다.
- 하네스 상태 문자열(pass/fail)을 이론 지위처럼 `paper/`에 복사하지 않는다.
