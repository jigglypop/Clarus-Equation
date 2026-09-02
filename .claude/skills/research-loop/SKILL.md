---
name: research-loop
description: 연구 질문(ledger/questions.yaml의 Q-id)을 잡고 attempt를 돌릴 때, 다음 에이전트를 누구로 할지 정할 때, 루프를 시작·재개·종료할 때 반드시 참조. 식을 먼저 세우고(추측 카드) 사다리 한 단씩 증명하는 추측 우선 상태기계이며, 탐색 단계에는 게이트를 두지 않고 논문 단계에만 둔다. /attempt, /conjecture, research-loop 드라이버, "attempt N을 돌려라" 요청이 트리거다.
---

# 연구 루프 (내부 루프 상태기계, 추측 우선)

한 세션 = 한 attempt. 오케스트레이터는 직접 유도·판정하지 않고 위임만 한다. 메인 문맥을
얇게 유지해야 다음 세션을 새 문맥으로 시작할 수 있다. attempt는 두 종류뿐이다. **카드 attempt**
(식을 세운다)와 **사다리 attempt**(단 하나를 닫는다). 카드 없는 사다리 attempt는 없다.
이유: 보조정리부터 고르면 문헌으로 수렴한다. 식이 먼저 있어야 증명할 것이 생긴다.

**일괄 카드 모드(사용자 지시 시만).** 사용자가 "최대한 많은 추측식"을 지시하면 한 세션에 카드 attempt를
질문마다 하나씩 **병렬**로 돌리고(질문 등록 → prover ×N → adversary ×N → sourcer ×N → judge ×N,
judge는 항목 파일만 쓰고 `after-attempt`는 오케스트레이터가 질문 순서대로 실행), 사다리 attempt는
같은 질문에서 **연속**으로 이어 갈 수 있다. 채택된 카드가 여럿이면 `after-attempt`가 하나만 active로
두고 나머지는 open 큐(사다리는 열린 채)에 남기며 `next-question`이 순서대로 꺼낸다. 병렬 judge가
`questions.yaml`·`index.md`를 동시에 쓰지 않도록 judge는 항목 파일과 `validate`만 담당한다.

## 0. 시작

1. active Q를 읽는다. 없으면 `.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py next-question`.
2. `bump-attempt <Q>`로 attempt 번호 N. Stop 훅이 이 번호의 항목을 찾는다.
3. `ledger.py ladder <Q>`로 카드·다음 단·`force_pivot`을 본다.
   - `card` 없음 또는 `force_pivot: conjecture` → 이번은 **카드 attempt**(§1).
   - `card` 있음 → **사다리 attempt**(§2), 대상은 `next` 단 하나.
   - `kind: kill_test`(문헌 kill test)는 카드를 면제한다. 통과 뒤 파생 질문은 카드로 시작한다.
4. 진전 원장 §2(훅이 주입)와 Q의 연결을 한 줄로 적는다. 없으면 `[범위변경]`으로 묻고 멈춘다.
5. `_workspace/<YYYYMMDD>-<트랙>-<주제>.md` 노트를 만들거나 이어 쓴다(주차장 포함).

## 1. 카드 attempt (conjecture-first)

```
[1] prover(모드: 추측) → derivations/<Q>/F-NN.formula.md  (저장 즉시 PostToolUse 훅 verify)
     ↳ `ledger.py card-check <파일>` PASS까지 같은 prover가 최대 2회 수정
[2] adversary(모드: 카드 감사 6종) → {status, severity, counterexamples(극한 파괴 등), hidden_assumptions}
     ↳ P1이면 prover가 같은 세션에서 카드 1회 수정 → adversary 재감사
[3] sourcer(모드: 신규성) → prior_art[].relation. identical·special_case면 카드는 죽는다 (필수 호출)
[4] judge → adopt(카드 등록, 사다리 열림, 진전 종류 "예측") | refute
     ↳ refute(재발견)이면 같은 세션에서 [1]로 돌아가 F-(NN+1)을 더 강하게 한 번 더 세운다.
        두 번째도 죽으면 항목을 남기고 종료(force_pivot은 유지된다)
```

## 2. 사다리 attempt

```
[1] 대상 단의 kind로 분기
     외부기존   → sourcer 인용만. judge가 ladder_cited로 닫는다. 유도 없음, 같은 attempt에서 다음 단 계속
     보조정리   → prover(모드: 유도) derivations/<Q>/attempt-NN.derivation.md (verify 블록 필수, 훅 검증)
     수치시험·예측시험 → prover가 verify/<Q>/attempt-NN/check_*.py (씨앗 20260902, tol·숫자는 카드에서)
[2] adversary → {counterexamples[], survived_checks[], hidden_assumptions[], status, severity}
[3] sourcer   → 반례 없을 때만(L3 후보). 보조정리가 문헌에 있으면 relation을 적고 그 단은 cited로 닫힌다
[4] judge     → promote(단 closed) | continue | pivot | refute(kill 발동, kill_triggered 필수)
                 + ledger/entries/ 항목(ladder_step 필수) + questions.yaml (validate PASS, check-current 0)
     ↳ Stop 훅 ledger-or-block이 항목 존재·스키마·등급 일치를 검사
```

prover 후보 모드는 카드 attempt에서 쓰지 않는다(카드가 후보다). 사다리 단이 막혀 pivot이
필요할 때만 부르고, 그때도 선택 기준은 **반증 가능성·신규성 우선, 비용은 그 다음**이다.
각 에이전트는 Agent 도구로 `subagent_type=<이름>`으로 띄우고 출력 마지막 fenced `json`만 넘긴다.
역할 카드를 메인 문맥에 읽어 들이지 않는다. 모델은 카드가 정한다.

## 3. 호출 생략 규칙 (속도·토큰)

- 훅 symbolic pass면 수치 스크립트 생략. adversary가 반례를 찾으면 sourcer 생략 — 단 카드
  attempt의 신규성 대조는 항상 한다.
- 외부기존 단은 attempt를 쓰지 않는다. 한 attempt에 cited 단 여러 개 + 보조정리 단 하나까지.
- 같은 byte의 green 검증을 반복하지 않는다. 각 출력의 `parking`은 노트 주차장에 옮기고 실행하지 않는다.

## 4. verdict별 다음 반복

| verdict | 질문 상태 | 다음 attempt |
|---|---|---|
| adopt | active, kind=conjecture, 사다리 전부 open | 1단부터 |
| promote (단) | active; 마지막 단이면 resolved(카드 status 정리) | 다음 open 단; 끝나면 /paper |
| continue | active | 같은 단 유도 보강 |
| pivot | active, `pivots_tried` += step | 확장(conjecture·generalize)이면 새 카드, 축소면 같은 단 다른 모양 |
| refute (kill 발동, 채택 카드) | parked, 진전 원장 §4 | 새 질문 또는 사용자 결정 |
| refute (재발견) | active, `force_pivot: conjecture` | 더 강한 카드 |
| refute (반례, 채택 전 카드) | active, `force_pivot: conjecture` — 반례는 축소 pivot 신호이지 질문의 죽음이 아니다 | 같은 세션에서 보정 카드 F-(NN+1) |
| park | parked | notes에 사유 |

상태 전이는 `ledger.py after-attempt <Q> <N>`이 결정적으로 수행한다. 재발견 2회 또는 축소
4단계 소진이면 `force_pivot: conjecture`가 자동으로 붙고 judge는 이를 어길 수 없다.
`consecutive_low`가 3이면 escalated로 바뀌어 사람에게 넘어간다.

## 5. 종료

1. judge가 `validate` PASS와 `check-current` 0을 보고했는지 본다.
2. `after-attempt <Q> <N>`. 진전 원장 §2(현재 추측·다음 단)와 §7에 한 행.
3. 마지막 메시지: 진전 종류(예측·닫힘·기각·축소·개시·정리) → 항목 id → 카드/단 →
   반례 수 → 주차장 항목 수 → 다음 attempt의 첫 단계 한 줄.

## 6. 원칙

- 탐색은 fail-open, 논문은 fail-closed(L3 미만 인용 금지). 게이트는 논문 단계에만 있다.
- 식이 먼저다. 유도 없이 세우고, kill이 죽이고, 사다리가 증명한다. 안전한 추측은 재발견이다.
- 반례는 경로 전환 신호, 재발견은 확장 신호다(pivot-playbook).
- 어려우면 adversary를 먼저 부른다. 부수는 것이 세우는 것보다 싸다.
- 하네스 상태 문자열(pass/fail)을 이론 지위처럼 `paper/`에 복사하지 않는다.
