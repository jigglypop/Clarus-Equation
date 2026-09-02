---
description: 연구 루프 attempt 한 바퀴 (카드 없으면 추측 카드, 있으면 사다리 한 단) 대화형 실행
---

research-loop 스킬에 따라 attempt 한 바퀴를 돌려라. 오케스트레이터는 직접 유도·판정하지 않고 Agent 도구로 위임만 한다.

질문: $ARGUMENTS (비어 있으면 `ledger.py next-question`이 고른 것)

1. `.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py next-question`으로 active 질문을 정하고 `bump-attempt <Q>`로 attempt 번호 N을 받는다. `ledger.py ladder <Q>`로 카드·다음 단·`force_pivot`을 본다. 진전 원장 §2 표적과의 연결을 한 줄로 적는다. 연결이 없으면 `[범위변경]`으로 묻고 멈춘다. `_workspace/` 노트를 연다.
2. **카드가 없거나 `force_pivot: conjecture`면 카드 attempt**: prover(모드: 추측) → adversary(카드 감사) → sourcer(신규성, 필수) → judge(adopt|refute). refute(재발견)면 같은 세션에서 prover(추측)부터 한 번 더, 더 강하게.
   **카드가 있으면 사다리 attempt**: 다음 open 단 하나. 외부기존 단은 sourcer 인용 → judge `ladder_cited`. 보조정리·수치시험·예측시험 단은 prover(모드: 유도) → adversary → (반례 없을 때만 sourcer) → judge. 각 단계는 직전 단계의 마지막 fenced json만 입력으로 받는다. 각 출력의 `parking`은 노트 주차장에 옮기고 실행하지 않는다.
3. judge가 `validate` PASS와 `check-current` 0을 보고했는지 확인한다.
4. `ledger.py after-attempt <Q> <N>`을 실행해 질문·사다리 상태를 전이시키고 진전 원장 §2(현재 추측·다음 단)·§7에 반영한다.
5. 마지막 메시지: 진전 종류(예측·닫힘·기각·축소·개시·정리), 항목 id, level/verdict, 카드/단, 반례 수, 주차장 항목 수, 질문의 새 status, 다음 attempt의 첫 단계 한 줄.
