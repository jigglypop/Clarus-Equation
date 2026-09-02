---
description: 연구 루프 attempt 한 바퀴 (prover→adversary→sourcer→judge) 대화형 실행
---

research-loop 스킬에 따라 attempt 한 바퀴를 돌려라. 오케스트레이터는 직접 유도·판정하지 않고 Agent 도구로 위임만 한다.

질문: $ARGUMENTS (비어 있으면 `ledger.py next-question`이 고른 것)

1. `.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py next-question`으로 active 질문을 정하고 `bump-attempt <Q>`로 attempt 번호 N을 받는다. 진전 원장 §2 표적과의 연결을 한 줄로 적는다. 연결이 없으면 `[범위변경]`으로 묻고 멈춘다. `_workspace/` 노트를 연다.
2. prover(모드: 후보; attempt 2+이면 questions.yaml notes의 잔여 후보 재사용) → 후보 1개 선택(반증조건이 가장 싸게 검사되는 것) → prover(모드: 유도) → adversary → (반례 없을 때만 sourcer) → judge 순으로 위임한다. 각 단계는 직전 단계의 마지막 fenced json만 입력으로 받는다. 각 출력의 `parking`은 노트 주차장에 옮기고 실행하지 않는다.
3. judge가 `validate` PASS와 `check-current` 0을 보고했는지 확인한다.
4. `ledger.py after-attempt <Q> <N>`을 실행해 질문 상태를 전이시키고 진전 원장 §7에 한 행을 남긴다.
5. 마지막 메시지: 항목 id, level/verdict, 유도·검증 경로, 반례 수, 주차장 항목 수, 질문의 새 status, 다음 attempt의 첫 단계 한 줄.
