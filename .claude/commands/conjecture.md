---
description: 추측 카드만 세운다 (예측식·예산식 → 카드 감사 → 신규성 → adopt/refute). 새 질문 개시나 재추측에 쓴다
---

conjecture-first 스킬에 따라 카드 attempt 하나를 돌려라. 오케스트레이터는 직접 식을 세우거나 판정하지 않고 Agent 도구로 위임만 한다.

대상: $ARGUMENTS (Q-id, 또는 "새 질문: <제목>" — 새 질문이면 먼저 `ledger.py add-question --id Q-xxxx --title "..." --kind conjecture --priority 1 --origin "paper/진전_원장.md §2"`로 등록하고 `next-question`·`bump-attempt`로 attempt 1을 연다. 대상이 여러 개(쉼표 구분 또는 "일괄 N개")면 research-loop §0 "일괄 카드 모드"로 질문마다 prover→adversary→sourcer→judge를 병렬로 돌리고 `after-attempt`만 순서대로 실행한다)

1. 진전 원장 §2와의 연결을 한 줄로 적는다. 카드가 겨냥할 **양 하나**(비율·장부·분포·스펙트럼)를 사용자 요청과 §2에서 고정한다. 없으면 `[범위변경]`으로 묻고 멈춘다.
2. prover(모드: 추측): 예측식 또는 예산식 카드 `derivations/<Q>/F-NN.formula.md`. 숫자(또는 항등식)·극한 복원·kill≥2·사다리≤7단·novelty·verify. `card-check` PASS까지.
3. adversary(카드 감사 6종) → P1이면 prover 수정 1회 → sourcer(신규성 대조, 필수) → judge(adopt|refute). refute(재발견)면 2번부터 더 강한 F-(NN+1)로 한 번 더.
4. `ledger.py after-attempt <Q> <N>` → `ledger.py ladder <Q>`로 사다리가 열렸는지 확인. 진전 원장 §2 "현재 추측"·§5 카드 행·§7 세션 로그(종류: 예측 또는 정리) 갱신.
5. 마지막 메시지: 카드 id·종류·식 한 줄, 사전등록 숫자(또는 항등식), kill 2개, 사다리 단 수, adopt/refute와 사유, 다음 attempt가 닫을 1단.
