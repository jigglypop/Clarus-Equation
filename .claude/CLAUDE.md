# CE 연구 하네스 (Claude Code)

## 정본·표적
- 정본: `ledger/`(yaml 원장·큐) `derivations/` `verify/` `paper/`(제자리 갱신). 시행착오·주차장은 `_workspace/<YYYYMMDD>-<트랙>-<주제>.md`(상한 20개·48KB·640KB·21일, harness가 강제).
- 표적은 `paper/진전_원장.md` §2 하나. 훅이 매 프롬프트에 주입. 첫 메시지에 "이번 세션이 닫을 고리" 한 문장. 무관한 요청은 `[범위변경]`으로 묻고, 표적 변경은 사용자만.
- 떠오른 아이디어는 실행 말고 주차장에 한 줄. 툴 호출 10개마다 `[정렬]`/`[이탈]`, `[이탈]` 2연속이면 표적 복귀.

## 에이전트 5종 (오케스트레이터는 위임만, 카드를 읽지 않음, 출력의 마지막 json만 전달)
prover(후보·유도, inherit) · adversary(반례·지위 감사, inherit) · judge(등급·verdict·ledger 기록, sonnet) · paper-writer(md 원장→원고, sonnet) · sourcer(문헌·기준선, haiku). inherit는 결과를 바꾸는 두 자리에만.

## 루프
한 세션 = 한 attempt: prover(후보)→선택→prover(유도, 훅 검증)→adversary→(sourcer)→judge. Stop 훅이 원장 항목을 요구(3회 후 INCOMPLETE).
L0~L4(evidence-ladder). **L3 이상만 논문 인용.** 반례는 정지가 아니라 pivot(partial→alt_derivation→reformulate→weaken). 탐색 fail-open, 논문 fail-closed.

## 라우팅
spot(단순 질문·한 줄 수정·하네스 최소 수정, 스킬 없음) · /attempt · /paper · /explain-plan · /audit /dim /validate · /status /gc. 코드(examples·tests)는 메인이 직접, 가장 작은 검증 하나.

## 명령 (정책 허용 시스템 Python만, `.venv`·uv·pip install·대화형 프롬프트 금지)
    .claude\hooks\python.cmd doctor|harness|source <p>|pytest <t> -q|links [--strict]|lint
    .claude\hooks\python.cmd python .claude\hooks\lib\ledger.py summary|validate <f>|check-current|next-question|bump-attempt <Q>|after-attempt <Q> <N>|reindex
    scripts\research-loop.cmd [--max-iters N] [--question Q] [--dry-run]
전체 pytest는 사용자가 `전체`/`full` 명시 때만. sympy 없으면 symbolic skipped(최고 L2).

## 정직성 (협상 불가)
실패를 통과로 쓰지 않는다. 안 돌린 검증을 돌렸다고 쓰지 않는다. 기계 문자열(pass/PASS)을 `paper/`에 지위처럼 쓰지 않는다.
지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐. 관측 근접은 증명이 아니다. 진전은 닫힘·기각·축소뿐(새 장·항목·게이트 분할은 개시·정리). 하위 게이트 3개 초과면 §2에 택일 기록 후 진행. tolerance·fixture·seed는 결과 본 뒤 안 바꾼다.

## 한국어·논문
툴 호출 사이 한국어 1–3문장. 판단 설명은 목표 계약→1학년 LaTeX→비유→정렬→지위. `paper/`는 결과 먼저, 과정·식별자 금지, 영어는 첫 등장 병기 후 한국어, 장 머리 세 줄 상자. 종료 전 진전 원장 §2·§7 갱신 + `links`.

## Git
루트 `AGENTS.md` main-only. subagent는 commit/push 금지. 발행은 명시 지시 때만, `check-large-data.cmd --commit/--push`. force push 금지. 메시지 `<종류>: <항목> — <한 줄>`(닫힘·기각·축소·개시·정리·하네스), "new" 금지.
