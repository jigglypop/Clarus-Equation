# CE 연구 하네스 (Claude Code)

## 정본·표적
- 정본: `ledger/`(yaml 원장·큐) `derivations/`(유도·추측 카드) `verify/` `paper/`(제자리 갱신). 시행착오·주차장은 `_workspace/<YYYYMMDD>-<트랙>-<주제>.md`(상한 20개·48KB·640KB·21일, harness가 강제).
- 표적은 `paper/진전_원장.md` §2 하나. 훅이 매 프롬프트에 주입. 첫 메시지에 "이번 세션이 닫을 고리" 한 문장. 무관한 요청은 `[범위변경]`으로 묻고, 표적 변경은 사용자만.
- 떠오른 아이디어는 실행 말고 주차장에 한 줄. 툴 호출 10개마다 `[정렬]`/`[이탈]`, `[이탈]` 2연속이면 표적 복귀.

## 추측 우선 (재발견 금지)
식이 먼저다. 질문에 추측 카드(`derivations/<Q>/F-NN.formula.md`, 예측식 또는 예산식)가 없으면 attempt는 카드 작성이다: 유도 없이 `[공리: 후보]`로 선언, 숫자(또는 장부 항등식)·극한 복원·kill≥2·사다리≤7단 사전등록, sourcer 신규성 필수, judge adopt/refute. 그 뒤 attempt 하나가 사다리 한 단을 닫는다. 재발견(identical·special_case)은 정지가 아니라 확장 신호: 카드는 refute 후 더 강한 카드, 보조정리는 cited로 닫고 다음 단, 질문을 park하지 않는다. 재발견 2회·축소 4단계 소진이면 `force_pivot: conjecture`가 자동으로 붙는다. 틀릴 수 있는 만큼 강하게 세우고, 결과를 본 뒤 숫자·kill·tol을 바꾸지 않는다.

## 에이전트 5종 (오케스트레이터는 위임만, 카드를 읽지 않음, 출력의 마지막 json만 전달)
prover(추측·후보·유도, inherit) · adversary(반례·카드 감사·지위, opus — Fable 안전장치 오탐 회피) · judge(등급·verdict·ledger 기록·사다리, sonnet) · paper-writer(md 원장→원고, sonnet) · sourcer(신규성·문헌·기준선, haiku). inherit는 prover만.

## 루프
한 세션 = 한 attempt(사용자가 일괄을 지시하면 카드 attempt 여러 질문 병렬·사다리 연속 허용, 채택 카드는 하나만 active). 카드 attempt: prover(추측)→adversary(카드 감사)→sourcer(신규성)→judge(adopt|refute). 채택 전 카드의 반례 기각은 질문을 살려 재추측. 사다리 attempt: prover(유도·수치)→adversary→(sourcer)→judge(promote 단 닫힘|continue|pivot|refute kill). Stop 훅이 원장 항목을 요구(3회 후 INCOMPLETE).
L0~L4(evidence-ladder). **L3 이상만 논문 인용.** 반례는 축소 pivot(partial→alt_derivation→reformulate→weaken), 재발견은 확장 pivot(conjecture→generalize). 탐색 fail-open, 논문 fail-closed.

## 라우팅
spot(단순 질문·한 줄 수정·하네스 최소 수정, 스킬 없음) · /attempt · /conjecture · /paper · /explain-plan · /audit /dim /validate · /status /gc. 코드(examples·tests)는 메인이 직접, 가장 작은 검증 하나.

## 명령 (정책 허용 시스템 Python만, `.venv`·uv·pip install·대화형 프롬프트 금지)
    .claude\hooks\python.cmd doctor|harness|source <p>|pytest <t> -q|links [--strict]|lint
    .claude\hooks\python.cmd python .claude\hooks\lib\ledger.py summary|validate <f>|check-current|next-question|bump-attempt <Q>|after-attempt <Q> <N>|ladder <Q>|card-check <f>|reindex
    scripts\research-loop.cmd [--max-iters N] [--question Q] [--dry-run]
전체 pytest는 사용자가 `전체`/`full` 명시 때만. sympy 없으면 symbolic skipped(최고 L2).

## 정직성 (협상 불가)
실패를 통과로 쓰지 않는다. 안 돌린 검증을 돌렸다고 쓰지 않는다. 기계 문자열(pass/PASS)을 `paper/`에 지위처럼 쓰지 않는다.
지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐(채택 카드는 `[공리: 후보]`, 그 숫자는 `[예측: 사전등록]`, 사다리 완주 뒤 `[정리]`). 관측 근접은 증명이 아니다. 진전은 예측(카드 채택)·닫힘·기각·축소 넷(새 장·항목·게이트 분할은 개시·정리). 하위 게이트 3개 초과면 §2에 택일 기록 후 진행. tolerance·fixture·seed·예측 숫자·kill은 결과 본 뒤 안 바꾼다.

## 한국어·논문
툴 호출 사이 한국어 1–3문장. 판단 설명은 목표 계약→1학년 LaTeX→비유→정렬→지위. `paper/`는 결과 먼저, 과정·식별자 금지, 영어는 첫 등장 병기 후 한국어, 장 머리 세 줄 상자. 종료 전 진전 원장 §2·§7 갱신 + `links`.

## Git
루트 `AGENTS.md` main-only. subagent는 commit/push 금지. 발행은 명시 지시 때만, `check-large-data.cmd --commit/--push`. force push 금지. 메시지 `<종류>: <항목> — <한 줄>`(예측·닫힘·기각·축소·개시·정리·하네스), "new" 금지.
