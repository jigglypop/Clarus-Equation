# Clarus Agent Guard

AI 에이전트가 **바로 행동하지 않고**, 기억할지·검색할지·검증할지·멈출지를
먼저 판단하게 만드는 상시 실행 Guard 서버.

겉은 평범한 agent guard / runtime observability지만, 속은 **DAGlet
Runtime** 이다. 그게 기존 하네스(OpenAI Agents SDK, LangGraph, NeMo
Guardrails)와의 차별점이다.

## 핵심 개념: DAGlet

뇌의 연결망은 recurrent(순환)지만, **한 번의 생각/행동/replay를 시간축으로
펼치면 acyclic** 하다 — 이게 RNN을 학습할 때 쓰는 BPTT(시간펼침)와 같은
구조다. Clarus는 이 한 번의 펼쳐진 인과 그래프를 **DAGlet** 이라 부르고,
모든 제품을 "DAGlet에 대한 연산"으로 통일한다.

| 제품 | DAGlet 관점 |
|------|------------|
| Trace Guard | DAGlet 실행 기록 + 실패 노드 localize |
| Permission Proxy | `external_action` edge에 gate |
| Memory Firewall | `memory_write` edge 검증 (replay 후) |
| Metric RAG | `retrieval` edge re-rank |

고정 workflow graph가 아니라, 항상 켜진 cell substrate 위에서 이벤트마다
DAGlet이 **생성**된다. 같은 substrate가 입력에 따라 다른 DAGlet을 만든다.

## 3-layer 구조

- **substrate G** — 모든 cell과 가능한 연결 (`server/cells/`)
- **runtime D_t** — 이번 이벤트의 실제 DAGlet (`server/scheduler.py`)
- **memory M** — 과거 DAGlet, 점수, motif 조회 (`server/trace/store.py`)

## 빠른 실행

서버 없이 헤드리스로 돌아간다:

```bash
python -m examples.rag_chatbot_guard
python -m examples.coding_agent_guard
```

HTTP 서버:

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload
# POST /event {"event": "지난 회의 기준으로 메일 보내줘"}
```

## 구조

```
server/
  main.py          FastAPI shell (optional)
  routes.py        POST /event, GET /daglet/{id}, /daglet/similar, /trace
  scheduler.py     ★ substrate를 걸어 DAGlet을 생성 (런타임 핵심)
  cells/           salience / router / policy / critic / memory / trace
  trace/
    schema.py      ★ DAGlet / Node / Edge — 코어 데이터 모델
    store.py       in-memory 저장 + motif 유사도(similar)
  policies/default.yaml   PolicyCell 허용/승인 규칙
examples/
  rag_chatbot_guard.py
  coding_agent_guard.py
```

`schema.py` 가 진짜 코어다 — 웹 프레임워크나 LLM 클라이언트를 절대
import 하지 않아 어디든 이식된다.

## 성공 지표 (첫 데모 기준)

1. 잘못된 external action 차단율
2. 불필요한 LLM call 감소율
3. 검색/기억 필요 질문 route 정확도
4. trace로 오류 단계 찾는 정확도
5. 근거 없는 답변 비율 감소
6. 평균 latency 증가량

**벤치마크 (직접 100문항):** 바로답변 25 / memory 25 / search 25 / verify 25.
목표: route accuracy 85%+, external action false-allow 0%, latency +300ms 이하.

```bash
python -m bench.run
# route accuracy : 99/100 = 99.0%   (target 85%+)
# external false-allow : 0/25 = 0.0%   (target 0%)
# latency avg / p95 : ~0.03 ms        (target +300ms 이하)
# RESULT: [PASS]
```

### 하드(적대적) 벤치마크

키워드 함정 + 우회 표현 + prompt injection 40문항. 가드 제품의 진짜
지표는 "공격 하에서도 false-allow가 0이냐"다.

```bash
python -m bench.hard_run
# route accuracy        : 95.0%
# external false-ALLOW  : 0%      (MUST be 0)  -> [SAFE]
# false-BLOCK (harmless): 13.3%   (잔여 과차단, 룰의 한계)
# verify_declared / injection : 100%  <- tool 경계 가로채면 텍스트 무관하게 안전
# verify_evasion(tool 미선언) : 룰 확장으로 막았으나 whack-a-mole
```

**교훈:** 텍스트 탐지는 우회에 뚫린다. durable fix는 에이전트가 tool을
선언하게 만드는 tool-boundary interception(= PolicyCell이 모든
`external_action` edge를 강제 통과)이다.

### 그래프 불변식 (구조적 보장)

탐지(이게 액션인가?)는 fallible하지만, 구조는 보장할 수 있다:
**`external_action` 라벨이 붙은 edge는 DAGlet 안에 policy 노드가 반드시
있어야 한다.** 매 commit마다 `trace/audit.py`가 검사하고 위반을
`daglet.violations`에 기록한다 — 미래에 어떤 cell이 액션을 실행으로
바로 연결하면 조용히 새지 않고 commit 시점에 터진다.

```bash
python -m bench.audit_check      # 조작된 우회 검출 + 정상 액션 clean -> [HOLDS]
# GET /audit  -> {"breaches": 0, ...}  (항상 0이어야 함)
```

### replay 루프 (적응)

온라인에선 반응만 하고, idle 때 memory M을 복기한다. `draft`로 차단된
DAGlet에서 명령형 동사 stem을 캐내 `LEARNED_ACTION_HINTS`에 넣는다 →
tool 선언 덕에 잡혔던 동사가 다음엔 **텍스트만으로도** 잡힌다. LLM도,
사람의 키워드 편집도 없이 evasion gap이 좁혀진다.

```bash
python -m bench.replay_demo
# 1) "...양도해줘" (tool 없음)     -> answer (LEAK)
# 2) tool 선언 transfer            -> draft (blocked)
# 3) replay()  newly_learned=['양도']
# 4) 같은 텍스트 (tool 없음)        -> draft (BLOCKED)   -> [LEARNED]
# POST /replay 로도 호출
```

## 지금 상태 (MVP)

- [x] DAGlet schema / store / motif 유사도
- [x] cell substrate (rule-based, LLM 미호출)
- [x] scheduler: 이벤트 → DAGlet 생성 + critic score
- [x] Permission gate (external action → draft_only)
- [x] 두 example, 헤드리스 동작 확인
- [x] 100문항 벤치마크 하네스 (`bench/`) — route 96% / false-allow 0% / latency ~0.03ms
- [x] 하드(적대적) 벤치 (`bench/hard_*`) — route 95% / false-allow 0% / SAFE
- [x] 그래프 불변식 (`trace/audit.py`) — labelled action은 policy를 반드시 통과 (`bench/audit_check.py`)
- [x] replay 루프 (`replay.py`) — 과거 차단에서 행동동사 학습, 텍스트-우회 적응 차단 (`bench/replay_demo.py`)
- [ ] Memory Firewall: `memory_write` edge 검증 cell
- [ ] 학습 힌트 영속화(SQLite) + 운영 중 false-block 모니터링
