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

## 개발자 SDK (3줄 통합)

```python
from server.capability import Capability
from server.sdk import ClarusGuard

guard = ClarusGuard(db_path="guard.db")          # 영속화

@guard.tool(side_effecting=True, cap=Capability.SEND_EMAIL, critical_args=("to",))
def send_email(to, body): ...

r = guard.call("send_email", user_text=user_msg, args={"to": to, "body": draft})
if r.status == "pending":                         # human-in-the-loop
    guard.approve(r.token)
```

`ClarusGuard`가 4개 코어(interception·capability·DAGlet·firewall)를 하나로
묶는다. injected→refused, side-effect→승인 대기, read-only→실행, 전부
SQLite에 기록·감사. 데모: `python -m examples.integrated_agent`.

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

### 간접 프롬프트 인젝션 (표준 벤치 축)

업계 표준 에이전트-보안 벤치(AgentDojo·InjecAgent·ASB)의 핵심 지표는
**ASR(Attack Success Rate)** — 공격이 사용자 입력이 아니라 검색결과·
툴출력·메모리에 *심겨* 들어온다. 적응형 공격 시 SOTA ASR은 85%+, 방어를
잘 쌓아야 73%→8.7%로 내려간다.

우리의 답은 **control/data 분리**: 비-`user` provenance 콘텐츠에서 발견된
명령은 데이터일 뿐, allow-list를 만족할 수 없다(quarantine → draft).

```bash
python -m bench.inject_run
# ASR (attack success) : 0/15 = 0%   (target 0; SOTA 11-85%)
# benign pass-rate     : 100%        (검색/메모리는 그대로 쓸 수 있음)
# INJECTION DEFENSE: [HELD]
```

### 한 번에 다 돌리기

```bash
python -m bench.all     # 6개 벤치 스코어보드 + CI 게이트(exit code)
```

### 공식 AgentDojo 공격 데이터 실측

자작 벤치가 아니라 **AgentDojo v1의 실제 4 스위트**(workspace/travel/banking/
slack)를 설치해, 27개 injection task의 *공격자 ground-truth 툴콜*을 우리 실제
`Executor`에 흘려보낸 결과. 전체 user×injection 매트릭스 = **629 보안 조합**
(AgentDojo 공식 수치와 일치).

```bash
python -m venv .venv-adojo && source .venv-adojo/Scripts/activate
pip install agentdojo
PYTHONUTF8=1 python -m bench.agentdojo_official
# TOTAL security combinations : 629
# ASR (any injected side effect) : 0/629 = 0.0%
# BENIGN authorized calls execute : 80/80 = 100%  (blanket-deny 아님)
```

범위/정직성: **보안(ASR) 차원만** 측정 — utility는 라이브 LLM 실행이 필요해
미수행(별도 유료 런). 측정 중 발견한 분류 버그(`get_scheduled_transactions`의
`"scheduled"`가 `schedule` 동사로 오매칭 → 무해한 read를 side-effect로 오판,
초기 0.5% 거짓 ASR)는 read-접두사 우선으로 수정 후 0%로 정정. 진짜 공격
(send_money 등)은 처음부터 전부 차단돼 있었음.

### 벤치마크 한눈에

| bench | 측정 | 결과 |
|-------|------|------|
| `bench.run` | route 정확도 (100문항) | 96% / false-allow 0% |
| `bench.hard_run` | 적대적 (함정·우회·injection) | 95% / false-allow 0% / SAFE |
| `bench.inject_run` | 간접 인젝션 ASR | **0%** / benign 100% / HELD |
| `bench.audit_check` | 게이트 우회 불변식 | HOLDS |
| `bench.replay_demo` | 차단에서 동사 학습 | LEARNED |
| `bench.firewall_run` | 메모리 poisoning (ASB 축) | poison 100% 차단 / 무단덮어쓰기 0 / HELD |
| `bench.test_capability` | capability 소프트니스 (탐지 무관) | fuzz 150회 0실행 / SOUND |
| `bench.agentdojo_suite` | AgentDojo-style (utility+ASR) 재현 | utility 100% / ASR 0% (F1·F2) |
| `bench.test_interception` | interception 무우회 chokepoint | 직접호출 불가 / NON-VACUOUS |

### 탐지에 의존하지 않는 구조 방어 (capability layer)

키워드 cell은 *탐지* 레이어 — 난독화에 뚫린다. `capability.py` + `executor.py`는
*집행* 레이어로, 공격의 표현을 인식하지 않아도 막는다. 두 불변식:

- **I1 권한 출처:** side effect 권한(capability)은 `authorize()`가 **신뢰된
  사용자 입력에서만** 발급. 데이터에서는 절대 파생 불가.
- **I2 taint 단조성:** 모든 값은 신뢰 등급을 갖고, 결합 시 **최소 신뢰**를
  택함. web/tool/memory 데이터는 스스로를 상위 신뢰로 세탁할 수 없음.

결과: 인젝션이 제안한 액션은 executor에 **권한 없이** 도달 → 표현이
아무리 교묘해도 거부. 탐지는 UX(친절한 에러)용이지, 방어의 본체가 아니다.
`Executor.execute`만이 부작용 경로라서(interception) 우회 래핑이 불가능 →
게이트 불변식이 *비공허*해진다.

```bash
python -m bench.test_capability
# keyword-evading injection refused   (키워드 0개여도 차단)
# exfiltration via tainted arg refused (권한 있어도 오염된 인자 차단)
# fuzz: 150 untrusted attempts, 0 execute
# unrecognised real action -> DENY (safe; 기본 거부)
# STRUCTURAL DEFENSE: [SOUND]
```

정직한 트레이드오프: 구조 방어는 **기본 거부(default-deny)**다. authorize가
못 알아본 *진짜* 사용자 액션도 막힌다(안전하지만 over-block). recall을
올리는 건 classifier cell의 몫 — 그러나 under-allow(=breach)는 구조가 0으로
고정한다. 참고: Google DeepMind **CaMeL**(*Defeating prompt injections by
design*, 2025)이 이 capability/dataflow 분리의 본격판.

표준 벤치 참고: [AgentDojo](https://www.emergentmind.com/topics/agentdojo-benchmark),
[InjecAgent](https://arxiv.org/pdf/2403.02691),
[Agent Security Bench (ICLR'25)](https://proceedings.iclr.cc/paper_files/paper/2025/file/5750f91d8fb9d5c02bd8ad2c3b44456b-Paper-Conference.pdf),
[Indirect PI: Firewalls or Stronger Benchmarks?](https://arxiv.org/abs/2510.05244).

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
- [x] Memory Firewall (`memory_firewall.py`) — poisoning/faithfulness/preservation 3-gate (`bench/firewall_run.py`)
- [x] capability/taint 집행 레이어 (`capability.py`, `executor.py`) — 탐지 무관 구조 방어, soundness 증명 (`bench/test_capability.py`)
- [x] executor를 PolicyCell allow-path에 정식 결선 — salience가 권한 발급(I1), policy가 refuse/approval 구분, read-only는 chokepoint 실행
- [x] AgentDojo/InjecAgent 위협모델 재현 (`bench/agentdojo_suite.py`) — 4 env, F1/F2 공격족, utility 100% / ASR 0% (※동형 재현; 공식 repo+LLM 실측은 미수행)
- [x] interception 배관 (`interception.py`) — `@gate.guarded`로 툴 봉인, `dispatch()`만이 호출 경로; 직접 호출/함수참조 누수 불가 (`bench/test_interception.py`)
- [ ] `@guarded`를 실제 MCP 서버 / function-call 스키마 / 셸 래퍼에 적용 (현재는 in-proc 봉인까지 증명)
- [ ] 학습 힌트 + 메모리 영속화(SQLite) + 운영 중 false-block 모니터링
- [ ] 룰 셀 → small classifier 셀 교체 (recall↑/false-block↓; under-allow는 capability가 0 고정)
