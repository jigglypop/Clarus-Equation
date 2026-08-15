# 런타임·패키지 아키텍처 감사

분석일: 2026-08-15  
범위: `reality_stone/`, `clarus-agent-guard/`, `ex-codex/`  
방법: 정적 호출/패키지 경계 추적 + 읽기 전용 import probe + Torch/Rust 동일 상태 1-step 비교. 외부 웹은 사용하지 않았고 제품 소스는 수정하지 않았다.

## 1. 결론

세 디렉터리는 하나의 단일 런타임이 아니라 다음 세 제품 경계다.

1. `reality_stone/`: Python이 제어면, Torch/NumPy가 호환 계산면, 두 개의 독립 PyO3 크레이트가 네이티브 계산면인 혼합 패키지다. 상위 저장소 설치는 Clarus 코어를 `reality_stone.clarus._rust`로, 하위 패키지 설치는 기하 코어를 `reality_stone._rust`로 빌드한다 (`pyproject.toml:38-42`, `reality_stone/pyproject.toml:21-24`, `reality_stone/README.md:47-53`).
2. `clarus-agent-guard/`: HTTP DAGlet 데모와 SDK/VS Code 실효 실행 경로가 서로 다른 두 런타임이다. HTTP는 `scheduler -> cells -> DAGlet store`, VS Code는 `webview -> LLM provider -> AgentSession -> JSONL child process -> ClarusGuard -> real tools`로 흐른다 (`clarus-agent-guard/server/scheduler.py:18-38`, `clarus-agent-guard/vscode-extension/README.md:16-31`).
3. `ex-codex/`: 애플리케이션 라이브러리가 아니라 Codex 연구 워크플로 하네스다. hook JSON과 수동 CLI가 shell/PowerShell/cmd 래퍼를 거쳐 단일 Rust 바이너리로 들어가고, 결과는 JSON stdout 또는 `_workspace/ce`의 Markdown/포인터 파일로 나온다 (`ex-codex/hooks.json:3-22`, `ex-codex/skills/ce-research/core/src/main.rs:27-46`).

즉 직접 코드 의존 방향은 대체로 `ex-codex -> (명령으로) reality_stone/guard`, `Python -> Rust/CUDA`이며, guard와 Reality Stone 사이의 런타임 import 결합은 없다. `ex-codex`는 검증 스킬에서 Reality Stone 스크립트/테스트를 호출하고 guard 벤치 스킬에서 별도 제품 벤치를 호출할 뿐이다 (`ex-codex/skills/ce-validate/SKILL.md:10-27`, `ex-codex/skills/clarus-guard-bench/SKILL.md:10-25`).

## 2. 진입점·공개 API·언어 경계

| 영역 | 진입점/공개면 | 언어 경계 | 주 데이터 흐름 |
|---|---|---|---|
| Reality Stone 기하 | `import reality_stone`; `reality_stone.api.pipeline`; `layers`, `models`, `optim`; `reality_stone._rust` | Python/Torch/NumPy -> PyO3 outer Rust -> 선택적 CUDA | 텐서 -> Python autograd/facade -> Rust ndarray 또는 CUDA -> Torch 텐서. 최상위 공개 목록은 `reality_stone/python/reality_stone/__init__.py:193-259`, 파이프라인 공개면은 `reality_stone/python/reality_stone/api/__init__.py:1-30`이다. |
| Clarus 런타임 | `CEEngine`, `BrainRuntime`, `BrainDaemon`; 다수 `python -m reality_stone.clarus.<module>` | Python 제어면 -> `ce_ops` dispatch -> Torch / `reality_stone.clarus._rust` / 선택적 CUDA | `.ce.pt` -> `CEEngine` tokenizer/가중치 -> prompt state/decoder -> `BrainRuntime` sparse state/replay/STDP -> snapshot/output. 제어/수치 분리는 `reality_stone/python/reality_stone/clarus/ce_ops.py:1-8`, 엔진 로드는 `.../engine.py:137-165`, runtime step은 `.../runtime.py:1005-1065`에 나타난다. |
| Guard HTTP | `uvicorn server.main:app`; `/event`, `/daglet/*`, `/trace`, `/audit`, `/replay`, `/memory/propose` | Python stdlib core -> 선택적 FastAPI/Pydantic | HTTP event -> keyword salience -> router/policy/memory cells -> DAGlet audit -> process-global in-memory stores (`clarus-agent-guard/server/main.py:1-22`, `.../routes.py:20-87`, `.../scheduler.py:45-91`). |
| Guard SDK | `ClarusGuard.tool/call/approve/remember/history/audit` | Python callables -> capability/taint executor -> SQLite decision trace | user text -> lexical capability -> argument `Value` -> dry validation -> pending token -> approval -> raw callable (`clarus-agent-guard/server/sdk.py:51-135`). |
| Guard VS Code | extension `activate`; chat webview; Anthropic/OpenAI provider; Python JSONL bridge | TS/Node fetch + child process -> JSONL -> Python -> filesystem/shell | user text -> remote model tool call -> path resolution -> guard call -> modal approval for writes/commands -> bridge side effect -> tool result back to remote model (`clarus-agent-guard/vscode-extension/src/extension.ts:4-12`, `.../agentLoop.ts:27-66`, `.../guardBridge.ts:20-55`). |
| ex-codex | `hooks/run.cmd`, `run.ps1`, `run.sh`; Rust `hook/init/status/check/revise/gc` | JSON stdin/stdout + shell -> Rust -> filesystem Markdown | Codex event JSON -> route/stop decision; 수동 명령 -> stage sentinel 검사/수정 카운트/archive (`ex-codex/hooks/run.ps1:5-28`, `ex-codex/skills/ce-research/core/src/main.rs:49-85`, `:143-165`, `:228-279`, `:349-397`). |

### Reality Stone 내부 의존 방향

```text
reality_stone.__init__ / reality_stone.api
  -> Python layers/models/optim
  -> reality_stone._rust (outer geometry PyO3) OR _rust.py/_fallback.py

reality_stone.clarus.engine / daemon / runtime
  -> ce_ops (canonical dispatch)
     -> Torch fallback
     -> reality_stone.clarus._rust (nested Clarus PyO3)
     -> optional CUDA ops
```

Outer Rust는 `bindings -> layers/ops`를 등록한다 (`reality_stone/src/lib.rs:1-6`, `reality_stone/src/bindings/mod.rs:23-42`). Nested Rust는 sparse/runtime/attention/Pre-EQ 함수를 별도 `_rust` 모듈에 직접 등록한다 (`reality_stone/python/reality_stone/clarus/core/src/lib.rs:676-700`). Python은 이 둘을 같은 이름 `_rust`이지만 서로 다른 패키지 위치로 소비한다.

### Guard 내부 의존 방향

```text
HTTP demo: routes -> scheduler -> cells -> trace.audit -> in-memory DAGletStore

Product SDK: ClarusGuard -> ToolGate(private Executor) -> real callable
                       \-> independent MemoryStore
                       \-> independent SQLite linear trace

VS Code: webview -> provider -> AgentSession -> GuardBridge -> vscode_bridge.py
                                                     -> ClarusGuard -> file/shell
```

README는 `ClarusGuard`가 interception/capability/DAGlet/firewall 네 코어를 통합한다고 서술하지만 (`clarus-agent-guard/README.md:33-51`), 실제 SDK import와 상태는 capability executor, ToolGate, MemoryStore, SQLite뿐이다 (`clarus-agent-guard/server/sdk.py:28-38`, `:51-67`). DAGlet scheduler/store는 HTTP 쪽의 별도 process-global 경로다.

### ex-codex 내부 의존 방향

```text
hooks.json
  -> run.cmd | run.sh
       -> prebuilt ce-research-core (hook은 없으면 no-op)

manual run.cmd/run.ps1/run.sh
  -> stale check / cargo build to external cache
  -> ce-research-core
       -> _workspace/ce/<run>/{stage markdown, artifacts, revisions, pointers}
```

스킬/역할 카드는 정책과 사람/모델의 의미 판정을 담당하고, Rust 코어는 파일명·상태 sentinel·수정 횟수·archive 이동을 담당한다. Rust 코어 자체는 Reality Stone이나 guard를 링크하지 않는다 (`ex-codex/skills/ce-research/core/Cargo.toml:1-11`).

## 3. P0 — 즉시 차단해야 하는 결함

### P0-1 [보안] Reality Stone 아티팩트 로더가 임의 pickle 코드 실행을 허용한다

- `CEEngine(path)`는 외부 경로를 곧바로 `torch.load(..., weights_only=False)`로 연다 (`reality_stone/python/reality_stone/clarus/engine.py:137-140`). CLI의 필수 `--engine`이 그대로 이 경로에 들어간다 (`.../engine.py:1796-1806`, `:1855-1862`). Teacher clone 금지는 역직렬화 **후** 검사이므로 이 위협을 줄이지 않는다 (`.../engine.py:305-325`).
- 일반 공개 API `HierarchicalLLM.from_pretrained`도 checkpoint를 `torch.load`하며 (`reality_stone/python/reality_stone/api/pipeline.py:29-43`), 이어서 `strict=False` 불일치를 조용히 버린다 (`:47-58`). 지원 범위가 `torch>=2.0`이므로 버전별 기본 `weights_only` 동작에도 의존한다 (`pyproject.toml:6-12`).
- 영향: 내려받았거나 공유받은 `.pt/.ce.pt`를 여는 순간 모델 검증 전에 Python 코드가 실행될 수 있다. 이는 런타임 아티팩트를 데이터로 취급한다는 경계를 무너뜨린다.
- 조치: tensor-only 포맷(safetensors 또는 명시적 tensor/JSON 분리)을 정본으로 삼고, 이행기에는 `weights_only=True` + 허용 키/타입/shape/dtype/크기 상한을 역직렬화 직후 강제한다. 불일치 키를 오류로 승격한다.

### P0-2 [보안] VS Code의 무승인 read 경로가 작업공간 밖 비밀을 원격 모델로 반출할 수 있다

- 도구 스키마가 절대 경로를 명시적으로 허용한다 (`clarus-agent-guard/vscode-extension/src/tools.ts:16-25`). `resolvePath`도 absolute path는 그대로 두고 상대 경로의 `..` 경계를 검사하지 않는다 (`.../agentLoop.ts:23-25`, `:45-49`).
- Python bridge의 `read_file`/`list_dir`는 side-effecting이 아니므로 capability나 승인 없이 임의 `Path`를 연다 (`clarus-agent-guard/server/vscode_bridge.py:27-34`). SDK는 side-effect가 아닌 도구를 즉시 실행한다 (`clarus-agent-guard/server/sdk.py:77-100`).
- 읽은 값은 다음 모델 호출의 tool result로 들어간다. Anthropic은 tool result를 user content로, OpenAI는 tool message로 원격 API에 보낸다 (`clarus-agent-guard/vscode-extension/src/providers.ts:36-67`, `:98-116`).
- 영향: repo 문서의 간접 prompt injection이나 모델 오작동만으로 SSH 키, 환경 파일, 브라우저/클라우드 설정 등 확장 프로세스 권한으로 읽을 수 있는 파일이 사용자 승인 없이 외부 API로 전송될 수 있다.
- 조치: 모든 파일 도구에 canonical `realpath` 기반 workspace allow-root, symlink 재검사, 파일 크기/종류 상한을 적용한다. workspace 밖 읽기와 민감 패턴은 별도 capability + modal approval 대상으로 바꾼다. tool result를 원격으로 보내기 전에 redaction/egress policy를 둔다.

### P0-3 [보안·정확성] 실제 LLM 통합이 생성 인자를 전부 `user` provenance로 승격해 I2 보장을 무효화한다

- 보안 코어의 핵심 전제는 user만 권한 원천이고 tool/doc 값은 UNTRUSTED라는 것이다 (`clarus-agent-guard/server/capability.py:9-22`, `:42-55`). Executor도 critical argument가 UNTRUSTED일 때만 거부한다 (`clarus-agent-guard/server/executor.py:75-95`).
- 그러나 실제 `AgentSession`은 **LLM이 생성한 모든 tool args**를 `provenance='user'`로 bridge에 보낸다 (`clarus-agent-guard/vscode-extension/src/agentLoop.ts:42-50`). SDK는 tuple 표기가 없는 모든 인자를 그 단일 provenance로 감싼다 (`clarus-agent-guard/server/sdk.py:77-88`). 따라서 검색/파일/tool output에서 유래한 path/command도 USER trust가 된다.
- 벤치는 공격 인자를 수동으로 `Value(..., 'tool')`로 만들어 이상적인 코어만 시험한다 (`clarus-agent-guard/bench/agentdojo_suite.py:99-110`). 전체 scoreboard에도 VS Code/bridge end-to-end 항목이 없다 (`clarus-agent-guard/bench/all.py:16-35`).
- 영향: VS Code README의 “tool output은 권한을 만들 수 없고, untrusted에서 가져온 critical arg는 거부된다”는 제품 보장이 실제 호출 경로에서 성립하지 않는다 (`clarus-agent-guard/vscode-extension/README.md:3-14`). 사용자가 “테스트를 실행해”처럼 넓은 RUN_COMMAND 권한을 준 턴에서는 injected command도 USER trust로 dry check를 통과해 승인창까지 도달한다.
- 조치: LLM 메시지/도구 결과 단위의 provenance graph를 유지하고, 각 argument가 어느 span/value에서 유도됐는지 taint를 전파한다. 그것이 없으면 LLM 생성 args는 기본 UNTRUSTED로 두고 사용자가 승인 UI에서 critical args를 직접 확정한 경우에만 신뢰를 올린다. `AgentSession -> bridge -> real sink`를 실제 공격 corpus로 E2E 벤치해야 한다.

## 4. P1 — 결론은 유지되나 방치 불가

### P1-1 [정확성] 기본 CPU `auto`/Rust 경로가 기본 활성화된 axon-delay 동역학을 무시한다

- `BrainRuntimeConfig` 기본은 `axon_delay=True`다 (`reality_stone/python/reality_stone/clarus/runtime.py:98-120`). Torch 경로는 ring buffer를 읽고 갱신한다 (`:923-951`).
- CPU `auto`는 Rust kernel이 있으면 무조건 Rust를 선택한다 (`:827-834`), 그러나 `_step_rust`에는 delay buffer/index가 전달되지 않는다 (`:836-897`). Nested Rust `brain_step` 인자에도 delay 상태가 없고 현재 activation으로 바로 recurrent matvec을 한다 (`reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs:198-256`).
- 기존 parity test는 이 차이를 피하도록 명시적으로 `axon_delay=False`를 사용한다 (`tests/test_runtime_contracts.py:120-160`).
- 최소 재현 probe: `artifacts/delay_backend_probe.py`. 동일 seed/상태/가중치, noise 0, `axon_delay=True`에서 두 backend를 한 step 실행한다.

  ```powershell
  .\.venv\Scripts\python.exe -B _workspace\ce\repository-code-analysis-20260815\artifacts\delay_backend_probe.py
  ```

  2026-08-15 실행 원문 요약:

  ```text
  has_rust=True
  max_activation_diff=0.008280158042908
  torch_delay_idx=1
  rust_delay_idx=0
  torch_delay_sum=2.057143211364746
  rust_delay_sum=0.000000000000000
  mismatch_reproduced=True
  ```

  따라서 Torch는 delay ring을 한 칸 전진·기록했지만 Rust는 delay state를 전혀 갱신하지 않았고 activation도 달라졌다.
- 조치: delay state를 Rust ABI에 포함하거나 `axon_delay=True`이면 Torch로 dispatch한다. 기본 설정 그대로의 multi-step parity test를 추가한다.

### P1-2 [패키징·성능] 같은 distribution 이름의 두 build root가 서로 다른 native 모듈과 dependency 계약을 만든다

- 저장소 루트와 하위 디렉터리 모두 `name="reality_stone"`, `version="0.2.10"`이지만 root는 Clarus native module을, nested는 geometry native module을 빌드한다 (`pyproject.toml:1-12`, `:34-42`; `reality_stone/pyproject.toml:1-14`, `:21-24`). dependency 범위도 root `numpy>=1.24`, nested `numpy>=1.21,<3` 등으로 다르다.
- README가 차이를 설명하므로 의도된 구조이지만 (`reality_stone/README.md:47-53`), 사용자는 동일 이름/버전 wheel이 어느 디렉터리에서 빌드됐는지로 native API와 성능이 달라진다. 한 wheel에서 두 PyO3 코어를 모두 빌드하지 못한다.
- 조치: 단일 build root에서 두 extension을 workspace/member로 패키징하거나 distribution을 `reality-stone-geometry`와 unified runtime으로 분리하고 버전/의존 계약을 분명히 한다.

### P1-3 [API 정확성] 공개 conversion API가 존재하지 않는 모듈 import 실패를 `None`으로 숨긴다

- 최상위는 `.conversion`에서 두 함수를 import하려다 모든 `Exception`을 잡아 `None`으로 두고 (`reality_stone/python/reality_stone/__init__.py:103-107`), 그래도 `__all__`에 공개한다 (`:228-229`). 현재 tree에는 `python/reality_stone/conversion.py`가 없다.
- 읽기 전용 import probe에서도 두 값이 모두 `None`이었다.
- 영향: import는 성공하지만 호출 시 늦고 불명확하게 실패하고, 타입/문서상 공개 API와 런타임 계약이 다르다.
- 조치: 구현을 복구하거나 공개 목록에서 제거하고 명시적 feature error를 발생시킨다. broad `except Exception` 대신 예상한 optional dependency 오류만 처리한다.

### P1-4 [보안·안정성] Python에 raw CUDA pointer 함수를 노출하며 pointer/shape/device 검증이 없다

- nested PyO3가 7개 raw `u64` pointer와 크기를 받아 곧바로 `unsafe` CUDA 호출에 넘긴다 (`reality_stone/python/reality_stone/clarus/core/src/lib.rs:505-532`). Python wrapper는 정상 경로에서 contiguous/f32/sync를 준비하지만 (`.../ce_riemann_attn.py:339-363`), `_rust` 함수 자체는 직접 import 가능하다.
- 영향: 잘못된 정수, 다른 device pointer, 수명 종료 버퍼, overflow 크기로 process crash 또는 GPU memory corruption이 가능하다. Python 내부 low-level API이므로 원격 권한 상승은 아니지만 런타임 격리/안정성 경계를 깬다.
- 조치: raw pointer PyFunction을 비공개 Rust API로 내리고 Torch custom op/DLPack capsule처럼 device·dtype·shape·lifetime을 검증할 수 있는 소유 객체를 받는다. 최소한 checked multiplication, null/device/buffer-size 검사와 명시적 stream handle을 요구한다.

### P1-5 [성능] native hot path가 매 step 대량 복사하고 CUDA attention은 전-device barrier를 건다

- Rust runtime step 전 Python이 state/input/CSR 14개 이상을 `.cpu().numpy().astype(...)`로 매번 복사하고, 결과 7개를 다시 NumPy/Torch/device로 복사한다 (`reality_stone/python/reality_stone/clarus/runtime.py:836-904`). Rust도 mutable state 7개를 다시 `to_vec()`하고 새 NumPy arrays로 반환한다 (`.../clarus/core/src/lib.rs:356-390`).
- Riemann CUDA attention은 호출마다 `torch.cuda.synchronize(q.device)`로 device 전체를 동기화한다 (`.../ce_riemann_attn.py:339-363`). 이는 stream overlap을 제거한다.
- 조치: persistent native state/capsule 또는 in-place writable buffers를 사용하고 CSR은 초기화 시 한 번만 native에 보관한다. CUDA는 현재 stream을 전달하고 event/stream dependency만 설정한다.

### P1-6 [책임 중복·검증 공백] Guard의 DAGlet 제품면과 실제 SDK/extension 집행면이 분리돼 있다

- scheduler는 cells와 process-global `DAGletStore`를 사용한다 (`clarus-agent-guard/server/scheduler.py:18-39`, `:45-91`); store는 명시적으로 in-memory다 (`.../trace/store.py:1-10`, `:32-37`, `:85-86`).
- SDK는 자체 `ToolGate`, 자체 `MemoryStore`, 자체 SQLite 선형 trace를 만들며 DAGlet/scheduler/audit를 import하지 않는다 (`clarus-agent-guard/server/sdk.py:28-38`, `:51-67`, `:120-136`). 실제 bridge도 이 SDK만 쓴다 (`.../vscode_bridge.py:21-47`).
- 영향: HTTP/benchmark에서 보인 DAGlet invariant, replay, motif, trace가 VS Code real sink에는 적용되지 않는다. 반대로 SDK 승인/실효 tool 결과는 DAGlet 관측면에 나타나지 않는다. README의 “네 코어 통합/전부 SQLite” 설명과 다르다 (`clarus-agent-guard/README.md:33-51`).
- 조치: 하나의 `ExecutionService`가 provenance, capability, approval, DAGlet audit, memory, persistence를 모두 소유하게 하고 HTTP/SDK/bridge는 transport adapter만 되게 한다.

### P1-7 [HTTP 정확성] `/daglet/similar`가 앞선 동적 route에 가려진다

- `/daglet/{daglet_id}`가 먼저 등록되고 그 뒤에 `/daglet/similar`가 등록된다 (`clarus-agent-guard/server/routes.py:41-49`). Starlette/FastAPI는 선언 순서로 첫 path match를 택하므로 `similar`가 `daglet_id`로 소비된다.
- 조치: static route를 dynamic route보다 먼저 선언하거나 `/daglets:similar`처럼 충돌 없는 경로로 바꾼다. routing E2E test를 둔다.

### P1-8 [보안·정확성] Memory Firewall의 “raw episode backing”은 SDK에서 호출자가 임의 문자열로 자가 증명한다

- firewall은 episode ID가 `raw_episodes` set에 있기만 하면 faithfulness를 통과시키지만 실제 episode payload/hash는 저장하지 않는다 (`clarus-agent-guard/server/memory_firewall.py:58-78`).
- SDK `remember(..., episode=...)`는 어떤 문자열이든 먼저 `add_episode`하고 즉시 commit한다 (`clarus-agent-guard/server/sdk.py:113-118`). 반대로 HTTP `/memory/propose`에는 episode 등록 경로가 없어 정상 제안도 기본적으로 거부된다 (`clarus-agent-guard/server/routes.py:75-87`).
- 조치: raw episode를 immutable content/hash와 함께 ingestion 경계에서만 등록하고, memory proposal은 존재하는 episode reference만 받게 한다. HTTP/SDK가 같은 store/service를 써야 한다.

### P1-9 [보안·운영] API key는 VS Code 일반 설정에 평문으로 저장 가능하고 HTTP 관측면에는 인증이 없다

- extension이 API key를 일반 configuration string으로 노출한다 (`clarus-agent-guard/vscode-extension/package.json:43-51`) and reads it directly from settings/env (`.../chatViewProvider.ts:38-48`). SecretStorage를 사용하지 않는다.
- FastAPI app은 router만 포함하며 인증 middleware/dependency가 없고 (`clarus-agent-guard/server/main.py:12-17`), routes는 원문 event/trace/audit를 조회하고 replay/memory mutation도 수행한다 (`.../routes.py:28-38`, `:52-87`). localhost 밖에 bind하면 정보 노출·상태 변조 면이 열린다.
- 조치: `ExtensionContext.secrets`로 key를 이동하고 기존 setting migration/redaction을 제공한다. HTTP는 기본 loopback bind를 명시하고 token/auth, CORS/host 제한, trace redaction을 둔다.

### P1-10 [정확성·운영] 선언형 policy 파일이 로드되지 않아 운영자 변경이 아무 효과가 없다

- `PolicyCell` docstring은 `policies/default.yaml`을 로드한다고 하지만 실제 생성자는 hard-coded `DEFAULT` executor만 받는다 (`clarus-agent-guard/server/cells/policy_cell.py:1-3`, `:15-31`). YAML은 allow/require_approval 목록을 정의한다 (`clarus-agent-guard/server/policies/default.yaml:1-19`), server 코드에는 이를 읽는 import/call이 없다.
- 조치: policy loader/schema validation을 단일 등록 경계에 연결하거나 YAML을 삭제하고 code-defined policy임을 명시한다. 로드된 effective policy를 health/debug surface에 노출한다.

### P1-11 [가용성] GuardBridge child process 장애 시 요청 Promise가 영구 대기한다

- bridge는 stdout line FIFO만 처리하고 stderr를 무제한 누적한다 (`clarus-agent-guard/vscode-extension/src/guardBridge.ts:15-35`). `send`에는 timeout/write failure 처리가 없고 (`:42-46`), child `error`/`exit` 시 pending resolver를 reject하는 handler도 없다.
- 영향: Python 미설치, import 실패, bridge crash, stdout protocol 손상 시 chat session이 멈추고 메모리도 증가한다.
- 조치: startup handshake, request ID, per-call timeout/cancel, child error/exit fan-out rejection, bounded stderr ring buffer, restart policy를 추가한다.

### P1-12 [검증 정확성] ex-codex machine gate는 의미가 아니라 forge 가능한 sentinel만 검사한다

- stage 통과는 파일 어디든 정확한 `Status: COMPLETE` 한 줄이 있으면 되고 (`ex-codex/skills/ce-research/core/src/main.rs:303-336`), gate는 `Gate:` 뒤 문자열이 `PASS`로 **시작하기만** 하면 된다 (`:339-346`). `check`는 이 결과와 파일 존재를 조합할 뿐 claim ID, P0 목록, 근거, 역할 schema를 파싱하지 않는다 (`:228-279`).
- 영향: `Gate: PASSING`, 비어 있는 COMPLETE 문서, 인용 블록 속 sentinel도 machine PASS가 될 수 있다. 역할 카드의 의미 감사와 Rust gate의 보장 수준을 혼동하면 완결성 주장이 과장된다.
- 조치: line grammar를 exact match하고, stage별 최소 schema/필수 섹션/열린 P0=0/claim 근거를 구조화 JSON sidecar 또는 엄격 Markdown parser로 검증한다. 기계 gate는 “형식 통과”라고 명명한다.

### P1-13 [가용성] ex-codex route/stop hook은 바이너리 부재·깨진 JSON에서 의도적으로 fail-open이다

- Windows/Unix hook은 prebuilt binary가 없으면 성공 no-op한다 (`ex-codex/hooks/run.cmd:5-11`, `ex-codex/hooks/run.sh:12-15`; README에도 `ex-codex/README.md:29-33`). Rust는 JSON parse 실패를 `Null`로 바꾸고 (`ex-codex/skills/ce-research/core/src/main.rs:51-59`), stop cwd도 `.`로 기본화한다 (`:70-81`).
- 영향: 최초 설치, stale/missing binary, malformed hook payload에서 routing/completion guard가 조용히 사라진다. 이는 성능을 위한 의도된 tradeoff지만 “Stop guard”를 강제 제어로 사용할 수 없음을 명시해야 한다.
- 조치: 설치 self-check와 눈에 보이는 degraded-state telemetry를 제공하고, stop만큼은 선택적 strict mode에서 fail-closed하도록 한다.

### P1-14 [동시성·데이터 무결성] 단일 `.active-run`과 lock 없는 revision log가 병렬 연구에 안전하지 않다

- workspace당 active pointer는 하나이며 init할 때 무조건 덮어쓴다 (`ex-codex/skills/ce-research/core/src/main.rs:15-17`, `:145-160`). stop은 이 한 포인터만 본다 (`:99-128`).
- revise는 전체 log를 읽고 횟수를 센 뒤 전체 파일을 다시 써서 append를 흉내 내며 lock/atomic replace가 없다 (`:351-364`). 두 agent가 동시에 실행하면 update loss 또는 revision limit 우회가 가능하다.
- 조치: active-run을 session/run ID별 registry로 만들고 lockfile + append/transaction 또는 SQLite를 사용한다. revision key를 `(run, role, attempt_id)`로 원자적으로 유일화한다.

### P1-15 [파괴적 운영] `gc`는 전달된 임의 workspace의 자식 디렉터리를 sentinel 하나만 보고 이동한다

- `gc(ws)`는 모든 비숨김 자식 디렉터리를 순회하고 `40-final-report.md`가 COMPLETE/ABANDONED이면 `_archive`로 rename한다 (`ex-codex/skills/ce-research/core/src/main.rs:369-388`). canonical `_workspace/ce`인지, run manifest가 있는지, archive destination 충돌이 없는지 사전 검증하지 않는다.
- 조치: workspace marker + canonical path containment + 완전 stage chain을 요구하고, dry-run을 기본으로 하며 move plan을 모두 검증한 뒤 실행한다.

## 5. P2 — 유지보수·표기·정리

### P2-1 Reality Stone public surface가 optional failure를 `None`으로 정상 import처럼 보이게 한다

Clarus는 170개 가까운 이름을 먼저 `None`으로 만들고 다수 `ImportError`를 조용히 삼킨 뒤 모두 `__all__`에 싣는다 (`reality_stone/python/reality_stone/clarus/__init__.py:9-174`, `:176-223`, `:329-460`, `:463-620`). import-safe 목표는 이해되지만 feature availability 객체/명시적 error 대신 런타임 `NoneType` 실패가 된다. 기능별 lazy module과 `has_<feature>()`/typed exception으로 좁히는 편이 낫다.

### P2-2 CUDA build가 basename만으로 archive 이름을 만들어 중복 `mobius.cu` 책임이 충돌한다

build script는 모든 `.cu`의 stem만으로 static library 이름을 만들고 compile한다 (`reality_stone/build.rs:50-56`, `:66-91`). `src/ops/cuda/mobius.cu`와 `src/layers/cuda/mobius.cu`가 같은 `mobius_add_cuda`/`mobius_scalar_cuda` 심볼을 별도 구현한다 (`reality_stone/src/ops/cuda/mobius.cu:14-52`, `:110-114`; `reality_stone/src/layers/cuda/mobius.cu:16-32`, `:37-52`). 하나로 통합하거나 경로 기반 고유 archive 이름을 써야 한다.

### P2-3 Guard route는 오류/거부에도 HTTP 200 payload를 반환한다

Audit refusal과 not-found가 status code 대신 정상 JSON body로 반환된다 (`clarus-agent-guard/server/routes.py:28-44`). 클라이언트/관측에서 성공과 실패를 분리하기 어렵다. 4xx/409/422와 typed response model을 사용해야 한다.

### P2-4 Guard extension packaging은 sibling Python repo layout에 결합돼 있다

extension은 `extensionUri/..`를 guard repo로 가정하고 그 cwd에서 `python -m server.vscode_bridge`를 spawn한다 (`clarus-agent-guard/vscode-extension/src/chatViewProvider.ts:55-61`, `.../guardBridge.ts:20-24`). README도 sibling shipping을 전제한다 (`clarus-agent-guard/vscode-extension/README.md:44-51`). VSIX에 Python server가 실제로 포함/설치되는 독립 패키징 계약과 version handshake가 필요하다.

### P2-5 ex-codex는 배포 사본이라 `.codex`와 이중 정본 위험이 있다

README 자체가 `ex-codex`를 `.codex/` 개발 트랙의 배포 사본이라고 정의한다 (`ex-codex/README.md:1-4`). 감사 시점의 대응 파일은 hash상 동일했지만, 역할 카드·SKILL·hook을 수동으로 두 군데 유지하면 drift 가능성이 상존한다. 생성/동기화 스크립트와 CI hash gate로 단일 정본을 강제해야 한다.

## 6. 책임 중복 요약

| 중복 | 현재 상태 | 권장 소유자 |
|---|---|---|
| Reality Stone native 코어 | outer geometry `_rust`와 nested Clarus `_rust`가 별도 Cargo/PyO3/version/build root를 가짐 | Cargo workspace + 단일 Python distribution manifest |
| Reality Stone backend policy | 최상위 `__init__`, Python layer별 import, `clarus.__init__`, `ce_ops`, `runtime._use_rust`가 각각 availability/dispatch를 판단 | 하나의 typed backend registry (`ce_ops`를 확장) |
| Guard 집행/관측 | scheduler/PolicyCell/DAGlet과 SDK/ToolGate/SQLite가 분리 | 단일 `ExecutionService`; HTTP/SDK/bridge는 adapter |
| Guard policy | `default.yaml`, `DEFAULT` executor, VS Code `TOOL_DEFS`, bridge decorators에 tool 성격이 반복 | versioned policy/tool registry에서 Python/TS schema 생성 |
| Guard memory | DAGletStore motif memory와 SDK MemoryStore가 별도, 둘 다 process-local | durable repository 하나 + episode/provenance schema |
| CE 하네스 정본 | `.codex` 개발본과 `ex-codex` 배포본 | 한 source tree에서 배포 artifact 생성 |

## 7. 우선 실행 순서

1. Guard의 `read_file/list_dir` workspace sandbox와 remote egress 제한을 먼저 적용하고, LLM args를 기본 UNTRUSTED로 바꾼다. 그 후 실제 extension/bridge E2E 공격 테스트를 CI 필수 gate로 둔다.
2. 모든 `.pt/.ce.pt` reader를 안전 tensor-only schema로 전환한다. 안전 loader가 없으면 외부 artifact 실행을 중단한다.
3. `BrainRuntime`의 기본 config parity test를 추가하고, delay ABI 구현 전까지 `axon_delay=True`에서 Rust auto dispatch를 금지한다.
4. Guard의 두 실행 스택을 단일 service로 합치고 policy/tool/provenance/memory/trace의 정본을 하나로 만든다.
5. Reality Stone build root와 public API 계약을 통합하고 `None` 공개 심볼을 제거한다.
6. ex-codex gate를 exact/schema 기반으로 강화하고 active/revision/gc에 containment·transaction을 추가한다.

## 8. 실행한 비변경 검증

- Import probe: checkout shim은 inner package를 로드했고, geometry `_rust`는 Python fallback, Clarus `_rust`는 native로 로드됐다. `convert_to_full_riemannian`과 `convert_to_hyperbolic`은 모두 `None`이었다. 이 결과는 dual native build 및 P1-3과 일치한다 (`reality_stone/__init__.py:12-27`, `reality_stone/python/reality_stone/__init__.py:7-25`, `:103-107`).
- Torch/Rust delay probe: Rust kernel 존재 환경에서 동일 초기 상태 1-step 결과가 달랐고 Rust delay index/buffer가 갱신되지 않았다. P1-1의 정적 호출 그래프를 수치로 확인했다.
- 외부 웹/네트워크 검증은 하지 않았다. 벤치 전체 실행은 이 아키텍처 레인의 요청 범위가 아니므로 실행했다고 주장하지 않는다.
