# Clarus-Equation 비-RBE 저장소 전체 코드 분석

Status: COMPLETE

Final verdict: BLOCKED — 분석은 완료됐지만 open P0 때문에 현재 checkout을 release-ready 또는 형식적으로 완결됐다고 판정할 수 없음

분석 기준일: 2026-08-15  
분석 범위: `reality_stone/`, `clarus-agent-guard/`, `ex-codex/`, 관련 `tests/`, `examples/`, `experiments/`, `benchmarks/`, `.claude/`, `.codex/`, 핵심 문서와 build metadata  
명시적 제외: `RBE/` 전체, 가상환경·cache·과거 CE run의 내용, 외부 웹 자료  
작업 원칙: 현재 dirty worktree를 그대로 감사했고 제품 소스와 사용자 변경은 수정하지 않았다.

## 1. 한줄 결론

저장소는 연구 상태 분류, exact arithmetic, fail-closed fixture, Python/Rust 경계, adversarial Guard core bench 등 좋은 기반을 갖췄지만, **보안 경계 3건, V17 exact 반례, 검증 false-green, 동일 배포 좌표의 이중 wheel, 전체 native suite stack overflow**가 열려 있어 현재 상태로 배포하거나 “검증 완료”라고 부를 수 없다.

## 2. 시스템 지도

| 제품면 | 실제 책임과 흐름 | 현재 경계 문제 |
|---|---|---|
| Reality Stone geometry | Python/Torch façade → outer PyO3 `_rust` 또는 fallback → 선택적 CUDA | 동일 name/version의 다른 build root, fallback/native 정체성 혼재, raw pointer 및 복사 비용 |
| Clarus runtime | `CEEngine`/`BrainRuntime` → `ce_ops` → Torch 또는 nested `clarus._rust` | unsafe artifact load, 기본 delay semantics의 backend 불일치, public optional import가 `None` |
| Guard HTTP | FastAPI routes → scheduler/cells → in-memory DAGlet store | SDK/extension 집행과 별도 상태, route shadow, 인증·policy loading 공백 |
| Guard SDK/VS Code | webview → remote LLM → `AgentSession` → JSONL bridge → `ClarusGuard` → 파일/쉘 | workspace 밖 read/egress, provenance 승격, secret/child lifecycle 문제 |
| ex-codex | hook wrapper → Rust core → `_workspace/ce` stage·pointer·revision | sentinel 중심 gate, fail-open, 단일 pointer, 동시성·gc 안전성 |

직접 runtime 결합은 대체로 `Python → Rust/CUDA`이고 Guard와 Reality Stone 사이 import 결합은 없다. ex-codex는 두 제품을 명령으로 검증하는 연구 하네스다. 세 하위트리를 하나의 단일 애플리케이션으로 보기보다 독립 배포·보안 경계로 관리하는 편이 맞다.

## 3. 잘 된 부분

1. README와 핵심 문서는 정리·공리·산출·경험식·미완성을 구분하고, 미출판 연구 가설을 확정 이론처럼 표현하지 않는다.
2. 수학/core 집중 테스트 242개가 warning strict 모드에서 통과했고, narrow identity/chart 궤도의 V17 정리와 V16의 rank-one congruence 구현은 반례 후에도 보존된다.
3. nested Clarus Rust 32개, outer lib 20개, ex-codex core 7개가 각각 통과하고 VS Code TypeScript compile도 성공했다.
4. Guard core 벤치는 injection ASR 0/15, AgentDojo-style ASR 0/12, external false-allow 0/25를 기록했고 memory poison 5/5를 차단했다. core 단위의 capability/taint 설계는 좋은 출발점이다.
5. Python 647파일을 AST로 읽었을 때 syntax error는 0이고, 전체 pytest는 2,604개를 수집한다. 경계·변조·fail-closed 테스트가 많아 결함을 고정할 재료는 충분하다.

이 강점은 아래 P0를 상쇄하지 않는다. 특히 core bench 성공을 실제 VS Code E2E 보안 성공으로, focused math pass를 broad public API 정리의 증명으로 확대하면 안 된다.

## 4. P0 — 먼저 차단할 7건

### 4.1 제품 보안·정확성 P0

| ID | 결함 | 직접 영향 | 핵심 근거 |
|---|---|---|---|
| A-P0-01 | `CEEngine`이 외부 model artifact를 `torch.load(..., weights_only=False)`로 연다 | 검증 전에 pickle code execution 가능 | `reality_stone/python/reality_stone/clarus/engine.py:137-140`, CLI `:1796-1806`, `:1855-1862` |
| A-P0-02 | VS Code read/list가 absolute path와 `..`를 workspace containment 없이 허용하고 결과를 원격 LLM에 반환한다 | 확장 권한으로 읽는 로컬 비밀의 무승인 egress | `tools.ts:16-25`, `agentLoop.ts:23-25,45-49`, `vscode_bridge.py:27-34`, `providers.ts:36-67,98-116`; `C:\Windows\win.ini` 읽기 재현 |
| A-P0-03 | 실제 `AgentSession`이 LLM 생성 tool args 전부를 `user` provenance로 전송한다 | Guard I2 taint 보장이 실제 제품 경로에서 무효화 | `agentLoop.ts:42-50`, `sdk.py:77-88`; 같은 command가 `user`면 pending, `tool`이면 refused임을 재현 |
| M-P0-01 | V17 public API가 임의 SPD를 허용하지만 write 시 `p=2`만 검사한다 | cue `+1`을 `-1`로 읽는 exact wrong-sign | `G=[[50.5,-49.5],[-49.5,50.5]]`, terminal costs `(-,+)=(4,200)`; `homogeneous_signed_cue.py:154-160,197-220,240-256` |

V17 반례는 선언된 identity 또는 `diag(GL(d),1)` chart 궤도의 좁은 정리를 깨지 않는다. 문제는 public constructor가 그보다 넓은 SPD를 정상 상태로 승인하면서 같은 보장을 암시한다는 점이다.

### 4.2 릴리스·검증 P0

| ID | 결함 | 직접 영향 | 핵심 근거 |
|---|---|---|---|
| Q-P0-01 | CUDA 미가용 경로가 `pytest.skip`이 아닌 조기 return이고 consistency는 assertion이 없다. validation CAUTION도 exit 0이다 | 실행·검증되지 않은 경로가 녹색으로 집계 | `reality_stone/tests/test_rsulf_cuda.py:25-28,66-72,145-164,197-203,329-335`; 실제 `8 passed` 중 CUDA 5경로 미실행 |
| Q-P0-02 | 두 `pyproject.toml`이 동일 `reality_stone==0.2.10`으로 서로 다른 native extension을 만든다 | 같은 wheel 좌표가 작업 디렉터리에 따라 다른 API/성능을 의미 | root `pyproject.toml:2-3,39-42`; nested `reality_stone/pyproject.toml:6-7,21-24`; 실제 root wheel은 `_rust.py` fallback + `clarus/_rust.pyd` |
| Q-P0-03 | outer crate 전체 test composition이 Windows stack overflow로 종료한다 | native release suite를 녹색으로 재현 불가 | `cargo test --locked --manifest-path reality_stone/Cargo.toml` exit 1; 단독 대형 test는 pass, 전체 rsulf 직렬 실행은 재실패 |

## 5. 높은 우선순위 P1

### 수치·수학 계약

- 기본 `BrainRuntimeConfig.axon_delay=True`지만 Rust 경로에는 delay buffer/index가 전달되지 않는다. 동일 seed/상태 1-step에서 activation max diff `0.008280158042908`, Torch/Rust delay index `1/0`, delay sum `2.057143211364746/0`을 재현했다. 기존 parity test는 `axon_delay=False`라 이 차이를 피한다.
- Poisson fixed-point 식은 low root `0.048646633337214085`와 `q=1`을 모두 갖는다. public Newton은 initial guess 1에서 `q=1`을 정상 반환해 최소 survival root 계약을 강제하지 않는다.
- dimensionless checker는 `mass/time`을 `PASS`로 잘못 판정하고 parse error가 있어도 exit 0이며 소스 트리 report를 쓴다. `sympy`도 package dependency 원장에 없다.
- V15 symmetry tolerance는 다른 node의 거대 scale에 오염될 수 있고, V16 tie tolerance의 `max(1,cost)`는 숨은 단위 1을 넣어 공통 scale 불변성을 깬다.
- `HybridState(2,0,0)` 직접 생성이 cube 정의역을 우회하고 L8의 `sigma`는 public state에 있지만 관측된 transition에 참여하지 않는다.

### Guard 제품 통합

- HTTP DAGlet/scheduler/store와 실제 SDK/VS Code 집행은 서로 다른 상태·trace·memory를 사용한다. HTTP benchmark의 invariant가 real sink에 자동 적용되지 않는다.
- `/daglet/{daglet_id}`가 `/daglet/similar`보다 먼저 등록돼 static route가 가려진다. FastAPI가 현 환경에 없어 정적 판정이며 dynamic route probe는 실행하지 못했다.
- memory firewall의 raw episode backing은 payload/hash가 아니라 호출자가 넣은 문자열 set이고, YAML policy는 코드에서 로드되지 않는다.
- API key를 일반 VS Code setting에 둘 수 있고 HTTP surface에는 인증이 없다. bridge child의 error/exit/timeout 처리도 없어 Promise가 영구 대기할 수 있다.

### 패키징·성능·API

- top-level conversion API 두 개와 Clarus 다수 symbol이 optional import 실패를 `None`으로 숨기면서 `__all__`에는 남는다.
- nested native API가 raw CUDA pointer `u64`를 직접 받고 pointer/device/size/lifetime을 검증하지 않는다.
- Rust runtime hot path는 매 step CPU/NumPy 변환과 복사를 반복하고 CUDA attention은 device-wide synchronize를 건다.
- Clarus Python/Cargo/distribution 버전은 각각 1.2.0/1.1.0/0.2.10으로 분리돼 산출물 provenance를 어렵게 한다.

### 테스트·연구 재현성

- 추적된 CI, 단일 품질 command, coverage floor, timeout/warning gate가 없다. network/model-cache/performance/native/GPU 테스트도 marker로 충분히 분리되지 않는다.
- Cargo lock 두 개, `.claude` 설정, `.codex/hooks.json` 등이 ignore 규칙에 걸려 현재 성공 환경을 clean clone에서 재구성하기 어렵다.
- RNG 사용 테스트 69파일 중 28파일에서 file-local seed가 보이지 않았고, benchmark 변환기는 input 0건도 성공한다. experiment corpus도 현재 dirty tree에 의존하면서 content digest를 남기지 않는다.
- ex-codex core는 `Status: COMPLETE`와 `Gate: PASS...` prefix/sentinel 중심으로 판단하고, hook fail-open, 단일 active pointer, lock 없는 revision, 넓은 `gc`를 쓴다. agent 카드의 read-only tool 권한과 필수 stage write 계약도 충돌한다.

## 6. 전체 검증 결과

| 층 | 결과 | 판정 |
|---|---|---|
| Python AST | 647 files, 195,665 lines, syntax error 0 | 파싱 건전성만 통과 |
| Python 전체 pytest | **32 failed, 2,517 passed, 14 skipped, 41 errors, 2 warnings** / 350.02s | 실패; release gate 불가 |
| 수학/core focused | 242 passed with `-W error` | 등록된 좁은 계약 통과; 새 반례 미방어 |
| CE baseline | 61 passed/2 warnings; `-W error`에서는 60 passed/1 failed | warning hygiene 실패 |
| standard validation | 11 PASS, 1 CAUTION, exit 0 | OVERALL CAUTION; machine gate 아님 |
| outer Rust lib | 20 passed | 전체 crate 대체 불가 |
| outer Rust 전체 | stack overflow, exit 1 | 실패 |
| nested Clarus Rust | 32 passed | 통과 |
| ex-codex Rust core | 7 passed | 현재 약한 sentinel semantics 구현 통과 |
| Guard VS Code TS | compile 성공 | runtime/E2E 보안 미검증 |
| Ruff | 감사 범위에 따라 231/242 errors | 품질 gate 실패 |

Python 실패의 주요 묶음은 다음과 같다.

1. `.claude`/`.codex` agent policy mirror drift와 machine verdict token.
2. `.research-tmp/ScienceDB-j00186-00813-V1` 고정 원자료 6개 결손에 따른 fusion fixture 연쇄 error/failure.
3. `benchmarks/neural_tree_algorithm_census_v1.json`, `benchmarks/q0_minimal_abelian_higgs_v1.json` 결손. 두 경로는 `.gitignore`상 추적 허용이지만 실제 파일이 없다.
4. 현재 V18b production bytes/module name과 sealed manifest 불일치. 기존 dirty-tree 사용자 변경과 연관돼 자동 수정하지 않았다.

따라서 73개 비성공 node를 모두 하나의 코드 회귀로 부를 수는 없지만, 현재 checkout이 clean-clone 재현과 전체 gate를 만족하지 않는다는 결론은 변하지 않는다.

## 7. Guard 벤치의 정확한 해석

공식 7종 벤치 결과는 capability 6/6, interception 5/5, injection ASR 0/15, hard route 38/40, external false-allow 0/25, hard false-block 2/15, memory poison block 5/5, AgentDojo-style ASR 0/12, crafted bypass fail-closed다.

이 수치는 core에 공격 provenance를 수동 주입한 조건의 산업 지표다. 실제 VS Code 경로는 LLM args를 `user`로 승격하며 workspace 밖 read와 tool-result egress가 corpus에 없다. 따라서 “Guard core bench는 강하다”는 결론은 가능하지만 “VS Code 제품의 injection ASR가 0%다”라는 결론은 불가능하다.

## 8. 권고 순서

1. **보안 hotfix:** artifact loader를 tensor-only/allowlist로 전환하고, VS Code filesystem containment·egress approval를 추가하며, LLM args를 기본 UNTRUSTED로 바꾼다.
2. **정확성 hotfix:** V17 public domain을 좁히거나 broad invariant를 재설계하고, `axon_delay=True` backend parity를 고정한다.
3. **검증 신호 복구:** CUDA return을 real skip으로 바꾸고 독립 assertion을 추가한다. CAUTION/FAIL을 machine-readable nonzero exit로 전달한다.
4. **배포 정체성 단일화:** 한 build root에서 두 extension을 명시적으로 묶거나 distribution을 분리한다. clean venv wheel-install smoke와 backend identity assertion을 둔다.
5. **전체 gate 복구:** outer Rust stack overflow를 고치고, 누락 benchmark/payload의 추적·취득 계약을 정하며, default Python suite를 hermetic unit와 opt-in integration으로 나눈다.
6. **수학 경계 강화:** fixed-point branch, dimension registry, V15/V16 tolerance, finite-host state 정의역을 public constructor에서 검사하고 반례를 회귀 테스트로 편입한다.
7. **아키텍처 정리:** Guard의 HTTP/SDK/extension을 하나의 execution service·trace·memory에 연결하고 ex-codex gate를 구조화 schema와 원자적 state로 바꾼다.
8. **CI·provenance:** OS/Python/native matrix, lint/coverage/warnings, seed·input/output hash, dirty-tree 기록을 단일 품질 command로 고정한다.

## 9. 재현물과 상세 감사

- `artifacts/architecture.md`: 런타임·패키지·보안·성능·ex-codex 상세 추적
- `11-math.md`: 형식 지위, exact 반례, 무차원성·고정점·허용오차 감사
- `artifacts/quality.md`: 테스트·빌드·CI·benchmark·experiment 품질 감사
- `artifacts/validation-summary.md`: 실제 명령과 종료 결과 원장
- `artifacts/math_counterexamples.py`: 수학/API 반례 재현
- `artifacts/delay_backend_probe.py`: Torch/Rust delay mismatch 재현
- `artifacts/python_inventory.py`: 비-RBE Python AST inventory

## 10. 최종 판정

분석 산출물은 완료됐다. 그러나 open P0가 7건이며 `20-audit.md`의 Gate는 정직하게 `BLOCKED`다. 소스 수정이 요청 범위가 아니었으므로 구현·구현 후 검증은 SKIPPED했다. 다음 작업은 기능 확장보다 보안 경계, public 수학 정의역, 검증 신호, 배포 정체성을 먼저 고정하는 것이 맞다.
