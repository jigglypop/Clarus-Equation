# 비-RBE 저장소 통합 상태 감사

Status: COMPLETE

Gate: BLOCKED

PREDECESSOR: 12-routes.md

## 판정

감사 자체는 완료됐지만 구현·릴리스 게이트는 통과하지 못했다. 재현 가능한 P0가 열려 있고, 사용자의 요청은 분석이지 소스 수정이 아니므로 임의로 결함을 고치거나 상태를 PASS로 승격하지 않았다. `RBE/`는 근거·실행·결론에서 완전히 제외했다.

| Claim | 형식 지위 | 통합 심각도 | 판정 |
|---|---|---:|---|
| C1 아키텍처·공개 API·책임 경계가 일관적인가 | 미완성 | P1 | 세 제품 경계는 식별되지만 이중 build root, Guard의 HTTP/SDK 분리, silent optional API가 공개 계약을 갈라 놓는다. |
| C2 수학·수치 구현이 선언된 정의역과 보장을 지키는가 | 미완성 | P0 | V17 공개 SPD 정의역의 exact wrong-sign 반례와 기본 axon-delay backend 불일치가 있다. 좁은 identity/chart 궤도 정리와 집중 테스트 통과는 보존된다. |
| C3 테스트가 실제 실행·실패 경로를 정직하게 방어하는가 | 미완성 | P0 | CUDA 조기 return이 pass로 집계되고 수치 consistency assertion이 없으며 CAUTION도 exit 0이다. 전체 Python suite도 녹색이 아니다. |
| C4 설치·빌드·실행·연구 run이 재현 가능한가 | 미완성 | P0 | 동일 name/version의 두 wheel 정의, ignored lock, CI 부재, 전체 outer Rust suite stack overflow, 누락 benchmark/payload가 있다. |
| C5 구체적인 P0/P1/P2가 무엇인가 | 산출 | P0 | 아래 open ledger와 세 독립 감사 산출물에 재현 근거를 고정했다. |

## Open P0 ledger

| ID | 영역 | 재현·근거 | 게이트 해제 조건 |
|---|---|---|---|
| A-P0-01 | 아티팩트 보안 | `reality_stone/python/reality_stone/clarus/engine.py:137-140`이 외부 `.ce.pt`를 `weights_only=False`로 역직렬화한다. CLI 입력은 `:1796-1806`, `:1855-1862`에서 직결된다. | tensor-only 포맷 또는 `weights_only=True`와 key/type/shape/dtype/size allowlist, hostile artifact negative test |
| A-P0-02 | 파일 egress | Guard VS Code schema가 absolute path를 허용하고 `agentLoop.ts:23-25,45-49`에 workspace containment가 없다. `vscode_bridge.py:27-34`의 읽기 결과는 `providers.ts:36-67,98-116`을 통해 원격 모델로 돌아간다. `C:\Windows\win.ini` 무승인 읽기를 재현했다. | canonical realpath allow-root, symlink 재검사, 외부 읽기 승인, redaction/egress E2E test |
| A-P0-03 | provenance | 실제 `agentLoop.ts:42-50`이 LLM 생성 args 전부를 `user` provenance로 전송하고 `sdk.py:77-88`이 그대로 신뢰한다. core bench의 수동 `tool` taint와 제품 경로가 다르다. | argument-level provenance/taint 또는 기본 UNTRUSTED, 실제 extension-to-sink injection E2E bench |
| M-P0-01 | 수학·API | V17 public constructor는 임의 SPD를 받고 write는 `p=2`만 확인한다. `G=[[50.5,-49.5],[-49.5,50.5]]`, cue `+1`에서 비용 `(-,+)=(4,200)`으로 `-1`을 읽는 exact 반례를 재현했다. | API 정의역을 identity/선언 chart 궤도로 제한·검증하거나 broad SPD에서 성립하는 invariant로 재설계 |
| Q-P0-01 | 검증 신뢰성 | `test_rsulf_cuda.py`의 CUDA 5경로가 미가용 시 return하지만 `8 passed`; consistency는 출력만 하고 assert하지 않는다. `run_validation.py`는 CAUTION인데 exit 0이다. | skip/pass 분리, 독립 수치 oracle/assertion, machine-readable nonzero gate |
| Q-P0-02 | 산출물 정체성 | 루트와 `reality_stone/pyproject.toml`이 모두 `reality_stone==0.2.10`이면서 각각 `clarus._rust`와 `_rust`를 빌드한다. 실제 root wheel은 fallback `_rust.py` + native `clarus/_rust.pyd`였다. | 단일 정본 wheel 또는 배포명 분리, clean venv wheel-install backend identity test |
| Q-P0-03 | native release suite | `cargo test --locked --manifest-path reality_stone/Cargo.toml`이 unit/layers/ops 후 `tests/rsulf.rs`에서 `STATUS_STACK_OVERFLOW`; 단일 테스트는 pass하나 전체 rsulf 직렬 실행도 실패한다. | clean/locked 전체 suite 안정 통과와 실패 seed/resource profile 보존 |

## Open P1 묶음

세부 ledger는 `artifacts/architecture.md`, `11-math.md`, `artifacts/quality.md`에 있다. 통합 우선순위는 다음과 같다.

1. 기본 `axon_delay=True`인데 Rust dispatch가 delay state를 전달·갱신하지 않는다. 동일 상태 1-step에서 max activation diff `0.008280158042908`, delay index `1/0`을 재현했다.
2. dimensionless checker가 `mass/time`을 거짓 PASS하고 parse error가 있어도 exit 0이며 보고서 파일까지 쓴다. 고정점 API는 `q=1` 가지를 허용해 최소 survival root 계약을 강제하지 않는다.
3. V15 symmetry tolerance가 다른 node scale에 오염되고, V16 tie 판정은 숨은 단위 1 때문에 공통 scale에 불변이 아니며, finite-host public state/coupling 경계가 정의보다 넓다.
4. Guard DAGlet HTTP 면과 실제 SDK/extension 집행면이 분리되고, `/daglet/similar` route shadow, 비어 있는 raw-episode backing, 미사용 YAML policy, SecretStorage 부재, bridge child 장애 무기한 대기가 있다.
5. public conversion 함수와 다수 Clarus symbol이 import 실패를 `None`으로 숨긴다. raw CUDA pointer API, 매 step CPU/NumPy 복사, device-wide synchronize도 안정성·성능 계약을 약화한다.
6. CI/coverage/test taxonomy가 없고 lock·hook 일부가 무시되며 network/cache/벽시계/RNG 의존 테스트와 구현 독립 oracle 공백이 섞인다.
7. benchmark·experiment 입력 hash/seed/corpus provenance가 닫히지 않고 빈 변환도 성공한다. 현재 전체 Python 실패에는 누락된 고정 JSON과 ScienceDB payload가 포함된다.
8. ex-codex gate는 forge 가능한 status prefix/sentinel, fail-open hook, 단일 active pointer, lock 없는 revision, 넓은 gc를 사용하며 agent의 write 권한과 필수 산출물 계약도 충돌한다.

## 검증 원장

- Python: 2,604 collected; 전체 `32 failed, 2,517 passed, 14 skipped, 41 errors, 2 warnings`.
- 수학/core 집중: 242 passed with `-W error`; 반례 scratch는 별도로 모두 재현.
- Rust: outer lib 20 passed; outer 전체는 stack overflow; nested Clarus 32 passed; ex-codex core 7 passed.
- TypeScript: Guard VS Code extension compile 성공.
- Guard 공식 7종 core bench: injection ASR 0/15, AgentDojo-style ASR 0/12, false-allow 0/25, false-block 2/15. 단 실제 VS Code provenance와 filesystem egress는 미포함.
- Ruff: 선택 범위 231건, 표준 품질 범위 242건.

## Gate 근거

`Gate: PASS`의 필요조건인 open P0=0을 만족하지 않는다. 이번 run은 결함을 은폐하지 않고 분석 결과를 전달하기 위한 것이므로 `30-implementation.md`와 `31-validation.md`는 명시적으로 SKIPPED한다. Gate 해제에는 최소한 위 P0의 소스 수정, hostile/negative test, clean wheel/native 전체 검증이 필요하다.
