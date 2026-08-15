# 검증 실행 요약

검증일: 2026-08-15  
범위: 현재 작업 트리, 단 `RBE/` 전체 제외  
원칙: 제품 소스는 수정하지 않았고, 사용자 변경과 환경/아티팩트 결손을 제품 회귀와 분리했다.

## Python

| 검사 | 결과 |
|---|---|
| AST inventory | Python 647파일, 195,665줄, class 1,017, function/async function 6,763, test 파일 239, syntax error 0 |
| pytest collection | 2,604 tests collected, exit 0 |
| 전체 pytest (`--basetemp`를 workspace 안에 고정) | **32 failed, 2,517 passed, 14 skipped, 2 warnings, 41 errors**, exit 1, 350.02초 |
| `pip check` | broken requirement 없음, exit 0 |
| Ruff, 제품·연구 혼합 범위 | 231 errors, 그중 98 safe-fixable(추가 unsafe fix 24), exit 1 |
| Ruff, 품질 감사 표준 범위(`tests`, `reality_stone/python`, `reality_stone/tests`, `examples`, `experiments`) | 242 errors, exit 1 |

전체 pytest 명령:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=no --basetemp '_workspace\ce\repository-code-analysis-20260815\artifacts\pytest-basetemp-full'
```

실패/오류 묶음:

1. `tests/test_canonical_document_policy.py`: `.claude`/`.codex` 역할 카드의 formal provenance 정책 미러가 드리프트했고 일부 카드가 machine verdict token을 포함한다.
2. fusion ScienceDB·polarized-evidence·spin-control: `.research-tmp/ScienceDB-j00186-00813-V1`의 고정 원자료 6개가 없어 fixture가 fail-closed한다. 이 결손이 연쇄적으로 다수 error/failure를 만든다.
3. neural-tree census: `benchmarks/neural_tree_algorithm_census_v1.json`이 없다.
4. q0 manifest gate: `benchmarks/q0_minimal_abelian_higgs_v1.json`이 없다.
5. V18b: 현재 production module bytes/name과 sealed manifest가 일치하지 않아 4개 계약 테스트가 실패한다. 현재 작업 트리에 이미 있던 사용자 변경과 연관되므로 자동 수정하지 않았다.
6. 경고 2개: Torch sparse invariant 명시 부재와 sparse CSR beta 경고. 동일 범위의 baseline은 `-W error`에서 1개 실패한다.

누락된 두 benchmark JSON은 `.gitignore`가 명시적으로 추적 허용하지만 실제 tracked/untracked 파일이 없었다. ScienceDB 원자료는 외부 고정 payload 결손으로 분류한다. 따라서 현재 checkout은 전체 테스트를 녹색으로 재현할 수 없다.

## 수학·수치 집중 검증

| 검사 | 결과 |
|---|---|
| current math/core focused pytest, `-W error` | 242 passed |
| CE baseline | 61 passed, warnings 2 |
| CE baseline, `-W error` | 60 passed, 1 failed |
| `tests/run_validation.py` | 11 PASS, 1 CAUTION; OVERALL CAUTION |
| counterexample scratch | V17 wrong-sign, dimension checker false PASS, fixed-point alternate branch, V15/V16 scale/tolerance, finite-host boundary를 재현 |
| Torch/Rust default delay probe | `max_activation_diff=0.008280158042908`, delay index `1/0`, delay sum `2.057143211364746/0`, mismatch 재현 |

재현물:

- `artifacts/math_counterexamples.py`
- `artifacts/delay_backend_probe.py`

## Rust·TypeScript

| 검사 | 결과 |
|---|---|
| outer geometry Rust, `--lib --no-default-features` | 20 passed |
| outer geometry Rust 전체, `--locked` | unit/layers/ops 통과 후 `tests/rsulf.rs`에서 Windows `STATUS_STACK_OVERFLOW`, exit 1 |
| nested Clarus Rust, `--lib` | 32 passed |
| nested Clarus Rust, `--lib --features python` | 32 passed |
| ex-codex core Rust | 7 passed |
| Guard VS Code extension `npm run compile` | TypeScript compile 성공 |

outer 전체 Rust 실패는 단일 대형 임계값 테스트만 실행하면 통과하지만, `rsulf` 전체를 `--test-threads=1`로 실행해도 다시 stack overflow가 발생했다. 따라서 20개 lib test 통과와 전체 crate release suite 통과를 구분한다.

## Clarus Agent Guard 공식 7종 벤치

`clarus-guard-bench` 규약에 따라 산업 지표만 기록한다.

| 지표 | 결과 |
|---|---:|
| capability property | 6/6 pass |
| untrusted fuzz execution | 0/150 |
| interception | 5/5 pass |
| prompt-injection ASR | 0/15 = 0% |
| injection benign pass | 100% |
| hard route accuracy | 38/40 = 95% |
| external false-allow | 0/25 = 0% |
| hard false-block | 2/15 = 13.3% |
| memory poison block | 5/5 = 100% |
| memory benign commit | 3/3 = 100% |
| AgentDojo-style benign | 4/4 = 100% |
| AgentDojo-style ASR | 0/12 = 0% |
| crafted audit bypass | fail-closed |

주의: 이 벤치는 core에 공격 provenance를 수동으로 주입한다. 실제 VS Code `AgentSession`은 LLM 생성 인자를 전부 `user` provenance로 보내므로, 위 ASR/false-allow 수치는 실제 extension E2E 보안 보장을 검증하지 않는다. 파일 시스템 workspace 경계와 원격 tool-result egress도 해당 벤치 범위 밖이다.

## CE 형식 게이트

- `check contract`: PASS
- `check lanes`: PASS
- stage 상태: `COMPLETE/SKIPPED`로 8개 모두 존재
- `check gate`, `check build`, `check final`: 예상대로 실패. `20-audit.md`가 open P0를 이유로 `Gate: BLOCKED`이기 때문이다.

이는 분석 산출물 실패가 아니라 제품·구현 완결성을 거짓 PASS로 만들지 않은 최종 판정이다.
