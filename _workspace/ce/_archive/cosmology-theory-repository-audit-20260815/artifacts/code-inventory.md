# 우주론 코드 의존 경로 독립 감사

Status: **COMPLETE**  
Snapshot: `5414336ae2ff20197efe3bf8a92ec5183ad079aa` + 현재 dirty worktree  
Contract: `../00-contract.md`의 C1--C6 및 P0/P1 기준  
Method: `rg --files`, 참조/호출 검색, Python AST import/test-function 감사, JSON/경로 존재 검사. 이 레인은 소스와 role stage 파일을 수정하지 않았고 테스트를 재실행하지 않았다. 무시된 `/.tmp/` 복제본은 live evidence에서 제외했다.

## 1. 결론

- C1의 저근 고정점과 항등근 분리는 `core_axioms.py` 및 12개 테스트에서 가장 잘 닫혀 있다. 반면 구형 `bootstrap_solver.py`는 저근만 시험한다.
- C2는 **기계 판정 지위가 잘못 승격된 P0 후보**다. `omega_dm0`, `omega_lambda0`를 `ce_prediction`으로 내보내고 테스트가 이를 고정하지만, 같은 저장소 scorecard는 이를 `Phenomenology`로 둔다.
- C3의 residual FLRW/CPL/GR 및 DESI 공분산 계산은 테스트가 촘촘하다. 그러나 고정 모형의 실제 회귀 결과는 `REJECT`인데 CLI는 항상 exit 0이다.
- C4의 `hubble_tension.py`는 고립된 Open/Phenomenology toy이고, `cosmological_constant_holographic_gate.py`는 정본 validator가 명시적으로 `CODE_ONLY`로 격리하는 legacy artifact다. 둘 다 독립 관측 검증으로 승격할 수 없다.
- C5는 동결된 DESI 내장 snapshot까지만 재현 가능하다. 원시 mean/cov 파일, 실제 recombination history, H0 covariance bundle 및 배정된 future holdout은 없다.
- C6: 선정한 12개 test module의 **116개 test 함수에는 AST상 무-assert test가 0개**다. 위험은 assertionless pytest가 아니라, 상수 복제·자기참조 fixture·보고서 CLI의 false-green exit다.

## 2. 범위와 규모

| 묶음 | 파일 | 물리 줄 수 | 비고 |
|---|---:|---:|---|
| `reality_stone/.../clarus` 코어 | 6 | 2,678 | constants, legacy bootstrap, core axioms, exact/heuristic dimensionless, model selection |
| `examples/physics` 우주론/게이트 | 14 | 4,266 | ratio, residual, legacy cosmology/H0, primordial/Λ 및 보조 gate |
| validator/runner | 6 | 8,790 | manuscript loops, holdout validator, run_validation, scorecard |
| 직접 관련 테스트 | 12 | 2,237 | 116 test functions |
| 정본/보조 문서 | 29 | 7,406 | README, 상수/우주론/등호이전/형식 원장 |
| JSON 원장/산출물 | 4 | 1,225 | core manifest, holdout v1/v2, scorecard result |
| **선정 live set** | **71** | **26,602** | 키워드 전체가 아니라 실제 생성·전달·검증 경로 |

주요 문서군은 `README.md`, `docs/{README,axium,상수,경로적분}.md`, `docs/3_상수/{3,7,9,10,11,12,13}_*.md`, `docs/9_등호이전/05{n,o}_*.md`, `docs/참조/{형식적_수학_모델과_증명,핵심_정리_증명,무차원_감사_수학,이론물리_보존_원장,epsilon_제1원리_유도}.md` 및 경로적분/유도 보조 문서다.

## 3. 실제 의존 경로

| 생성/전달 경로 | 소비자 | 감사 결과 |
|---|---|---|
| `constants.py:26-28` literal 3-layer 비율 | `cosmology_ratio_audit.py:20-41`이 AST literal로 읽음 | import가 아니라 `AnnAssign` literal 파서라 식으로 바꾸면 깨짐 |
| ratio audit `CE_RATIOS` | `ce_residual_forward_model.py:25-26,46-58` | residual의 기본 밀도·H0·rd·sigma8·CPL/GR 경계 |
| `cosmology.py:14-100` 수치 유틸 | residual `:25`; H0 toy `:78` | 유틸은 live, `cosmology.py`의 자체 LO background/CLI는 별도 legacy 경로 |
| residual + 내장 DESI `:1077-1125` | residual tests; `improvement_loop_engineering.py:216-229,545-659` | 동적 import, 고정 모형/동일자료 scale-fit/CV 진단 |
| rejection + improvement loop | `validate_manuscript.py:17-26,1214-1264` | fail-closed 문서/산술/라우팅 검사; 외부 artifact 검증은 `NOT_RUN` (`:1237-1240`) |
| `core_axioms.py` | `core_axiom_loop.py`; 12 tests | 두 근·안정도·임계 조건을 분리 |
| `benchmarks/core_claims_v1.json` | `core_model_selection.py:293-310,850-875`; gate/tests | 현재 내용을 hash하나 외부에 고정된 expected digest는 없음 |
| primordial gate | scorecard `:28-33`, proof attempt, 6 tests | Bridge/Phenomenology readout; 관측 `A_s`도 같은 모듈에 고정 |
| holographic Λ gate | validator path check `:440-475`와 독립 재계산 `:853-890` | 함수 import/직접 pytest 없이 구현을 복제 검산 |
| `hubble_tension.py` → discrimination gate | 문자열/개념 참조만 | README/정본 validator/pytest에 연결되지 않은 code island |

## 4. 테스트 매핑

| 대상 | 테스트 수 | 확인 범위 | 비어 있는 핵심 범위 |
|---|---:|---|---|
| `core_axioms.py` | 12 | 저근/항등근, 안정도, 분기 임계, 민감도 | 물리적 Ω 식별은 별도 bridge |
| `bootstrap_solver.py` | 3 | Newton/Brent 저근, residual, Jacobian (`test_bootstrap_solver.py:8-36`) | 초기값 1의 항등근, 분기 라벨, CLI 실패 exit |
| core model selection | 12 | 27-model grid, manifest schema, underidentification | manifest expected digest의 외부 anchor |
| exact dimensionless API | 15 | 단위군/exp/가산/분수 거듭제곱 | heuristic registry와 실제 우주론 식 연결 |
| ratio audit | 3 | 네 baseline에 4% 근접, 상수 source | 오류/공분산/출처, zero-reference edge, stale coverage flags |
| residual forward model | 25 | background/growth, early-rd, BAO/covariance/parser/provenance | CLI exit, NaN/비물리 density/CPL/μ, 임의 covariance PSD |
| recombination adapter | 5 | hash/order/metadata/error/수치 적분 | 실제 CLASS/CAMB/HyRec export; fixture는 `test_recombination_drag_adapter.py:28-72`의 manufactured analytic table |
| primordial readout | 6 | 후보 수치/상태 | 상태 문자열 자체가 구현에 고정; 독립 data likelihood 없음 |
| proof ledger | 5 | 지위/열린 항목 회귀 | 자연에 대한 검증 아님 |
| holdout manifest | 18 | digest/schema/unassigned/readiness/kill rule | 배정된 실제 미래자료 없음 |
| run_validation | 7 | summary/dimension API/현재 CAUTION | 실제 cosmology source 식을 파싱하지 않음; process exit |
| manuscript loop | 5 | mutation fail-closed, text/JSON exit parity, release artifact 요구 | 네트워크/외부 artifact 자체는 별도 |

직접 pytest import가 없는 중요 실행물: `hubble_tension.py`, `cosmological_constant_holographic_gate.py`, `cosmology_discrimination_gates.py`; `cosmology.py`는 residual이 네 유틸만 간접 사용하며 자체 background/growth/CLI는 직접 시험되지 않는다.

## 5. P0/P1 후보

| 등급 | 근거 | 영향/조치 |
|---|---|---|
| **P0** | `ce_residual_forward_model.py:825-855`가 Ωb/ΩDM/ΩΛ 모두 `ce_prediction`; `test_ce_residual_forward_model.py:494-504`가 이를 강제. 반면 `scorecard.py:173-205`는 Ωb=`Bridge`, ΩDM/ΩΛ=`Phenomenology`. | C2 형식 지위가 실행 경로마다 모순. DM/DE를 empirical/phenomenology 또는 supplied bridge로 낮추고 테스트를 바꿔야 함. |
| **P0 후보** | 고정 DESI 모형은 `test_ce_residual_forward_model.py:308-368`에서 χ²=37.10026, p=3.9957e-4, `REJECT`; CLI는 verdict를 출력(`ce_residual_forward_model.py:1653-1659`)한 뒤 무조건 `return 0` (`:1708`). | exit code를 gate로 쓰면 관측 결론이 뒤집힌다. report-only로 이름/문서화하거나 REJECT 시 nonzero wrapper 필요. |
| **P0 후보** | `run_validation.py:342-360`이 `CAUTION`을 반환하지만 `:363-364`는 `SystemExit` 없이 종료. 현재 `CAUTION`은 `test_run_validation_consistency.py:109-113`이 고정. `scorecard.py:372-399,501-511`도 FAIL/WARN/CAUTION과 무관하게 exit 0. | README의 validation 명령을 CI gate로 쓰면 false green. 구조화 결과를 명시적 exit policy에 연결해야 함. |
| **P1 (승격 시 P0)** | `hubble_tension.py:259-287`의 `om_b_h2`는 docstring상 고정 baryon density지만 함수 본문에서 사용되지 않음; 기본 0.02237은 `:290-304`. toy sound speed는 `:10,46-48`; target 5.6을 최적화하는 분기는 `:481-511`; 항상 exit 0 (`:514`). | 입력 변화 불변 테스트를 추가하고 baryon-loaded sound speed/실제 recombination을 연결하기 전 H0 수치 결론 금지. 현재 헤더가 Open/Phenomenology로 철회(`:18-20`)하므로 code-only toy로 유지. |
| **P1** | background default가 셋: `cosmology.py:344-355` LO식은 Ωm≈0.307918, ΩΛ≈0.692082; calibrate 기본은 `:320-334`의 0.315/0.685; residual은 `constants.py:26-28`의 raw 0.3110/0.6891을 `ce_residual_forward_model.py:89-98`에서 Ωm≈0.310969/ΩΛ≈0.689031로 정규화. | 동일 이름의 실행물이 서로 다른 우주를 기본값으로 생성. canonical residual 외 legacy CLI를 명시적으로 격리하고 default provenance 출력 필요. |
| **P1** | ratio audit는 background/growth가 생긴 뒤에도 `has_background_expansion_model=False`, `has_growth_model_for_s8=False` (`cosmology_ratio_audit.py:165-174`); residual은 둘 다 True (`ce_residual_forward_model.py:105-117`). | coverage summary가 stale. 합쳐진 inventory를 source of truth로 만들 것. |
| **P1** | Λ gate는 “not independent”, “zero free parameters”라 주장(`cosmological_constant_holographic_gate.py:1-17,68-70`)하면서 ΩΛ=0.6891과 관측 2.24 meV를 고정(`:24-29`)해 0.2% 기준으로 exit(`:79-103`). validator는 그 문구를 `CODE_ONLY`로 격리(`validate_manuscript.py:440-475`)하고 공식을 별도 복제(`:853-890`). | 정본 theorem이 아니라 supplied entropy/H0 bridge + observational comparison. 직접 테스트/공용 함수 없이 이중 구현이 drift할 수 있음. |
| **P1** | `bootstrap_solver.py:54-95,206-243`는 초기값에 따라 항등근도 정답으로 받을 수 있으나 3 tests는 저근만(`test_bootstrap_solver.py:8-36`). CLI PASS/FAIL 출력 뒤 exit policy 없음(`bootstrap_solver.py:306-321`). | branch-aware API/테스트와 nonzero failure exit 필요. C1 정본은 `core_axioms.py`로 한정. |
| **P1** | `dimensionless_checker.py:482-525`는 문자열에 `exp`나 `/`가 있으면 PASS 가능한 heuristic이며 `:598-616`은 source tree에 보고서를 쓰고 exit gate가 없다. 실제 residual/H0/Λ 식은 registry에 연결되지 않는다. `parse_expr` 직접 의존(`:20`)은 pyproject direct dependency가 아니고 torch의 transitive sympy(`pyproject.toml:6-12`, `uv.lock:3285-3303`)에 기대고 있다. | exact `dimensionless.py` API로 실제 표현식을 구성하고 heuristic 결과는 proof로 사용 금지; sympy를 직접 선언. |
| **P1** | `CEForwardParams.__post_init__`는 rd/H0/Tcmb/Neff만 검사(`ce_residual_forward_model.py:60-70`); density, `w0/wa`, σ8, μ, finite/zero norm은 미검사. BAO parser도 finite/PSD를 보장하지 않음(`:1141-1194,1352-1368`). | NaN/음의 density/비양정 covariance가 결론을 바꿀 수 있는 미시험 domain. |
| **P1** | `cosmology.py:31-54` Simpson은 점 수가 짝수면 마지막 점을 조용히 버림. | public utility의 적분 구간이 축소됨. 짝수 입력을 거부하거나 마지막 구간 보정 및 테스트 필요. |

## 6. 관측 입력·fixture·누락 참조

| 항목 | 현재 상태 |
|---|---|
| density baselines | `cosmology_ratio_audit.py:109-138`에 Planck/ACT/SPT 네 압축값을 URI, release, covariance 없이 literal로 저장; `relative_error(..., observed=0)`는 0을 반환(`:141-144`). |
| DESI DR2 | 13-vector/13×13 covariance가 residual `:1077-1125`에 내장. upstream commit/URL/bytes/SHA256는 `improvement_loop_engineering.py:36-56`에 별도로 pin되어 있어 정적 provenance는 보강됐으나 raw 파일은 checkout에 없음. |
| future holdout | v2는 DESI를 이미 본 exploratory data라고 명시(`cosmology_future_holdout_v2.json:187-205`); future slot은 unassigned/null (`:207-225`). confirmatory claim 불가. |
| core manifest | `benchmarks/core_claims_v1.json`은 존재하고 현재 hash는 출력하지만 `test_core_model_selection.py`에 외부 expected digest anchor가 없다. |
| scorecard JSON | `tests/scorecard.py:501-511`만 `tests/scorecard_results.json`을 덮어쓰며 다른 live code reader가 없다. generated dead-end artifact. |
| recombination | 실제 CLASS/CAMB/HyRec fixture 없음. 5 tests는 “not a solver” synthetic bytes를 즉석 생성(`test_recombination_drag_adapter.py:28-72,107-123`). |
| H0 bundle | `docs/3_상수/10_H0_readout_law_audit.md:33-43`, `11_TDCOSMO_real_covariance_audit.md:3-5,44-52`, `12_H0_source_role_readout_paper_draft.md:8-10`가 요구하는 `examples/physics/h0_readout/`, target manifest, source scout, TDCOSMO JSON/converters/Fisher suite, paper package gate가 모두 없음. 문서가 `[미완성]`으로 공개하므로 silent break가 아니라 P1 재현성 gap. |
| DESI raw refs | `docs/9_등호이전/05o_...md:193-200`의 BGS/ALL mean/cov 파일이 없음. |
| broken ref | `docs/참조/무차원_감사_수학.md:107`의 `examples/physics/flrw_scale_map_gate.py`가 없음. 이 참조는 주변에서 `[미완성]`으로 표시되지 않아 정정 필요. |

## 7. 권장 validation 명령

아래 첫 명령이 코드 회귀의 최소 묶음이다. 단, 통과는 C6 구현 일관성만 뜻한다.

```powershell
uv run --extra dev python -m pytest -q tests/test_core_axioms.py tests/test_bootstrap_solver.py tests/test_core_model_selection.py tests/test_dimensionless.py tests/test_cosmology_ratio_audit.py tests/test_ce_residual_forward_model.py tests/test_recombination_drag_adapter.py tests/test_primordial_spectrum_readout_gate.py tests/test_proof_completion_attempt.py tests/test_holdout_preregistration.py tests/test_run_validation_consistency.py docs/2_경로적분과_응용/test_manuscript_loop_engineering.py
uv run python experiments/preregistration/validate_holdout_manifest.py
uv run python docs/2_경로적분과_응용/validate_manuscript.py --json
uv run python examples/physics/core_model_selection_gate.py --json
```

판정 주의: `ce_residual_forward_model.py`, `tests/run_validation.py`, `tests/scorecard.py`, `hubble_tension.py`, `primordial_spectrum_readout_gate.py`는 현재 process success를 과학적 PASS로 해석하면 안 된다. release 주장 전에는 improvement loop의 `--require-live-artifacts` 경로로 실제 DESI mean/cov 및 고정된 외부 artifact hash를 공급해야 한다.

## 8. 우선 처리 순서

1. C2 provenance role 모순과 그 테스트를 정정한다(P0).
2. REJECT/CAUTION/FAIL을 exit code 또는 명시적 report-only contract에 연결한다(P0 후보).
3. H0/Λ code-only 경로를 정본에서 계속 격리하고, unused baryon 입력·실제 recombination/covariance가 생기기 전 승격하지 않는다.
4. residual 하나를 canonical background로 지정하고 LO/calibrate/H0 toy의 서로 다른 default를 명시한다.
5. stale coverage, missing `flrw_scale_map_gate.py`, raw fixture/provenance 및 domain/PSD 테스트를 닫는다.
