# 비-RBE 검증·테스트·빌드·연구 워크플로 품질 감사

- 감사일: 2026-08-15 (Asia/Seoul)
- 범위: `tests/`, `reality_stone/tests/`, `benchmarks/`, `examples/`, `experiments/`, `.claude/`, `.codex/`, 루트 및 중첩 `pyproject.toml`, CI/빌드 설정
- 명시적 제외: `RBE/` 전체와 그 아래 Cargo 테스트·설정·산출물
- 방식: 읽기 전용 정적 감사와 재현 가능한 로컬 명령 실행. 소스와 기존 사용자 변경은 수정하지 않았다.
- 판정 원칙: 기계 테스트의 성공, 수학 명제의 증명, 경험적 주장의 지지, 릴리스 산출물의 검증을 서로 다른 상태로 취급한다.

## 결론

현재 비-RBE 저장소의 품질 신호는 릴리스 게이트로 신뢰할 수 없다. 특히 (1) 실행하지 않은 CUDA 테스트가 `PASS`로 집계되고 핵심 일치성 검사가 수치만 출력하며 assertion을 하지 않고, (2) 같은 이름·버전으로 서로 다른 네이티브 확장을 만드는 두 빌드 정의가 공존하며, (3) 비-RBE 최상위 Rust 전체 테스트가 현재 구성에서 스택 오버플로로 실패한다. 그 위에 CI가 전혀 없고, 표준 검증 스크립트가 `CAUTION`/`FAIL`을 출력해도 종료 코드 0을 반환한다. 따라서 “명령이 녹색이었다”는 사실만으로 코드·수학·연구 주장의 어느 층도 승격해서는 안 된다.

감사 심각도는 **P0 3건, P1 9건, P2 4건**이다.

| ID | 등급 | 요약 | 직접 관측 |
|---|---:|---|---|
| Q-P0-01 | P0 | 검증 신호가 실제 미실행·무검증을 `PASS`/exit 0으로 위장한다 | CUDA 미가용인데 8 passed, 그중 5개는 `return`; consistency는 assertion 없음; validation은 `CAUTION`인데 exit 0 |
| Q-P0-02 | P0 | 동일한 배포 이름·버전이 서로 다른 Rust 확장을 빌드한다 | 루트와 `reality_stone/`의 두 `pyproject.toml` 모두 `reality_stone==0.2.10`; 각각 `clarus._rust`와 `_rust`를 빌드 |
| Q-P0-03 | P0 | 현재 비-RBE Rust 전체 테스트가 실패한다 | `reality_stone/Cargo.toml` 전체 테스트가 `rsulf` 대형 임계값 테스트 부근에서 `STATUS_STACK_OVERFLOW` |
| Q-P1-01 | P1 | CI와 단일 품질 게이트가 없다 | 추적된 workflow/CI 설정, `project.scripts`, nox/tox/Makefile 없음 |
| Q-P1-02 | P1 | clean clone에 재현 잠금과 훅 구성이 빠진다 | 두 Cargo lock 및 중첩 uv lock이 무시됨; `.claude/` 전체와 `.codex/hooks.json`도 무시됨 |
| Q-P1-03 | P1 | 기본 pytest 수집에 네트워크·모델 캐시·벽시계 의존 테스트가 섞인다 | 빈 오프라인 캐시에서 기본 수집 GPT-2 테스트 실패; 1초 성능 임계값 존재 |
| Q-P1-04 | P1 | 테스트 계층·커버리지 정책이 없고 전체 Python suite가 대량 실패한다 | marker는 `cuda` 하나, coverage/timeout/warnings gate 없음; 2,604개 중 32 failed, 41 errors |
| Q-P1-05 | P1 | 난수와 테스트 합성에 대한 전역 재현 정책이 없다 | RNG 사용 Python 테스트 69개 중 28개는 파일 내 seed 없음; Rust `rand::random` 다수 |
| Q-P1-06 | P1 | 여러 테스트의 oracle이 구현 독립적이지 않거나 아예 없다 | GPT-2 last-layer 오차 무assert, LLM inference dict 반환, validation adapter가 동일 scorecard와 자기 비교 |
| Q-P1-07 | P1 | 벤치마크 변환·모델·결과 provenance가 clean clone에서 닫히지 않는다 | 변환기가 입력 0건도 성공; 모델/CSV/JSON 산출물 무시; end-to-end 수·해시 회귀 없음 |
| Q-P1-08 | P1 | 실험 결과가 현재 작업 트리 자체를 데이터로 사용하고 기록이 모순된다 | corpus digest 없음; read error 묵살; OOD 결과 문서가 3 seeds와 single seed를 동시에 주장 |
| Q-P1-09 | P1 | 연구 코어 및 agent 카드가 형식 완결성을 실제로 강제하지 못한다 | 문자열 prefix 상태 승인, 불완전 run archive, 두 번째 stop 허용, read-only agent에 산출물 쓰기 요구 |
| Q-P2-01 | P2 | 정적 품질 부채가 크고 독립 실행 경로가 깨진다 | Ruff 242건; `test_rsulf_inference_pipeline` 미정의 호출 |
| Q-P2-02 | P2 | 현재 수집 수가 untracked 테스트에 의존한다 | untracked 테스트 14개가 228개 node 추가 |
| Q-P2-03 | P2 | 패키지 내부 버전 원장이 갈라져 있다 | 배포 0.2.10, Clarus Python 1.2.0, Clarus Cargo 1.1.0 |
| Q-P2-04 | P2 | prereg 기본 성공은 holdout 준비 완료가 아니다 | 기본 validator는 exit 0/NOT_READY; 강제 옵션은 exit 1/INVALID |

## P0 상세

### Q-P0-01 — 녹색 신호가 실행과 검증을 보장하지 않는다

`reality_stone/tests/test_rsulf_cuda.py:25-28`, `:66-72`, `:107-113`, `:158-164`, `:197-203`, `:329-335`는 CUDA가 없으면 `pytest.skip(...)`가 아니라 메시지를 출력하고 단순 `return`한다. 로컬 실행은 `8 passed`였지만 stdout상 CUDA 테스트 5개가 실제로는 실행되지 않았다. 더 나아가 CPU/CUDA consistency 경로 `reality_stone/tests/test_rsulf_cuda.py:145-156`은 차이를 출력할 뿐 허용 오차 assertion이 없다. 파일의 독립 실행 블록은 존재하지 않는 함수를 호출한다(`reality_stone/tests/test_rsulf_cuda.py:379-400`, 특히 `:388`).

동일한 false-green 패턴이 표준 검증 스크립트에도 있다.

- `tests/run_validation.py:342-364`: 종합 상태와 결과 dict를 반환할 뿐 non-PASS에 `SystemExit`을 내지 않는다. 실제로 `CAUTION`을 출력하고 exit 0이었다.
- `tests/bootstrap_solver.py:306-321`: `FAIL` 분기를 출력해도 프로세스 실패로 전환하지 않는다.
- `tests/scorecard.py:501-511`: 결과 JSON을 쓰고 scorecard를 반환하지만 gate exit code가 없다. 결과 파일은 루트 `*.json` ignore 규칙 때문에 추적되지 않는다.
- `examples/physics/proof_completion_attempt.py:157-176`: obstruction/candidate ledger를 출력해도 exit 0이다.
- 이 명령들은 `.codex/skills/ce-validate/SKILL.md:14-20`과 `.claude/skills/ce-validate/SKILL.md:14-20`의 baseline에 포함된다. 즉, 자동화가 `PASS`, `CAUTION`, `FAIL`, obstruction을 종료 코드만으로 구별할 수 없다.

재현:

```powershell
.venv\Scripts\python.exe -m pytest reality_stone/tests/test_rsulf_cuda.py -q -rA -p no:cacheprovider --basetemp "$env:TEMP\ce-quality-cuda"
# 8 passed; stdout에는 CUDA 미가용으로 5개 경로가 조기 return했다는 메시지

.venv\Scripts\python.exe tests/run_validation.py
# Overall Status: CAUTION, exit code 0
```

영향: GPU backend가 전혀 검증되지 않았어도 배지가 녹색이 될 수 있고, 수치 불일치·이론 obstruction도 성공 프로세스로 소비된다. 모든 검증 스크립트는 명시적 상태 enum과 실패 exit code를 가져야 하며, `skip`과 `pass`를 분리하고 핵심 수치에는 독립 oracle/허용 오차 assertion을 둬야 한다.

### Q-P0-02 — 동일 배포 좌표가 두 가지 네이티브 산출물을 뜻한다

루트 `pyproject.toml:2-3`은 `reality_stone==0.2.10`이고 `pyproject.toml:39-42`는 중첩 Clarus core를 `reality_stone.clarus._rust`로 빌드한다. 그런데 `reality_stone/pyproject.toml:6-7`도 똑같이 `reality_stone==0.2.10`이며 `reality_stone/pyproject.toml:21-24`는 다른 crate를 `reality_stone._rust`로 빌드한다. 작업 디렉터리에 따라 동일 wheel 이름·버전이 서로 다른 native surface를 의미한다.

루트에서 만든 실제 wheel은 다음을 포함했다.

```text
reality_stone/_rust.py
reality_stone/clarus/_rust.pyd
reality_stone/clarus/core/src/...
```

즉, 최상위 `_rust`는 Python fallback이고 Clarus만 native다. 현재 소스 환경에서도 `reality_stone._rust`는 fallback(`_has_rust_ext=False`)이고 `reality_stone.clarus._rust`는 `.pyd`였다. `reality_stone/README.md:47-53`은 두 extension path를 설명하지만, 동일 배포 이름·버전 충돌을 해소하지는 않는다.

재현:

```powershell
$env:CARGO_TARGET_DIR="$env:TEMP\ce-quality-wheel-target"
uv build --wheel --out-dir "$env:TEMP\ce-quality-wheel"
tar -tf "$env:TEMP\ce-quality-wheel\reality_stone-0.2.10-cp38-abi3-win_amd64.whl"

.venv\Scripts\python.exe -c "import reality_stone, reality_stone._rust as t, reality_stone.clarus._rust as c; print(reality_stone.__file__); print(t.__file__, getattr(t, '_has_rust_ext', None)); print(c.__file__)"
```

관측: wheel 빌드 성공(약 83초), 위 혼합 구성이 확인됐다. 배포 단위를 분리하거나 단일 정본 `pyproject.toml`만 유지하고, 설치한 wheel을 깨끗한 venv에서 두 API surface에 대해 smoke/integration 테스트해야 한다.

### Q-P0-03 — 비-RBE Rust 전체 suite가 현재 실패한다

`reality_stone/Cargo.toml`의 전체 테스트는 unit 20개, `layers.rs` 25개, `ops.rs` 12개를 통과한 뒤 `tests/rsulf.rs`의 대형 임계값 테스트 부근에서 Windows `STATUS_STACK_OVERFLOW`로 종료했다. 문제 테스트를 단독 실행하면 통과하고, `--test-threads=1`로 전체 `rsulf`를 직렬화해도 같은 위치에서 실패했다. 단순 race라기보다는 테스트 합성 시의 누적 스택/자원 민감성이다.

```powershell
$env:CARGO_TARGET_DIR="$env:TEMP\ce-quality-rs-target"
cargo test --locked --manifest-path reality_stone/Cargo.toml
# exit 1, 약 155.2초, process didn't exit successfully: STATUS_STACK_OVERFLOW

cargo test --locked --manifest-path reality_stone/Cargo.toml rsulf_폴딩_정확도_임계값_대형 -- --exact --nocapture
# 단독 실행은 통과(약 7.25초)

cargo test --locked --manifest-path reality_stone/Cargo.toml --test rsulf -- --test-threads=1
# 전체 rsulf 구성에서는 다시 STATUS_STACK_OVERFLOW
```

릴리스 gate에는 반드시 전체 binary 조합 실행을 포함해야 한다. 단독 test pass를 전체 suite pass로 대체하면 이 실패를 숨긴다.

## P1 상세

### Q-P1-01 — CI/품질 gate 부재

추적된 비-RBE 파일에서 GitHub Actions, GitLab CI, Azure Pipelines, CircleCI, Jenkinsfile, Makefile, tox/nox 설정을 찾지 못했다. 루트 `pyproject.toml`에도 `[project.scripts]`가 없다. 따라서 Python 버전/OS/native backend별 설치·test·lint·wheel smoke matrix가 저장소에서 정의되거나 강제되지 않는다.

재현:

```powershell
git ls-files | rg -i "(^|/)(\.github/workflows|\.gitlab-ci|azure-pipelines|\.circleci|Jenkinsfile|Makefile|tox\.ini|noxfile)"
rg -n "^\[project\.scripts\]" pyproject.toml reality_stone/pyproject.toml
# 일치 없음
```

### Q-P1-02 — clean clone에서 lock과 훅이 사라진다

`reality_stone/.gitignore:48-50`은 `uv.lock`, `Cargo.lock`, `target`을 무시한다. 로컬에는 `reality_stone/Cargo.lock`과 `reality_stone/python/reality_stone/clarus/core/Cargo.lock`이 있지만 둘 다 ignored/untracked다. 따라서 여기서 성공한 `cargo test --locked`는 clean clone에서 같은 방식으로 재현되지 않는다. 루트 `uv.lock`만 추적되고 중첩 Python 프로젝트의 lock은 없다.

또한 루트 `.gitignore:23`이 `.claude/` 전체를 무시해 `git ls-files .claude`는 비어 있다. `.claude/README.md:9`가 요구하는 `settings.json`도 clone에 오지 않는다. 루트 `.gitignore:31`의 `*.json` 때문에 `.codex/hooks.json` 역시 ignored/untracked인데 `.codex/README.md:14`는 이를 설치 파일로 설명한다. hook runner 자체도 binary가 없으면 조용히 성공한다(`.codex/hooks/run.cmd:10-11`, `.claude/hooks/run.cmd:9-10`).

```powershell
git check-ignore -v .claude/settings.json .codex/hooks.json reality_stone/Cargo.lock reality_stone/python/reality_stone/clarus/core/Cargo.lock
git ls-files .claude .codex/hooks.json reality_stone/Cargo.lock reality_stone/python/reality_stone/clarus/core/Cargo.lock
```

### Q-P1-03 — 기본 suite가 네트워크·캐시·성능 환경에 의존한다

`reality_stone/tests/llm/test_gpt2_manifold_learner.py:7-10`, `:42-45`는 marker나 opt-in 없이 `from_pretrained("gpt2")`를 호출하며 revision/hash도 고정하지 않는다. 기본 수집 대상인데 빈 오프라인 cache에서는 실패한다.

```powershell
$env:HF_HOME="$env:TEMP\ce-quality-empty-hf"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
uv run --extra dev python -m pytest reality_stone/tests/llm/test_gpt2_manifold_learner.py::test_gpt2_manifold_learner_collect_weights -q -p no:cacheprovider --basetemp "$env:TEMP\ce-quality-offline"
# FAILED, gpt2 config가 cache에 없고 outgoing traffic disabled, exit 1
```

`reality_stone/tests/llm/test_spd_performance.py:32-49`는 `fast_time < 1.0`이라는 절대 벽시계 임계값을 일반 테스트로 강제한다. 로컬 단독 실행은 통과했지만(`1 passed in 8.85s`) 공유 runner·저사양 호스트에서 기능 회귀 없이 실패할 수 있다. network/model/performance/native/GPU를 marker로 분리하고 기본 unit suite를 hermetic하게 만들어야 한다.

### Q-P1-04 — 분류·coverage 정책 부재와 전체 Python suite 실패

루트 `pyproject.toml:48-52`의 pytest 설정은 `testpaths`와 `cuda` marker 하나뿐이다. slow/network/integration/native/performance marker, timeout, warning strictness, addopts가 없고 `pyproject.toml:29`의 dev dependency에도 `pytest-cov`가 없으며 coverage 설정도 없다.

현재 수집은 2,604개(`tests/` 2,449, `reality_stone/tests/` 155)다. 조정된 전체 실행 결과는 **32 failed, 2,517 passed, 14 skipped, 41 errors, 2 warnings**였다. 즉 실패와 setup/collection error를 합쳐 73개의 비성공 node가 있다. 반면 문서상 핵심 subset은 58 passed, README Reality Stone subset은 46 passed/3 skipped였다. 선택 subset의 녹색은 default 2,604개 전체나 설치 wheel을 증명하지 않는다.

```powershell
uv run --extra dev python -m pytest --collect-only -q -p no:cacheprovider --basetemp "$env:TEMP\ce-quality-collect"
# 2604 tests collected

.\.venv\Scripts\python.exe -m pytest -q --tb=no --basetemp '_workspace\ce\repository-code-analysis-20260815\artifacts\pytest-basetemp-full'
# 32 failed, 2517 passed, 14 skipped, 2 warnings, 41 errors in 350.02s

.venv\Scripts\python.exe -m pytest tests/test_bootstrap_solver.py tests/test_dimensionless.py tests/test_layer_a.py tests/test_bridge_gates.py -q -p no:cacheprovider --basetemp "$env:TEMP\ce-quality-canonical"
# 58 passed, 2 warnings in 39.75s

.venv\Scripts\python.exe -m pytest -q reality_stone/tests/layer reality_stone/tests/test_unified_riemannian.py reality_stone/tests/llm/test_metric_attention.py reality_stone/tests/llm/test_metric_router.py reality_stone/tests/api/test_pipeline_api.py -p no:cacheprovider --basetemp "$env:TEMP\ce-quality-readme"
# 46 passed, 3 skipped in 17.49s
```

### Q-P1-05 — 난수 재현 정책 부재

Python 테스트 중 RNG 호출 파일은 69개이고 그중 28개는 파일 내부 seed 설정을 찾을 수 없었다. 전역 `conftest.py` seed fixture도 없다. 예를 들어 `reality_stone/tests/test_unified_riemannian.py:68`, `:77-78` 이후 다수 경로가 난수를 사용한다. Rust `reality_stone/tests/rsulf.rs:25-36`, `:385-386`, `:411-412`도 `rand::random`을 사용한다. 이 때문에 실패 입력을 재구성하기 어렵고, P0-03 같은 composition failure의 원인 분리가 더 어렵다.

고정 seed 하나만 강제하기보다 test별 seed를 로그하고, property/fuzz 테스트는 실패 seed를 보존하며, 통계 테스트는 반복 횟수·허용 오차·신뢰수준을 명시해야 한다.

### Q-P1-06 — oracle/assertion의 구조적 공백

- `reality_stone/tests/llm/test_gpt2_last_layer.py:57-74`: cosine similarity와 relative error를 계산하지만 둘 중 어느 것도 assert하지 않는다. 유한하지만 틀린 student output도 통과한다.
- `reality_stone/tests/test_llm_inference.py:11-59`: 기본은 opt-in skip이고 dev dependency에 없는 `accelerate`를 요구하며, 핵심 함수가 정확성 assertion 대신 dict를 반환한다.
- `reality_stone/tests/test_unified_riemannian.py:14-18`: 패키지 import 성공만으로 `HAS_RUST=True`를 둔다. 실제 `_has_rust_ext=False` fallback 환경에서도 “Rust” 테스트가 돈다.
- `tests/test_run_validation_consistency.py:10-32`는 adapter와 같은 `ConstantsScorecard()` 원천을 비교한다. wiring drift는 잡지만 잘못된 기준 상수 자체는 검출하지 못한다. `tests/run_validation.py:90-115`도 같은 scorecard를 감싼다.

외부 또는 독립적으로 고정된 golden/reference vector, backend identity assertion, mutation/negative control을 추가해야 한다. 기계적 일관성 통과를 이론 정확성으로 표기해서는 안 된다.

### Q-P1-07 — benchmark provenance와 변환 검증 공백

33개의 benchmark artifact가 추적되며 RAGTruth raw/converted와 FaithBench zip 같은 대형 데이터가 포함된다. 반면 로컬 학습 모델 JSON(`claim_span_detector_model.json`, `hashed_detector_model.json`, `supervised_residual_model.json`)과 CSV error output은 광범위한 `*.json`/`*.csv` ignore 규칙 때문에 추적되지 않는다.

`examples/pre_eq/convert_faithbench.py:35-55`는 입력 디렉터리가 틀리거나 비어 있어도 0개를 변환하고 성공하며, main `:58-73`은 `records: 0`을 출력한 뒤 exit 0이다. `tests/test_faithbench_conversion.py:7-17`은 label policy helper만 검사하고 end-to-end file count, schema, content hash를 확인하지 않는다. RAGTruth 변환에도 입력/출력 manifest·hash를 고정하는 전체 회귀를 찾지 못했다.

원본 출처/버전/라이선스, 입력 hash, 변환기 commit, 출력 row count/schema/hash, 학습 seed와 hyperparameter를 하나의 추적 manifest로 묶고 빈 변환을 실패 처리해야 한다.

### Q-P1-08 — 실험 데이터가 현재 작업 트리에 종속되고 기록이 모순된다

`experiments/recursion_probe.py:39-51`과 `experiments/ood_length_repro.py:40-52`는 실행 시점 저장소의 `.py` 파일을 corpus로 사용하고 read error를 조용히 삼킨다. 파일 목록·content hash·git tree/dirty 상태를 기록하지 않아 소스 수정과 untracked 파일이 결과를 바꾼다. stdout 외에 환경·설정·데이터 digest가 묶인 산출물도 없다. recursion/OOD/aliasing experiment를 실행하는 회귀 테스트는 찾지 못했고 prereg validator만 테스트된다.

`experiments/RESULTS_ood_length.md:24-37`은 3 seeds/full 32x와 thesis supported를 주장하지만 같은 문서 `:53-56`은 아직 single seed/16x이고 exact corpus가 없다고 적는다. 결론과 caveat가 동시에 참일 수 없어 상태가 stale하다.

### Q-P1-09 — 연구 코어와 agent 계약이 완결성을 강제하지 못한다

`.codex/skills/ce-research/core/src/main.rs`의 현재 gate는 의미 검증보다 문자열·파일 존재 검증에 가깝다.

- `:228-277`, `:320-346`: 파일 어디에든 `Status: COMPLETE`가 있으면 승인하며 gate verdict는 prefix라 `PASSING`도 `PASS`로 취급될 수 있다. `SKIPPED`는 이유를 요구하지 않는다.
- `:281-300`: stray-root hygiene가 `.md`만 막아 txt/json 등 추가 산출물은 허용한다.
- `:351-364`: `revise` role 입력을 정규화/허용 목록 검증하지 않아 별칭·대소문자로 counter를 우회할 수 있다.
- `:369-396`: `gc`는 전체 check 없이 final 상태 하나만 보고 archive한다. 단위 테스트 `:478-490`도 final 파일만 있는 불완전 run archive를 성공 동작으로 고정한다.
- `:493-505`: unfinished run의 첫 stop만 경고하고 두 번째 stop은 허용하는 동작을 테스트가 명시적으로 승인한다.

core 자체 Rust 테스트 7개는 통과했지만, 이는 위의 약한 규칙을 구현대로 만족했다는 뜻이지 연구 완결성의 증명이 아니다.

Agent 카드도 실행 계약과 충돌한다. `.claude/agents/ce-status-auditor.md:3-4`는 Glob/Grep/Read만 허용하면서 `:41-49`는 `20-audit.md` 작성을 요구한다. `.claude/agents/ce-math-verifier.md:3-10`도 write 금지를 명시하지만 `:33`, `:39-49`, `:57`은 `11-math.md`와 artifact 작성을 요구한다. physics sourcer와 route explorer도 같은 형태다. 그런데 `.claude/skills/ce-research/SKILL.md:8`은 이 agent들에게 lane을 직접 위임한다. 별도의 부모 writeback 계약이 없으면 필수 stage 파일을 만들 수 없다.

정책도 서로 다르다. `.codex/AGENTS.md:35-36`은 열린 P1이면 revise 또는 BLOCKED여야 한다고 하지만 `.codex/agents/ce-status-auditor.md:14-18`은 open P0만 없고 상태가 일관되면 PASS를 허용한다. 따라서 P1 proof gap이 있어도 구현을 unlock할 수 있다.

## P2 상세

### Q-P2-01 — lint 부채와 깨진 독립 실행 경로

```powershell
.venv\Scripts\ruff.exe check tests reality_stone/python reality_stone/tests examples experiments --statistics --exclude RBE
# exit 1, 242 errors:
# 104 F401, 56 E702, 27 E402, 23 F841, 10 F541, 8 E741,
# 6 E701, 3 E731, 2 F405, 1 E401, 1 F403, 1 F821
```

F821은 `reality_stone/tests/test_rsulf_cuda.py:388`의 미정의 `test_rsulf_inference_pipeline`이다. pytest collection은 `__main__` 블록을 실행하지 않아 숨겨지지만 파일을 script로 실행하면 마지막 “All tests passed”에 도달할 수 없다. lint를 gate로 만들되 generated/data 파일을 명시적으로 분리하고 신규 오류 0 정책부터 적용하는 편이 현실적이다.

### Q-P2-02 — 수집 수가 untracked 테스트에 의존

현재 untracked `tests/test*.py` 14개가 228개의 node를 추가한다. 예를 들어 `tests/test_delayed_linear_credit.py` 68개, `tests/test_homogeneous_signed_cue.py` 26개, `tests/test_covariant_metric_flow.py` 19개와 L3-L8/v16-v18b 계열이 포함된다. 따라서 현재의 2,604 수집 결과는 clean clone baseline이 아니다. 감사 결과는 이 dirty-tree 현실을 보존해 측정했지만, CI 기준 수는 반드시 tracked tree에서 별도 기록해야 한다.

### Q-P2-03 — 버전 원장 분리

루트와 중첩 배포는 0.2.10이지만 `reality_stone/python/reality_stone/clarus/__init__.py:7`은 1.2.0, `reality_stone/python/reality_stone/clarus/core/Cargo.toml:3`은 1.1.0이다. Q-P0-02의 빌드 정체성 문제와 결합하면 bug report 및 산출물 provenance가 모호해진다. 단일 version source와 build-time consistency test가 필요하다.

### Q-P2-04 — prereg `VALID`와 evaluation readiness 혼동 가능

`experiments/preregistration/validate_holdout_manifest.py:994-1038`의 기본 검사는 미래 holdout이 미배정이어도 manifest 구조를 `VALID`로 보고 별도로 `NOT_READY`를 출력한다. 강제 옵션에서는 올바르게 실패한다.

```powershell
.venv\Scripts\python.exe -m experiments.preregistration.validate_holdout_manifest
# exit 0; v2 manifests VALID, holdout unassigned, evaluation NOT_READY

.venv\Scripts\python.exe -m experiments.preregistration.validate_holdout_manifest --require-assigned-holdout
# exit 1; 두 manifest INVALID, future holdout not assigned
```

이는 validator 버그라기보다 호출자가 기본 성공을 실험 준비 완료로 해석할 위험이다. 평가/발표 gate에서는 반드시 강제 옵션을 사용하고 readiness를 별도 machine-readable 상태로 내보내야 한다.

## 실행 결과 원장

아래는 RBE를 전혀 호출하지 않은 실제 결과다.

| 검증 | 명령 요약 | 결과 | 해석 한계 |
|---|---|---|---|
| Python 수집 | `uv run --extra dev python -m pytest --collect-only -q` | 2,604 collected | untracked 테스트 228개 포함 |
| Python 전체 | `.\.venv\Scripts\python.exe -m pytest -q --tb=no --basetemp '_workspace\ce\repository-code-analysis-20260815\artifacts\pytest-basetemp-full'` | 32 failed, 2,517 passed, 14 skipped, 2 warnings, 41 errors in 350.02s | 73개 비성공 node; 선택 subset 녹색과 불일치 |
| 문서상 CE 핵심 subset | bootstrap/dimensionless/layer_a/bridge_gates | 58 passed, 2 warnings, 39.75s | 전체 suite·이론 증명 아님 |
| README Reality Stone subset | layer/unified/metric/API | 46 passed, 3 skipped, 17.49s | source tree/fallback 중심, 설치 wheel 아님 |
| validation script | `python tests/run_validation.py` | 11 PASS, 1 CAUTION(12 scored), exit 0 | non-PASS gate 아님 |
| root wheel | `uv build --wheel` | 성공, cp38-abi3 wheel, 약 83s | `_rust.py` fallback + `clarus/_rust.pyd` 혼합 |
| Clarus nested Cargo | `cargo test --locked --manifest-path .../clarus/core/Cargo.toml` | 32 passed, 0 failed, 0.49s | ignored local lock에 의존 |
| Reality Stone Cargo 전체 | `cargo test --locked --manifest-path reality_stone/Cargo.toml` | exit 1, stack overflow, 약 155.2s | 현재 릴리스 blocker |
| ce-research core Cargo | `cargo test --locked --manifest-path .codex/skills/ce-research/core/Cargo.toml` | 7 passed | 약한 semantics 자체를 승인 |
| GPT-2 빈 오프라인 cache | default-collected singleton | failed, exit 1, 8.69s | 기본 suite 비-hermetic 증거 |
| CUDA pytest | `test_rsulf_cuda.py` | 8 passed, 6.68s | 5개 CUDA 경로는 조기 return, skip 집계 아님 |
| performance singleton | `test_spd_fast_mixing_performance` | 1 passed, 8.85s | 절대 1초 임계값은 환경 민감 |
| Ruff | non-RBE Python 범위 | 242 errors, exit 1 | CI gate 없음 |
| prereg 기본/강제 | validator 두 모드 | exit 0 NOT_READY / exit 1 INVALID | readiness와 schema validity 분리 필요 |

전체 Python 2,604개 실행은 실패했다. 다만 이 숫자에는 현재 untracked 테스트 228개와 네트워크·모델 cache·환경 의존 테스트가 함께 들어가므로, clean-clone code regression과 환경/setup failure는 CI에서 marker와 failure taxonomy로 분리해야 한다. 이 분류 문제를 감안해도 73개 비성공 node와 구조적 false-green은 현재 상태를 release-ready로 판정할 수 없게 한다.

## 권고되는 차단 순서

1. **즉시 release 차단:** Q-P0-03 전체 Rust suite를 안정적으로 통과시키고, CUDA 조기 `return`을 실제 `pytest.skip`으로 바꾸며 CPU/CUDA 수치 assertion을 추가한다.
2. **산출물 정체성 단일화:** 두 `pyproject.toml`의 동일 name/version 충돌을 제거하고, clean venv에 wheel을 설치한 뒤 backend identity와 두 API surface를 검사한다.
3. **실패를 exit code로 전달:** validation/scorecard/bootstrap/proof scripts가 machine-readable verdict와 nonzero exit를 내도록 한다. `CAUTION` 허용 여부는 호출자가 명시한다.
4. **clean-clone CI 생성:** tracked lock만 사용해 lint, hermetic unit, integration/network opt-in, native Rust 전체, wheel install smoke를 OS/Python matrix로 실행한다.
5. **테스트 분류와 oracle 강화:** slow/network/model/performance/GPU/native marker, 고정된 reference vectors, backend assertion, negative/mutation controls, seed 기록, coverage floor를 둔다.
6. **연구 provenance 닫기:** benchmark 및 실험마다 input/content hash, corpus manifest, commit+dirty 상태, seed/config/environment, output hash를 기록하고 empty conversion을 실패 처리한다.
7. **agent/gate 계약 정합화:** 쓰기 금지 agent는 결과를 구조화해 부모가 stage file로 materialize하도록 명시하거나 제한된 write 권한을 부여한다. P1 정책을 하나로 통일하고 research core가 full check 없이 archive/stop하지 못하게 한다.

이 조치를 마치기 전에는 subset `passed`나 검증 스크립트 exit 0을 “이론 검증 완료”, “GPU 검증 완료”, “릴리스 준비 완료”로 표현하면 안 된다.
