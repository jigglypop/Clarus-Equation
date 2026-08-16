# Phase A implementation preflight

Status: COMPLETE

Date: 2026-08-16 (Asia/Seoul)

Scope: Phase A 계약의 새 격리 표면 네 개를 구현하기 전, 기존 benchmark·preregistration·runner·test 관례, Python 의존성, import 경계, Git 충돌과 최소 회귀 명령을 읽기 전용으로 조사했다. 이 문서는 구현 승인이 아니며 제품 코드·테스트·정본·stage 파일을 수정하지 않았다.

## 1. 결론

구현은 가능하며 정확한 네 대상 파일은 모두 현재 존재하지 않아 파일명 충돌은 없다. 다만 다음 두 경계는 구현 전에 고정해야 한다.

1. 새 production module은 `__init__.py`에 export하지 않는다. 해당 파일은 이미 사용자 변경이 있고 계약도 수정을 금지한다(`00-contract.md:81-86`). test와 runner는 새 파일의 immutable bytes를 직접 load하는 격리 패턴을 사용한다.
2. `experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`은 현재 `.gitignore:33`의 `*.json`에 걸리며 정확한 예외가 없다. `.gitignore`도 이미 dirty이므로 구현 agent가 묵시적으로 수정해서는 안 된다. 파일을 로컬에서 생성·검증하는 것은 가능하지만 repository 산출물로 추적하려면 별도 승인으로 정확한 한 줄 예외를 추가하거나 commit 단계에서 그 파일만 명시적으로 force-add해야 한다.

SciPy는 필요하지 않다. NumPy만으로 generator, joint least squares, rank certificate, NLL, paired bootstrap을 구현할 수 있으며 이것이 기존 synthetic bridge의 `numpy_only` 경계와도 일치한다(`sparse_causal_bridge.py:969-978`).

## 2. 승인된 대상과 충돌 조사

계약의 승인 후보는 다음 네 파일이다(`00-contract.md:77-86`). 2026-08-16 preflight 시점의 `Test-Path` 결과는 모두 `False`였다.

| 대상 | 존재/충돌 | 판정 |
|---|---:|---|
| `reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py` | 없음 | 신규 생성 가능 |
| `tests/test_causal_recurrent_geometry_benchmark.py` | 없음 | 신규 생성 가능 |
| `experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json` | 없음 | 이름 충돌 없음; Git ignore 문제 있음 |
| `examples/agi/causal_recurrent_geometry_development_run.py` | 없음 | 신규 생성 가능 |

정확한 대상에는 dirty overlap이 없지만 인접 표면은 깨끗하지 않다.

- `reality_stone/python/reality_stone/clarus/__init__.py`는 tracked dirty이며 V16/V17/V18b 및 다른 현재 작업의 export 변경을 포함한다. 새 export를 추가하지 않는다.
- `.gitignore`는 tracked dirty이며 CE target 및 cosmology JSON 예외 변경을 포함한다. 새 JSON 예외를 같은 구현 patch에 섞지 않는다.
- `tests/`와 `clarus/`에는 다수의 unrelated dirty/untracked 파일이 있다. 새 파일 네 개 외의 formatting·cleanup·bulk rewrite를 하지 않는다.
- clean reusable references인 `sparse_causal_bridge.py`, `test_sparse_causal_bridge.py`, `nested_scc_memory_benchmark.py`, `test_nested_scc_memory_benchmark.py`, `local_cloud_development_run.py`, `sparse_causal_bridge_v8.json`에는 scoped `git status` 변경이 없었다.
- predecessor가 금지한 dirty V15, untracked V16/V17, V9 STOP candidate와 infinite-tail 코어는 import하지 않는다(`repository-reuse-map.md:225-247`).

실행한 ignore 판정은 다음과 같다.

```text
.gitignore:33:*.json experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
```

권고 P0는 다음 중 하나를 명시적으로 선택하는 것이다.

- repository 추적이 필요하면 별도 좁은 patch로 `.gitignore`에 `!experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`을 추가한다. 현재 사용자 변경과 겹치므로 승인·diff 검토가 필요하다.
- 현재 연구 run의 로컬 실행만 수행한다면 JSON을 생성하되 최종 보고에 ignored 상태를 명시한다. 추후 commit에서 `git add -f -- <exact-path>`를 사용할 수 있으나 자동 실행하지 않는다.

## 3. 의존성과 packaging

루트 package는 Python 3.10 이상을 요구하고 NumPy를 core dependency로 둔다(`pyproject.toml:5-12`). SciPy는 `science` optional extra다(`pyproject.toml:14-22`). 별도 `reality_stone/pyproject.toml`도 NumPy를 core dependency로 둔다(`reality_stone/pyproject.toml:5-14`). 현재 `.venv`에서 읽기 전용 import probe는 Python `3.14.4`, NumPy `2.3.5`, SciPy `1.18.0`으로 성공했다.

권고 구현 의존성은 다음뿐이다.

```python
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
```

- fitting: `np.linalg.lstsq`; rank certificate: singular values 또는 `np.linalg.matrix_rank`;
- stability scaling: `np.linalg.eigvals`;
- Gaussian NLL, coefficient/support error, bootstrap: NumPy;
- canonical JSON과 SHA-256: 표준 라이브러리.

SciPy를 사용하지 않으면 optional extra가 없는 환경에서도 benchmark가 실행되고, lock manifest에 dependency surface가 늘어나지 않는다. RNG는 legacy global state가 아니라 `np.random.Generator`만 사용한다.

## 4. `__init__.py` 수정 없는 import

### 4.1 권고 방식: byte-hash isolated load

V18b runner는 production file을 직접 읽고 SHA-256을 계산한 뒤 `importlib.util.spec_from_file_location`으로 module object를 만들고, 해시한 동일 byte buffer를 `compile`/`exec`한다(`run_v18b_benchmark.py:83-106`). 이 방식은 package initializer를 실행하지 않으며 source loader가 경로를 다시 읽는 TOCTOU도 피한다(`:98-105`). test는 module path와 private sealed module name까지 확인한다(`tests/test_v18b_benchmark.py:89-96`).

Phase A에서도 runner와 test 공용으로 다음 패턴을 사용한다.

```python
def _isolated_load(path: Path):
    resolved = path.resolve(strict=True)
    source = resolved.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = f"_ce_phase_a_crg_{digest}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, str(resolved), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, digest
```

production module은 self-contained NumPy module로 유지하고 package-relative import를 두지 않는다. runner는 `reality_stone.clarus`나 top-level `reality_stone`을 import하지 않는다. top-level initializer는 Torch와 여러 layer를 즉시 import하므로(`reality_stone/python/reality_stone/__init__.py:1-78`) 격리 benchmark에는 불필요하게 넓다.

### 4.2 일반 library import

개발 편의상 `from reality_stone.clarus.causal_recurrent_geometry_benchmark import ...`은 `__init__.__all__` export 없이도 submodule import 문법상 가능하다. 기존 sparse bridge test도 이 직접 submodule 방식을 쓴다(`tests/test_sparse_causal_bridge.py:10-25`). 그러나 부모 package initializer는 실행되므로 dirty initializer와 설치 상태의 영향을 받는다. hash-bound runner와 integrity test의 정본 경로로는 사용하지 않는다.

## 5. 재사용할 benchmark 패턴

### 5.1 generator와 RNG 분리

clean sparse bridge는 simulation stream을 `SeedSequence.spawn`으로 initial state, latent, process noise에 분리한다(`sparse_causal_bridge.py:177-217`). paired intervention은 order, process, sensor stream을 다시 분리하고 양 arm에 동일 process noise를 주어 nuisance를 상쇄한다(`:220-284`). intervention sign permutation negative control도 별도 구현되어 있다(`:297-310`).

Phase A는 이 원칙만 가져오고 기존 nonlinear SCM이나 V4 점수는 재사용하지 않는다. manifest에는 숫자 범위만 두지 말고 다음 역할을 별도 field로 둔다.

- `pilot_graph_seeds`: sanity only, scored aggregate에서 제외;
- `development_graph_seeds`: model/debug 선택 가능;
- `confirmation_graph_seeds`: `reserved_unopened`, 이번 runner가 소비하지 않음;
- graph seed별 namespace: `graph`, `train_trajectory`, `heldout_intervention`, `evaluation_noise`, `shuffle`, `bootstrap`.

단순 `seed + offset`보다 V18b의 SHA-256 namespace 파생(`run_v18b_benchmark.py:329-358`)을 축약 재사용한다. payload는 schema tag, graph seed, role ASCII string, coordinates를 고정 byte order로 serialize한다. 같은 namespace replay는 동일하고 서로 다른 role digest는 달라야 한다.

### 5.2 관측과 개입 경계

production API는 truth matrix를 estimator에 전달하지 않도록 자료형을 분리한다.

- generator-owned `GroundTruth`: $A_z$, $B$, observation type와 support;
- learner-visible `TransitionBatch`: `x`, `u`, `context`, `x_next`만;
- evaluator-owned `HeldOutBatch`와 truth scoring;
- `IdentificationCertificate`: observation type, design rank, declared model class, exact-edge allowed 여부.

known mask와 unknown mix는 prediction fixture를 만들 수 있어도 `exact_edge_allowed=False`로 고정한다. rank deficiency도 같은 certificate 경로에서 fail closed해야 한다. sparse bridge가 effective intervention과 observational selection을 분리하는 방식(`sparse_causal_bridge.py:706-787`)과 truth를 report의 evaluation-only field에 두는 방식(`:979-1007`)이 참고 패턴이다.

### 5.3 estimator 설계와 rank 조건

공유 $B$와 문맥별 $A_z$는 target coordinate별로 하나의 joint design을 만든다. row $t$의 feature는 해당 context의 $n$차원 block에만 $x_t$를 놓고 마지막 $m$차원 block에 $u_t$를 놓는다. 전체 design shape은 `N x (K*n + m)`이며 exact certificate는 이 joint design이 full column rank일 때만 true다. context별 `[X_z, U_z]` rank를 따로 검사하는 것만으로는 공유 $B$의 joint 유일성을 충분히 보장하지 않는다.

- factorized dof: $K n^2 + nm$;
- pooled dof: $n^2 + nm$;
- 공개 차이: $(K-1)n^2$.

noiseless fixture는 `np.linalg.lstsq`의 residual이 아니라 반환 rank와 최대 coefficient error를 모두 검사한다. ridge가 0보다 크면 exact recovery theorem fixture와 development estimator를 구분한다.

primary Gaussian NLL의 noise scale은 evaluator-owned manifest의 positive generator sigma를 양 arm에 동일하게 적용하는 것이 가장 좁다. 이를 estimator 입력에 넘기지 않는다. 대안인 train-residual sigma를 쓰려면 양 arm 공통의 train-only 추정량을 별도 등록해야 하며 모델별 sigma를 쓰면 parameter/likelihood 회계가 달라진다. 최종 선택은 math/audit gate의 결정을 따른다.

### 5.4 matched baseline과 결과 구조

factorized와 pooled arm은 동일 train batch, held-out batch, ridge, evaluation sigma를 받는다. 결과는 seed별 `nll_factorized`, `nll_pooled`, `delta_nll`, design rank, coefficient error, finite flag를 보존하고 aggregate에서 mean, median, paired bootstrap interval을 계산한다. frame을 bootstrap unit으로 쓰지 않는다.

기존 sparse bridge report는 registration hash, selection, evaluation-only truth, negative controls, checks와 resource usage를 분리한다(`sparse_causal_bridge.py:979-1040`). Phase A 결과도 다음 typed 영역을 유지한다.

```text
schema / mode / manifest_sha256 / implementation_sha256
seed_roles / protocol / dof_accounting
per_graph_seed
aggregate / primary_gate / integrity_gates
claim_certificate / exclusions
```

## 6. manifest hash와 confirmation sealing

### 6.1 canonical manifest

기존 holdout validator는 top-level `manifest_sha256`을 제거한 canonical UTF-8 JSON을 SHA-256으로 묶어 self-digest 순환을 피한다(`experiments/preregistration/validate_holdout_manifest.py:1-5,86-100`). Phase A 전용 runner가 같은 정책을 작게 구현하되 기존 cosmology/quantum validator schema에 억지로 연결하지 않는다.

manifest의 최소 고정 field는 다음과 같다.

- schema/version/status와 `manifest_hash_policy`;
- exact contract, production, test, runner relative path와 SHA-256;
- dimensions, contexts, steps, noise, ridge, intervention distribution;
- pilot/development/confirmation seed roles와 non-overlap rule;
- primary endpoint, bootstrap seed/sample count, STOP rule;
- dof formula와 allowed claim boundary;
- confirmation `status: reserved_unopened`, `execution_authorized: false`.

path는 POSIX relative canonical form만 받고 absolute path, `..`, backslash, duplicate JSON key를 거부한다. V18b는 exact required-path set과 repository-relative path를 검증한다(`run_v18b_benchmark.py:147-180` 및 `tests/test_v18b_benchmark.py:243-266`).

### 6.2 development one-shot

`local_cloud_development_run.py`는 repository root, 등록 hash, seed 역할 비중첩을 확인하고(`:28-46`), 기존 result가 있으면 거부한 뒤 temporary file과 `os.replace`로 결과를 원자적으로 기록한다(`:47-66`). Phase A development runner도 같은 정책을 사용한다.

- CLI는 manifest 하나만 받고 `mode=development`만 허용한다.
- manifest와 required artifact hash를 점수 계산 전에 검증한다.
- output이 이미 있으면 rerun을 거부한다.
- result는 `allow_nan=False`, sorted keys, UTF-8 newline로 직렬화한다.
- manifest hash, loaded production byte hash, Python/NumPy version을 기록한다.

### 6.3 confirmation은 실제로 봉인하고 열지 않음

이번 run에서는 confirmation 함수나 CLI mode를 제공하지 않는 것이 가장 강한 봉인이다. reserved seed 숫자는 manifest metadata로만 존재하고 production generator에 전달하는 public path가 없어야 한다. test는 다음을 확인한다.

1. development seed와 confirmation seed 집합이 disjoint;
2. runner CLI에 confirmation 선택지가 없음;
3. development 평가 함수가 정확한 development block 외 범위를 거부;
4. result에 `confirmation_status: reserved_unopened`가 남음;
5. confirmation 범위의 첫 seed를 전달해도 seed namespace 생성 전에 거부.

후속 별도 run에서 confirmation을 열 때는 receipt를 seed 접근보다 먼저 exclusive-create하는 V18b 패턴(`run_v18b_benchmark.py:1658-1701`), 실행 뒤 manifest와 module을 재해시하고 result를 exclusive-create하는 closing pattern(`:1704-1740`)을 사용한다. V9/V16/V17/V18b의 과거 confirmation seed나 result는 재사용하지 않는다.

## 7. 필수 test matrix

새 test 하나에 적어도 다음 fixture를 둔다.

| 분류 | fixture |
|---|---|
| exact positive | noiseless known-identity joint full-rank에서 최대 $A_z,B$ 오차 `<=1e-10` |
| rank refusal | duplicated/zero excitation으로 joint design rank 부족, exact certificate false |
| observation refusal | known mask와 unknown mix에서 exact-edge false |
| PA-N1 no-go | similarity-related 두 LTI의 latent support가 다르지만 관측열이 동일 |
| determinism | 같은 manifest·seed의 canonical result bytes 동일 |
| seed isolation | graph/trajectory/intervention/shuffle namespace가 모두 다르고 confirmation 접근 거부 |
| no hidden/future | learner signature에 truth/test target이 없고 test target mutation이 fitted coefficient를 바꾸지 않음 |
| orientation | `A[target, source]` convention exact fixture |
| primary comparison | pooled/factorized 동일 batch·ridge·sigma, seed-level delta 보존 |
| integrity negative | intervention tag/time shuffle이 intact보다 악화되지 않으면 PA-H2 STOP |
| domain | NaN, infinity, nonpositive sigma, 빈 context, shape mismatch 거부 |
| sealing | manifest tamper/path traversal/duplicate key/source hash mismatch/result overwrite 거부 |
| exclusion | SCC, memory, biology, consciousness, AGI certificate는 false/absent |

기존 test는 direct matrix orientation과 paired intervention을 검사한다(`tests/test_sparse_causal_bridge.py:36-99`). V18b test는 seed role 결정성·분리를 검사한다(`tests/test_v18b_benchmark.py:111-120` 이후), manifest 누락·traversal과 receipt ordering을 별도로 공격한다(`tests/test_v16_benchmark.py:95-181`; `tests/test_v18b_benchmark.py:243-320`). 구조만 재사용하고 과거 seed나 판정을 복제하지 않는다.

## 8. 최소 회귀 명령

현재 `uv`는 sandbox 밖 기본 cache 접근에서 권한 오류가 났으므로 이 checkout의 `.venv` executable을 직접 쓰는 것이 재현 가능한 최소 경로다. 구현 후 순서는 다음과 같다.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_causal_recurrent_geometry_benchmark.py -q
.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py tests/test_latent_causal_bridge.py tests/test_nested_scc_memory_benchmark.py -q
.venv\Scripts\ruff.exe check reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
.venv\Scripts\python.exe -m compileall -q reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
git diff --check -- reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json examples/agi/causal_recurrent_geometry_development_run.py
git check-ignore -v --no-index experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
```

그 뒤 development one-shot은 audit가 고정한 정확한 CLI로 한 번만 실행한다. confirmation 명령은 이 run에 존재해서는 안 된다. 전체 suite는 dirty worktree의 unrelated 변경과 실패를 분리하기 위해 focused 회귀가 통과한 뒤 별도 최종 검증으로 실행한다.

## 9. 구현 handoff

구현 agent가 지켜야 할 순서는 다음과 같다.

1. math/audit gate에서 joint rank 조건, NLL sigma와 implementation scope를 승인받는다.
2. self-contained NumPy production module과 isolated-load test를 먼저 작성한다.
3. development/pilot/confirmation 역할과 required artifact hash를 가진 manifest를 작성한다.
4. development-only one-shot runner를 작성하고 confirmation code path가 없음을 test한다.
5. focused test·legacy causal regression·Ruff·compile·diff check를 실행한다.
6. ignored JSON 처리 결정을 별도 기록한다. `.gitignore`나 `__init__.py`를 조용히 수정하지 않는다.

이 경계에서는 새 Phase A benchmark가 기존 사용자 변경, V9 STOP, dirty metric 계열과 분리된 채 구현될 수 있다.
