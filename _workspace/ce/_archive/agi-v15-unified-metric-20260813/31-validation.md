# 31-validation — AGI V15 Unified Metric Agent

Status: COMPLETE

## 1. 구현·회귀 검사

| 범위 | 명령 요약 | 원래 결과 |
|---|---|---:|
| V15 focused | `pytest tests/test_unified_metric.py` | `17 passed in 3.68s` |
| dimensionless + Clarus-field + V15 | 세 test 파일 | `46 passed in 3.13s` |
| SCC compatibility | `test_scc_atlas`, `test_nested_scc_tower`, V15 | `114 passed in 3.79s` |
| runtime/public compatibility | `test_runtime_contracts`, `test_reality_bridge`, V15 | `34 passed, 2 warnings in 3.31s` |
| CE core slice | bootstrap/dimensionless/layer-A/bridge/Clarus/V15 | `89 passed, 2 warnings in 4.49s` |
| local-cloud compatibility | V10/small-gain/V13/Clarus/V15 | `89 passed in 8.81s` |
| geometry compatibility | manifold diffusion/Riemann PE/MRA/V15 | `40 passed, 1 skipped in 3.48s` |
| changed implementation style | `ruff check` on new implementation/test/example/artifact and dimensionless files | `All checks passed!` |
| compile | `python -m compileall -q` on new module/example/artifact | exit 0 |
| patch whitespace | `git diff --check` on V15 scope | exit 0; 기존 CRLF 전환 경고만 있음 |

runtime과 CE core의 두 경고는 기존 `runtime.py`의 PyTorch sparse invariant/beta 경고다. broad lint 첫 시도에 기존 `clarus/__init__.py:192-199`의 `ce_ops` alias 8개가 F401로 보고됐다. 이 행들은 V15 import와 무관한 기존 optional export이며 이번 승인 범위 밖이라 수정하지 않았다. 새 구현 및 직접 수정한 계산 파일을 분리해 실행한 ruff 결과는 모두 통과했다.

## 2. 독립 affine·barrier 계량

고정 seed 150013에서 SPD metric과 condition이 제한된 비직교 affine Jacobian 128개를 생성했다. 각 trial은 point, tensor와 고정 adjacency를 함께 운반했다. `artifacts/validate_unified_metric.py` 결과는 다음과 같다.

| 측정량 | 결과 | 사전 허용치 |
|---|---:|---:|
| local quadratic 최대 상대오차 | $6.344\times10^{-16}$ | $10^{-10}$ |
| finite edge 최대 상대오차 | $6.301\times10^{-16}$ | $10^{-10}$ |
| shortest cost 최대 상대오차 | $4.778\times10^{-16}$ | $10^{-10}$ |

identity diamond graph의 goal minimizer는 $(1,2)$이고 path는 non-unique다. node 1에 $4I$ barrier metric을 두면 goal minimizer는 $(2)$, plan은 $(0,2,3)$, cost는 $2.8284271247461903$이다. 같은 metric deformation이 edge cost, memory delta, plan, critic gate와 goal readout을 함께 바꾸는 focused test도 통과했다.

projection no-go fixture에서는 coordinate transport 직후 length squared가 $1.0000000000000002$였지만 default fixed-chart clipping 뒤 25가 됐다. covariance defect 24는 `projection_affine_covariant=False`를 유지해야 한다는 음성 대조다.

bounded source fixture의 고유값 범위는 $[0.875,1.25]$, condition number는 $1.4285714286$으로 configured $[0.5,2]$와 bound 4 안이다. persistent state는 `metric_only`, 역할 parameter 수는 0으로 기록됐다.

## 3. 수학 반례 재현

`artifacts/verify_unified_metric_math.py`는 exit 0으로 다음을 재현했다.

- affine quadratic length 최대 상대오차 $2.339\times10^{-14}$,
- spectral clipping covariance defect 9인 별도 fixture,
- 완비 $\mathbb R^d$의 bounded non-$L^2$ source 반례,
- $\operatorname{tr}_g\dot g>4\lambda$인 time-varying metric energy 증가,
- fixed-point 없는 reflection isometry의 singleton goal no-go,
- 같은 finite endpoint metric에서 continuum length 2.0000 대 1.02936 비식별.

이 수치들은 `11-math.md` 정리의 기계적 예시와 반례 회귀이며 증명의 대체물이 아니다.

## 4. 결정론적 예시

`examples/agi/unified_metric_demo.py`는 exit 0으로 identity goal $(1,2)$, deformed goal $(2)$, path $(0,2,3)$, condition number 4를 출력했다. 함께 출력된 `full_geodesic_verified`, `continuum_limit_verified`, `agi_evidence`는 모두 false다.

## 5. CE 하네스

- bootstrap solver: exit 0, $\varepsilon^2=0.0486466333$, residual $2.08\times10^{-17}$.
- scorecard: 23 total, 12 scored, 11 PASS, 1 CAUTION, aggregate `CAUTION`; 최대 주의는 $\Omega_bh^2=-1.80\sigma$.
- `tests/run_validation.py`: bootstrap PASS, scorecard CAUTION, dimension 7/7, overall `CAUTION`.
- `proof_completion_attempt.py`: exit 0; raw $V_{cb}$, tree $V_{us}$, raw $A_s$ obstruction도 원문 그대로 유지.

이 하네스는 V15 task utility, 물리 관측 또는 AGI 검증이 아니다. 저장소 전체 pytest는 기존 dirty fixture/policy 문제 때문에 이번 범위에서 재실행하지 않았다.

Status: COMPLETE
