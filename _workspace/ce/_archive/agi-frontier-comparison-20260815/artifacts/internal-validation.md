# CE-AGI internal validation log

Snapshot: 2026-08-15 current working tree. `RBE/` was excluded from every
command, import, input, and conclusion.

## Focused executable surface

Command family:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='reality_stone/python'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp '_workspace\ce\agi-frontier-comparison-20260815\artifacts\pytest-tmp' `
  tests/test_runtime_contracts.py tests/test_agent.py tests/test_consciousness.py `
  tests/test_sleep.py tests/test_stdp.py tests/test_episodic_memory.py `
  tests/test_local_memory.py tests/test_local_memory_verifier.py `
  tests/test_nested_scc_tower.py tests/test_nested_scc_memory_benchmark.py `
  tests/test_adaptive_scc_tower_controller.py tests/test_local_cloud_kernel.py `
  tests/test_local_cloud_benchmark.py tests/test_local_cloud_confirmation_runner.py `
  tests/test_local_cloud_ood_benchmark.py tests/test_learnable_small_gain_local_cloud.py `
  tests/test_local_cloud_v13.py tests/test_reliability_rollout_bridge.py `
  tests/test_parent_anchored_rollout_bridge.py tests/test_sparse_causal_bridge.py `
  tests/test_delayed_linear_credit.py tests/test_universe_life_kernel.py `
  tests/test_l3_ne2_open_set.py tests/test_l3_nonlinear_las.py `
  tests/test_l4_weighted_routing.py tests/test_l5_role_split.py `
  tests/test_l6_activity_closure.py tests/test_l7_region_loop.py `
  tests/test_l8_internal_kernel.py
```

Result: **541 passed, 2 warnings in 39.76 s**. The warnings are the existing
PyTorch sparse-invariant and sparse-CSR-beta warnings from `runtime.py:395`.
This is internal conformance, not an external AGI benchmark.

The L3--L8/finite-host subset was repeated separately: **84 passed in 9.68 s**.
Those source and test files are untracked in the current git worktree, so the
result is current-worktree reproducibility rather than clean-clone
reproducibility.

The V15/V16 metric subset plus MRA was repeated separately: **61 passed in
10.25 s**.

## V18b sealing state

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp '_workspace\ce\agi-frontier-comparison-20260815\artifacts\pytest-v18b-tmp' `
  tests/test_v18b_benchmark.py --tb=short
```

Result: **4 failed, 9 passed in 13.04 s**. The current production module name
contains a hash suffix while one test expects the old fixed name, and the
current production bytes do not match the sealed manifest. Therefore V18b's
unit implementation exists, but a current sealed confirmation result is not
valid.

## Episodic-memory synthetic benchmark

```powershell
.\.venv\Scripts\python.exe examples\agi\episodic_memory_bench.py
```

Result: exit 0, `hard_gate=true`, `grade=GO`, 32 seeds. Candidate latest-value,
evidence, abstention, and delete accuracy were all `1.0`; composite lower
confidence bounds versus existing/FIFO/merge-off were `0.75/0.25/0.75`.
The evaluator's own claim limit is “synthetic bounded-capacity key/value
memory mechanics only.”

## STDP efficacy benchmark

```powershell
.\.venv\Scripts\python.exe examples\agi\stdp_efficacy_bench.py
```

Result reproduced the documented seven-seed values:

- on-minus-off next-step improvement: `-0.38330 +/- 0.49436` — `NO-EFFECT`;
- on-minus-off slope: `+0.00172 +/- 0.00282` — no improvement;
- held-out guard delta: `+0.34797 +/- 0.06621` versus tolerance `0.02` — `FAIL`;
- 60 STDP updates per seed and weight drift about `5.56--5.65`, so the failure
  is not a dead update path.

## Independent scratch

```powershell
.\.venv\Scripts\python.exe `
  '_workspace\ce\agi-frontier-comparison-20260815\artifacts\internal_math_probe.py'
```

Key outputs:

- `diag(2, 0.4)` has determinant squared `0.64 <= 1` but spectral norm `2 > 1`;
- with scalar `V=2`, `eta=0.2 < 1/lambda_max(V^T V)=0.25`, the documented
  LBONorm update factor is `1.6`, not a contraction;
- V10/V12 declared finite small-gain certificates are internally certified
  (`q=0.935555...` and `q=0.96125` respectively);
- all stored V13/V13b/V13c development artifacts report `STOP`; only their
  integrity gate is true;
- the finite-host public constructor accepts `HybridState(2,0,0)`, and changing
  host `sigma` from 0 to 1 leaves the sensor/action next state unchanged;
- `ConsciousnessMonitor.metacognition_step(1)` returns the prescribed scalar
  sequence `[1, 0.155, 0.024025]`; this is not task-level self-correction.

