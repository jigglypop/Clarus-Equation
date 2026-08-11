# Validation

Status: COMPLETE

## V7 validation execution

Exactly one registered validation run was executed:

```powershell
.\.venv\Scripts\python.exe examples/agi/reliability_rollout_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v7.json --split validation
```

It wrote `artifacts/agi/sparse_causal_bridge_validation_v7.json` and returned
exit code 1 because the registered conjunction was false.  Artifact SHA-256:
`f172447ecc0d19ac206c6625bf5911805f28214bb5adc1d2a215c59dc3bc4e12`.

## Primary model means

All values are mean seed-level normalized H20 path RMSE; lower is better.

| Model | Mean |
|---|---:|
| symmetric dense consensus | 0.5633889431 |
| sparse consensus | 0.5634060101 |
| V5 sparse parent | 0.5569150278 |
| persistence | 0.5738975375 |
| no-sparse consensus | 0.5839581541 |
| stable adaptive dense | 0.6427024216 |

## Registered comparisons

| Comparison | Estimate | 95% interval | Clause |
|---|---:|---:|---|
| sparse contribution (`no_sparse - sparse`) | +0.0205521440 | [+0.0081743398, +0.0329299482] | true |
| V5 repair (`v5 - sparse`) | -0.0064909824 | [-0.0274308279, +0.0144488632] | false |
| persistence improvement | +0.0104915274 | [-0.0164872019, +0.0374702568] | false |
| adaptive-dense geometric error ratio | 0.8919786278 | log ratio [-0.1598832441, -0.0687429690] | true |
| symmetric-dense geometric error ratio | 0.9998693014 | log ratio [-0.0005790968, +0.0003176826] | true |

All leakage, finite-output, scale, absolute-bound, latent-AR, convex-weight,
and read-index clauses were true.  Future reads were zero and the maximum
observed state index was 80.  The maximum dynamic-component pathwise Jacobian
radius was `1.1143092447`, above the registered `0.98` limit, so the pathwise
stability clause was false.

The sparse-ablation sub-comparison is positive, but the registered V7 result is
conjunctive.  The failed parent-repair, persistence, and stability clauses close
the route.  This is not evidence of AGI capability, broad causal discovery, or
a global contraction theorem.

## Locked test disposition

- `test_unlocked`: false
- `test_opened`: false
- `artifacts/agi/sparse_causal_bridge_test_v7.json`: absent

The test split was not executed and must remain unopened.  No second V7 route
was attempted.

## Resource and integrity accounting

- validation seeds: 96
- forecast origins per seed: 1
- component rollouts per seed: 8
- total component rollouts: 768
- free-rollout steps per component: 20
- external downloads: 0 bytes
- evaluation probe pairs: 0
- trajectory files: 0
- measured wall time: 0.8598877001 seconds

## Supporting validation

- `tests/test_dimensionless.py`: 10 passed.
- `dimensionless.py`: exit 0.
- local-memory verifier plus V1--V7 focused bridge tests: 44 passed.
- V7 API poisoning tests confirmed bit-identical behavior under future-state
  and hidden-state poisoning.
