# Validation

Status: COMPLETE

## Focused implementation test

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_endogenous_competition_homeostasis.py
```

Result: `4 passed` in 4.65 seconds. The two warnings are the existing PyTorch
sparse-CSR beta/invariant warnings; no test failed. The test covers config and
Rust fail-closed behavior, delayed state, snapshot continuation, reset, absence
of hard allocator inputs, and the frozen calibration seed.

## Adjacent runtime regression

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_delay_events.py tests\test_cloudcell_kleisli.py::test_full_snapshot_is_a_sufficient_state_for_continuation tests\test_cloudcell_kleisli.py::test_snapshot_and_restore_do_not_alias_runtime_config_or_tensors tests\test_cloudcell_kleisli.py::test_restore_rejects_malformed_delay_state tests\test_runtime_neuronwise_thresholds.py::test_scalar_threshold_mutation_remains_live tests\test_runtime_neuronwise_thresholds.py::test_snapshot_preserves_vectors_and_nontrivial_delay_continuation
```

Result: `9 passed` in 3.86 seconds, with the same two existing sparse warnings.

## Frozen calibration and development

Calibration seed `97091` returned `CALIBRATION_PASS` without changing any
coefficient. Its binding was a bijection, minimum winner margin was about
`0.0220521`, every source washed out in 53 ticks, the $\lambda=0$ collision
fraction was `0.25`, and uniform weights abstained.

Development command:

```text
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_endogenous_competition_homeostasis_benchmark --stage development --output _workspace\ce\brainruntime-endogenous-competition-homeostasis-20260822\artifacts\development-results.json
```

Result: `DEVELOPMENT_GO`. All 16 seeds passed every apparatus gate and formed
four-source bijections. Mean collision was `0.0` with persistent homeostasis and
`0.28125` at $\lambda=0$, so the paired mean reduction was `0.28125`, above the
frozen `0.20` gate. Winner margins ranged from `0.0220521` to `0.205139`;
washout was 53 ticks for every episode. Uniform weights abstained in 16/16,
row permutation and snapshot continuation passed in 16/16, and confirmation
remained unopened.

The source-independent row-bias control also produced positional bijections in
16/16. It is deliberately classified `SOURCE_UNIDENTIFIED`, confirming why a
bijection alone is not evidence of source-specific local information.

The four changed Python sources also passed in-memory `compile()` without
writing bytecode. Every source and artifact digest in
`artifacts/source-freeze.json` was recomputed after the final focused test and
matched exactly. `run.cmd check ... build`, `run.cmd check ... final`, and the
scoped `git diff --check` all passed. No full suite was run because the focused
and adjacent runtime checks cover the changed API and the user requested rapid
iteration rather than a release gate.
