# V9 runtime integration validation

Status: COMPLETE

## Result

The authorized runtime integration and cleanup pass their focused, related, static,
dimensionless, and deterministic-demo gates. No V9-related test failed.

## Executed checks

| Check | Result |
|---|---|
| agent+tower+controller initial focus | `186 passed` |
| agent+tower+controller+runtime contracts | `200 passed` |
| final related suite including dimensionless, warning-as-error | `210 passed in 4.41s` |
| dimensionless suite | `10 passed in 2.86s` |
| dimensionless checker | exit code 0 |
| Ruff changed execution/test/demo files | `All checks passed!` |
| Ruff format | `8 files already formatted` |
| Python compileall | exit code 0 |
| deterministic runtime demo | exit code 0 |

The warning-as-error command ignored only `UserWarning` from
`reality_stone.clarus.runtime`, where PyTorch emits two existing sparse-CSR constructor
warnings. Running without that ignore produced `209 passed, 1 failed`, and the sole failure was
the promoted pre-existing sparse warning rather than a numerical or assertion failure.

## Causal and negative controls

- The returned action equals `read_policy(issued_token).selected_action`.
- The issued token is exactly the controller's latest token.
- Identical current evidence after two different legal histories yields different policies.
- All-false masks and nonfinite supplied observations fail before runtime tick advancement.
- Belief-control plus V9 is rejected.
- A controller supplied while V9 is disabled is not retained and legacy action selection is
  unchanged.
- Manifest cleanup tests assert that the two dormant fields and old misleading metadata name
  no longer exist.
- Demo output was:

```text
tick 0, depth 1, action 0
tick 1, depth 2, action 1
tick 2, depth 3, action 2
```

## Full-suite audit

The repository-wide suite completed with:

```text
2145 passed, 14 skipped, 28 failed, 41 errors
```

The failures/errors are outside the changed V9 path and split into two pre-existing worktree
conditions:

1. deleted/missing benchmark and ScienceDB payload fixtures, including
   `benchmarks/neural_tree_algorithm_census_v1.json` and
   `benchmarks/q0_minimal_abelian_higgs_v1.json`;
2. `.codex`/`.claude` policy mirror drift in `test_canonical_document_policy.py`.

The canonical-policy failure was independently rerun and names only those policy mirror files;
it does not reference V9 code or documentation. Those unrelated user changes were preserved.

## Interpretation

These checks establish implementation integrity, finite state mediation, and regression safety
for the opt-in path. They do not prove task utility, convergence of the combined BrainRuntime
and tower, predictive superiority, AGI, or biological identity.
