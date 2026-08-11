# Implementation

Status: COMPLETE

## Scope

Implemented only the status-auditor-approved fresh development pilot in the
research run directory.  No canonical AGI module, historical registration,
historical artifact, or locked test split was modified or opened.

Added:

- `fresh_parent_anchor_pilot.py`: deterministic 256-seed development runner
  for training-only parent-anchored shrinkage.
- `21-pilot-lock.md`: algorithm, seed block, controls, gates, critical value,
  and implementation hash recorded before first execution.

The runner recomputes the sparse gain from inherited observational training
data and requires exact reproduction of
`g=0.7868543064870357`.  It fits symmetric dense and zero-bridge controls by
the identical training-window procedure, verifies that seeds `79100..79355`
do not overlap any V1--V7 registered role, and accepts only prefix states
through index 80.

Implementation SHA-256 before and after execution:

`b3e07dec5895e670fc4babd1dbd261a2fc9795de90f1e576ba69f76fc2de0a41`

## Invariants

- one origin, one H20 path, H5 nongating;
- no evaluation-prefix gain fitting;
- no adaptive dense component inside the candidate;
- no future/hidden input to the candidate;
- symmetric dense and zero-bridge controls use their own fitted gain;
- V7 test remains unopened;
- output is a postprocessing path and is never fed back recursively.

## Verification commands

```powershell
$env:PYTHONPATH='reality_stone/python'
.\.venv\Scripts\python.exe _workspace\ce\agi-v8-breakthrough-20260811\fresh_parent_anchor_pilot.py
.\.venv\Scripts\ruff.exe check _workspace\ce\agi-v8-breakthrough-20260811\scratch_diagnostics.py _workspace\ce\agi-v8-breakthrough-20260811\pilot_parent_anchor.py _workspace\ce\agi-v8-breakthrough-20260811\fresh_parent_anchor_pilot.py
.\.venv\Scripts\python.exe -m pytest tests\test_local_memory_verifier.py tests\test_sparse_causal_bridge.py tests\test_latent_causal_bridge.py tests\test_free_rollout_bridge.py tests\test_reliability_rollout_bridge.py tests\test_dimensionless.py -q --basetemp .pytest_tmp_agi_v8_final
```

Observed:

- fresh pilot: exit 0;
- ruff: `All checks passed!`;
- focused regression and dimensionless tests: `54 passed in 8.05s`.
- CE research core: `OK lanes`, `OK gate`, `OK build`, `OK final`.

The skill text's generic `<codex-home>/skills/ce-research/core` manifest path
did not exist in this checkout.  The first core invocation therefore failed
before compilation.  Re-running the same locked core from the repository-local
`.codex/skills/ce-research/core/Cargo.toml` completed all four checks.  Cargo
also printed a non-fatal warning that it could not canonicalize
`C:\Users\22310326` in the sandbox.
