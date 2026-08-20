# M4-R focused validation

Status: COMPLETE

Focused command:

```powershell
.codex\hooks\python.cmd pytest tests\test_runtime_self_selecting_deformation.py -q -p no:cacheprovider
```

Result: `1 passed` (two existing PyTorch sparse-CSR beta warnings).

Discovery command used the dedicated CLI with only authorized discovery seed
lists and merged exactly `97401..97408`; its output asserts that validation and
confirmation were not opened.  `git diff --check` passed for the owned source,
test, and run paths.

Final source/result freeze: `artifacts/source-freeze.json`.

Observed discovery outcome: Loop 8 basic task gate is 8/8, but every seed has
`min_control_advantage = 0`; Loop 9 basic task gate is 4/8 and its advantage is
never positive.  Thus this is a selection-causal-gate failure, not a source
parse/test failure.  Corrected fold receipts have no instability; max-scale
trigger is true for Loop 8 seeds 97401, 97406, 97407 and Loop 9 seeds 97402,
97407.  Fold remains disabled pending separate Revision 2 math/audit.
