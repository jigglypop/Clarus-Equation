# BA-TR27 validation

Status: `PASS`.

Calibration seed `114001` passed before development.  Fresh development seeds
`114101..114116` produced `16/16` passes, balanced as eight hidden
$\kappa=1$ and eight hidden $\kappa=2$ fixtures.  Learned and oracle routes
were both `16/16`; joint lookup, canonical coordinate, wrong cue,
packet-binding shuffle, and no-context controls were each `0/16`.
Cue/content association shuffle was rejected before endpoint in every row.

Every seed admitted 36 gauge-equivalent candidates with class labels
`[1,2]`.  Maximum selected residual was
`3.980046983219726e-16`; minimum additive residual was
`0.4202366510382134`; maximum query content error was
`9.437797496072644e-16`; minimum binding margin was
`0.7924058284059866`.  All query-gauge spreads were below `2.6e-15`.

Each main probe produced exactly two positive first-arrival hidden coordinates
and two targets.  Positive hidden activation was
`0.25735095143318176`; positive target activation was
`0.0022818935103714466`.  One-shot receipts were
`[0,0,0,3,0,0,0]` and `[0,3,0,0,0,0,0]`.

Focused validation:

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests/test_runtime_z3_twisted_content_transfer.py
4 passed, 2 warnings in 8.31s
```

The warnings are existing PyTorch sparse-CSR beta/invariant warnings.
Source-freeze verification passed all twelve declared entries.
