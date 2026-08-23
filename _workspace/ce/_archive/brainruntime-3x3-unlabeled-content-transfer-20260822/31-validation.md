# BA-TR26 validation

Status: `PASS`.

Calibration seed `113001` passed before development.  Fresh development seeds
`113101..113116` then produced `16/16` conditional rank-four passes.  Learned
and remapped-oracle routes were both `16/16`; absent-row joint lookup,
canonical-coordinate memorization, wrong cue, packet-binding shuffle,
rank-three, and no-context controls were each `0/16`.  Cue/content association
shuffle was rejected before endpoint on every row.

The maximum affine training residual was
`1.2354301123036062e-15`; maximum held-out content error was
`1.8043358472135524e-15`; minimum relative binding margin was
`0.7924058284059857`.  Every main probe had exactly two positive first-arrival
hidden coordinates and two target coordinates.  Positive hidden activation
was `0.25735095143318176`; positive target activation was
`0.0022818935103714466`.

Focused validation:

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests/test_runtime_3x3_unlabeled_content_transfer.py
4 passed, 2 warnings in 3.58s

.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests/test_dimensionless.py
17 passed in 0.41s
```

The two warnings are existing PyTorch sparse-CSR beta/invariant warnings.
`artifacts/source-freeze.json` passed all eleven declared SHA-256 comparisons.
