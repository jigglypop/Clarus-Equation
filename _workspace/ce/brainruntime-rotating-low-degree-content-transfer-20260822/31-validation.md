# BA-TR28 validation

Status: `COMPLETE / DEVELOPMENT_STOP`.

Across 16 seeds and 400 rotating missing-cell folds, the full degree-two
operator reconstructed every query with maximum relative error
`2.7445789784409926e-15`.  All 400 association shuffles rejected before the
endpoint.  Minimum binding margin was `0.016097139425517472`, minimum wrong
route separation `0.01545712512055814`, and maximum candidate norm ratio
`1.223648234993142`.

The matched affine predictor retained minimum per-seed mean content error
`0.07641872934410789`, but the random three-packet endpoint was too coarse:
its correct packet-selection fraction reached `0.60`.  The frozen maximum was
`0.50`, so six seeds failed that single aggregate gate and the run stopped at
`10/16`.  No threshold or seed was retuned.

Focused combined validation after the successor repair:
`.codex\hooks\python.cmd pytest -q -p no:cacheprovider
tests\test_runtime_rotating_low_degree_content_transfer.py
tests\test_runtime_low_degree_hard_negative_transfer.py
tests\test_dimensionless.py` returned `24 passed` with two pre-existing
PyTorch sparse-CSR warnings.
