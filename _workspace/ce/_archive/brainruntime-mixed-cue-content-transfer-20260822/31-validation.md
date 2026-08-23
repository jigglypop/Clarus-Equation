# BA-TR25 validation

Status: `PASS`.

Calibration seed `112001` passed before development opened.  Fresh development
seeds `112101..112116` then produced `16/16` conditional rank-two transfer
passes.  Learned and oracle routes were `16/16`; absent-row joint lookup,
absolute-coordinate memorization, wrong cue, packet-content shuffle, rank-one,
and no-context controls were each `0/16`.

The largest cue parallelogram residual was
`2.603703785810335e-16`; the largest held-out content residual was
`3.510833468576701e-16`.  The smallest content-pair binding margin was
`0.0108672003341486`.  Every delayed input receipt was
`[0,0,0,3,0,0,0]`, every ring-write receipt was
`[0,3,0,0,0,0,0]`, and every store gate passed.

Focused validation:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_context_packet_relevance_gate.py tests\test_runtime_factor_context_relevance_composition.py tests\test_runtime_mixed_cue_content_transfer.py tests\test_dimensionless.py -q -p no:cacheprovider
22 passed, 2 warnings in 4.43s
```

The warnings are the existing PyTorch sparse-CSR beta/invariant warnings.

