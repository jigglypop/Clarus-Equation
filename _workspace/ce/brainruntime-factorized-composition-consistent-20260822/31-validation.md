# BA-TR19 validation

Focused validation:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_factorized_composition_consistent.py -q -p no:cacheprovider
2 passed in 4.23s
```

Fresh calibration seed `106001` passed all gates: atomic 4/4,
source-factorized pairs 4/4, independent union 4/4, legacy global WTA 0/4,
and one-tick-misaligned provenance 0/4.

Fresh development seeds `106101..106116` produced:

- row gate: 16/16;
- atomic recall: 64/64;
- source-factorized pair recall: 64/64;
- independent atomic union: 64/64;
- adaptive post-sum top-2: 54/64, descriptive only;
- legacy global WTA: 0/64;
- misaligned source receipt: 0/64;
- first-arrival hidden positive count: exactly two in all 64 pair probes;
- correct target activation: `7.52075284253806e-4` to
  `2.15593795292079e-3`, with wrong target activation zero;
- source delay-ring receipt: `[0,0,0,2,2,2,2]` in every pair probe.

The store-cutoff gate required zero temporal and hippocampal rows and every
probe ended with zero hippocampal rows.  The aggregate artifact serializes the
boolean gate and per-probe hippocampal count, but not the raw cutoff receipt;
that limitation is retained.  `endpoint_opened=false` is a status field, while
source inspection separately confirms that this module calls no endpoint API.

Atomic decoding is a single-label argmax-plus-margin rule. Pair and union
decoding are multi-label threshold sets. They share the `1e-5` activation
floor but are not the same decision operator.

