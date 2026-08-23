# BA-TR17 validation

Focused command:

```text
.codex\hooks\python.cmd pytest tests/test_runtime_adaptive_competition_composition.py -q -p no:cacheprovider
```

Result: `3 passed` in `3.61 s`.

Calibration `104001` passed: atomic `4/4`, adaptive pairs `4/4`, legacy pairs
`0/4`, misaligned-count pairs `0/4`.

Fresh development `104101..104116` stopped:

- row pass: `8/16`;
- atomic recall: `64/64`;
- adaptive simultaneous pairs: `47/64`;
- legacy count-blind pairs: `0/64`;
- misaligned-count pairs: `0/64`;
- independent union: `64/64`;
- every adaptive pair had exactly two positive first-arrival H units.

The failure is not insufficient winner capacity: exactly two H units survived
in all 64 probes. In failing probes, top-2 of the summed candidate vector did
not equal the union of the two separately selected atomic winners, producing a
wrong or subthreshold Y component.

