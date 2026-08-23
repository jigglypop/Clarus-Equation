# BA-TR24 validation

Focused test:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_factor_context_relevance_composition.py -q -p no:cacheprovider
1 passed in 3.87s
```

Fresh calibration seed `111001` passed. Fresh development seeds
`111101..111116` produced:

- row gate: 16/16;
- heldout `11` factor-context composition: 16/16;
- oracle `11`: 16/16 and bit-exact with factor gate;
- joint lookup with absent-`11` fallback: 0/16;
- factor-A cue shuffle: 0/16;
- factor-B cue shuffle: 0/16;
- no-context all-input rule: 0/16;
- training context exclusion, counts `[2,1]` for both factors, exact four
  compilers, gate immutability, and zero stores: 16/16.

Confirmation remains sealed.

