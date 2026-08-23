# BA-TR18 validation

Focused command:

```text
.codex\hooks\python.cmd pytest tests/test_runtime_factorized_competition_composition.py -q -p no:cacheprovider
```

Result: `2 passed` in `3.69 s`.

Calibration `105001` passed the frozen contract: factorized `4/4`, global
adaptive top-2 `2/4`, legacy `0/4`, misaligned `0/4`.

Fresh development `105101..105116` produced:

- factorized simultaneous route: `64/64`;
- atomic recall: `64/64`;
- global adaptive top-2: `48/64`;
- legacy WTA: `0/64`;
- misaligned source receipt: `0/64`;
- independent-union control: `62/64`;
- row gate pass: `15/16`.

The sole failing row was seed `105104`, and only the independent-union control
failed. One singleton output was `1.8561986507847905e-5`, above the atomic
decoder's existing `1e-5` threshold but below the composition control's
`2e-5` threshold. The actual factorized simultaneous outputs for the affected
pairs were `2.123687881976366e-4` and passed. No factorized pair failed.

