# BA-TR16 validation

Focused command:

```text
.codex\hooks\python.cmd pytest tests/test_runtime_binding_composition_no_go.py -q -p no:cacheprovider
```

Result: `2 passed` in `3.47 s`.

Calibration seed `103001` confirmed the predicted no-go: atomic `4/4`, actual
simultaneous `0/4`, independent union `4/4`.

Fresh development seeds `103101..103116` gave:

- no-go witnesses: `16/16`;
- atomic nonidentity recall: `64/64`;
- actual simultaneous two-component recall: `0/64`;
- independent-union recovery: `64/64`;
- every simultaneous first hidden arrival had at most one positive unit;
- stores remained zero; confirmation remained sealed.

This result is exact for the frozen global max-relative WTA branch, not a
threshold miss. Raising the BA-TR15 write or lowering the output threshold
cannot make that branch carry two concurrent hidden winners.

