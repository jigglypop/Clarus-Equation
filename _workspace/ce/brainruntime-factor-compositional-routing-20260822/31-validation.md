# Validation

Status: COMPLETE

## Focused test

Command:

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_factor_compositional_routing.py
```

Result: `3 passed` in approximately five seconds. The tests cover unequal-frequency count normalization, compiler locality/tie failure, and one full delayed-runtime held-out `11` seed.

## Frozen development run

Command:

```text
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_factor_compositional_routing_benchmark --stage development --output _workspace\ce\brainruntime-factor-compositional-routing-20260822\artifacts\development-results.json
```

Result: `GO`, `16/16` seed passes.

| arm | mean held-out joint accuracy |
|---|---:|
| `ORACLE` | 1.000000 |
| `FACTORWISE_LEARNED` | 1.000000 |
| `A_FACTOR_SHUFFLE_TRAIN` | 0.000000 |
| `B_FACTOR_SHUFFLE_TRAIN` | 0.000000 |
| `A_LESION_STATIC_0` | 0.500000 |
| `B_LESION_STATIC_0` | 0.500000 |
| each `STATIC_00/01/10/11` | 0.250000 |
| `RANDOM_MATCHED_24` | 0.080295 |
| `FULL_32` | 0.000000 |

The four seed mapping pairs occurred exactly four times each. The learned advantage over the strongest exact-24 non-oracle control was `0.50`. In the A-shuffle arm, A accuracy was `0` while B stayed `1`; the B-shuffle arm was exactly symmetric. Every preflight, freeze, zero-store, direct-sum, edge-budget, and joint-lookup-abstention gate passed. Confirmation seeds `99701..99732` were not opened.

Final result SHA-256: `fc46066ca91fa2ab2dc7c40522eddc5d1fa2ed1445a9a755632e7fac6cbaffd2`.
