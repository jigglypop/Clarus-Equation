# Validation

Status: COMPLETE

Focused command:

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_broad_edge_selector.py
```

Result: `3 passed` in approximately five seconds. It checks 32-by-2 count normalization, weight-blind top-four compilation and tie failure, plus one full held-out delayed-runtime seed.

Development command:

```text
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_broad_edge_selector_benchmark --stage development --output _workspace\ce\brainruntime-broad-edge-selector-20260822\artifacts\development-results.json
```

Result: `GO`, `16/16`.

| arm | mean held-out joint accuracy |
|---|---:|
| `ORACLE` | 1.000000 |
| `EDGE_FIELD_LEARNED` | 1.000000 |
| factor-A/B training shuffle | 0.000000 |
| each `STATIC_00/01/10/11` | 0.250000 |
| `RANDOM_MATCHED_16` | 0.018663 |
| `FULL_72` | 0.000000 |

Every factor had 32 physically nonzero candidate edges with the identical weight one. Both weight-only and cue-pooled selectors had an exact top-four boundary tie and abstained before endpoint. The four mapping parity pairs occurred exactly four times each. Learned advantage over the strongest exact-16 non-oracle endpoint control was `0.75`. All local-input, strict-margin, mask-budget, freeze, zero-store, and held-out-exclusion gates passed. Confirmation `99801..99832` remains sealed.

Machine result SHA-256: `1c311658401796a34ed009fb9a1bf44659007a42bc245c3d5bfb0fc0a5fc1628`.
