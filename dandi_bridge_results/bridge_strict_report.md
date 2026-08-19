# DANDI 001695 strict multi-animal bridge test

Status: `COMPLETE`

Source workflow run: `32212213054`; job: `95946976649`.

Locked analysis: bin `0.05 s`, PCA rank `5`, history `3`, horizon `1`, ridge `1.0`. One official behavior+ecephys NWB session was used for each of M01, M02, M03 and M05. The target model included target history and the third-region control; the bridge model added the candidate source region. `Delta NLPD` is baseline minus bridge, so positive values mean the source improved held-out future-state prediction.

| path | mean forward ΔNLPD | mean reversed-time ΔNLPD | forward - reversed | exact p forward > 0 | exact p forward > reversed |
|---|---:|---:|---:|---:|---:|
| CA3 -> CA1 | 0.017693 | 0.009516 | 0.008177 | 0.0625 | 0.1250 |
| CA1 -> RSC | 0.018724 | 0.008173 | 0.010551 | 0.0625 | 0.0625 |
| CA1 -> CA3 | 0.021161 | 0.017765 | 0.003395 | 0.0625 | 0.1875 |
| RSC -> CA1 | 0.025435 | 0.029876 | -0.004441 | 0.0625 | 0.8125 |

Expected anatomical-direction contrasts at equal lag:

- `CA3 -> CA1` minus `CA1 -> CA3`: mean `-0.003468`, exact one-sided sign-flip `p = 0.6875`.
- `CA1 -> RSC` minus `RSC -> CA1`: mean `-0.006711`, exact one-sided sign-flip `p = 0.9375`.

## Interpretation boundary

The official recordings contain cross-region information that can improve future-state prediction. However, with four animals the exact one-sided resolution is coarse, and the expected anatomical forward routes did not dominate the reverse routes. This analysis therefore supports predictive inter-regional coupling, but not a unique directed routing law or causal bridge.
