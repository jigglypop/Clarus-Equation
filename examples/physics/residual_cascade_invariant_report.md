# Residual Cascade Invariant Gate

## Cascade state

| quantity | value |
|---|---:|
| D_eff | 3.17775842 |
| x | 0.04864672 |
| sigma | 0.95135328 |
| contraction D_eff*x | 0.15458752 |
| gamma_eff | 0.76063719 |
| N_e | 57.19965162 |
| Q_total | 0.05474276 |
| Q_source | 0.04628022 |
| Q_phase | 0.02946290 |
| Q_GER | 0.02836622 |

## A_s cascade

| readout | A_s | pull | status |
|---|---:|---:|---|
| total susceptibility | 7.83532001e-09 | +191.18 | rejected |
| source residual | 5.60007760e-09 | +116.67 | source only |
| half-cycle source | 2.26962596e-09 | +5.65 | undershoot |
| GER source | 2.10380875e-09 | +0.13 | selection candidate |

## Invariants

| name | equation | value | status | guardrail |
|---|---|---:|---|---|
| raw gain | Q_total/Q_source = 1/(1-D_eff x) | 1.18285456 | algebraic source of raw A_s overshoot | do not use total susceptibility as scalar amplitude readout |
| half-cycle projection | P_phase = 2/pi | 0.63661977 | shared d0->d3 projection candidate | projection alone undershoots A_s; it is not the final scalar readout |
| GER projection | P_GER = (2/pi) sigma^(D_eff/(D_eff+1)) | 0.61292326 | single projection reused by A_s and large-angle handles | selection candidate, not exact theorem |
| A_s GER pull | pull(A_s[Q_GER]) | 0.12695822 | inside broad scalar amplitude gate | must survive running/tensor/common-readout tests |
| tensor-running lock | r_tensor/(-alpha_spec) | 6.00000000 | exact N_e-family ratio | open until joint primitive spectrum likelihood |
| hemispherical identity | 2 Q_GER/sigma = 2 P_GER x | 0.05963341 | large-angle amplitude handle | does not select a preferred axis by itself |

## Verdict

The scalar residual cascade is coherent: raw total sensitivity is rejected, while a single GER projection reuses the same source residual across A_s and large-angle amplitude handles.

The next hard test is data-facing: running, tensor, and CMB large-angle likelihoods must reuse this same cascade without adding observable-specific recursion.
