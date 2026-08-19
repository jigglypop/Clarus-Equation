# Geometry-route validation

Status: COMPLETE — G1, G2, and G3-D development routes closed.

## Focused implementation validation

`python -m pytest tests/test_runtime_metric_intervention.py -q -p no:cacheprovider --basetemp C:\\tmp\\clarus-g1-pytest-20260819b`

Result: 5 passed. The warnings are PyTorch's existing sparse-CSR beta/invariant warnings; no test
failed. `py_compile` and `git diff --check` also passed.

## G1 development result

Artifact: `artifacts/g1-development-results-v1.json`

- circuits: 16 (`97401..97416`)
- integrity: 16/16
- circuit-level GO: 0/16
- mean held-out endpoint advantage over the strongest within-circuit control: `0.0311778448`
- minimum endpoint advantage: `0.0066936463`
- 10,000-resample 95% bootstrap lower bound of the mean: `0.0247662745`
- minimum calibration cross-response advantage: `0.0517854753`
- sign-specific first-passage gate: 16/16
- held-out linearization-error gate: 16/16
- AIRM treatment-versus-sham range: `[0.6168415173, 0.6265180657]`
- route verdict: `STOP`

The sole universal failing gate was the preregistered per-circuit endpoint advantage `>=0.05`:
the observed range was `[0.0066936463, 0.0455462234]`. The noise-only arm was the strongest endpoint
control in all 16 circuits. Treatment endpoints themselves were narrow,
`[0.0458138362, 0.0473878216]`; seed-dependent noise-control endpoints reduced the paired margin.

This supports only a sub-threshold simulator effect: the directed weight intervention changed the
declared calibration response and, on average, the held-out target endpoint. It does not satisfy the
frozen effect-size requirement, identify `g-to-x`, establish metric mediation, or justify
confirmation. Seeds `99401..99432` remain unopened.

## G2 focused validation and development result

Focused G2 tests: 5/5 passed. The adjacent legacy runtime snapshot-continuity test passed 1/1.
The tests include default-off bit parity, all-negative forced selection, snapshot persistence,
dedicated G2 fixture separation, codebook/noise disjointness, all-tick 48/48 masks, fixed W/CSR,
transform covariance, coefficient ledger, and non-repackaging identity.

Artifact: `artifacts/g2-development-results-v1.json`

- fresh circuits: 16 (`97601..97616`)
- integrity: 16/16
- cross-seed noise intervals disjoint: yes
- circuit-level GO: 0/16
- mean/min/max worst-adversary contrast: `-18.37506985 / -20.56424350 / -15.76409597`
- one-sided 95% bootstrap lower bound: `-19.04150933`
- no-repackaging prediction residual maximum: `0`
- metric-feature chart residual maximum: `5.3291e-15`
- route verdict: `STOP`

Every circuit was beaten by horizon-matched direct `B_h`: `raw_Bpath` was the strongest comparator
in 11 circuits and fitted `D+Bpath` in 5. Mean Gaussian loss was `15.78889148` for `D+g` versus
`-2.48452940` for raw `B_h`; mean MSE was `0.00389502` versus `0.00024034`. `D2` and `D+Cterms`
also beat `D+g` in all 16 circuits. The permuted metric beat the named metric in 8/16, so orientation
specificity was not stable either.

Thus the SPD precision is a valid compressed response representation but has no incremental path-
prediction utility against direction- and horizon-preserving calibration in this experiment. G2
confirmation seeds `99601..99632` remain unopened.

## G3-D focused validation and development result

Focused G3-D tests: 8/8 passed. They cover full-default frozen-M1 parity, strict AIRM inputs,
hippocampal/temporal zero-store probes, structural no-alias reconstruction, fresh null-lesion forks,
same-arm bootstrap rows, exact/retired/confirmation seed seals, forged-manifest rejection, and the
float32 representable-target boundary.

The immutable first artifact, `artifacts/g3-diagnostic-development-results-v1.json`, is
`APPARATUS_INVALID` and quarantined by
`artifacts/g3-development-results-v1-apparatus-invalid.md`. All 256 lesion installs had
intended-delta reconstruction residuals `[1.64868e-7,1.88207e-7]` against the old `1e-7` gate.
Those inspected seeds `97701..97716` were retired; none of their scientific fields is combined with
the replacement run.

Valid artifact: `artifacts/g3-diagnostic-development-results-v2.json`

- replacement circuits: 16 (`97801..97816`)
- integrity: 16/16
- directional/joint-positive circuits: 5/16 (`31.25%`)
- matched continuous recall mean: `0.69362753`
- target-shuffled/no-replay/weight-permuted recall means: `-0.22998225 / 0 / -0.00268058`
- simultaneous 95% LCB for mean recall advantage: `0.66444402`
- matched/target-shuffled/weight-permuted SPD-change means: `0.30047165 / 0.29697132 / 0.20936721`
- simultaneous 95% LCB for mean SPD-change advantage: `-0.04261143`
- same-arm correlations (shuffle/no replay/permuted): `-0.50810648 / 0.02393755 / 0.12320795`
- simultaneous 95% LCB for same-arm correlation: `-0.80389991`
- calibration-null falsifiers: `0/16`
- maximum selected lesion recall shift: `0.00510978`
- maximum final-target reconstruction residual: `0`
- route verdict: `STOP`

The matched condition robustly acquires the cue/value recall relation, but the independent global
SPD response-change summary neither dominates the target-shuffle/structural controls reliably nor
co-varies positively with recall. Failure to find a null-lesion falsifier is not sufficiency
evidence. `mediation_status` remains `BLOCKED_NOT_IDENTIFIED`, and confirmation seeds
`99701..99732` remain unopened.
