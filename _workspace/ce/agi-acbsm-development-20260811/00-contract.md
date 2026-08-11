# ACBSM development-only promise contract

Status: COMPLETE

## Objective

Implement one frozen `Adaptive Causal Belief-State Model` core and measure its
directional promise once. This is route selection, not V9 confirmation.

## Prohibited actions

- Do not create `sparse_causal_bridge_v9.json`.
- Do not simulate V8 locked test seeds 81100..81355.
- Do not treat V8 validation or any prior development split as new evidence.
- Do not change V1–V8 registrations, implementations, or artifacts.
- Do not choose another ACBSM route after seeing the new development block.

## Frozen new development role

- Environment: inherited `ood` synthetic family.
- Seeds: 82100..82355 inclusive, exactly 256 independent episodes.
- Origin: 80; observed prefix x[0]..x[80].
- Target: x[81]..x[100], H20.
- Historical raw-role overlap: zero at contract creation.
- This block is permanently development-only after its first execution.

The block must not be simulated until the implementation, unit tests, model
configuration, comparison set, and score formula are written and hashed.

## One model and fixed comparisons

Candidate: frozen sparse mechanism plus rank-two fast/slow residual state,
prefix-only posterior filter, and transition-internal correction.

Comparisons:

1. frozen V5 sparse parent;
2. persistence;
3. same observer on a zero-cross-chart-bridge mechanism;
4. same observer on the matched dense mechanism;
5. rank-one sparse state ablation;
6. frozen V8 R1 as historical readout reference only.

## Directional statistics

Use per-seed normalized H20 path RMSE and paired two-sided Student-t 95 percent
endpoints with df 255 critical value 1.9693105698498752.

## Promise score, 0--100

- V5 transfer: 30 points. Linear from 0 at lower CI <= 0 to 30 at lower CI >= +0.005.
- Persistence and zero bridge: 10 points each for strictly positive lower CI.
- Rank-two contribution: 10 points for strictly positive lower CI versus rank one;
  5 further points if both modes are noncollapsed and ordered.
- Dense symmetry: 10 points if paired log-RMSE-ratio upper <= log(1.02).
- Stability: 10 points if every retained component pole and pathwise radius <= 0.98,
  covariance is PSD, and all predictions are finite.
- Integrity: 10 points for exact prefix boundary, zero future/hidden reads,
  deterministic H20/H5 slicing, disjoint seeds, and unchanged lock hashes.
- Parsimony: 5 points if rank remains exactly two with no regime, memory,
  graph adaptation, beam, or planning module enabled.

Classification also requires all stability and integrity conditions:

- 75--100 and V5 lower CI >= +0.005: PROMISING, eligible for later V9 design.
- 60--74, or 75+ without the V5 buffer: HOLD, useful signal but insufficient.
- below 60: REJECT the current ACBSM core.

The numerical score is a development decision aid, not a probability of AGI
or scientific truth.
