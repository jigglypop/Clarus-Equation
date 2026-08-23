# Stable-snapshot G1 audit

Status: COMPLETE

Gate: PASS for G1 development execution. No G1 result is implied by this gate.

## First audit: REVISE

The first independent mathematics/status audit found two P0 defects and several specification
holes:

1. `C=BB^T+lambda I` was incompatible with a general non-orthogonal chart reconstruction unless
   the regularizer reference tensor transformed too.
2. G1 could establish joint effects of `do(W)` on a derived SPD summary and behavior, but not the
   causal arrow `Delta g -> Delta x` or mediation.
3. A zero background matrix made all nominal seed circuits deterministic clones.
4. The runtime injection map, first-passage indexing, bootstrap statistic, and noise protocol were
   incomplete.

## Repairs and final audit

The contract now uses `C=BB^T+lambda R0` with `R0'=P R0 P^T`, an explicit physical injection `U`
and output chart `F`, a numerical non-orthogonal `P`, seed-varying arm-common background circuits and
coordinate permutations, sign-by-sign held-out gates, independent reset forks, a deterministic
noise policy, and an exact 10,000-resample seed-level bootstrap.

The claim was narrowed to joint `do(W) -> {Delta SPD summary, Delta behavior}`. G2 retains the
separate utility question; G3 is labelled randomized-contingency association rather than causal
mediation.

The final mathematics audit returned PASS with no P0. Remaining P1 conditions are now explicit in
the contract and must be asserted by tests and artifacts:

- rebuild transformed tensors from `B'=PB` and `R0'=P R0 P^T`;
- log determinant, condition number, and bytes of `P`;
- restore an independent identical arm snapshot for every pulse;
- aggregate held-out signs exactly as frozen and never treat pulses/ticks as independent units;
- keep Temporal/Hippocampus stores empty and automatic STDP disabled;
- recompute numeric `B,C,g` per post-update weight when the frozen procedure is later reused.

This PASS authorizes G1 implementation only. It is not a positive result and does not authorize G2
or G3 execution before their remaining exact contracts are frozen.

## Pre-development implementation audit: REVISE

The first executable one-circuit smoke check exposed a runtime-interface defect not visible in the
algebra-only audit. With unit block entries `1/4`, calibration amplitude `0.5`, and external gain
`0.45`, the pulse salience was below the runtime default active threshold `0.22`. The pulse produced
zero active senders, so treatment, sham, and scrambled matrices all had exactly zero recurrent
cross-response. This was an invalid disconnected apparatus, not a negative result; no development
seed range or confirmation seed was opened.

The contract now freezes `active_threshold=0.04`, below the analytical external-salience floor
`0.04375`, and requires a driven-coordinate-active audit on every pulse. The same review also found
that mutable diagnostic configurations and merely treatment-vs-scramble matching could have
received `GO`; the implementation now requires the complete frozen parameter set, exact 256-edge
support, +0.08 entries, applied-delta reconstruction, 1.28 norm, transform bytes, and finite
invertible `P`. Norm comparisons use a declared float32 tolerance of `1e-6`.

## Post-repair stable-snapshot audit: PASS

The independent re-audit found no remaining P0 and authorized the 16-seed development run. It
verified the analytical threshold lower bound and the per-pulse driven-coordinate gate; complete
frozen-protocol enforcement; exact post-row/pre-column support, value, norm and reconstruction;
arm-common seed-varying backgrounds; rebuilt covariant SPD tensors and fixed-transform byte audit;
and the sign-specific held-out/first-passage plus circuit-level bootstrap rules.

The main execution environment independently ran
`tests/test_runtime_metric_intervention.py` with cache disabled and a temporary directory outside
the repository: 5 tests passed. This gate validates apparatus and implementation invariants only.
