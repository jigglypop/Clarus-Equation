# Research contract

Status: COMPLETE

## Objective

Re-audit the CE-AGI sparse causal bridge line through V6 and restart it as a
new V7 preregistered experiment on the current `main` branch.  The primary
question is narrower than AGI: can a prefix-only predictor improve genuine
20-step free rollout in the frozen four-chart synthetic OOD family without
future reads, hidden-state access, target-window tuning, or an unfair probe
budget?

## Frozen scope

- Historical source: Git branch `itself` at `33836b8`; it is evidence, not the
  implementation base.
- Canonical base: current `main` at `fcb754e`.
- Independent unit: one fresh simulation seed.
- Primary horizon: H20; H5 is a diagnostic and must be the exact prefix of the
  same H20 rollout.
- Validation and test seeds must be disjoint from V1--V6 and from any V7
  development/pilot seeds.
- V7 registration must be written and hash-locked before its implementation or
  validation outcomes are inspected.

## Claim boundary

Passing V7 may support only a synthetic forecasting/controller claim in one
fully observed, known-family, matched-basis world.  It does not establish AGI,
causal discovery in open worlds, brain equivalence, autonomous long-horizon
agency, consciousness, or physical CE correspondence.

## Decision rule

1. Preserve all V1--V6 failures and report them as development evidence.
2. Reject any V7 route that changes gates after seeing V7 validation outcomes.
3. Compare against persistence, the strongest stable dense prefix model, the
   frozen V5/V6 parent where reproducible, and an equal-information/equal-probe
   control.
4. Require finite/stable rollouts, zero future observation reads, paired
   seed-level uncertainty, and explicit resource accounting.
5. If no defensible single-change V7 can be preregistered, stop with a blocked
   report rather than manufacture a success.

## Deliverables

- Independent source/result, mathematical/statistical, and alternative-route
  lane reports.
- Formal status audit separating definitions, derivations, empirical results,
  incomplete claims, and rejected claims.
- At most two P0/P1 fixes.
- A locked V7 preregistration plus the minimum implementation/tests needed to
  run validation, or a documented blocker if that would violate the lock.
- Reproducible validation commands and a final report.
