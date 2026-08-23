# Stable-snapshot status audit

Status: COMPLETE

Gate: PASS

## Provenance verdict

The contract preserves the predecessor's confirmed split: native causal-STDP failed binding and
factor transfer, while the bounded supervised projection is a capacity ceiling. It does not reuse
the PFC covariance result, SCC formalism, or bounded next-state prediction as evidence for a missing
causal mechanism.

## First audit and revisions

The first stable-snapshot audit returned `Gate: REVISE` with no P0 and four P1 items:

1. M1 scalar modulation was not operationally unique.
2. The applied post/pre block contrast lacked an exact statistic and tolerance.
3. M3 fitting and freeze boundaries were under-specified.
4. M0 SVD construction, normalization, constraints, controls, and minimum-rank rule were incomplete.

The contract and mathematical lane were revised without opening development or confirmation
results. The second independent read-only audit returned `Gate: PASS`.

## Closed predicates

- M0 uses the raw supervised desired matrix $W^*$, truncates before subtracting initial weight,
  disables Dale/structural projection, installs under norm `5.0`, matches the random control's
  singular spectrum, norm-matches cue-only, and selects the first rank passing at least 80% of
  development circuits.
- M1 uses exactly a target-, state-, reward-, memory-, and condition-blind `+1.0` block-end clock
  pulse. Zero and sign lesions alone change it to `0.0` or `-1.0`.
- M1 computes the frozen $B(\Delta W)$ statistic on the actual post-install matrix and requires
  `B > 1e-6` plus a target-shuffled margin above `1e-6` in at least 80% of confirmation circuits.
- Temporal reversal changes event order but not pairing; target shuffle changes assignment but not
  order; no-replay preserves ticks and pulses; eligibility reset occurs between cue and target.
- M3 fits a circuit-local ridge predictor on exactly 64 native transitions, freezes it before all
  held-out scoring/writes, and cannot upgrade prediction success to binding without separate gates.
- Physical store cutoff, zero-input evaluation, fixed decoder, same-snapshot arms, seed-level units,
  and unopened confirmation seeds are explicit common gates.

## Implementation authorization

The authorized first slice is M0 followed by M1 only. M2, M3, G1--G3, C1, and S1 remain frozen but
pending. Confirmation seeds `99201..99232` remain prohibited until the implementation, primary
endpoints, thresholds, test results, and source hashes are frozen.

Implementation validation must directly prove:

- independent-reset construction of M0 $W^*$ and singular-spectrum/norm control parity;
- no M1 weight mutation inside a replay block;
- one block-end update and post-install $B(\Delta W)$ measurement;
- identical event, tick, and pulse counts across all M1 arms;
- physical zero-store state before and after every scored rollout;
- deterministic development results and unchanged legacy runtime defaults.

No implementation result from this run may be described as biological consolidation, real-brain
geometry, general metacognition, selfhood, or consciousness.

