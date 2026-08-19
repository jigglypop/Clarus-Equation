# Stable-snapshot audit

Status: COMPLETE

Gate: PASS

## Revision history before implementation

The first mathematics audit found three M3 P0 defects: inconsistent feature dimension, use of the
reset-preceding cue as if it were the actual predictor predecessor, and an unspecified three-tick
update. The revision introduced one typed feature schema, distinguished cached cue credit from the
true immediate state, and fixed the exact three-term matrix.

The second audit found two P0 counterexamples: projection could change M2 weight even under identical
phases, and M3 omitted the target-dependent replay drive from its feature while fitting no replay
rows. The final revision requires a verified projection fixed point and includes exact effective
replay drive plus independent replay calibration in the frozen predictor.

The final mathematics and independent status audits both returned PASS with no remaining P0/P1 at
the contract level.

## Frozen implementation conditions

- T1 reuses the predecessor M1 learner and threshold without edits.
- M2 begins from a diagonal-zero fixed point of the exact declared projection and directly tests
  identical-phase actual delta at tolerance `1e-7` before task scoring.
- M3 uses one `12*d+5` feature schema everywhere, logs the actual effective replay drive and actual
  STP/mask-gated recurrent pre, and reconstructs its declared three-term update exactly.
- Automatic STDP is disabled and zero automatic updates are asserted for M2/M3.
- Fit, held-out predictor scoring, and association learning restore independent named snapshots.
- Task codewords cannot enter predictor replay calibration. Held-out `(1,1)` cannot enter any
  learning row, collector, update, threshold, or calibration.
- All arms retain equal schedules, physical store cutoff, zero-store rollout, dense/sparse parity,
  finite-state checks, and circuit-level decisions.

## Authorized order

1. T1 held-out factor transfer.
2. M2 invariant tests, then binding and transfer.
3. M3 predictor gate without writes, then binding and transfer.

Confirmation seeds `99301..99332` remain sealed until the final source and results are frozen. The
maximum claims remain synthetic schedule-bound transfer, supervised contrastive acquisition, and
teacher-forced heuristic replay-residual learning. Biological consolidation and consciousness are
outside the claim space.

## Logged implementation revisions

M2 revision 1 iterated the declared projection after a focused test found a float32 Frobenius
residual of `2.31e-7`. Three 48-dimensional development matrices still entered one-ulp normalization
cycles. Revision 2 therefore derived an exactly unit-norm binary-amplitude support from the same
Gaussian ranks and signs. All 16 development seeds then had fixed-point residual exactly `0.0`.
Neither revision changed a task threshold, learning rate, decoder, or observed task result.

M3 revision 1 replaced its non-fixed random starting matrix by the same exact fixed-point family
with a disjoint seed offset. This prevents a zero residual from acquiring a projection-only write.
The predictor was rerun from scratch after that change. Its gate still failed 16/16. No M3 task
threshold or predictor ridge was revised.
