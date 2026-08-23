# BA-TR27 implementation

Status: `COMPLETE`.

The implementation is isolated in `runtime_z3_twisted_content_transfer.py`;
BrainRuntime and BA-TR26 remain unchanged.  It reuses the frozen 30D packet
apparatus but replaces the affine completion law with a finite nonseparable
$\mathbb Z_3$ incidence family.

Nine raw cues are supplied, while only eight current content sums are observed.
The generic learner enumerates raw-cue charts, opens all three incidence
classes, checks operational rank, fits each by pseudoinverse, rejects the
additive class, and requires every zero-residual nonzero class to agree on the
query prediction.  Binding then compares this prediction with pairs of the
three packet columns actually present after the held-out coordinate remap.

The synthetic endpoint is opened only after chart, rank, residual, additive,
gauge, binding, remap, association-shuffle, row-order, coordinate-chart, and
arbitrary-completion receipts pass.  The runtime event and ring logic are
unchanged from BA-TR26.
