# BA-TR26 implementation

Status: `COMPLETE`.

The experiment is isolated in
`runtime_3x3_unlabeled_content_transfer.py`; no BrainRuntime transition was
changed.  A 30-coordinate runtime contains a twelve-coordinate input pool, six
competition coordinates, an unused six-coordinate block, and six output
coordinates.  Six equal-norm positive content columns are moved by a fresh
injective episode map.  The held-out episode moves every column into the
second input block.

The learner performs a rank-four SVD fit on eight shuffled opaque rows and
enumerates cue parallelograms without grid metadata.  Recall compares the
predicted content sum with all pairs among the three current packet columns.
The expected target is built only after preflight as the union of two atomic
responses from the current snapshot; no fixed role-to-target table is used.

One source event produces input-ring receipts `[0,0,0,3,0,0,0]` and
`[0,3,0,0,0,0,0]`.  The synthetic endpoint is opened only after rank, fit,
span, rectangle, binding, association-shuffle, row-order, chart, coordinate,
and alternative-completion receipts pass.
