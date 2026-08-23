# BA-TR25 implementation

Status: `COMPLETE`.

The implementation is isolated in
`runtime_mixed_cue_content_transfer.py`; no `BrainRuntime` transition was
changed for this experiment.  The generic learner accepts only three raw cue
rows and three co-occurring packet-content sums.  Its compiler receives a raw
cue, the coordinates of packets that are currently present, their current
weight columns, and the response coordinates.  Factor tuples, semantic roles,
fixed source coordinates, targets, decoder values, rewards, stores, and the
held-out answer are absent from those APIs.

All state, cue, weight, content, residual, threshold, and norm values in this
synthetic fixture are normalized dimensionless quantities.  The absolute
rank and binding margins, both `1e-8`, are therefore dimensionless numerical
certification cutoffs for this frozen fixture rather than physical scales or
general rank theorems.

The held-out runtime probe copies all four learned packet columns to a
seed-specific permutation of coordinates `4..7`, emits two relevant packets
plus one matched learned-column distractor, and exposes only one ring packet.
No structural projection, endpoint fitting, or store lookup is used.

