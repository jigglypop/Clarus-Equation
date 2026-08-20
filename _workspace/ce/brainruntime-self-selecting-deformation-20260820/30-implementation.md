# M4-R implementation

Status: COMPLETE

`runtime_self_selecting_deformation.py` implements the frozen M4-R candidate
bank `(lambda, scale) = {0.50,0.80,0.95} x {0.50,1,2}` separately from frozen
M1/T1 modules.  Every epoch records raw `D`, rank at tolerance `1e-6`, pairwise
cosines, all candidate scores, deterministic selection, projected/actual delta,
held-out row count, codebook hash, parity and store-cutoff receipts.

The actual recurrent write is one selected projected candidate.  `identity`,
`no_selection`, `target_shuffled`, `trace_shuffled`, and `sign_flipped` use the
same schedule and candidate bank.  Loop 8 and Loop 9 share configuration and
candidate-bank receipt.  Loop 9 excludes `(1,1)` from collection, scoring,
and selection; it is decoded only after the sealed endpoint.

The first discovery implementation measured instability from raw input and
included the first driven state.  It was replaced before final interpretation:
`x0` is now the post-input state and only subsequent zero-input rollout states
are tested.  The superseded artifact is retained, never merged.

Discovery used only `97401..97408`, sharded deterministically with one Torch
thread per independent process.  Fold is receipt-only and remains inactive;
no validation or confirmation seed was opened.
