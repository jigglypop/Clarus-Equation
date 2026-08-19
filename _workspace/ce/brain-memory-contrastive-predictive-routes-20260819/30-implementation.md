# T1/M2/M3 implementation record

Status: COMPLETE

## Code boundary

All successor mechanisms are isolated in
`reality_stone/python/reality_stone/clarus/runtime_contrastive_predictive_memory.py` with a separate
benchmark CLI and focused test file. The confirmed predecessor
`runtime_alternative_memory.py` was imported read-only; its M1 learner, threshold, and confirmation
artifacts were not edited.

T1 reuses the frozen delayed signed eligibility learner on three factor combinations. It adds only
factor codebooks, held-out evaluation, complete store cutoff, snapshot parity, schedule manifests,
and route-level aggregation.

M2 implements a supervised positive-minus-negative virtual lag collector. Every positive and
negative phase forks the same fixed recurrent snapshot, executes cue, transient reset, and three
NREM ticks, and leaves weight unchanged until one bounded additive install. The initial recurrent
matrix is a Gaussian-rank/sign-derived exact fixed point of the declared projection. Raw,
projected, requested, clipped, installed, and actual deltas are separately recorded. Automatic STDP
is disabled.

M3 implements one typed `12*d+5` native feature builder, exact effective replay reconstruction,
actual STP/mask-gated recurrent pre, 64-row ridge fitting, 16 disjoint held-out predictor forks, and
three-term teacher-forced replay-residual writes. The predictor freezes before scoring and all task
writes. Its controls preserve the native replay trajectory while changing only residual-credit
pairing, timing, sign, replay value, target assignment, or the write boundary.

## Focused invariants

- row-post/column-pre outer-product direction;
- exact projection fixed point and identical-phase zero update;
- virtual collector first-term reconstruction;
- `12*d+5` feature dimension and frozen predictor hash;
- exact replay vector and exact gated recurrent pre;
- cyclic residual-credit derangement and causal one-block delay;
- one bounded additive native install per block, zero automatic STDP;
- physical Temporal/Hippocampus cutoff, snapshot parity, and dense/CSR parity.

The development implementation used at most the two declared revisions per mechanism. Confirmation
seeds were never opened.
