# AGI V18 formal audit

Status: COMPLETE

Gate: BLOCKED

## 1. Decision

The contract cannot authorize implementation. E1, E2 and E4 close, but E3's
registered exact realized paired-accuracy claim has a complete counterexample.
The math-verifier revision budget is exhausted at 2/2, so silently adding the
missing premise would violate the preregistration process.

## 2. Blocking P0

P0-1 concerns D2 and E3. D2 couples the post-training checkpoint and terminal
policy draw for $q,-q$, but not the whole post-checkpoint distractor/message
trajectory or every metric-update and ensemble seed. Let an allowed
sign-independent nuisance bit determine the terminal action independently in
the two branches. Their labels are opposite, yet their realized actions may be
equal or opposite. Pair accuracy can be $0$, $1/2$ or $1$. Only its expectation
is $1/2$ under symmetric nuisance.

The minimal successor premise is byte-identical coupling of all
post-checkpoint observations, distractors, messages, delays, topology and every
update/ensemble/policy seed, with fixed-seed pointwise sign-even updates and a
fixed action order. Then state/action equality propagates pathwise and exact
paired accuracy $1/2$ follows.

## 3. Other findings

- P1-1: A3 is state-coordinate-matched, not exactly compute-matched; no FLOP
  or operation ledger exists.
- P1-2: every positive route needs independent trace and reward lesions, and
  reward update must reread the lesioned state rather than a cached cue.
- P1-3: classifier timing must be checked on all 32 training episodes, before
  reward at every intermediate step, not merely one episode.
- P1-4: a successor seal must bind every imported production dependency,
  including V17 factor-flow dependencies if imported.
- P1-5: dense-query labels and zero-margin rejection should use an integer
  Rademacher dot with a bounded redraw loop and fail-closed exhaustion.
- P1-6: homogeneous state must reset atomically after every reward update;
  45 independent SPD coordinates must be distinguished from 81 serialized
  matrix slots.
- P2-1: the public cue marker solves event selection. The surviving positive
  theorem is delayed memory plus binary reward decoding, not learned causal
  discovery.

## 4. Authorization boundary

No production code, evaluator, development run, manifest or confirmation seed
is authorized for this run. A successor may reuse the closed E1/E2/E4 proofs
but must use a fresh contract and fresh seed block with the repaired E3
quantifiers.
