# Research contract

Status: IN_PROGRESS

## Objective

Promote the R1 training-only parent-anchored shrinkage checkpoint into one
formally preregistered V8 validation/test workflow while preserving all
historical failures, successful engineering checkpoints, and unopened test
splits.

The candidate is fixed before any V8 outcome is generated:

\[
\widehat Y=P+g(S-P),
\qquad
g=\Pi_{[0,1]}
\frac{\sum_w\langle(S_w-P_w)/s,(Y_w-P_w)/s\rangle}
     {\sum_w\lVert(S_w-P_w)/s\rVert^2}.
\]

`S` is the frozen V5 sparse-parent H20 path, `P` is persistence, and the gain
must be recomputed from inherited observational-train seeds `45100..45107` at
origins `80,100,...,500`.  It must reproduce
`0.7868543064870357` within a locked tolerance.

## Frozen checkpoint stack

1. V4 pooled-AR one-step mechanism checkpoint.
2. V5 immutable-prefix, one-origin, uninterrupted H20 rollout contract.
3. V7 training-only normalization, paired seed inference, symmetric controls,
   leakage instrumentation, finite output, and locked-test discipline.
4. V7 positive sparse/no-sparse development subcheck, without promoting the
   failed V7 endpoint.
5. R1 fresh-development result on seeds `79100..79355`, which is development
   evidence only and may not enter V8 inference.

## Data roles

- inherited train/probe data: unchanged;
- disclosed development: all V1--V7 validation data and R1 seeds
  `79100..79355`;
- V8 validation: 256 fresh OOD seeds `80100..80355`;
- V8 locked test: 256 fresh OOD seeds `81100..81355`;
- independent unit: one simulation seed;
- one forecast origin at 80 and one H20 path per seed;
- H5 is the exact H20 prefix and is nongating.

All blocks must be pairwise disjoint.  No historical locked test split may be
read or reused.

## Primary validation conjunction

The candidate must satisfy every clause:

1. paired Student-t 95% lower improvement above zero versus the unshrunk V5
   parent;
2. paired Student-t 95% lower improvement above zero versus persistence;
3. paired Student-t 95% lower improvement above zero versus an identically
   fitted zero-bridge shrinkage control;
4. paired log-error-ratio 95% upper below `log(1.02)` versus an independently
   fitted same-probe dense shrinkage control;
5. finite outputs, maximum state read index 80, zero future reads, and exact
   H5/H20 identity;
6. sparse, zero-bridge, and symmetric-dense dynamic components at or below
   pathwise radius `0.98`, and every latent AR magnitude at or below `0.98`;
7. exact registration, implementation, test, parent-artifact, and gain locks.

Frozen V7 consensus and stable adaptive dense are reported as secondary
historical comparators.  They do not enter the candidate recursion.

## Test rule

The V8 test split may run only after one unchanged V8 validation artifact
satisfies the full conjunction and all registration/code/test hashes match.
No alternative V8 route, gain, window grid, seed block, margin, or gate may be
substituted after validation is observed.

## Claim boundary

A validation and locked-test pass may support only training-only
covariance-aware convex shrinkage and sparse-bridge contribution in this
fully observed, matched-basis, four-chart synthetic H20 family.  Symmetric
dense superiority is not claimed.  AGI, open-world causal discovery, brain
equivalence, consciousness, unseen-environment transfer, and physical CE
correspondence remain unsupported.

## Provenance gate

Before canonical V8 implementation or any V8 seed is generated, the exact V8
registration must be committed or independently timestamped without absorbing
unrelated worktree changes.  If this cannot be done safely, stop before V8
execution and report `BLOCKED`, rather than relabeling another development run
as confirmatory.

