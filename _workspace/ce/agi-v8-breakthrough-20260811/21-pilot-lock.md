# Fresh development pilot lock

Status: LOCKED BEFORE EXECUTION

This is a development checkpoint, not a confirmatory registration.  It does
not open or reuse any V1--V7 locked test split.

## Candidate

\[
\widehat Y=P+g(S-P),
\qquad
g=\Pi_{[0,1]}
\frac{\sum_w\langle(S_w-P_w)/s,(Y_w-P_w)/s\rangle}
     {\sum_w\lVert(S_w-P_w)/s\rVert^2}.
\]

- `S`: frozen V5 sparse-parent H20 path.
- `P`: persistence path from the last observed state.
- Fit data: inherited observational-train seeds `45100..45107` only.
- Fit origins: `80,100,...,500`, exactly 22 nonoverlapping H20 target
  windows per seed and 176 windows total.
- Locked sparse gain: `0.7868543064870357`.
- Candidate output is postprocessed once and never recursively fed back.

## Fresh block

- environment: inherited synthetic `ood`
- seeds: `79100..79355`
- independent units: 256 seeds
- historical registered-seed overlap: zero, checked before lock
- origin: 80
- horizon: 20
- observed input: `x[0:81]`
- target: `x[81:101]`

## Fixed controls

1. unshrunk V5 sparse parent;
2. persistence;
3. same-probe dense parent with its own identically fitted scalar gain;
4. zero-bridge parent with its own pooled residual AR and scalar gain;
5. stable adaptive dense as an external comparator only;
6. frozen V7 sparse and no-sparse consensuses as historical comparators.

## Development checkpoint clauses

- paired Student-t 95% lower improvement above zero versus V5;
- paired Student-t 95% lower improvement above zero versus persistence;
- positive paired lower improvement versus zero-bridge shrinkage if a bridge
  contribution is discussed;
- paired log-ratio 95% upper below `log(1.02)` versus symmetric dense;
- sparse mechanism pathwise radius at most `0.98` and latent AR magnitude at
  most `0.98`;
- all outputs finite, maximum observed index 80, and zero future reads.

Failure of any clause is preserved.  No alternate gain, window grid, route, or
seed block may replace this pilot after execution.

## Implementation lock

- `fresh_parent_anchor_pilot.py` SHA-256:
  `b3e07dec5895e670fc4babd1dbd261a2fc9795de90f1e576ba69f76fc2de0a41`
- Student-t critical value: `1.9693105698498752` (`df=255`, two-sided 95%).

