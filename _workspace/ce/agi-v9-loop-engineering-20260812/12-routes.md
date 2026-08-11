# Route and benchmark design

Status: COMPLETE

## Route comparison

| Route | Core idea | New DOF | Decisive failure | Rank |
|---|---|---:|---|---:|
| A | exact append-zero direct limit | 0 | already refuted for nonzero upward coupling | rejected |
| B | weighted infinite $\ell_\infty$ operator plus uniform tail residual | 0 | $q\ge1$, $\lambda\ge1$, or numerical residual exceeds theorem | 1 |
| C | learned quotient/lumpability map | many | quotient defect or target-aware fitting | deferred |
| D | structural zero upward gain | 0 | removes the cross-scale mechanism under study | negative control |

Route B is selected because it preserves nonzero bidirectional cross-scale coupling without
reviving the false exact theorem.

## Two-timescale benchmark

Each episode has four actions representing an ordered pair `(slow_bit, fast_bit)`. At the early
tick the observation is one of

```text
slow 0: (+1,+1,-1,-1)
slow 1: (-1,-1,+1,+1)
```

and at the later tick it is one of

```text
fast 0: (+1,-1,+1,-1)
fast 1: (-1,+1,-1,+1)
```

All other ticks contain registered zero-mean finite noise. The decision tick contains no label
cue. The target index is `2*slow_bit + fast_bit`. Development uses registered delays and noise;
confirmation, if opened, uses longer untouched delays.

## Arms

| Arm | State | Purpose |
|---|---|---|
| `v9` | full finite tower, public token policy | candidate |
| `stateless` | current evidence only | proves present-input insufficiency |
| `level0` | one recurrent shell | tests whether one timescale is enough |
| `upper_reset` | full tower with upper levels reset before decision | real state lesion |
| `cross_cut` | full state count with cross-level messages structurally zero | real cross-scale lesion |
| `monolithic` | same scalar state count in one recurrent vector | topology control |

Storage scalar count and executed multiply-add estimates are reported separately. Neither is
called trainable parameter count or capacity.

## Anti-bypass and look-elsewhere

Candidate decisions must equal the public token-policy output. Candidate code cannot call the
target function after episode construction. The four cue templates, task mapping, arm list,
delays, thresholds, 256 seed roles, and paired bootstrap procedure are frozen before results.
There is one primary hypothesis and no post-result route selection.

## Whole-brain route

L6 uses a family $G_s$ indexed by registered scale/estimator $s$, with typed node maps and
directed edges. It must report maximal SCCs within each $G_s$, never call spectral communities
SCCs, and test cross-scale causal predictions through perturbations. Existing public evidence
does not execute this route, so its outcome in this run is design-only.
