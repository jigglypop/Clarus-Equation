# V9 sequential loop-engineering contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v9-runtime-integration-20260812

Mode: full

## Objective

Advance the V9 nested-SCC program through separately falsifiable loops without merging
mathematical closure, executable causality, task utility, confirmation, brain mapping, or AGI
into one claim.

## Loop ledger

| Loop | Claim | Entry status | Exit condition |
|---|---|---|---|
| L1 | nonzero cross-scale infinite tower has a controlled finite-prefix approximation | open because exact append-zero commutation is false | theorem with explicit invariant space, uniform $q$, tail defect, and rollout/fixed-point bound |
| L2 | Runtime-to-V9 action cascade is bounded and state-mediated | unit implementation exists | theorem plus tests that no external output bypass enters the V9 branch |
| L3 | nested levels causally help a two-timescale memory task | untested | preregistered benchmark, real lesions, matched controls, locked hashes |
| L4 | development result meets all registered gates | `0/256` | exactly one 256-seed development execution after pre-run audit |
| L5 | an untouched block confirms a development GO | not registered | run only if L4 is GO; otherwise SKIPPED/STOP |
| L6 | scale-indexed nested SCC is a viable whole-brain hypothesis | mathematical design only | explicit graph construction/data/intervention contract; no biological promotion without data |

## Mathematical objects

Let $w$ be finite shell width and

$$
X=\ell_\infty(\mathbb N_0;\mathbb R^w),
\qquad \lVert x\rVert_X=\sup_{\ell\ge0}\lVert x_\ell\rVert_\infty.
$$

The infinite previous-tick map uses the existing recurrence operator $R$, upward bridge
$U_\ell$, downward bridge $D_\ell$, level-zero input, and componentwise $\tanh$. The frozen
default bounds are

$$
\lVert R\rVert_\infty\le r,
\quad \lVert U_\ell\rVert_\infty\le u\lambda^{\ell+1},
\quad \lVert D_\ell\rVert_\infty\le d\lambda^{\ell+1},
\quad q=r+u+d<1.
$$

Exact append-zero compatibility with $u>0$ remains false. L1 is allowed to prove an
approximate infinite-tail theorem; it may not relabel the false exact theorem.

## Benchmark hypothesis

The development task contains an early slow cue and a later fast cue. The final four-way
action is their ordered pair. Current decision input alone is uninformative. Every arm receives
the same raw stream and frozen encoder.

Primary candidate: finite V9 tower through the public token-policy API.

Required controls: stateless, level-zero recurrent, upper-state reset, cross-scale cut, and a
same-state-size monolithic recurrent control. Parameter/storage/MAC labels must be separately
typed; the old manifest scalar metadata is forbidden as a matching statistic.

## Development and confirmation lock

- Development seeds: `0..255`.
- Confirmation seeds: `10000..10255`; forbidden to generate or inspect unless development is
  GO and a post-development confirmation audit authorizes it.
- No V8 locked split or ACBSM fresh block may be opened.
- Benchmark code, config, normalizer, arms, metrics, thresholds, and seed roles must be hashed
  in `artifacts/preregistration.json` before development.
- Only one development invocation is allowed. Failure is preserved as STOP.

Primary GO requires all of:

1. paired mean accuracy improvement over the strongest non-lesion comparator at least `0.02`;
2. two-sided 95% paired bootstrap lower bound greater than `0.0`;
3. upper-state reset loss at least `0.05`;
4. cross-scale-cut loss at least `0.05`;
5. no integrity, future-read, duplicate-seed, nonfinite, or arm-alias violation;
6. exact preregistration/hash match.

Any failed conjunction makes L4 STOP. No threshold may be edited after results exist.

## Authorized implementation

- Infinite-tail and cascade certificates with unit/property tests.
- A deterministic synthetic mechanism benchmark and its real control arms.
- Pre-registration and one development execution after a separate pre-run audit.
- Confirmation only under the conditional rule above.
- Canonical V9/CodeMap updates and code-level removal of newly identified dead metadata.

## Forbidden claims

Task success is not AGI. Synthetic results are not biological data. A scale-indexed graph
tower is not a nested partition into maximal SCCs of one fixed graph. No literal physical
infinity, consciousness, whole-brain identity, or universal generalization claim is authorized.
