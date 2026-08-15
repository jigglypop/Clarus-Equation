# V9 nested infinite-SCC isolated unit implementation

Status: COMPLETE

## Scope and formal status

This report records mathematical/software quality validation of the isolated
finite generator and controller authorized by `20-audit.md`. It is not security
work, a development result, V9 confirmation, a biological model validation, or
an AGI result. No code was changed during this independent validation.

The implementation consists only of the following five reviewed files:

- `reality_stone/python/reality_stone/clarus/nested_scc_tower.py`;
- `reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py`;
- `tests/test_nested_scc_tower.py`;
- `tests/test_adaptive_scc_tower_controller.py`;
- `examples/agi/nested_scc_tower_demo.py`.

There is no default runtime adapter, development runner, registration, result
artifact, historical-parent adapter, or output blend in this unit.

## Frozen source identities

The raw checkout-byte SHA-256 values independently recomputed on 2026-08-12
(Asia/Seoul) are:

| File | SHA-256 |
| --- | --- |
| `nested_scc_tower.py` | `18A13966CBBE69F244D686D7F5C7DC58A2D1A8F20057BD79F8DCE12B01138F81` |
| `adaptive_scc_tower_controller.py` | `9204DDDBF893A0C15DC34DE503E1E9C853A14FAAD164FA5AFB0F32BB1822E028` |
| `test_nested_scc_tower.py` | `18FAC6E512D928CD8E50DD58BCE3977D3812452EE37373DB0ADE1E5059CB032A` |
| `test_adaptive_scc_tower_controller.py` | `7D7D3FBC04E441E7F33711C5B1D0026E9FB2EADE24142D01FB62B23BCFF877ED` |
| `nested_scc_tower_demo.py` | `595D518A1BB1B5689054451F7891E7A98BC5785905AB4560F4D9A28747967896` |

Exact recomputation command:

```powershell
Get-FileHash -Algorithm SHA256 'reality_stone\python\reality_stone\clarus\nested_scc_tower.py','reality_stone\python\reality_stone\clarus\adaptive_scc_tower_controller.py','tests\test_nested_scc_tower.py','tests\test_adaptive_scc_tower_controller.py','examples\agi\nested_scc_tower_demo.py'
```

All five values match the expected independent-validation boundary.

## Graph and mathematical implementation

The generator constructs finite bidirected path shells and reciprocal bridges.
Every prefix is nonempty and strongly connected; each later prefix has a strict
vertex superset and an edge superset. The independent SCC partition therefore
returns one maximal SCC for the larger fixed graph, while lower prefixes remain
proper strongly connected subgraphs rather than nested distinct maximal SCCs.

The complete ideal-union predecessor rule has maximum in-degree five. Finite
backward causal-cone queries return a finite node set, a conservative cardinality
bound, the maximum birth depth, and a predecessor-completeness manifest hash.
The positive-delay event unroll advances the tick on every edge and is reported
as acyclic with singleton SCCs; it is not conflated with the recurrent template.

The declared update schedule is exactly `previous_tick_jacobi`. In the global
coordinate sup norm, the implementation uses

```text
q = recurrence_gain + upward_gain + downward_gain.
```

The within-shell row sums are one, cross-level gains decay from their registered
upper bounds, and `tanh` is one-Lipschitz. Thus `q` is a depth-independent bound
for the actual scheduled update. Certification requires finite `q`, `q < 1`,
and `q <= contraction_cap`; a schedule mismatch or non-strict bound is refused.
The default deterministic demo reports `q = 0.54` against cap `0.95`.

Exact inclusion is deliberately narrower than topology or contraction. The
append-zero embedding is exact on the invariant zero-state/zero-input singleton.
It is exact on the declared unit cube only when upward boundary coupling is
structurally zero. Every nonzero upward gain is refused as generic exact
compatibility, including tested tiny positive gains; the controller grows
conservatively until the finite maximum and reports exhaustion without claiming
a truncation or infinite-horizon error certificate.

## Controller implementation

The controller accepts an exact `CausalEvent`, normalizes observations by frozen
positive reference scales, uses finite normalized recurrent state, and commits
an observation only after validation, update, trace, and token construction all
succeed. Reset, observation, and snapshot-load rejection paths leave prior state
unchanged.

Forecast and policy readouts require the latest immutable `TowerStateToken`.
The token binds controller identity, episode generation, tick, active depth,
parameter hash, state, and delayed-message state. The forecast API accepts only
the token. The policy API additionally accepts only a feasibility mask. Raw
events, targets, analytic posteriors, persistence, and V5/V8/ACBSM outputs have
no readout parameter or cached bypass.

The generator specification, generated operator arrays, manifest, parameter
hash, controller-bound generator identity, and operator identity are sealed and
revalidated at public boundaries. Returned operator/state arrays are copies.

Snapshots carry an HMAC integrity tag under a random process-local key. This is
explicitly same-process provenance and exact continuation, not external
authentication or cross-process persistence. A valid snapshot restores state,
active depth, tick, delayed up/down messages, episode generation, current token,
depth decision, and a pending intervention. A pending-intervention continuation
is reproduced bitwise; contradictory, forged, foreign-parameter, nonfinite,
inactive-hidden-state, or malformed snapshots fail before commit.

All six registered intervention families mutate the next consumed update data:

- `LevelReset` zeroes the selected active level before message generation;
- `CutUp` zeroes the selected consumed upward message;
- `CutDown` zeroes the selected consumed downward message;
- `TimeShift` consumes the registered one-tick-old up/down message;
- `SignFlip` negates the selected consumed message direction;
- `StateShuffle` applies a full, hash-bound shell permutation before update.

Intervention arms own distinct state/message arrays and leave the source arm
unchanged. Indices, directions, delay, permutation type, and permutation hash are
revalidated when scheduled and again before consumption or snapshot restoration.

## Dimensionless boundary

Graph reachability, SCC membership, birth levels, ticks, depths, and counts are
combinatorial. Observations enter the numerical core only after division by named
positive finite `observation_scales`. Recurrent, input, upward, downward, decay,
contraction, and tolerance gains are finite dimensionless scalars; normalized
states are constrained to `[-1,1]`. The contraction factor `q`, cap, state
differences, compatibility defects, and readout values therefore share the
declared dimensionless normalized state convention. Booleans, encoded numeric
text, nonpositive scales, nonfinite values, overflowed normalization/drives, and
noncanonical schedule/type fields fail closed.

This dimensional consistency establishes software typing only. It does not
establish physical validity, truncation accuracy, empirical utility, or a limit
system realized by a brain.

## Preserved execution boundary

No V8 locked test, ACBSM reserved block, V9 evidence/development data, historical
result data, or seed block was opened. No runtime integration was added or
exercised. The implementation remains an isolated deterministic finite unit
fixture. Development execution and every confirmatory, biological, and AGI claim
remain blocked exactly as stated in `20-audit.md`.

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
