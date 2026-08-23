# G1 implementation record

Status: COMPLETE

The G1 harness is isolated in
`reality_stone/python/reality_stone/clarus/runtime_metric_intervention.py`, with its CLI in
`runtime_metric_intervention_benchmark.py` and focused tests in
`tests/test_runtime_metric_intervention.py`.

Every circuit constructs a seed-specific background recurrent matrix and a seed-specific disjoint
`S/T/N` coordinate permutation. Every arm forks the same serialized reset snapshot. Treatment and
scrambled writes use the public bounded native recurrent-delta boundary, after which dense/CSR
parity and the complete applied matrix difference are audited. Calibration and held-out probes each
restore their own arm snapshot, so signs, axes, and noise histories cannot contaminate one another.

The first executable smoke check exposed a disconnected input apparatus under the runtime default
active threshold. Before the development range was executed, the contract and code froze the
analytically justified `active_threshold=0.04`, added a driven-coordinate-active assertion to every
pulse, and tightened frozen-configuration, edge-support, transform-byte, and applied-delta gates.
An independent post-repair audit returned PASS.

The CLI records all circuit rows, calibration trajectories, held-out endpoints, first-passage
times, SPD matrices, transform audits, source hashes, runtime versions, and a canonical results
hash. Confirmation mode requires a separate hash-bound freeze manifest whose development verdict
is `GO`; no such manifest is created for a stopped route.

## G2 implementation

G2 is isolated in `runtime_metric_sufficiency.py` with a separate benchmark CLI and focused test
file. A default-false `BrainRuntimeConfig.force_all_active_selection` switch was added at the shared
active-selection seam; every predecessor retains the legacy path, while G2 explicitly sets it true.
The old threshold-zero smoke block was retired after it exposed both mask failures and losses.

The G2 harness builds independent calibration, fit, and test rollouts with collision-free native
noise intervals, never changes W, constructs horizon-wise `B_h`, `C`, `g`, and raw endpoint
precision, and fits the frozen 6/7/12/21-coefficient predictor family. It audits non-aliased
`C-to-inverse-to-g` prediction identity, covariant metric-feature equality, every active mask,
coefficient counts, and the circuit-level worst-adversary statistic.

## G3-D implementation

G3-D is isolated in `runtime_metric_memory_diagnostic.py` with its own benchmark CLI and focused
tests. It duplicates the untouched frozen M1 body only to retain post-learning W snapshots; a full
48-dimensional excluded-seed test proves exact predecessor report, final-W hash, and continuous
recall parity. Every response and recall probe restores a fresh, physically zero-store snapshot.

The structural control installs an explicit `P W P^T` on a fresh zero-gate branch. The null-lesion
bank is selected using only the six-horizon calibration response stack; recall is evaluated once,
only after selection. The first development run exposed an over-strict intended-delta audit at the
float32 addition boundary and is quarantined as apparatus-invalid. The repaired path first freezes
`W_target=fl32(W+d*)`, derives the native-representable actual delta, uses fixed `.250001` numerical
headroom, and audits the final target in float64. Inspected seeds `97701..97716` are blocked at all
public execution APIs; replacement development uses `97801..97816`.

Confirmation cannot be reached through single/range helpers. The stage API parses a hash-bound
development artifact, revalidates exact seed rows and source hashes, recomputes the summary, and
requires a genuine `DIAGNOSTIC_PASS` before opening confirmation.
