# V8 canonical implementation plan

Status: READ-ONLY DESIGN COMPLETE

## Outcome

The minimal canonical promotion is four new source files plus the eventual
evidence artifact.  Historical V1--V7 code, registrations, tests, and
artifacts remain byte-unchanged.

1. `experiments/preregistration/sparse_causal_bridge_v8.json`
2. `reality_stone/python/reality_stone/clarus/parent_anchored_rollout_bridge.py`
3. `tests/test_parent_anchored_rollout_bridge.py`
4. `examples/agi/parent_anchored_rollout_bridge_gate.py`
5. after the registration provenance gate and implementation checks only,
   `artifacts/agi/sparse_causal_bridge_validation_v8.json`

The example runner should remain the repository-standard seven-line shim:
import the canonical module's `main` and exit with its return code.  No V8
logic belongs in the example or research workspace.

The new module should import the frozen machinery from
`reliability_rollout_bridge`, `free_rollout_bridge`,
`latent_causal_bridge`, and `sparse_causal_bridge`.  It should not modify the
V7 module or copy its simulation and fitting internals.

## Required chronology

This ordering is a hard implementation precondition.

1. Write the exact V8 registration, including seed blocks, candidate, all
   controls, gates, expected gains, tolerances, and filenames covered by the
   lock.
2. Commit or independently timestamp that registration alone.  Record the
   provenance identifier in the run log.  Do not simulate any seed in
   `80100..80355` or `81100..81355` before this point.
3. Add the canonical module, tests, and thin example runner.
4. Hard-code in the V8 test the registration file SHA-256, merged-chain
   SHA-256 returned by `_load_registration`, and canonical merged-object
   SHA-256.  Run development-only unit tests using inherited disclosed seed
   `76999`, never a V8 evidence seed.
5. Run validation once and save atomically.  A failed artifact is preserved.
6. Run the locked test only if the unchanged validation artifact passes every
   registered clause and all current locks equal those stored in it.

Implementation must stop before step 5 if the provenance gate cannot be
established safely.

## Registration contract

V8 should extend V7 to preserve the complete inherited environment and
mechanism chain, but override the active runner, models, validation/test
roles, controller, gate, resource limits, and claim boundary.  Recommended
identifiers are:

- `schema_version: 8`
- `experiment: sparse_causal_bridge_v8`
- `runner: parent_anchored_shrinkage_confirmation`
- `active_gate: confirmation_gate`
- validation seeds `80100..80355`, exactly 256;
- locked test seeds `81100..81355`, exactly 256;
- origin `80`, horizon `20`, 100 transitions per evaluation episode;
- two-sided 95% Student-t critical value for df 255:
  `1.9693105698498752`;
- dense noninferiority log-margin `log(1.02) =
  0.01980262729617973`.

Registration validation must additionally scan every V1--V8 registration
file and reject overlap of either V8 evidence block with any historical role.
The normal merged V8 object no longer contains V7's overridden validation and
test arrays, so `base._validate_registration()` alone cannot prove this
historical non-overlap.

The disclosed R1 development seeds `79100..79355` and inherited unit-test seed
`76999` must also be declared non-evidence and disjoint from both V8 blocks.
No V7 locked test seed may be read merely to perform this check; reading the
registration arrays is sufficient.

## Frozen algorithm and context

Add a frozen `V8TrainingContext` containing:

- the V5 sparse mechanism and pooled scalar residual AR;
- the same-probe dense mechanism and its independently pooled residual AR;
- a zero-bridge mechanism and its independently pooled residual AR;
- the three independently fitted scalar gains;
- V7 training-only scales and normalized train-state q99;
- frozen V4 parent and V5 failure reports and raw hashes;
- optional V7 failure-artifact provenance hash for disclosure, not fitting.

The zero-bridge control is precisely the sparse parent's local coefficient
array with a zero cross-chart bridge matrix and an empty edge set.  It is a
no-cross-chart-bridge control, not a no-dynamics control.

Implement one shared gain fitter with an input surface limited to training
state arrays, a mechanism, its AR, scales, registered origins, and horizon.
It must enumerate observational-train seeds `45100..45107`, origins
`80,100,...,500`, exactly 22 disjoint H20 target blocks per episode and 176
blocks total.  Pool every normalized chart/lead/window coordinate before the
single division, then clip once to `[0,1]`:

`g = clip(sum(d*r) / sum(d*d), 0, 1)`.

Reject a nonpositive or nonfinite denominator and any nonfinite gain.  The
following values and absolute tolerance `1e-15` should be registered and
reproduced before evaluation:

- sparse: `0.7868543064870357`;
- same-probe dense: `0.7835668486813699`;
- zero bridge: `0.882857758971467`.

The predictor API should be exactly
`predict_from_prefix(prefix_states, context, registration)`.  It must accept
an immutable `(81, 4)` prefix and no episode, truth, future, hidden, outcome,
or evaluation-fit object.  For horizon 20 it materializes:

- `S`: frozen V5 sparse rollout;
- `P`: 20 copies of `prefix[-1]`;
- candidate `P + g_sparse * (S - P)`;
- unshrunk V5 sparse parent;
- persistence;
- same-probe dense shrinkage using `g_dense`, never `g_sparse`;
- zero-bridge shrinkage using `g_zero`, never `g_sparse`;
- stable adaptive dense and frozen V7 sparse/no-sparse consensuses as
  secondary reported comparators only.

The candidate path is postprocessed once and is never recursively fed into a
mechanism.  H5 is obtained only as `prediction_h20[:5]`; there must be no H5
rollout call or H5 gate.

## Exact primary controls and inference

For each seed, compute one normalized H20 path RMSE over 20 leads and four
charts using training-only scales.  The seed, not a lead or chart, is the
independent unit.  Candidate pass requires the conjunction of:

1. paired Student-t 95% lower improvement strictly above zero versus the
   unshrunk V5 sparse parent;
2. the same strictly above zero versus persistence;
3. the same strictly above zero versus independently fitted zero-bridge
   shrinkage;
4. paired log-RMSE-ratio 95% upper at or below `log(1.02)` versus independently
   fitted same-probe dense shrinkage;
5. all registered integrity, leakage, finiteness, stability, and lock clauses.

The stable adaptive dense and both frozen V7 consensuses are secondary.  They
must not enter candidate recursion, select the gain, or decide the primary
pass.  In particular, the adaptive model's known radius excursions must be
reported separately and must not be folded into the retained-component
stability maximum.

Use strict `> 0` for the three superiority lower endpoints and `<=` for the
registered noninferiority/stability upper bounds.  Persist the entire 256
seed vectors so paired calculations are reproducible.

## Leakage and integrity instrumentation

Reuse `PrefixReader` and persist, for each split:

- maximum observed state index, required `<= 80`;
- total future observation reads, required `0`;
- nonfinite prediction count, required `0`;
- exact model shapes `(20, 4)`;
- exact H5/H20-prefix identity;
- component and candidate gains and fit-window count;
- historical and disclosed-development seed overlap, required empty.

The unit test must poison `x[81:101]` by a large deterministic offset and
poison all hidden states, then require bit-identical component paths,
candidate/control paths, and gains.  It must also inspect the predictor
signature and reject future-bearing parameters.  Hidden data should never be
placed in `V8TrainingContext`; a shape-only hidden test is insufficient.

Record a coordinatewise convex-envelope diagnostic for each shrinkage output:
every value must lie between its persistence and learned-parent endpoint up
to a registered floating tolerance.  This is the correct safety property of
R1; it is not a claim that the output itself defines a recursive Jacobian.

## Stability instrumentation

Persist per seed, per component, not merely one aggregate maximum:

- sparse mechanism pathwise maximum Jacobian radius;
- same-probe dense mechanism pathwise maximum Jacobian radius;
- zero-bridge mechanism pathwise maximum Jacobian radius;
- adaptive comparator radius as a nongating diagnostic;
- sparse, dense, and zero-bridge pooled residual AR magnitudes.

The primary stability gate applies to the sparse, dense, and zero-bridge
recursive components and requires each pathwise radius and AR magnitude to be
`<= 0.98`.  Candidate and control outputs are nonrecursive path blends, so
report their convex-envelope checks separately.  Do not average radii across
seeds and do not let the secondary adaptive comparator close an otherwise
valid retained-component gate.

As a stronger audit field, record the frozen sparse augmented common-norm
certificate (`0.96786` upper bound under the documented fixed weighted norm)
and its prerequisites.  Keep it distinct from the empirical pathwise spectral
radius; do not recompute or transfer this certificate to dense or adaptive
models without a matching proof.

## Hash-lock bundle

At validation start compute and at validation end recompute an identical lock
bundle.  Abort rather than save if it changed during the run.  Store:

- V8 registration file raw SHA-256;
- full merged registration-chain SHA-256;
- canonical merged-object SHA-256;
- implementation SHA-256 map for
  `parent_anchored_rollout_bridge.py`, `reliability_rollout_bridge.py`,
  `free_rollout_bridge.py`, `latent_causal_bridge.py`, and
  `sparse_causal_bridge.py`;
- test SHA-256 for `test_parent_anchored_rollout_bridge.py`;
- frozen V4 parent artifact and V5 failure artifact raw SHA-256;
- disclosed V7 failure artifact and R1 pilot implementation/provenance hashes;
- all three recomputed gains, expected gains, tolerance, origins, and exact
  count 176.

The source/test filename sets themselves must be fixed by registration.  A
missing file is a hard error, not the string `MISSING` in an otherwise usable
lock.

## Test unlock and artifact behavior

Use fixed paths:

- validation:
  `artifacts/agi/sparse_causal_bridge_validation_v8.json`;
- test: `artifacts/agi/sparse_causal_bridge_test_v8.json`.

Before simulating even one test episode, `_assert_test_unlocked` must require:

- the canonical validation artifact exists and parses;
- experiment is V8, split is `validation`, and `passed is True`;
- every registered primary check is present and true;
- validation registration raw/merged/canonical hashes equal current values;
- current implementation and test hash maps exactly equal validation maps;
- current parent-artifact hashes and recomputed gain-lock bundle exactly equal
  validation values;
- the validation artifact's own raw SHA-256 is captured into the test report.

Any mismatch raises `PermissionError`.  There is no `--force`, alternate
unlock artifact, relaxed margin, or second V8 route.  A failed validation is
saved and leaves test locked.

The CLI should write atomically and refuse to overwrite an existing canonical
validation or test artifact.  Avoid a canonical `--no-save` path that exposes
the evidence block without preserving its result.  A noncanonical `--output`
must not be allowed for evidence execution.  The library function may return
a report for tests, but unit tests must use only disclosed development seed
`76999`, never either V8 block.

## Focused tests

The new test file should cover:

1. exact registration raw/merged/canonical hashes, provenance status, roles,
   critical value, and historical/disclosed seed disjointness;
2. exact parent hashes, training scales, 176-window enumeration, all three
   gains and their independent fit;
3. restricted prediction signature and immutable prefix shape;
4. exact candidate formula, distinct control gains, zero bridge matrix/edge
   set, model names and `(20, 4)` shapes;
5. future and hidden poisoning bit identity and `PrefixReader` counters;
6. H5 as the exact H20 slice and nongating status;
7. finiteness, convex envelopes, per-component radii and AR limits on seed
   `76999`;
8. complete implementation/test hash filename sets and no missing hashes;
9. negative unlock cases for absent, failed, registration-tampered,
   source-tampered, test-tampered, parent-tampered, and gain-tampered
   validation artifacts;
10. a synthetic positive unlock artifact proving the success path without
    simulating a test seed;
11. canonical single-write/atomic artifact behavior.

Estimated focused regression command:

```powershell
$env:PYTHONPATH='reality_stone/python'
.\.venv\Scripts\python.exe -m pytest tests\test_parent_anchored_rollout_bridge.py tests\test_reliability_rollout_bridge.py tests\test_free_rollout_bridge.py tests\test_latent_causal_bridge.py tests\test_sparse_causal_bridge.py tests\test_dimensionless.py -q --basetemp .pytest_tmp_agi_v8_confirm
```

Based on the existing 54-test run (`8.05 s`) and the fresh 256-seed pilot
(`9.1 s` observed during this audit), the focused unit/regression suite should
remain roughly 10--15 seconds because it must not execute either full V8
evidence block.  A single canonical 256-seed validation should be budgeted at
approximately 10--20 seconds on the current machine; the locked test has the
same order of cost.

## Pitfalls that must be blocked

- Extending V7 and checking only the merged object silently loses visibility
  of overridden historical validation/test seeds; scan raw historical roles.
- Reusing the sparse gain for dense or zero controls makes the controls
  asymmetric.
- Fitting any gain on the evaluation prefix backtest changes R1 into a
  target-adaptive controller and invalidates the checkpoint.
- Treating the trajectory blend as a recursive map creates an invalid
  Jacobian argument.  Gate components and audit the output envelope.
- Aggregating stability into one maximum repeats V7's attribution failure.
- Including adaptive dense in the primary stability maximum would let a known
  secondary instability veto the registered R1 mechanism.
- A hidden array of inert zeros inside the predictor API is unnecessary and
  weakens the leakage boundary; hidden should be absent.
- Running H5 separately permits a second trajectory; slice H20 only.
- `--no-save`, overwrite, or alternate-artifact flags can expose or replace a
  confirmatory block without preserving the registered endpoint.
- Hashing only the new module misses inherited executable dependencies;
  hashing a `MISSING` sentinel is not a lock.
- Capturing hashes only after evaluation leaves a time-of-check/time-of-use
  gap; compare start and end bundles.
- Re-running a failed validation, changing a margin/gain/window grid, or
  opening test after a partial pass violates the one-route rule.
- Passing validation supports only the narrow synthetic H20 shrinkage claim;
  dense superiority, AGI, and open-world causal discovery remain unsupported.

