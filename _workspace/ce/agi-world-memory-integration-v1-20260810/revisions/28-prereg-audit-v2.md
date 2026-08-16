# G9-CBM V2 preregistration reverse audit

Status: COMPLETE

Gate: **REVISE**

## Scope and evidence boundary

This is a static, field-by-field reverse comparison of
`revisions/00-contract-v2-draft.md` against
`experiments/preregistration/agi_world_memory_integration_v2.json`. It checks
scientific literals and formulas, codebooks, gates, registered seeds, the
29-field resource vector, the allocation ledger, and the artifact state
machine. No implementation module, runner, calibration, development seed,
registered train seed, validation seed, or locked-test seed was executed.

The audited immutable-byte snapshot was:

```text
contract bytes: 58,768
contract SHA-256: 842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70
preregistration bytes: 42,573
preregistration SHA-256: 92866c92dbfec1e577a75aae6e8272b0e1f304f145c4157d0745783faccf2554
```

The JSON's `preregistration_integrity.contract_raw_sha256` is byte-equal to the
actual contract SHA. Both files are UTF-8 without BOM, contain no CR/CRLF, and
have exactly one terminal LF. Strict parsing found no duplicate object key and
no `NaN`, `Infinity`, or other nonfinite JSON number.

## Static checks that pass

| Item | Result | Evidence |
|---|---|---|
| Contract identity | PASS | JSON lines 10--24 names the correct contract path and exact raw SHA. |
| Registered roles | PASS | Train `86100..86139` (40), validation `87100..87139` (40), and test `88100..88159` (60) are complete, internally unique, and pairwise disjoint. |
| Seed collision scan | PASS | No other preregistration seed-valued field contains any of the 140 V2 seeds. A repository word-boundary scan found no other registered declaration outside this research run. |
| Main dimensions/codebook values | PASS in part | `d=4`, `m=2`, `H=20`, `K=8`, `O`, `M`, `D`, `B`, `G`, action vectors, candidate order, stream IDs, and pair-reason numeric assignments agree where serialized. |
| 29-vector | PASS | Names, order, values, declared length 29, and uniqueness are exactly the contract vector at contract lines 1042--1071 / JSON lines 1295--1413. |
| Allocation arithmetic | PASS in part | The 28 declared ledger entries sum to `393,216`; the non-padding subtotal is `312,584`, padding is `80,632`, and the caps are `524,288` persistent / `1,048,576` temporary. Every entry with a real scalar NumPy dtype has correct shape-times-itemsize arithmetic. |
| Artifact paths | PASS | All ten paths at contract lines 1167--1176 are reproduced at JSON lines 1677--1687. |

These mechanical passes do not close the executable registration. The following
P0 items make the JSON non-equivalent to the contract and prevent an
implementation lock or registered-seed opening.

## P0-1 — the purported locked registration is ignored and untracked

`git check-ignore -v` resolves the new JSON to `.gitignore:29:*.json`. It is not
returned by `git ls-files`; only the new LF rule in
`experiments/preregistration/.gitattributes` is present as a tracked-file
modification. This conflicts with JSON `status="locked_pre_implementation"` and
with a state machine whose implementation lock and later validation/test chain
must name stable registration bytes. An ignored working-tree file can disappear
or change without repository provenance.

**Exact P0 fix:** before implementation lock, add
`!experiments/preregistration/agi_world_memory_integration_v2.json` after the
global `*.json` rule (or force-add the exact bytes), retain the LF attribute,
and require the registration path to be Git-tracked and byte-equal to the
approved preregistration SHA. Until all findings below are patched and this
audit is rerun, the JSON must not assert a completed mechanical registration
gate and train must remain closed.

## P0-2 — generator serialization is incomplete and one append order is contradictory

The contract defines the phase residual exactly at lines 99--112, the
episode-drift sign at lines 162--165, constant wake action `A[j]` at lines
224--230, and canonical order `c,p,i,j,h` with `(i,j)` taken once from displayed
`O`. The JSON only serializes phase row ranges; it never serializes
`P`, `C`, and `S+E` as the three values of `b`, never serializes
`q_episode=h*0.040*u_theta[c,p,i,j]`, and never states that each wake record uses
`A[j]` for all 12 transitions.

Worse, JSON lines 563--568 give both `"prefix"` and
`"displayed_observed_binding"` as independent append-order fields. Since an
observed binding already is `(i,j)`, literal nesting of that order produces
`2*4*3*6*2=288` records, contradicting both the required 96 and JSON's own
`records_per_seed=96`.

**Exact P0 fix:** serialize all of the following literals, without a second
prefix loop:

```text
b(c,p,i,j,phi) = P[c,p,i]              for phi 0..4
                 C[c,p]                for phi 5..6
                 S[c,p,j]+E[c,p,i,j]   for phi 7..11
q_episode = h*0.040*u_theta[c,p,i,j]
wake action rows 0..11 = A[j]
append order = context, port, displayed O binding (i,j), sign -1,+1
```

Also serialize the contract's exact common-noise key
`(seed,origin,lead)` across candidates/cells and its prohibition from candidate
inputs. These are generator literals, not implementation choices.

## P0-3 — the registered learner and planner equations are missing

Several outcome-determining equations exist only in the Markdown contract even
though JSON declares `standalone=true`:

- the only LTM estimator
  `q_hat=mean(completed_view_raw[:,0:4]-schema_anchor_raw,axis=0)`
  (contract lines 346--358);
- the recursive candidate transition
  `x_hat[t+1]=f_hat(x_hat[t],a[t])+q_hat+s_hat_raw[...]`
  and lead-row convention (lines 420--435);
- the normalized 20-lead state/action cost, including both divisors `20*4` and
  `20*2` (lines 517--542);
- `k_hat`, `k_opt`, true regret, success, the `r>=-1e-12` rule, serialization of
  tiny negative regret as zero, and the pre-candidate feasible/infeasible-goal
  assertions (lines 544--557).

JSON has the core predictor, goal rule, three cost literals, and a prose regret
label, but those do not determine the omitted arithmetic. Different candidate
implementations can therefore conform to the JSON while returning different
predictions, costs, selections, and gates.

**Exact P0 fix:** add machine-readable formula fields for the four bullets
above, including exact array indices, normalization, invalid-cost override,
lexicographic tie rule, and true-vs-inferred cost ownership. Add the exact raw
schema inverse and state phase selection to the rollout formula. The JSON must
state explicitly that all eight candidates recurse for all 20 leads and that
invalid-token padding uses numeric action plus component fallback while
returning key `-1` and source `0`.

## P0-4 — metric and all-of gate names do not freeze their formulas

JSON lines 1028--1180 preserve most thresholds and denominators, but omit
outcome-determining definitions required by contract lines 736--932:

1. `E_recall` has no formula, no selected-ledger target, no same-standardizer
   rule, and no rejection-as-raw-mean fallback despite being used by the
   no-antagonism gate.
2. The invalid-predicted-transition numerator omits the exact nonfinite, bound,
   inferred-valid, key-range, and key/source/context/component/phase/action
   audit union from contract lines 776--783.
3. Named relative-reduction thresholds do not serialize exact `RR_L`, the two
   matched dream contrasts, `RR_joint`, M11-vs-persistence, or `RR_regret`
   formulas. This leaves ratio aggregation ambiguous despite the separate
   `ratios_use_cell_means` flag.
4. `no_antagonism` gives a margin and an upper-CI threshold but not the exact
   paired vectors `E_recall11-1.02*E_recall10` and
   `E_uv11-1.02*E_uv01`.
5. Attribution labels do not freeze the exact H20 metric and positive paired
   difference used in each of the four contract expressions.
6. The mandatory maximum-seed lure-rate report and exhaustive per-metric
   interaction/sign/tie serialization are absent from the report requirements.
7. Join calibration does not serialize the exact two residual-only statistics
   `R[4]-R[5]` and `R[6]-R[7]`, the four-coordinate RMS, or the non-unique
   selector hard failure.

**Exact P0 fix:** copy these exact formulas into structured metric, calibration,
and gate fields; name H20/H5 explicitly for every contrast; and add an ordered
`required_reports` list. Validation and test must each recompute every Boolean
from the serialized primitive seed vectors and independently satisfy the same
all-of mapping. A threshold name alone is not an executable formula.

## P0-5 — codebooks and the 240-to-24 lesion traversal are under-specified

The numeric assignments that are present agree, but the JSON drops the hard
domain and transition rules that make them auditable:

- pair reasons do not say `10..255` invalid or that left rejection wins when
  both joins fail;
- schema sources do not say `4..255` invalid; schema keys omit `int16`, exact
  valid range `0..71`, and unresolved `-1` as the only negative value;
- recall scopes omit `3..255` invalid and the scope-0/2 invariants
  `accepted=false`, `identity=-1`, finite confidence `-2.0`; physical identity
  range `0..95` is absent;
- dream-output provenance `0=empty/rejected, 1=synthetic_hypothetical` and
  lesion provenance `0=empty, 1=valid_missing, 2=invalid_cross_port,
  3..255 invalid` are absent;
- JSON records only lesion counts. It does not freeze traversal by
  learner-visible context, 12 prefix slots, and 12 suffix slots in
  first-occurrence order, rejection of the 48 observed keys, then selection of
  the first 24 of the remaining 240. Thus the asserted `3/21` composition is not
  derivable from the JSON.

The candidate API also lists field names and shapes without the contract's exact
scalar/array dtypes for results and audits. That prevents static enforcement of
the above codebooks.

**Exact P0 fix:** add each valid domain, invalid range, rejection priority,
sentinel invariant, field dtype, and the exact canonical lesion pseudocode.
Require the computed 288-index traversal hash, the first-24 selected-index hash,
and the `3 valid / 21 invalid` typed provenance counts to match in every
condition; non-lesion conditions must retain the bytes but exclude them from the
scientific invalid-splice metric.

## P0-6 — three allocation entries are not executable NumPy dtypes

The arithmetic total is correct, but JSON ledger entries
`uint8_or_bool8`, `bool8_int16_float64_uint8`, and `bool8_uint8` are prose, not
NumPy dtypes. In particular, a four-field structured audit may have padding and
an implementation-dependent itemsize, so `(72)` plus `864 bytes` does not freeze
the same payload or allocation SHA. This contradicts the exact
shape/dtype/name-order requirement at contract lines 1086--1130.

**Exact P0 fix:** replace the three composite rows by real ordered arrays while
preserving the byte total:

```text
dream_occupancy       (24) bool8     24
dream_provenance      (24) uint8     24
lesion_occupancy      (24) bool8     24
lesion_provenance     (24) uint8     24

recall_accepted       (72) bool8     72
recall_identity       (72) int16    144
recall_confidence     (72) float64  576
recall_scope          (72) uint8     72

pair_check_flags     (288) bool8    288
pair_reason_codes    (288) uint8    288
```

The subtotal remains `312,584`, padding remains `80,632`, and total remains
`393,216`; update the ordered allocation-ledger SHA over these real entries.
Alternatively an explicit structured dtype must freeze names, offsets,
endianness, `align=false`, and itemsize, but separate arrays match the contract's
semantic ownership more directly.

## P0-7 — the artifact state machine is reduced to booleans

JSON lines 1677--1698 reproduce paths and several flags, but do not serialize
the exact transition and payload contract at contract lines 1179--1199. Missing
requirements include:

- the implementation-lock manifest for registration, implementation module,
  runner, both tests, inherited G7-M V2/V1 dependencies, callable/codec-strip
  boundaries, NumPy version, and ordered allocation SHA;
- exact stage preconditions, artifact-absence/no-overwrite rules, and the rule
  that deleting an artifact never authorizes a second registered run;
- calibration's mandatory core, normalizer, threshold, selector-pool/count, and
  dependency hash fields;
- validation/test's complete Boolean mappings and the recomputable identity
  `passed = performance_passed && integrity_passed && resource_passed`;
- exact `UnlockRecordV2` fields: validation raw SHA, registration,
  implementation, calibration and dependency SHAs, plus
  `test_unlocked=true`;
- integrity/verify as read-only stages that generate no registered world and do
  not mutate scientific artifacts.

The generic field
`exact_registration_implementation_calibration_dependency_hashes_required=true`
cannot prove which hashes are required or that the same bytes are reused.

**Exact P0 fix:** add an explicit ordered `artifact_state_machine` with
`preconditions`, `seed_role`, `run_count`, `input_hash_fields`,
`required_output_fields`, `no_overwrite`, and `postcondition` for
preregistration, implementation lock, calibration, validation, test, integrity,
and verify. Require the registration itself to be tracked/clean before the
implementation lock; require a committed clean HEAD-identical validation plus
recomputed component mappings before construction of the exact in-memory unlock
record.

## Final decision

**REVISE.** The byte transport, contract SHA, seed partition, 29-vector values,
allocation sum, and artifact paths are sound, but they are insufficient for an
executable lock. The untracked registration, contradictory wake append order,
missing generator/learner/planner/metric equations, incomplete codebooks and
lesion traversal, non-dtype allocation rows, and under-specified artifact state
machine are P0 blockers.

Do not create the implementation lock and do not open `86100..86139`. Patch the
registration only from the already-frozen contract (no empirical choice), make
its repository provenance explicit, recompute its raw SHA, and repeat this
static reverse audit. Any change to the contract's scientific design rather
than a faithful serialization repair requires the contract's V3/fresh-seed
rule.
