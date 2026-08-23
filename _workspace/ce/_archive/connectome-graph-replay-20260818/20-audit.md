# Formal status audit — C. elegans structural-connectome replay MVP

Status: COMPLETE

Gate: PASS

## Audit boundary and decision

This audit compares the completed contract and the three completed lanes; it does not rederive their numerical results. The source object, mathematical definitions, and alternative route are mutually usable after the claim narrowing and schema decisions below. The gate authorizes only the implementation envelope frozen here. It is not evidence that the implementation claims have already passed.

There is no open P0. `P0-MATH-01` is closed by selecting R1 with the exact scope stated below. The broader interpretation that arbitrary reordering of the frozen raw CSV must retain the same artifact is removed: such a byte change necessarily changes the C1 authority object, its hash, and its source ordinals. The surviving C5 theorem concerns only a permutation of already parsed observations that retain immutable frozen-source ordinals.

The absence of an explicit repository license is a real limitation, not permission. It does not prevent a local, user-supplied-byte replay gate, but it prohibits this run from claiming redistribution rights. The upstream CSV and the full derived replay artifact must remain run-local; the committed fixture must be wholly synthetic.

## Source and contract consistency

- `00-contract.md:9-15` limits the target to a frozen adult-hermaphrodite structural materialization and makes the exact included population input-defined. This is consistent with `10-sources.md:11-17,21-23`: the immutable OpenWorm commit and byte object are identified, while 302 neurons and rounded literature-scale counts are contextual rather than acceptance constants.
- `00-contract.md:19-25` separates source observations, chemical direction, unordered electrical endpoints, aggregation, and canonical bytes. `11-math.md:7-50` supplies a conditional algebra for those objects. The one ambiguity is the meaning of reciprocal electrical weights; it is closed here by labeling their aggregate strictly as an **exported-row weight sum**, not a physical gap-junction count.
- `00-contract.md:31-37` registers C1-C7. The source lane supplies the immutable transport facts (`10-sources.md:11-17`); the math lane supplies conditional preservation and canonicalization results (`11-math.md:27-60`); implementation and replay evidence are intentionally still prospective.
- `11-math.md:70,76` correctly identifies the ordinal/order conflict. `12-routes.md:13-21` gives a zero-free-choice resolution under R1 and explicitly rejects its use for identity-free raw-file permutations.
- `10-sources.md:16,23` does not establish a license for the selected connectivity bytes. The OCP licensing statement in `10-sources.md:14` applies only as corroboration and is not transferred to the selected OpenWorm byte object.

## R1 resolution of P0-MATH-01

R1 is selected and closes `P0-MATH-01` under all of the following rules:

1. `source_record_ordinal` is the zero-based logical CSV data-record position following the single header in the exact hash-verified byte object.
2. For a complete parsed object with $N$ accepted observations, ordinals must be exactly the set $\{0,\ldots,N-1\}$. Duplicate, missing, negative, Boolean, or out-of-range ordinals are rejected.
3. The parser assigns the ordinal once. Canonicalization and aggregation must preserve it; they must never renumber an input container.
4. The C5 permutation fixture shuffles the parsed observation container while carrying each ordinal with its observation. Both permutations must yield identical canonical values, bytes, and SHA-256.
5. A raw CSV row permutation is outside C5. It produces different frozen bytes and must fail C1 against the registered hash before CSV decoding or parsing.

Accepted C5 statement: **Permuting the parsed observation container while retaining each frozen-source ordinal leaves canonical bytes and SHA-256 unchanged.**

Deleted C5 interpretation: **Arbitrarily permuting the rows of the frozen source file without retained identities leaves replay authority and bytes unchanged.** This interpretation has a direct counterexample from the changed raw SHA-256 and changed R1 ordinals.

## Claim dispositions

| Claim | Current stated status | Actual formal status at this gate | Evidence and dependency | Disposition |
|---|---|---|---|---|
| C1 | Frozen bytes are the only authority. | [공리: 외부 입력] plus [예측: transport check]. | `00-contract.md:15,31,42-43,49-50`; immutable commit, URL, byte length, and SHA-256 in `10-sources.md:11-12`. | Coherent. Implementation must hash raw bytes before source decode/parse. Reuse status is “not established,” not “openly redistributable.” |
| C2 | Parsing is deterministic and fail-closed. | [정의] plus [미완성: 구현·검사]. | Closed mathematical domain in `11-math.md:7-25`; rejection evidence is not yet produced. | Eligible for implementation only with the exact parser profile and negative fixtures below. |
| C3 | Replay preserves every accepted observation and provenance. | [정리: 조건부 보존/합 보존] plus [예측: full replay]. | Invariants in `11-math.md:27-50`; source row vector in `10-sources.md:17`. | Conditional proof is sound. Full-source row equality, ordinal resolution, conservation, canonical digest, and repeated bytes remain implementation evidence. |
| C4 | Chemical direction and electrical unordered normalization match source semantics. | [공리: source field meaning] plus [산출: endpoint normalization], with one semantic restriction. | `10-sources.md:21`; normalization in `11-math.md:15-39`. | Accept. Electrical `released_weight_sum` is only the sum of released export rows. It must not be described as a biological gap-junction count; released orientations remain visible per observation. |
| C5 | Canonical output is independent of input row order and repeated execution. | [정리] only in the selected R1 domain; broader raw-byte statement deleted. | Conditional theorem in `11-math.md:48-50`, counterexample in `11-math.md:76`, route in `12-routes.md:13,17-21`. | P0 closed by the five R1 rules above. Fixture and repeated full replay must supply implementation evidence. |
| C6 | Full-release replay equals a registered exact count vector. | [예측]. | Exact frozen-byte metrics in `10-sources.md:12,17`; comparison rule in `11-math.md:52-60`. | The registered vector is frozen below. Rounded 302/2,990/890 context values cannot pass or fail C6. Fixture-only results cannot discharge C6. |
| C7 | Artifact is structural, not a functional brain simulation. | [정의: scope restriction] plus [미완성: output label]. | `00-contract.md:13,37`; `11-math.md:72,79`. | Output metadata must contain the exact structural-only label below; no dynamics, activity, behavior, learning, or human-connectome claim is authorized. |

## Frozen implementation envelope

### Exact repository paths

Implementation may create or modify only these product/test paths, in addition to the required CE stage files `30-implementation.md` and `31-validation.md`:

- `.gitignore` — add the single exception for the registered JSON manifest.
- `experiments/preregistration/c_elegans_connectome_replay_v1.json` — immutable source/parser/expected-count manifest.
- `reality_stone/python/reality_stone/clarus/connectome_replay.py` — standard-library parsing, validation, canonicalization, serialization, and digest library.
- `examples/brain/c_elegans_connectome_replay.py` — offline CLI with required `--manifest`, `--source`, and `--output` arguments.
- `tests/fixtures/c_elegans_connectome_tiny.csv` — wholly synthetic fixture; no row copied from the unresolved-license source.
- `tests/test_connectome_replay.py` — the sole focused test target.

The full source stays at `_workspace/ce/connectome-graph-replay-20260818/artifacts/herm_full_edgelist.csv`. The full canonical artifact stays at `_workspace/ce/connectome-graph-replay-20260818/artifacts/c_elegans_connectome_replay.full.json`. Neither is a repository redistribution artifact.

### Manifest v1 fields

The manifest is UTF-8 JSON and contains exactly these semantic groups; naming may not be weakened or repurposed:

- `schema_version`: integer `1` with `type(value) is int`.
- `dataset_id`: `openworm_celegansneuroml_herm_full_edgelist`.
- `scope`: `adult_hermaphrodite_structural_graph_only`.
- `source`: immutable `url`, `commit`, repository `path`, `byte_length`, and lowercase `sha256`; `redistribution_permission` is `not_established`; `repository_handling` is `manifest_only_raw_and_full_output_run_local`.
- `population`: `all_normalized_endpoint_identifiers_in_frozen_file_including_non_neuron_cells`.
- `parser`: exact header array `['Source','Target','Weight','Type']`; strict UTF-8; BOM forbidden; closed classes `chemical` and `electrical`; endpoint normalization removes only leading/trailing ASCII SPACE (`0x20`) bytes and then requires a nonempty ASCII alphanumeric identifier; weight grammar is `0|[1-9][0-9]*`; self-loops are accepted and preserved; ordinal policy is the selected R1 rule; duplicate identity means duplicate ordinal, while equal-value observations with distinct ordinals remain distinct.
- `electrical_weight_semantics`: `sum_of_released_row_weights_not_physical_gap_junction_count`.
- `expected_source_metrics`: the complete registered vector below.

The selected source constants are:

- `url`: `https://raw.githubusercontent.com/openworm/CElegansNeuroML/b36380a36d2a6dda0f03c946c433524b25ea2268/herm_full_edgelist.csv`
- `commit`: `b36380a36d2a6dda0f03c946c433524b25ea2268`
- `path`: `herm_full_edgelist.csv`
- `byte_length`: `252842`
- `sha256`: `0ab9baab5f404895b8dbeb8daa453c86e8f342961bc458cd19bf1b5f6a38d859`

The implementation must read raw bytes, compare byte length and SHA-256, and only then perform strict UTF-8 decoding and CSV parsing. A source-hash mismatch must therefore win over every possible schema error in the same byte object.

### Parser and observation rules

- The header must match the four registered column names and order exactly; no extra or missing column is accepted.
- Identifier padding normalization removes all leading/trailing ASCII SPACE bytes only. Generic Unicode `strip`, case folding, Unicode normalization, and internal-whitespace removal are forbidden. The raw hash and ordinal remain the exact provenance anchor.
- `Weight` is parsed only from the canonical nonnegative decimal grammar. Signs, decimal points, exponents, whitespace after field parsing, empty values, and numeric coercion are rejected.
- A programmatic observation accepts a weight/ordinal only when `type(value) is int`; `bool` is rejected.
- Chemical observations retain released `source_id -> target_id` direction. Electrical observations retain released `source_id,target_id` orientation in the observation table while setting normalized `endpoint_a=min(source_id,target_id)` and `endpoint_b=max(source_id,target_id)`.
- All 48 source self-loop observations are accepted and preserved. Exact-value duplicate rows would also be preserved if their immutable ordinals differed; a repeated ordinal is rejected.

### Canonical output v1

The top-level object has exactly `format`, `metadata`, `nodes`, `observations`, `connections`, and `summary`:

- `format`: `clarus.connectome_structural_replay.v1`.
- `metadata`: `dataset_id`, `scope='structural_graph_only'`, `source_url`, `source_commit`, `source_path`, `source_byte_length`, `source_sha256`, `redistribution_permission='not_established'`, `population`, and `electrical_weight_semantics`. It contains no timestamp, host path, or other run-dependent value.
- `nodes`: objects `{'id': identifier}` sorted by identifier.
- `observations`: objects with `connection_class`, normalized `endpoint_a`, normalized `endpoint_b`, released-orientation `source_id`, released-orientation `target_id`, `released_weight`, and `source_record_ordinal`; sorted by `(connection_class, endpoint_a, endpoint_b, source_record_ordinal)`.
- `connections`: objects with `connection_class`, `endpoint_a`, `endpoint_b`, `released_observation_count`, `released_weight_sum`, and ascending `source_record_ordinals`; sorted by `(connection_class, endpoint_a, endpoint_b)`. For electrical records the sum is an export aggregate only, and the referenced observations expose reciprocal rows and unequal reciprocal weights.
- `summary`: `node_count`, `canonical_observation_count`, `aggregate_connection_count`, `chemical_observation_count`, `chemical_released_weight_sum`, `electrical_observation_count`, `electrical_released_weight_sum`, `total_released_weight_sum`, `self_loop_observation_count`, `electrical_unordered_pair_count`, `electrical_reciprocal_two_row_pair_count`, `electrical_unequal_reciprocal_weight_pair_count`, `electrical_max_observations_per_pair`, and `exact_duplicate_released_row_count`.

Canonical bytes are produced with recursive key sorting, compact JSON separators, `ensure_ascii=False`, non-finite values forbidden, UTF-8 encoding, and exactly one appended LF. Arrays use the order above. The artifact SHA-256 is computed over those exact bytes.

### Registered full-source count vector

The C6 vector is exact and is interpreted after the registered ASCII-space endpoint normalization:

| Component | Expected integer | Meaning |
|---|---:|---|
| `source_byte_length` | 252842 | Raw authority-object bytes; transport check, not a graph count. |
| `source_observation_count` | 7379 | CSV data records after the header. |
| `normalized_endpoint_identifier_count` | 448 | Union of normalized released source/target identifiers; includes non-neuron cells. |
| `self_loop_observation_count` | 48 | Accepted source observations with equal normalized endpoints. |
| `chemical_observation_count` | 4681 | Released chemical rows. |
| `chemical_released_weight_sum` | 27019 | Sum over released chemical row weights. |
| `electrical_observation_count` | 2698 | Released electrical rows before unordered grouping. |
| `electrical_released_weight_sum` | 12683 | Sum over released electrical row weights; not a physical pair count. |
| `total_released_weight_sum` | 39702 | Exact arithmetic sum of the two class weight sums. |
| `electrical_unordered_pair_count` | 1359 | Distinct normalized electrical endpoint pairs. |
| `electrical_reciprocal_two_row_pair_count` | 1339 | Pairs represented by two released orientations. |
| `electrical_unequal_reciprocal_weight_pair_count` | 13 | Reciprocal two-row pairs whose released weights differ. |
| `electrical_max_observations_per_pair` | 2 | Maximum released rows in one normalized electrical pair. |
| `exact_duplicate_released_row_count` | 0 | Equal `(Source,Target,Weight,Type)` released rows, independent of ordinal. |

`aggregate_connection_count` and the first full canonical-output SHA-256 are output evidence, not silently invented preregistered source facts. They must be recorded in `31-validation.md`, then reproduced by a second execution. Conservation requires `canonical_observation_count=7379`, total connection `released_observation_count=7379`, and total connection `released_weight_sum=39702`.

## Validation distinction and required evidence

The focused fixture test is necessary but not a full connectome replay. It must remain offline and cover:

- source-hash failure before source decode/parse;
- malformed UTF-8, wrong header/arity, empty endpoints, non-ASCII/internal-whitespace identifiers, invalid class, invalid decimal weight, negative/nonintegral/Boolean programmatic weights, and duplicate/missing ordinal rejection;
- chemical reversal remaining distinct;
- electrical endpoint normalization with equal and unequal reciprocal weights, explicit released orientations, and no biological-count label;
- self-loop preservation, aggregation conservation, provenance resolution, R1 parsed-container permutation invariance, repeated canonical bytes, and terminal-LF/digest stability.

C1 and C6 require a separate full frozen-byte execution using the local artifact and registered manifest. The full gate must verify raw byte length/hash, every registered vector component, referential integrity, exact conservation, canonical byte generation, and byte-identical/digest-identical repetition. Passing only `tests/test_connectome_replay.py` must be reported as “fixture validation,” never “full-connectome reproduction.”

No network access is allowed during either validation. Acquisition remains a separate documented action.

## Findings by severity

### P0

- `P0-MATH-01` (C5) — **CLOSED by R1 in this audit.** Scope and rejection rule are frozen above. No open P0 remains.

### P1

- `P1-AUDIT-01` (C4) — **CLOSED by schema restriction.** Reciprocal electrical rows are preserved and their aggregate is named `released_weight_sum`; it is not a physical gap-junction count. Any implementation field or report that calls 12,683 a junction count reopens this item and requires `ce-impl-engineer` revision.
- `P1-AUDIT-02` (C1) — **SCOPED LIMITATION, not a permission finding.** No explicit license for the selected OpenWorm connectivity byte object is established. Raw bytes and the full derived graph stay run-local; only metadata, code, and a synthetic fixture may be committed. A future redistribution claim requires `ce-physics-sourcer` revision with source-specific permission evidence.
- `P1-AUDIT-03` (C2, C3, C5, C6) — **EXPECTED IMPLEMENTATION EVIDENCE.** Parser rejection, full replay, count-vector equality, provenance, conservation, canonical digest, and repetition are not yet established. This is the work authorized after the gate, not a defect requiring a pre-implementation lane revision.
- `P1-MATH-01` from `11-math.md:77` — **SUPERSEDED by the subsequently completed source lane.** `10-sources.md:11-17` now supplies the frozen release, byte hash, and registered exact vector. The license limitation remains separately scoped by `P1-AUDIT-02`.
- `P1-MATH-02` from `11-math.md:78` — **TRANSFERRED to implementation validation** with the exact fixture list above.

### P2

- `P2-AUDIT-01` (C7) — Output and final report must preserve `structural_graph_only`; neither graph connectivity nor deterministic serialization supplies dynamics or behavior.
- `P2-AUDIT-02` (C6) — 302 neurons and approximate 2,990/890 published-scale values remain contextual only because their population/layer/count conventions differ or are rounded. They cannot be promoted to exact acceptance values.

## Concrete revision scopes

No pre-implementation lane revision is required. If implementation violates the frozen envelope, revision ownership is limited as follows:

| Trigger | Role | Exact revision scope |
|---|---|---|
| Raw-file row permutation is again claimed invariant, or ordinals are regenerated after shuffle. | `ce-math-verifier` | Reopen C5 only; either restore R1 semantics or replace it with R2/R3 and revise the corresponding test claim. |
| Electrical aggregate is presented as a physical junction count or released orientations disappear. | `ce-impl-engineer` | Rename/restructure the aggregate and restore per-observation orientation/provenance; do not change source evidence. |
| Redistribution or open-license status is claimed. | `ce-physics-sourcer` | Supply source-object-specific license/permission evidence; an unrelated OCP license or README morphology statement is insufficient. |
| Full expected counts differ after a verified raw hash match. | `ce-impl-engineer` first, then `ce-physics-sourcer` only if independent parse evidence contradicts S7. | Inspect parser normalization and metric definitions without editing expected values to fit the implementation. |
| Fixture passes but full frozen-byte replay is absent. | `ce-impl-engineer` | Run the registered offline full command and record exact counts/digests; do not relabel the fixture as full replay. |

## Audit census

- Registered claims inspected: 7 (`C1`-`C7`).
- Conditional mathematical results retained: 3 (observation preservation/conservation, endpoint normalization separation, R1 canonical-order theorem).
- Previously hidden choices made explicit: 5 (ASCII-space identifier normalization, self-loop acceptance, immutable logical-row identity, duplicate-ordinal semantics, electrical export-sum semantics).
- Open implementation-evidence classes: 4 (manifest/hash transport, parser/negative fixtures, full replay/conservation/digest, structural-only metadata).
- Parent claims deleted: 0.
- Overbroad claim interpretations deleted: 1 (identity-free raw-file row permutation under C5).
- Complete counterexamples affecting the retained narrow claims: 0.

The claim statuses are therefore internally consistent: the mathematical results are conditional, full-release equality remains a prediction until execution, the licensing status is an explicit limitation, and the gate has no open P0.
