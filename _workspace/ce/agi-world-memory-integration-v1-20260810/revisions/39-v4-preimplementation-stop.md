# G9-CBM V4 pre-implementation stop

Status: COMPLETE

Gate: BLOCKED_PRE_IMPLEMENTATION

## Decision

[산출] G9-CBM V4 is stopped before implementation lock. The frozen registration
and amendment are retained as a negative implementation-boundary witness; they
are not an empirical experiment result.

[산출] No V4 registered train/calibration, validation, or locked-test seed was
opened. No V4 implementation lock, intent marker, calibration artifact,
validation artifact, test artifact, or integrity artifact exists. Consequently
V4 supplies no evidence for or against C1--C5, AGI, long-term memory, dream-like
recombination, world prediction, or planning performance.

## Locked inputs

- V4 registration raw SHA-256:
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- V4 amendment raw SHA-256:
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- Recursive V2 -> V3 -> V4 canonical merge SHA-256:
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.

The original self-audit in `35-v4-prereg-audit.md` is preserved for provenance,
but its PASS disposition is superseded by the independent audits below.

## Complete counterexamples and unresolved P0s

[미완성] Independent audit found that the executable contract is not unique or
irreversible enough to authorize implementation:

1. The registered 10-source/5-dependency manifest conflicts with inherited
   8-source/3-dependency hash recipes.
2. Inherited recall fields still name nonexistent `scoped_hard_recall_v2`,
   while the callable manifest names V4/V3 wrappers.
3. Exact V4 candidate, control, arena, and registered-stage capability APIs are
   absent.
4. Deleting an uncommitted implementation-lock or scientific artifact can erase
   every specified opening witness and permit a second run.
5. Empty/partial hard-crash bytes, exception payload hashing, and per-origin
   feasibility failures do not have one total terminal schema.
6. The 29-row budget mixes event counters, owner measurements, and registered
   caps without one row-by-row observed-source recipe.
7. The allowed V3 call graph can allocate transient owners outside
   `TempArenaV4`, contradicting the claimed closed ownership boundary.
8. On Windows, two predecessor Markdown contracts materialize as CRLF and fail
   their locked raw hashes in the V3 structural loader.

The exact evidence and proposed closures are recorded in:

- `36-v4-independent-prereg-audit.md` (`Gate: REVISE`);
- `37-v4-executable-contract-redteam.md` (`Gate: REVISE`);
- `38-v4-byte-merge-audit.md` (`Gate: REVISE`).

## Next authorized step

[예측] A new V5 preregistration may close only these pre-implementation
boundaries, use new V5 paths and fresh disjoint seed roles, and retain the V4
scientific equations and gate thresholds unless an explicit change is declared.
V5 must pin all hash-locked contract Markdown to LF, replace every stale merged
field, define exact APIs and row provenance, and use a durable pre-RNG opening
witness that survives deletion and hard interruption.

V5 may not open a registered seed until its raw bytes, recursive merge, source
manifest, API, resource ownership, and state machine independently receive
`Gate: PASS`. V4 itself must never be resumed or reinterpreted as a failed
scientific run.
