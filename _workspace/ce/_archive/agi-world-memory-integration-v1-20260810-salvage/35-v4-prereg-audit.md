# G9-CBM V4 preregistration static audit

Status: COMPLETE

Scope: static bytes, recursive merge, boundary completeness, role separation,
and mechanical consistency only. No registered seed, scientific world,
calibration, validation, test, V3 runner, or V4 runner was executed.

## Byte locks

- V4 amendment: 15,468 bytes; raw SHA-256
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- V4 registration: 30,676 bytes; raw SHA-256
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- Fully recursive V2 -> V3 -> V4 canonical merged payload: 75,088 bytes;
  SHA-256
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.
- V3 base registration raw SHA and V3 amendment raw SHA match the exact values
  declared by V4.
- Amendment and registration are UTF-8 without BOM, contain LF only, and end in
  exactly one LF. JSON parsing with duplicate-key rejection passed.

## Merge audit

- Exact segment-array deletions: 4 requested, 4 present, 4 applied.
- Allowed new override paths: 13 declared, 13 encountered exactly once as a
  set, no undeclared addition, and no unused allowance.
- Arrays replaced rather than concatenated.
- V4 top-level records were assigned only after the recursive override merge.
- The inherited allocation ledger retains its listed object-key order; its
  unsorted-key recipe yields
  `7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4`
  and its byte sum remains exactly 393,216.

## Role, path, and manifest audit

- Train: exactly `95100..95139` (40).
- Validation: exactly `96100..96139` (40).
- Locked test: exactly `97100..97159` (60).
- The roles are pairwise disjoint. A recursive integer scan of every other JSON
  file in `experiments/preregistration` found zero collisions.
- Thirteen registered V4 source/artifact/intent paths are unique and carry the
  V4 suffix; predecessor V3 paths are dependency-only and not reused as outputs.
- The ordered source manifest has 10 unique paths: five V4 primary paths, the
  committed V3 registration/module, and three G7-M dependencies.
- The callable manifest has 22 unique symbols: nine V4 wrappers/evaluator/
  runner/arena boundaries, twelve explicitly permitted V3 callables, and the
  inherited scoped recall boundary. V3 runner/stage/unlock callables are absent.

## Closure audit

- Shuffled binding fixes direction `i -> (i+1) mod 3`, completes all three
  accepted-or-fallback views before projection, uses origin `i`'s own schema
  anchor, and preserves origin `i`'s own audit.
- The lesion control rolls out exactly M00 and exposes only the common
  `21/24` lesion diagnostic.
- Zero-synthetic valid missing slots return source 3 and never source 2.
- Persistent arrays are actual `ConditionArenaV4` destinations; observed work
  is owned by `ActualCounterV4`; every nonpersistent owner is tracked and
  released by `TempArenaV4` under the inherited cap.
- Symbolic `REJECT_ALL` is an actual lexicographic calibration candidate and a
  winning symbolic candidate makes calibration infeasible.
- Every validation/test origin requires a generator-valid true `J<=25` witness
  and a distinct-side true `J>25` witness without using action lures.
- Mutating stages require their exact registered output path and expose no dry
  run. Scientific roles require committed intent, exclusive create, an fsynced
  pre-RNG nullable FAIL envelope, immutable finalization, deletion-history
  refusal, and exact fixed-order history records.
- Implementation-lock handcrafted evidence is structural command/hash/count/
  exit/stdout/stderr evidence; a literal success Boolean is forbidden.
- Test unlock recomputes exactly 55 performance, 17 hard-zero, and 12 resource
  checks from primitive/observed evidence and rehashes current sources,
  callables, NumPy, allocation, and the artifact chain before touching test
  intent.

## Verdict

**PASS for preregistration closure.** This does not predict or assert an
empirical PASS. Implementation lock remains forbidden until the V4 module,
runner, tests, and all manifested callable/path hashes exist and pass the exact
handcrafted evidence gate.
