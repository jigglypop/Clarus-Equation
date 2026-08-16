# G9-CBM V4 independent byte and recursive-merge audit

Status: COMPLETE

Gate: REVISE

Audit scope: frozen preregistration bytes, recursive V2 -> V3 -> V4 merge,
declared delete/add paths, role separation, allocation ledger, exact key sets,
registered paths, Git ignore/attribute behavior, and fresh-checkout hash
reproducibility. No registered seed, scientific world, calibration, validation,
test, V3 runner, or V4 runner was opened.

Audited worktree HEAD:
`ce628ce2ed3d7342ca27df4285db0937a26d6e04` on
`research/agi-world-memory-integration-v1`.

## 1. Frozen V4 byte evidence

[Observation] The actual restored files have these bytes:

- V4 registration: 30,676 bytes, raw SHA-256
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- V4 amendment: 15,468 bytes, raw SHA-256
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- V4 self-audit: 4,135 bytes, raw SHA-256
  `12a30ec6f6c2fb9317fd61b1276ae2d0dda4e4e82df20d1782fc945b6b252cdc`.
- V2 registration: 77,134 bytes, raw SHA-256
  `b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2`.
- V3 registration: 13,990 bytes, raw SHA-256
  `bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b`.

[Observation] V2, V3, and V4 registrations parse as strict finite JSON with
duplicate object keys rejected. The V4 registration, V4 amendment, and V4
self-audit are UTF-8 without BOM, contain no CR bytes, and end in exactly one
LF. All V4 internal references to the V3 registration, V3 amendment, and V4
amendment hashes equal their declared locked values when compared with the Git
blob bytes intended by the registration.

## 2. Recursive merge reproduction

[Derivation] Applying the registered algorithm independently gives five valid
V3 deletions and eighteen valid V3 additions, followed by the following exact
V4 deletions:

1. `preregistration_integrity / scientific_post_lock_change_requires_v4_fresh_seeds`
2. `data_roles / train / fresh_after_v2_block`
3. `data_roles / validation / fresh_after_v2_block`
4. `data_roles / test / fresh_after_v2_block`

[Derivation] The exact thirteen V4 additions encountered once each are:

1. `preregistration_integrity / predecessor_status`
2. `preregistration_integrity / v4_pilot_executed`
3. `preregistration_integrity / v4_static_content_audit_required`
4. `preregistration_integrity / scientific_post_lock_change_requires_v5_fresh_seeds`
5. `data_roles / train / fresh_after_v3_block`
6. `data_roles / validation / fresh_after_v3_block`
7. `data_roles / test / fresh_after_v3_block`
8. `test_lock / calibration_intent_path`
9. `test_lock / validation_intent_path`
10. `test_lock / test_intent_path`
11. `artifact_state_machine / stages / calibration_intent`
12. `artifact_state_machine / stages / validation_intent`
13. `artifact_state_machine / stages / test_intent`

[Derivation] No requested deletion was absent, no undeclared addition occurred,
no declared addition was unused, arrays were replaced rather than concatenated,
and V4 top-level records were assigned after override merge. Canonical JSON with
sorted keys, ASCII escaping, compact separators, and nonfinite values forbidden
is 75,088 bytes with SHA-256
`8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.

## 3. Roles, allocation, and exact key sets

[Observation] V4 roles are exactly train `95100..95139` (40), validation
`96100..96139` (40), and locked test `97100..97159` (60). They are pairwise
disjoint. A recursive integer scan of all 56 other JSON files under
`experiments/preregistration` parsed every file with duplicate-key rejection
and found zero collisions with the 140 V4 role integers.

[Derivation] The inherited allocation ledger contains 36 uniquely named
objects, sums to exactly 393,216 bytes, and its registered listed-order,
unsorted-object-key canonical payload hashes to
`7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4`.

[Observation] Exact-field arrays are internally unique and mechanically
consistent: all three intent stages use the same 11-field intent schema; the
opened-failure schema has 15 fields; a history record has 6 fields; handcrafted
test evidence has 14 fields; validation has 25 required output fields; and test
has those same 25 plus exactly `validation_raw_sha256` and `unlock_record`.
Unlock counts are exactly 55 performance + 17 hard-zero + 12 resource = 84.

## 4. Path and repository-state evidence

[Observation] The thirteen registered V4 paths are unique, normalized safe
repository-relative paths, and every stem carries `v4`. The ten ordered source
paths and twenty-two callable symbols are each unique; the ordered source list
is exactly five primary paths followed by five dependency paths. All five
dependency files exist, are tracked, clean, and have Git history. At this
preimplementation point only the V4 registration exists among the five V4
primary files; the module, runner, and two tests are correctly still absent.

[Observation] All eight registered V4 implementation-lock, intent, scientific,
and integrity artifact paths are absent, untracked, and have no history witness.
This is consistent with zero V4 role opening. The V4 registration is ignored by
`.gitignore:29:*.json`, so committing it requires an explicit force-add. The
working `experiments/preregistration/.gitattributes` contains exactly one
`agi_world_memory_integration_v4.json text eol=lf` rule.

## 5. P0: predecessor raw bytes are not reproducible in a fresh Windows checkout

[Observation] The current Git installation has `core.autocrlf=true`. Neither
of the two predecessor Markdown files hashed by
`load_merged_registration_v3` has a matching `text eol=lf` attribute. A fresh
checkout therefore materialized CRLF bytes:

- `00-contract-v2-draft.md`: working tree 59,999 bytes, 1,231 CR bytes, SHA-256
  `3077c391ce5e95b848d3ab0953d0e74a67b4837be890fa8aef91299dd91292e2`;
  locked Git-blob form 58,768 bytes, zero CR bytes, SHA-256
  `842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70`.
- `31-v3-boundary-amendment.md`: working tree 4,145 bytes, 96 CR bytes,
  SHA-256
  `b420c0bcda641baeeba6aad2c24373357c546f20e56704b557fe31bfb1b94a9c`;
  locked Git-blob form 4,049 bytes, zero CR bytes, SHA-256
  `9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340`.

[Observation] Calling only the read-only structural loader
`load_merged_registration_v3()` on this fresh checkout, without constructing a
world or RNG, terminates with
`PermissionError: base contract SHA256 changed`.

[Incomplete, P0] This blocks V4 implementation because V4 explicitly inherits
and invokes the V3 merged-registration loader. The same portability defect would
also affect the currently LF V4 amendment after it is committed and checked out
again unless that path is pinned.

Minimum repair before preregistration commit:

1. Add exact LF attributes for `00-contract-v2-draft.md`,
   `31-v3-boundary-amendment.md`, and
   `34-v4-implementation-closure-amendment.md` in the repository-root
   `.gitattributes`.
2. Refresh the tracked predecessor files from the index, or construct a second
   fresh worktree, and prove their actual `Path.read_bytes()` hashes equal the
   locked hashes above.
3. Re-run strict JSON, recursive merge, path/history, and loader checks without
   opening a registered role. Keep the V4 registration and amendment bytes
   unchanged if their hashes remain the values in section 1.

No other P0, P1, or P2 was found within this byte/merge audit scope.

## Verdict

Gate: REVISE

The frozen V4 registration content is mechanically self-consistent, but the
recursive predecessor byte chain is not executable from a normal fresh Windows
checkout. PASS is forbidden until the LF checkout invariant is committed and a
fresh-checkout loader proof succeeds.
