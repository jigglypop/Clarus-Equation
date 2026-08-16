# G9-CBM V4 transport-only fresh-checkout recheck

Status: COMPLETE

Gate: PASS_FOR_TRANSPORT_ONLY

This recheck addresses only the CRLF portability counterexample in
`38-v4-byte-merge-audit.md`. It does not change the V4 registration or
amendment, does not clear the executable-contract P0s, and does not authorize
implementation or any registered seed.

[산출] Commit `3dbd723582f3ea6ffeaf495d729039e5e7899cb2` adds LF attributes for the
G9-CBM run Markdown and preserves the frozen V4 evidence. A new detached
Windows worktree was created from exactly that commit under
`core.autocrlf=true`.

[산출] In that fresh checkout, all three hash-locked contracts materialized as
`w/lf` and matched their registered raw SHA-256 values:

- `00-contract-v2-draft.md`:
  `842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70`;
- `31-v3-boundary-amendment.md`:
  `9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340`;
- `34-v4-implementation-closure-amendment.md`:
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.

[산출] The read-only structural call
`load_merged_registration_v3(Path('experiments/preregistration/agi_world_memory_integration_v3.json'))`
then completed and returned the V3 experiment with 40 train-role seeds. It
constructed no registered world or RNG and opened no scientific role.

[미완성] Overall V4 remains `BLOCKED_PRE_IMPLEMENTATION` under
`39-v4-preimplementation-stop.md`. The manifest, stale recall boundary, typed
API, durable-opening, failure-totalization, and resource-owner P0s must be
closed only in a fresh V5 registration before implementation may begin.
