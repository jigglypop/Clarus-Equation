# G9-CBM V2 lock-amendment audit

Status: COMPLETE

Gate: **PASS**

## Scope and meaning

This audit checks only the three post-PASS lock amendments in the latest
`00-contract-v2-draft.md`: C-order codec mapping, finite recall rejection
semantics, and the all-condition shadow lesion. No code, test, calibration,
development seed, or registered seed was run.

PASS means that implementation may continue under the frozen V2 contract. It is
not evidence that any performance, safety, validation, or test claim will pass.

## P0 closure

| Amendment | Gate | Audit |
|---|---|---|
| `vec_C` / reshape | **PASS** | `00-contract-v2-draft.md:227-244,318-330` fixes C-order flattening and `(12,8)` reconstruction for storage-derived standardization, inherited masked recall, schema construction, inverse-standardization, hashes, and recall scoring. The common `mu_codec/sigma_codec` bytes and array shapes are unchanged, so no factor, provenance, API, or budget difference is introduced. |
| finite sentinel / symbolic `REJECT_ALL` | **PASS** | `:568-590` exhaustively separates `scope=0` disabled/no-queryable LTM, `scope=1` valid 12-row scope, and `scope=2` invalid context/component. Scopes 0/2 serialize finite `-2.0`, false acceptance, and identity `-1`; invalid precheck precedes store availability, so cross-port diagnostics remain scope 2 in every factor cell. Scope 1 is finite in `[-1,1]`. Internal `-inf` is neither serialized nor used by candidate arithmetic. `:895-929` makes `REJECT_ALL` symbolic, outside confidence pools and artifacts; a winning `REJECT_ALL` stops calibration, while every operational threshold is finite. The existing float64 audit field and 72-call ledger remain unchanged. |
| all-condition shadow lesion | **PASS** | `:626-664` makes every cell/control reuse the same already-enumerated 288 indices for deterministic `240 -> 24` post-classification. It adds no pair enumeration, join, proposal, schema update, or dream pass; therefore the registered 288 enumerations, 48 endpoint values, 24 lesion slots, and `N_dream_passes=1` remain exact (`:988-1027`). The separate preallocated lesion buffer is hashed in every condition but cannot enter the 72-key schema/action index, rollout, selection, constrained provenance counts, or non-lesion scientific `invalid_splice_rate`. Only the registered lesion control exposes it to its diagnostic metric. Its occupancy/provenance codebook is fixed and does not create episodic identity or LTM writes. |

## Isolation checks

- **Factor isolation:** LTM still changes only queryable real recall and `q_hat`;
  dream still changes only accepted hypothetical missing-slot payloads. Sentinel
  and shadow audit bytes cannot affect either treatment path.
- **Provenance:** synthetic hard-zero invariants are unchanged. Lesion audit codes
  are buffer-local diagnostics, not `observed`, `recalled`, episode IDs, schema
  provenance, or LTM insertion attempts.
- **Budget:** no dtype, shape, owned array, call slot, enumeration, join, update,
  pass, or byte cap changes. The `(29,)` vector and exact `393,216`-byte payload
  remain applicable.
- **API/leakage:** `OriginRecallAuditV2` and `SeedRecallAuditV2` retain their fixed
  types and order. Diagnostic cues and lesion bytes do not enter
  `execute_candidate_v2`; all hashes precede evaluator access.
- **Claim boundary:** the amendments are definitions and integrity rules, not
  theorems or empirical outputs. C1--C4 remain untested predictions and C5 remains
  a reporting rule.

Final decision: **PASS**. The three additional P0 findings are closed without
contaminating factor attribution, provenance, budget equality, or the candidate
capability boundary. All original stop, lock, validation, and test-unlock rules
remain in force.
