# G9-CBM V3 preregistration reverse audit

Status: COMPLETE

Gate: PASS

Audit type: independent static content and boundary audit. No registered seed,
generator, learner, rollout, calibration, validation, or test code was run.

## Locked byte witnesses

| Object | Raw SHA-256 | Bytes |
|---|---:|---:|
| `experiments/preregistration/agi_world_memory_integration_v3.json` | `bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b` | 13,990 |
| `experiments/preregistration/agi_world_memory_integration_v2.json` | `b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2` | 77,134 |
| `revisions/31-v3-boundary-amendment.md` | `9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340` | 4,049 |
| `revisions/00-contract-v2-draft.md` | `842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70` | 58,768 |

All four declared SHA relations match the raw files. The V3 registration and
amendment are UTF-8 without BOM, LF-only, and have exactly one terminal LF.

## Deterministic merge reproduction

I independently implemented the declared merge literally:

1. deep-copy the raw-SHA-verified V2 object;
2. interpret every delete path as an array of exact object-key segments;
3. delete in listed order, failing on a missing segment or leaf;
4. recursively merge objects while replacing arrays and scalars;
5. allow a new key only when its exact segment array is registered; and
6. assign every V3 top-level key except `overrides`, then serialize with the
   registered sorted-key, ASCII-escaped, compact JSON recipe.

The reproduction consumed all 18 allowed additions exactly once, performed all
5 deletions exactly once, encountered no unknown override, and produced:

```text
merged_registration_sha256 = 37e7bfb6ee100c47164bec49f2e151234a647964839189ba47bf504552e1644b
canonical_merged_bytes       = 62491
```

The segment-array representation removes the former ambiguity caused by dots
inside literal dtype keys such as `PartialCue.cue_values`.

## V2 disposition and zero-execution witness

The V2 file in the worktree is byte-identical to Git commit
`407a9714483ffd417660d9fe24db83b96f162301`: both resolve to Git blob
`7021699d08c52b0beb12ddac433a8f2a7d105cd7`. That commit contains the V2
registration but no V2 world-memory implementation, runner, unit test,
integrity test, implementation-lock artifact, calibration artifact, validation
artifact, test artifact, or integrity artifact. Those nine exact paths are also
absent from the current worktree and have no matching commit in this branch.

This filesystem/Git evidence is consistent with, and materially supports, the
registered predecessor counters: implementation lock `false`, train `0`,
validation `0`, and test `0`. V2 therefore remains a blocked, unexecuted
negative audit witness rather than a consumed scientific split.

## Fresh V3 roles

The arrays are exact consecutive ranges and their declared counts match:

- train: `92100..92139`, 40 seeds;
- validation: `93100..93139`, 40 seeds;
- locked test: `94100..94159`, 60 seeds.

All 140 values are unique, the three roles are pairwise disjoint, and no value
overlaps V2's `86100..86139`, `87100..87139`, or `88100..88159`. A recursive
seed-field scan of the other 68 preregistration/artifact JSON files found no V3
seed collision. The merged state-machine roles contain only the V3 ranges; no
old registered seed value survives.

## Actual inherited cue boundary

Static source inspection confirms that
`episodic_ltm_dream_bridge.PartialCue` declares, in order:

```text
context_token, prefix_token, suffix_token, cue_values, cue_mask
```

`episodic_ltm_dream_bridge_v2.PartialCue` is a direct alias of that class. The
merged registration has exactly this order, records `cue_values` as
`float64[12,8]` and `cue_mask` as `bool[12,8]`, and contains neither obsolete
`PartialCue.raw_codec` nor obsolete `PartialCue.mask`.

The paired lure boundary is outcome-complete and deterministic: stream 11 makes
one positive mask per canonical origin; its lure reuses that exact mask and
makes no second stream-11 draw; stream 12 visits visible cells in ascending
global C order and consumes 24 positive Normals immediately followed by 24
fresh lure Normals; cross-port diagnosis reuses the complete noisy positive cue
byte-for-value and changes only `suffix_token`. Lure state/fingerprint/drift
construction remains inherited from stream 13 and neither lure data nor cue
enters the 96-record ledger.

## Paths, callables, and inherited gates

Every operational world-memory primary path and every implementation-lock,
calibration, validation, test, and integrity artifact path has suffix `v3`.
The first five callable boundaries name V3 symbols; the sixth intentionally
names the inherited, byte-locked
`episodic_ltm_dream_bridge_v2.hard_cue_anchored_recall`. The only merged
world-memory V2 path references are provenance metadata (`extends`,
`supersedes`, and `base_registration_path`). The inherited G7-M/V2 dependency
paths are intentionally retained and hash-locked, not stale V3 outputs.

Canonical comparison shows byte-for-meaning equality for all 23 untouched V2
scientific/gate subtrees, including generator, learned core, R1 rollout,
factorial design, controls, metrics, paired inference, all-of gate, calibration,
provenance, claim boundary, and resources. Apart from the registered V3
preregistration pre/postconditions and the three fresh seed-role strings, the
artifact state machine is identical to V2. All no-overwrite, append-only run
count, deletion-never-authorizes-rerun, committed-validation unlock, three-way
pass identity, read-only integrity, and read-only verify rules therefore remain
active.

The inherited resource ledger also remains exact: 29 unique budget fields,
allocation total `393216`, persistent cap `524288`, temporary cap `1048576`,
metadata cap `32768`, and allocation-ledger SHA-256
`7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4`.

## Gate decision and remaining procedural stop

The frozen V3 content passes this static reverse audit. This PASS does **not**
open any registered seed. At audit time the V3 JSON is still ignored by the
repository-wide `*.json` rule and the amendment/audit are not yet committed.
Before implementation lock, the exact audited V3 JSON must be force-added, the
V3 JSON, amendment, and this audit must be committed and clean at their paths,
and the V3 JSON/amendment must be byte-identical to `HEAD`. Any change to the
audited V3 or amendment bytes invalidates this PASS; any outcome-determining
change after registration requires V4 with fresh seeds.
