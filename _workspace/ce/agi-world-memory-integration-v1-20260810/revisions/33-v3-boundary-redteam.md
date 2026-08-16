# G9-CBM V3 boundary red-team

Status: COMPLETE

Gate: PASS

## Scope and locked bytes

This is a read-only red-team of the V3 boundary amendment, not a new audit of
the inherited V2 scientific design. No registered/development seed and no
scientific implementation code was executed.

- V3 registration raw SHA-256:
  `bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b`
  (`13,990` bytes, LF-only, one terminal LF).
- Amendment raw SHA-256:
  `9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340`.
- Audited amendment:
  `revisions/31-v3-boundary-amendment.md`.
- Audited inherited boundary:
  `reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py`
  and its V2 alias module.

## P0 findings

None.

## Boundary checks

| Check | Gate | Evidence and reasoning |
|---|---|---|
| Paired-mask lure construction | PASS | Amendment lines 29--45 give each canonical origin one independently drawn positive mask and require its lure to reuse that exact mask without another stream-11 draw. Both therefore expose the same 10/4/10 coordinate geometry. The only intended distributional difference is stored-real versus fresh-unstored content. |
| Fresh-noise fairness | PASS | Lines 37--45 assign the first 24 stream-12 standard Normals to positive visible cells and the immediately following 24 fresh Normals to lure visible cells, in the same ascending C-order coordinate order and with the same coordinatewise `0.01*sigma_codec` scale. Thus the noises are independent but identically distributed; neither query receives the other's realized measurement noise. This is a fair false-recall control rather than an accidental shared-noise advantage. |
| Unique RNG chronology | PASS | The inherited RNG is one fixed PCG64/SeedSequence child per `(master_seed,stream_id)` with NumPy version locked. Canonical origin order is `(context,port,prefix_local)`. Per seed, stream 11 therefore performs exactly `24*3=72` slot permutations and no lure permutation; stream 12 assigns exactly `24*(24+24)=1,152` consecutive Normal draws in the order `positive(origin 0), lure(origin 0), ..., positive(origin 23), lure(origin 23)`. Sorting selected cells by global C-order before assignment removes the last indexing ambiguity. |
| Lure provenance and ledger isolation | PASS | Lines 43--50 use the separately generated fresh 12-row lure prefix codec while retaining the corresponding valid tokens, and forbid the lure fingerprint/drift from the 96-record real ledger. The final registration also forbids the lure prefix and cue from insertion. Threshold calibration therefore queries an unstored item rather than a disguised real record. |
| Cross-port byte preservation | PASS | Lines 46--47 require reuse of the complete noisy positive cue and replacement of only `suffix_token` by the same-local token of port `(p+1) mod 4`. With the five-field `PartialCue`, this preserves `context_token`, `prefix_token`, every `cue_values` byte including masked poison, and every `cue_mask` byte; it consumes no RNG. The final registration serializes the same rule as “byte-for-value.” |
| Actual inherited `PartialCue` | PASS | The inherited dataclass at `episodic_ltm_dream_bridge.py:51-57` is ordered exactly `context_token`, `prefix_token`, `suffix_token`, `cue_values`, `cue_mask`; G7-M V2 aliases that class at `episodic_ltm_dream_bridge_v2.py:31-35`. Amendment lines 55--64 and final V3 registration lines 287--300 match those names and order and delete the obsolete `raw_codec`/`mask` dtype paths. The source annotations are `str`/`np.ndarray`, but the dataclass performs no runtime type enforcement and the inherited recall uses token equality/hash only, so registered opaque hashable tokens are compatible. |
| Boolean dtype | PASS | The amendment names executable `bool`/`numpy.bool_` with itemsize one and explicitly excludes removed alias `bool8`. This agrees with the inherited recall's `np.asarray(..., dtype=bool)` boundary. |
| Fresh registered roles | PASS | V3 replaces the roles with train `92100..92139`, validation `93100..93139`, and locked test `94100..94159`. Static set comparison gives 140 unique V3 seeds, zero within-V3 duplicates, and zero overlap with the 140 V2 seeds. V2 has zero implementation-lock/train/validation/test executions, and all V3 artifacts use new `v3` paths. |
| Claim boundary and stop rule | PASS | V3 does not override `claim_boundary`; deterministic inheritance retains the V2 restriction to this exact dimensionless synthetic generator and forbids general-world-model, biological-memory/dream, consciousness, AGI, open-world, and beyond-generator claims. Any further outcome-determining change requires V4 with fresh seeds, so the new cue rule cannot be tuned using a V3 registered seed. |

## Required implementation assertions

These are direct tests of the locked amendment, not discretionary design
choices:

1. assert per origin that positive and lure masks are array-equal, each has slot
   counts `10/4/10`, and the lure caused no stream-11 draw;
2. assert stream-12 assignment is the interleaved `24+24` chronology above and
   visible-coordinate assignment uses ascending global C-order;
3. assert cross-port `cue_values.tobytes(order="C")` and
   `cue_mask.tobytes(order="C")` equal the positive bytes, context/prefix tokens
   compare equal, and only the suffix token differs;
4. assert no lure object, prefix, fingerprint, drift, or cue is present in the
   96-record real-ledger identity/hash set;
5. assert the imported `PartialCue` dataclass field tuple equals the five names
   above before implementation lock.

The amendment is therefore executable, paired fairly, and narrow enough to
proceed to the independent full-registration audit and implementation lock.
