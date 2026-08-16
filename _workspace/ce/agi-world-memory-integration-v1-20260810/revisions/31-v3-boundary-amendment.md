# G9-CBM V3 pre-implementation boundary amendment

Status: LOCKED_PRE_IMPLEMENTATION

## Disposition of V2

G9-CBM V2 is **BLOCKED_PRE_IMPLEMENTATION** and is preserved at Git commit
`407a9714483ffd417660d9fe24db83b96f162301`. Its registration raw SHA-256 is
`b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2`.
No V2 implementation lock, train/calibration seed, validation seed, or locked
test seed was opened. The blocker was found by a read-only implementation-boundary
audit after the V2 preregistration commit.

V2 left two outcome-determining details unresolved:

1. its typed API named inherited `PartialCue` fields `raw_codec` and `mask`,
   while the byte-locked G7-M boundary actually exposes `cue_values` and
   `cue_mask`;
2. it did not determine the mask and observation noise of each unstored-lure
   `PartialCue`, although those values enter train recall calibration and the
   false-lure gate.

Because item 2 can change calibration and scientific outcomes, this is not a
clerical patch to locked V2. V3 uses fresh registered seeds and new artifact
paths. V2 remains an unexecuted negative audit witness.

## Sole scientific amendment

For every origin in the existing canonical `(context, port, prefix_local)`
order, V3 performs this exact cue chronology:

1. stream 11 independently permutes the C-order cells within rows `0:5`,
   `5:7`, and `7:12`, taking the first `10`, `4`, and `10` cells. This is the
   positive cue mask;
2. the corresponding unstored lure reuses that exact Boolean mask and makes no
   additional stream-11 draw;
3. within each mask, cells are visited in ascending global C-order flat index;
4. stream 12 draws 24 iid standard Normals for the positive visible cells,
   followed immediately by 24 fresh iid standard Normals for the lure visible
   cells;
5. each visible perturbation is
   `0.01*reshape(sigma_codec,(12,8))[r,h]*normal_draw`;
6. the lure cue values come from its separately generated fresh 12-row lure
   prefix codec. Its context, component, prefix, anchor-suffix, and action tokens
   remain the identical valid tokens of the corresponding positive origin;
7. the cross-port diagnostic reuses the complete noisy positive cue and changes
   only the suffix token to the same-local suffix token of port `(p+1) mod 4`.

Masked cells remain poison. Neither a lure nor its fingerprint/drift enters the
96-record real ledger. This paired-mask/fresh-noise rule is fixed before any V3
implementation or registered seed is opened.

## API amendment

The inherited `PartialCue` field order and exact names are:

```text
PartialCue(
  context_token: OpaqueToken,
  prefix_token: OpaqueToken,
  suffix_token: OpaqueToken,
  cue_values: float64[12,8],
  cue_mask: bool[12,8])
```

The executable NumPy Boolean dtype is `bool` (`numpy.bool_`, itemsize one). The
removed NumPy-2 alias `bool8` is not used as a dtype string.

## Fresh roles and paths

Registered roles are replaced, not extended:

```text
train/calibration: 92100..92139 (40)
validation:        93100..93139 (40)
locked test:       94100..94159 (60)
```

All V3 implementation, runner, test, implementation-lock, calibration,
validation, test, and integrity paths carry suffix `v3`. No V2 artifact path is
reused.

## Inheritance and stop rule

Every other scientific literal, generator equation, stream ID, state/action
space, 96-record ledger, R1 route, residual-only dream rule, 2x2 factorial,
controls, metrics, thresholds, 29-field budget, 393,216-byte allocation,
provenance rule, three-way pass identity, test unlock, and claim boundary is
inherited byte-for-meaning from the locked V2 contract and registration.

The V3 registration must declare deterministic recursive merge semantics,
delete the two obsolete V2 API paths, serialize the new cue rule above, and pass
an independent static reverse audit before implementation lock. Any further
outcome-determining ambiguity or post-registration scientific change requires
V4 with fresh seeds. No registered V3 seed may be used to choose an
implementation detail.
