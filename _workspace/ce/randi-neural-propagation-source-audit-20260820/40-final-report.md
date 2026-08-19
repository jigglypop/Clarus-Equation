# Final report: Randi E2SYT intervention-data admission

Status: COMPLETE

Final decision: `SCHEMA_AUDIT_COMPLETE / EMPIRICAL_ROUTE_BLOCKED_CONDITIONAL`

## Outcome

The Randi et al. E2SYT/DANDI release is a genuine next-step brain-algorithm
input: the inspected NWB exemplar contains optogenetic event objects,
time-indexed whole-brain fluorescence arrays and canonical-identity machinery.
It is not yet an admissible canonical source-to-target causal table.

This run downloaded no large imaging payload.  It fixed all 223 published
DANDI asset metadata, then selected by a predeclared minimum-size rule and
inspected one 1,273,970-byte segmentation NWB.  Its bytes and SHA-256 match the
published receipt exactly.  No neural value, effect, model or endpoint was
inspected.

## What is established

1. The public release is stable and versioned: DANDI `001075`, version
   `0.240920.1434`, 113 subjects and 223 assets.
2. The exemplar exposes three optogenetic event rows with start/stop, power,
   target, pattern and site schema.
3. Green and red response-series schema contains 393 time rows by 105 ROI rows
   and corresponding timestamps.
4. The response-side identity path is explicit for matched ROIs:
   `GreenSignal.table_region -> PumpProbeGreenPlaneSegmentation -> neuropal_ids
   -> NeuroPAL labels/comments/confidences`.
5. The inspector traversed 151 objects without reading dataset values; a second
   run produced an identical JSON artifact.

These findings justify only `PASS_EVENT_TRACE_SCHEMA_CANDIDATE`.

## What blocks the causal route

The event source path is
`OptogeneticStimulusTable.targets -> OptogeneticStimulusTarget* ->
targeted_rois -> TargetPlaneSegmentation`.  The final target table has depth,
pixel-mask and image metadata, but no explicit `neuropal_ids`, label,
confidence or PumpProbe/NeuroPAL table reference.  Canonical `A_id` is therefore
`BLOCKED_EXPLICIT_CANONICAL_SOURCE_JOIN`.  Inferring it from spatial overlap
would be a new matching algorithm and cannot be smuggled in as a schema fact.

The exemplar also establishes no no-light, sham, failed-event or assignment
condition field.  This does not prove that every object in the complete source
release lacks one; it means the current inspected apparatus has no eligible
comparator.  Consequently:

- light-versus-no-light `tau` is `BLOCKED_CONTROL`;
- exact randomization strata and positivity are `UNVERIFIED`;
- post-stimulation autoresponse exclusions cannot be used as controls;
- direct/monosynaptic connectivity, endogenous `do(A)` and `G -> tau`
  mediation remain blocked.

## Brain-algorithm implication

The scientifically live hypothesis is narrower than “the connectome routes
signals.”  The release may permit a future test of whether *which identified
source is stimulated*, under a fixed stimulation policy and state, changes a
fixed target's later activity.  That is an effective source-targeting policy
effect, not a direct synaptic edge and not a consciousness mechanism.

The prospective active-control estimand is

`E[Y_B(Z=A;q) - Y_B(Z in C;q)]`,

where `C`, assignment strata, dose/geometry matching, target observability,
missingness and carryover rules must all be fixed before outcomes are read.
Calcium sampling also limits any later result to a prespecified response
window, not millisecond latency or synaptic direction.

## Next admissible step

A new contract may inspect source-identity join values and assignment/control
fields across a predeclared metadata-first subset of segmentation assets.  It
must not read response matrices while choosing the matching rule.  Entry gates
are:

1. an explicit or independently validated event-target-to-NeuroPAL map;
2. event-level assignment/randomization strata and a priori active controls;
3. treatment-independent missingness and no autoresponse conditioning;
4. fixed target, response window, dose/geometry/state matching and
   animal/session-level inference;
5. fresh audit authorization before any response effect is computed.

Without those fields, the correct endpoint is this run's blocked/conditional
status.  No threshold, source partition or response window should be tuned to
rescue it.

## Reproducibility artifacts

- `artifacts/e2syt-public-manifest.json`
- `artifacts/e2syt-exemplar-schema.json`
- `artifacts/e2syt-exemplar-schema-audit.md`
- `artifacts/fetch_e2syt_public_manifest.py`
- `artifacts/inspect_e2syt_nwb_schema.py`
- `20-audit.md`
- `31-validation.md`
