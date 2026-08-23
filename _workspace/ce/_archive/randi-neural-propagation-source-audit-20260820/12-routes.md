# Acquisition and validation routes

Status: COMPLETE

## R1 — DANDI segmentation-event route

**Status: recommended; schema inspection required.**

Use published DANDI `001075/0.240920.1434`. Do not acquire the 4.07-TB image
layer. First download exactly one segmentation asset selected solely as the
smallest byte-size asset, with path as the deterministic tie-break, and inspect
its NWB schema without computing neural effects. The exemplar must establish:

- optogenetic event onset/duration and targeted source identity;
- whole-brain target traces and a common timestamp;
- canonical neuron labels and identity confidence/missingness;
- all assignments, including autoresponse failures if retained;
- dose/session/subject fields and repeated-event order.

If this passes, freeze all 110 segmentation asset UUID/path/size/SHA-256 values
before acquiring the approximately 893-MB subset. Subjects 20, 23 and 33 are
missing segmentation assets and must be marked missing, not silently replaced.

The first estimand is randomized source-choice versus active-source controls,
not light versus no-light. Failure to reconstruct the source-choice assignment
mechanism changes the route to observational intervention-label association.

## R2 — processed Nature source-data replication

**Status: PASS_SOURCE, processed-atlas claim only.**

The article's compact source-data tables can reproduce selected pair means,
observation counts, q-values and WT/`unc-31` atlas contrasts. This route is
useful as an oracle for conversion parity and published-result reproduction.
It cannot establish event-level assignment, recover excluded failures, or
support animal-clustered reanalysis if only pair aggregates are retained.

## R3 — OSF/pumpprobe native event route

**Status: BLOCKED_OSF_MANIFEST in the current environment.**

OSF `E2SYT` and `pumpprobe` are the publication-native route. Reopen only when
the OSF API tree/object checksums can be captured or when an official compact
manifest is supplied. Native `Funatlas`/`Fconn` objects may preserve event
kernels and observation mappings, but code semantics cannot substitute for
the actual source bytes.

## R4 — DANDI raw-image conversion audit

**Status: DEFERRED_PAYLOAD, not required for the first routing test.**

The 113 full NWB assets total about 4.073 TB. They could independently audit
segmentation and fluorescence extraction, but acquiring them before the
segmentation-event route proves necessary fields would waste time and storage.
Open this route only if a concrete conversion/preprocessing falsifier cannot
be resolved with the compact NWB and source tables.

## R5 — alternative datasets

**Status: deferred.**

- IBL Brain-wide Map remains the mammalian observational replication route.
- CloudCell remains metric-only and GFP-falsifier support.
- Neither repairs a failed Randi event schema by changing thresholds or
  redefining arbitrary neural groups.

## Decision tree

1. Freeze DANDI version metadata and the 223-asset manifest.
2. Pass the stable source/math audit.
3. Acquire the deterministic smallest segmentation exemplar only.
4. Inspect NWB fields with no effect calculation.
5. If event assignment, identities and traces pass, preregister the
   source-choice effect and only then acquire all compact segmentation assets.
6. If fields fail, stop R1 and retain R2 processed-atlas reproduction; do not
   open the 4-TB raw route without a named falsifier.

## Permanent claim ceiling

Even a positive R1 result means that a targeted-light policy changed a
canonical target's measured calcium response relative to matched active-source
controls in this immobilized *C. elegans* apparatus. It does not establish a
monosynaptic edge, endogenous information flow, behavioral control, a physical
Riemannian brain metric, metric mediation, mammalian memory or consciousness.
