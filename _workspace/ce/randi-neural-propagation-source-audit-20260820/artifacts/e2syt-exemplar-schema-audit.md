# E2SYT segmentation exemplar schema audit

Status: COMPLETE

Scope: object paths, shapes, dtypes, table metadata and HDF5 references only.
No dataset value, response effect or endpoint was read or computed.

## Frozen input

- DANDI: `001075`, version `0.240920.1434`
- asset UUID: `d076d282-162a-4946-a1c8-68e72b6cce54`
- path: `sub-24/sub-24_ses-20211102-101248_desc-segmentation_ophys+ogen.nwb`
- bytes: `1,273,970` (exact match)
- SHA-256: `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e`
  (exact match)
- file signature: HDF5 `89 48 44 46 0d 0a 1a 0a`
- reader: Python 3.11.9, h5py 3.15.1, HDF5 1.14.6

The complete inventory has 151 HDF5 objects and is recorded in
`e2syt-exemplar-schema.json`, SHA-256
`45e53bb20739b3e1bbe61e9108422ad5e0f85cce9516af37f0b1df631433e54f`.
An independent second invocation was byte-identical.

## Event and signal schema

The file contains `/intervals/OptogeneticStimulusTable` with three rows and
the declared columns `start_time`, `stop_time`, `power`, `targets`,
`stimulus_pattern`, and `stimulus_site`.  `targets` is a length-three HDF5
reference vector.  The table description says that approximately every
30 seconds a neuron visible in tagRFP-T was selected at random and targeted by
the two-photon apparatus.  That procedural description is not an event-level
randomization receipt.

Three `/general/OptogeneticStimulusTarget*` objects each expose a
`targeted_rois` DynamicTableRegion into the three-row
`TargetPlaneSegmentation`.  That target table has pixel masks, depth and
reference-image schema, but no canonical NeuroPAL label column.

The response-side schema is materially richer:

| Object | Shape | Join |
|---|---:|---|
| `GreenSignal/data` | `393 x 105` | `table_region` points to the PumpProbe green ROI table |
| `GreenSignal/timestamps` | `393` | synchronized signal clock candidate |
| `RedSignal/data` | `393 x 105` | `table_region` points to the PumpProbe red ROI table |
| `RedSignal/timestamps` | `393` | synchronized signal clock candidate |
| PumpProbe green/red ROI tables | `105` rows each | each has a `neuropal_ids` UTF-8 column |
| NeuroPAL plane segmentation | `121` rows | labels, comments and confidence columns exist |

Subject, session identifier/start-time and genotype fields also exist.  No
`trials` table, `control` field, `sham` field or named no-light assignment path
is present in this exemplar.  This is not evidence that every other release
asset lacks such a comparator; it means that no eligible comparator has been
established by the inspected schema.

## Required event tuple

| Component | Schema verdict | Reason |
|---|---|---|
| `animal`, `session` | `PASS_SCHEMA` | subject and session metadata paths exist |
| `event`, `t_stim`, dose | `PASS_SCHEMA` | event ID, start/stop and power columns exist |
| targeted source object | `PASS_SCHEMA` | event target references and target ROI table exist |
| canonical `A_id` | `BLOCKED_EXPLICIT_JOIN` | target ROI table has no explicit NeuroPAL-ID column |
| canonical `B_id` | `PASS_SCHEMA_CANDIDATE` | signal ROI table links to `neuropal_ids` |
| `xB_pre`, `xB_post` | `PASS_SCHEMA_CANDIDATE` | signal matrix and timestamps coexist; values not inspected |
| `condition` | `PASS_SCHEMA` | genotype metadata exists; this mirror is WT-only |
| `u=0` / sham | `BLOCKED_CONTROL` | no machine-readable no-light or sham arm appears |
| failed/autoresponse status | `UNVERIFIED` | no such field was established by schema inspection |

## Decision

- `PASS_NWB_BYTE_IDENTITY`
- `PASS_EVENT_TRACE_SCHEMA_CANDIDATE`
- `BLOCKED_EXPLICIT_CANONICAL_SOURCE_JOIN`
- `BLOCKED_CONTROL`
- `BLOCKED_LIGHT_VS_NOLIGHT_TAU`
- `CONDITIONAL_RANDOM_SOURCE_ACTIVE_CONTROL`

The last route is only a prospective possibility: it requires validating the
actual source-choice assignment mechanism and strata, within-stratum positivity
for a fixed target, an a priori active-control source set, canonical source and
target mappings, treatment-independent missingness, matched power/duration/
geometry/expression/pre-state, and prespecified carryover handling without
conditioning on the post-stimulation autoresponse.  Its estimand would be the
source-targeting policy contrast
`E[Y_B(Z=A;q) - Y_B(Z in C;q)]`, not a no-light effect or endogenous `do(A)`.
This audit does not establish those conditions or numerical event-trace
synchronization.

Nothing here identifies a direct or monosynaptic edge, endogenous `do(A)`,
`G -> tau` mediation, mammalian routing, memory or consciousness.
