# Primary-source and acquisition lane

Status: COMPLETE

Checked: 2026-08-20. This lane uses the primary Nature article, its official
source-data/code links, and the published DANDI NWB mirror. No large neural
payload or empirical outcome was downloaded.

## Primary publication

- Randi, Sharma, Dvali et al., “Neural signal propagation atlas of
  *Caenorhabditis elegans*,” *Nature* 623, 406–414 (2023), DOI
  `10.1038/s41586-023-06683-4`, version of record 2023-11-01.
- The article is open under CC BY 4.0. Its data-availability statement names
  OSF `E2SYT`; code availability freezes `pumpprobe` at Zenodo `8312985`,
  `wormdatamodel` at `8247252`, `wormneuronsegmentation-c` at `8247242`,
  `wormbrain` at `8247254`, and acquisition code at `8247258`.
- WT-background experiments include 113 animals, each recorded for up to
  40 min. A mostly randomly selected sequence of individual neurons was
  stimulated approximately every 30 s while whole-brain calcium activity was
  recorded. The final atlas covers 23,433 source–target pairs and 186 of 188
  head neurons.
- The perturbation is spatially restricted two-photon activation of the
  GUR-3/PRDX-2 actuator. NeuroPAL fluorophores provide canonical neuron
  identity. The paper reports stimulus durations of 0.3 or 0.5 s and uses an
  upstream-neuron calcium autoresponse as an inclusion requirement for the
  published pair atlas.
- An `unc-31` mutant background is used to study dense-core-vesicle signalling.
  It is a mechanistic genotype contrast, not an untreated or sham arm for each
  WT stimulation event.

## Source-data and interpretation boundary

Nature supplies XLS/XLSX/CSV source data for Figs. 2–6 and Extended Data
Figs. 2–7 and 9. These files support reproduction of the processed pair atlas,
response amplitude/q-value summaries and selected temporal-kernel results.
They do not by themselves prove that every stimulation event, failed
autoresponse, trace, assignment order and missing identity is retained.

The publication explicitly describes the atlas as effective signal
propagation, not a set of direct or monosynaptic connections. Anatomically
indirect and extrasynaptic paths are part of the measured response. A positive
pair value therefore cannot be promoted to a direct synaptic edge.

## Published DANDI mirror

The authors and CatalystNeuro published an NWB conversion as DANDI `001075`,
version `0.240920.1434`, DOI `10.48324/dandi.001075/0.240920.1434`. The version
is open access under CC BY 4.0 and names the Nature article as its describing
resource. DANDI's versioned metadata reports:

| Field | Value |
|---|---:|
| subjects | 113 |
| assets | 223 |
| total bytes | 4,073,427,051,047 |
| raw/full NWB assets | 113 |
| raw/full bytes | 4,072,533,594,007 |
| segmentation NWB assets | 110 |
| segmentation bytes | 893,457,040 |
| data standard | NWB |
| species | *C. elegans* |

The metadata-only acquisition receipt is now frozen locally at
`artifacts/e2syt-public-manifest.json` (204,834 bytes; file SHA-256
`8ae094546532cc654dd6d49f3ffe5284f734598e7f6e05369a762579bba60e88`).
It records all 223 asset UUIDs, paths, byte sizes, public content URLs and
published SHA-256 values.  The canonical hash of the normalized asset list is
`ff206a13191908e92167817c60644265066ba578d8a58ea1ad1011466dcb47d5`.
The receipt was produced without downloading an NWB payload.

The deterministic schema exemplar is the minimum `(bytes, path)` segmentation
asset: `sub-24/sub-24_ses-20211102-101248_desc-segmentation_ophys+ogen.nwb`,
1,273,970 bytes, asset UUID `d076d282-162a-4946-a1c8-68e72b6cce54`, published
SHA-256 `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e`.
Selecting it used metadata only and therefore inspected no response outcome.

The 110 segmentation files have path form
`sub-*/sub-*_ses-*_desc-segmentation_ophys+ogen.nwb`. Subjects `sub-20`,
`sub-23` and `sub-33` have no segmentation asset in this published version.
Every asset has a DANDI asset UUID, byte size, `dandi:sha2-256`, session start
and public content URL. The DANDI asset metadata labels this published mirror
WT; it does not expose the 18-animal `unc-31` panel as a separate genotype
subset.

The conversion repository `catalystneuro/leifer_lab_to_nwb` is BSD-3-Clause
and explicitly identifies DANDI `001075` as the Randi Nature 2023 conversion.
This mirror is therefore a stable, versioned acquisition route independent of
the currently inaccessible OSF API.

## OSF and local state

- OSF `E2SYT` is the publication's official machine-readable source, but its
  API/file tree could not be retrieved from this Windows environment during
  the audit. Exact OSF object IDs, file sizes and checksums remain
  `BLOCKED_OSF_MANIFEST`.
- No E2SYT, `pumpprobe`, `wormbrain`, conversion repository or DANDI asset is
  currently present under the repository data roots.
- Raw research payload belongs under ignored `data/`, not in Git. Only compact
  manifests, acquisition receipts, code and derived summaries may enter the
  worktree.

## Source verdict

| Requirement | Verdict | Evidence ceiling |
|---|---|---|
| publication/source identity | `PASS_SOURCE` | article DOI, code DOIs, OSF ID and DANDI DOI fixed |
| canonical neuron identity | `PASS_SOURCE` | NeuroPAL identification is part of the experiment/NWB conversion |
| source-specific intervention | `PASS_SOURCE` | targeted two-photon stimulation with recorded whole-brain response |
| stable acquisition manifest | `PASS_DANDI_MANIFEST` | published DANDI version, asset UUID/size/SHA-256 available |
| compact feasible subset | `PASS_CANDIDATE` | 110 segmentation assets total 893 MB; schema still requires inspection |
| event-level untreated/sham control | `UNVERIFIED` | article describes repeated stimulation, not a randomized no-light arm |
| all-event/autoresponse failures | `UNVERIFIED` | published atlas conditions on autoresponse; NWB schema not yet inspected |
| `unc-31` event-level mirror | `BLOCKED_DANDI_SUBSET` | DANDI published metadata exposes WT 113-subject panel only |
| raw independent conversion audit | `CONDITIONAL_PROCESSED_MIRROR` | 4.07 TB raw layer is available but not acquired |

## Primary links

- Nature article: `https://www.nature.com/articles/s41586-023-06683-4`
- OSF DOI: `https://doi.org/10.17605/OSF.IO/E2SYT`
- DANDI version: `https://dandiarchive.org/dandiset/001075/0.240920.1434`
- DANDI manifest: `https://dandiarchive.s3.amazonaws.com/dandisets/001075/0.240920.1434/assets.yaml`
- NWB conversion: `https://github.com/catalystneuro/leifer_lab_to_nwb`
- Analysis code: `https://github.com/leiferlab/pumpprobe`
