# Source and biological-eligibility lane

Status: COMPLETE

Access date: 2026-08-19

## 1. Source standard

Only primary papers, author repositories, and institutional data repositories
are used. A dataset is not eligible merely because its title says geometry or
frontal cortex. Eligibility is determined separately for region, physical
anatomy, structural connectivity, activity, perturbation, behavior,
longitudinal unit identity, sample unit, license, and accessible schema.

No new biological file was downloaded and no outcome was inspected in this
lane. Repository pages and data dictionaries are source metadata, not
scientific outcomes.

## 2. Primary PFC learning source

### E-PFC-01: Wójcik et al. macaque lateral PFC

- Paper: M. J. Wójcik et al., *Learning shapes neural geometry in the primate
  prefrontal cortex*, Nature Neuroscience 29 (2026), DOI
  [10.1038/s41593-026-02333-w](https://doi.org/10.1038/s41593-026-02333-w).
- Official data: Dryad
  [10.5061/dryad.c2fqz61kb](https://doi.org/10.5061/dryad.c2fqz61kb),
  published 2024-11-12.
- Official analysis code:
  [m-j-wojcik/pfc_learning](https://github.com/m-j-wojcik/pfc_learning).
- Species and region: two rhesus macaques; recordings from the lateral surface
  of PFC, with released categorical `cell_loc` codes covering locations dorsal
  and ventral to the principal sulcus, principal-sulcus banks, lateral OFC,
  and beyond arcuate-sulcus boundaries.
- Experiment 1: XOR-rule exposure from the first learning session; the paper
  reports 376 sequentially recorded neurons and 25 sessions from two animals.
- Longitudinal identity: not available. The paper states that electrodes were
  moved between sessions to obtain a new neuron sample. Learning-stage
  populations are therefore pseudopopulations, not tracked cells.
- Data release: 50 session archives (`m1_ses1` through `m1_ses25` and
  `m2_ses1` through `m2_ses25`), 32.11 GB total. Each archive contains smoothed
  firing rates with shape `trials x neurons x 3500 timepoints`, categorical
  `cell_loc`, and trial identity `meta`.
- Behavior: reward contingency and trial completion/termination are available
  in the study; the released neural `meta` encodes color, shape, irrelevant
  width, and XOR reward class.
- Anatomy: no continuous cortical surface coordinate, cortical thickness,
  signed depth, layer, surface normal, or subject-specific MRI registration is
  declared in the Dryad schema. `cell_loc` is categorical, not $(u,v,\ell)$.
- Connectivity and perturbation: no direct $W^s$ and no randomized circuit
  perturbation.
- License: the paper is open access and the data are publicly downloadable;
  the dataset page did not expose a machine-readable license string in the
  retrieved metadata. The exact data reuse license must be captured in the
  acquisition manifest before local scientific use. The GitHub repository did
  not expose a license in the inspected root listing, so its code is reference
  material until a license file is verified.
- Disposition: `ELIGIBLE_METADATA_ONLY_PENDING_LICENSE`; scientific analysis
  remains `ACCESS_BLOCKED` until exact reuse terms are in the acquisition
  manifest, and can become only `PFC_FEASIBILITY_ONLY` after Gate B. It cannot test
  physical ribbon $h$, same-cell $g_0\to g_t$, $W^s\to g$, or the full chain.

## 3. Independent PFC comparison sources

### E-PFC-02: Calangiu et al. macaque dlPFC multi-task data

- Official record: I. Calangiu, V. Mante, and J. Reppas,
  *Prospective and retrospective representations of saccadic movements in
  primate prefrontal cortex*, Zenodo v1,
  [10.5281/zenodo.14360532](https://doi.org/10.5281/zenodo.14360532),
  2025-02-12.
- Institutional mirror and raw-data description: ETH Research Collection,
  [10.3929/ethz-b-000721620](https://doi.org/10.3929/ethz-b-000721620),
  licensed CC BY-SA 4.0.
- Species and region: three rhesus macaques, Utah arrays in dorsolateral PFC.
- Tasks: reward decision, perceptual decision, and instructed action; neural
  and eye-movement measurements; released bins are 10 ms.
- Size: one 13.5 GB Zenodo archive, MD5
  `d4487a9235edf897b40656f4d1d89881`.
- Missing for the full claim: no verified subject-specific ribbon surface,
  depth/layer coordinate, direct $W^s$, or same-unit cross-task/longitudinal
  identity in the retrieved metadata.
- Disposition: `ELIGIBLE_METADATA`; at most cross-task
  `PFC_FEASIBILITY_ONLY` after schema and unit-identity inspection. It is not a
  learning-before/after replication of E-PFC-01.

### E-PFC-03: Kiani et al. area 8Ar population data

- Official record: NYU Data Catalog UID 10430,
  [High-density neural recordings of macaque monkeys performing a direction
  discrimination task](https://datacatalog.med.nyu.edu/dataset/10430).
- Species and region: three rhesus macaques, 96-channel arrays in area 8Ar of
  prearcuate gyrus.
- Released scale: monkey 1 has 1,206 trials and 219 simultaneously recorded
  units; monkey 2 has 1,778 trials and 169 units; monkey 3 has 1,047 trials and
  220 units.
- Task: 800 ms random-dot motion, variable delay, and saccadic choice. All
  animals were trained before implantation, so it is not a learning-onset
  cohort.
- Access: institutional catalog says `Free to All` and links to the Kiani Lab.
  An exact downloadable-byte license and hash still require acquisition.
- Missing for the full claim: subject-specific ribbon/depth coordinates,
  direct $W^s$, learning manipulation, and verified cross-session unit
  identity.
- Disposition: `ELIGIBLE_METADATA`; smallest external simultaneous-population
  trajectory/choice feasibility route, never $W^s\to g$ evidence.

## 4. Anatomy-aware sources and why they do not close the chain

### E-ANAT-01: HCP Young Adult

The official [HCP Young Adult page](https://www.humanconnectome.org/study/hcp-young-adult)
reports 1,071 processed subjects in the 2025 release, 1,113 subjects with
unprocessed imaging, and 46 subjects with 3T retest data in the S1200 release.
It supplies high-quality structural, diffusion, functional, and behavioral
imaging and can support human cortical-surface/thickness and connectome
controls. It is not paired with the macaque single-unit PFC datasets above and
does not measure identified synaptic $W^s$.

### E-ANAT-02: ABCD MRI

The official [ABCD MRI protocol](https://abcdstudy.org/scientists/protocols-mri/)
includes 3D T1, 3D T2, multi-shell HARDI, resting-state fMRI, and task fMRI.
The official imaging dictionary exposes cortical thickness, sulcal depth,
surface area, and longitudinal imaging visits. Access is controlled through
the NIMH Data Archive. It can test anatomy-aware population associations, not
the same-unit macaque ribbon chain.

These sources demonstrate that folds, thickness, diffusion, and functional
activity can be measured in the same human participant. They do not justify
merging unrelated participants, species, or spatial scales into a synthetic
same-unit arrow.

## 5. Region exclusions

- Predecessor E17 is rodent anterolateral motor cortex, not PFC.
- The predecessor GRID route is rat medial entorhinal cortex.
- E15 is hippocampal sleep/replay data.
- MICrONS is mouse visual cortex.
- DANDI 000037 is visual-cortex OpenScope data, not longitudinal PFC.
- FEF, DMFC, motor, visual, entorhinal, and hippocampal recordings are not
  relabeled as PFC because they are anatomically or computationally adjacent.

They may motivate methods or negative controls but cannot be counted as PFC
replications in this run.

## 6. Typed eligibility ledger

| Route | PFC | Activity | Behavior | continuous $(u,v,\ell)$ | $h$/folds | $W^s$ | perturbation | same-unit longitudinal | Status before acquisition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R-PFC-WOJCIK` | yes | yes | yes | no | no | no | no | no | `ELIGIBLE_METADATA_ONLY_PENDING_LICENSE` |
| `R-PFC-CALANGIU` | yes, dlPFC | yes | yes/eye | unverified | no | no | no | unverified | `ELIGIBLE_METADATA` |
| `R-PFC-KIANI` | yes, 8Ar | yes | yes | no | no | no | no | no | `ELIGIBLE_METADATA` |
| `R-PFC-RIBBON` | required | required | required | required | required | required | required | required | `UNTESTABLE_MISSING_INPUT` |

## 7. Acquisition rule

No 13.5 GB or 32.11 GB archive is downloaded merely to discover its schema.
The official README, per-file size list, license, and a minimal session archive
must first be frozen in `artifacts/source-manifest.json`. Full acquisition is
permitted only after Gate A and Gate B pass, because Gate B failure seals every
biological outcome. Every downloaded byte receives SHA-256, original filename,
official URL, access time, and license metadata. Partial or transformed arrays
are never treated as raw originals.
