# Neural Riemannian metric validation: primary sources

Status: COMPLETE

Checked 2026-08-18. This is an access/evidence ledger only. No cross-study chain is claimed.

## Tier rule

Tier A requires same-cell/same-synapse pre/post identity, direct structural or synaptic measurement, intervention, independent animals, and accessible future-trial trajectories. Tier B requires longitudinal same-cell activity plus behavior, or direct synaptic/effective-connectivity plus intervention without complete linkage. Tier C is activity-only, cross-subject, condition-mean, or non-longitudinal.

## Ranked candidates

### 1. `NRM-E17` — Tier B candidate; Tier A not established

Maristany de las Casas et al., *Science* (2026), DOI https://doi.org/10.1126/science.adx4358; official G-Node archive https://doi.org/10.12751/g-node.etlk5k/. The official ZIP was acquired at `artifacts/realdata/NRM-E17-g-node.etlk5k.zip`: 483,324,456 bytes, SHA-256 `b9962e7760ac7299cc968fa4a23d2c965342d78abdded4f937a4081588f09ba3`, ZIP CRC `PASS`, CC BY 4.0. It contains rule-switching learning behavior, two-photon dendritic calcium, synaptic glutamate/iGluSNFR imaging, Neuropixels, and NDNF optogenetic/chemogenetic manipulation.

Archive inspection changes the eligibility judgment from unverified to a verified split boundary. Figure 3 supplies unequal condition-summary synapse columns (Rule A 345, Rule B 101, Rule A-prime 120) without a same-synapse ID map or future single-trial trajectories. Figure 4 tracks dendrites across days but exposes summary arrays without animal IDs. Figure 2 exposes 11 saline/DCZ single-trial branch sessions from three animals, with 180 time points and 2--6 retained ROIs per inspected session, but it is not linked unit-by-unit to Figure 3 or Figure 4. Released array order has no verified timestamp provenance. E17 is therefore suitable for retrospective effective-dynamics and candidate-eligibility discovery, not Tier A, future-chronology confirmation, or one same-unit $\Delta W^s\to\Delta g\to\Delta x$ chain. The byte manifest is `artifacts/realdata/e17-acquisition-manifest.json`; the schema ledger is `artifacts/realdata/e17-mat-schema.json`.

### 2. `NRM-E19` — Tier B

Lee & Chen, *eLife* (2022), article https://doi.org/10.7554/eLife.72549; Dryad https://doi.org/10.5061/dryad.q573n5tjj. Public `Data_For_Dryad.zip` is 702.25 MB, published 2022-12-07. The official record states longitudinal two-photon M1 imaging on conditioning day 1 and day 7; `.mat` dF/F files and WaveSurfer H5 behavior log cue, reward, and mouse lick timing at 30 fps. Same-cell identity is plausible from longitudinal imaging but the matching manifest, animal counts, and trial-level file schema are **UNVERIFIED**. There is no direct synaptic measurement or causal intervention. Use for effective/representational geometry only.

### 3. `NRM-E20` — Tier B control

Gillon et al., *Scientific Data* (2023), DOI https://doi.org/10.1038/s41597-023-02214-y; DANDI 000037 https://dandiarchive.org/dandiset/000037; version DOI https://doi.org/10.48324/DANDI.000037/0.230426.0054. Public NWB data contain 50 sessions from 13 mice: ROI dF/F, masks, ROI tracking indices, plane image, running velocity, pupil traces, stimulus table, optional stimulus images and motion-corrected stacks. Basic files are 130 MB–1.7 GB each; 33 QC-passing basic files are about 15 GB; image versions 1.5–3.1 GB and stack versions about 45 GB each. The article reports ~100–500 plausible matched ROIs per plane. No learning intervention, direct synapse, or causal perturbation; license is **UNVERIFIED** from this check. Best for fixed-identity chart/noise controls.

### 4. `NRM-E21` — Tier B effective-dynamics only

LINK, Chestek Lab: project page https://chesteklab.github.io/LINK_dataset/ (DANDI link provided there). One rhesus macaque, 312 sessions over 303 days spanning 1,242 days. NWB files contain 96-channel threshold crossings and spiking-band power plus synchronized finger joint angles, position, and velocity, in 375 mostly contiguous trials/session. Chronic electrode/channel identity is available, but same-cell identity over years, direct synaptic W, independent animals, and intervention are absent or **UNVERIFIED**. Strong long-horizon future-trajectory benchmark; not direct W validation.

### 5. `NRM-E22` — Tier B candidate

Serradj et al., G-Node https://doi.org/10.12751/g-node.sbrmy3/; associated preprint https://doi.org/10.1101/2021.03.22.436415. Public archive is 631 MiB, published 2023-04-14, CC0 1.0. It contains calcium imaging during precise versus imprecise corticospinal motor training. Exact same-cell longitudinal identity, trial schema, intervention timing, and direct synapse are **UNVERIFIED** from the landing page. Use only for condition-held-out activity/behavior comparisons until manifest inspection.

### 6. `NRM-E02` — Tier C

Wójcik et al., *Nature Neuroscience* (2026), article https://www.nature.com/articles/s41593-026-02333-w; Dryad https://doi.org/10.5061/dryad.c2fqz61kb; code https://github.com/m-j-wojcik/pfc_learning. The public release is 32.11 GB with 50 session ZIPs plus README. Each ZIP contains smoothed firing rates (`n_trials x n_neurons x 3500_timepoints`), `cell_loc`, and trial `meta`; two macaques are represented. Electrodes were moved between sessions to obtain new neurons, so same-cell identity across learning is unavailable. No direct synapse or intervention. This is a session-held-out geometry feasibility source, not Eq. (1) evidence.

### 7. `NRM-E23` — Tier C

Public reaching comparators: DANDI 000070 https://dandiarchive.org/dandiset/000070/draft and RTT DANDI 000129 https://dandiarchive.org/dandiset/000129. NWB units/spikes, trial intervals, position and kinematics are accessible through DANDI. Same-cell longitudinal learning identity, direct synaptic measurement, intervention, and independent-animal chain criteria are not established. Use for baseline future-trajectory prediction only.

## Decision and boundaries

`NRM-E17` is the strongest locally acquired feasibility source because it combines learning, a direct synaptic proxy, and causal manipulation across the paper, but archive inspection proves that the released figures do not form the required same-unit chain. `NRM-E20` and `NRM-E21` remain strong future controls for fixed-identity chart stability and long-horizon effective dynamics. E02 is explicitly lower-tier because its moving electrodes prevent same-cell linkage. No source currently proves all Tier A requirements; `NRM-H1A` remains **UNVERIFIED**. Public activity or iGluSNFR is not a complete synaptic weight matrix.

## Candidate-formula provenance

The candidate registry separates provenance of a mathematical object from evidence that the brain uses it.

- Mahalanobis' covariance-normalized distance supports `S1` as a statistical comparator: https://cir.nii.ac.jp/crid/1370017279880961561.
- Kalman's linear-systems work supports the distinct controllability and observability constructions and their required inputs: https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD724842.xhtml. Gu et al. apply control-theoretic ideas to structural brain networks but do not identify process-noise covariance with control energy: https://doi.org/10.1038/ncomms9414.
- Haller's finite-time deformation review supports the Cauchy--Green pullback object in `S6`, not a noise-reachability or synaptic metric interpretation: https://doi.org/10.1146/annurev-fluid-010313-141322.
- Fisher information in correlated neural populations supports `S8` as response-distribution information geometry: https://pmc.ncbi.nlm.nih.gov/articles/PMC4451760/. The pullback construction in `S9` is supported mathematically by https://papers.neurips.cc/paper/6873-principles-of-riemannian-geometry-in-neural-networks.pdf.
- Effective resistance and diffusion distance require the graph assumptions stated in the registry: https://doi.org/10.1007/BF01164627 and https://doi.org/10.1016/j.acha.2006.04.006.
- Freidlin--Wentzell supports `D2` only as a small-noise SDE rate functional: https://doi.org/10.1007/978-3-642-25847-3. Randers' original construction supports the mathematical form of `D3`: https://doi.org/10.1103/PhysRev.59.195.
- Benamou--Brenier supports the dynamic representation of the same Wasserstein-2 distance in `P1/P2`: https://doi.org/10.1007/s002110050002.

No cited primary source validates the complete empirical chain $\Delta W\to\Delta g\to\Delta x$. `S2`, the neural interpretation of inverse process noise, raw-operator `S14`, residual pullback `S15`, and the forced sleep update `R4` remain operational or phenomenological candidates rather than sourced biological laws.
