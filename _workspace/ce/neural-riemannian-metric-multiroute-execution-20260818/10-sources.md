# Multiroute source and access ledger

Status: COMPLETE  
Checked: 2026-08-18  
Scope: frozen public routes only. No route outcome, score, or mathematical claim was inspected here.

Access rule: official paper/repository existence was checked first. A file is
marked downloaded only when a direct request returned the file and its byte
size/hash could be recorded. Initial sandboxed DANDI and Figshare failures are
superseded by the successful official-API retries recorded below. Large
MICrONS, Dryad BCI, and DANDI payloads were not downloaded unless an exact
local object is listed.

| Route | Official primary source / access | File or API identity | Size / SHA-256 | Schema and sample identity | License / version date | Declared endpoint support | Eligibility |
|---|---|---|---|---|---|---|---|
| R-GRID-TORUS | Gardner et al., Nature 2022, DOI [10.1038/s41586-021-04268-7](https://doi.org/10.1038/s41586-021-04268-7); official data/code release [Figshare 16764508](https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508) | Figshare article version 6; `Toroidal_topology_grid_cell_data.zip`, file id `35078602`; official code repository at commit `ce920c1d849edc6b6eb40dc091f38f27b1226009` | 319,902,753 bytes; MD5 `379BFDCA61CD54D5F58CAB9D3BA477DE`; SHA-256 `56842A7AE8E8F9144E390C9CDD72494D8EA2C48F2CE89CDEE4780A49ED495372`; supplied and computed MD5 match | ZIP contains 4 NPZ plus 4 session TXT files for rats Q/R day 1/R day 2/S. NPZ schemas contain `spikes_mod*` object dictionaries plus `x`, `y`, `t`, `z`, `azimuth`; unit counts: Q mods 97/66, R day 1 166/168/149, R day 2 189/172/183, S 140. `utils.py` freezes explicit REM/SWS intervals for Q, R day 2 and S, enabling independent spike-block splits | Data CC0; code GPL-3.0; article 2022-01-12; code commit frozen above | Supports topology-metric dissociation using independent spike blocks across wake/REM/SWS. No structural W and no causal metric intervention; maximum claim is topology stability versus separately estimated local metric/transition statistics | ELIGIBLE (topology-metric dissociation only) |
| R-DANDI-37 | Official [DANDI 000037 record](https://dandiarchive.org/dandiset/000037) and DANDI download documentation [official guide](https://docs.dandiarchive.org/user-guide-using/accessing-data/downloading/) | API identifies 000037 as `Allen Institute Openscope - Responses to inconsistent stimuli in somata and distal apical dendrites in primary visual cortex`, published version `0.240209.1623`, 151 assets | Aggregate size `2,484,974,036,912` bytes; per-file manifest/hash not used because source identity is wrong for frozen route | This is not the declared longitudinal M1 learning dataset. The earlier description of 000037 as longitudinal M1 activity/behavior was incorrect and is corrected here | Published version 0.240209.1623; exact per-asset licenses require a different route-specific manifest | Frozen R-DANDI-37 endpoint requires longitudinal M1 pre-period geometry, later activity/behavior and relative deformation; this Dandiset does not establish those typed inputs | UNTESTABLE_MISSING_INPUT (source mismatch) |
| R-ALLOPTICAL | Wagenmaker et al., NeurIPS 2024 primary paper [PMC12466919](https://pmc.ncbi.nlm.nih.gov/articles/PMC12466919/); official NeurIPS DOI [10.52202/079017-0994](https://doi.org/10.52202/079017-0994) | Official checklist states data/code were not released; no public raw-data manifest or repository was identified | No downloadable raw file, size, hash, or license verified | Published method reports stimulation/response measurements, but no verified same-animal independent unperturbed target or held-out manifest | Paper 2024; raw-data license unavailable | Frozen perturbational endpoint cannot be executed from the released material | UNTESTABLE_MISSING_INPUT |
| R-BCI | Sadtler et al., Nature 2014 [10.1038/nature13665](https://doi.org/10.1038/nature13665), plus successor Busch et al., Nature Neuroscience 2026 and official Dryad record [10.5061/dryad.9cnp5hr0w](https://doi.org/10.5061/dryad.9cnp5hr0w) | Sadtler raw neural/session files remain absent. Dryad v3 has `avatarRT_subject_data.zip` (file 4685339, 38,823,011,866 bytes, SHA-256 `b03ff1ca2cf4b90a5b3c1f8127f6dba9d3e373145426ba55798c776f42e0d684`) and `README.md` (21,501 bytes, SHA-256 `b2c0883858d296a5de89191a66552faf29cbb4b70a2a1fa04baabca7d900a838`). Official analysis commit `a1c827e3d3a8543d6e922a9c11fbedb90056561a` is local and contains subject-level derived behavior/manifold tables | Shell download endpoints returned WAF/authorization errors, so the 38.8 GB archive was not locally hashed or schema-audited. The local official code repository exposes derived subject/session tables, not the frozen raw payload | N=18, 4--5 fMRI sessions, baseline joystick calibration, IM/WMP/OMP perturbations, subject/session/trial/latent/behavior fields are documented. Derived tables permit a subject-level directional reproduction but not an independently reconstructed metric length from raw baseline activity | Dryad v3, 2026-04-01, CC0. The GitHub repository has no source-object license establishing that its derived tables replace the Dryad raw-input audit, so they remain local descriptive inputs | Reproduce only the frozen subject-level accessibility direction from official derived tables; do not count it as a raw held-out metric test | PARTIAL_DESCRIPTIVE |
| R-MICRONS | Official [MICrONS Explorer](https://www.microns-explorer.org/), [DANDI 000402 v0.230307.2132](https://dandiarchive.org/dandiset/000402/0.230307.2132), primary paper DOI [10.1038/s41586-025-08790-w](https://doi.org/10.1038/s41586-025-08790-w), and BossDB/CAVE query endpoints | DANDI 000402 v0.230307.2132: 19 assets, aggregate 1,347,665,448,061 bytes. Anonymous BossDB probes retrieved coreg CSV (14,392 bytes), nucleus CSV (10,988,660 bytes; path includes `minnie/minnie65/nucleus_detection`), functional DB SQL (116,199,864,342 bytes), and synapse graph approximately 47.5 GB. Local coreg/nucleus/README/technical-PDF files were downloaded | Selective local files have no single biological full-graph hash here; functional values plus matching structural edges for the same roots are still required | Co-registration and nucleus tables are present, but static route is at most descriptive until functional values and structural edges are linked on identical roots; no longitudinal Delta W | MICrONS portal terms [official terms](https://www.microns-explorer.org/terms-and-conditions); query/CAVE token required for selective structural access; DANDI version above | Supports only a static structure-function association if same-root functional and structural records are obtained; no Delta W | PARTIAL_DESCRIPTIVE |

## Evidence notes

- Gardner's paper explicitly identifies Figshare article 16764508 as the data and code release; this is the official mirror used for R-GRID-TORUS.
- The DANDI API retry retrieved the versioned 000037 record and established a
  source mismatch. The earlier connection failure is attempt history only.
- The Wagenmaker official checklist confirms that data/code were not released; method capability is not treated as data eligibility.
- The Sadtler Nature page exposes source-data workbooks, but those aggregated figure tables do not establish raw subject/session identity or the route's required held-out split.
- MICrONS official DANDI/BossDB resources are real and selective files were retrieved, but same-root functional values plus structural edges and a reproducible query manifest remain necessary; no Delta W route is implied.

## Access log

| Attempt | Result |
|---|---|
| `https://api.figshare.com/v2/articles/16764508` | Later acquisition completed: clean single-transfer ZIP passed MD5 and SHA-256; initial Range/resume corrupted transfer preserved separately |
| `https://api.dandiarchive.org/api/dandisets/000037/` | Later retry succeeded; active disposition is the Openscope source mismatch, not an access block |
| Dryad files 4685339/4685375 | Metadata API succeeded; shell file endpoints returned HTTP 401/403 WAF responses. Official derived tables were acquired through analysis commit `a1c827e3d3a8543d6e922a9c11fbedb90056561a` only |
| MICrONS Explorer/BossDB | Portal and anonymous S3 listing succeeded; small coregistration, nucleus, access README, and technical-document files were downloaded. Large functional/synapse payloads were not downloaded |
| Nature/PMC primary pages | Primary publication and stated data-availability information reachable |

The later DANDI API retry superseded the earlier failed-request note: 000037
was retrieved and identified as the Openscope visual-cortex dataset, not the
declared longitudinal M1 source. The later Figshare retry supplied the v6
archive metadata and the clean acquisition verified the archive hash and
schema described above.

### API retry correction

The later official API retry supersedes the earlier failed-request note for
DANDI 000037: the record was retrieved and identified as the Openscope visual
cortex inconsistent-stimulus dataset, version 0.240209.1623, 151 assets and
2,484,974,036,912 aggregate bytes. It is therefore a source mismatch for the
frozen R-DANDI-37 route, not an access block. The later Figshare API metadata
retrieved article version 6 and file id 35078602 with the byte size, MD5 and
CC0 metadata recorded above; extraction/schema and session split verification
are complete as summarized in the R-GRID-TORUS row.

No route is promoted to biological evidence by this ledger. The statuses are
input/access dispositions only and must be consumed by the execution and audit
ledgers.

## Local-route provenance audit

The following routes use files already present in predecessor workspaces. The
audit records provenance and typed-input limitations only; it does not inspect
new route outcomes.

| Route | Exact local inputs / provenance | Observed metadata | Missing typed inputs for frozen endpoint | Eligibility |
|---|---|---|---|---|
| R-E17-F3 | `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure3/Selectivity/Data/SelectivityStatsGluSNFR.mat`; `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure3/FunctionalClustering/Data/SpatialCorr_RuleARuleB.mat`; Figure 3 READMEs and MATLAB code under `.../Figure3/` | Released MAT summaries, source READMEs and reproduction code exist | No verified animal-level or same-unit independent held-out clustering/task endpoint; no typed pre-transition metric and future endpoint pairing | PARTIAL_DESCRIPTIVE |
| R-E17-F4 | `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure4/Data/DataSummary_CaImagingDendrites.mat`; `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure4/Data/DataRepDrift_CaImagingDendrites.mat`; `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure4/README_Figure4_Data.md`; code under `.../Figure4/Code/` | Released dendrite summary/representation-drift MAT files with README and reproduction code | No verified independent animal/same-unit split linking an earlier frozen geometry proxy to later held-out drift | PARTIAL_DESCRIPTIVE |
| R-E17-F5 | `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure5/Data/Figure5Data_TransitionError.mat`; `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure5/Data/Figure5Data_Selectivity_RepDrift.mat`; `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure5/README_Figure5_Data.md`; code under `.../Figure5/Code/` | Released transition-error/selectivity-drift MAT files and source code | No verified earlier frozen geometry, independent future transition endpoint, or same-unit chronology sufficient for the declared prediction | PARTIAL_DESCRIPTIVE |
| R-E17-F2 | `_workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/` and predecessor E17 tournament artifacts | Same released E17 reference family used by the predecessor tournament | It is the locked reference, not an independent replication; reusing its outcomes cannot supply an independent endpoint | INELIGIBLE_DEPENDENT |
| R-SLEEP-E19 | `_workspace/ce/_archive/sleep-replay-routing-realdata-20260818/artifacts/realdata/e19_data/suball_sleep_staging.mat`; `suball_neural_sleep_param.mat`; `encode_corr_conds_final.mat`; `closeeye_corr_conds_final.mat`; cluster MAT files and `figure2.m`--`figure5.m` in the same directory | Source-selected cluster-association material; n=34 rows/subjects as described in predecessor provenance | Chronology between predictor and later endpoint, subject-level independent split, and license/receipt provenance remain unresolved; no structural W or same-unit Delta W | PARTIAL_DESCRIPTIVE |
| R-SLEEP-E15 | `_workspace/ce/_archive/sleep-replay-routing-realdata-20260818/artifacts/realdata/e15_repo/README.md`; `ProcessedData/` arrays; `interactive_figures/`; `analysis_and_figure_code/` | 104 processed rows with 13 session labels in predecessor provenance; replay/sleep-deprivation processed material and scoring/code are present | Independence of rows/session labels and subject-level held-out endpoint are not established; license/receipt provenance unresolved; no structural W | PARTIAL_DESCRIPTIVE |
| R-CELEGANS | `_workspace/ce/connectome-graph-replay-20260818/artifacts/herm_full_edgelist.csv`, frozen OpenWorm commit `b36380a36d2a6dda0f03c946c433524b25ea2268` | 252,842 bytes; SHA-256 `0ab9baab5f404895b8dbeb8daa453c86e8f342961bc458cd19bf1b5f6a38d859`; 7,379 edge rows; chemical/electrical types stay separate | No real future endpoint. The selected repository tree has no source-object license; predecessor audit permits local research replay but not redistribution. It can serve only as a run-local structural fixture plus synthetic perturbation source | PARTIAL_DESCRIPTIVE |

The Figshare route is now acquired and hash-verified. An initial Range/resume
attempt produced a corrupted transfer and was preserved separately; a clean
single-transfer acquisition passed the official MD5 and independent SHA-256.
`R-DANDI-37` is no longer an access block:
the API succeeded and demonstrated a source-identity mismatch, so it is
`UNTESTABLE_MISSING_INPUT`. The earlier longitudinal-M1 description was an
incorrect provenance statement and must not be reused. Other public access
blocks remain provisional until their official API/manifests are retried.

## Anatomy-aware follow-up sources

These sources were identified after the frozen 13-route outcome lock. They are
not added to this run and none of their outcomes was inspected. They define a
separate future test of whether a learned functional metric adds information
beyond the subject's pre-existing cortical surface geometry $h$.

| Future source | Official release and typed inputs | Smallest valid use | Boundary |
|---|---|---|---|
| HCP Young Adult 2025 | [Official 2025 release](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release), released 2025-08-11: 1,071 processed participants, 45 retest participants, structural/FreeSurfer surfaces, diffusion processing, resting fMRI and task fMRI | Fit $h$ from the fixed subject surface and anatomy; fit $g_0$ from visit-one activity; ask whether $g_0$ predicts retest activity beyond $h$, tract length, depth, area and a flat-pullback null | Retest is observational and only 45 participants. It can test incremental $H_G$, not $\Delta W^s\to\Delta g$ causality. Do not mix the 2025 and S1200 processed releases. |
| ABCD | [Official MRI protocol](https://abcdstudy.org/scientists/protocols-mri/) and [release-6 derivative documentation](https://docs.abcdstudy.org/latest/documentation/imaging/data_derivatives.html): T1w/T2w, multi-shell HARDI, resting fMRI and task fMRI in a longitudinal cohort | Freeze anatomy-aware $h$, surface/layer and diffusion controls at baseline; predict a later-visit activity or behavior endpoint with family/site blocked splits | Development, scanner/site and motion are compulsory nuisance fields. A positive longitudinal prediction remains $H_G$ unless a typed synaptic $W^s$ intervention is measured. |
| OpenNeuro ds006072 v1.1.0 | [Official dataset](https://openneuro.org/datasets/ds006072/versions/1.1.0), CC0: seven original participants, four later replication participants, 11 BIDS participant entries, 22 sessions, structural T1w/T2w, diffusion, repeated resting/task fMRI before and after psilocybin or methylphenidate | Within participant, keep the structural surface $h$ fixed, estimate pre-drug $g_0$, and test preregistered post-drug trajectory prediction and relative deformation against drug, motion, global gain/noise and flat-pullback controls | Small $N$, pharmacological state change and repeated scans make this a high-resolution exploratory $H_G$ route, not a structural-plasticity or sleep claim. The listed archive is about 3 TB and was not downloaded in this run. |

The decisive contrast for all three is incremental rather than substitutive:
the baseline model contains $h$ and anatomical nuisances, while the candidate
adds $g_0$ or the relative field $g_0^{-1/2}g_tg_0^{-1/2}$. A result that
disappears after $h$, depth, area, wiring length, motion, gain/noise or a smooth
flat coordinate pullback is controlled does not support learned geometry.
