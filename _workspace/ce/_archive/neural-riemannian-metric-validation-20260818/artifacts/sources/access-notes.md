# Source acquisition notes

Checked 2026-08-18. No datasets downloaded.

| ID | Official access and exposed size/license | Confirmed variables | Missing for higher tier |
|---|---|---|---|
| NRM-E17 | G-Node DOI https://doi.org/10.12751/g-node.etlk5k/; 461 MiB ZIP; CC BY 4.0 | Rule switching, behavior, calcium, glutamate/iGluSNFR, Neuropixels, opto/chemogenetics | Manifest, same-cell/synapse map, intervention timestamps, independent animals, future trial endpoint |
| NRM-E19 | Dryad DOI https://doi.org/10.5061/dryad.q573n5tjj; 702.25 MB ZIP; license **UNVERIFIED** | Day 1/day 7 dF/F, cue/reward/lick H5 timing, 30 fps | ROI map, animal/session manifest, direct synapse/intervention |
| NRM-E20 | DANDI 000037; 50 sessions/13 mice; 130 MB–1.7 GB basic, ~15 GB QC subset, ~45 GB stack; license **UNVERIFIED** | dF/F, masks, tracking, running/pupil/stimulus, raw stack | Learning manipulation, direct synapse, causal perturbation |
| NRM-E21 | LINK page; 312 sessions/303 days/1,242 days; NWB; exact DANDI version/license **UNVERIFIED** | TCR/SBP, synchronized kinematics, trials | Same cells, direct synapse, independent animals, intervention |
| NRM-E22 | G-Node DOI https://doi.org/10.12751/g-node.sbrmy3/; 631 MiB; CC0 1.0 | Calcium imaging, precise/imprecise motor training | Same-cell map, intervention/timing, direct synapse, trial manifest |
| NRM-E02 | Dryad DOI https://doi.org/10.5061/dryad.c2fqz61kb; 32.11 GB; 50 ZIPs; license **UNVERIFIED** | Trial x neuron x timepoint, cell locations, metadata | Same-cell identity, direct synapse, intervention |
| NRM-E23 | DANDI 000070/000129; public NWB/API; exact version/license/size **UNVERIFIED** | Units/spikes, intervals, position/kinematics | Learning identity, direct synapse, intervention |

Access decision: inspect E17's archive manifest first; use E20/E21 as controls and E02 only as a lower-tier comparator. No cross-study concatenation is permitted.
