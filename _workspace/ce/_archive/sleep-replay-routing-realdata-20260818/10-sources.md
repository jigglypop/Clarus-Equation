# Primary data and code sources

Status: COMPLETE

Checked 2026-08-18. This ledger records repository/access facts only; it does not evaluate CE claims.

## E15 — Giri et al., Nature (2024)

- Article: https://www.nature.com/articles/s41586-024-07538-2 (DOI 10.1038/s41586-024-07538-2; version of record 2024-06-12).
- Official data statement: processed group data are at https://doi.org/10.7302/73hn-m920 as NumPy `.npy` files used for most figures; the remainder of long-duration datasets is available from the corresponding author on request.
- Official code: https://github.com/diba-lab/sleep_loss_hippocampal_replay (v0.2), with general-purpose NeuroPy at https://github.com/diba-lab/NeuroPy (v0.1). The repository is public and displays a GPL-3.0 license. Its README identifies `ProcessedData`, `analysis_and_figure_code`, `interactive_figures`, and `misc_code`; processed data are not equivalent to raw spike/LFP event files.
- Event-level status: **PARTIAL / UNVERIFIED**. The public processed-group DOI and code are confirmed, but the article explicitly places the remaining long-duration datasets behind author request. No public animal/session raw spike/SWR event archive was confirmed in this check.
- Access/license/size: DOI landing page was not machine-readable here; exact file list, byte sizes, and data license are **UNVERIFIED**. GitHub code license is GPL-3.0. Do not claim raw event-level access from the public processed bundle.

## E02 — Wójcik et al., Nature Neuroscience (2026)

- Article: https://www.nature.com/articles/s41593-026-02333-w (DOI 10.1038/s41593-026-02333-w; version of record 2026-06-25).
- Official data: Dryad DOI https://doi.org/10.5061/dryad.c2fqz61kb, titled “Electrophysiological recordings of prefrontal activity over learning in non-human primates,” published 2024-11-12, version size 32.11 GB.
- Files: 50 session ZIPs (`m1_ses1`–`m1_ses25`, `m2_ses1`–`m2_ses25`) plus `README.md` (4.56 KB). The Dryad page exposes per-file sizes; examples include m1_ses1 169.48 MB, m2_ses1 1.78 GB, m2_ses4 2.73 GB. Each ZIP contains three `.npy` arrays: smoothed firing rates (`n_trials × n_neurons × 3500_timepoints`), `cell_loc`, and trial `meta`; trial IDs encode colour/shape/width/XOR/reward condition.
- Official code: https://github.com/m-j-wojcik/pfc_learning. The article states custom Python code; repository license was not established from the checked source and is **UNVERIFIED**.
- Event-level status: **VERIFIED for processed trial/session-level neural arrays**, not raw broadband. The Dryad README explicitly exposes animal/session IDs, trials, neurons, 3500 time points, and condition labels. Raw acquisition waveforms are not claimed.

## E19 — Communications Biology (2025)

- Article: https://www.nature.com/articles/s42003-025-08812-3 (DOI 10.1038/s42003-025-08812-3; 2025).
- Official data and code statement: “All preprocessed data necessary to evaluate the main conclusions” and custom MATLAB code are available at OSF https://osf.io/4e7bx/.
- Repository/access: OSF is the official repository and the article gives a public project URL. This checker received HTTP 403 from the OSF landing page, so file list, sizes, version date, license, and downloadability are **UNVERIFIED**.
- Event-level status: **UNVERIFIED / likely preprocessed participant-level only**. The article statement guarantees preprocessed data sufficient for main conclusions, not raw EEG/fMRI/event-level acquisition. Do not infer raw event access until the OSF project manifest is independently retrieved.

## E13 — Xu et al., Nature Neuroscience (2024)

- Article: https://www.nature.com/articles/s41593-023-01536-9 (DOI 10.1038/s41593-023-01536-9; version published February 2024).
- Official data statement: raw datasets are >10 TB of raw neural broadband, stored in a cost-efficient manner not immediately accessible to the Internet; data are available upon request at `khengen@wustl.edu`. Source data are provided with the paper.
- Official code: https://github.com/hengenlab/Sleep_restores_criticality. Public GitHub repository; repository license was not established from the checked page and is **UNVERIFIED**.
- Event-level status: **RESTRICTED / UNVERIFIED**. Raw broadband event-level data require author request; only source-data files are immediately linked from the article. No public raw-data repository, file list, checksum, or size manifest was confirmed.

## Boundary

These records do not authorize downloading or redistribution. “Public code” and “source data” do not imply public raw neural recordings. Any E15/E13 raw-data analysis requires an explicit acquisition receipt and access/usage terms; E19 requires recovery of the OSF manifest before granularity can be asserted.
