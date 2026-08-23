# Sources

Status: COMPLETE

The primary source is Maristany de las Casas et al., *Science* (2026), DOI
`10.1126/science.adx4358`, with the official G-Node archive DOI
`10.12751/g-node.etlk5k` (CC BY 4.0). The archived predecessor source lane
records the acquired ZIP SHA-256 as
`b9962e7760ac7299cc968fa4a23d2c965342d78abdded4f937a4081588f09ba3`
and ZIP CRC `PASS`.

This run reads the already extracted official Figure 2 files under
`_workspace/ce/_archive/neural-riemannian-metric-validation-20260818/artifacts/realdata/NRM-E17-extracted/Figure2/Data`.
There are 11 sessions from three animals (`DCO1`, `DCO2`, `DCO4`). Each file
contains saline and DCZ trial blocks, 180 frames per trial, and same-session
time-by-ROI calcium arrays. The official Figure 2 README defines the trial
chart as six seconds at 30 Hz, the instructed delayed-response task, saline
control, and chemogenetic NDNF activation by DCZ.

The archived input audit is authoritative for limitations: Figure 2, Figure 3,
and Figure 4 do not form one same-unit longitudinal chain; the released trial
array order has no verified acquisition-timestamp semantics; session and ROI
identities must not be pooled across animals. The result therefore concerns
held-out trials in released order, not verified future prediction or synaptic
consolidation.
