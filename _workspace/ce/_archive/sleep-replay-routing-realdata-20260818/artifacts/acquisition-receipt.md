# Acquisition receipt

Checked and acquired 2026-08-18. Hashes and byte sizes are in `realdata-manifest.csv`.

| Dataset | Official object | Local object | Provenance and access |
|---|---|---|---|
| E15 | `https://github.com/diba-lab/sleep_loss_hippocampal_replay` | `realdata/e15_repo` | commit `4b405bca0a18b82560a491e744ba064f7f8cddd9`; project code GPL-3.0; processed data correspond to Deep Blue DOI `10.7302/73hn-m920`, whose landing page states CC BY 4.0 |
| E19 | `https://api.osf.io/v2/nodes/4e7bx/files/` | `realdata/e19_*.json`, `realdata/e19_data/` | official public OSF API manifests; downloaded participant sleep parameters, retrieval/encoding RSA matrices, selected cluster files, and Figure 2--5 MATLAB code; node metadata are public but `node_license` is null, so no reuse license is asserted |
| E13 | `https://github.com/hengenlab/Sleep_restores_criticality` | `realdata/e13_repo`, `realdata/e13_box_share.html` | commit `0c6509f1969cbb6e9d07d7c0b2ff0bbc0328f347`; source-data workbooks included; repository has no LICENSE file; README-linked Box sample manifest is public/download-enabled but has no declared reuse license, so its large neural arrays were not downloaded; raw broadband remains author-request only |
| E02 | Dryad DOI `10.5061/dryad.c2fqz61kb`, version `418011` | `realdata/e02_dryad_*.json`, `realdata/e02_data/*.error.*` | metadata and SHA-256 digests acquired; CC0-1.0; file downloads returned HTTP 403 or required OAuth bearer token, so no neural ZIP was analyzed |

The E02 target was fixed before download as monkey 1 sessions 1, 2, 24 and 25. Their official metadata sizes total 733,144,673 bytes. Error bodies were renamed and never treated as ZIP or neural data.
