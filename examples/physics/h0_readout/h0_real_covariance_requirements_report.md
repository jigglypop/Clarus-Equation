# H0 real covariance requirements gate

- blocking requirements: 6
- channel plans: 4

## Promotion requirements

| requirement | reason | required for promotion |
|---|---|---|
| public source URL or DOI | The covariance/provenance must be independently recoverable. | `True` |
| pinned version, commit, release, or dataset date | Source drift must be detectable before selector comparison. | `True` |
| machine-readable Fisher/covariance or posterior-derived covariance | The q selector needs edges, not a final scalar H0. | `True` |
| node labels including observable, local anchors, and global priors | q_F is defined from source roles; unlabeled matrices are insufficient. | `True` |
| documented role map made before H0 comparison | Prevents fitting q after seeing the H0 branch. | `True` |
| positive definite or validated invertible covariance | The normalized Fisher-edge rule requires stable diagonals and edges. | `True` |
| negative/ablation case | Static all-local/all-global/flipped maps should not explain the channel equally well. | `False` |

## Channel plans

| priority | channel class | expected branch | minimum artifact | required labels | failure mode |
|---:|---|---|---|---|---|
| 1 | BAO+SN inverse distance ladder | global/low or low-side bridge | labelled compressed covariance JSON | BAO observable, sound horizon/ruler priors, SN nuisance/population nodes | matrix selects local/high before H0 refit |
| 2 | SH0ES/CCHP local ladder | local/high or semi-local high | calibration covariance graph JSON | Cepheid/TRGB/JAGB calibrators, anchors, SN/Hubble-flow nodes | endpoint-dominated graph selects global/low |
| 3 | GW standard sirens | bridge/intermediate | event/posterior covariance JSON | GW distance node, host/redshift/environment anchor nodes | event covariance collapses to either endpoint without bridge behavior |
| 4 | CMB acoustic-scale covariance | global/low | parameter covariance adapter JSON | acoustic-scale observable, horizon/ruler priors, nuisance nodes | same role map yields bridge/local q_F |

## Verdict

Real covariance promotion requires labelled, version-pinned matrix or posterior covariance before H0 comparison; scalar H0 rows are not enough.
