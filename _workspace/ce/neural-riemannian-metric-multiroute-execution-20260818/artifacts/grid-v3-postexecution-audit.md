# Grid-Torus v3 postexecution audit

Status: INTEGRITY_PASS_PARTIAL_DESCRIPTIVE

All preexecution code, design, contract and raw-input hashes match the lock.
The expensive verifier reloaded the three official NPZ files, parsed the
official interval definitions, rebuilt every 10 s block, C/A/B split, wake-C
chart, topology signature, mobility precision and contrast, and matched all
six stored modules at `rtol=1e-8`, `atol=1e-10`.

## Frozen heuristic

All 12 REM/SWS module-state comparisons had raw mobility AIRM above the
within-state split reference in both roles. Only R3-REM also kept the topology
ratio at or below one in both primary and swapped roles:

| Comparison | Primary topology / metric ratio | Swap topology / metric ratio | Compatible |
|---|---:|---:|---|
| R3-REM | 0.9852 / 27.5194 | 0.9966 / 26.6398 | yes |
| Other 11 REM/SWS comparisons | at least one topology ratio above 1 | metric ratios all above 1 | no |

Thus the exact frozen outcome is 1/6 REM, 0/6 SWS and 1/12 overall. Primary
topology alone was below one in 2/12 comparisons, while the swapped topology
was below one in 8/12, exposing substantial split sensitivity. There is no
animal-level population test; six modules come from three rats.

## Scale decomposition

For generalized-eigenvalue logs `ell_i` between the state and wake precision
matrices, the AIRM decomposes exactly as

$$
d_{\rm AIRM}^2
= d_{\rm scale}^2+d_{\rm shape}^2,
\qquad
d_{\rm scale}=\sqrt{6}\,|\bar\ell|,
\qquad
d_{\rm shape}=\left\|\ell-\bar\ell\mathbf 1\right\|_2.
$$

Across the 24 state-role matrices, the median shape share
`d_shape^2 / d_AIRM^2` was 0.02436, with range 0.00524--0.08141. Hence the
median squared distance was about 97.6% common scale. For example, Q1-REM
role A had total 5.245, scale 5.169 and shape 0.891; Q1-SWS role A had total
4.497, scale 4.452 and shape 0.634.

This exact postexecution decomposition does not alter the frozen estimator.
It shows that global activity/noise scale can explain most of the large raw
mobility distance. A scale-normalized SPD shape statistic, rate/variance
nuisance regression and state-matched bootstrap require a separately frozen
follow-up route; they cannot be added now as confirmatory repairs.

Final status: one R3-REM descriptive topology-mobility pattern was observed,
but physical fold geometry h, structural W, a local g field, curvature,
longitudinal deformation and held-out trajectory prediction were not measured.
