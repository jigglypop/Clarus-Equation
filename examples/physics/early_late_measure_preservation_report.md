# Early-Late Measure Preservation Gate

## Core invariant

| quantity | value |
|---|---:|
| D_eff | 3.17775842 |
| x | 0.04864672 |
| sigma | 0.95135328 |
| N_e | 57.19965162 |
| phase area pi^2/2 | 4.93480220 |
| d=3 adjoint phase measure | 4.93480220 |
| integrated defect pi delta sigma | 0.53127806 |
| endpoint defect delta sigma | 0.16911106 |
| I_phase | 282.26896669 |

## Channel-corrected preservation

| channel | q_graph | q_obs | invariant residual | invariant pull | H0_pred | H0 pull | role |
|---|---:|---:|---:|---:|---:|---:|---|
| Planck 2018 base LCDM | 0.0000 | 0.0268 | -0.004538 | -0.31 | 67.247 | -0.31 | global horizon |
| DESI DR2 BAO no-CMB calibration | 0.2500 | 0.2200 | +0.005070 | +0.30 | 68.684 | +0.30 | mostly global ruler |
| CCHP 2025 JWST-only JAGB | 0.1000 | 0.0968 | +0.000539 | +0.01 | 67.818 | +0.01 | endpoint diluted by population |
| CCHP 2025 TRGB HST+JWST | 0.5000 | 0.5402 | -0.006795 | -0.12 | 70.151 | -0.12 | mixed endpoint/global |
| SH0ES JWST update | 1.0000 | 0.9983 | +0.000292 | +0.01 | 73.181 | +0.01 | local endpoint |
| TDCOSMO+SLACS hierarchical lenses | 0.2500 | 0.0268 | +0.037740 | +0.35 | 68.684 | +0.35 | hierarchical lens |
| Megamaser Cosmology Project | 1.0000 | 1.1157 | -0.019563 | -0.24 | 73.181 | -0.24 | local geometric endpoint |
| GW standard siren representative | 0.5000 | 0.5250 | -0.004236 | -0.03 | 70.151 | -0.03 | mixed distance-redshift bridge |

## Verdict

invariant chi2/dof = 0.379/8
phase-adjoint error = 0.000e+00

Early-late measure preservation survives as a channel-corrected Bridge: source topology chooses q, and q-corrected late horizon readouts return to the same primordial phase measure.

This is still not an Exact theorem: the bridge must eventually ingest real covariance/Fisher edges and justify the late horizon entropy readout from dynamics.
