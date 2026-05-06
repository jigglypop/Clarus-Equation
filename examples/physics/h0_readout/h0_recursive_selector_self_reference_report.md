# H0 Recursive Selector Self-Reference Gate

## q-space closure

| channel | q_graph | q_obs | sigma_q | q pull | q_back drift | H0_pred | H0_obs | H0 pull | source-blind rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Planck 2018 base LCDM | 0.0000 | 0.0268 | 0.0877 | -0.31 | +0.00e+00 | 67.247 | 67.400 | -0.31 | early acoustic horizon endpoint only |
| DESI DR2 BAO no-CMB calibration | 0.2500 | 0.2200 | 0.1001 | +0.30 | -1.18e-13 | 68.684 | 68.510 | +0.30 | one local abundance anchor plus three global ruler/population anchors |
| CCHP 2025 JWST-only JAGB | 0.1000 | 0.0968 | 0.4745 | +0.01 | -4.71e-14 | 67.818 | 67.800 | +0.01 | one stellar endpoint diluted by nine population anchors |
| CCHP 2025 TRGB HST+JWST | 0.5000 | 0.5402 | 0.3252 | -0.12 | +1.01e-13 | 70.151 | 70.390 | -0.12 | one stellar endpoint and one cross-instrument closure |
| SH0ES JWST update | 1.0000 | 0.9983 | 0.1390 | +0.01 | -1.35e-13 | 73.181 | 73.170 | +0.01 | local Cepheid/SN endpoint closure |
| TDCOSMO+SLACS hierarchical lenses | 0.2500 | 0.0268 | 0.6405 | +0.35 | -1.18e-13 | 68.684 | 67.400 | +0.35 | one lens endpoint plus three hierarchical/global closures |
| Megamaser Cosmology Project | 1.0000 | 1.1157 | 0.4801 | -0.24 | -1.35e-13 | 73.181 | 73.900 | -0.24 | one-step local geometric distance |
| GW standard siren representative | 0.5000 | 0.5250 | 0.8664 | -0.03 | +1.01e-13 | 70.151 | 70.300 | -0.03 | absolute GW distance plus redshift/environment bridge |

## Verdict

q-space chi2/dof = 0.379/8
H0-space chi2/dof = 0.381/8
max algebraic q_graph -> H0_pred -> q_back drift = 1.349e-13

The algebraic loop closes by construction, so the real test is q_graph versus q_obs. This keeps the self-reference in the selector layer instead of fitting a new H0 correction per channel.
