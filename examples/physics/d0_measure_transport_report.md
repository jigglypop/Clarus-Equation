# d=0 Measure Transport Gate

## Transport state

| quantity | value |
|---|---:|
| x0 | 1.00000000 |
| sigma0 | 0.00000000 |
| S_R0 | 0.00000000 |
| D_eff | 3.17775842 |
| x | 0.04864672 |
| sigma | 0.95135328 |
| S_R | 3.02317090 |
| S_R-D_eff(1-x) | +2.220e-15 |
| contraction k | 0.15458752 |
| reverse amplification 1/k | 6.46882737 |
| N_e | 57.19965162 |
| S_R/N_e | 0.05285296 |
| residual after 10 steps k^10 | 7.79371261e-09 |
| residual after 20 steps k^20 | 6.07419563e-17 |
| steps for 1e-60 residual | 74.00 |

## Transport rows

| name | equation | value | status | guardrail |
|---|---|---:|---|---|
| identity boundary | x0=1, sigma0=0, S_R0=0 | 0.00000000e+00 | closed | boundary identity, not a physical location |
| recursive entropy | S_R=-log(x)=D_eff(1-x) | 3.02317090e+00 | closed dimensionless transport | not absolute thermodynamic entropy |
| branch contraction | k=D_eff*x | 1.54587523e-01 | stable forward d=3 branch | reverse map amplifies residuals; no finite-time arrival at d=0 |
| source measure | Q_source=x(1-x) | 4.62802163e-02 | closed residual source | must pass through readout taxonomy before becoming observable |
| half-cycle transport | Q_phase=(2/pi)Q_source | 2.94629008e-02 | projection candidate | not sufficient alone for A_s |
| +1 spatial transport | Q_GER=(2/pi)sigma^(D/(D+1))Q_source | 2.83662213e-02 | selection candidate | not exact until shared likelihood tests survive |
| curvature dilution | exp(-2N_e) | 2.07497978e-50 | closed dimensionless flatness direction | not an Omega_k measurement without FLRW scale map |
| recursive residual erasure | k^N_e | 4.17888453e-47 | closed dimensionless residual suppression | not reheating or horizon entropy |

## Verdict

d=0 remains usable as a zero-measure boundary condition.  The allowed transport is dimensionless entropy/residual projection, not motion to a physical d=0 state.

The next unresolved step is a true scale lift: derive how this dimensionless transport maps to FLRW curvature, reheating, or late horizon entropy without importing the answer.
