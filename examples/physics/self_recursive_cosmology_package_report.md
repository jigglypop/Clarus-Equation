# Self-Recursive Cosmology Package Gate

## Core numbers

| quantity | value |
|---|---:|
| D_eff | 3.17775842 |
| x | 0.04864672 |
| sigma | 0.95135328 |
| fixed-point residual | -1.041e-16 |
| contraction | 0.15458752 |
| N_e | 57.19965162 |
| H0 global q=0 | 67.247245 |
| H0 endpoint q=1 | 73.180689 |
| branch gap | 5.933444 |
| Q_GER | 0.02836622 |

## Promotion table

| lever | layer | status | evidence | guardrail | next gate |
|---|---|---|---|---|---|
| minimal fixed-point kernel | core kernel | Closed/minimal | residual=-1.0e-16, tuned AIC 10.493 > minimal AIC 2.684 | no c/kappa deformation without independent derivation | kernel derivation only, not observable tuning |
| d0 measure transport | boundary | Boundary principle | S_R identity error=+2.2e-15, contraction=0.15458752 | do not call d=0 a physical location or reachable state | FLRW/reheating/horizon scale lift |
| residual cascade | readout | Selection candidate | raw A_s pull=+191.18, GER A_s pull=+0.13 | no observable-specific recursion; same cascade must serve running/tensor/CMB handles | joint primitive-spectrum and CMB large-angle likelihood |
| H0 q-selector | selector | Channel-corrected Bridge | q-space chi2/dof=0.379/8 | q must be assigned from source/covariance graph before H0 comparison | real covariance/Fisher edge ingest |
| early-late phase measure | horizon bridge | Channel-corrected Bridge | phase_area=4.93480220, early-late chi2/dof=0.379/8 | not local slow-roll entropy growth and not Exact | dynamical derivation of late horizon readout |

## Verdict

The package is a Selection/Bridge package, not Exact.  The safe use of self-reference is now concentrated in readout/selector/measure-preservation layers.  Kernel deformation is blocked unless derived before data contact.
