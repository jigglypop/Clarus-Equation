# Self-recursive cosmology prediction ledger

- passed: `True`
- predictions: 7

| target | layer | expected result | decisive check | falsifier | status | next artifact |
|---|---|---|---|---|---|---|
| real BAO+SN covariance | H0 q-selector | global/low or low-side bridge before H0 comparison | source/covariance graph gives q_F near the global family with public covariance labels fixed first | labelled covariance gives stable local/high q_F before any H0 refit | future data-facing | real BAO+SN Fisher JSON bundle |
| real SH0ES/CCHP ladder covariance | H0 q-selector | local/high or semi-local high depending on population closures | calibrator/anchor graph predicts q_F before final scalar H0 is used | source graph selects global/low despite endpoint-dominated calibration chain | future data-facing | ladder covariance role adapter |
| primitive spectrum joint likelihood | residual cascade | A_s, running, and tensor remain compatible with one N_e/A3c cascade | same Q_GER and N_e family fit scalar amplitude while r and alpha_spec remain within bounds | running/tensor data require an observable-specific projection not shared by A_s | future data-facing | primitive spectrum common-readout likelihood gate |
| CMB large-angle map/covariance likelihood | residual cascade | fixed A_H=2Q_GER/sigma improves or remains competitive against null after mask/trials handling | map likelihood tests fixed amplitude without fitting preferred-axis strength | fixed A_H performs worse than null or requires amplitude refit beyond uncertainty | future data-facing | CMB large-angle fixed-amplitude likelihood gate |
| FLRW/reheating/horizon scale lift | d0 measure transport | dimensionless S_R transport maps to scale quantities only with a derived physical scale | scale map derives curvature/reheating/horizon factor without importing observed H0 as the answer | scale lift needs an unconstrained calibration per target quantity | theory-facing | FLRW scale-lift derivation gate |
| late horizon readout dynamics | early-late bridge | late horizon entropy reads channel-corrected primordial phase measure | dynamical argument reproduces I_late = I_phase without changing pi^2/2 or q definitions | dynamics selects local slow-roll entropy growth instead of boundary phase measure | theory-facing | late horizon phase-readout dynamics gate |
| core kernel deformation | kernel guardrail | no c/kappa deformation is promoted unless fixed before data contact | new kernel term comes from independent derivation and improves multiple observables under AIC | best result relies on tuning c or kappa to one observable and degrades shared readouts | guardrail | kernel no-free-parameter derivation gate |
