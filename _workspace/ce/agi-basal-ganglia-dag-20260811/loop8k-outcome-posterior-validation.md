# Loop 8K validation — factorized outcome posterior

Status: LOCKED VALIDATION RUN ONCE — 80/100 STOP

The finite-state filter identities passed to floating-point tolerance: maximum
Bayes residual `5.55e-17`, posterior normalization error `4.44e-16`, transition
normalization error `4.44e-16`, zero degenerate evidence, no legacy decay, and
no explicit reset.

The candidate improved over the hard recurrent reference on both accuracy and
NLL: ID accuracy `0.7063` versus `0.6973`, OOD `0.3723` versus `0.3275`; ID NLL
`1.0802` versus `13.0875`, OOD `2.1373` versus `29.8352`. It also beat the
directional reset, support derangement, and outcome-sign-flip controls.

Two locked gates failed. Candidate-minus-signed accuracy LCB was `+0.00732` ID
and `+0.02869` OOD, below the required `+0.02` in ID. Post-switch LCB was
`+0.01407` ID and `-0.05123` OOD, below `+0.05` in both. The OOD environment
switches with hazard `0.12`, while the candidate correctly retained its locked
model hazard `0.06`.

Interpretation: factorized outcome credit fixes overall inference and
calibration, but a fixed transition prior cannot close the switch-rate shift.
The next admissible mechanism is an explicit posterior over transition hazard
or change-point responsibility. Retuning the fixed hazard to OOD is forbidden.
