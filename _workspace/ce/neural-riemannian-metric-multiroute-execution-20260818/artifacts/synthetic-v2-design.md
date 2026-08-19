# Synthetic v2 frozen design

Status: INVALID_INCOMPLETE_SUPERSEDED_BY_V3

The source hash and specific defects are recorded in
`synthetic-v2-invalidation.md`. This document remains as the historical design
record only and is not executable evidence.

For G1--G6: d=3, eight nodes, 20 independent training circuits and 20
disjoint test circuits, each with 24 trajectories, 320 Euler-Maruyama steps
at dt=0.02. Training trajectories apply e1/e2 only; test trajectories apply
the actually simulated held e3 intervention over their last 160 steps. Training seeds are `11000+1000g+i`; test
seeds are `21000+1000g+i`, i=0..19.

The blind fitter API is `fit_candidates(train_observed)` followed by
`score_candidates(bundle, test_observed)`. Observations contain W, its frozen
Phi(W), paths, intervention direction and time only: no generator label,
truth, true g, drift or covariance is available to the fitter. Truth remains
evaluator-only.

G1 computes Phi(W) from normalized W degree, which causally controls alpha in
log g=alpha e3e3^T; e3 is the frozen common structural anisotropy. The SDE
uses Q=sigma^2 g^-1 and v=-g^-1Kx. The fitter estimates normalized precision
g_hat(w)=exp(S0+w*S1), sigma and K from training increments only, then scores
increments with Q_hat dt. It does not substitute an unrelated covariance for g.

Test Gaussian conditional log scores compare metric to five distinct frozen
baselines: unrestricted direct v/Q, gain-only, noise-only,
Euclidean/persistence (the one explicit consolidated baseline), and nonlinear
flat-pullback coordinates. Save g_hat, log-SPD direction/error and every score
for each of the fixed 20 test circuits per generator.

G1 requires finite SPD estimates, at least 18/20 positive log-SPD directional
alignments, and Holm-surviving held-out superiority against every baseline.
G2--G6 retain a 100-test fixed denominator for any Holm-surviving false metric
advantage; missing/nonfinite records fail.

Curvature is `NOT_IMPLEMENTED`: no C2 Riemann tensor plus chart-resampling
flat-pullback test is present, therefore v2 has status
`FAILED_INCOMPLETE_UNTIL_CURVATURE` and its verifier reports
`FAIL_CURVATURE_NOT_IMPLEMENTED`, never PASS or evidence.
