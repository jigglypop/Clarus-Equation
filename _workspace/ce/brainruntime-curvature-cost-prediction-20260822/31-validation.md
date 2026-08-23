# BA-TR12 validation

Focused test: `4 passed` in 5.68 s for
`tests/test_runtime_curvature_cost_prediction.py`. Source compile passed.

The exact flat nonlinear pair had identical origin metric and zero curvature
cost, yet held-out distortions `.2384058440` and `.1389428284`. This rejects
curvature-only sufficiency without an empirical fit.

Across the frozen 16 BA-TR10 learned matrices, six equal-origin-metric route
rotations, and eight directions, curvature cost selected the lowest-distortion
route with mean hit rate `.7890625` and mean regret `.00552670`. The matched
metric-strain cost achieved `.2734375` and `.03906262`. Origin-metric mismatch
was at most `5.32965e-15`; signed-permutation equality residual was at most
`3.74700e-16`. No output-side or semantic endpoint was opened.

