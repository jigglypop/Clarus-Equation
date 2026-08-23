# BA-TR13 fresh-geometry confirmation contract

Date: 2026-08-22

The BA-TR12 curvature and metric-strain costs, calibration radii `0,.1,...,.5`,
and learned BA-TR10 matrices are unchanged. Six new hidden rotations, eight
interleaved directions `(k+1/2)pi/8`, and held-out amplitude `1.25` are frozen
before execution. No BA-TR12 route angle or direction is reused.

The curvature selector is admitted only if, across all 16 matrices and 8
directions, its exact best-route hit rate is at least `.70`, its mean distortion
regret is at most `.01`, and both metrics are strictly better than the frozen
metric-strain selector and fixed-route-0 baseline. Equal origin metrics and a
signed-permutation equality receipt must pass at `1e-12`. Otherwise STOP.

The permanent flat nonlinear counterexample remains: even a pass cannot make
curvature sufficient or identify memory. This test concerns only restricted
out-of-catalog route selection in the declared tanh family.

