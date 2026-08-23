# BA-TR11 final report

Status: COMPLETE

Verdict: `CURVATURE_MEMORY_IDENTITY_REJECTED / DERIVED_NONLINEAR_SIGNATURE_PASS`.

Curvature is not the stored memory code. Distinct full-rank linear codes all
have constant pullback metrics and zero intrinsic curvature. The pre-learning
uniform matrix is degenerate, so its curvature is undefined rather than zero.

After a nonlinear tanh readout and a declared two-dimensional source plane,
the learned code induces a state-dependent metric and nonzero Gaussian
curvature. This is a derived geometric signature. It cannot identify hidden
labels: a hidden-row permutation changes every labeled source-to-hidden winner
while preserving the entire metric and curvature field to machine precision.

The next admissible question is functional rather than ontological: whether a
frozen curvature-derived cost predicts finite-amplitude saturation or route
distortion beyond the origin metric, under equal-small-signal controls.

