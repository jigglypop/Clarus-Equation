# BA-TR28 dimensionless audit

Status: `PASS`.

The raw cue coordinates, the learned two-dimensional chart, the quadratic
feature vector, packet-content descriptors, fitted coefficient matrix, and
runtime activations are normalized synthetic quantities and therefore
dimensionless.  Feature-space residuals, condition numbers, cosine/norm
ratios, binding margins, and success fractions are consequently
dimensionless as well.  No physical time, energy, length, curvature, or
biological unit enters the learner or endpoint rule; delay values are integer
tick counts.

The focused dimensionless regression was run together with the two changed
experiment tests and returned `24 passed` (two pre-existing PyTorch sparse-CSR
warnings).
