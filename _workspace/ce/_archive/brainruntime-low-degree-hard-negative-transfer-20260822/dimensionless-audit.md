# BA-TR29 dimensionless audit

Status: `PASS`.

The raw cue coordinates, quadratic and affine predictions, true/decoy packet
descriptors, midpoint hard negative, current-column binding distances, norm
ratios, route margins, and success fractions are normalized synthetic
quantities and therefore dimensionless.  The midpoint coefficient `1/2`, all
thresholds, and all comparison ratios are dimensionless.  Runtime delays are
integer tick counts; the reported runtime cost is not a physical-energy
measurement.

The focused dimensionless regression was run together with the two changed
experiment tests and returned `24 passed` (two pre-existing PyTorch sparse-CSR
warnings).
