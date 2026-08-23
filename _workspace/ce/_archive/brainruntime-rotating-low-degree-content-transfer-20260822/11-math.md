# BA-TR28 math note

Status: `PASS FOR CALIBRATION`.

The discarded 3-by-3 generic bilinear route has nine categorical bilinear
features.  Removing one of nine rows leaves design rank 8 while the augmented
train-plus-query design has rank 9.  Its query is therefore not identified;
regularization would merely choose an answer.

The replacement 5-by-5 fixture has the six full quadratic features
$1,z_1,z_2,z_1^2,z_1z_2,z_2^2$.  Every 24-row rotating training design has
rank 6, so its query feature lies in the row span and $\widehat y_q$ is unique
inside the declared class.  All variables are normalized and dimensionless;
the pseudoinverse, residual ratios, singular-value cutoff, condition number,
binding margin, and runtime proxy amplitudes are therefore dimensionless.

This does not identify an arbitrary missing value.  Two worlds with the same
24 observations and query cue but values $y_q$ and $y_q+\delta$ are
observationally identical to the learner.  Conditional interpolation and
query-only class-external detection cannot both be claimed without an
additional observed response channel.
