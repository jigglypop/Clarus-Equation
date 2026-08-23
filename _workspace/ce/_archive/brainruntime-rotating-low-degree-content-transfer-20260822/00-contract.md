# BA-TR28 rotating low-degree content transfer

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR27 selected from a hand-enumerated three-class table.  BA-TR28 removes
that table.  It uses one generic full degree-two vector operator on an opaque
rank-two cue plane and rotates every one of 25 cells through the missing-query
position.

For learned intrinsic coordinates $z=(z_1,z_2)$, define the fixed
dimensionless feature vector

$$
\phi(z)=(1,z_1,z_2,z_1^2,z_1z_2,z_2^2)^\top\in\mathbb R^6.
$$

The learner receives only 24 shuffled pairs $(r_i,y_i)$ in each fold.  It
centers the raw cues, discovers their rank-two plane by SVD, builds $\Phi$
from the projected coordinates, and fits

$$
\widehat C=\Phi^+Y,
\qquad
\widehat y(r_q)=\phi(z_q)^\top\widehat C.
$$

The relative SVD cutoff is $10^{-10}\sigma_{\max}$.  Every fold must have cue
rank 2, feature rank 6, condition number at most $10^4$, observed relative fit
error at most $10^{-10}$, query cue/feature span error at most $10^{-10}$, and
held-out relative content error at most $10^{-9}$.  The full quadratic block
must carry at least 5 percent of the coefficient Frobenius norm.  The matched
affine model must have mean rotating-query error at least $10^{-3}$ and may
route at most half of the queries correctly.  The exact finite lookup has no
query row and must abstain.

Each seed uses fresh raw-cue orientation and fresh dense signed coefficients;
the resulting six-vector content is shifted and globally scaled before the
experiment so all runtime packet columns remain positive.  At each query, the
correct current packet and two nonzero distractors arrive simultaneously at
fresh physical coordinates.  The gate compares its prediction only with the
three current $H\leftarrow I$ columns and requires relative best/second margin
above $10^{-4}$.  A runtime-local source mask then transmits the compiled
coordinate through the unchanged $L=2$ ring.  Required packet and write
receipts are `[0,0,0,3,0,0,0]` and `[0,3,0,0,0,0,0]`.

Frozen controls: affine degree one, absent-row exact lookup, wrong cue,
cue/content association shuffle rejected before endpoint, packet-column
shuffle, fixed canonical coordinate, and no-context/all-packet routing.  Row
order and an orthogonal raw-cue chart change must preserve predictions.  All
25 cells are held out exactly once per seed.  Candidate norm ratio is at most
1.75, learned/oracle runtime relative error at most $10^{-6}$, and a wrong
current packet must differ by at least $10^{-5}$.

The strongest no-go is part of the result, not a tunable control.  Replacing
only the unobserved value by $y_q+\delta$ leaves the 24 fit rows unchanged, so
no fit-only learner can both predict the in-class value and detect that
query-only mutation.  BA-TR28 must record this as
`QUERY_ONLY_OUTSIDE_CLASS_NONIDENTIFIABLE`; it must not call it learned
abstention.  An off-plane raw cue, which is observable, does fail closed.

Fresh calibration seed: `115001`.  Fresh development seeds:
`115101..115116`.  Degree, feature order, ranks, cutoffs, coefficient
generator, all 25 rotations, thresholds, event timing, remaps, distractors,
controls, and seed rows are frozen before calibration.  No development or
confirmation endpoint may change them.

Claim ceiling: a pass means conditional synthetic degree-two interpolation
and current-packet routing on this frozen distribution.  It is not sparse
support recovery, unrestricted interaction or factor discovery, semantic
memory, biological routing, curvature/folding evidence, physical energy, or
AGI evidence.
