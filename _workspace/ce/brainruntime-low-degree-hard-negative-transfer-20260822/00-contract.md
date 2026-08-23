# BA-TR29 low-degree hard-negative transfer

Status before execution: `REVISION_1_CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR28 fitted all 400 rotating degree-two queries at machine precision, but
its randomly spaced three-packet endpoint let the affine approximation select
the same packet on as many as 60 percent of a seed.  BA-TR29 does not change
the learner, coefficients, degree, rank, cutoff, or runtime timing.  It changes
only the predeclared endpoint resolution on fresh seeds.

The primary, model-independent panel contains the true content $y_q$ and the
five closest other current content descriptors under Euclidean distance.  It
is randomly permuted over six fresh physical coordinates.  The degree-two
prediction must select $y_q$ on every rotating query.  The affine control may
select truth on at most $25\%$ of a seed.  Candidate norm ratio is at most
1.25, nearest relative separation at least $5\times10^{-3}$, and binding
margin at least $5\times10^{-3}$.

The secondary affine-resolution panel is derived only after both models are
fit and frozen from the same 24 observed rows.  Let

$$
p_q=\operatorname{QuadFit}(D_{-q},r_q),
\qquad
a_q=\operatorname{AffFit}(D_{-q},r_q),
\qquad
v_q=a_q-y_q.
$$

The R0 orthogonal-skew third descriptor crossed the positive packet floor on
calibration seed `116001`, so R0 stopped before development.  Revision 1 does
not lower that floor.  It uses the positive convex midpoint, giving the three
current descriptors

$$
\mathcal P_q=\left\{y_q,\ a_q,\ \frac{y_q+a_q}{2}\right\}.
$$

They are freshly permuted after fitting.  The quadratic prediction must select
$y_q$ and the affine baseline must select its own $a_q$ decoy on all 25 folds.
The model separation is at least $10^{-2}$, every panel component at least
0.1, norm ratio at most 1.75, and both binding margins at least $10^{-3}$.

Both panels use the actual $L=2$ runtime ring.  The three-packet receipts are
`[0,0,0,3,0,0,0]` and `[0,3,0,0,0,0,0]`; the six-packet receipts replace 3
by 6.  Runtime route error is at most $10^{-6}$ and wrong-route separation at
least $10^{-4}$.  Frozen controls are wrong cue, association shuffle rejected
before endpoint, packet-column shuffle, fixed canonical coordinate,
no-context/all-packet routing, finite lookup abstention, source/store cutoff,
and gate immutability.

Revision 1 calibration seed: `116002`.  Development seeds: `116101..116116`.  No panel,
distance, margin, degree, coefficient generator, seed, remap, timing, or
control changes after calibration.

Claim ceiling: this is a synthetic endpoint-discrimination stress test under
one declared degree-two data family.  The affine panel is deliberately
adversarial and the nearest panel is query-centered by the environment.  A
pass is not generic interaction/factor discovery, naturalistic candidate
generation, biological routing/memory, curvature/folding, energy, or AGI.
The query-only $y_q+\delta$ nonidentifiability from BA-TR28 remains.
