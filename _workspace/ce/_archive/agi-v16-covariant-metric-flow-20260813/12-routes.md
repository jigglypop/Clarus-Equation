# V16 alternative-route audit

Status: COMPLETE

## 1. Fixed objective and comparison discipline

The route must learn a hidden full SPD quadratic cost from only the executed
pair $(x_t,c_t)$ while retaining exactly one persistent semantic tensor $g_t$.
The required symmetry is covariance under every $J\in GL(d)$, not merely under
rotations.  SPD preservation, one-state compliance, and the G-CHART condition
are structural gates; finite development regret cannot compensate for their
failure.

All benchmark learners receive the same seeded candidate generator and noise
law but select their own actions, so their executed training streams can differ.
This is a fair end-to-end closed-loop comparison, not a pure optimizer
comparison.  A pure optimizer comparison would additionally require a shared
logged excitation stream.  No confirmation seed was opened in this lane.

Let $m=d(d+1)/2$, equal to 3, 6, and 10 for $d=2,3,4$.

## 2. Route matrix

| Route | Persistent learned dof | General affine covariance | SPD guarantee | Per-step work | Identifiability | Target-aware status |
|---|---:|---|---|---|---|---|
| R-A: V16.1 affine-invariant rank-one flow | $m$; only $g$ | Yes, exactly in real arithmetic and without reprojection | Yes; congruence factor has eigenvalues $1$ and $(c/p)^\eta>0$ | $O(d^2)$ | Full $g_*$ only if $\{xx^T\}$ spans $\operatorname{Sym}(d)$; this does not by itself prove convergence | The structure was engineered for M1--M4 and uses executed $c$; $\eta$ is selected on development regret only |
| R-B: additive log-loss SGD plus spectral projection | $m$; only $g$ | No; both the raw $xx^T$ step and eigenvalue clipping depend on chart | Yes after a successful finite eigendecomposition and clipping | $O(d^3)$ because of projection | Same full-span information condition; projection can bias a chartwise optimum | Contract-mandated full-matrix comparator; its own $\eta$ is selected on development regret |
| R-C: conformal $g=\alpha I$ | 1 | Only for similarity charts $J=sQ$, not general $GL(d)$ | Yes, since the scalar update is exponential | $O(d)$ | A scalar is identifiable for exactly conformal truth; anisotropic $g_*$ is outside the model regardless of excitation | Structural low-capacity control; its own $\eta$ is selected on development regret |
| R-D1: vectorized batch/RLS | $m+m(m+1)/2$ if $g$ and symmetric information state $P$ persist: 9, 27, 65 | Conditional only if the fourth-order $P$ and prior are transported; $P_0=\delta I$ is not generally invariant | No for unconstrained RLS; projection reintroduces the R-B chart defect | $O(m^2)=O(d^4)$, plus $O(d^3)$ if projected | Full feature rank can identify the noiseless quadratic, subject to consistency | Analytically attractive because it uses target residuals and curvature memory, but extra tuning and state make it ineligible in this run |
| R-D2: one-state log-Euclidean gradient | $m$ | No under general congruence; matrix log is compatible with orthogonal similarity, not arbitrary $J^{-T}gJ^{-1}$ | Yes through the matrix exponential | Normally $O(d^3)$ | Same full-span information boundary; optimization convergence is separate | Target-loss driven, but adding it to the scored family now would be an unregistered model search |

Detailed algebra and numerical counterexamples are in
`artifacts/route-comparison-calculations.md`.

## 3. R-A: primary route and H1 boundary

V16.1 is the only route in this comparison that simultaneously satisfies the
one-state restriction, exact general-affine covariance, real-arithmetic SPD,
and $O(d^2)$ work.  It also fits the selected observation geometrically:

$$
p^+=p^{1-\eta}c^\eta.
$$

A new proof route exists for the noiseless case.  For
$c=x^Tg_*x$, the log-det/Burg divergence

$$
\mathcal D(g_*\Vert g)
=\operatorname{tr}(g^{-1}g_*)-\log\det(g^{-1}g_*)-d
$$

obeys, with $z=c/p$,

$$
\mathcal D(g_*\Vert g^+)-\mathcal D(g_*\Vert g)
=z^{1-\eta}-z+\eta\log z
\le\eta(1-z+\log z)\le0. \tag{R-A1}
$$

This closes a narrower route-lane theorem candidate for a finite nonzero
spanning direction set visited at least once in every fixed-length window.
Monotone convergence of $\mathcal D_t$ forces the actual residual to zero;
compactness then forces update norms to zero.  A bounded number of intervening
vanishing updates transfers that limit to every spanning direction at every
late time, and injectivity of the finite-dimensional measurement map forces
$g_t\to g_*$.  The complete proof chain is in the calculation artifact and
still requires independent math/audit promotion.

It does not prove the contract's noisy H1.  With continuing iid multiplicative
noise and fixed $\eta$, the $d=1$ recursion is an AR(1) process in $\log g$ with
positive stationary variance.  Exact convergence to $g_*$ is therefore false.
H1 must remain open or be narrowed to one of: noiseless convergence, a
stationary error bound, expected finite-run risk, or diminishing-$\eta_t$
stochastic convergence.  The current fixed-rate benchmark measures finite-run
risk only.

Killing tests for R-A are:

1. an unused scaled shear $J$ must give update agreement at machine tolerance
   without projection;
2. $p^+/c=(p/c)^{1-\eta}$ and SPD must survive residuals near the finite numeric
   boundary, otherwise the routine must reject explicitly;
3. a rank-deficient direction family must expose two distinct observationally
   identical SPD tensors rather than suggest identification;
4. the one-dimensional persistent-noise construction must reject any claim of
   fixed-rate point convergence.

## 4. Why the alternatives are not implementation route 1

R-B has equal parameter count and is therefore the correct learned
full-matrix comparator, but it fails before projection.  For
$g=I$, $x=e_1$, $c=e$, $\eta=0.1$, and $J=\operatorname{diag}(2,1)$, updating
then transporting gives first diagonal entry $0.275$, whereas transporting then
updating gives $0.65$.  Projection did not activate.  Its eigenvalue floor and
cap are a second noncovariant operation.  Killing tests are this exact
counterexample, an unused shear metamorphism, and an input that activates both
clipping bounds.

R-C is useful because it isolates whether the environment needs anisotropy.  It
cannot distinguish equal-Euclidean-norm candidates.  With
$g_*=\operatorname{diag}(0.25,4)$, $e_1$ and $e_2$ have true costs $0.25$ and
$4$ but receive the same conformal prediction.  Its killing tests are this
equal-norm ranking task and any nonsimilarity chart transform.  Passing finite
regret on nearly isotropic episodes would not repair this structural deficit.

R-D1 can be sample-efficient because $P$ records excitation geometry, and a
properly transported fourth-order $P$ can in principle make the algebra
covariant.  That strength is precisely extra persistent optimizer state, which
the contract forbids.  Unconstrained RLS also lacks an SPD invariant; spectral
repair reduces it to the R-B covariance problem.  Its killing tests are the
persistent-state ledger, transformed-prior metamorphism, an inconsistent noisy
positive-cost set producing an indefinite regression estimate, and measured
$O(d^4)$ work.

R-D2 should not be confused with R-A's affine-invariant exponential map.
Although $\exp H$ is SPD, $H=\log g$ does not transform tensorially under a
general congruence.  For $g=I$ and $J=\operatorname{diag}(2,1)$, the original
log state is zero while the transported metric has log
$\operatorname{diag}(-\log4,0)$.  Its killing tests are this state-level chart
defect followed by a one-step scaled-shear action disagreement.

## 5. Hyperparameters, target awareness, and look-elsewhere control

For the registered R-A, R-B, and R-C learners, the only searched value is their
separate learning rate in
$\{0.05,0.1,0.2,0.4,1.0\}$.  Each rate is chosen by mean held-out route regret
on development seeds 917000--917063, with the smaller rate winning exact ties.
Using a single common rate would unfairly favor one geometry.  The additive
projection interval $[10^{-6},10^6]$, noise law, horizon, and exploration rule
are fixed by the contract and are not searched.

The selection metric is target-aware: it uses development route regret, rather
than invariant tensor error.  This is declared model selection, not
confirmation evidence.  Hidden $g_*$ may be used by the development evaluator
to compute regret but is never supplied to an update.  Confirmation seeds may
be opened only after code, per-learner rates, gates, and artifact hash are
frozen.

RLS and log-Euclidean variants would introduce new choices such as prior scale,
forgetting factor, ridge/noise level, projection rule, or matrix-gradient
variant.  They remain analytic routes only.  Tuning or adding either to the
confirmation comparison requires a new preregistered run; it may not be used as
a post-confirmation rescue.

The development-only scratch run in `artifacts/explore_metric_flow_dev.py` and
`artifacts/explore_metric_flow_dev.log` reported the following lowest-regret
choices on seeds 917000--917063:

| Learner | Development-selected $\eta$ | Held-out accuracy | Mean held-out regret | Median invariant metric error |
|---|---:|---:|---:|---:|
| V16.1 | 0.4 | 0.964355 | 0.000465892 | 0.0357665 |
| Additive | 0.2 | 0.909912 | 0.00565183 | 0.146190 |
| Conformal | 0.05 | 0.247070 | 0.427350 | 0.680958 |
| Identity | not applicable | 0.247070 | 0.427350 | 0.826997 |

These are target-aware development diagnostics, not evidence for a confirmation
gate and not a convergence proof.  Every protocol candidate has unit Euclidean
norm and every held-out route contains exactly two unit vectors, so a conformal
learner predicts mathematical ties for all choices, independent of $\alpha$
and $\eta$.  An initial raw `numpy.argmin` probe spuriously made choices depend
on normalization-roundoff differences.  The contract and scratch probe were
repaired to use the declared tolerance tie with lowest-index selection before
confirmation was opened.  The corrected conformal action metrics are identical
to identity for all five rates; the smaller-rate tie rule therefore selects
$\eta=0.05$.  Its metric error can still differ because $\alpha$ is learned,
but that scalar cannot affect ranking in this unit-vector protocol.  This is a
structural-control result, not a learned-action improvement.

## 6. Strict route selection

**Implementation priority 1: R-A, equation (V16.1).**

The selection is structural: R-A is the sole route with $m$ persistent degrees
of freedom, exact $GL(d)$ covariance, unconditional real-arithmetic SPD, and
$O(d^2)$ work.  R-B and R-C remain mandatory scored controls.  R-D1 is excluded
from the one-state claim, and R-D2 is excluded from G-CHART and from the frozen
benchmark family.

This selection does not promote H1.  A V16 narrow GO still requires every
preregistered math, numeric, learning, chart, and closed-loop gate.  The most
promising theorem extension after the frozen implementation is the noiseless
bounded-gap spanning result suggested by (R-A1), followed by a separately
contracted stationary-noise or diminishing-step analysis.
