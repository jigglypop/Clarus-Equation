# AGI V16 covariant metric-flow agent: research contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v15-unified-metric-score-20260813

Mode: full. V15's proved affine readout and no-go results are referenced, not
rederived. V16 introduces a new metric-learning law and new scored claims.

## 1. Research question

Can one symmetric positive-definite metric $g_t$ remain the only persistent
semantic state while a finite agent (i) learns it from executed displacement
and scalar cost observations, (ii) uses it for prediction, memory, planning,
critic, and goal readouts, and (iii) closes an action--environment--update loop?

The sought result is a narrow mechanism breakthrough, not an AGI declaration.
Static Riemannian symmetry, continuum SCC convergence, delayed credit, raw-pixel
perception, biology, cosmology, and unrestricted intelligence remain outside the
positive claim unless separately scored.

## 2. Domain and notation

- Dimension $d\in\{2,3,4\}$.
- Semantic state $g_t\in\mathbb S_{++}^d$; its canonical numerical storage may
  be one lower-triangular positive-diagonal factor $L_t$ with $g_t=L_tL_t^T$.
  The factor is an encoding of $g_t$, not a second semantic state. No
  role-specific persistent weights, replay
  buffer, optimizer moment, covariance estimator, or eligibility trace.
- Executed displacement $x_t\in\mathbb R^d\setminus\{0\}$.
- Observed positive squared cost $c_t>0$.
- Prediction $p_t=x_t^Tg_tx_t>0$.
- All synthetic features and squared costs are dimensionless. Equivalently, in
  a dimensional application $p_t$ and $c_t$ must use the same reference unit,
  so $p_t/c_t$ is dimensionless.
- Learning rate $0<\eta\le1$ and dimensionless residual

$$
r_t=\log\frac{p_t}{c_t}.
$$

The candidate rank-one metric flow is

$$
a_t=e^{-\eta r_t}-1,
\qquad
g_{t+1}=g_t+\frac{a_t}{p_t}(g_tx_t)(g_tx_t)^T. \tag{V16.1}
$$

The five current readouts are deliberately narrow: world prediction is
$x^Tg_tx$, memory is the changed $g_t$, planning sums this cost along candidate
routes, critic is $r_t$, and goal is the minimum predicted route. Only $g_t$ is
persistent semantic state; the environment, current observation, episode index,
and transient candidate list are not agent memory.

## 3. Mathematical claims to audit

### M1. SPD preservation

For every allowed input, equation (V16.1) maps SPD $g_t$ to SPD $g_{t+1}$
without spectral clipping.

### M2. General affine covariance

For $y=Jx$, $h=J^{-T}gJ^{-1}$ and the same scalar $c$, updating $(h,y,c)$
must equal $J^{-T}g_{t+1}J^{-1}$ for every $J\in GL(d)$.

### M3. Exact same-observation residual contraction

If the updated metric is queried again on the same $x_t$, then

$$
p_{t+1}=p_t^{1-\eta}c_t^\eta,
\qquad
\log\frac{p_{t+1}}{c_t}=(1-\eta)r_t. \tag{V16.2}
$$

At $\eta=1$, the selected observation is fit exactly in exact arithmetic.

### M4. Natural-gradient identity

Equation (V16.1) is the affine-invariant SPD exponential-map step for
$\frac12[\log(p/c)]^2$. This status requires a complete tangent-gradient and
rank-one exponential proof; numerical agreement is insufficient.

### M5. Measurement identifiability boundary

The hidden SPD metric is uniquely identifiable from noiseless pairs
$(x_t,c_t=x_t^Tg_*x_t)$ iff the matrices $x_tx_t^T$ span
$\operatorname{Sym}(d)$. If they do not span it, the verifier must construct two
distinct SPD metrics with identical measurements.

### H1. Multi-direction convergence

Global convergence of repeated noisy rank-one updates under persistent
excitation is an open hypothesis. It may be promoted only with a proof; finite
benchmark success remains a numerical result.

## 4. Numeric repair claims inherited from the V15 STOP

R1. Public shortest-path reconstruction terminates with a simple path or raises
a bounded explicit exception; it never follows a predecessor cycle.

R2. Distance relaxation uses strict improvement. Any tie-count structure must
be a distance-oriented DAG with $D(u)<D(v)$ on every predecessor edge. Exact
float ties may be reported as ties; numerically ambiguous ties must not replace
the representative strict-relax predecessor.

R3. Stable local and edge lengths are correct at coordinate scales
$10^k$, $k\in\{-150,-16,0,16,150\}$, whenever the mathematical length is
representable. Nonrepresentable public output is rejected explicitly rather
than returned as NaN or an invalid SPD state.

R4. Surprise gating remains correct for reference scales $10^{-200}$ through
$10^{200}$ by comparing in a stable ratio or log domain. A diagnostic magnitude
may saturate at 0 or infinity, but the hard gate may not flip because of that
saturation.

R5. The V16 update is evaluated through a stable factor/congruence algorithm,
not the subtractive outer-product formula when that would destroy a
representable positive eigenvalue. All accepted factors have finite entries and
strictly positive diagonal. Inputs for which the mathematically required factor
or public scalar cannot be represented are rejected explicitly.

The unique chain $(0,10^{-16},2\times10^{-16})$, source 2, target 0, the eight
V15 adversarial probes are mandatory regression inputs.
The V16 numeric killing cases include scalar transitions
$(g,x,c,\eta)=(1,1,10^{-300},1)$,
$(1,10^{-150},1,1)$, $(10^{308},1,10^{-308},1/2)$,
and representable near-equality residuals. The min-subnormal endpoint is
diagnostic only because a positive Cholesky factor whose square equals the
smallest subnormal is not itself representable in binary64.

## 5. Alternative routes to compare before implementation

At least three routes are required:

- R-A: equation (V16.1), affine-invariant rank-one natural flow;
- R-B: one-state additive Euclidean log-loss SGD with SPD projection;
- R-C: conformal scalar metric learning;
- optional R-D: batch/RLS or log-Euclidean methods, counted as extra-state or
  noncovariant when applicable rather than silently treated as one-state.

Each route must report persistent state degrees of freedom, coordinate
covariance, SPD guarantee, per-step complexity, identifiability requirements,
and a killing test. Target-aware choices and hyperparameter searches must be
recorded.

## 6. Synthetic closed-loop protocol

No hidden metric tensor is supplied to the agent. For each seed:

1. draw $d=3$ and hidden $g_*$ with Haar-like random eigenvectors and eigenvalues
   log-uniform on $[0.25,4]$; specifically use `numpy` QR on an i.i.d. standard
   normal $3\times3$ matrix and form $Q\operatorname{diag}(\lambda)Q^T$;
2. initialize the learner with $g_0=I$;
3. for 128 steps draw four i.i.d. standard-normal vectors and normalize each to
   Euclidean unit length; a zero draw is deterministically rejected and redrawn;
4. on steps $t\equiv0\pmod4$ execute candidate $(t/4)\bmod4$; otherwise execute
   the minimum predicted $x^Tg_tx$. Values within
   $64\epsilon\max(1,\max_k|p_k|)$ of the minimum are a declared numerical tie,
   broken by lowest index; the evaluator uses the same explicit decision
   convention but computes its costs independently;
5. the environment returns only the executed $x_t$ and
   $c_t=x_t^Tg_*x_t\exp(0.05z_t)$, $z_t\sim N(0,1)$;
6. update from that single observation; unexecuted true costs are used only by
   the evaluator to compute regret;
7. after training, score 64 new four-choice route sets. Each route contains two
   fresh independently normalized standard-normal displacements and its
   noiseless cost is the sum of the two quadratic costs. Online regret is also
   scored against noiseless hidden costs and normalized by the true candidate
   minimum; observation noise is used only in the learner update.

Development seeds are 917000--917063. Candidate learning rates are
$\{0.05,0.1,0.2,0.4,1.0\}$, selected per learner by lowest development mean
held-out regret, with the smaller rate winning ties. Confirmation seeds
918000--918255 are opened once after code, rate choice, thresholds, and artifact
hash are frozen.

The baselines are:

- identity $g=I$, no learning;
- conformal $g=\alpha I$ with
  $\alpha^+=\alpha\exp[-\eta\log(\alpha\lVert x\rVert^2/c)]$;
- same-state full additive learner

$$
g^+=\Pi_{[10^{-6},10^6]}
\left(g-\eta\frac{r}{p}xx^T\right), \tag{V16.3}
$$

  using the same selected observation and its own development-selected rate.

Identity and conformal are structural controls. Equation (V16.3) is the learned
full-matrix comparator. It has the same parameter count but not identical
operations; no claim of exact compute matching is allowed without a measured
compute ledger.

## 7. Metrics and preregistered gates

The coordinate-invariant metric error is the RMS log generalized eigenvalue of
$(g_*,g)$; prediction and regret ratios are dimensionless.

### G-MATH

M1--M5 must all close with killing tests and no open P0.

### G-NUMERIC

R1--R5 and all prior adversarial fixtures must pass; all public outputs in the
declared representable domain must be finite and valid.

### G-LEARN

On confirmation seeds, V16 must satisfy all of:

- finite-run rate 100%;
- held-out route accuracy at least 90%;
- mean normalized held-out regret at most 0.05;
- median invariant metric error at most 0.25;
- mean regret improvement over identity at least 0.10;
- mean regret improvement over conformal at least 0.05;
- no more than 0.02 mean-regret worse than the additive full-matrix learner.

### G-CHART

For each confirmation episode, transport the initial state, hidden metric,
candidate sequence and observations by an unused affine $J$ with singular
values in $[0.1,10]$, without reprojection. For confirmation episode seed $s$,
create an independent generator with seed $s+1{,}600{,}000{,}000$. Draw two
i.i.d. standard-normal $3\times3$ matrices, take their reduced QR decompositions,
and multiply each column of $Q$ by the sign of the corresponding diagonal of
$R$ (using sign $+1$ when that diagonal is exactly zero). Draw three i.i.d.
$u_k\sim U[-\log10,\log10]$, set
$J=Q_1\operatorname{diag}(e^{u_1},e^{u_2},e^{u_3})Q_2^T$, and use exactly this
one $J$ for that episode. A numerically singular or nonfinite draw is an episode
failure, not a resampling opportunity. V16 must have 100% action agreement
and maximum relative prediction difference at most $10^{-10}$. This is a chart
metamorphism, not semantic OOD.

### G-CLOSED-LOOP

The action must be selected before its cost is observed, only the executed
action may update $g_t$, and mean normalized online regret after the first 32
steps must be lower than identity by at least 0.05.

## 8. Decision rule and AGI boundary

- `V16 NARROW GO`: G-MATH, G-NUMERIC, G-LEARN, G-CHART and G-CLOSED-LOOP all
  pass on the unopened confirmation block.
- `V16 STOP`: any required gate fails.
- A narrow GO may count evidence for vector-observation metric learning and a
  synthetic closed loop. It does not establish raw sensory representation,
  delayed credit assignment, learned compute-matched semantic OOD, tool use,
  continuum SCC convergence, or AGI.
- `AGI GO` is forbidden in this run. The strongest permitted conclusion is a
  one-state covariant metric-learning primitive with specified finite scope.

## 9. Reproducibility and contamination controls

- No external data or literature is required; the sources lane is skipped.
- Development and confirmation results must be stored separately.
- Confirmation may be executed once. A code or threshold change afterward
  requires a new run and new seed block; it cannot repair this score.
- Before confirmation, record a single SHA-256 manifest covering the production
  module, public export file, confirmation evaluator, this contract, selected
  per-learner rates, and all gate thresholds. The evaluator must verify every
  entry before opening seed 918000. A mismatch aborts without scoring and
  requires a new run; the manifest itself is stored in `artifacts/`.
- Numerical agreement is never substituted for M1--M5 proofs.
- Existing repository changes outside this run and the later Gate-approved
  implementation scope are preserved.
