# Mathematical successor audit

Status: COMPLETE

## 1. Target, domain, and claim boundary

The target is one prefix-only H20 forecast in the frozen four-chart,
dimensionless synthetic family.  A seed supplies `x[0:81]`; the candidate must
produce `x[81:101]` without target-window updates or hidden-state access.  For
training-only chart scales `s_j`, the seed endpoint remains

\[
R_s(m)=\left\{\frac1{80}\sum_{h=1}^{20}\sum_{j=1}^4
\left(\frac{x_{s,80+h,j}-\hat x^{(m)}_{s,80+h,j}}{s_j}\right)^2
\right\}^{1/2}.
\]

The independent inferential unit is a simulation seed, not a lead, chart, or
training window.  Everything below is a controller/dynamics proposal for this
synthetic family.  It is not an AGI, open-world causal-discovery, brain, or CE
physics result.

## 2. Why the V7 inverse-root controller failed

Let `e_i` be the stacked normalized error vector of expert `i` on the sole
pseudo-origin-60 H20 backtest.  V7 used

\[
E_i=\frac{\lVert e_i\rVert_2^2}{80},\qquad
w_i=\frac{(E_i+10^{-12})^{-1/2}}
{\sum_k(E_k+10^{-12})^{-1/2}}.
\]

Because `E_i` is MSE, this is inverse-RMSE weighting.  For the aggregate error,

\[
L(w)=\left\lVert\sum_iw_ie_i\right\rVert_2^2=w^TGw,
\qquad G_{ij}=e_i^Te_j.
\]

The V7 rule uses only the diagonal `G_ii`; it discards cross-expert error
covariance.  Convexity gives

\[
\left\lVert\sum_iw_ie_i\right\rVert^2
\leq \sum_iw_i\lVert e_i\rVert^2,
\]

but gives no inequality against the best component or the V5 parent.  Hence
nonnegative weights summing to one are not a no-regret anchor.

The disclosed V7 diagnostics are consistent with this defect:

- mean weights for `[sparse parent, adaptive dense, persistence]` were
  `[0.346157, 0.327509, 0.326334]`; the parent never received a majority weight
  (observed sparse-weight range about `0.198..0.474`);
- inner-to-outer absolute-error correlations were `-0.079` (sparse), `-0.029`
  (adaptive), and `+0.159` (persistence);
- inner-to-outer relative-advantage correlations were `-0.053` for sparse
  versus adaptive and `+0.041` for sparse versus persistence;
- winner agreement was `0.469` and `0.573`, respectively.

Thus the one inner H20 block carried almost no observed ranking information
for the next H20 block.  These are development diagnostics, not estimates from
fresh evidence.

The three registered failures have separate causes:

1. `V5 - V7 = -0.006491`, CI `[-0.027431,+0.014449]`: V7 had no parent-anchor
   constraint and mixed roughly two-thirds non-parent output into a parent
   that already had the best mean of its three experts.
2. `persistence - V7 = +0.010492`, CI `[-0.016487,+0.037470]`: the mean effect
   was small relative to paired SD `0.133150`; 96 seeds could not establish it.
3. Maximum component radius `1.114309 > 0.98`: development attribution shows
   the adaptive dense component was responsible (`19/96` seeds above `0.98`),
   while sparse max was `0.781420` and same-probe dense max was `0.821632`.

## 3. Exact stability result for the frozen V5 sparse parent

The frozen sparse mechanism has no cubic term and has

\[
D=\operatorname{diag}(0.14060858,0.11800805,0.78141996,0.76110462),
\]

with only `A -> C = +0.52184254` and `C -> D = -0.48295590`.  Therefore

\[
J_f(x)=D+B\operatorname{diag}(\operatorname{sech}^2x)
\]

is triangular in topological order for every `x`, so its eigenvalues are
exactly the diagonal of `D` and

\[
\rho(J_f(x))=0.78141996\quad\text{globally}.
\]

Write the rank-one residual state as a scalar `u`, with fitted unit direction
`v` and train-pooled AR coefficient `a=0.9369267940`:

\[
u^+=b+au,\qquad x^+=f(x)+c+v u^+.
\]

Its augmented Jacobian is block triangular,

\[
J_{\rm aug}(x)=
\begin{pmatrix}J_f(x)&av\\0&a\end{pmatrix},
\]

and consequently has global pointwise spectral radius `0.9369267940 < 0.98`.
There is also a common induced-norm certificate.  For

\[
\lVert(x,u)\rVert_W=
\max(|x_A|,|x_B|,|x_C|/3,|x_D|/8,|u|/0.04),
\]

using `|v_i| <= 1`, the five worst row bounds are

`[0.17809, 0.15549, 0.96786, 0.94690, 0.93693]`.
Hence every frozen-parent rollout step is a contraction by at most `0.96786`
in this particular weighted norm.  This result depends on the frozen DAG,
coefficients, zero cubic term, unit residual direction, and pooled AR.  It does
not transfer automatically to a refitted graph or adaptive dense model.

Component and output stability must remain distinct.  Spectral radius is not
convex: matrices

\[
A=\begin{pmatrix}.9&.4\\0&.9\end{pmatrix},\qquad
B=\begin{pmatrix}.9&0\\.4&.9\end{pmatrix}
\]

each have radius `.9`, while `(A+B)/2` has an eigenvalue `1.1`.  Moreover V7
mixed already-generated trajectories; it did not recursively apply one map,
so there is no valid identity `J_consensus=sum w_i J_i`.  A state-dependent
weight would add the further derivative term `sum F_i grad(w_i)^T`.

## 4. R1: training-only parent-anchored scalar projection

The strongest defensible next route is the two-endpoint segment

\[
c=p+g(s-p)=g s+(1-g)p,
\]

where `s` is the full frozen V5 sparse-parent trajectory and `p` is the
constant persistence trajectory.  On inherited observational-training windows,
stack normalized

\[
d=(s-p)/s_{\rm chart},\qquad r=(y-p)/s_{\rm chart}.
\]

Define

\[
\hat g=\operatorname{clip}_{[0,1]}
\frac{\sum\langle d,r\rangle}{\sum\lVert d\rVert^2},
\]

with `g=0` if the denominator vanishes.  The current development pilot used
22 nonoverlapping H20 origins `80,100,...,500` in each of eight inherited
training seeds: 176 fitted windows but only eight independent training
trajectories.  It obtained `g=0.7868543065`; leave-one-training-seed-out gains
ranged `0.754692..0.839645`.

### Calibration no-regret theorem

Let

\[
L_T(g)=\sum\lVert r-gd\rVert^2.
\]

This is a convex quadratic.  The displayed formula is its Euclidean projection
onto `[0,1]`, so a minimizer exists and, because both endpoints are feasible,

\[
L_T(\hat g)\le L_T(0)=L_T(p),\qquad
L_T(\hat g)\le L_T(1)=L_T(s).
\]

This is an exact theorem only on the fitted training windows.  It is not a
future, seedwise, or OOD no-regret theorem.

On disclosed V7 validation, now development data, R1 produced mean RMSE
`0.5449545190`, compared with parent `0.5569150278` and persistence
`0.5738975375`.  Paired lower endpoints were `+0.003779731` versus parent and
`+0.001995271` versus persistence.  These favorable values select a research
route; they do not confirm it.

R1 removes the offending adaptive component.  Its dynamic components are the
globally certified sparse parent and constant-anchor persistence.  If
persistence is implemented as `x^+=x_80` with the observed anchor frozen, its
internal-state Jacobian is zero.  Conditional on that frozen anchor, the H-step
output sensitivity is bounded by `g` times the sparse-parent sensitivity.  If
instead the derivative includes perturbing `x_80` itself, persistence contributes
an identity term; no `<0.98` input-to-output contraction follows.  The next
registration must name which derivative it gates.

The symmetric same-probe dense route, fitted by the identical training rule,
gave `g_dense=0.7835668487` and development RMSE `0.5453418191`.  Sparse/dense
geometric error ratio was `0.998972`, with paired log-ratio interval
`[-0.002117,+0.000059]`.  Thus R1 presently supports a shrinkage hypothesis,
not sparse-specific superiority.  The original V7 `+0.020552` sparse-ablation
checkpoint remains frozen development evidence and must be replayed unchanged;
it cannot be transferred algebraically to R1.

## 5. Structurally distinct alternatives

| Route | Definition | Fitted degrees of freedom | Analytic property | Main risk |
|---|---|---:|---|---|
| R1 training-only trajectory projection | `p + g(s-p)` with one global training gain | 1 global scalar; 0 evaluation-seed parameters | training calibration loss no worse than either endpoint; stable components | nonrecursive output, no OOD theorem |
| R2 recursive anchor homotopy | `x^+=x_80+g(F_s(x,u)-x_80)`, `u^+=b+au` | 1 global scalar | at `g=1` exactly V5, at `g=0` persistence; common-norm bound cannot exceed frozen-parent bound for `0<=g<=1` | unpiloted; recursive trajectory is not R1's trajectory blend |
| R3 stability-projected adaptive correction | anchor adaptive map at `x_80`, scale its recurrent part by `gamma=min(1,0.98/L_W)`, then mix with sparse map using a frozen coefficient | adaptive fit plus 1 controller scalar; projection gain is determined by the safety bound | common-norm contraction if both maps use the same fixed norm and weights are frozen | many more fitted coefficients; projection may collapse the dense expert and V7 showed poor transfer |

R1 is ranked first: it changes only one direction, uses no evaluation-prefix
weight fitting, exactly contains both failed comparators, and has the only
positive development pilot.  R2 is the stronger dynamical-stability experiment
but must be piloted separately.  R3 should be killed unless R1/R2 fail for a
mechanistic reason, not tried in parallel and selected on the same fresh block.

Free protocol choices for R1 are not zero: endpoint membership, clipping
interval, training-origin start, H20 stride, equal normalized squared loss,
and nonrecursive trajectory blending are six design choices.  The fitted model
has one continuous scalar.  The route was chosen after V7 disclosure and after
comparison with per-prefix gains, so look-elsewhere is real even though the
number `0.7868543065` itself used training data only.

## 6. Impossibility and tradeoffs

No nontrivial prefix-only predictor can be guaranteed to dominate the parent
on every continuation.  For a fixed prefix, if candidate `c` differs from
parent `s`, choose the future truth `y=s`; parent loss is zero and candidate
loss is positive.  The same construction applies to persistence.  Therefore
"safe anchor" can defensibly mean only:

1. exact inclusion of the parent endpoint;
2. training-calibration no-regret;
3. structural stability; and
4. a fresh paired kill gate against target degradation.

It cannot mean deterministic future dominance without additional assumptions
such as stationarity or a calibrated distribution-shift bound.  Forcing a high
sparse coefficient protects parent identity but limits possible persistence
gain; allowing it to approach zero improves adaptivity but weakens a sparse
contribution claim.  Exact persistence also has Jacobian one if represented as
the identity map, while the path-equivalent frozen-anchor representation has
Jacobian zero.  Stability claims are therefore representation- and
perturbation-dependent and must be registered explicitly.

## 7. Paired successor evaluation

Do not open or reuse the V7 locked test.  After R1, its gain, code, controls,
and derivative convention are frozen, use wholly fresh disjoint validation and
test seeds.  For every seed report paired differences against:

- the unmodified V5 sparse parent;
- persistence, which is also R1's exact no-sparse endpoint;
- the original V7 no-sparse controller as a frozen regression baseline;
- a same-rule, same-probe dense gain controller;
- the stable adaptive dense baseline.

Require positive Student-t 95% lower endpoints for `parent - R1` and
`persistence - R1`, the registered dense noninferiority bounds, finite paths,
zero future reads, and per-seed/per-component stability records.  Treat the
whole decision as one preregistered conjunction; all leadwise, H5, weight, and
winner analyses remain secondary.

Using the development paired effects and 20% SD inflation in

\[
n=\left\lceil
\left((z_{.975}+z_{.80})\,1.2s_d/\delta\right)^2
\right\rceil
\]

gives approximate floors `129` seeds for parent improvement and `239` for
persistence improvement.  Therefore a 96-seed successor is underpowered for
the smaller conjunctive endpoint if these development effects are the design
targets.  This calculation is planning only; no version or seed block should
be assigned until the algorithm is frozen.

## 8. Findings by severity

- **P0:** Calibration projection cannot be reported as future/OOD no-regret.
  Any such claim is defeated by the identical-prefix adversarial continuation.
- **P0:** V7 validation and the R1 pilot are development data; V7 test remains
  unopened and cannot confirm the selected successor.
- **P1:** Component radius, recursive-map contraction, and forecast sensitivity
  to the observed prefix are different quantities.  Register and report them
  separately.
- **P1:** R1's favorable pilot does not show sparse-specific superiority; the
  symmetric dense route was essentially tied.
- **P1:** The 176 training windows are dependent within eight seeds.  They fit
  one scalar but do not create 176 inferential replicates.
- **P2:** The generic Jacobian helper omits the derivative of a cubic local
  term.  V7's relevant mechanisms have cubic coefficient zero, but any future
  reuse with a nonzero cubic must add `3c_i x_i^2`.

## 9. Reproduction and scratch boundary

No locked test was run.  Development-only commands used were:

```powershell
$env:PYTHONPATH='reality_stone\python'
.\.venv\Scripts\python.exe _workspace\ce\agi-v8-breakthrough-20260811\scratch_diagnostics.py
.\.venv\Scripts\python.exe _workspace\ce\agi-v8-breakthrough-20260811\pilot_parent_anchor.py
```

Scratch inputs are
`_workspace/ce/agi-v8-breakthrough-20260811/scratch_diagnostics.py` and
`_workspace/ce/agi-v8-breakthrough-20260811/pilot_parent_anchor.py`.
Canonical files and locked-test artifacts were not modified.

CE_RUN=_workspace/ce/agi-v8-breakthrough-20260811
