# Alternative routes after the V7 checkpoint

Status: COMPLETE

## 1. Fixed target, domain, and status boundary

For the frozen, fully observed four-chart synthetic family, the target is a
positive reduction in independent-seed normalized H20 free-rollout RMSE versus
both the V5 sparse parent and persistence, while retaining a sparse/no-sparse
ablation and keeping every recursively evaluated dynamic component at or below
the inherited `0.98` stability limit.  Every predictor may read only the
training/probe data and the episode prefix `x[0:81]`.

This is an algorithm-development question about one matched-basis synthetic
forecast family.  It is not an AGI, open-world causal-discovery, biological, or
physical-CE claim.  All route numbers below are empirical development results,
not confirmation.  The disclosed V7 validation seeds were used only as
development data; the locked V7 test split was not simulated or opened.

## 2. Frozen checkpoint versus remaining freedom

### 2.1 Frozen

Every route below preserves the following checkpoints and protocol constraints.

1. One origin at 80, one uninterrupted H20 path, H5 equal to the first five
   rows of that path, no target-window update, no hidden-state read, and an
   instrumented maximum observation index of 80.
2. Training-only chart scales, seed-level paired inference, finite component
   trajectories, symmetric equal-probe dense controls, and immutable future
   registrations.
3. The historical V7 sparse addition checkpoint remains in the ledger:
   `no_sparse - sparse = +0.020552`, paired 95% interval
   `[+0.008174, +0.032930]`.  It is not reinterpreted as sparse-specific causal
   superiority because the V7 sparse and symmetric-dense consensuses were
   effectively tied.
4. The V7 failures remain failures: improvement over V5 had mean `-0.006491`,
   improvement over persistence had mean `+0.010492` with a lower interval
   below zero, and the maximum dynamic pathwise radius was `1.114309`.
5. All V1--V7 artifacts and unopened historical test splits remain untouched.

### 2.2 Free

The only free choice is the next controller, dynamics projection, or residual
state estimator.  No new version number, seed block, gate, or noninferiority
margin is assigned here.  A route must be selected and fully frozen before a
new validation block is generated.

The disclosed V7 data localize the stability failure.  Recalculation gave
maximum pathwise radii `0.781420` for the V5 sparse mechanism, `0.821632` for
the same-probe dense mechanism, and `1.114309` for the episode-adaptive dense
mechanism.  Thus removing or analytically projecting the adaptive component is
a targeted repair; changing the already-stable sparse mechanism is not.

## 3. Look-elsewhere disclosure

These routes are target-aware at the **family-selection** level because they
were proposed after V7 validation was disclosed.  The fitted coefficient in
Route 1 is nevertheless computed from inherited observational-training data
only and is invariant to every V7 validation value.

At least five candidate specifications were inspected across this development
run: training-only global shrinkage, the three-block median gate, a Kalman
residual filter, an unanchored contractive consensus, and the subsequently
anchored contractive consensus.  The last route was formed after seeing the
unanchored contractive pilot, so its target-awareness is especially high.
Mandatory no-sparse and symmetric-dense controls are not counted as extra
candidates because they cannot be selected as the sparse route.  This is a
lower bound on search, not a multiplicity-adjusted experiment.  None of the
development intervals below may be cited as fresh evidence.

## 4. Route R1 -- training-only convex correction shrinkage

### 4.1 Algorithm

Let `S` be the frozen V5 sparse-parent H20 path and `P` the persistence path.
Fit one global coefficient using inherited observational-training episodes
only:

\[
d_w=\frac{S_w-P_w}{s},\qquad
r_w=\frac{Y_w-P_w}{s},\qquad
g=\operatorname{clip}_{[0,1]}
\frac{\sum_w\langle d_w,r_w\rangle}
     {\sum_w\lVert d_w\rVert_2^2}.
\]

Here division by `s` is coordinatewise by the frozen training scales.  The
windows are the 176 nonoverlapping H20 windows from inherited
`observational_train` seeds `45100..45107`, at origins
`80,100,...,500` (22 windows per episode).  No V7 validation row enters this
fit.  The resulting locked value is

\[
g=0.7868543064870357,
\qquad \widehat Y=P+g(S-P).
\]

Thus the candidate keeps 78.69% of the parent displacement away from
persistence and shrinks 21.31%.  It is a convex, nonrecursive output operation;
its output is never fed back into `S`.

### 4.2 Degrees of freedom and choices

- New fitted degrees of freedom: **one global scalar**, zero episode-specific
  controller parameters.
- Fixed protocol choices: expert pair `{V5 sparse, persistence}`, the 176-window
  enumeration, normalized squared loss, pooling across charts/leads/windows,
  clipping interval `[0,1]`, and one global rather than prefix-adaptive gain.
- Leave-one-training-episode-out gains ranged only from `0.754692` to
  `0.839645`.  This is a stability diagnostic, not independent validation.
- Target-aware risk: moderate for choosing the route after V7; low for the
  numerical value of `g`, which is training-only.

### 4.3 Analytic safety and predicted effect

The output lies pointwise in the convex hull of `P` and `S`, so it introduces
no new recursive state, cannot exceed the component envelope coordinatewise,
and falls back to persistence or the parent at the two endpoints.  The only
learned recursive component is the frozen sparse parent.  Its acyclic bridge
ordering makes its mechanism Jacobian triangular, with global spectral radius
given by the maximum diagonal self coefficient (`0.781420` here); its latent AR
magnitude is `0.936927`.  Both are below `0.98`.  Persistence is materialized as
a fixed path, not recursively coupled to the candidate.

The predicted effect is lower long-horizon variance and less overshoot at the
cost of modest bias toward the final observation.  This is not a theorem of
future RMSE improvement; the safety guarantee concerns recursion and bounded
correction, not unknown outcomes.

On the disclosed V7 development set:

- candidate mean H20 RMSE: `0.5449545190`;
- mean improvement over V5: `+0.011960509`, paired 95% lower endpoint
  `+0.003779731`;
- mean improvement over persistence: `+0.028943018`, paired 95% lower endpoint
  `+0.001995271`.

This is the best low-DOF development result in the current search, but it is
not confirmatory because the route was selected after V7 was disclosed.

### 4.4 Fair controls and claim boundary

The symmetric control applies the identical formula and the identical 176
training windows to the same-probe dense parent.  It independently gives
`g_dense=0.7835668486813699` and disclosed-development RMSE `0.5453418191`.
The sparse/dense geometric error ratio is `0.998971657`, with log-ratio interval
`[-0.002116572, +0.000058827]`.  Sparse-specific superiority is therefore not
supported.  A successful future result could support **training-only convex
forecast shrinkage**, while sparse value remains only the mandatory
sparse-versus-no-sparse ablation.

Required controls are:

1. persistence as the no-learned-dynamics/no-sparse endpoint;
2. unshrunk V5 parent;
3. the independently fitted same-probe dense shrinkage above;
4. frozen V7 consensus and stable adaptive dense as historical comparators;
5. a zero-bridge parent processed by the same residual-filter and shrinkage
   code if a bridge-specific contribution is claimed.

Sparse and dense arms must retain identical observational data, paired probes,
window enumeration, prefix handling, scale, and scalar-fit procedure.  The
dense value may not reuse the sparse `g`.

### 4.5 Kill tests

Kill R1 before any test opening if any of the following occurs on one newly
locked validation block:

- the paired 95% lower improvement is not positive versus either V5 or
  persistence;
- the candidate does not beat its registered no-sparse/zero-bridge ablation;
- it violates the registered noninferiority margin against the symmetric dense
  shrinkage;
- recomputing the 176 training windows does not reproduce `g` exactly within a
  preregistered numerical tolerance;
- future or hidden poisoning changes `g`, `S`, or the output;
- any recursive component has radius above `0.98`, or the output is nonfinite;
- any second gain, alternate window grid, clipping interval, or lead-dependent
  gain is tried after the new validation block is seen.

## 5. Route R2 -- V5-anchored globally contractive consensus

### 5.1 Algorithm

For a cyclic dense mechanism

\[
F_i(x)=b_i+a_i x_i+\sum_j B_{ij}\tanh(x_j),
\]

let `D=diag(s)` contain the frozen training scales and define the normalized
row bound

\[
q_i=|a_i|+\sum_j |B_{ij}|\frac{s_j}{s_i},\qquad
c_i=\min\left(1,\frac{0.98}{q_i}\right).
\]

At the current prefix anchor `x_o`, replace the row by

\[
F_i^{(c)}(x)=F_i(x_o)+c_i\{F_i(x)-F_i(x_o)\}.
\]

This preserves the mechanism prediction at the observed anchor and guarantees

\[
\left\|D^{-1}J_{F^{(c)}}(x)D\right\|_\infty\le 0.98
\]

for every `x`, because `|sech^2(x_j)| <= 1`.  Apply this operator to the inner
and outer adaptive-dense experts and symmetrically to cyclic dense controls.
Do not project the sparse parent: its acyclic triangular Jacobian already has
global radius `0.781420`.

Form the inherited V7 inverse-root-error consensus `C_c(S,A_c,P)` from sparse
parent `S`, contractive adaptive expert `A_c`, and persistence `P`.  The final
candidate is the fixed anchored stack

\[
\widehat Y=\tfrac12 S+\tfrac12 C_c(S,A_c,P).
\]

Because `C_c` is convex, the effective V5-parent weight is at least one half.
The final aggregate is not fed back into a component.

### 5.2 Degrees of freedom and choices

- Episode-fitted controller degrees of freedom: two simplex coordinates inside
  the inherited three-expert consensus.
- New freely fitted degrees of freedom: zero.  Each row scale `c_i` is fixed by
  the analytic bound and the existing `0.98` checkpoint.
- Fixed additional choice: the outer `1/2` anchor weight.
- Target-aware risk: **high**.  The unanchored contraction was first inspected
  at RMSE `0.549043`; the fixed half-anchor was then inspected and improved to
  `0.545168`.  The half weight therefore cannot be portrayed as a priori.

### 5.3 Development effect and safety

On disclosed V7 development data, the anchored route gave:

- mean H20 RMSE `0.5451683681`;
- paired improvement versus V5 `+0.011746660`, interval
  `[+0.000361366, +0.023131954]`;
- paired improvement versus persistence `+0.028729169`, interval
  `[+0.003641364, +0.053816974]`;
- maximum component pathwise radii `0.781420` (sparse), `0.898547`
  (projected adaptive), and `0.791272` (projected symmetric dense control).

The matched no-sparse fills are deliberately both reported rather than
selecting the easier one after outcomes:

- replace the sparse slot/anchor by projected adaptive: mean `0.568702`, paired
  sparse improvement interval `[+0.003430, +0.043638]`;
- replace the sparse slot/anchor by persistence: mean `0.568142`, interval
  `[+0.000302, +0.045645]`.

The symmetric projected-dense anchored controller had mean `0.546971`; the
sparse candidate's geometric ratio was `0.994096`, with log-ratio interval
`[-0.009637, -0.002206]` on this disclosed development set.  These favorable
numbers motivate, but cannot confirm, the route.

### 5.4 Fair controls and cross-predictions

Use the exact operator

\[
T(E,A,P)=\tfrac12E+\tfrac12 C_c(E,A,P)
\]

for all controls.  Compare `E=S` with `E=D_c` (same-probe dense), `E=A_c`, and
`E=P`; refit each controller's inner weights symmetrically.  Apply the same
contraction projection to every cyclic dense expert, even if an unprojected
control would happen to score better.  Keep unprojected V5 and persistence as
external anchors.

Registered cross-predictions should include: projection activates only when a
prefix-fitted row bound exceeds `0.98`; its benefit should be concentrated in
those seeds; and the preprojection bound should positively predict the
absolute path change.  Failure of these directional diagnostics makes an
apparent mean gain compatible with accidental averaging.

### 5.5 Kill tests

Kill R2 if the analytic induced-norm test fails at any state probe, any
component exceeds radius `0.98`, the effective V5 weight drops below `0.5`, or
either no-sparse fill has a nonpositive paired lower improvement.  Also kill it
for a nonpositive paired lower endpoint versus V5 or persistence, dense
noninferiority failure, leakage, nonfinite output, or any post-lock change to
the `0.98` projection or `1/2` anchor.  Because R2 was selected adaptively on
V7, it must not share a fresh selection block with R1 and then be promoted as
the winner.

## 6. Route R3 -- median-of-three prefix gate between V5 and persistence

### 6.1 Algorithm

Use three disjoint H20 backtests wholly inside the observed prefix: origins
20, 40, and 60, with target blocks 21--40, 41--60, and 61--80.  For block `b`,
compute

\[
g_b=\operatorname{clip}_{[0,1]}
\frac{\langle (P_b-S_b)/s,(Y_b-S_b)/s\rangle}
     {\lVert(P_b-S_b)/s\rVert_2^2},
\qquad g=\operatorname{median}(g_{20},g_{40},g_{60}),
\]

with deterministic `g_b=0` for a zero denominator.  At origin 80 return

\[
\widehat Y=S+g(P-S).
\]

This is a robust stacking/gating route: the median has a one-block breakdown
point and the V5 parent is the exact fallback at `g=0`.

### 6.2 Degrees of freedom, predicted effect, and pilot

- Applied degrees of freedom: one episode-specific scalar; three intermediate
  clipped block gains.
- Fixed choices: the two experts, three disjoint origins, H20 block size,
  normalized squared error, coordinatewise pooling, median, and `[0,1]` clip.
- Analytic safety: convex output, no adaptive expert, no new recursion, sparse
  radius `0.781420`, latent AR magnitude `0.936927`.
- Predicted effect: resistance to one anomalous prefix era, but possible loss
  when early-prefix regimes do not predict the outer latent state.

The disclosed-development result was weak: mean RMSE `0.555840`, paired mean
improvement versus V5 `+0.001075` with interval
`[-0.020696, +0.022845]`, and improvement versus persistence `+0.018057` with
interval `[-0.003528, +0.039643]`.  The applied gain was exactly zero on 38.5%
of seeds and exactly one on 32.3%, showing severe boundary collapse.  This
route is therefore not recommended for the next locked run.

### 6.3 Fair controls and kill tests

The symmetric dense parent must receive its own three block gains; persistence
is the no-sparse endpoint.  Report the entire gain distribution and the
block-to-outer error-rank association.  Kill the route if either boundary mass
exceeds a preregistered ceiling, if the prefix block ranking fails to predict
outer relative error, or if paired lower improvement is nonpositive versus V5
or persistence.  The existing development screen already triggers the last
condition, so further tuning of origins or robust summaries on V7 is forbidden.

## 7. Route R4 -- uncertainty-aware rank-one Kalman residual

### 7.1 Algorithm

For sparse mechanism `F_S`, form prefix residuals

\[
e_t=x_{t+1}-F_S(x_t).
\]

Use the existing prefix center `c`, leading covariance eigenvector `u`, and
training-frozen residual AR coefficient `phi`.  With covariance eigenvalues
`lambda_1 >= ... >= lambda_4`, define without a tuned threshold

\[
R=\frac{\lambda_2+\lambda_3+\lambda_4}{3},\qquad
V=\max(\lambda_1-R,0),\qquad Q=(1-\phi^2)V.
\]

Treat `y_t=u^T(e_t-c)` as a noisy observation of a scalar AR(1) residual state,
run the exact scalar Kalman recursion over the prefix, and roll out the
conditional residual mean jointly with `F_S`.  This replaces the V5 filter's
assumption that the last projected residual is noise-free.

### 7.2 Degrees of freedom, safety, and predicted effect

- No tuned hyperparameter.  Two nonnegative prefix variance estimates (`R,V`)
  and one posterior scalar state are data-dependent; `phi`, the mechanism, and
  chart scales remain frozen.
- Fixed choices: mean of the three minor eigenvalues as isotropic noise floor,
  stationary initial variance, Gaussian scalar filter, and no robust clipping.
- Analytic stability: the augmented transition is block triangular, so its
  eigenvalues are the sparse mechanism eigenvalues and `phi`; their maxima are
  `0.781420` and `0.936927`, below `0.98`.
- Predicted effect: attenuate noisy last-residual extrapolation and reduce
  seed-tail variance.  It can introduce bias when the top residual component
  is mostly signal rather than measurement noise.

The latter risk appeared on disclosed V7 development.  Mean RMSE was
`0.559093`, compared with `0.556915` for V5.  Paired improvement versus V5 was
`-0.002178`, interval `[-0.005639, +0.001283]`; improvement versus persistence
was `+0.014805`, interval `[-0.019358, +0.048967]`.  The final Kalman gain had
mean `0.8045` and range `0.6081..0.9532`.  This does not justify a fresh locked
run.

### 7.3 Fair controls and kill tests

Run the identical covariance decomposition and scalar filter around the
zero-bridge parent and the same-probe dense parent; neither may reuse sparse
directions, variances, or posterior states.  Include the deterministic V5
residual filter as the direct parent.  Kill on nonpositive paired improvement
versus V5 or persistence, failure of the sparse/no-sparse ablation, dense
noninferiority failure, any estimated variance below zero before explicit
clipping, or any leakage/stability failure.  The disclosed development result
already activates the parent-improvement kill; do not rescue this route by
tuning a noise multiplier on V7.

## 8. Ranking

| Rank | Route | Structural locus | New fitted DOF | Analytic safety | Disclosed V7-development signal | Decision |
|---:|---|---|---:|---|---|---|
| 1 | R1 training-only convex shrinkage | output correction magnitude | 1 global, 0 per seed | strong; convex and only stable sparse recursion | `0.544955`; positive paired lower endpoints vs V5 and persistence | recommend freezing |
| 2 | R2 anchored contractive consensus | dense dynamics plus fixed V5 anchor | 2 inherited per-seed simplex coordinates; 0 new fit | strongest global dense bound, but more machinery | `0.545168`; positive development intervals and sparse ablations | retain as separate backup, not co-select |
| 3 | R3 three-block median gate | robust prefix controller | 1 applied per seed (3 intermediates) | strong; convex and no adaptive recursion | `0.555840`; both key intervals cross zero, high boundary mass | development-killed |
| 4 | R4 Kalman residual | latent residual uncertainty | 2 prefix variances plus posterior state; 0 tuned hyperparameters | strong augmented-state bound | `0.559093`; slightly worse than V5 | development-killed |

The ranking prioritizes low adjustment freedom and an analytic safety property,
then the chance of clearing both V5 and persistence at H20.  R1 ranks above R2
despite nearly equal disclosed RMSE because R1 has one training-only global
coefficient, eliminates the unstable adaptive expert entirely, and was not
created by tuning an outer half-anchor after inspecting a precursor on V7.

## 9. Recommended next move

Freeze **R1 only** as the next pilot candidate.  Before assigning any fresh
seed block:

1. implement the 176-window scalar fit as a deterministic, hash-locked training
   stage and reproduce `g=0.7868543064870357`;
2. implement sparse, zero-bridge/no-sparse, and independently fitted
   same-probe-dense arms with identical data and accounting;
3. retain the complete V7 prefix reader, poisoning tests, H5-prefix identity,
   normalization, per-seed endpoint, and component stability audit;
4. lock one H20 conjunction requiring positive paired lower improvement versus
   V5 and persistence, positive sparse/no-sparse contribution, dense
   noninferiority, finiteness, and stability;
5. state the claim as **training-only convex forecast shrinkage**.  Do not claim
   sparse-causal superiority unless a future sparse-versus-symmetric-dense
   comparison itself clears a preregistered superiority gate.

R2 is a scientifically useful backup if R1 is rejected before a new run is
registered.  It must receive a separate preregistration and fresh development
block; running R1 and R2 together and selecting the winner would reopen the
same look-elsewhere problem.  R3 and R4 should remain preserved negative routes
rather than being retuned on V7.

CE_RUN=_workspace/ce/agi-v8-breakthrough-20260811
