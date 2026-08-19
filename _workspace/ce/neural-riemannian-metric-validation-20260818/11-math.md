# Neural operational metric: mathematical audit

Status: COMPLETE

This lane audits equations (2)--(5) in `00-contract.md`.  The resulting $g_H$ is a defined operational estimator when its inputs are frozen; it is not uniquely implied by raw connectivity, nor a proof that neural trajectories are Riemannian geodesics.

## Well-posedness, type, and coordinate rule

For time-invariant $J,Q$, equation (3) is

$$
C_H=\sum_{k=0}^{H-1}J^kQ(J^k)^\top.
$$

For a time-varying linearization beginning at $t$, replace $J^k$ by $\Phi_{t,k}=J_{t+k-1}\cdots J_t$, with $\Phi_{t,0}=I$.  If $Q\succeq0$, then $C_H\succeq0$.  Hence $g_H=(C_H+\lambda I)^{-1}$ is symmetric positive definite exactly when $C_H+\lambda I\succ0$: automatically for $\lambda>0$, and for $\lambda=0$ only when the finite-horizon noise-reachability matrix has full rank.  Stability of $J$ is unnecessary at finite $H$, though it is needed for the infinite-horizon limit to be finite.

P0 type correction: $C_H$ as written is not a controllability Gramian, since it contains $Q$ rather than the input channel $B$ and a control-cost matrix.  It is the finite-horizon covariance/reachability Gramian of process noise.  The local control Gramian would be

$$
G_H^{\rm ctrl}=\sum_{k=0}^{H-1}\Phi_{t,k}B_{t+k}R_{t+k}^{-1}B_{t+k}^\top\Phi_{t,k}^\top.
$$

No conclusion about controllability follows from (3) unless an explicit identification equates the relevant noise and input channels, which is an additional assumption.

Under an invertible linear chart change $z'=Pz$, the consistent transformations are $J'=PJP^{-1}$, $Q'=PQP^\top$, and $C'_H=PC_HP^\top$.  With no ridge,

$$
g'_H=(C'_H)^{-1}=P^{-\top}g_HP^{-1},
$$

so $\dot z'^\top g'_H\dot z'=\dot z^\top g_H\dot z$ and (4) is invariant.  With the coordinate-specific ridge in (3), in general

$$
(PC_HP^\top+\lambda I)^{-1}\ne P^{-\top}(C_H+\lambda I)^{-1}P^{-1}.
$$

It is invariant only for orthogonal $P$, $\lambda=0$, or a covariant regularizer: replace $\lambda I$ in the transformed chart by $P(\lambda I)P^\top$ (more generally transform a declared reference tensor).  Nonlinear charts add Jacobian-dependent noise transformation and, for stochastic continuous-time models, an Ito/Stratonovich convention.  Therefore claims of chart invariance require a fixed admissible chart family and covariantly transformed regularization.

## Drift is not a Riemannian metric

Equation (4) is a reversible, undirected length because $g_H$ is SPD.  Equation (5) is a directed transition residual/action: for a nonzero drift $F$, reversing a path changes $x_{t+1}-F(x_t)$, so the action is not a Riemannian length.  A directional geometry would require a specified Finsler/Randers-type object and its own axioms; neither follows from (2).  A fitted $F$ may improve trajectory prediction without making actual trajectories minimizers of $L_g$, and an SPD metric cannot encode the direction of $F$.

## Complete counterexamples to strong negative claims

### N1: raw $W$ cannot uniquely determine a local metric

Fix scalar raw connectivity $W=1$.  Let $F(z;W,A)=Az$ and $Q=q>0$.  For $H=1$, $C_1=q$, and for $H>1$, $C_H=q\sum_{k=0}^{H-1}A^{2k}$.  Keeping $W$ fixed while changing gain $A$ or noise $q$ changes $g_H$.  Thus one raw $W$ has infinitely many metrics.  Conversely, $W'=PWP^{-1}$ is a gauge-equivalent representation under a chart change while its entries change.  This falsifies N1's uniqueness at both the mechanistic and coordinate levels.

### N2: representational geometry alone does not identify $\Delta W\to\Delta g$

For each $a\in(-1,1)$ consider $z_{t+1}=az_t+\epsilon_t$ with $\operatorname{Var}(\epsilon_t)=1-a^2$.  Every process has stationary variance one, hence identical one-time Euclidean representational geometry, while the effective connection $W=a$, temporal covariance, and $C_H$ vary with $a$.  Alternatively, a changing latent chart can alter representational distances with fixed physical dynamics.  Therefore observing geometry change neither proves a connectivity change nor a metric change.

### N3: an effective-dynamics reproduction cannot establish a causal spatial mechanism

Let an unobserved training/context variable $U$ cause both $W^e$ and the later trajectory $X$, while $g_H=f(W^e,A,Q)$ is computed deterministically.  Then $W^e$, $g_H$, and $X$ can be associated and held-out predictable even if intervening on the alleged spatial feature has no effect on $X$.  A second complete counterexample is $W$ fixed with context-dependent $A,Q$, which changes both $g_H$ and $X$.  H1B establishes predictive reproducibility only; it cannot distinguish these causal graphs from $\Delta W\to\Delta g\to\Delta X$.

## Identifiability and mediation

The map $(W,A,Q,\text{chart},H,\lambda)\mapsto g_H$ is many-to-one and its inverse is non-identifiable.  Trial activity alone cannot separately identify directed $W$, gain/delay state $A$, and process noise $Q$ without model assumptions, perturbations, or calibration.  Estimate $J,Q$ on calibration/training trials only, report posterior/bootstrap uncertainty for their induced $g_H$, and treat small eigenvalues/condition numbers as an exclusion or regularization diagnostic fixed before testing.

Mediation requires an intervention on $W$ (or a stated valid instrument), temporal ordering $W\to g\to X$, no unmeasured treatment--mediator, mediator--outcome, or treatment--outcome confounding after stated covariates, and no mediator--outcome confounder affected by treatment.  A product-of-coefficients from observational activity data is descriptive only.  Because $g_H$ is calculated from estimated dynamics, uncertainty propagation and split separation are mandatory.

## Held-out estimators, baselines, and kill tests

For each outer group split, fit cell matching, latent chart, $F,J,Q$, missingness handling, $H,\lambda$, and all hyperparameters only in outer training data.  Compute test $g_H$ from frozen rules before looking at test outcomes.  The primary score is group-aggregated future-trial next-state log predictive density (or one predeclared proper energy score).  Use paired group-level score differences and a cluster/group bootstrap; report the number of groups, never window/trial count as confirmation $n$.

Required baselines are persistence, Euclidean latent distance, firing rate/covariance, raw $W$, direct state-space dynamics without the metric feature, parameter-matched unrestricted SPD, and label-only.  `NRM-H2` survives only if the frozen geometry feature has a positive adjusted held-out score interval against the direct-dynamics and parameter-matched SPD baselines, not merely against a weak baseline.  Run geometry shuffle, time reversal, condition-label permutation, and admissible-chart/cell-resampling stability checks.  K1--K8 from the contract are immediate kill/downgrade conditions; in particular a gain/noise-only model matching the score (K6) leaves a $W$-mechanism unsupported.

For H3, use a trained-pair versus matched-untrained-pair interaction on prespecified path action, hitting time, or endpoint error, with direction fixed before test data.  For H4, require a context/gain-only switch control with anatomy fixed and a causal manipulation.  For H5, require paired sleep/wake and replay/sham controls with a global scaling/time-awake baseline.  None can be inferred from condition means.

## Independence and multiplicity

The independent unit is animal; if and only if animal identity is unavailable but sessions are documented biologically independent, it is session.  All trials/windows/cells of one unit remain in one outer fold.  Repeated measures may improve a within-group estimate but add no independent population units.  A single animal supports technical/time-blocked feasibility only, not a population confirmation.  All chart, horizon, ridge, smoothing, target, metric family, and route choices selected after test outcomes are exploratory.  Treat NRM-H1A, H1B, H2, H3, H4, and H5 as a six-claim Holm family unless a pre-data amendment declares a narrower primary family.

## Deterministic synthetic fixture

`artifacts/math/metric_fixture.py` calculates finite-horizon covariances for: a known $J$ change (positive), an identical system (null), and a fixed-$W$ gain/noise change (confounded).  It also verifies exact unregularized congruence invariance, the failure of isotropic ridge under a non-orthogonal chart, and restoration by a covariant ridge.  The generated output is `artifacts/math/metric_fixture_output.json`.  This is a mathematical fixture, never empirical evidence.

Run:

```powershell
python _workspace/ce/neural-riemannian-metric-validation-20260818/artifacts/math/metric_fixture.py
```

## Findings

- P0: equation (3) is mislabeled if called a controllability Gramian; its $Q$-based construction is a noise covariance/reachability object.
- P0: isotropic $\lambda I$ regularization destroys general coordinate congruence.  A claim of chart invariance without a covariant ridge or restricted charts is false.
- P0: equations (4) and (5) have different types.  A reversible Riemannian length cannot by itself describe directed drift or a path transition law.
- P1: N1, N2, and N3 have complete counterexamples above; raw connectivity, observed representation, or held-out prediction alone cannot establish the stated causal bridge.
- P1: `NRM-H1A` needs synchronized pre/post direct-connectivity measurement and intervention; `NRM-H1B` is only an effective-dynamics predictive result.

## Candidate-universe v2 amendment

This amendment audits the frozen bytes in `artifacts/candidate-equation-registry.md`.  It replaces the earlier single-`S4` implementation scope with a finite, type-stratified candidate universe.  The previous E17 $H=5$ output had already been inspected, so the expanded E17 tournament is mathematically useful discovery but cannot be a locked test.

### Time-varying covariance theorem

Let

$$
z_{t+1}=J_tz_t+\varepsilon_t,
\qquad
\operatorname{Cov}(\varepsilon_t)=Q_t,
$$

with independent zero-mean innovations.  Recursion gives

$$
z_{t+H}=\Phi_{t,t+H}z_t+
\sum_{k=0}^{H-1}\Phi_{t+k+1,t+H}\varepsilon_{t+k}.
$$

Therefore

$$
\operatorname{Cov}(z_{t+H}\mid z_t)=
\sum_{k=0}^{H-1}\Psi_{t,H,k}Q_{t+k}\Psi_{t,H,k}^{\top},
\qquad
\Psi_{t,H,k}=\Phi_{t+k+1,t+H}.
$$

Cross terms vanish by innovation independence.  The earlier expression $\Phi_{t,t+k}Q_{t+k}\Phi_{t,t+k}^{\top}$ transports every innovation from the initial time toward its injection time, not from its injection time to the common endpoint, and is false for a generic time-varying system.  The same derivation with $\varepsilon_{t+k}=B_{t+k}u_{t+k}$ and $\operatorname{Cov}(u_{t+k})=R_{u,t+k}^{-1}$ gives `S10-H`.

### Covariance and metric tensor roles

For $z'=Pz$, covariance shapes obey $M'=PMP^\top$.  Their inverses obey

$$
(M')^{-1}=P^{-\top}M^{-1}P^{-1},
$$

which is the metric transformation law.  A ridge added before inversion must therefore be another covariance tensor $R_C'=PR_CP^\top$.  In contrast, a reference tensor added directly to a pullback, Fisher or Hessian metric must obey $G_0'=P^{-\top}G_0P^{-1}$.  One symbol cannot fill both roles.  Fixed $R_C=G_0=I$ is preserved only by orthogonal $P$, so the E17 tournament makes a fixed-chart claim rather than a general affine-invariance claim.

### Candidate gates

| Family | Mathematical gate | E17 disposition |
|---|---|---|
| `S0`--`S5` | covariance shape plus ridge is SPD; `S5` also needs $\rho(J)<1$ | calculate when gate passes |
| `S6/S7/S14/S15` | direct quadratic form plus metric ridge is SPD; `S14` is fixed-chart only | calculate the registered proxy endpoint |
| `S8/S9` | Fisher/pullback may be rank deficient; positive ridge is required unless full rank | condition-label feasibility only |
| `S10/S11` | measured $B,R_u,Q_x$ and control assumptions are required | `UNTESTABLE_MISSING_INPUT` |
| `S12/S13` | optimizer returns a finite SPD matrix under the frozen parameterization | flexible discovery baselines |
| `S16` | independently frozen nonquadratic $V$, affine chart and positive Hessian/ridge | `UNTESTABLE_MISSING_INPUT` |
| `G1/G2` | nonnegative symmetric connected conductance; reversible Markov kernel for `G2` | fixed symmetrization proxy only |
| `G3a/G3b` | positive registered edge costs and finite reachable pairs | directed quasi-metric proxy only |
| `D1` | $Q_t\succ0$ and full log-determinant normalization for cross-model scoring | discrete action feasibility |
| `D2/D3` | calibrated SDE or identifiable navigation one-form | `UNTESTABLE_MISSING_INPUT` |
| `P1/P2` | finite second moments and frozen ground metric | one $W_2$ class, not two candidates |
| `R3/R4` | smooth state-dependent field, derivatives, boundary/gauge and update inputs | `UNTESTABLE_MISSING_INPUT` |

For `G1`, $K=K^\top\succeq0$ is not a sufficient graph condition.  The required object has $K_{ij}\ge0$, $K_{ii}=0$, $D=\operatorname{diag}(K\mathbf1)$ and a connected support.  For `D1`, the quadratic action alone is not a proper score across covariance candidates: the $\log\det Q_t+d\log(2\pi)$ terms are mandatory.  For `S8`, `S9` and `S11`, positive semidefiniteness is not positive definiteness; they are metrics only after the registered SPD gate.  A constant session-level metric has zero coordinate derivatives and flat curvature in its fixed linear chart, so it cannot validate `R3/R4`.

### Finite closure and inference

The registry freezes horizons, ridges, ranks, penalties, graph symmetrizations, edge costs, optimizer, split fractions, tie rules, Monte Carlo counts and seed.  Each outcome is attached to an ID plus this complete tuple.  An ineligible candidate remains in the universe with a reason code instead of receiving an improvised proxy.  This makes the enumeration finite and auditable; it does not prove that nature must use one member.

The outer E17 procedure can test whether a hyperparameter rule transfers across three held-out animals while allowing session-local calibration, but three already-open animals do not provide a population winner or an independent lock.  Any selected formula must be carried unchanged to a new cohort.  `NRM-H1A` remains untestable because E17 does not align direct $W^s$, metric inputs and later trajectories in the same units.
