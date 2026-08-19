# Full 3D cortical-ribbon metric and PFC validation contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/neural-riemannian-metric-validation-20260818
PREDECESSOR: _workspace/ce/neural-riemannian-metric-independent-tests-20260818
PREDECESSOR: _workspace/ce/neural-riemannian-metric-multiroute-execution-20260818

## 1. Objective and claim boundary

Define and precheck a genuine three-dimensional Riemannian metric on a
cortical ribbon, implement the eligible estimator and falsification suite in
high-performance Rust, and only then inspect synthetic or biological outcomes.
The target causal chain is the hypothesis

$$
\Delta W^s
\xrightarrow{\Phi}
\Delta g_M(x,c)
\xrightarrow{\mathcal B}
\Delta p(X_{0:T},\tau_A\mid X_0,c).
\tag{1}
$$

Here $\Phi$ and the dynamical bridge $\mathcal B$ are separate model choices.
Neither is a theorem. A successful synthetic recovery validates only the
frozen estimator family. A public PFC activity dataset without direct
structural connectivity, cortical depth, and same-unit longitudinal identity
cannot validate (1) and is restricted to PFC geometry feasibility.

The predecessor's synthetic false-selection rate, 72 of 100 null-family
circuits, failed calibration. No biological PFC outcome may be opened in this
run until the replacement synthetic familywise-error gate passes.

## 2. Typed full-3D canon

Let $M$ be a cortical ribbon, a three-manifold with boundary, with local
coordinates

$$
x=(u,v,\ell),
\tag{2}
$$

where $(u,v)$ are intrinsic cortical-surface coordinates and $\ell$ is a
registered depth or laminar coordinate. The following are distinct:

- $h(x)\in\operatorname{Sym}^+(T_x^*M)$: measured anatomical metric,
  including pre-existing folding and cortical thickness;
- $g_0(x,c)\in\operatorname{Sym}^+(T_x^*M)$: baseline functional metric;
- $g_t(x,c)\in\operatorname{Sym}^+(T_x^*M)$: induced functional metric;
- $W^s$: measured structural connectivity between identified units;
- $W^e$: estimated effective connectivity, never substituted for $W^s$;
- $v(x,c)$ and $a(x,c)=B(x,c)B(x,c)^\top$: drift and diffusion;
- $g_z$ on a neural state space $Z$: a different metric unless pulled back by
  a declared map $F:M\to Z$.

A full metric has six independent fields:

$$
g(x,c)=
\begin{pmatrix}
g_{11}&g_{12}&g_{13}\\
g_{12}&g_{22}&g_{23}\\
g_{13}&g_{23}&g_{33}
\end{pmatrix}\succ0.
\tag{3}
$$

The predecessor's diagonal one-parameter $D=3$ fixture is not the canon and
cannot satisfy the full-3D recovery gate.

When the physical ribbon is a volume $\Omega\subset\mathbb R^3$ and its chart
map is a local diffeomorphism $r:M\to\Omega$, its induced metric
$h=r^*\delta$ is intrinsically flat in the interior even though folds,
boundaries, metric components, and Christoffel symbols can be nontrivial.
The folded surface geometry, ribbon boundary, depth, and extrinsic second
fundamental form are anatomical covariates; they are not by themselves
evidence that the functional metric has nonzero Riemann curvature.

## 3. Units, SPD parameterization, and chart law

Physical coordinates are normalized with a frozen diagonal scale
$D=\operatorname{diag}(L_s,L_s,L_\ell)$ and reference distance $L_0$. Define

$$
\bar h(y)=\frac{D^\top h(x)D}{L_0^2},
\qquad x=Dy.
\tag{4}
$$

All arguments entering exponentials, logarithms, and fitted tensor
coefficients are dimensionless. Let $e^a$ be an oriented $\bar h$-orthonormal
coframe and let $\mathcal S_\theta$ be a dimensionless $\bar h$-self-adjoint
endomorphism. The production SPD field is defined intrinsically by

$$
g_\theta(u,v)=\bar h\!\left(\exp(\mathcal S_\theta)u,v\right),
\qquad
(g_\theta)_{ij}=e^a_i[\exp(S_\theta)]_{ab}e^b_j.
\tag{5}
$$

$S_\theta$ is the symmetric matrix of $\mathcal S_\theta$ in that coframe and
uses all six basis elements. No principal square root of a coordinate metric
matrix defines (5). Under $x'=\varphi(x)$ with
$J=\partial x'/\partial x$,

$$
g'(x')=J^{-\top}g(x)J^{-1}.
\tag{6}
$$

The relative deformation is defined only in the same tangent fiber. Let
$A_t=g_0^{-1}\circ g_t$ be the positive $g_0$-self-adjoint endomorphism:

$$
\mathcal L_t(x)=\log A_t(x),
\qquad
D_x(g_0,g_t)=\sqrt{\operatorname{tr}(\mathcal L_t^2)}.
\tag{7}
$$

If a state-space metric is used, the only allowed ribbon metric is

$$
F^*g_z=DF^\top g_z DF,
\tag{8}
$$

with a separately measured or training-frozen $C^1$ immersion $F$ satisfying
$\operatorname{rank}DF=3$. Otherwise (8) is singular or gauge-unidentified.

## 4. Frozen producer and bridge families

`M-H` is anatomy-only: $g=\bar h$. `M-W` uses a finite, preregistered set of
dimensionless $\bar h$-self-adjoint endomorphism fields derived from spatial
$W^s$ and $h$:

$$
\mathcal S_\theta(x,c)
=\sum_{m=1}^{p}\theta_m\mathcal T_m(W^s,h,x,c).
\tag{9}
$$

Allowed $T_m$ families are identity, trace-free anatomical Hessians of
dimensionless local connection fields, tensor products of their covariant
gradients, and their symmetric cross-products. Kernel radii, tensor list,
smoothness, parameter count, optimizer, bounds, and regularization must be
enumerated in `11-math.md` before implementation. Raw adjacency entries may
not be treated as coordinate tensor components.

The primary stochastic bridge is specified by a coordinate-covariant
generator relative to the anatomical connection $\nabla^h$:

$$
\mathscr L f=
\left[-R^{ij}\partial_jU+\nabla^h_jR^{ij}\right]\partial_i f
+D^{ij}\nabla^h_i\nabla^h_j f,
\qquad
R=D=\kappa g_\theta^{-1}
\tag{10}
$$

Distance, curvature, drift, noise, and first-passage time remain separate
readouts. The coordinate Itô drift includes the corresponding
$-D^{jk}\Gamma(h)^i{}_{jk}$ term. Equation (10), not $g$ alone, generates
trajectories.

Every scored metric candidate must compete against independently nested,
parameter-budgeted controls: independently parameterized direct response and
diffusion; anatomy-only;
Euclidean; persistence; gain-only; noise-only; degree/weight-preserving $W$
surrogate; spatial-autocorrelation-preserving surrogate; circuit-permuted $W$;
and nonlinear flat-pullback.

## 5. Frozen route universe

| ID | Input and purpose | Maximum claim |
|---|---|---|
| `R-SYNTH-3D` | Folded Euclidean-ribbon pullback $h$, spatial $W$, independently curved six-component $g_0\to g_t$, paths and interventions | estimator recovery only |
| `R-NULL-3D` | Flat pullback, direct-dynamics, gain, noise, anatomy-only, randomized-$W$, and exact-null generators | calibrated rejection only |
| `R-PFC-WOJCIK` | Wójcik et al. macaque lateral-PFC XOR learning sessions | session/stage-held-out representational feasibility |
| `R-PFC-CALANGIU` | public macaque dlPFC multi-task recordings | cross-task PFC feasibility if schema and identity permit |
| `R-PFC-KIANI` | public macaque area-8Ar simultaneous population recordings | fast external trajectory/choice feasibility |
| `R-PFC-RIBBON` | same-subject PFC anatomy, depth/layer, $W^s$, activity, perturbation, and behavior | full chain (1), only if every typed input exists |

No route or hyperparameter may be added after any outcome from its family is
opened. Source acquisition, schema inspection, and non-outcome fixtures do not
count as outcome opening. A route lacking its typed inputs receives a machine
status rather than a proxy substitution.

## 6. Pre-outcome gates

Implementation and data access are staged.

### Gate A: definition and numerical kernel

1. Recover all six SPD components, including nonzero off-diagonal fields, on
   analytic fixtures.
2. Pass SPD exponential/logarithm round-trip, Cholesky, affine and nonlinear
   pullback, length, and curvature-invariance fixtures in `f64` against a
   trusted reference oracle.
3. Pass deterministic seeded parallel reduction and serialization checks.
4. Demonstrate that nonconstant flat pullbacks have zero curvature within a
   frozen numerical tolerance and that a known curved field does not.

Failure blocks all scientific outcomes.

For day-to-day development, **Gate A-KERNEL** is the mandatory gate: the
locked Rust unit tests, the 39 analytic fixtures, and the independent NumPy
oracle must pass from source. Executable copies, compiler fingerprints,
create-only lineage manifests, and mutation campaigns belong to the optional
**Gate A-LOCK** publication gate. Gate A-LOCK is required only immediately
before a one-shot Gate B scientific execution or a release claim. Its absence
does not block source work, non-outcome microbenchmarks, or kernel refactoring.
Build directories and executable binaries are never committed to Git.

### Gate B: independent synthetic calibration

1. Fit only on independent training circuits; score only held-out circuits,
   animals, intervention directions, and paths.
2. Treat independent circuit as the inferential unit. Time bins and repeated
   trajectories never increase the inferential sample size.
3. Across at least 1,000 independent null datasets, the frozen run-level FWER
   must be at most 0.05 and its declared one-sided 95% upper confidence bound
   must meet the threshold specified in `11-math.md`.
4. On the known full-3D generator, recover deformation direction and magnitude,
   including off-diagonal terms; beat every non-nested frozen control and be
   noninferior to the 12-parameter direct response/diffusion alternative by
   the frozen margin, with a ten-test Holm correction.
5. False selection under exact null, $W$ surrogates, direct-only, gain-only,
   noise-only, and flat-pullback generators is a hard failure, not evidence for
   a more flexible metric.

Failure blocks `R-PFC-WOJCIK`, `R-PFC-CALANGIU`, and `R-PFC-KIANI` outcome
inspection. It does not falsify every possible neural metric; it rejects the
frozen estimator and selection rule.

### Gate C: biological eligibility

For each PFC route, record official URL, version, license, file hashes, animal,
region, session/unit identity, anatomy/depth fields, behavior, split, and
missingness before analysis. A public activity dataset without $h$, $W^s$,
depth, or same-unit identity cannot be promoted to `R-PFC-RIBBON`.

## 7. PFC-specific frozen boundary

The Wójcik dataset contains lateral-PFC learning data from two macaques and 25
experiment-1 sessions, but electrodes were moved between sessions to obtain
new neurons. Its released `cell_loc` is a categorical region code, not a
continuous $(u,v,\ell)$ registration. Therefore it can test stage/session
transfer of a frozen representational estimator and later trial/behavior
prediction, but not $\Delta W^s\to\Delta g_M$, same-cell deformation, physical
folding, laminar curvature, or population confirmation beyond two animals.

ALM, motor, entorhinal, hippocampal, visual-cortex, FEF, and DMFC data are not
counted as PFC evidence in this run. Published PCA, decoding, selectivity, or
representational geometry is not itself an observed Riemannian metric.

## 8. Rust implementation and speed contract

The scientific kernel is an isolated Rust crate under `artifacts/rust/`.
Heavy computation uses fixed-size `f64` 3-by-3 symmetric matrices, analytic
eigendecomposition or a reviewed linear-algebra backend, and `rayon` across
independent datasets/circuits. Input is streamed or memory-mapped where the
official format permits. Python may perform only source conversion and an
independent reference oracle; it is not the production fitter.

Parallel tasks receive deterministic counter-based seeds. Reductions use a
fixed order or store per-unit scores before a serial aggregate. Release builds
use optimization, but `fast-math`, silent SPD projection, GPU nondeterminism,
and `f32` are forbidden. Runtime optimization may not change sample units,
candidate budgets, numerical tolerances, or decision rules.

## 9. Kill conditions and allowed conclusions

Hard stops include rank-deficient $DF$, domain mismatch, missing six-component
recovery, coordinate-law failure, failed null calibration, leakage, nested
pseudoreplication, nonfinite output, silent regularization changes, or a metric
candidate that is worse than the 12-parameter direct response/diffusion model
by more than the frozen noninferiority margin.

Allowed final statuses are:

- `ESTIMATOR_PASS` or `ESTIMATOR_FAIL` for synthetic recovery/calibration;
- `PFC_FEASIBILITY_ONLY` for eligible public PFC activity routes;
- `UNTESTABLE_MISSING_INPUT` for the full cortical-ribbon biological chain;
- `FAILED_EXECUTION` for an eligible frozen computation that cannot complete.

No result in this run may state that the brain's canonical metric has been
found. A positive result only supports the exact frozen producer and bridge at
its declared tier.

## 10. Deliverables and completion

- `10-sources.md`: official provenance and biological eligibility.
- `11-math.md`: exact finite candidate universe, estimands, tolerances, scores,
  multiple-testing rule, dimensions, counterexamples, and kill gates.
- `12-routes.md`: dependency-ordered execution matrix and fallback statuses.
- `20-audit.md`: independent pre-implementation gate.
- `30-implementation.md`: Rust/Python ownership, hashes, commands, and outputs.
- `31-validation.md`: non-outcome fixtures, lock, outcome validators, and
  result ledger.
- `40-final-report.md`: Korean synthesis with formal claim status.

Completion requires a disposition for every frozen route, a stable-snapshot
audit before implementation, a second stable-code audit before first outcome,
and machine-readable provenance for every opened result. If Gate A or B fails,
the run still completes with the failure and leaves biological outcomes sealed.
