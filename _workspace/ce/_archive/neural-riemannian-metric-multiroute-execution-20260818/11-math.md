# Multiroute estimands and independence audit

Status: COMPLETE

## Frozen objects and common decision rule

The target chain is

$$
\Delta W^s\longrightarrow\Delta g(z,c)\longrightarrow
\Delta p(x_{0:T},\tau_B\mid x_0,c).
\tag{M0}
$$

`W^s` means registered structural edges or a declared structural proxy;
`g(z,c)` is an SPD field in a chart fit only on the declared training block;
`v,Q` remain separately modelled drift and process covariance.  A metric is
not a velocity field.  A route may use the bridge
`v=-g^{-1}\nabla V`, only when `V`, the stochastic convention, time scale, and
the parameter-matched unconstrained `v,Q` comparator are frozen before scoring.

For every eligible route, the inferential row is an animal, subject, or an
independently randomized circuit.  Cells, spines, trials, windows, paths, and
sessions are nested.  The route-level primary contrast is the sample-level
mean difference in a strictly held-out proper score (conditional log score;
energy score for samples; censored first-passage log score when applicable).
Use a paired animal-level permutation or cluster bootstrap confidence interval.
The confirmatory family is the set of eligible primary contrasts for
`R-E17-F3,F4,F5`, `R-SLEEP-E19,E15`, `R-GRID-TORUS`, `R-DANDI-37`,
`R-ALLOPTICAL`, `R-BCI`, and `R-MICRONS`; apply Holm to that fixed family only.
`R-E17-F2`, `R-CELEGANS`, and `R-SYNTH` are descriptive or estimator-validation
routes and do not enter biological confirmatory inference.  A route that has
only one animal/subject reports no population p-value.

All `H_G` claims require simultaneous advantage over: persistence/Euclidean,
direct `v,Q`, gain-only, noise-only, and flat-pullback models with no larger
effective parameter budget.  `H_W` additionally requires a frozen covariant
map `Phi(W^s,c)` and independently estimated later `g`; `H_C` requires a
randomized or validly instrumented structural change and a retained direct
`W^s -> x` term.  Identity permutation, chart resampling, pre-period placebo,
and baseline producers `P-h/P-W/P-D/P-C` are common falsifiers where typed
inputs exist.

## Route estimand matrix

| Route | Typed estimand and independent source | Unit / feasible split | Score and primary nulls | Eligibility and kill rule |
|---|---|---|---|---|
| R-E17-F3 | `E[task-or-clustering outcome | frozen spine selectivity/clustering proxy]`; Figure 3 predictor versus separately released task/cluster endpoint | animal; source-supported held-out dendrite/cell pair or leave-animal-out. No array-order chronology assumption | Bernoulli/multinomial log score if raw labels exist; otherwise descriptive correlation. Null: rate, selectivity, dendrite identity, pair permutation | If only aggregated figure table or predictor/endpoint are the same summary: `PARTIAL_DESCRIPTIVE`/`INELIGIBLE_DEPENDENT`. Never claims `H_W` or `H_G`. |
| R-E17-F4 | earlier frozen representation/metric proxy predicts later held-out drift `d(z_later,z_pred)` | animal, then future session/block; cell identities must be established before outcome access | Gaussian/energy score; baselines rate, selectivity, identity permutation, persistence | A representation fitted from the same later data is circular. No independent chronology or unit IDs means descriptive only. |
| R-E17-F5 | pre-transition proxy predicts later transition distribution/error conditional on pre-state | animal, earlier transition block to later block | transition conditional log score; direct Markov `v,Q`, rate/selectivity, label permutation | Geometry must predate transitions. Source statistic alone is descriptive; outcome-derived geometry is ineligible. |
| R-E17-F2 | no new estimand; locked reference to predecessor tuple scores | none | no new score | `INELIGIBLE_DEPENDENT` as independent evidence. A rerun may only verify hash/reproducibility. |
| R-SLEEP-E19 | subject-level sleep-state summary predicts later encoding/trajectory endpoint: `E[Y_later | S_sleep, covariates]` | participant; sleep windows for predictor, later encoding trials for target; leave-subject-out | conditional log/energy score; REM/SWS composition, total sleep, arousal/time, state-label permutation | No same subject IDs or later target: `PARTIAL_DESCRIPTIVE`. This is sleep association, never `W -> g`. |
| R-SLEEP-E15 | sleep/deprivation/replay dose predicts later replay or behavior after predeclared temporal adjustment | animal/session label if animal IDs exist; dose window must precede target window | count likelihood or survival score; time, state composition, sham/condition permutation | Aggregated windows or no independent animal identity: descriptive. Same replay count used both sides is algebraically dependent. |
| R-CELEGANS | edge-held-out probability `p(W_ij=1 | structural metric features)` and synthetic perturbation recovery | neuron/edge with blocked node-pair and degree-stratified split; no population biological inference | Bernoulli log score; degree, Euclidean/path length, shortest path, diffusion, control-energy, configuration-model shuffle | Static graph cannot estimate `Delta W -> Delta g`; direct structural synthetic baseline only. If no independent perturbation generator, link prediction remains descriptive structural association. |
| R-GRID-TORUS | state contrast in independently estimated topology `T` and local transition/mobility statistic `G`: test `T_wake \approx T_REM \approx T_SWS` while `G` differs | animal/module; split spikes into topology block and metric/transition block, then state blocks | topology bootstrap distance and conditional transition score; rate, occupancy, time-shift, flat-pullback | Shared spikes for topology and metric create source dependence. Stable topology does not prove a metric field or `W` link. |
| R-DANDI-37 | frozen earlier-day `g0`/deformation predicts next-day activity or behavior beyond earlier dynamics | animal, tracked ROI/day; train early days, test later days, leave animal out when possible | next-day conditional log/energy score; persistence, `v,Q`, gain, noise, cell-match shuffle | No verified ROI identity, later endpoint, or animal grouping makes it descriptive. Activity-only data can test longitudinal `H_G`, not `H_W`. |
| R-ALLOPTICAL | stimulation-estimated response ellipsoid/mobility predicts unperturbed held-out path law: `p(x_future | x0,g_response)` | animal; stimulation trials/directions for field, separate unperturbed trials and held-out directions | conditional log/energy score; stimulus amplitude/spread, direct `v,Q`, covariance-only, gain/noise | Same responses used to fit and score are circular. Without separate unperturbed target or animal split, only system-ID description. Maximum claim `H_G`. |
| R-BCI | preregistered decoder perturbation length predicts held-out learning curve or transfer | subject/session; fit baseline metric before decoder change, hold out decoder perturbation classes/subjects | learning-curve likelihood or slope prediction; Euclidean decoder displacement, decoder condition, rate, task difficulty | Decoder geometry is imposed action-space structure, not neural `g` unless calibration establishes the map. Maximum behavioral accessibility. |
| R-MICRONS | static `Phi(W^s)` predicts held-out functional similarity/response conditional on anatomy | cell/edge; block by animal/area and hold out cell pairs; prevent shared-neuron leakage | conditional Gaussian/categorical score; physical distance, layer/type, degree, shuffled connectome | Endpoint connectome is static: cannot test change or causal mediation. If `Phi` and target share the same functional response summary, source leakage kills bridge claim. |
| R-SYNTH | recovery of known generator class and held-out trajectory law under M0 | independently seeded simulated circuits; train/test seeds and perturbations disjoint | class recovery, SPD-field error, path NLL/energy, empirical false-positive rate | Estimator validation only. It cannot add biological evidence regardless of perfect recovery. |

## Algebraic and causal boundary checks

1. A covariance/metric computed from fitted `J,Q` and scored on trajectories
generated by that same fitted model is an algebraic re-expression.  Cross-fold
prediction does not create an independent `g -> x` intervention.
2. At horizon one, `C_{Q,1}=Q`; hence any common-ridge `S4-H=1` and `S3`
comparison is structurally tied, not convergent evidence.
3. Nonconstant coordinates do not imply curvature: polar Euclidean
`ds^2=dr^2+r^2dtheta^2` has nonzero Christoffels but zero Riemann tensor.
Curvature needs a `C2` SPD field, tensor statistic, chart-resampling, and a
flat-pullback comparator.
4. `d_g(A,B)` does not determine first-passage time.  With the same distance
from 0 to 1, `dz=2dt` has hitting time `1/2` whereas `dz=dt/2` has time `2`.
Time claims require an explicit SDE/control law and matched direct dynamics.
5. `W^s` alone does not uniquely determine `g`; a named `Phi` can be tested,
but no route may infer uniqueness from a positive association.

## Common synthetic ground-truth suite

Use the frozen chart `z in R^d`, `d=3`, eight named nodes, 20 independent
seeds per generator, and a train/test split by seed plus held-out intervention
directions.  Simulate `dz=v(z)dt+B(z)dB_t`, record the true field, paths,
first-passage outcomes, and structural graph where applicable.  Fit each
candidate exactly as for the public route and score on unseen seeds.

| Generator | Truth | Required result / falsifier |
|---|---|---|
| G1 true metric bridge | `M=M0+alpha(z)uu^T`, `g=M^-1`, `v=-g^-1 grad V`, structural `W` drives `alpha` through frozen `Phi` | recover positive deformation and beat direct/gain/noise only when bridge is correctly specified |
| G2 direct dynamics only | `v` changes with `W`; `g=g0` | reject metric mediation; direct `v,Q` wins or ties |
| G3 flat pullback | Euclidean latent dynamics observed under smooth diffeomorphism `f`; `g=f*I` | flat-pullback ties; curvature test has calibrated false-positive rate |
| G4 gain/noise | fixed geometry, condition-dependent scalar gain and/or `Q` | gain/noise baseline absorbs effect; no metric-specific discovery |
| G5 topology-only | add/delete graph shortcut while local SDE and `g` remain fixed | graph/topology feature wins; metric-deformation claim rejected |
| G6 null | fixed `W,g,v,Q`, randomized labels | nominal familywise error after Holm; no systematic deformation |

Pass criteria are prespecified: SPD violations equal zero after declared
regularization; G1 has directionally correct recovery on at least 90% of seeds;
G2--G6 yield no Holm-significant metric advantage above 5% familywise false
positive frequency; flat-pullback curvature false-positive frequency is at most
5%.  Failure invalidates the estimator for all biological `H_G/H_C` claims,
not the biological hypothesis itself.

The machine numerator is fixed as follows. For each of G2--G6 and each of its
20 seeds, apply Holm once to the complete frozen metric-specific claim family
for that seed. Let $I_{g,s}=1$ when any metric-specific claim survives. The
false-positive numerator is $\sum_{g=2}^{6}\sum_{s=1}^{20} I_{g,s}$ over the
fixed denominator 100 and must be at most 5. For G3 curvature, let $C_s=1$
when its frozen curvature claim survives its within-seed Holm family; the
denominator is 20 and $\sum_s C_s$ must be at most 1. G1 directional recovery
uses denominator 20 and requires at least 18 successes. Missing or nonfinite
seed scores fail the suite rather than reducing a denominator.

## Reproducibility commands

The future implementation must write its seed, generator configuration, true
parameters, candidate estimates, and held-out scores to a machine-readable
artifact.  Minimal required focused commands are declared as:

```powershell
uv run python _workspace/ce/neural-riemannian-metric-multiroute-execution-20260818/artifacts/run_synthetic_suite.py
uv run python _workspace/ce/neural-riemannian-metric-multiroute-execution-20260818/artifacts/verify_synthetic_suite.py
```

These commands are declarations only at this pre-implementation lane; no
outcome has been inspected here.
