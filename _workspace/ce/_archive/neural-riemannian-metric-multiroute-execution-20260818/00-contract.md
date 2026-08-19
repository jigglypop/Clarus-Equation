# Neural Riemannian metric multiroute execution contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/neural-riemannian-metric-independent-tests-20260818
PREDECESSOR: _workspace/ce/neural-riemannian-metric-validation-20260818
PREDECESSOR: _workspace/ce/_archive/sleep-replay-routing-realdata-20260818

## 1. Objective

Execute as many genuinely distinct validation routes as current local and
officially accessible public data allow for the hypothesis

$$
\Delta W^s\longrightarrow\Delta g(z,c)
\longrightarrow\Delta p(x_{0:T},\tau_B\mid x_0,c).
\tag{1}
$$

The run must distinguish a successful computation from biological evidence.
It must attempt every frozen route below, execute every eligible route, and
record exact missing inputs or access failures for every unexecuted route.

## 2. Frozen route universe

| ID | Data target | Independent question | Maximum claim |
|---|---|---|---|
| `R-E17-F3` | released Figure 3 spine selectivity/clustering | does functional synaptic organization predict a separate clustering or task endpoint? | static structure-function component |
| `R-E17-F4` | released Figure 4 dendrite representation-drift data | does an earlier representation/metric proxy predict later drift on held-out units or animals? | exploratory $H_G$ component |
| `R-E17-F5` | released Figure 5 transition-error data | does a frozen pre-transition geometry proxy predict later transition error beyond rate/selectivity baselines? | exploratory $H_G$ component |
| `R-E17-F2` | predecessor Figure 2 tournament | locked reference only; do not count a rerun as independent evidence | retrospective feasibility |
| `R-SLEEP-E19` | local E19 sleep-stage/neural/encoding MAT files | do independently separated sleep-state summaries predict later encoding or trajectory fidelity? | sleep association only |
| `R-SLEEP-E15` | local replay/sleep-deprivation processed data | do sleep/replay dose changes predict independent later endpoints beyond time/state composition? | sleep association only |
| `R-CELEGANS` | local full *C. elegans* structural edge list | do structural diffusion/control/topology metrics yield distinct predictions under synthetic held-out perturbations? | structural/synthetic baseline |
| `R-GRID-TORUS` | Gardner et al. official wake/REM/SWS data | is topology stable while a separately estimated local metric or transition law changes across state? | topology-metric dissociation |
| `R-DANDI-37` | official DANDI 000037 longitudinal M1 data | does pre-period $g_0$ predict later activity/behavior and relative deformation across learning days? | longitudinal $H_G$ only |
| `R-ALLOPTICAL` | official Wagenmaker et al. response/stimulation data | do held-out stimulation response ellipsoids predict separate unperturbed paths beyond direct dynamics? | perturbational $H_G$ only |
| `R-BCI` | official Sadtler or successor BCI data | does frozen metric length of imposed decoder perturbation predict held-out learning curves? | behavioral accessibility |
| `R-MICRONS` | official MICrONS query/API subset | does a preregistered static $\Phi(W^s)$ predict held-out functional similarity beyond anatomy and degree? | static bridge, not $H_W$ |
| `R-SYNTH` | generated known-ground-truth systems | can the full estimator distinguish true metric, flat pullback, gain/noise, direct dynamics, and null cases? | estimator validation only |

No route may be added after its outcomes are inspected. A public target may be
replaced only by an official mirror of the same frozen dataset.

## 3. Eligibility and provenance

Every route receives exactly one status:

- `ELIGIBLE`: all required raw inputs, independent target, sample identity,
  license, and split fields exist;
- `PARTIAL_DESCRIPTIVE`: a calculation is possible but cannot test the declared
  link because the target or sample unit is aggregated;
- `UNTESTABLE_MISSING_INPUT`: a required typed input is absent;
- `ACCESS_BLOCKED`: the official data exist but cannot be retrieved after a
  documented attempt;
- `INELIGIBLE_DEPENDENT`: predictor and target are algebraic or data-source
  duplicates;
- `FAILED_EXECUTION`: eligible inputs existed but the frozen implementation
  failed.

Record official URL, access date 2026-08-18, file size, SHA-256, license,
schema, sample unit, and missingness before outcome analysis. Existing local
files are authoritative inputs and must not be overwritten.

## 4. Common analysis rules

1. Freeze preprocessing, chart, candidate family, horizon, regularization,
   endpoint, split, tie rule, and random seed before reading route outcomes.
2. Use animal or independently randomized circuit as the inferential unit.
   Cells, spines, windows, paths, and sessions are nested observations.
3. Fit charts and $g_0$ on training/pre-period data only. Compare baseline
   producers `P-h`, `P-W`, `P-D`, and `P-C` only when their typed inputs exist.
4. Any $H_G$ route must beat parameter-matched direct $v,Q$, Euclidean,
   persistence, gain-only, noise-only, and flat-pullback controls on an
   independently held-out endpoint.
5. Metric, directed action, graph distance, topology, control energy,
   representational geometry, and distribution distance remain separate
   leaderboards.
6. Multiple tests use a frozen family and Holm correction. Descriptive routes
   report effect direction and uncertainty without confirmatory population
   language.
7. Reusing predecessor E17 data, formulas, or outcomes is not an independent
   replication and must be labeled as such.

## 4A. Existing folds and relative deformation

Three geometries remain distinct in every executable route:

- $h$ is measured physical surface geometry: cortical folds, laminar depth,
  physical separation, wiring length, and cell type belong here or enter as
  explicit anatomical nuisance covariates.
- $g_0(z,c)$ is the pre-existing baseline state-space metric. It is fit on
  training or pre-manipulation data only and is never inferred from a
  post-manipulation endpoint.
- $g_t(z,c)$ is the post-state or induced candidate metric. A route tests the
  relative change from $g_0$, not the existence of an unconstrained new
  geometry.

For SPD fields in a common chart, report the relative deformation through the
generalized eigenvalues of $(g_t,g_0)$ and the affine-invariant magnitude

$$
D(g_0,g_t)=\left\|\log\left(g_0^{-1/2}g_tg_0^{-1/2}\right)\right\|_F.
$$

Under a chart change $y=Px+b$, both metrics transform as
$g'=P^{-T}gP^{-1}$; the regularizer and uncertainty model must transform with
the same declared law. Every nonconstant-field claim must beat a nonlinear
flat-pullback null. Longitudinal biological routes additionally control for
$h$, depth/layer, physical and wiring distance, and cell type when those
fields exist. E17's sessionwise constant SPD candidates measure neither the
folded physical surface $h$ nor an independently observed nonconstant $g_0$
and remain descriptive.

## 5. Route-specific primary endpoints

- `R-E17-F3`: held-out clustering/task statistic under source-supported unit
  pairing; otherwise descriptive reproduction only.
- `R-E17-F4`: later drift or future-session representation error, with baseline
  selectivity/rate and identity permutation controls.
- `R-E17-F5`: transition-error proper loss or source statistic, predicted from
  a strictly earlier/frozen geometry proxy.
- `R-SLEEP-E19/R-SLEEP-E15`: subject/animal-level later encoding or replay
  endpoint; window/event counts cannot be the sample size.
- `R-CELEGANS`: edge-held-out structural link prediction and synthetic
  perturbation recovery only, with degree/shortest-path/diffusion/control
  baselines.
- `R-GRID-TORUS`: topology signature from one block and local metric/transition
  statistics from a disjoint block, compared across wake, REM, and SWS.
- `R-DANDI-37`: next-day activity/behavior proper score from frozen earlier-day
  baseline and deformation.
- `R-ALLOPTICAL`: held-out input directions and animals; stimulation-derived
  response geometry predicts unperturbed future response/path data.
- `R-BCI`: subject/session-held-out learning curve from preregistered decoder
  perturbation length.
- `R-MICRONS`: edge/cell-held-out functional similarity, with physical distance,
  cell type, degree, and shuffled-connectome controls.
- `R-SYNTH`: recovery and false-positive rates across known true, direct-only,
  flat-pullback, gain/noise, and null generators.

## 6. Kill conditions

A route cannot support the theory if its effect disappears under sample-level
resampling, survives identity permutation, is matched by direct dynamics or a
baseline producer, requires outcome-driven preprocessing, changes direction
across independent animals without a declared moderator, or lacks the typed
input for the claimed arrow. A failed full-chain arrow is not rescued by a
different partial route.

## 7. Deliverables and completion

- `10-sources.md`: official access/provenance and eligibility ledger.
- `11-math.md`: route-specific estimands, independence audit, and nulls.
- `12-routes.md`: execution matrix and fallback order.
- `20-audit.md`: preregistration gate before implementation.
- `30-implementation.md`: scripts, input manifests, and run disposition.
- `31-validation.md`: hashes, focused tests, outputs, and validator results.
- `40-final-report.md`: Korean result synthesis with no biological promotion.

Completion requires an attempted disposition for every frozen route, successful
execution and validation for every `ELIGIBLE` route, machine-readable raw result
and failure ledgers, an independent stable-snapshot audit, and a final statement
of which parts of (1) remain untested.
