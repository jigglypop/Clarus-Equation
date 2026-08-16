# G9-CBM V2 independent red-team contract

Status: COMPLETE

## Verdict

Implement **R1 only**, but do not carry the G7-M V2 generator into the
integrated claim.  That generator makes every held-out valid item the exact
known-slot concatenation of a prefix, connector, and suffix, and its hard recall
copies the hidden coordinates of a stored exemplar.  Extending it with an action
head would make both advertised effects true by construction.  Reuse only the
real-only/provenance boundary, fixed-capacity store shape, and constrained
proposal discipline.  Use fresh V2 seeds and a new generator.

The smallest defensible experiment is a four-dimensional controlled system in
which a previously observed entity is encountered again under independent noise
and new actions.  The old record identifies an entity-specific drift but contains
none of the scored future.  Separately, wake data omit three of nine valid
prefix/action compositions per context-local port; an offline pass may estimate
those slots from observed fragments.  Thus LTM and schema completion solve
different missing variables and neither path receives evaluator truth.

## 1. Frozen minimal generator

All quantities below are dimensionless.  Use `float64`, CPU/NumPy, `p=4`, scalar
action value `a in {-1,0,+1}`, `H=20`, and one evaluation origin per seed.

### 1.1 World and wake ledger

For each seed create two contexts `c in {0,1}`, four context-local ports
`g in {0,1,2,3}`, three opaque prefix types `r in {0,1,2}`, and three opaque
action types `u in {0,1,2}` with public numeric values `a=(-1,0,+1)[u]`.
Independently permute every context, prefix, action, and port token; strings may
not contain an integer role, seed, validity bit, or episode ordinal.

For every `(c,g)`, expose exactly the six wake pairs

```
E_obs = {(0,0),(0,1),(1,1),(1,2),(2,2),(2,0)}
```

and hold out the three generator-valid pairs

```
E_miss = {(0,2),(1,0),(2,1)}.
```

There are two independent real entities per observed pair, hence
`2 contexts * 4 ports * 6 pairs * 2 = 96` append-only wake records.  Each entity
has a four-coordinate fingerprint `f_e` and drift `theta_e`; they are independent
draws.  The fingerprint is observable, noisy, and used only as a partial content
cue.  It is statistically independent of `theta_e`, schema parameters, goal,
future noise, and candidate rank.  Include near-neighbour unstored fingerprints
as lures, so recall is not exact ID lookup.

Each real record is a `12 x 8` observed history `Y_t=(x_t,f_e)` and ends before
the evaluation origin.  Its three phases are prefix steps `0..4`, connector steps
`5..6`, and suffix steps `7..11`.  With fixed diagnostic actions, their additive
drifts are respectively

```
theta_e + alpha[c,g] + beta[c,g,r]
theta_e + alpha[c,g]
theta_e + alpha[c,g] + gamma[c,g,u] + eta[c,g,r,u].
```

Use independent bounded draws

```
theta in {-0.06,+0.06}^4
alpha in U[-0.04,0.04]^4
beta,gamma in U[-0.05,0.05]^4
eta in U[-0.01,0.01]^4
epsilon_t in U[-0.01,0.01]^4.
```

The scored controlled transition is

```
x[t+1] = clip(D*x[t] + B*tanh(x[t]) + C*a[t]
              + theta_e + alpha[c,g] + beta[c,g,r]
              + gamma[c,g,u[t]] + eta[c,g,r,u[t]] + epsilon[t+1],
              -1, 1)
```

with `D=diag(0.55,0.50,0.45,0.40)`, `B[1,0]=0.08`,
`B[3,2]=-0.07`, all other `B=0`, and
`C=(0.12,-0.10,0.08,0.06)^T`.  The learner is not handed these generator
parameters.  It fits the registered diagonal/two-edge/action model by one
`ridge=1e-6` closed-form solve on the identical real wake ledger in all cells;
the edge set and feature order are frozen, with no edge or model selection.

The evaluation entity is a uniformly chosen context-0 wake entity and is
re-encountered with the same `f_e,theta_e` but a new 12-step diagnostic prefix,
new process noise, and new future actions.  The stored record and evaluation
episode IDs, timestamps, action arrays, and all noise arrays are disjoint.  The
current suffix after the origin is never inserted.  Selection of the entity and
all evaluation arrays happens before any cell is constructed.

### 1.2 R1 memory and schema adapters

The current cue exposes 32 of 96 coordinates: 16 fingerprint coordinates and
16 state coordinates selected by frozen per-slot permutations, with bounded
cue noise.  Values outside the mask are poison and must never be read.  Hard
recall must rank only records in the cue's learner-inferred context/component;
the inherited G7-M V2 implementation does **not** do this after its precheck and
therefore needs a V2 wrapper/remediation, not a false byte-equivalence claim.

On acceptance, clamp all observed current coordinates, fill masked history
coordinates from the recalled real record, discard `episode_id`, and compute the
fixed prefix condition

```
q = mean_t(completed_x[t+1]
           - core(completed_x[t], diagnostic_a[t])
           - schema_fallback(observed_key[t])).
```

Rejection uses the same formula after causal/schema imputation.  `q` is frozen
for all candidates and is added at every rollout step.  It cannot contain a
stored post-origin row.

Build the learner graph solely from the 96 wake records.  For each accepted
G7-M-shaped `DreamBinding`, decode prefix, connector, and suffix transition
residual means `R_P,R_C,R_S` and write

```
s_hat[c,g,r,u] = R_P + R_S - R_C
```

to the already allocated missing slot, unit weight, once.  The small target
interaction `eta` is not exactly reconstructible; therefore dream is an
estimate, not an oracle concatenation.  Observed slots and episode storage are
immutable.  The no-dream schema-only fallback is the component mean computed
from exactly the same wake residuals; it may not be zeroed or weakened.

The active candidate is only

```
xhat[t+1] = clip(Dhat*xhat[t] + Bhat*tanh(xhat[t]) + Chat*a[t]
                 + q + s_hat[c,g,r,u[t]], -1, 1).
```

No R2 residual recursion, R3 nearest-edge rollout, R4 posterior, V6 consensus,
or V7 AR replacement may be run or selected on train.

## 2. Exact action set, validity, goal, cost, and planner

Use six valid numeric codewords, fixed before seeds:

```
V0 = [-1]*20                    V3 = [-1]*10 + [+1]*10
V1 = [ 0]*20                    V4 = [+1]*10 + [-1]*10
V2 = [+1]*20                    V5 = repeat([-1,0,+1])[:20].
```

An action also carries an opaque source-port token.  A step is generator-valid
iff `(context token, current prefix token, action token)` belongs to the same
true context-local port, the numeric value matches that action token's public
value, and the previous output/current input continuity token matches.  Port is
held fixed in this minimal V2; continuity still must be checked at every step.

Add two invalid lures: copy `V3` and replace only step 7's action token by the
same-valued token from port `(g+1) mod 4`; copy `V4` and replace only step 13 by
the same-valued token from the other context.  Thus values remain in bounds and
invalidity cannot be detected from magnitude.  All eight sequences are rolled
for 20 steps.  Presentation order is independently permuted, while tie breaking
uses the lowest pre-permutation canonical index.

Use one evaluator-only common noise array for all eight counterfactuals and all
four cells.  Uniformly choose one of `V0..V5` with a separate task stream and set
the public terminal goal `g` to that valid counterfactual's `x[20]`.  The chosen
index is never exposed.  A seed-world may be rejected only until it has at least
one valid candidate with cost `<=0.05` and one valid candidate with cost `>0.20`;
the rejection routine may inspect generator outcomes but may never instantiate
or score M00/M10/M01/M11.  Its transcript and count are evaluator-only and
reported.  Resampling for an LTM/dream effect is forbidden.

For a valid sequence define

```
J_valid(k) = (1/4)*sum_j (x[k,20,j]-g[j])^2
             + 0.01*(1/20)*sum_t a[k,t]^2.
J(k) = J_valid(k) if generator-valid, else P_invalid=5.
```

Because `x,g in [-1,1]^4`, every valid cost is at most `4.01`, so the invalid
penalty is bounded without reading outcomes.  The candidate uses the identical
formula on `xhat` and assigns 5 only from its wake-inferred graph.  Select the
minimum `(predicted_cost, canonical_index)`.  Evaluator regret and success are

```
k* = argmin_k (J(k), canonical_index)
R  = J(khat)-J(k*)
S  = 1[J(khat)<=0.05 and khat is generator-valid].
```

Assert `R>=-1e-12`; a lower value is evaluator failure.  Prediction scores use
all six valid rollouts.  `E_uv` contains every scalar at a lead whose `(r,u)` is
in `E_miss`, whether or not a dream proposal was accepted.  Invalid lures are
excluded from state-NRMSE but included in invalid-selection rate.  H5 is exactly
`H20[:5]`.

## 3. Effects that would otherwise be constructed or trivial

Any one of the following invalidates the intended interpretation:

1. A stored row shares evaluation future noise, actions, episode ID, timestamp,
   or a post-origin observation; this tests lookup leakage, not memory.
2. Fingerprints/tokens encode `theta`, port number, validity, missing-edge status,
   goal action, oracle rank, or future identity.  Fingerprints must be noisy,
   partial, independent keys with near-neighbour unstored lures.
3. The target missing slot is an exact concatenation/linear decode of the dream
   fragments.  The independent bounded `eta` term is required and all fixed
   missing keys are scored, including rejected proposals.
4. The baseline fallback is zero, denied the same real wake data, or has smaller
   schema capacity.  Use the strongest fixed component-mean fallback in every
   cell.
5. A recalled episode ID, stored suffix, confidence-selected extra computation,
   or hidden cue coordinate reaches the rollout model.
6. The true validity graph filters proposals, candidates, or predictions.  Only
   the evaluator may use it after outputs and selection are hashed.
7. Invalid lures are out of bounds or visibly marked.  They must differ only by
   opaque context/port/continuity incompatibility.
8. Candidate order or construction encodes oracle rank; goal generation exposes
   the selected codeword; or seed rejection examines a treatment result.
9. Dream accepts only cases known to improve error, overwrites an observed slot,
   creates an episode identity, or enters LTM even as a rejected insertion
   attempt.
10. Hard recall performs its current all-bank argmax after a component precheck;
    cross-context accepted recall must be structurally impossible, not merely
    rare.
11. Metrics omit abstentions, nonfinite outputs, uncovered missing keys, hard
    cases, or losing seeds; H5 is a second rollout; or candidate/cell noise differs.
12. C1 is inferred from M11-M00 rather than the factorial LTM marginal, C2 from
    an averaged dream contrast rather than both matched simple effects, or a
    positive interaction is selected after inspecting several metrics.
13. Per-cell tuning, validation recalibration, unequal proposal/query/rollout
    calls, Python early stopping, or a larger active schema/store makes the
    treatment a capacity/compute bundle.

## 4. Non-negotiable pre-seed tests

All tests use explicitly off-range development seeds; no registered train,
validation, or test seed may be imported or generated.

- Candidate signatures accept only the masked observed prefix, opaque public
  tokens, the eight action sequences, public goal/cost, frozen core, real-only
  store facade, learner schema, and learner graph.  They cannot accept a world,
  seed/stream, episode object, future, latent, true validity, realized cost, or
  evaluator handle.
- Independently mutate future states/noise, `theta/eta`, true validity labels,
  realized costs, goal-source index, and split metadata while holding candidate
  inputs fixed.  Predictions, audit bytes, and selected index must be bit-equal.
- Poison every cue and prefix value outside its observed mask, including NaN and
  huge finite values.  Recall, `q`, rollout, and selection must be bit-equal.
- Assert ledger maximum timestamp `<` evaluation origin; evaluation IDs/noise/
  actions are disjoint; store contents are hash-identical before and after all
  calls; and no current evaluation suffix can be inserted before scoring.
- Evaluate all 24 cell orders and several presentation permutations.  Per-cell
  outputs mapped to canonical action order must be bit-equal.  Worlds, masks,
  goals, actions, wake ledgers, core bytes, RNG arrays, and inferred-graph bytes
  must hash identically across cells.
- Assert H5 is a bitwise slice of H20; every candidate receives 20 transitions;
  all ties use canonical first index; and invalid lures have the same numeric
  arrays as their valid templates except opaque tokens.
- Hard-zero integer invariants: synthetic with episode ID, synthetic observed,
  synthetic recalled, synthetic-to-LTM attempts, synthetic-to-LTM successes,
  nonledger real LTM records, observed overwrite/hash change, cross-context or
  cross-component accepted splice/recall, action/port/continuity accepted
  violation, held-out/evaluator reads, early test reads, nonfinite outputs.
- Shuffled binding permutes entity-to-trajectory content within context/port
  strata while preserving fingerprints, acceptance-score marginals, capacity,
  order, and calls.  Require paired LTM benefit over shuffled.  A zero-`q`
  ablation must remove the LTM prediction/planning effect.
- Unconstrained dream uses identical fragments, proposal order, 24 checks and 24
  update slots, removing only graph/join rejection.  It must have a strictly
  higher paired generator-invalid splice/transition rate; zeroing synthetic slots
  must remove the dream effect.  Otherwise constraint/schema attribution fails.
- Test unlock requires a passing validation artifact plus byte hashes of raw
  preregistration, generator, candidate, evaluator, tests, inherited provenance
  module, calibration, and budget ledger.  Any change means a new version and
  fresh seeds.

## 5. Exact common budget and gates

Per seed and cell freeze the integer ledger

```
(N_wake,T_wake,U_core,P_core,K,H,N_transition,N_score,
 B_ltm,Q_ltm,B_schema,B_proposal,U_dream,N_pass) =
(96,12,1,10,8,20,160,8,96,17,72,24,24,1).
```

`Q_ltm=17` is the integrated origin plus eight positive recall audits and eight
near-neighbour unstored lures.  Allocate `96*12*8=9216` LTM float64 slots
(`73,728` bytes) and `72*4=288` schema float64 slots (`2,304` bytes) in every
cell.  Inactive factors use unqueryable inert arrays and execute fixed-shape
no-op calls; results are discarded without reading proposals.  Cap temporary
numeric workspace at `262,144` bytes per cell and fail on over-budget or unequal
counts.  Wall time is report-only.

Use train-standardized pooled RMSE exactly as defined in `11-math.md`, with every
valid rollout and every unseen-valid coordinate retained.  Retain the corrected
registered all-of gates: marginal LTM and both matched dream reductions `>=10%`,
paired lower CI `>0`, strict seed wins `>=0.65`; M11/M00 regret reduction `>=20%`
with paired lower CI `>0`; success gain `>=0.10` with paired lower CI `>0`;
M11 success `>=0.75`; H20/H5 `<=2`; and predicted generator-constraint violation
rate on valid rollouts `<=0.01`.  Also require M10 and M11 recall coverage and
identity `>=0.80`, both wrong rates and mean lure false recall `<=0.05`, plus the
hard-zero provenance counts above.  Report every simple effect and interaction;
`synergy` is allowed only for one preregistered metric with lower CI `>0`.

The locked test remains unopened after any validation failure.  A PASS supports
only a conditional software-component claim in this finite generator, not
general world modelling, biological memory, dreaming, consciousness, or AGI.
