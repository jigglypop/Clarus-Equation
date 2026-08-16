# G9-CBM V1 integration-route exploration

Status: COMPLETE

## 1. Fixed target, scope, and evidence boundary

The target is an outcome-blind, CPU-only integration which, in the frozen small
synthetic domain, lowers autonomous H20 state NRMSE (with the registered M11/M00
ratio at most 0.90), lowers unseen-valid H20 NRMSE in each matched dream contrast
by at least 10%, and lowers planning regret (M11/M00 ratio at most 0.80), while
all provenance, invalid-transition, false-recall, test-lock, and equal-budget
gates remain satisfied.

The four routes below are **gate-aware but registered-outcome-blind**. They were
constructed after reading the frozen V1 contract and inherited results, but no
G9-CBM train, validation, or test seed was run. The failures of G9-CB V5--V7 are
used as inherited counterevidence, not as data for tuning this experiment.

### 1.1 What the inherited modules actually support

| Inherited item | Reusable fact | Active limitation that must remain visible |
|---|---|---|
| G9-CB V4 | `BridgeModel.predict(x)` implements the sparse mechanism; pooled-train scalar AR plus prefix-only residual center/direction/intercept passed a sequential one-step gate. | Four fully observed charts, rank-one residual, matched `tanh` basis, no actions, and true state was reread at every scored step. |
| G9-CB V5 | `free_rollout(mechanism, x_previous, x_anchor, horizon, residual_filter)` is a good leakage pattern: it accepts no `Episode`, future, outcome, or hidden state and recursively feeds predictions. | Finite H20 rollout nevertheless failed four registered robustness checks; paired CIs versus persistence were negative. |
| G9-CB V6 | Prefix-only expert backtesting and `weighted_state_consensus` show a leak-free arbitration implementation. | Consensus was 3.63% worse than V5 at H5 and 1.26% worse at H20; prefix expert ranking did not transfer. It should not be the active V1 route. |
| G9-CB V7 | Nonzero-lag AR estimation was dimensionless, stable, and much closer to the programmed scalar. | Better scalar identification did not improve the misspecified whole rollout: H20 was 1.70% worse than V5. Do not equate parameter accuracy with world-model accuracy. |
| G7-M V2 | The exact real-only `PersistentEpisodicStore`, `hard_cue_anchored_recall(PartialCue)`, `constrained_missing_binding_dream(...)`, and `update_missing_slow_binding(...)` boundary passed validation and locked test. A 96-item bank is 73,728 trace bytes. | It is deterministic exemplar completion plus known-slot schema recombination, not an attractor, consolidation, world rollout, or planner. Its interaction was approximately zero. |
| memory_replay_planning G7 V4 | Fixed candidate enumeration and model-based macro-action selection demonstrated that remembered actuator semantics can change behavior: cost was 68.5% below zero action and 92.7% below recency MPC in its toy regime. | It stores five-parameter prototypes, not episodes. `_macro_plan_cost(model_coefficients, true_coefficients, cfg)` co-locates predictor and truth and must not be reused as the new evaluator-facing API. |

The resulting design rule is: use V4 as the supported causal core, V5 only as
the no-future-read interface pattern, keep V6/V7 as negative controls or
counterevidence, reuse the G7-M V2 provenance implementation exactly, and split
G7-style candidate rollout from evaluator execution.

### 1.2 Forced versus free structure

Forced by the contract:

- The cells are `M00`, `M10`, `M01`, and `M11`; worlds, wake records, observed
  prefixes, candidate action arrays, seeds, model capacity, and budgets are
  paired exactly.
- Episodic storage contains real wake records only. Synthetic output is always
  `source=synthetic, epistemic_status=hypothetical`, is never an episode
  identity, and can update only a slow schema/world component.
- The candidate never reads a scored future, evaluator latent, evaluator
  validity label, counterfactual outcome, or test record. H5 is the first five
  rows of the same H20 rollout.
- Every one of the fixed action sequences is rolled out before deterministic
  selection. An inferred graph may mark or penalize a candidate but may not
  remove candidates or consult generator truth.
- Planning selection and evaluator scoring are separate calls. The oracle is
  diagnostic and cannot supply a candidate prediction or selection score.
- Validation is run once and test stays locked unless every validation gate
  passes. Any post-freeze change creates V2 with fresh seeds.

Free, and therefore the subject of this route search:

- where the recalled exemplar enters the world model;
- how hypothetical bindings update the slow world component;
- whether memory acts in observation space, residual-state space, transition
  graph space, or parameter-posterior space;
- the additional model capacity and train-only calibration choices.

## 2. Common leak-free interface and data flow

All four routes use the same learner/evaluator separation. This is part of each
route, not an optional test convenience.

```python
@dataclass(frozen=True)
class PrefixPacket:
    cue: PartialCue                 # exact G7-M type; values [12, 8], mask [12, 8]
    past_states: np.ndarray         # [P + 1, d], observed rows only
    past_state_mask: np.ndarray     # same shape; masked values are never read
    past_actions: np.ndarray        # [P, m], actions already taken
    public_tokens: tuple[str, ...]  # opaque observed context/port tokens only

@dataclass(frozen=True)
class RolloutRequest:
    prefix: PrefixPacket
    candidate_actions: np.ndarray   # exact shape [K, 20, m]

@dataclass(frozen=True)
class CandidateBatch:
    states: np.ndarray              # [K, 20, d]
    inferred_valid: np.ndarray      # [K, 20], learner graph only
    provenance_audit: tuple[dict, ...]
    recall_audit: dict

def predict_candidates(
    model: FrozenActionWorldModel,
    request: RolloutRequest,
    components: CellComponents,
) -> CandidateBatch: ...

def choose_action(
    predictions: CandidateBatch,
    actions: np.ndarray,
    public_task: PublicTaskSpec,
) -> int: ...                       # minimum predicted cost, first-index tie break

def score_selected_action(
    evaluator_handle: EvaluatorOnlyHandle,
    selected_index: int,
) -> EvaluatorMetrics: ...          # called only after selection is frozen
```

`predict_candidates` has no seed, episode object, evaluator handle, future,
hidden state, reward outcome, or generator-validity argument. Poisoning every
masked cue value, every scored future, every evaluator latent, and every
counterfactual outcome must leave all non-oracle candidate bytes and the chosen
index unchanged.

### 2.1 Exact reuse of the G7-M V2 boundary

A frozen deterministic `TrajectoryCodec` maps an already-observed wake fragment
to the exact V2 object:

```python
EpisodicRecord(
    episode_id=opaque_real_id,
    context_token=observed_context_token,
    prefix_token=observed_input_port_token,
    suffix_token=registered_action_output_port_token,
    trajectory=observed_only_float64_array_12_by_8,
    provenance=REAL_PROVENANCE,
)
```

The 12-by-8 view contains only history available before the prediction origin;
it may encode past state, action, and public port features but never the current
case's scored future or a generator latent. Action/output-port tokenization is
registered before train and is opaque to the memory implementation. The codec,
mask convention, and learner/evaluator stripping function must be hash-locked
with the implementation.

The only permitted memory calls are:

```python
store.insert_real(real_record)
recall = hard_cue_anchored_recall(store, prefix.cue)

bindings = constrained_missing_binding_dream(
    tuple(store.records), standardizer, frozen_join_threshold
)
accepted = update_missing_slow_binding(slow_table, binding)
```

For `M01`, an identical read-only real wake snapshot is used for the offline
pass and then destroyed; its evaluation-time store has zero records. For
`M00/M01`, recall is executed against an empty-store facade so the call count is
the same. A decoded `DreamBinding` retains synthetic/hypothetical provenance
through every later object; no constructor is allowed to cast it to real or
recalled. If byte reuse of these functions is impossible, semantic equivalence
of insertion rejection, masked-coordinate non-read, first-index tie breaking,
component precheck, observed overwrite rejection, and provenance fields is a
stop condition before registered train.

### 2.2 Shared causal/action core

Let `d` be observed-state width, `m` registered action-feature width, `E` the
frozen sparse directed edge set, and `k_t` an inferred public
context/port/action schema key. The common deterministic mean model is

\[
\widehat x_{t+1}
= D\widehat x_t+B\tanh(\widehat x_t)
+G\phi(a_t)+s(k_t)+c(q)+\widehat r_{t+1}.
\]

`D,B` follow the V4 intervention/freeze pattern. `G` is fit from randomized
wake actions only. `s(k)` is a preallocated slow-schema entry; absent entries
use one frozen schema-only fallback. `q` is a prefix condition and
`r` is optional route-specific residual state. Generator truth is used only by
the evaluator, never by the learner validity graph.

The learned common-core dimension is

\[
p_{core}=d+|E|+d m+1,
\]

for diagonal self terms, sparse bridge terms, action effects, and one temporal
scalar, before the preallocated `dK` slow-schema slots. Every cell has the same
`p_core+dK` capacity; dream changes contents of permitted missing slots, not
capacity. No-dream cells perform a fixed-size no-op update schedule so all four
cells receive the same registered call/step budget. Stochastic innovations are
not predicted.

The inherited G7-M calibration *rules* contribute three common train-derived
scalar outputs for this new domain (pre/post-bank hard-recall thresholds and one
join threshold). Their algorithms, pooling, strict comparison, and write-once
artifact are frozen before train; their new numeric values are not copied from
the old G7-M seed family. They are common calibration outputs, not route-specific
knobs.

The inferred `ValidityGraph` is built only from real wake co-occurrence,
observed port/action compatibility, and observed continuity. It exports
`check(previous_public_state, action, proposed_public_state)` and never imports
the generator's validity predicate. The evaluator applies generator truth only
after prediction/selection.

## 3. Route R1 — early prefix completion plus slow-schema completion

**Rank: 1. Recommended minimal V1 implementation.**

### 3.1 Structure and exact flow

1. **[Inherited verified]** Build the V4-style `D,B,G` causal/action core from
   common real wake data and freeze it for all four cells.
2. **[Inherited verified boundary]** Insert at most 96 exact V2
   `EpisodicRecord`s into `M10/M11`; `M00/M01` contain zero queryable records.
3. **[New definition]** Call hard recall once. A prefix-only projector clamps all
   actually observed coordinates, fills only masked coordinates of the current
   12-by-8 history view, discards `episode_id` before model invocation, and
   deterministically extracts a fixed condition `q`. Rejection uses the
   unchanged schema fallback. The model never sees a stored episode suffix
   outside the registered history view.
4. **[Inherited verified boundary + new adapter]** In `M01/M11`, run exactly one
   V2 constrained recombination pass over identical real snapshots. Decode an
   accepted `DreamBinding` into hypothetical transition sufficient statistics,
   apply the inferred context/component/port/action/continuity graph, and update
   only the already-allocated missing `s(k)` slot. Unit weight and one pass are
   fixed; observed slots are immutable.
5. **[Inherited leak-free pattern]** For each of the `K` fixed action sequences,
   call

   ```python
   model.rollout(
       x_previous=completed_prefix[-2],
       x_anchor=completed_prefix[-1],
       prefix_condition=q,
       actions=candidate_actions[j],       # [20, m]
       slow_schema=cell.slow_schema,
       validity_graph=cell.inferred_graph,
   )
   ```

   and recursively feed only predicted states/residuals. H5 is a slice of this
   H20 array.
6. **[New integration]** Score every candidate with the frozen public task cost,
   choose one index, and only then call the evaluator.

The two components are orthogonal by construction: LTM changes `q` but no
parameter; dream changes missing `s(k)` slots but no episodic data or identity.
This deliberately predicts a small interaction, consistent with G7-M V2's
observed zero interaction, without assuming it must be zero.

### 3.2 Degrees of freedom and look-elsewhere

- Added learned continuous parameters over the common core: **0** if `q` is the
  preregistered fixed projection of the completed view. The `dK` schema capacity
  already exists in every cell.
- Episodic storage: at most `96 * 12 * 8 = 9,216` float64 values, exactly 73,728
  bytes, plus metadata; no learned LTM weights.
- Added route-specific tunable scalars: **0** beyond the three common
  train-derived G7-M calibration outputs. The threshold-selection rules are
  inherited; synthetic weight is 1, passes are 1, tie break is first index, and
  no memory/schema blend is tuned.
- New structural choices that must be frozen: codec layout, fixed projection
  defining `q`, and the additive schema-key definition (3 choices, not a sweep).
- Search disclosure: this report considered four route families. R1 is selected
  before train. Running all four on train and then selecting would violate the
  contract's pre-train implementation freeze; it would require a newly declared
  selection protocol or V2 with fresh registered seeds.

### 3.3 Fairness

- Identical base-model bytes, schema capacity, inferred graph, candidate order,
  H20 calls, and public cost in all cells.
- `M01/M11` use byte-identical wake snapshots and offline budgets. `M00/M10` use
  fixed no-op buffers of the same shapes; they do not inspect dream proposals.
- Empty-store calls in `M00/M01` and real-store calls in `M10/M11` occur once per
  query; no confidence-based extra compute or early stopping is allowed.
- Inferred-invalid candidates remain in the candidate set and receive one
  preregistered finite model-side penalty; all are still rolled for 20 steps.
- Required persistence, frozen causal rollout, schema-only fallback, oracle
  diagnostic, shuffled-binding, and unconstrained-recombination controls use the
  same request object and evaluation cases.

### 3.4 Cross-predictions

- The LTM benefit should increase with cue missingness and accepted recall
  confidence, be largest when an action-semantic/context condition is otherwise
  ambiguous, and approach zero for fully observed prefixes.
- The dream benefit should be localized to unseen-valid schema keys; it should
  be approximately zero on already observed bindings because overwrite is
  forbidden.
- Shuffling recalled episode binding while preserving the confidence and
  acceptance distribution should remove or reverse the LTM H20 benefit.
- Removing the recombination graph should weakly increase apparent coverage but
  increase cross-context/port/continuity violations and evaluator-invalid
  transitions.
- Because memory changes `q` and dream changes `s(k)`, the factorial interaction
  should be small relative to both main effects; a positive interaction is not
  assumed and is called synergy only under the frozen CI rule.

### 3.5 Likely failure modes and route-specific falsification

- V5--V7 show that a correct one-step mechanism and a better temporal scalar do
  not guarantee seed-robust H20 rollout. Early completion can be overwhelmed by
  recursive model error, so C1 may simply fail.
- A 12-by-8 template can match identity while carrying the wrong current action
  semantics. High-confidence accepted-wrong recall can then amplify H20 error.
- V2 join continuity was designed for known slots, not dynamic transitions. The
  additional inferred action/port/continuity check can reject most dream output,
  leaving too little C2 signal; accepting it too freely can violate the 0.01
  invalid-transition gate.
- The R1 memory mechanism is falsified if its paired benefit over the shuffled
  binding control has a non-positive lower CI, even if an unrelated M11/M00
  contrast happens to pass.
- The R1 dream mechanism is falsified if benefit appears mainly on observed
  bindings, persists when synthetic updates are disabled, or does not disappear
  when every accepted missing slot is replaced by the schema fallback.
- Any target/future poison changing predictions, any synthetic episode identity,
  any observed overwrite, or any candidate-set difference is an integrity
  failure, not a weak performance result.

## 4. Route R2 — recalled residual-state prior carried through the rollout

**Rank: 2. Stronger H20 mechanism, but more assumptions and inherited risk.**

### 4.1 Structure and exact flow

R2 keeps the same store and slow-schema pass, but memory acts after the causal
state update rather than only through completed prefix features.

1. Hard recall returns one real reconstruction. A pure action-aware residual
   decoder computes residuals of that previously observed history under the
   frozen base model; it never reads the current scored future.
2. From recalled residuals, fit a rank-one tuple
   `(center_mem, direction_mem, intercept_mem, score_mem)`. Independently fit the
   same tuple from the current observed prefix.
3. Sign-align directions using their dot product and combine them with a
   preregistered confidence-to-precision map. Rejection has memory precision
   zero. No episode ID is passed to the world model.
4. Dream bindings update only a context/action-keyed slow prior over the same
   residual tuple. Synthetic statistics cannot enter the real residual bank.
5. For every lead and action,

   \[
   \widehat z_{t+1}=c+\rho\widehat z_t,
   \qquad
   \widehat r_{t+1}=\bar r+u\widehat z_{t+1},
   \]

   and the resulting residual is added to the shared action-conditioned causal
   step. The combined residual state is frozen before candidate rollout and is
   identical across all candidate sequences except for registered
   action-conditioned schema terms.

Suggested API:

```python
prior = residual_prior_from_recall(
    recall.reconstruction, prefix.cue.cue_mask, frozen_model
)
filter_state = combine_prefix_and_memory_residuals(
    prefix_prior, prior, recall.confidence, frozen_precision_rule
)
paths = model.rollout(..., actions=actions_j, residual_state=filter_state)
```

### 4.2 Degrees of freedom and fairness

- Added persistent learned parameters: 0 if residual tuples are derived on
  demand from real reconstructions. Added per-query transient state is
  `2d + 2` continuous values (center, direction, intercept, score).
- At least **2 additional calibration choices** must be frozen: the
  confidence-to-memory precision map and whether `rho` is shared, retrieved, or
  combined. Rank is fixed to 1; considering ranks 1 and 2 would double the
  structural search.
- Slow residual-schema entries add up to `(2d+1)K` stored values unless they
  share the R1 `dK` table. This is materially more capacity than R1 and must be
  preallocated in every cell.
- Fairness otherwise matches R1. All cells execute the same rank-one algebra;
  no-LTM cells use zero memory precision, and no-dream cells use the same-shaped
  fallback prior. Candidate-dependent refitting is forbidden.

### 4.3 Cross-predictions and falsification

- Unlike R1, the LTM benefit should persist or increase from H5 to H20 because
  the retrieved state is recursively carried, not used only at the origin.
- Benefit should track alignment between recalled and current prefix residual
  directions. A sign-permuted or context-shuffled residual prior should be
  actively harmful.
- Rank-two latent or changed-AR stress worlds should degrade R2 more sharply
  than R1; this is a useful negative cross-prediction.
- Removing residual recursion while keeping the same completed prefix should
  remove most R2-specific H20 benefit. If it does not, the claimed path is
  falsified.
- R2 is falsified as a residual-memory explanation if H20 improvement is no
  larger than H5, residual alignment does not predict per-seed gain, or shuffled
  residual binding performs equivalently.

Likely failures are exactly those exposed by V7: better estimation of one
rank-one time state may not repair source-basis or action-interaction
misspecification. Recalled residuals also risk negative transfer across context,
and a confidence-derived precision creates more train-only selection freedom.

## 5. Route R3 — episodic residual graph with constrained transition stitching

**Rank: 3. Strongest structural validity story, but highest discrete tuning and
nearest-neighbor risk.**

### 5.1 Structure and exact flow

R3 retains the shared causal core but represents the slow correction as an
inferred transition graph rather than a dense schema table.

```python
@dataclass(frozen=True)
class TransitionKey:
    context: str
    input_port: str
    action_token: str
    output_port: str

@dataclass(frozen=True)
class SchemaEdge:
    key: TransitionKey
    anchor: np.ndarray
    residual: np.ndarray
    provenance: Provenance

def graph_step(
    graph: FrozenObservedGraph,
    causal_prediction: np.ndarray,
    action: np.ndarray,
    belief_node: int,
) -> tuple[np.ndarray, int, bool]: ...
```

Real wake transitions create real graph edges. Hard recall chooses one real
episode exemplar and initializes the graph belief from its prefix-only
reconstruction; it does not expose its future or copy its transition path.
At each predicted step, the graph selects the closest real/schema edge matching
the predicted public state, current action token, and inferred component. Its
residual corrects the causal prediction. A DreamBinding may add only a
synthetic/hypothetical missing `SchemaEdge`; it never creates a real edge or
node identity. Every candidate follows its own graph belief for all 20 steps.

The graph is inferred from wake data. Generator-level context, port, continuity,
and validity predicates remain evaluator-only. A graph miss is handled by the
same schema fallback, not by querying truth or pruning an action.

### 5.2 Degrees of freedom and fairness

- Stored continuous values are `O(Fd)` for `F` observed/synthetic edges, in
  addition to the common core. The exact edge and node caps must be identical in
  all cells.
- At least **4 adjustable choices** are unavoidable: state-distance
  standardization/bandwidth, neighbor count or deterministic tie rule,
  residual-mix weight, and continuity/projector tolerance. An inferred-invalid
  planning penalty is a fifth choice unless inherited globally.
- Alternative metrics, `k`, and projectors create a large look-elsewhere family;
  all must be fixed before train. Nearest-neighbor selection must never use
  future error.
- All cells traverse a graph with the same caps and 20-step call count. Empty
  slots are explicit fallbacks. No route may reduce compute by terminating an
  inferred-invalid candidate early.

### 5.3 Cross-predictions and falsification

- R3 should have the lowest invalid-transition rate and its advantage should be
  concentrated at port/action composition boundaries, not within well-covered
  smooth segments.
- Per-seed gain should increase with wake-graph coverage and decrease with
  nearest-edge distance. Outside observed/inferred graph support it should
  revert to the causal baseline.
- The unconstrained lesion may improve raw coverage or NRMSE but must increase
  invalid/cross-context transitions; otherwise the graph constraint is not doing
  the claimed work.
- Shuffled episodic binding should mainly damage the first few graph choices;
  dream should restore reachability for unseen-valid keys without restoring an
  episode identity.
- The route explanation is falsified if graph coverage/distance does not predict
  benefit, constrained and unconstrained lesions have the same invalid rate, or
  synthetic edges can be returned as recalled identities.

Likely failures are discontinuous nearest-neighbor rollouts, state-bin explosion,
false confidence outside graph support, and implicit target imitation if a
developer tunes distance or `k` against H20 outcomes. It is also harder to show
equal effective capacity than R1.

## 6. Route R4 — memory-conditioned posterior over causal parameters

**Rank: 4. Most expressive; unsuitable as the minimal V1.**

### 6.1 Structure and exact flow

Let `theta` contain the shared `D,B,G,rho` and slow-schema parameters. Real wake
data fit one frozen diagonal base posterior. For a recalled real reconstruction,
a prefix-only sufficient-statistic function constructs an episode-conditioned
likelihood. Dream bindings contribute bounded hypothetical likelihood terms to
the slow prior only; they never create episodic likelihoods. A fixed Bayesian
update produces one query posterior, which is held constant across all action
candidates.

```python
base = fit_shared_posterior(real_wake_only)
episode_stats = stats_from_recalled_prefix(recall.reconstruction, frozen_codec)
schema_stats = cell.slow_schema.hypothetical_stats()
posterior = condition_world_model(base, episode_stats, schema_stats)
paths = rollout_common_quadrature(
    posterior, request.prefix, request.candidate_actions, common_draws
)
```

All candidates use the same fixed quadrature points/common random numbers.
Planning uses expected public cost, with a risk coefficient only if it is frozen
before train. The evaluator is still called only after selection.

### 6.2 Degrees of freedom and fairness

- For `p=p_core+dK`, a diagonal posterior stores `2p` mean/precision values; a
  full covariance would add `p(p+1)/2` and is excluded from a CPU-minimal route.
- Each real episode may add up to `2p` sufficient-statistic values unless they
  are derived on demand. This greatly exceeds the fixed V2 trace budget.
- At least **4 tuning choices** arise: base prior precision, synthetic
  likelihood weight, recall-confidence-to-precision map, and planning risk
  weight or quadrature size. Posterior family and covariance structure add
  discrete look-elsewhere choices.
- Equal budget requires a fixed posterior update and identical quadrature count
  for every cell/action, with zero-precision placeholders in disabled factors.

### 6.3 Cross-predictions and falsification

- R4 uniquely predicts improved predictive calibration as well as mean NRMSE;
  posterior variance should track per-seed error and action regret.
- Its largest planning gain should occur when candidate mean costs are close but
  uncertainty differs. Removing the risk/uncertainty term should remove that
  extra regret gain.
- LTM and dream precision can interact, so a nonzero factorial interaction is
  plausible; antagonism is also a serious risk through synthetic overconfidence.
- Shuffled binding should broaden or miscenter the posterior and worsen both
  calibration and regret. If it does not, the memory-conditioned posterior
  explanation is falsified.
- The route is falsified if uncertainty is uncalibrated, regret improvement
  survives uncertainty ablation, or synthetic updates reduce variance while
  increasing invalid transitions/accepted-wrong recall.

The principal failure modes are posterior collapse, synthetic overconfidence,
excess capacity, CPU-budget mismatch, and tuning to the planning target. These
costs are not justified for a first integration gate.

## 7. Ranked comparison

| Rank | Route | Integration topology | Added adjustable choices | Best cross-prediction | Main inherited risk | Decision |
|---:|---|---|---:|---|---|---|
| 1 | R1 | recall completes one prefix condition; dream fills preallocated slow-schema slots; one causal rollout | 0 scalar, 3 frozen structural definitions | LTM effect follows cue missingness; dream effect is unseen-key-local; small interaction | V5-style recursive error can erase the early advantage | **Implement as V1** |
| 2 | R2 | recall initializes a persistent rank-one residual state; dream updates residual prior | at least 2 | H20 benefit persists/grows relative to H5 and follows residual alignment | V7 showed accurate rank-one temporal state need not improve H20 | Reserve for V2 after a clean R1 result/failure with fresh seeds |
| 3 | R3 | recall initializes graph belief; real/synthetic residual edges constrain every step | at least 4--5 | lowest invalid rate; gain follows graph coverage and boundary transitions | nearest-neighbor discontinuity and large look-elsewhere space | Useful lesion/research branch, not minimal V1 |
| 4 | R4 | recall and dream update a parameter posterior used for expected-cost planning | at least 4 plus posterior family | uncertainty calibration predicts regret gain | overcapacity, overconfidence, compute and selection freedom | Reject for V1 |

The ranking prioritizes fewer degrees of freedom, exact provenance reuse, a
candidate API whose leak-freedom is easy to prove, and strong differential
lesions. It does **not** rank routes by unobserved expected score.

## 8. Minimal implementation recommendation

Freeze and implement **R1 only** for G9-CBM V1:

1. Reuse the G7-M V2 provenance classes/store/recall/dream/update functions by
   import and hash-lock them. Add only a deterministic observed-history codec
   and prefix-only exposure projector.
2. Extend the V4-style causal mean with an observed-action term and one
   preallocated slow-schema residual table. Fit the common core once from real
   wake/calibration data and share its bytes across all cells.
3. Use a V5-shaped action rollout API that accepts only two observed anchor
   states, one frozen prefix condition, one `[20,m]` action array, and learner
   schema/graph objects. Do not include V6 consensus or the V7 AR replacement in
   the active candidate.
4. Build the four cells by changing only store availability and the one offline
   synthetic-schema pass. `M01`'s wake workspace is destroyed before evaluation.
5. Roll every fixed action sequence, select with a deterministic public cost,
   and move generator execution/scoring behind an evaluator-only handle.
6. Include the required controls with matched inputs and budgets. The shuffled
   binding must shuffle reconstruction-to-query association without changing
   acceptance counts; the unconstrained lesion must preserve proposal/update
   count while removing only inferred graph checks.
7. Before any registered train/calibration seed, freeze the raw LF contract,
   codec/mask semantics, action/token map, inferred-graph construction, model,
   runner, unit/integrity tests, inherited-module hashes, and the V2 boundary
   equivalence proof.

R1 is intentionally capable of failing. In particular, G9-CB V5--V7 make a
failure of the H20 robustness gates plausible. Such a failure would be useful:
it would separate "safe exemplar/schema coexistence" from "memory improves
autonomous planning," preserve the artifact, leave test unopened, and force any
R2/R3/R4 follow-up into a new version with fresh seeds.

## 9. Universal stop/falsification conditions

Stop before implementation if the learner-facing generator cannot produce a
prefix packet without scored futures/latents, if the fixed action set differs by
cell, if compute cannot be equalized, or if the V2 provenance boundary cannot be
reused or proven equivalent.

Stop before validation on any nonzero future/evaluator read, synthetic-to-LTM
insert, synthetic `recalled` tag, observed overwrite, cross-context splice,
candidate-set difference, nonfinite output, unlocked test access, or unresolved
P0/P1 defect.

At validation, the integrated performance claim is falsified if any frozen
all-of gate fails, including H20 M11/M00 above 0.90, either matched dream
improvement below 10%, planning-regret M11/M00 above 0.80, success gain below
0.10, invalid predicted transitions above 0.01, accepted-wrong or lure false
recall above 0.05, synthetic-to-LTM inserts above zero, or either no-antagonism
upper-CI check above 2%. Preserve the failed artifact and do not open test.
