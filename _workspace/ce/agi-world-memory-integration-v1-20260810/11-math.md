# Mathematical verification — G9-CBM V1

Status: COMPLETE

Contract disposition: BLOCKED before implementation/validation.

The audit is complete, but the frozen contract is blocked before implementation under
its own stop conditions. The present text does not define a unique experiment:
the C1 estimand and its gate disagree, the rollout and planning metrics have no
mathematical definition, the candidate/evaluator boundary is not executable, the
provenance rate gate contradicts the provenance definition, and “same compute budget”
has no ledger. No registered seed was run.

## 1. Target, scope, and inspected evidence

The target is 00-contract.md, SHA-256
61083634d45b1dc60b610612827cf044c0bac4be0e5e7df426fa5c2a39ca4107.
The audit treats the seed as the independent sampling unit and the four cells as
within-seed software interventions. It does not treat numerical agreement as a proof
of a memory, dream, or planning mechanism.

Relevant inherited material inspected:

| artifact | SHA-256 | relevant fact |
|---|---|---|
| episodic_ltm_dream_factorial_v2.json | 973e90111ee98862a5c9ffc3f86509b46ee4e263b5a977e7e1504e00109092b9 | Defines the inherited 2×2 signs, train-standardized pooled hidden error, strict thresholding, and provenance rules. |
| episodic_ltm_dream_bridge_v2.py | edace8bb0ff63eb04bf87518a2f69c4f2960d5a200fec2ac9f1b2bf2a89fa6b1 | Implements fixed-96 real-only hard exemplar recall; this is not a recurrent attractor or a rollout model. |
| sparse_causal_bridge_v7.json | bef9a01be7a17002a6bee9371759963785b3852c2156922b1072187fd2d987f8 | Registers one-origin H5/H20 free rollout, but V7 failed validation. |
| invariant_prior_rollout_bridge.py | f7a64266ad167b58eecb5d3b27907f3c4f30b66fa660e50f52b88b27c72fc90f | Uses prefix-only prediction and paired baseline-minus-candidate error signs. |

Inherited facts that may be reused without relabelling:

- G7-M V2’s “NRMSE” is exactly pooled RMSE in coordinates standardized by a
  train-only mean and population standard deviation. It has no second
  test-target denominator.
- Its provenance tuple is source, epistemic status, observed flag, recalled flag.
  Synthetic output is synthetic/hypothetical/false/false and has no episode ID.
- G9-CB V5–V7 use baseline error minus candidate error for a positive benefit.
  H5 is required to be the first five rows of the same H20 rollout.
- Both inherited runners contain numeric read counters set to zero. Their separate
  API-signature and future-poisoning tests provide the meaningful leakage evidence;
  a self-reported zero counter alone does not.

## 2. Blocking findings

| ID | severity | finding | consequence |
|---|---|---|---|
| P0-1 | P0 | C1 is a marginal LTM claim, but its H20 gate is M11 versus M00, which changes LTM and dream together. | C1 can be false while every stated H20 comparison passes. |
| P0-2 | P0 | State/action spaces, transition equation, wake/evaluation chronology, candidate action construction, public goal, counterfactual cost, success, and invalid-lure cost are undefined. | Prediction, regret, success, and “counterfactual” are not reproducible or identifiable. |
| P0-3 | P0 | “NRMSE,” unseen-valid H20 error, H20/H5 ratio, regret, and the denominators of invalid-transition rates are undefined. | Gates can change under equally plausible implementations and dimensional consistency cannot be audited. |
| P0-4 | P0 | The contract forbids leakage in prose but does not freeze a candidate API, chronology, poison tests, cell isolation, or an evaluator-after-return order. | A target read can coexist with a hard-coded read counter of zero. |
| P0-5 | P0 | Synthetic output is definitionally unavailable as recalled identity, yet a recalled-synthetic rate up to 0.01 is allowed. “Synthetic-to-LTM inserts” also does not say attempts or successful writes. | One semantic violation per 100 items can pass a purported hard boundary. |
| P0-6 | P0 | Internal LTM, dream join, rollout blend, planner, success, and invalid-penalty thresholds have no train-only selection rule. “Both absolute component gates” is referenced but no two component gates are defined. | Model selection and the meaning of C3 remain free after seeing data. |
| P0-7 | P0 | The four treatment cells cannot literally have the same resources as written: M11 has a store and offline update while M00 has neither. No operation/capacity ledger or inactive-cell padding is specified. | Component effects are confounded with unbounded extra storage, updates, and calls. |
| P0-8 | P0 | The sign of an error interaction, the aggregation of relative gates, zero-denominator behavior, strict ties, and whether “2%” is relative or 0.02 error units are not frozen. | Opposite synergy decisions and opposite PASS/FAIL decisions are possible. |

Therefore validation must not start, and implementation should not start until a
revised preregistration resolves P0-1 through P0-8.

## 3. Exact factorial estimands and paired signs

Let c = ld denote cell Mld, with l,d in {0,1}. Let E(s,c) be a
lower-is-better seed-level error, Q(s,c) a higher-is-better seed-level quantity,
R(s,c) planning regret, and S(s,c) binary task success after aggregating every
origin/query within seed s.

For a lower-is-better error, define positive benefit as:

~~~
B_L^E(s) = 0.5 * [(E(s,00) - E(s,10)) + (E(s,01) - E(s,11))]
B_D^E(s) = 0.5 * [(E(s,00) - E(s,01)) + (E(s,10) - E(s,11))]
I^E(s)   = E(s,10) + E(s,01) - E(s,00) - E(s,11)
~~~

For a higher-is-better metric:

~~~
B_L^Q(s) = 0.5 * [(Q(s,10) - Q(s,00)) + (Q(s,11) - Q(s,01))]
B_D^Q(s) = 0.5 * [(Q(s,01) - Q(s,00)) + (Q(s,11) - Q(s,10))]
I^Q(s)   = Q(s,11) - Q(s,10) - Q(s,01) + Q(s,00)
~~~

Positive I is benefit-oriented super-additivity on the named metric’s frozen
scale. Interactions of NRMSE, regret, and success are different estimands; a positive
interaction on one does not license a generic “synergy” claim.

For seed values v(s), use:

~~~
mean(v) = sum_s v(s) / n
CI95(v) = mean(v) ± t[0.975,n-1] * sd(v,ddof=1) / sqrt(n)
~~~

The exact registered critical values should be 2.022690911734728 for n=40 and
2.0009953780882674 for n=60. A wrong seed count, any omitted seed, or any nonfinite
value is a hard failure. A strict win is v(s) > 0; v(s)=0 is a tie and not a win.
The confidence-bound comparisons in the current contract say “strictly positive,”
so the implementation must compare CI lower > 0, not greater-than-or-equal to zero.

### Exact replacement for C1 and the H20 gate

Use E_all(s,c,20), defined in section 4. Replace the current use of M11 versus M00
as evidence for C1 with:

~~~
RR_L = 1 - [mean(E_all(10,20)) + mean(E_all(11,20))]
             / [mean(E_all(00,20)) + mean(E_all(01,20))]

C1 passes iff:
  RR_L >= 0.10
  lower95(B_L^E_all) > 0
  mean(1[B_L^E_all(s) > 0]) >= 0.65
~~~

Here the cell labels inside mean abbreviate the seed arrays. If the denominator is
not finite and strictly positive, the gate is hard invalid, not a pass and not zero.
Report both simple LTM effects E00-E10 and E01-E11. If the intended claim is
“LTM helps at both dream levels,” require the same ratio, CI, and win gates for
each simple effect separately; a marginal gate alone supports only an averaged
factorial effect.

The existing joint comparison may be retained, but it must be renamed:

~~~
RR_joint = 1 - mean(E_all(11,20)) / mean(E_all(00,20))
G_joint passes iff RR_joint >= 0.10,
lower95(E_all(00,20)-E_all(11,20)) > 0,
and strict win fraction >= 0.65.
~~~

It is an integration contrast, not an LTM marginal effect.

### Exact replacement for C2

For E_uv(s,c,20), require both matched dream effects:

~~~
RR_D|L=0 = 1 - mean(E_uv(01,20)) / mean(E_uv(00,20))
RR_D|L=1 = 1 - mean(E_uv(11,20)) / mean(E_uv(10,20))

For each pair independently:
  relative reduction >= 0.10
  lower95(E_control - E_dream) > 0
  mean(1[E_control(s) - E_dream(s) > 0]) >= 0.65
~~~

Each control mean must be finite and strictly positive. Also report the marginal
dream effect B_D and its interval.

### Exact replacement for planning

Once regret and success are defined as in section 5:

~~~
RR_regret = 1 - mean(R(11)) / mean(R(00))
B_R(s)    = R(s,00) - R(s,11)
B_S(s)    = S(s,11) - S(s,00)

C3 passes iff:
  mean(R(00)) is finite and > 0
  RR_regret >= 0.20
  lower95(B_R) > 0
  mean(B_S) >= 0.10
  lower95(B_S) > 0
~~~

If mean R(00)=0, relative improvement is undefined and C3 fails. Substituting an
epsilon after outcomes are known is forbidden.

### Exact replacement for no antagonism

The phrase “2%” should mean relative, not 0.02 standardized-error units. Define:

~~~
A_recall(s) = E_recall(s,11) - 1.02 * E_recall(s,10)
A_uv(s)     = E_uv(s,11,20) - 1.02 * E_uv(s,01,20)

No-antagonism passes iff:
  upper95(A_recall) <= 0
  upper95(A_uv) <= 0
~~~

This paired formulation remains defined when a reference error is zero. If the
authors instead intend an absolute margin, the prose must say “0.02
train-standardized error units” and the formula must be
upper95(E11-Ereference) <= 0.02.

## 4. State error and dimensionless audit

The present contract has no units or normalizer, so its NRMSE is not currently
auditable. The least ambiguous reuse of G7-M is a train-standardized RMSE.

For state coordinate j, fit only on the completed real-wake train ledger:

~~~
mu_j    = train population mean
sigma_j = train population standard deviation with ddof=0
q_j     = max(sigma_j, epsilon * r_j)
z_j(x)  = (x_j - mu_j) / q_j
~~~

r_j is a preregistered reference with the same unit as x_j; if the synthetic state
is declared dimensionless, use r_j=1 and epsilon=1e-8. The arrays mu and q are
written once and reused byte-identically in every cell and split. Validation/test
future ranges, variances, and norms may not be normalizers.

For one registered origin and K_eval fixed valid evaluation action sequences:

~~~
E_all(s,c,H) =
  sqrt( sum[k,t=1..H,j] (z_j(xhat[s,c,k,t])-z_j(xstar[s,k,t]))^2
        / (K_eval * H * p) )
~~~

The origin is not included. H5 must be the exact first five rows of the H20 call,
not a separately tuned or separately randomized rollout.

Let U_s be the evaluator-only set of scalar coordinates belonging to generator-valid
transitions whose registered composition was absent from every candidate-visible
real-wake record. U_s must be constructed before cell execution, have the same
nonzero cardinality in every cell, and never be passed to the candidate:

~~~
E_uv(s,c,H) =
  sqrt( sum[i in U_s, lead(i)<=H] (zhat_i-zstar_i)^2
        / count(i in U_s, lead(i)<=H) )
~~~

Every fixed query/transition is scored. Abstention, invalid output, or missing schema
coverage may not silently remove a hard case from either denominator. Nonfinite
prediction is a hard failure and must still increment a failure count.

The H20/H5 stability gate should be frozen as:

~~~
mean(E_all(s,11,20)) / mean(E_all(s,11,5)) <= 2.0
~~~

with a finite, strictly positive denominator. This is a growth gate, not an absolute
accuracy gate.

Dimension table:

| quantity | dimension | audit result after the replacement |
|---|---:|---|
| x_j, q_j, r_j | coordinate-specific state unit | q_j and r_j match x_j |
| z_j | 1 | dimensionless |
| E_all, E_uv, recall standardized RMSE, join RMS | 1 | dimensionless |
| masked cosine, confidence, LTM threshold | 1 | dimensionless |
| normalized action u=a/a_ref | 1 | dimensionless |
| cost J, regret R, all CIs and margins | 1 | dimensionless only if section 5 is adopted |
| rates, success, win fractions, relative reductions | 1 | dimensionless |
| H20/H5 and cell error ratios | 1 | dimensionless |

No executable integrated expression exists yet, so the repository dimensionless
checker cannot verify this contract. No checker or seed runner was executed in this
read-only lane. Dimensional consistency would not by itself establish physical,
biological, or planning validity.

## 5. Minimum mathematical definition of planning

The contract must freeze p-dimensional state, m-dimensional action, horizon H=20,
the number and canonical order of action sequences K, action bounds, public goal
g, state/action reference scales, cost weights, invalid penalty, and success
threshold. Without these, delete C3 and every use of “counterfactual planning.”

One viable dimensionless definition is:

~~~
u_t = a_t / a_ref

J_valid(s,k) =
  (1/(H*p)) * sum[t,j] wx_j * (z_j(xstar[s,k,t])-g_j[t])^2
  + (lambda/(H*m)) * sum[t,h] wa_h * u[k,t,h]^2

J(s,k) = J_valid(s,k) if sequence k is generator-valid,
         P_invalid     otherwise.
~~~

All weights, lambda, and P_invalid are fixed dimensionless numbers. The goal and
cost function are public candidate inputs; evaluator counterfactual states,
validity labels, and realized costs are not. The generator must guarantee at least
one valid successful candidate, at least one valid nonsuccessful candidate, and the
registered number of invalid lures per seed. P_invalid must exceed the registered
maximum valid-sequence cost by construction, not by inspecting validation/test.

The candidate computes the same cost form using only its rollout and its inferred
constraint graph:

~~~
khat(s,c) = argmin_k [Jhat(s,c,k), canonical_action_index(k)]
kopt(s)   = argmin_k [J(s,k),      canonical_action_index(k)]
R(s,c)    = J(s,khat(s,c)) - J(s,kopt(s))
S(s,c)    = 1[J(s,khat(s,c)) <= tau_success
               and selected sequence is generator-valid]
~~~

Argmin ties use the lowest canonical action index. Candidate ordering must not encode
oracle rank; permuting the presented order and mapping back must preserve selection
except for exact registered ties. Assert R>=0 up to one preregistered numerical
tolerance; a more negative value is a hard evaluator error.

Invalidity needs three separate denominators:

1. accepted synthetic splice violation count divided by accepted proposals;
2. predicted state/port/context/continuity violation count divided by
   H times the number of generator-valid candidate rollouts;
3. invalid selected-action count divided by the number of seeds.

Invalid action lures cannot be included in denominator 2, because their action
constraint is invalid by construction. They are tested by denominator 3 and the
finite penalty in J. The current single phrase “invalid predicted-transition rate”
does not distinguish these quantities.

For a genuine counterfactual claim, actions must enter a registered structural
transition x[t+1]=F(x[t],do(a[t]),context,noise[t+1]). Wake action assignment must
be exogenous or its policy fully exposed to the learner. Evaluator counterfactuals
must use the same registered exogenous-noise coupling across action candidates and
all four cells. Otherwise the experiment is action-conditioned prediction, not an
identified counterfactual planning test.

## 6. Leakage, chronology, and cell isolation

### Exact replacement candidate boundary

For each seed, create an append-only real-wake ledger W_s whose every episode ends
strictly before the evaluation origin. The evaluation episode ID and every future
row after the origin are disjoint from W_s. Dream input is an immutable snapshot
of W_s only. The current evaluation prefix may be observed through the origin but
its suffix may not be inserted before scoring.

The non-oracle candidate API may receive only:

~~~
observed prefix through origin
proposed length-20 action sequence or fixed candidate set
public task goal and public cost specification
read-only frozen core-model parameters
read-only real-wake LTM when l=1
read-only slow schema, including hypothetical entries, when d=1
opaque observed context/cue tokens
~~~

It may not receive a World/Episode object, master seed, RNG stream ID, held-out
future, target episode ID, hidden/evaluator latent, generator validity graph or
labels, counterfactual outcome/cost, oracle rank, or split result. Evaluation begins
only after all candidate outputs and selections are returned and hashed.

Required pre-registration tests:

- mutate every held-out future, reward/cost outcome, evaluator latent, and validity
  label while holding candidate inputs fixed; predictions, provenance, and selected
  action must be bit-identical;
- poison cue values outside the observed mask; recall and rollout must be
  bit-identical;
- change seed/stream metadata while holding actual candidate inputs fixed; output
  must be identical;
- verify H5 equals the H20 prefix bit-for-bit;
- evaluate the four cells in every order, or at minimum all 24 permutations in an
  off-range harness; each cell result must be bit-identical;
- verify worlds, prefixes, action arrays, masks, and real-wake ledger hashes are
  identical across cells;
- deny test-path reads before a passing validation lock.

Access counters must be capability-instrumented or independently tainted. A literal
result field equal to zero is not evidence of nonaccess.

### Generator-side leakage requirements

- Separate RNG streams for mechanisms, wake observations, evaluation noise,
  action candidates, candidate-order permutation, cue masks/noise, and lures.
- Candidate code never receives the master seed or stream identifiers.
- Prefix/suffix/context tokens are opaque and independently permuted; they may not
  encode port, validity, oracle action, or future identity.
- The inferred dream graph is a pure function of W_s. Generator truth is used only
  after candidate return.
- “Unseen-valid” membership is defined against W_s, not against a cell-dependent
  store or a model’s own output.

Until these tests and chronology are in the frozen contract, C4 is not identifiable.

## 7. False-memory and provenance gates

Reuse the G7-M V2 semantic tuples exactly:

| object | source | epistemic_status | observed | recalled | episode_id |
|---|---|---|---:|---:|---|
| real wake record in LTM | real | observed | true | false | unique real ledger ID |
| accepted recall output | real | recalled | false | true | ID present in LTM |
| synthetic schema output | synthetic | hypothetical | false | false | null |
| schema fallback | schema_fallback | inferred | false | false | null |

Hard invariants, all as integer counts, must equal zero in every seed and cell:

~~~
synthetic_with_episode_id
synthetic_tagged_observed
synthetic_tagged_recalled
synthetic_to_ltm_insert_attempts
synthetic_to_ltm_successful_inserts
nonledger_real_record_in_ltm
observed_record_overwrite_or_hash_change
cross_context_or_cross_component_accepted_splice
accepted_port_continuity_or_action_constraint_violation
~~~

The current synthetic-tagged-recalled <=0.01 gate must be replaced by count=0.
An invariant is not a 99% performance target. “Synthetic-to-LTM inserts” must report
both attempts and successes; the inherited G7-M field named
synthetic_to_ltm_insert_count actually counts rejected attempts.

For positive recall set P_s and unstored-lure set L_s, report:

~~~
coverage(s)           = accepted positives / |P_s|
identity_accuracy(s)  = accepted correct identities / |P_s|
wrong_all(s)          = accepted wrong identities / |P_s|
wrong_given_accept(s) = accepted wrong identities / max(accepted positives,1)
false_lure(s)         = accepted unstored lures / |L_s|
~~~

At minimum, freeze |P_s|, |L_s|, cue masks/noise, and require coverage and identity
accuracy >=0.80 in M10 and M11, both wrong rates <=0.05, and mean false_lure <=0.05.
A safer P1 gate is upper95(false_lure) <=0.05 as well. Coverage is necessary because
an always-abstain store otherwise achieves perfect false-memory safety.

The question’s phrase “without increasing false recall” is mathematically
incompatible with comparison to M00/M01, whose no-store false recall is structurally
zero, while the gate permits 0.05. Replace it with “without exceeding the
preregistered false-recall ceiling.” If literal nonincrease is intended, the
required gate is false_lure=0 exactly.

The inherited hard recall first checks that the cue endpoints share some inferred
component, but then computes its winner over every record in the store. It does not
filter candidate traces to that context/component. Reusing its provenance boundary
is sound; assuming it enforces context-local retrieval is not. Either add
accepted-cross-context-recall count=0 or create a new version that filters candidates
and obtain fresh equivalence/remediation evidence.

## 8. Thresholds that must be frozen

Every threshold or blend coefficient that can change output must be either a literal
in the raw preregistration or the result of an exact train-only algorithm. One
defensible reuse of G7-M V2 is:

1. Fit coordinate means/scales on the 40 train seeds only with ddof=0 and the
   registered scale floor.
2. For each registered LTM bank size/mechanism, form candidates from sorted unique
   train initial masked-cosine confidences plus positive infinity.
3. Accept iff confidence > tau. Among thresholds with pooled unstored-lure false
   recall <=0.025 and wrong_all <=0.025, lexicographically maximize correct identity
   accuracy with abstentions wrong, minimize lure false recall, then choose the
   largest tau. Use the same frozen tau in M10 and M11.
4. Define dream join RMS in train-standardized coordinates. Set tau_join to
   numpy.quantile of registered real adjacent-fragment discontinuities at q=0.99
   with method=linear. Accept only when both joins are <= tau_join and the inferred
   graph permits the splice. Use the same tau_join in M01 and M11.
5. Freeze any rollout memory blend, schema-update weight, invalid penalty,
   success threshold, and planner tie rule by a finite, preregistered train-only
   grid/objective. No cell-specific tuning is allowed except the factor being
   disabled.

The calibration artifact must include raw registration SHA, implementation/tests/
runner/inherited-module SHAs, every numeric array and threshold, the exact selection
pool and tie result, and its own byte hash. Validation/test recalibration is forbidden.
If the integration mechanism needs no threshold, the contract should say so
explicitly rather than leave a free parameter.

“Both absolute component gates” currently has no referent. Either define:

- LTM absolute gate: coverage, identity, wrong_all, wrong_given_accept, lure false,
  and real-only provenance; and
- dream absolute gate: fixed valid-binding coverage, fixed accepted-proposal budget,
  zero constraint violations, hard-zero provenance violations, and a preregistered
  absolute standardized-error cap;

or delete that phrase from C3. Thresholds from the old known-slot problem cannot be
silently transferred to a new rollout scale.

## 9. Common-budget ledger

Wall time is not a scientific equality because scheduling and cache effects differ.
Freeze integer capacities and call/update counts. For every seed and cell, serialize:

| ledger field | equality rule |
|---|---|
| real wake episodes presented to the core; bytes; canonical hashes | identical |
| core optimizer/update steps, minibatch order, parameters, and final core SHA | identical |
| H20 calls, H5 calls, action candidates K, predicted transitions K×20, planner score calls | identical |
| LTM capacity and lookup-call slots | identical allocation/call count; inactive cells use an unqueryable inert buffer and discard results |
| offline proposal slots, graph checks, update slots, and passes | identical allocation/call count; inactive cells use a disposable shadow schema and discard writes |
| maximum persistent bytes and temporary workspace bytes | identical preallocation/cap |
| RNG draw arrays used by the core/evaluator | pregenerated once and identical |
| queryable real episodic records | treatment variable: 0 for l=0 and the fixed registered count for l=1 |
| accepted hypothetical schema updates | treatment variable: 0 for d=0 and at most the fixed registered cap for d=1 |

Define a numeric budget vector before implementation:

~~~
b(c,s) = (
  N_wake, U_core, P_core, K, H, N_rollout, N_score,
  B_ltm_capacity, Q_ltm_slots,
  B_dream_proposals, U_dream_slots, N_dream_passes,
  persistent_byte_cap, workspace_byte_cap
)
~~~

Require b(c,s)=b_registered for all c,s, plus equal input/core hashes. The inherited
defaults B_ltm_capacity=96 and one offline pass may be used only if the new generator
actually preserves that design; K, proposal cap, and every other entry still need
numeric values. Any over-budget or unequal count is a hard resource failure.

If inactive-cell no-op padding is rejected, replace “same registered compute budget”
with “same core training and rollout budget; LTM/dream overhead is part of the
treatment and is reported.” In that weaker design, the estimand is the whole
component-plus-resource bundle, not a memory-content effect.

## 10. Identifiability and required controls

After the P0 fixes, the factorial effects identify conditional software-treatment
contrasts only if all cells are isolated copies with identical input/core hashes and
the factor toggles are the only differences. Seed CIs then generalize over the
registered evaluation-seed generator conditional on one frozen train/calibration
artifact. They do not include uncertainty over a new training sample or a different
generator family.

Current claim audit:

| claim | current status | reason |
|---|---|---|
| C1 marginal LTM reduction | not identifiable | joint M11-M00 gate is not the marginal contrast |
| C2 dream reduction on unseen-valid H20 | not defined | unseen-valid set and error denominator are missing |
| C3 counterfactual planning | not defined | no do-action generator, cost, candidate construction, regret, or success |
| C4 no leakage/contamination mechanism | not identifiable | semantic invariants and counters exist only in prose; API/poison gates absent |
| C5 metric-specific interactions | calculable only after fixes | sign and metric list are missing |

The required controls are named but have no decision role. For content-specific
attribution, preregister:

~~~
B_binding(s) = E_shuffled_binding(s) - E_real_binding(s)
require lower95(B_binding) > 0 with identical capacity/calls.

B_constraint(s) = invalid_rate_unconstrained(s)
                  - invalid_rate_constrained(s)
require lower95(B_constraint) > 0 with identical proposal/update caps,
and constrained hard violations = 0.
~~~

The shuffled control must permute episode-to-content bindings within registered
context/length strata while preserving storage order, capacity, score marginals,
and call count. The unconstrained lesion must use the same fragments and proposal
budget and differ only by removal of the graph/join constraint. If these remain
diagnostic, the allowed conclusion must be about enabling a software bundle, not
about memory content or constraint mechanism.

Persistence, frozen G9-CB rollout, and schema-only fallback also need frozen paired
contrasts or must be labelled diagnostic. The current H20/H5 <=2 gate is not an
absolute accuracy gate: E5=100 and E20=100 passes. If “world-model adequacy” is to
be claimed, add a preregistered absolute E20 cap and/or paired superiority to
persistence/frozen G9-CB. A relative improvement from a poor baseline supports only
“lower error,” not “accurate world model.”

Unadjusted intervals are acceptable for the single all-of intersection claim, but
choosing whichever of many interactions has a positive lower bound does not control
family-wise error. Preselect the metric attached to any synergy claim or adjust the
interaction family. Report every interaction regardless of sign.

## 11. Counterexamples and boundary calculations

### C1 can fail while the current H20 gates pass

Take deterministic seed-level H20 errors:

~~~
E00=1.00, E10=1.00, E01=0.80, E11=0.80.
~~~

M11 <= 0.90*M00 passes, and each dream cell improves its matched no-dream cell by
20%. But:

~~~
B_L = 0.5*[(1-1)+(0.8-0.8)] = 0.
~~~

There is no LTM effect. Thus the current gate cannot establish C1.

### Error interaction sign is otherwise reversed

Let E00=1.0, E10=0.8, E01=0.8, E11=0.5. The joint reduction 0.5 exceeds the
sum of the two baseline simple reductions 0.4 by 0.1. Benefit-oriented interaction:

~~~
I^E = 0.8+0.8-1.0-0.5 = +0.1.
~~~

The raw error interaction E11-E10-E01+E00 is -0.1. Requiring a positive lower
bound without freezing orientation can produce the opposite synergy verdict.

### “At least 10%” has two inequivalent aggregations

For two seeds, baseline errors [1,100] and treatment errors [0.5,91]:

~~~
ratio of means = 91.5/101 = 0.905940594...  -> fails a 0.90 ratio
mean seed ratio = (0.5/1 + 91/100)/2 = 0.705 -> passes
~~~

The replacement uses a ratio of seed-level metric means and paired absolute
differences for the CI; it never averages seed ratios.

### Provenance rate contradicts the definition

One synthetic item tagged recalled among 100 gives rate 0.01 and passes <=0.01,
while directly violating “unavailable as an episode identity.” The correct gate is
an integer violation count equal to zero.

### “2%” cannot mean an absolute 0.02 by accident

If recall error changes from E10=0.004 to E11=0.020, the absolute increase 0.016
is below 0.02, but the relative degradation is 400%. The paired quantity
E11-1.02*E10 correctly fails.

### Accepted-wrong denominator matters

With 100 positives, one accepted query, and that identity wrong:

~~~
wrong_all = 1/100 = 0.01
wrong_given_accept = 1/1 = 1.00.
~~~

Both must be reported, and a positive coverage gate is necessary.

### “No false-recall increase” is stronger than the registered ceiling

M00 has no queryable store and hence false recall 0 by construction. M10 false
recall 0.04 satisfies the proposed 0.05 ceiling but is an increase. The wording
must be changed or the gate tightened to exact zero.

## 12. P1 and P2 findings

P1:

1. Give shuffled-binding and unconstrained-lesion controls an exact paired decision
   rule if mechanism-specific attribution is desired.
2. Add an absolute H20 accuracy/comparator gate; H20/H5 alone permits uniformly
   terrible prediction.
3. State that seed intervals are conditional on one frozen train/calibration
   artifact, or replicate the whole training procedure if training-sample
   generalization is claimed.
4. Add upper95(false_lure)<=0.05 and report the maximum seed lure rate. The inherited
   G7-M V2 locked test had a mean below 0.05 but a seed maximum above 0.08.
5. Either filter inherited hard-recall candidates by inferred context/component or
   add a zero accepted-cross-context recall gate.
6. Preselect the interaction metric for any synergy claim or apply a multiplicity
   rule; no generic cross-metric synergy claim is identified.

P2:

1. Rename the metric “train-standardized pooled RMSE” or define “NRMSE” by the exact
   formula above; do not mix it with G9-CB raw-coordinate path RMSE.
2. Report all four seed arrays, means, standard deviations, CIs, strict wins, ties,
   and denominators, not only booleans.
3. Report the complete resource vector and hashes for every cell, plus cell-order
   invariance results.
4. Freeze comparison semantics with no hidden floating tolerance. If a tolerance is
   needed for regret nonnegativity or constraint geometry, register its value and
   dimension.

## 13. Minimum viable claim boundary and required deletions

If the full generator, planner, and common-budget contract can be frozen, retain
C1–C5 only after adopting sections 3–10.

If that cannot be done before implementation, the minimum honest V1 is narrower:

- delete C3, “counterfactual action selection,” “planning,” action success, regret,
  and invalid-action lures;
- delete “without increasing false recall” and use “within preregistered ceilings”;
- delete “both absolute component gates” unless the two gates are enumerated;
- delete generic “synergy”; at most report metric-specific benefit-oriented
  interactions;
- call the dream path “known-slot constrained schema completion” unless it actually
  updates a frozen transition/world model and affects autonomous rollout;
- do not call the inherited G7-M hard completion an attractor and do not relabel the
  failed G9-CB V5–V7 robustness results as support;
- if budget padding is not used, describe effects as component-plus-resource bundle
  effects;
- if shuffled/lesion controls are diagnostic only, delete memory-content and
  constraint-mechanism attribution.

A viable narrow allowed conclusion would be:

> Conditional on one frozen train-only calibration artifact, in the registered
> synthetic family, enabling the specified real-only episodic lookup and/or
> hypothetical schema-completion software paths changed the preregistered H20
> train-standardized rollout errors while the exact leakage and provenance
> invariants remained at zero.

Only restore planning language after section 5 is numerically frozen and its P0
tests pass.

## 14. Reproduction and scratch notes

Read-only inspection commands used:

~~~
Get-Content -Raw -Encoding UTF8 .codex\agents\ce-math-verifier.md
Get-Content -Raw -Encoding UTF8 .codex\skills\ce-dimensionless\SKILL.md
Get-Content -Raw -Encoding UTF8 docs\참조\무차원_감사_수학.md
Get-Content -Raw -Encoding UTF8 .tmp\agi-world-memory-v1\_workspace\ce\agi-world-memory-integration-v1-20260810\00-contract.md
rg --hidden --no-ignore -n "NRMSE|regret|paired|provenance|heldout|budget" .tmp\agi-world-memory-v1
Get-FileHash -Algorithm SHA256 <inspected artifact>
~~~

Scratch calculations are embedded in section 11. Scratch path: none. No validation,
test, train/calibration, pilot, dimensionless-test, or registered-seed command was
executed. The only file modified by this lane is 11-math.md.
