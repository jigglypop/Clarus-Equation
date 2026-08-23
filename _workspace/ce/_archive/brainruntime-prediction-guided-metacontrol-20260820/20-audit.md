# C1 status and gate audit

Status: COMPLETE

## Scope

This audit covers only the stable C1 contract and its two internal review
lanes: `00-contract.md`, `10-sources.md`, `11-math.md`, and `12-routes.md`.
No implementation result is present in this run.  The decision below is an
admission decision for the frozen simulator protocol, not a result claim.

## Claim status

| Claim | Status | Basis and boundary |
|---|---|---|
| A frozen action-indexed predictor can be tested at a predictor-to-planner edge | `[예측]` / admissible | `00-contract.md` defines the edge, fixed task, frozen predictor, and four decision gates. It is not yet empirically established. |
| A positive result would show causal dependence of the simulated policy on that edge | `[미완성]` until execution | The intact/edge-shuffle intervention is causally interpretable only if the logged identity and one-step integrity gates pass. |
| Loop-10 is useful predecessor evidence for prediction | `[경험식]` / carried forward | The contract preserves the 32/32 result and explicitly excludes action selection, biology, and consciousness claims. |
| C1 identifies a brain algorithm, metacognition, consciousness, or a biological mechanism | `REJECTED` boundary | Explicitly excluded in `00-contract.md`; the simulator contains no biological measurement or intervention. |
| S1 SCC lesion is ready to execute | `DEFERRED` | `12-routes.md` correctly requires a nontrivial SCC and fair matched outside-SCC controls first. |

## P0/P1 findings

No P0 or unresolved P1 defect remains.  The previously identified goal-bank
ambiguity is closed by the revision in `00-contract.md`: an independent
seed-`97901` CPU stream, float64 `16 x 4` draw, reduced QR with a fixed
positive-diagonal sign convention, individual row normalization, explicit
zero/nonfinite rejection, pre-normalization/final hashes, and a unit-norm
tolerance are all frozen.  The contract also explicitly disclaims mutual row
orthogonality, so the construction is well-defined in R4.

The other five
clarifications recorded in `11-math.md` are reflected in the contract:

- activation is disambiguated as `x^act`, separate from STP `x_t`;
- persistence uses the same 144 action-conditioned audit rows and reduction,
  with finite positive denominators required;
- context tables, action schedules, and surplus/sign choices are frozen and
  hash-bound;
- bootstrap resamples 16 circuit rows only, with explicit order statistics and
  finite-replicate requirements;
- the planner-port intervention and readout-only negative control have
  episode-level identity logs and byte-equivalence requirements.

The final contract additions are also coherent and implementable: recurrent
weights use an exact per-circuit CPU initializer before construction; the 240
snapshots are allocated as 128 fit, 48 audit, and 64 policy rows with ordered
four-tick WAKE tapes and separate context tapes; the ridge, target statistics,
and standardizer use the declared float64 conventions; schedules and the
fit-only median are literal rather than outcome-selected; and raw-summary MSE,
bootstrap index conventions, degeneracy/denominator stops, and per-episode
planner-port/readout logs are explicit.  These additions close reproducibility
and outcome-leakage seams rather than introducing a new claim.

These are protocol requirements, not evidence that an implementation has
already satisfied them.  The implementation must stop before confirmation if
any required hash, counter, identity, or finiteness check is absent.

## Causal seam and controls

The causal seam is sufficiently isolated for the stated simulator claim:
forecasts are computed algebraically from pretransition features, the planner
selects one action, and only that action receives one real runtime step.
Policy-test candidate rollouts are forbidden.  `edge_shuffle` changes only the
forecast-to-action labels at the planner port; `readout_shuffle` is a
display-only negative control and must be byte-identical in action, drive,
trace, and loss.  Persistence, balanced random, error-magnitude-only, and
reactive-mean-effect arms cover constant, state-independent, magnitude-only,
and non-counterfactual alternatives.  The minimum-across-adverse-arms rule
prevents a single favourable comparator from carrying the result.

## Leakage and statistical unit

The declared fit, predictor-audit, and policy-test snapshots are disjoint, and
the policy test is inaccessible without a hash-bound development manifest.
The feature excludes post-state, realized loss, target next state, and
candidate policy rollouts.  Context/goal banks, schedules, standardizer,
source, split, and recurrent-weight hashes are required to freeze before
evaluation.  The circuit seed, not the 64 correlated episode rows, is the
statistical unit; the bootstrap convention is consistent with that choice.

The remaining implementation risk is operational rather than conceptual:
the runner must prove that no hidden candidate `step` or test-row fit occurs,
and must persist the complete pre-map forecast/cost tensors and hashes.  The
allowed implementation scope is only R1 under this frozen contract; it may
not add R2/R3, retune any retired route, or promote simulator output to a
biological claim.

## Simulator/biology boundary

The boundary is explicit and adequate.  Even a four-gate confirmation would
support only a conditional synthetic statement about a frozen `BrainRuntime`
task.  It cannot be promoted to a brain algorithm, predictive coding,
metacognition, selfhood, phenomenal consciousness, or biological neural
mechanism.  `10-sources.md` being `SKIPPED` is appropriate because this run
makes no external empirical claim.

## Implementation readiness and decision

R1 is the only licensed implementation route.  R2 remains a prospective
discriminator and R3 remains conditional; neither may be introduced as a
post-outcome rescue.  Before any development `GO`, the implementation must
freeze the source/result manifest and then run the declared development gates.
Any failed gate is `STOP`, with confirmation sealed.  A development pass is
not a biological or consciousness result.

Gate: PASS

The pass means the C1 protocol is internally coherent and may enter its
pre-registered simulator implementation.  It does not mean C1 has passed an
experiment.
