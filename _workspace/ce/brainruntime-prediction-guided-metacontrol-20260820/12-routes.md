# C1 alternative-route assessment

Status: COMPLETE

These are mutually distinct prospective routes, not outcome-selected retries.
No route has been executed in this math lane.  Each must first incorporate the
P1 freezes in `11-math.md`; a seed, threshold, endpoint, or decoder-only change
is not a new route.

| Route | Mechanism and causal seam | Target-aware degrees of freedom | Required controls and falsifier | Assessment |
|---|---|---|---|---|
| R1: C1 action-conditioned forecast planner | Frozen next-summary predictor reaches action selector; action labels are deranged only at that port. | Ridge $\lambda=10^{-4}$, features, split, $u$, contexts, QR goal bank, all schedules, thresholds, bootstrap seed, and tie rule are fixed before development.  No outcome tuning is allowed. | Predictor audit vs explicitly defined persistence; edge shuffle; readout-only shuffle; persistence, random, error-magnitude-only, and reactive-mean-effect arms; zero candidate rollout and one actual step.  Falsified by any decision gate failure. | **Primary route.**  It tests the selected ledger seam directly after P1 clarifications.  A pass supports only synthetic policy dependence on forecast input. |
| R2: state-conditioned model-free action scorer | Replace the forecast vector with a separately frozen direct score map from the same pre-state/action features to declared loss, with no predicted next-state output exposed to the planner. | One predeclared model class/regularizer and the same fit/audit/test split.  It may not select a scorer after comparing development outcomes. | Match all snapshots/goals/actions and use a score-label shuffle at the planner port.  Falsified if it matches R1 under the same adverse controls, because then C1 cannot attribute its advantage specifically to next-state forecasts rather than a direct action-value readout. | **Secondary discriminating route.**  It does not rescue C1; it can narrow the claim if R1 passes by separating prediction-shaped control from generic state-conditioned scoring. |
| R3: BA-S1 structural SCC lesion preflight | Manipulate a recurrent support/SCC property while preserving an outside-SCC matched lesion, then measure the already-declared task. | Support threshold, SCC algorithm, lesion cardinality, matching rule, and seeds must be frozen before outcomes; none may be tuned to make an SCC appear favourable. | Verify a nontrivial SCC and feasible equal-size outside/SCC-preserving controls before execution.  Falsified/precluded by dense or giant-SCC support with no fair matched control, yielding `STRUCTURE_UNDEFINED_STOP`. | **Deferred conditional route.**  It is mechanism-distinct from C1 and may proceed only if its structural feasibility gate passes; C1 results cannot substitute for it. |

R1 is the only route licensed to enter implementation now.  R2 is a planned
claim-discriminator, and R3 remains conditional on topology feasibility.  None
licenses a biological or consciousness conclusion.
