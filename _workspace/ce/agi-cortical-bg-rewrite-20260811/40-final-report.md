# Verdict

The prior track did not fail because its Bayes filter was numerically wrong. It
failed to establish the claimed mechanism: the task supplied oracle-shaped
contexts and an XOR law, the graph lacked shared subactions, feedback updated a
context vector rather than graph edges, and the hazard bank solved volatility
estimation rather than basal-ganglia action gating.

The replacement is not another coefficient patch. It is a separable architecture:

`predictive cortical state -> conserved option flow -> co-active D1/D2 channels
-> STN/GPi HOLD gate -> action -> delayed local TD eligibility`.

The new equations guarantee finite within-tick execution, probability-mass
conservation, a non-cancelling STN effect, and normalized multi-path credit. They
do not yet prove that a shared DAG improves behavior or that the equations are a
literal biological implementation. Those claims are intentionally left pending
the topology, HOLD, credit, and cortical-state experiments in `12-routes.md`.

The mathematical core is now implemented and its invariants pass. The first
untouched-seed topology experiment found `0.999544` OOD accuracy for both the
shared DAG and the matched factorized-flat control, versus `0.0` for the atomic,
strict-tree, and destroyed-sharing controls. Thus reusable factor identity is a
GO, while DAG specificity remains unresolved rather than promoted.

Status: mathematics and core implementation validated; static factorization GO;
temporal option/DAG specificity remains the next open gate.

Experiment B then tested a Bellman-derived, non-hand-tuned HOLD decision. It
raised utility from `0.327880` to `0.551173` and accuracy from `0.663940` to
`0.818713`, while beating always-wait utility. The registered overall verdict is
still `STOP` because its entropy-quartile HOLD gap was `0.212027`, short of the
locked `0.25`. The common-offset STN control was exactly inert, as predicted.
