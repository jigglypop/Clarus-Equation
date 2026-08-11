# Verdict

Status: COMPLETE

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

The subsequent Indra-net correction changes the scaling architecture. A
countably expanding causal carrier can close on a finite group-orbit quotient
when its kernel is equivariant/equitable. The exact identity
`F_A L = L F_Abar` passed to `1.11e-16` while full network size grew and quotient
dimension stayed three. It also exposed a new boundary of the old SCC result:
every finite open truncation of `A[i,i+1]=2` becomes extinct, while the infinite
translation quotient has extinction `0.20318787`. Thus the proposed runtime is
now `collective quotient background + finite local causal deviation`, not an
ever-growing list of separately evaluated options.

The next dynamic finite-cover experiment now passes as a standalone mechanism.
For delayed signed `tanh` dynamics on cyclic covers of 32, 64, 128, and 256
cells, a fixed three-orbit quotient reproduced the full homogeneous trajectory
with error `0.0`.  A local perturbation was executed only on its causal cone and
reconstructed the dense reference to `2.78e-17`; the largest active slice was
13 nodes against a bound of 51.  Nonuniform translation tests had error `0.0`,
snapshot continuation was exact, same-tick edges were rejected, and budget
overflow failed closed.  The absolute small-gain certificate was `0.31 < 1`, so
the result controls nonzero spatial modes rather than certifying only the
homogeneous quotient mode.

This is `73/100`: GO for the standalone dynamic quotient/cone prototype, STOP
for BrainRuntime integration.  Missing points are approximate-budget error
curves, measured wall-time/memory scaling, and a matched behavioral task.  The
next experiment is therefore not another algebraic rewrite: it is a sidecar
task lane comparing quotient-only, local-cone, and dense full execution without
allowing the candidate to inspect the dense reference.
