# Topology routing with a feasible shared budget

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-topology-routing-delay-packets-20260822`

The predecessor repaired event-time delay and restored full M1 binding to
`16/16`, but stopped before factor endpoints because
$\lceil0.25\,\mathrm{nnz}(W)\rceil$ exceeded one cue's restricted
cluster/path support. This successor changes only the sparse budget.

For each sealed circuit, let $M_k$ be the number of learned edges admitted by
the frozen cluster/path support for cue $k$. Before any rollout, set

$$B=\max\left(1,\left\lceil0.25\min_k M_k\right\rceil\right).$$

Every sparse arm and every cue uses this same exact $B$; `FULL` remains
unmasked. $B$, $\min_kM_k$, and the fraction are receipted. No target,
decoder, endpoint, or post-rollout state enters this calculation.

All routes, thresholds, learning schedule, horizon, decoder, development
seeds, controls, and gates otherwise remain exactly those of the original
topology contract. Confirmation remains sealed. This is synthetic runtime
evidence only; no biological, clinical, curvature-memory, physical-energy,
anatomical, or AGI claim is authorized.
