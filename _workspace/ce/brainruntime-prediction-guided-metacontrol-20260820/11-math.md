# C1 math and protocol audit

Status: COMPLETE

## Scope and predecessor check

The contract is complete and its carried-forward evidence is consistent with the
route ledger: Loop 10 is a `CONFIRMED_SIMULATOR` *prediction* result, M3 is a
retired predictive-write route, and BA-G1--BA-G3D do not license a metric or
mediation feature.  C1 introduces a distinct action-selection seam rather than
retuning M3 or a geometric endpoint.  The direct predecessor's `12-routes.md`,
`31-validation.md`, and `40-final-report.md` support that narrow starting point:
its frozen native predictor beat persistence on 32/32 confirmation seeds, but it
did not choose an action and did not support a brain or consciousness claim.

## Independent reconstruction

For $d=48$, the seven native vector fields
$(a,r,m,\alpha,u,x,\ell)$ plus the proposed drive contribute $8d$ coordinates;
the action one-hot and intercept contribute four.  Thus

$$
8d+4=8(48)+4=388.
$$

The fit set has $128\times3=384$ action-labelled rows and the predictor audit
has $48\times3=144$ rows.  These agree with the contract.  The policy set has
64 episodes, so an exactly equal three-action `random_balanced` schedule is
arithmetically impossible: $64=3\cdot21+1$.

The causal time order is valid *if implemented as written*: snapshot and
precommitted $(c_e,q_e)$, compute three algebraic ridge forecasts, select one
action, then call `BrainRuntime.step` once.  Candidate restoration/step calls on
policy-test snapshots would turn the purported forecast into a rollout oracle;
the declared zero-candidate-step counter is therefore a necessary gate, not a
diagnostic convenience.

With fit-target mean $\mu$ and componentwise positive scale $\sigma$, the
standardizer must be exactly

$$
z(v)_k=(v_k-\mu_k)/\max(\sigma_k,10^{-8}),
\qquad g_e=z(y(s_e))+0.25q_e,
$$

where $q_e$ has Euclidean norm one.  Consequently the declared primary loss is
dimensionless.  This does not make it a biological objective; it only defines
the synthetic one-step objective.

For each circuit and adverse arm, $A_i^b$ is bounded in $(-1,1)$ when the two
mean losses are nonnegative and their denominator is positive.  Its stated
minimum-across-arms gate is correctly more stringent than testing a favourable
mean across arms.  Bootstrap resampling must operate on these already
per-circuit aggregated rows, never on the 64 correlated episodes as independent
units.

## Required clarifications before implementation (P1)

1. **Output-state name collision.**  The predecessor's Loop-10 summary is of
   `runtime.activation`, whereas the C1 formula writes $y(s)$ using an
   unqualified $x$ even though $x_t$ already denotes STP availability in
   $\phi$.  Minimum correction: replace the formula with
   $x^{\mathrm{act}}=\operatorname{activation}(s)$ and use
   $y=[\operatorname{mean}(x^{\mathrm{act}}),\ldots]$ everywhere.  Record this
   exact four-vector in source and result hashes.
2. **Predictor-audit denominator is not operationally fixed.**
   Action-conditioned persistence MSE must be declared as
   $\operatorname{mean}_{j,a}\lVert y(s'_{j,a})-y(s_j)\rVert_2^2/4$ on the 144
   audit transitions, with the same target components and reduction as model
   MSE.  Require a finite, strictly positive denominator in every circuit;
   otherwise `STOP`.  Without this, a zero denominator or a component/reduction
   mismatch can make the prediction ratio undefined or selectively favourable.
3. **Context and schedule reproducibility.**  Hash and record the full
   per-circuit context table before fitting/evaluation, including its generator,
   seed, ordering, and the association to snapshot IDs.  Define
   `random_balanced` as counts $(22,21,21)$ under a pre-frozen cyclic choice of
   the surplus action (or another explicitly fixed permutation).  The same
   requirement applies to the nonzero-sign schedule.  Balanced alone cannot
   specify the 64-episode schedule.
4. **Bootstrap convention.**  Fix the percentile rule (e.g. sorted empirical
   order statistic with indices), resample only 16 seed rows with replacement,
   and require every resample statistic finite.  Retain the 10,000 bootstrap
   draws/seed and report the number of passing circuit rows separately from the
   confidence bound.
5. **Intervention identity.**  Log for every episode the action-labelled
   forecast tensor/cost vector before the planner map, the action after the map,
   and the actual drive.  For `edge_shuffle`, require the same unordered
   forecast/cost multiset and a nonidentity action-label permutation; for
   `readout_shuffle`, require the selected action and actual drive to be
   byte-identical to intact.  This distinguishes a real planner-port lesion from
   an inert logging intervention.

These are P1 protocol ambiguities, not a P0 counterexample: the declared
one-step C1 simulator claim remains mathematically coherent once the stated
quantities are pinned down.  No claim may advance until they are incorporated
into the executable frozen manifest.

## Gate audit and claim boundary

The four decision gates jointly test (i) predictive information on held-out
transitions, (ii) lower loss than every named adverse planner/control arm,
(iii) behavioural sensitivity to only the action-label mapping at the planner
port, and (iv) a display-only negative control.  They do not identify the
predictor as a biological mechanism, establish a real connectome/SCC property,
or support metacognition, selfhood, predictive coding in brains, or
consciousness.  That boundary in `00-contract.md` is necessary and sufficient
for the current simulator evidence.

## Reproducibility record

Focused arithmetic check, run without the blocked workspace virtual environment:

```text
cmd.exe /c .codex\\hooks\\python.cmd python -c "d=48; assert 8*d+4==388; assert 128*3==384; assert 48*3==144; assert 64%3==1; print(...)"
```

Result: `feature_dim=388`, `fit_rows=384`, `audit_rows=144`,
`policy_episodes=64`, `balanced_remainder=1`.
