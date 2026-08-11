# V8 conclusion and next algorithm checkpoint

Status: COMPLETE

## Outcome

R1 survived as a useful, stable development controller but failed independent
confirmation against its own unshrunk V5 parent. Its average improvement was
`0.0022893931`, while the paired 95% interval was
`[-0.0031918816, 0.0077706678]`. Therefore the fixed parent-anchor route is
closed as a confirmatory superiority claim.

This failure is informative rather than catastrophic. R1 did confirm positive
value over persistence, a zero-cross-chart-bridge control, and the frozen V7
controller; it tied the matched dense shrinkage extremely closely. What did
not transfer was a reliable advantage over leaving the sparse parent
unshrunk.

## Post-failure route audit

The failed validation block was retired to development use. No V8 test seed
was read. Three route families were tested:

1. metric-aligned scalar, leadwise, chartwise, and lead-by-chart gains;
2. five completed prefix backtests with leave-one-training-episode-out
   selection of prior shrinkage;
3. a simplex ensemble of sparse rollouts launched 0, 10, and 20 steps before
   the forecast origin plus persistence.

None produced a credible breakthrough. The best metric-aligned scalar mean
was `0.5376955035`, but its V5 lower endpoint remained `-0.0018459594`.
Prefix adaptation worsened mean RMSE to `0.5396276290`. The multi-origin fit
assigned weights `[0.7860754, 0.0008001, 0, 0.2131245]`, effectively collapsing
back to R1, and had mean `0.5377894590`.

A reconstruction of every validation seed's interpolation loss showed that
even the hindsight global scalar minimizing validation mean used gain
`0.8235` and still had V5 lower endpoint `-0.0021420624`. This is strong
evidence that coefficient tuning inside the line segment between persistence
and V5 cannot solve the registered problem.

## Next algorithmic direction

The checkpoint is now:

`stable sparse mechanism + prefix-only API + preserved controls + failed scalar output correction`.

The next route must change the parent forecast's state representation or
residual dynamics, not add another scalar/lead/chart shrinkage or output
ensemble. The most defensible successor is a **training-fitted multi-timescale
latent residual state**:

- keep the frozen sparse cross-chart mechanism and training-only scales;
- replace the single pooled scalar residual AR with two stable latent residual
  modes, fitted only on inherited training trajectories;
- infer the two residual amplitudes from the observed prefix only;
- enforce a common contraction certificate for the augmented state;
- compare directly with the frozen V5 parent, persistence, zero bridge, and a
  symmetric dense two-mode control;
- use new validation/test seed blocks and preregister them only after a fresh
  development block shows a material, not marginal, V5 lower confidence
  bound.

Why this is the correct escalation: all failed variants changed only the
readout of the same one-mode parent and converged to the same solution. A
second residual timescale changes what the model can represent while
preserving the sparse mechanism, leakage boundary, symmetric controls, and
stability discipline already validated.

No V9 evidence block should be registered yet. First require a fresh
development result whose paired lower improvement over V5 has a safety buffer
substantially above zero; `>= 0.005` is a reasonable route-selection target,
not a future confirmatory gate.
