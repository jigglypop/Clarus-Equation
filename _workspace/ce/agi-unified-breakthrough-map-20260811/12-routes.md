# Breakthrough route map

Status: COMPLETE

## Attach to ACBSM core now

1. **Fast/slow residual state** — separates transient error from persistent
   bias. This changes internal memory rather than the final blend coefficient.
2. **Low-rank vector residual geometry** — lets chart residuals share a small
   number of directions instead of one pooled scalar score.
3. **Prefix Bayesian/Kalman observer** — estimates current latent amplitude
   and covariance sequentially instead of extrapolating one PCA score.
4. **Internal residual injection** — adds the inferred correction at every
   structural transition; it does not interpolate two completed H20 paths.
5. **Posterior-SNR trust** — reduces corrections only when the estimated state
   is uncertain. It is a safety/readout of the belief, not the performance engine.
6. **Robust state fitting** — Huber or Student-t innovations prevent a few
   shocks from becoming a false slow mode.
7. **Multi-horizon training loss** — aligns the state lifetime with H1–H20
   without giving each horizon an unrelated parameter set.
8. **Symmetric dense control** — swaps only the sparse structural graph while
   keeping the identical observer, uncertainty, and loss.

## Reserve as disabled modules in the same model

9. **Locked-origin regime switching** — normal/shock/recovery dynamics. Enable
   only if two fixed modes leave clustered innovations.
10. **Change-point reset** — clears stale slow state after a detected break;
    useful only if resets can be trained without target-aware thresholds.
11. **Latent environment belief** — represents loading/noise shifts as causes,
    rather than compensating them at the output.
12. **Episodic analogue memory** — retrieves a prior for `m_T,P_T`; retrieved
    trajectories must never be averaged directly into the answer.
13. **Latent-hypothesis beam rollout** — carries several state/regime futures
    when posterior multimodality is real.
14. **Sparse edge-strength adaptation** — permits small posterior changes to
    edge magnitudes while freezing graph support. This is deferred because it
    confounds state improvement with graph relearning.
15. **Receding-horizon consistency planner** — chooses among belief paths by
    stability and causal consistency. This belongs after forecasting state is
    demonstrated, not before.

## Reject for the current direction

- More scalar, lead, chart, or lead-by-chart output gains.
- Multi-origin averaging of completed trajectories.
- Full dense prefix refitting.
- Graph support relearning in the first model.
- Large neural regime gates or end-to-end joint optimization.
- Horizon-specific independent heads.
- More than two residual modes.
- Recomputing regime or trust from self-generated future values.
- Combining memory, regime, graph adaptation, and planning in the first run.

These either reproduced V8's local optimum, destroy attribution, or add more
degrees of freedom than the inherited training episodes can identify.
