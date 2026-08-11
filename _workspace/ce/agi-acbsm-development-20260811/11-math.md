# Mathematical verification

Status: COMPLETE

For each frozen mechanism, training residual lag covariances are decomposed
into up to two stable rank-one components with ordered poles. Signal matrices
are projected to the PSD cone, loading signs are canonical, `Q` and `R` are
PSD, and all poles are below 0.98.

The prefix filter applies predict then update to residuals
`x[t]-f(x[t-1])`. The final posterior is the state at the observed origin.
Forecasting first advances the belief and then injects its mean inside the
mechanism transition. No future observation, hidden state, or target is an
input.

Fast-mode evidence appeared in only four of eight raw leave-one-episode-out
fits, with unstable poles from 0.075 to 0.8. The predesigned fold-stability
rule therefore collapsed the candidate to rank one in every selected fold.
This is an identified absence of a second mode in this SCM, not a numerical
failure.

Across eight episode means, legacy RMSE was `0.6300720964` and ACBSM RMSE was
`0.6057545840`. Mean improvement was `0.0243175124` (3.8595 percent), but the
eight-fold paired 95 percent interval was `[-0.0223078218, 0.0709428467]`.
