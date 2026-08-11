# Loop 8L final validation — finite hazard ensemble

Status: LOCKED VALIDATION RUN ONCE — 70/100 STOP — TRACK TERMINAL

The joint HMM was numerically valid: joint normalization error at most
`6.66e-16`, action-mixture residual `2.22e-16`, outcome Bayes residual
`5.55e-17`, zero degenerate evidence, and no nonfinite event.

Hazard identification worked descriptively. Mean final expected hazard was
`0.05666` ID and `0.13857` OOD. In the matched stationary null, posterior mass
on `h=0` averaged `0.99949`, above every alternative.

The candidate did not create decision superiority. ID accuracy equaled the
fixed `h=.06` mean (`0.70081`) but paired LCB was `-0.00317`. OOD accuracy was
`0.37085` versus fixed `0.37610`, LCB `-0.01050`. OOD post-switch accuracy rose
from `0.28837` to `0.30525`, but its LCB was only `+0.00554`, below the locked
`+0.03`. Learned model weights did not beat frozen uniform weights in either
domain. NLL remained far better than the hard recurrent parent and causal
support/sign controls passed.

Conclusion: the filter can identify volatility regimes, but under these
pseudo-likelihoods model-weight adaptation does not improve action selection.
No more coefficient, grid, prior, temperature, or hazard tuning is authorized
in this synthetic track. The active checkpoint remains PFC–MD plus residual
feedback; soft DAG/outcome filtering remains an experimental component, not a
promoted basal-ganglia mechanism.
