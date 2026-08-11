# Future development checkpoint

Status: COMPLETE

No model test, evidence seed, or V9 gate was run.

When implementation is separately authorized, the order should be:

1. Audit inherited training residual spectrum, ACF, and innovation whiteness.
2. Fit rank-two `L,A,Q,R` with leave-one-training-episode-out selection.
3. Prove prefix-only behavior, PSD covariance, deterministic loading signs,
   pole ordering, and augmented stability on non-evidence unit inputs.
4. Freeze one model configuration and its ablations.
5. Use one fresh development block once; never touch 81100–81355.

Direction-selection buffer, not a confirmatory gate: paired 95% lower
improvement over V5 at least `+0.005`, positive lower improvement over
persistence and zero bridge, dense noninferiority, and all leakage/stability
checks. Failure means revise the state representation before any V9
registration; it does not authorize repeated route selection on the same block.
