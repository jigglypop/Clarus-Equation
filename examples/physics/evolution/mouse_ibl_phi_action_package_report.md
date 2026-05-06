# Mouse IBL Phi-action package gate

This packages the non-data bottlenecks after the Clarus residual audit.
Zebrafish continuous movement is excluded because it is a data bridge bottleneck.

## preregistered object

$$
\Phi_{\rm action,t}^{(s)}
\equiv
\epsilon_{t,S_{\rm train}}^{(s)}
\subset
H_t^{(s)}-\widehat H_t^{(s)}(X_t,R_t,H_{t-\ell}),
$$

$$
\Delta_{\Phi}^{(s,y)}
=
\mathrm{BA}(X,R,\widehat H,\Phi_{\rm action})
-
\mathrm{BA}(X,R,\widehat H).
$$

Promotion rule: axes/subspaces are selected inside train folds only; held-out test trials only score the preregistered rule.

## action metrics

| target | evidence | passed | interpretation |
|---|---|---|---|
| `first_movement_speed` | nested action subspace: 9/12, mean dBA 0.013697, median dBA 0.011018 | `True` | Phi_action survives train-selected residual-subspace testing |
| `wheel_action_direction` | nested action subspace: 8/12, mean dBA 0.020350, median dBA 0.017363 | `True` | Phi_action survives train-selected residual-subspace testing |
| `choice_sign` | nested choice subspace: 5/12, mean dBA 0.006472, median dBA -0.000203 | `False` | choice is not promoted to a universal Phi residual field |

## carrier metrics

| target | evidence | passed | interpretation |
|---|---|---|---|
| `speed_probe00` | only probe00: 7/9, mean dBA 0.008768; drop probe00: 3/6 | `True` | probe00 is a strong speed carrier, though weaker than full field |
| `wheel_probe00` | only probe00: 6/9, mean dBA 0.020289; full mean dBA 0.020350 | `False` | probe00 nearly matches wheel mean but misses strict replication |
| `wheel_top_probe00_units` | only top 16 probe00 units: 7/9, mean dBA 0.023757 | `True` | wheel action has a fold-local top-unit sufficient carrier |
| `speed_top_probe00_units` | only top 16 probe00 units: 6/9, mean dBA 0.008726 | `False` | speed weakens without top units but is not top-unit sufficient |

## decisions

| bottleneck | decision | status | next test |
|---|---|---|---|
| mouse action residual | Promote Phi_action = train-selected subspace of epsilon_t. | `selection_candidate` | Run the same preregistered Phi_action rule on a larger registered panel. |
| mouse action carrier | Treat probe00/top-unit evidence as carrier split, not full localization. | `mechanism_candidate` | Separate speed distributed-carrier and wheel top-unit-carrier hypotheses. |
| mouse choice residual | Do not promote choice Phi as a universal subspace. | `theory_refined` | Model policy/history and session-adaptive residual as separate terms. |
| stable named latent axis | Replace named-axis claim with train-selected residual-subspace claim. | `theory_refined` | Require axis identity only after anatomical/probe-ablation replication. |
| d=0 brain interpretation | Use d=0 only as zero-residual boundary condition. | `boundary_principle` | Measure residual contraction or residual entropy reduction, not d=0 arrival. |

## verdict

- Non-data action bottleneck is advanced to a preregistered Phi-action candidate.
- Choice and d=0 are advanced by narrowing the theory, not by claiming closure.
- The next empirical run should be a larger-panel Phi-action replication, not another post-hoc axis search.
