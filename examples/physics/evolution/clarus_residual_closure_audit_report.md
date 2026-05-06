# Clarus residual-closure audit

Question: can the blocked brain/evolution gates be explained by a Clarus field,
i.e. a self-referential residual direction that reduces the leftover from the 0-space boundary?

## criterion

Promotion rule:

$$
\Phi_t \;\widehat{=}\; \epsilon_t=H_t-\widehat H_t(X_t,R_t,H_{t-\ell}),
\qquad
\Delta_{\Phi}=\mathrm{score}(X,R,\widehat H,\Phi)-\mathrm{score}(X,R,\widehat H)>0.
$$

A bottleneck counts as Clarus-closed only when the residual is measured and improves held-out
prediction under train-only selection. Missing timestamps, missing behavior bridges, or unstable
post-hoc axes do not count.

## verdict table

| bottleneck | evidence | Clarus fit | verdict | next gate |
|---|---|---|---|---|
| zebrafish continuous movement decoding | activity-only and behavior-frame gates pass, but public chunks lack an explicit e2 timestamp or e2-resampled tail/stage movement trace | weak: a residual-reduction field cannot replace the missing behavior-time bridge | `not_solved_data_boundary` | find timestamp-certified neural-to-tail/stage alignment, then test residual readout |
| mouse region-only decision/action closure | region-only, same-window interaction, lagged region coupling, and strict temporal GLM failed or were partial; low-rank state transition and innovation-to-behavior survived | strong: the surviving term is explicitly an innovation residual after X, R, and H_hat | `partly_solved_by_residual_field` | treat Phi_t as residual innovation epsilon_t, not as a pure region loop |
| mouse action readout | epsilon_t after X,R,H_hat survives for speed 9/12 and wheel 7/12; nested action subspace survives speed 9/12 and wheel 8/12 | strong: action behavior is where residual-reduction/readout has repeated support | `yes_action_channel` | pre-register Phi_action subspace and test speed/wheel split on a larger panel |
| mouse choice residual | choice innovation reproducibility fails on 24-panel: 8/24 support, mean dBA 0.001288, top1 axis null p=0.930300 | limited: choice residual may be session-adaptive Phi, but not a stable universal subspace | `not_yet_choice_closure` | model policy/history and session-adaptive residual separately before promotion |
| stable named latent axis | best single axes can predict targets, but axis identity is unstable; only a broader top3 concentration survives in some panels | medium: Clarus field can be a projection rule over residual subspace, not a named axis | `subspace_not_axis` | use train-selected residual subspace with anatomical/probe ablations |
| d=0 / zero-residual interpretation in brain data | brain gates show residual reduction, not finite-time arrival at a zero-residual state | conceptual only: d=0 is a boundary condition/ideal mirror, not an observed brain state | `boundary_principle_only` | measure monotone residual contraction or residual entropy reduction, not d=0 itself |

## summary

- Yes for the mouse action channel: the surviving object is already a residual innovation term.
- No for zebrafish continuous decoding: the blocker is a missing timestamp/behavior bridge.
- Not yet for mouse choice: residual signal is weak and not reproducible as a stable subspace.
- The clean mathematical reading is not `brain reaches d=0`; it is `brain dynamics repeatedly reduces residuals relative to a zero-residual boundary condition`.
