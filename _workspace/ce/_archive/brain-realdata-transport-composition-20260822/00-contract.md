# Real-data transport composition contract

Status: COMPLETE

## Question and claim ceiling

This run tests whether a train-only neural-state chart from real trial-resolved
calcium imaging supports the held-out predictive relation

$$
T_{02}\simeq T_{12}\circ T_{01}.
$$

The maximum admissible claim is an observational, session-local predictive
composition over an equal-lag delay period. A pass is not evidence for a
synaptic edge, anatomical path, structural shortcut, consolidation, causal
routing, curvature, disease mechanism, or AGI.

## PREDECESSOR_EVIDENCE

| evidence | status | retained information | forbidden retry/upgrade |
|---|---|---|---|
| BA-TR11--13 in `_workspace/ce/brain-algorithm-route-ledger.md` | `STOP` for curvature identity/sufficiency | geometry can be a derived signature, not stored identity | do not use curvature in this test |
| BA-TR28/29 in the same ledger | conditional synthetic prediction only | current content/transition prediction is the surviving narrow route | do not treat a designed candidate bank as biology |
| archived E17 source audit in `_workspace/ce/_archive/neural-riemannian-metric-validation-20260818` | `PASS_INPUT / BLOCKED_LONGITUDINAL_CHAIN` | 11 Figure 2 sessions, 3 animals, trial-resolved calcium; same-session ROI identity | do not join Figure 2/3/4 as one same-unit chain; released trial order is not verified chronology |
| local Tafazoli/Wójcik raw spike candidates | `BLOCKED_INPUT` | appropriate trial design exists in code | raw spike arrays are absent; do not substitute pseudopopulation caches |
| local CloudCell GCaMP/GFP | `PASS_INPUT / NO_NATIVE_TRIALS` | useful continuous and nuisance-control follow-up | not the primary trial-level composition test |

## Frozen dataset and states

Use all 11 local E17 Figure 2 `DCO*_dff.mat` files. Analyze saline and DCZ
blocks separately and never align ROIs across sessions. Primary signal is
`cont_data.{Sal,DCZ}.dff`; `branch` is a sensitivity that cannot rescue the
primary decision. The official frame chart is 180 samples over `[-3,3]` s.

The air-puff cue is at about `-1.8 s` and Go is at `0 s`. To avoid an
unmodelled external event between states, freeze three equal-lag, post-cue,
pre-Go windows centered at `(-1.5,-0.9,-0.3) s`, width `0.2 s`. Each state is
the per-trial window mean. All preprocessing is fitted inside each outer
whole-trial fold: finite-ROI filtering, centering, scaling, and one common PCA
chart of rank at most 6. Use five contiguous released-order folds and ridge
`1.0`; no endpoint-dependent tuning is allowed.

## Equation and controls

For row-vector latent states $x_0,x_1,x_2$ fit affine maps on outer-train
trials only:

$$
x_1\approx x_0A_{01}+b_{01},\qquad
x_2\approx x_1A_{12}+b_{12},\qquad
x_2\approx x_0A_{02}+b_{02}.
$$

The composed held-out prediction is

$$
\widehat x_2^{\rm comp}
=(x_0A_{01}+b_{01})A_{12}+b_{12}.
$$

The primary normalized excess is

$$
G=\frac{\operatorname{SSE}_{\rm comp}-\operatorname{SSE}_{\rm direct}}
        {\max(\operatorname{SSE}_{\rm persistence},\epsilon)}.
$$

Controls are direct $T_{02}$, persistence $x_2=x_0$, train mean, trial-
deranged successors, a reversed intermediate-coordinate interface, a pooled
stationary one-step map applied twice, and time-reversed composition.

## Frozen decision

For the primary `dff` panel, aggregate Sal/DCZ within session and sessions
within animal by summed held-out SSE. A row is `core_consistent` only when:

1. $G\le0.10$;
2. composition beats persistence;
3. composition beats the train-fold mean predictor;
4. composition beats the deranged-successor composition.

Return `OBSERVATIONAL_TRANSPORT_COMPOSITION_CONSISTENT` only if the overall
primary aggregate and all three animal aggregates meet all three conditions.
Otherwise return `OBSERVATIONAL_TRANSPORT_COMPOSITION_STOP`. With only three
animals, either result is descriptive rather than population-confirmatory.
