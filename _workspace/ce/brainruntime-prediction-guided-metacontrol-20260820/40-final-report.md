# Prediction-guided metacontrol: development result

Status: COMPLETE

## Abstract

This experiment asked a deliberately sharper question than the earlier
self-prediction test: does a frozen next-state predictor improve action choice
when its three action-conditioned forecasts are delivered to a planner?  The
implementation separated forecast calculation from action execution, used an
action-label shuffle at the planner port, and used a display-only shuffle as a
negative wiring control.  On sixteen preregistered development circuits the
predictor was accurate and the planner port clearly affected behavior, but the
chosen actions did not beat an always-zero correction policy.  The frozen
route therefore stops before confirmation.  This is evidence against the
specific synthetic controller, not evidence about biological metacognition or
consciousness.

## 1. Question and design

The predecessor showed that current native state plus a committed action can
predict the next four-component activation summary.  That result left open
whether the prediction was actually useful.  C1 fitted the predictor on 128
independent warm states, evaluated prediction on 48 disjoint states, and used
another 64 states only for policy evaluation.  For each policy state it
computed three forecasts without rolling the runtime forward, selected one
action, and then executed exactly one transition.

The decisive comparison was not predictor accuracy alone.  The intact policy
had to outperform five adverse policies simultaneously: planner-port label
shuffle, zero correction, balanced random actions, magnitude-only alarm, and
a state-independent mean-effect controller.  A readout-only shuffle also had
to leave action and loss exactly unchanged.

## 2. What passed

[Empirical simulator result] The predictor MSE ratio passed in every circuit.
Its mean was `0.262754`, and the preregistered bootstrap upper bound was
`0.281168`, well below `0.90`.  The edge shuffle changed every selected action
on average (bootstrap lower bound `1.0`) and raised mean loss from `0.925336`
to `3.919636`.  The display-only shuffle reproduced the intact action, drive,
and loss exactly.  These observations establish that the implemented
predictor-to-planner port was real rather than a decorative readout.

## 3. What failed

[Empirical simulator result] Useful control failed in all sixteen circuits.
The always-zero correction and reactive mean-effect controls both achieved
mean loss `0.602820`, better than the intact planner's `0.925336`.  Consequently
the worst-control advantage was negative in every circuit and its bootstrap
lower bound was `-0.239823`, whereas the frozen gate required a value above
`0.05`.

This distinction matters.  The result does not say that forecasting was
impossible: forecasting passed.  It says that this forecast, action set, and
one-step planner did not convert accuracy into external task improvement.  A
readout can predict the next state while a conservative policy still makes a
better decision.

## 4. Formal status

- [Empirical simulator result] Next-summary prediction relative to persistence:
  supported in this development block.
- [Empirical simulator result] The planner causally depends on the forecast
  label mapping: supported by the port intervention and readout placebo.
- [Rejected prediction] The frozen C1 policy improves the declared goal loss
  over every adverse controller: rejected at development.
- [Unfinished/unsupported] Biological predictive control, metacognition,
  consciousness, and a brain algorithm: not tested.

No confirmation result exists.  The confirmation API requires a verified
development `GO` artifact, which this run cannot supply.

## 5. Consequence for the research program

C1 should not be repaired by changing its seed block, goal bank, thresholds,
or decoder.  Its useful contribution is the negative separation between
prediction and control.  The next route must therefore not be another
synthetic seed tournament.  It should begin from actual neural recordings and
ask two independently measurable questions: whether a preregistered
state-space SPD summary changes across task or learning conditions, and
whether held-out activity supports context-dependent directed routing beyond
matched undirected, gain, and common-input controls.

That empirical route may discover co-varying geometry and routing, but it must
not call covariance the brain's physical metric or infer synaptic causation
without same-unit longitudinal and intervention evidence.

## 6. Reproduction

Focused mechanics:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_prediction_guided_metacontrol.py -q
```

Development artifact:

```text
.codex\hooks\python.cmd python -B -m reality_stone.clarus.runtime_prediction_guided_metacontrol_benchmark --output _workspace\ce\brainruntime-prediction-guided-metacontrol-20260820\artifacts\c1-development-results.json
```

The first command passed 6/6 tests.  The second produced the frozen `STOP`
artifact cited in `31-validation.md`.
