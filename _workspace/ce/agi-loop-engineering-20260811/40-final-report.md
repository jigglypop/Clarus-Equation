# Checkpoint report

Status: COMPLETE

The first two mathematical gaps now have executable, feature-isolated experiments.

Action-conditioned belief plus adequate horizon passes the synthetic causal gates and scores `86.32/100 GO`, but it only ties a recurrent policy that has the same sufficient statistic. The honest conclusion is that the causal state and horizon matter; model-based planning superiority is not established. OOD calibration remains a concrete defect.

Signed delayed eligibility scores `100/100` on its tabular mechanism probe and collapses when the trace, reward sign, or reward alignment is removed. This justifies advancing the signed-credit equation to a runtime-scale A/B, not enabling STDP by default.

Runtime credit efficacy was tested in two registered stages. Loop 2b exposed a
dense-to-sparse structural projection confound. Loop 2c removed that confound by
starting every arm on the same projected manifold, but still scored `0/100
STOP`: signed credit did not beat OFF or reward shuffle and damaged the held-out
prediction guard. The external signed gate remains experimental and the default
STDP path remains off.

The next algorithmic checkpoint is observation/action-history latent-state
learning. This attacks the unresolved Loop 1 limitation directly: the current
planner was handed a sufficient statistic and therefore tied a recurrent
baseline. The next candidate must infer state from raw history, retain action
polarity and delay causality, and pass OOD calibration without validation-tuned
thresholds.

Loop 3 closes the narrow implementation gap: the candidate now consumes raw
ordered action/observation history, learns its decay and likelihood on train
data only, and scores `85/100 GO` on held-out and OOD synthetic episodes. The
causal effect disappears when action sign, alignment, or history is removed.
It only matches a correctly implemented tiny tanh RNN, so the breakthrough is
history-state discovery and calibration—not a claim that the CE state equation
is uniquely or generally superior. The next discriminating loop must test
intervention/reward transfer where an explicit belief and planner can adapt
without relearning a policy.

