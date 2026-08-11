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

That Loop 4 transfer attempt is `0/100 STOP`. The registered stale-planner
causal gate did not separate in four cells, one OOD oracle gap missed by
`0.000410`, and the small context policy RNN failed to learn SAFE at all. Hence
the apparently large candidate-RNN return gap is not accepted. The next route
must not tune this failed benchmark; it moves to an independently testable AGI
gap: hippocampal memory/replay efficacy under delay and interference.

Loop 5 closes the narrow memory-operation gap and scores `90/100 GO` on its
corrected bounded synthetic benchmark. Explicit UPDATE prevents duplicate cues
from retaining stale facts; evidence IDs, abstention, DELETE, and audit are all
causally required by the controls. This does not validate replay consolidation,
long-dialogue memory, or biological hippocampus. The candidate stays separate
from the default runtime until a downstream multi-session task and regression
guard justify integration.

