# Basal-ganglia DAG research verdict

The useful breakthrough is not an infinitely recursive tree. It is a recurrent
system composed of finite DAG execution slices:

`PFC/MD state -> forward proposal DAG -> reverse inhibitory reduction -> action
-> feedback -> next recurrent state`.

This matches the engineering need for indefinite operation without making one
forward call nonterminating. It is compatible with recurrent and convergent
features of basal-ganglia circuits only as a time-unrolled coarse-graining.

The recommended algorithm is a recurrent hierarchical mixture-of-experts/DAG
with reverse competitive inhibition and signed dopamine-like eligibility. An
XGBoost model belongs in the comparator set, not in the biological identity
claim.

Loop 8H implemented the predictor and ran the locked synthetic benchmark once.
It scored 80/100 but remains STOP. Recurrent accuracy, feedback alignment,
feedback sign, topology, boundedness, integrity, and null gates passed. NLL and
post-switch recovery failed because the recurrent context state saturated at
its norm cap and became overconfident. The next admissible route is a separately
preregistered surprise-dependent deconsolidation/reset mechanism, not retuning
Loop 8H after observing validation.

Loop 8I then replaced the hard content bottleneck with full probabilistic base
support. It removed all unreachable targets and cut NLL by more than an order of
magnitude, but missed the accuracy-noninferiority lower bound and remained
90/100 STOP. Loop 8J tested coefficient-free, confidence-gated directional
context labilization. It improved ID switch recovery but failed OOD, stationary,
and comparator gates (50/100 STOP). The remaining mathematical defect is causal
credit assignment: negative feedback must be attributed between sensory/base
uncertainty and context uncertainty before context state is updated.

Loop 8K implemented that factorization as an exact finite-state context filter.
It beat the hard parent in both accuracy and NLL, and passed support-alignment,
outcome-sign, reset, null, and mathematical-identity controls. It scored 80/100
STOP because the locked transition hazard `0.06` did not adapt to the OOD
environment hazard `0.12`; post-switch recovery remained below gate. The next
unresolved variable is therefore transition/change-point inference, not action
evidence, context likelihood, reset strength, or DAG topology.

Loop 8L closed the track with a finite joint hazard-by-context HMM. It correctly
separated stationary, ID, and OOD volatility, but learned hazard weights did not
beat fixed or frozen-weight alternatives in action accuracy and post-switch
gates. The run scored 70/100 STOP and is terminal by preregistration. This
removes hazard-grid tuning as the next route. Further evidence must come from a
broader task distribution, calibrated likelihoods, real neural/behavioral data,
or a separately authorized runtime experiment—not another version of this
synthetic benchmark.
