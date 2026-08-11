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
