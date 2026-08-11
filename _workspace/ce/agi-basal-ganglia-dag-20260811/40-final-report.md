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
