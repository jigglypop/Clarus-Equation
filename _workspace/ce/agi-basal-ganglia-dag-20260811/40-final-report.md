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

Status: the equation and falsification route are ready. The predictor,
preregistration JSON, synthetic benchmark, and numerical score are not yet
implemented in this run.
