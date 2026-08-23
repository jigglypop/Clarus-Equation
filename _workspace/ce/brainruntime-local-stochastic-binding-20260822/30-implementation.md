# BA-TR10 implementation

The default-off runtime extension adds a bounded packet-local competition
jitter (`competition_jitter_sigma`, `competition_jitter_seed`). It is evaluated
inside `_apply_local_competition`, after true axonal delivery and before
homeostatic attenuation and max-relative competition. It multiplies a positive
packet by `1 + sigma*tanh(xi)`; a zero packet stays exactly zero. Its stateless
seed is a function of the frozen jitter seed and snapshotted `step_index`.
Structural in-place mutation fails closed; Rust remains excluded by the
existing local-competition guard.

`runtime_local_stochastic_binding.py` starts the 4x4 `H <- S` support at exact
ones, captures the actual delay-ring packet before read-before-write delivery,
and applies only

`eta * post[:,None] * (pre[None,:] - post[:,None] * current_weight)`.

The installed matrix delta is zero outside `H <- S`. Training uses jitter;
evaluation reconstructs from a copied snapshot with jitter zero and performs no
write. All output-side structures and observations are absent.

