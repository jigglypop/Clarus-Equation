# Implementation lane

Status: COMPLETE

Implemented current checkpoint:

- `clarus/belief_control.py`: rank-1 robust observer, action-effect matrix, pure H2/H3 planning, state persistence.
- `RuntimeAgent`: optional `belief_control_enabled=False` path and external `task_goal`; default path remains reactive.
- `clarus/dpc_benchmark.py`: teacher-identified and train-only fitted DPC validation.
- `clarus/credit_control.py`: signed eligibility and separately signed slow homeostasis.
- `clarus/delayed_credit_benchmark.py`: delayed reward causal ablations.

Runtime credit checkpoint implemented after the original list:

- `BrainRuntimeConfig.stdp_gate_mode` keeps `critic_derivative` as the default
  and exposes an experimental `external_signed` path.
- signed signals accumulate across the STDP apply interval and survive
  snapshot/restore.
- `runtime_credit_benchmark.py` compares OFF, legacy, signed, sign-flip,
  absolute, trace-off, reward-shuffle, and homeostasis-only with matched streams.
- Loop 2c can start all arms from the same projected structural manifold.

Next implementation checkpoint:

1. replace synthetic sufficient-statistic feature with observation-history state learning;
2. repair OOD calibration without validation-selected thresholds;
3. stop the current STDP route: both registered runtime efficacy gates failed;
4. learn latent state from observation/action history, rather than giving the
   planner the sufficient statistic;
5. require held-out task gain, causal ablations, and OOD calibration before
   connecting that learned state to the default runtime path.

Loop 3 implements that isolated state learner in
`clarus/history_state_benchmark.py`: train-only decay selection, frozen
logistic likelihood, ordered raw history ingestion, true two-state tanh RNN
comparison, and causal history ablations. It is not yet wired into the default
`RuntimeAgent`; the next checkpoint is transfer/control integration rather than
another same-task classifier comparison.

Loop 4 added a frozen-belief reward-transfer harness and a context-conditioned
two-state policy RNN. The locked run failed its stale-policy and oracle-gap
gates. A post-run, non-scoring audit also found that the policy comparator never
learned its SAFE class, so this harness cannot establish planner superiority.
The route is closed without tuning; it remains useful as a regression witness.

Loop 5 adds `episodic_memory.py` with explicit ADD/UPDATE/DELETE/NOOP audit,
evidence-returning recall, margin abstention, and bounded novelty/priority/
recency eviction. `episodic_memory_benchmark.py` holds fixed the existing
priority-only memory, merge-off, abstention-off, FIFO, and no-memory arms. It is
not yet the default `BrainRuntime.hippocampus`; runtime integration requires a
separate downstream-task guard.

