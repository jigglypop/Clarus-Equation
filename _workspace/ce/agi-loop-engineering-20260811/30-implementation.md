# Implementation lane

Status: COMPLETE

## Loop 8B brain-geometry mechanism

- Added `brain_geometry_benchmark.py`: a bounded two-dimensional attractor
  benchmark with an exact two-node heat-flow control.
- Added five locked arms: pure diffusion, fixed attractor, continuous
  MD-modulated attractor, shuffled-context MD, and oracle-context MD.
- Every non-oracle arm receives the same causal cue filter and final readout;
  the candidate receives no latent context or future state.
- Added a CLI and focused tests. No canonical runtime state, Layer A–F equation,
  default STDP behavior, branch, or dependency was changed.
- Reproduction: `.venv/Scripts/python.exe examples/agi/brain_geometry_bench.py
  --output _workspace/ce/agi-loop-engineering-20260811/loop8b-brain-geometry-validation.json`.

## Loop 8C feedback-residual replay

- Extended the frozen Loop 8B model with one scalar residual state computed
  only after action-contingent feedback.
- Added checkpoint, residual, error-sign-flip, and oracle arms across ID, OOD,
  and stationary controls.
- The canonical hippocampus/runtime state remains unchanged; this is an
  isolated synthetic experiment.
- Reproduction: `.venv/Scripts/python.exe examples/agi/residual_replay_bench.py
  --output _workspace/ce/agi-loop-engineering-20260811/loop8c-residual-replay-validation.json`.

## Loop 8D conflict-adaptive STN boundary

- Added a read-only drift-diffusion decision layer over frozen Loop 8C memory
  traces with low, equal-mean matched, adaptive, and conflict-shuffle arms.
- Decision arms cannot mutate PFC, MD, or residual state; trace identity is
  checked after every arm.
- No canonical runtime state or dependency changed.
- Reproduction: `.venv/Scripts/python.exe examples/agi/stn_boundary_bench.py
  --output _workspace/ce/agi-loop-engineering-20260811/loop8d-stn-boundary-validation.json`.

## Loop 8E quasi-static gravitational decision field

- Added a finite-volume screened-Poisson solver, reusable source basis fields,
  underdamped motion, saddle-energy capture, DDM/STN comparators, and source
  shuffle/sign controls.
- It reads the frozen Loop 8C trace and does not modify canonical runtime state.
- Reproduction: `.venv/Scripts/python.exe examples/agi/gravitational_decision_bench.py
  --output _workspace/ce/agi-loop-engineering-20260811/loop8e-gravitational-decision-validation.json`.

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

Loop 6 adds `executive_control.py`: a normalized categorical rule belief,
hidden change hazard, action selection by expected rule correctness, and a
surprise-triggered next-step flexibility signal. The benchmark causally removes
hazard, surprise feedback, temporal feedback alignment, and gap retention. The
belief state is useful, but the scalar surprise boost failed its independent
efficacy gate and must not be wired into `RuntimeAgent`.

Loop 7 adds an expected rule-information term to action selection and includes
reward-only, information-only, surprise, shuffled-feedback, heuristic, and
oracle controls. It is retained as a negative witness: the existing card task
does not expose a probe action or action-dependent sensing, so information gain
cannot become a distinct executive control variable.

Implementation is now locked behind `loop8-unified-executive-equation.md`.
No additional RuntimeAgent branch or named INSPECT/WAIT/SWITCH handler is
allowed. The next code, after equation review, is an isolated exact finite-state
solver for one posterior and one policy functional.

