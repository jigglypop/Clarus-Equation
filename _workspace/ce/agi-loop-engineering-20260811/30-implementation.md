# Implementation lane

Status: COMPLETE

Implemented current checkpoint:

- `clarus/belief_control.py`: rank-1 robust observer, action-effect matrix, pure H2/H3 planning, state persistence.
- `RuntimeAgent`: optional `belief_control_enabled=False` path and external `task_goal`; default path remains reactive.
- `clarus/dpc_benchmark.py`: teacher-identified and train-only fitted DPC validation.
- `clarus/credit_control.py`: signed eligibility and separately signed slow homeostasis.
- `clarus/delayed_credit_benchmark.py`: delayed reward causal ablations.

Next implementation checkpoint:

1. replace synthetic sufficient-statistic feature with observation-history state learning;
2. repair OOD calibration without validation-selected thresholds;
3. feed signed TD eligibility into a copy of the runtime STDP efficacy protocol;
4. require held-out task gain and stability before changing the default STDP gate.

