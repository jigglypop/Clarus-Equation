# Next loop draft: V11 strong comparator and OOD

Status: DRAFT — no seed opened

## Purpose

The confirmed V10 task is too narrow to support an AGI claim. The next gate must test whether the
increment survives stronger learned recurrence and registered distribution shifts.

## Candidate and comparators

- Frozen V10 full kernel plus train-only ridge readout.
- Dense Elman RNN with 20 hidden scalars, trained end-to-end on exactly the same train episodes.
- GRU with 20 hidden scalars as a deliberately parameter-advantaged comparator.
- Compute-matched recurrent comparator whose hidden width is chosen before any result so its
  declared MAC budget does not exceed V10's registered budget.
- Existing local-only, cloud-only, and no-memory arms remain diagnostic factorial controls.

All train optimizers, initializations, epochs, stopping rules, and seed roles must be locked before
scoring. Parameter count, active state, MAC, training compute, effective degrees of freedom, and
wall time are separate ledgers.

## OOD panels

1. noise: `sigma 0.04 -> 0.08`;
2. horizon: `4 -> 8` steps with the cue locations fixed and additional distractor ticks inserted;
3. amplitude: local/shared cue scales sampled from a preregistered unseen interval;
4. combined noise+horizon shift.

No label rule or context matrix changes; those would test task transfer rather than state-memory
robustness and require a separate contract.

## Required gates

- V10 must retain positive paired accuracy increment over both dense learned comparators on ID and
  every OOD panel, or meet a preregistered noninferiority margin while using materially fewer
  trainable parameters/MAC.
- Cross-cut and both reset losses must remain positive under the frozen V10 readout.
- Calibration/Brier and accuracy are both reported; neither may be selected post hoc.
- Development and confirmation seed roles are disjoint from all V10 seeds.
- Failure on any primary panel is STOP, preserved without same-run tuning.
