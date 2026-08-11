# Validation lane

Status: COMPLETE

Results:

- focused runtime/new-feature regression: `76 passed`;
- CE dimensionless/bridge regression: `36 passed`;
- bootstrap fixed point: residual `2.08e-17`, PASS;
- Loop 1a teacher-identified DPC: `66.92/100 HOLD`;
- Loop 1b learned DPC: `86.32/100 GO` within claim limit;
- Loop 2 delayed credit: `100/100 GO` within tabular mechanism claim limit.
- Loop 2b runtime signed credit: `0/100 STOP`; first-update structural
  projection was a confound and signed-off LCB was `-0.41836`.
- Loop 2c matched structural manifold: `0/100 STOP`; signed-off mean
  `-0.02279`, LCB `-0.06021`, signed-shuffle LCB `-0.00208`, and held-out
  guard delta `+0.07479 > 0.02`.
- post-change focused regression: `52 passed`.
- Loop 3 raw-history state discovery: corrected comparator run `85/100 GO`;
  ID/OOD return `0.95879/0.89072`, ECE `0.02021/0.02815`.
- Loop 3 recurrent comparison: ID LCB `0.0`, OOD LCB `-0.00498`; this proves
  noninferiority under the frozen `-0.03` margin, not superiority.
- The first Loop 3 artifact is INVALID and retained only for provenance because
  its supposed recurrent comparator was a flattened MLP. The separately named
  corrected-RNN artifact is the scored result.
- final AGI-focused regression: `61 passed`;
- CE bootstrap residual: `2.08e-17`, PASS;
- CE canonical validation: `53 passed`; dimensionless formulas `7/7`;
- constants scorecard: `23` total, `12` scored, `11 PASS`, `1 CAUTION`,
  `1 INPUT` excluded, `1 OPEN TEST`; aggregate `CAUTION` due to
  `Omega_b h^2 = -1.80 sigma`;
- proof-completion harness retained the explicit LO/tree/raw obstructions and
  labelled the improved readouts as candidates only;
- research-core checks: `OK lanes`, `OK gate`, `OK build`, `OK final`.
- Loop 4 modular reward transfer: `0/100 STOP`; stale LCB failed in four of
  six domain/context cells and maximum oracle gap was `0.020410 > 0.02`.
- Loop 4 context-RNN comparison is non-claimable: train accuracy `68.76%`,
  label counts `7729/269/8002`, prediction counts `3725/0/12275`.
- Loop 5 first artifact is INVALID because the registered no-memory arm was
  omitted; corrected artifact is scored separately.
- Loop 5 corrected episodic memory: `90/100 GO`; latest/evidence/abstention/
  deletion each `1.0`; composite LCB `+0.75` existing, `+0.75` merge-off,
  `+0.25` FIFO; audit and capacity guards pass.

Loop 1b primary per delay:

- return `0.76797` vs reactive `0.15`;
- LCB improvement `0.57344`;
- action-agnostic and H1 controls both lose the effect;
- recurrent sufficient-statistic control ties the candidate;
- success `0.86328`, Brier `0.07232`, ECE `0.04401`.

OOD sigma 1.2:

- return remains above reactive (`0.54102` vs `0.15`);
- success `0.73633`;
- ECE `0.09278`, so the preregistered calibration diagnostic fails.

The locked confirmatory block was not opened. No STDP learning-rate or
threshold sweep was run after the registered Loop 2c failure.

