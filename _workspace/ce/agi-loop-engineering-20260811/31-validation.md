# Validation lane

Status: COMPLETE

Results:

- focused runtime/new-feature regression: `76 passed`;
- CE dimensionless/bridge regression: `36 passed`;
- bootstrap fixed point: residual `2.08e-17`, PASS;
- Loop 1a teacher-identified DPC: `66.92/100 HOLD`;
- Loop 1b learned DPC: `86.32/100 GO` within claim limit;
- Loop 2 delayed credit: `100/100 GO` within tabular mechanism claim limit.

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

The locked confirmatory block was not opened.

