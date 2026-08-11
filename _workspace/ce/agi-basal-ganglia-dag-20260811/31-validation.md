# Loop 8H validation

Status: RUN ONCE AGAINST LOCKED VALIDATION — STOP

Diagnostic promise score: **80/100**. The conjunctive hard gate failed.

## Passed

- finite valid within-tick DAG;
- recurrent state finite and norm-capped;
- recurrent accuracy above feedforward DAG: ID `0.6754` versus `0.4023`, OOD
  `0.3175` versus `0.2483`; paired LCBs `+0.2555` and `+0.0581`;
- feedback alignment causal control: LCB `+0.3517` ID, `+0.1057` OOD;
- feedback sign control: LCB `+0.6017` ID, `+0.1854` OOD;
- stationary and flat nulls;
- no future read, environment clone, same-tick feedback commit, topology cycle,
  or nonfinite event;
- 10 related regression tests and Ruff passed.

## Failed

1. Recurrent NLL did not beat the static boosted-stump baseline. Although the
   recurrent DAG had much higher accuracy, its NLL was `5.4472` ID and `11.8571`
   OOD versus boosted `2.0469` and `2.1924`. Paired NLL-improvement LCBs were
   `-3.6099` and `-9.8686`.
2. Post-switch trials 2–5 improvement LCB was only `+0.0030` ID and `+0.0327`
   OOD, below the locked `+0.08` threshold.

## Diagnosis

The state reached the declared norm cap (`6.0`). Signed feedback carries causal
information, but repeated reinforcement makes the context policy overconfident.
After a hidden-context switch, the old route remains dominant; errors receive
tiny assigned probability and therefore produce catastrophic NLL even when
mean accuracy improves.

This is not a topology or recurrence failure. It is a missing uncertainty and
surprise-reset mechanism. Loop 8H coefficients must not be retuned after this
result. A later loop may preregister negative-feedback-dependent state
deconsolidation or a hyperdirect/arkypallidal reset and must compare it against
temperature calibration and generic adaptive forgetting.
