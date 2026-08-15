# V11 strong-comparator/OOD result

Status: COMPLETE

## Outcome

V11 is `STOP`. The confirmed V10 local/cloud mechanism does not match strong learned recurrence
and is not robust to the registered horizon shifts. Elman-20 and GRU-20 nearly solved ID/noise;
GRU-20 remained near `0.998` on the horizon and combined panels. Even compute-matched Elman-3
outperformed V10 on all four panels.

## Consequence

The V10 claim is narrowed to its confirmed synthetic architectural ablation: explicit local/shared
interaction is better than its registered local-only/cloud-only/no-memory variants. Claims of
general recurrent superiority, parameter efficiency sufficient to offset accuracy, or OOD
robustness are rejected by this run.

This STOP does not invalidate the contraction theorem or the causal V10 lesion result. It shows
that those properties are insufficient for competitive learned sequence computation and AGI.
