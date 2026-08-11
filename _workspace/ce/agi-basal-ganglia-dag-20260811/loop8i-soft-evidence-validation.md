# Loop 8I validation — uncertainty-preserving DAG

Status: LOCKED VALIDATION RUN ONCE — 90/100 STOP

Soft marginalization eliminated unreachable true actions (`0` ID/OOD). NLL
fell from hard recurrent `13.2290` to `1.2127` ID and from `29.9461` to `2.2721`
OOD. Hard-minus-soft NLL LCB was `+11.4496` ID and `+26.9473` OOD.

Recurrence, feedback derangement, feedback sign, stationary null, flat null,
boundedness, support, and integrity gates passed.

The sole failed gate was accuracy noninferiority. Soft minus hard accuracy was
`-0.00525` ID and `-0.00403` OOD, but paired lower bounds were `-0.010742` and
`-0.013794`, just below the locked `-0.01` limit. No Loop 8I coefficient may be
changed after this result.

Interpretation: the hard bottleneck produced falsely sharp probabilities and
must not be restored, but Loop 8I is not a promoted checkpoint. A child route
may use it as an explicitly experimental base and must independently recover
accuracy while preserving its proper-score improvement.
