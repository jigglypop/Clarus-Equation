# C1 development validation

Status: COMPLETE

## Executed stage

The frozen development block `97901..97916` ran once through the system-Python
harness.  Confirmation seeds `99901..99932` were not executed.  The result is
stored at `artifacts/c1-development-results.json`.

- artifact SHA-256: `4e21f994ae1f6f2563a2c00bc13fba7ec9ac812e02701fed54d6eb6cdd49ae0b`
- canonical results SHA-256: `de2b59cb54de3a6d2007c3282b13b0c3bb80e45a55c5ec9af923003157be8b28`
- exact ordered circuits: 16/16
- integrity: 16/16
- independently regenerated summary: byte-value equal to the artifact summary

## Frozen gates

| Gate | Result | Decision |
|---|---:|---|
| Predictor MSE ratio at most 0.90 | 16/16; bootstrap mean UCB `0.281168` | PASS |
| Minimum advantage over every adverse policy above 0.05 | 0/16; bootstrap mean LCB `-0.239823` | **FAIL** |
| Edge shuffle changes action at least 0.20 | 16/16; bootstrap mean LCB `1.000000` | PASS |
| Readout equivalence and all integrity | 16/16 | PASS |

The route verdict is therefore `STOP`.

## Failure decomposition

Across circuits, intact prediction-guided control had mean loss `0.925336`.
The forecast/action-label shuffle (`3.919636`), balanced random policy
(`2.915771`), and error-magnitude-only policy (`3.472909`) were all worse, so
the predictor-to-planner edge was functionally connected.  However the
zero-action persistence policy and the reactive mean-effect policy both had
mean loss `0.602820`, below intact.  Their normalized advantage over intact
was `-0.200786` on average and exceeded the required `0.05` in 0/16 circuits.

The frozen predictor itself was accurate relative to persistence: the mean
raw-summary MSE ratio was `0.262754` (range `0.202431..0.347863`).  Accuracy of
the readout therefore did not imply useful control.  The intact planner chose
zero in 911/1024 episodes and a nonzero action in 113; the always-zero controls
won on the declared external loss.

## Integrity boundary

All weights remained fixed, dense/CSR views agreed, automatic STDP and memory
rows stayed zero, policy selection made zero candidate runtime transitions,
and every arm made one actual transition from the expected fixture snapshot.
The readout-only shuffle had identical action/drive trace hashes and identical
loss.  Thus the failed advantage gate is not attributed to leakage or an
unwired intervention.

Confirmation remains sealed because no valid development `GO` manifest can be
created from this artifact.  Thresholds, goals, actions, endpoints, or seed
blocks must not be retuned as a C1 rescue.
