# Synthetic v3 postexecution audit

Status: INTEGRITY_PASS_SCIENTIFIC_GATE_FAIL

The one-shot result was produced after the source lock and was not rerun or
tuned. The preexecution source hashes still match. The trace SHA-256 is
`3ab5657e96e8f64c08b0f21b32552beb237fa6a5c81e50a945d14eb1291a4bde`.
All 120 records, finite arrays, W-to-density values, seeds, directions,
first-passage values, truth meshes, score-derived randomization p-values,
Holm decisions and numerical curvature rows were independently recomputed.

## Frozen gates

| Gate | Result | Threshold | Status |
|---|---:|---:|---|
| G1 metric-beta recovery | 15/20 | at least 18/20 | FAIL |
| G1 predictive Holm contrasts | 5/5 | 5/5 | component PASS |
| G2--G6 any-Holm count | 72/100 | at most 5/100 | FAIL |
| G3 pullback-c recovery | 15/20 | at least 18/20 | FAIL |
| G3 numerical curvature false positives | 0/20 | at most 1/20 | numerical fixture PASS |

The G1 one-sided randomization p-values were 0.00805 against direct v/Q,
0.000244 against gain/noise, 0.000976 against noise-only, 0.000244 against
Euclidean, and 0.000488 against flat pullback; all survived Holm. This shows
that the frozen metric model predicted its own matching generator better than
the controls, but parameter recovery was not sufficiently reliable.

The 72 any-Holm rows were independently recovered from trajectory scores:
G2 14/20, G3 11/20, G4 15/20, G5 16/20 and even G6 null 16/20. Only one row,
G4 circuit 14, promoted against all five controls. The large any-reject count
therefore cannot be explained only by genuinely non-Euclidean alternatives.
It exposes poor calibration of the one-circuit fit and nested trajectory-level
selection protocol. In four of the five G1 recovery failures the training
graph density was 0.2679 or 0.2857, close to the frozen zero-signal density
0.28, matching the preregistered identifiability risk without excusing it.

The curvature result is only a deterministic numerical check that Euclidean
and smooth pullback fields remain flat while a known conformal field is not.
It is neither a sampling-based curvature test nor neural evidence.

Final status: synthetic estimator validation failed. No biological claim and
no structural-to-metric-to-trajectory arrow may be promoted from this result.

