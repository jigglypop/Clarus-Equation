# Synthetic v1 invalidation ledger

Status: INVALID_IMPLEMENTATION

The prior artifact `synthetic-suite-results.json` is invalid as estimator
evidence. Its SHA-256 is
`A2C973C441D65DF05B3AF00D730F4F2C66F9B5BB09D8F98EDB249E78D0A30264`.

Reasons: candidate selection recreated generator truth, non-G1 metric and
direct candidates were equal by construction, test circuits were not disjoint
from training circuits, and the flat-pullback curvature result used a literal
p-value instead of a C2 tensor test. Preserve it for audit only. It is never
PASS or evidence.
