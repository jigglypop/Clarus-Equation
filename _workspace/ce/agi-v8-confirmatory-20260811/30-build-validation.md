# V8 build and validation record

## Implementation

The canonical predictor accepts only `(prefix_states, context, registration)`.
It independently fits sparse, same-probe dense, and zero-bridge gains from the
eight inherited training episodes and 22 registered origins per episode.
All 176 windows reproduce the locked values:

- sparse: `0.7868543064870357`
- same-probe dense: `0.7835668486813699`
- zero bridge: `0.882857758971467`

The candidate is generated once as `P + g(S-P)`. It is not recursively fed
back into a component. Future and hidden poisoning is bit-identical, H5 is
only the first five rows of H20, and the three shrinkage outputs have zero
convex-envelope violation.

## Verification

- Focused V1–V8 regression: `57 passed in 9.12s`.
- V8 unit and lock regression: `8 passed in 5.30s`.
- CE bootstrap solver: PASS, residual `2.08e-17`.
- CE scorecard: CAUTION, 11/12 scored rows PASS, unchanged in meaning.
- CE dimensional analysis: 7/7 PASS.
- CE proof-completion runner: completed successfully; candidate labels remain
  candidates and obstructions remain obstructions.
- Full repository pytest exceeded the 60-second execution window. Its first
  observed failure was unrelated to V8: missing
  `.claude/agents/ce-paper-writer.md`; the focused AGI regression is green.

## Confirmatory result

Validation seeds 80100–80355 were executed once and atomically saved. Overall
status: `R1_NOT_CONFIRMED`.

| Comparison | Candidate/control mean RMSE | Registered endpoint | Result |
|---|---:|---:|---|
| candidate | 0.5377851901 | — | — |
| V5 sparse parent | 0.5400745832 | lower CI > 0 | **FAIL**, -0.0031918816 |
| persistence | 0.5851127789 | lower CI > 0 | PASS, 0.0274287793 |
| zero-bridge shrinkage | 0.5494444233 | lower CI > 0 | PASS, 0.0027033469 |
| frozen V7 consensus | 0.5489099948 | lower CI > 0 | PASS, 0.0015434470 |
| same-probe dense shrinkage | 0.5378828913 | log upper <= log(1.02) | PASS, 0.0004911860 |
| stable adaptive dense | 0.5962335725 | log upper <= log(1.05) | PASS, -0.0439161065 |

All leakage, shape, finiteness, component-stability, common-norm, scale, and
artifact-lock checks passed. The secondary adaptive comparator again reached
radius `1.1327361583`; this was correctly reported but excluded from the
registered retained-component stability maximum `0.8216411318`.

The test unlock was explicitly challenged and returned
`V8 validation did not pass its full conjunction`. Test artifact existence
remained false.
