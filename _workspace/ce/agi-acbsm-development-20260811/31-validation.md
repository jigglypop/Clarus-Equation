# Validation

Status: COMPLETE

Focused ACBSM unit tests: `8 passed in 9.13s`.

Focused V1–V8 plus ACBSM regression: `65 passed in 13.52s`.

Training-only eight-fold screen:

- score: `67.12584746369484`
- classification: `HOLD`
- ACBSM versus legacy relative improvement: `+3.8594809%`
- rank-one posterior versus legacy: `+3.8594809%`
- rank-two versus rank one: effectively `0%` after stable collapse
- core/dense RMSE ratio: `1.0020100`
- episode wins: `5/8`

No seed in 82100..82355 was simulated. No implementation lock was issued, so
the development runner cannot consume that block. No V9 or locked test was
run.
