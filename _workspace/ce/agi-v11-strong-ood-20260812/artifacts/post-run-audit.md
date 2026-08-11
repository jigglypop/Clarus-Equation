# V11 post-run audit

Status: PASS

- Single registered invocation completed in `19.8s` with `STOP`.
- Registration SHA-256: `411D8BB9A2FAD867E0AE7AB21D0BB3E58D298FBF63B2CCD9046EA36A348FF4D6`.
- Result SHA-256: `456E95F5E0DC7BE89E86924F721C1A01696BF2D6DA71AD73C8414AC2B6167181`.
- Exactly 16 registered seed rows occur in order; intersection with 140 prior seeds is zero.
- Independent recomputation matches every panel/model accuracy and Brier mean, all strong-contrast
  means, and all compute-matched contrast means.
- Seven registered hashes and embedded registration SHA match.
- Four integrity counters are zero.
- Ten of fourteen primary gates failed; no alternate threshold or run was opened.
- A second runner call failed before evaluation with `V11 result already exists`.
