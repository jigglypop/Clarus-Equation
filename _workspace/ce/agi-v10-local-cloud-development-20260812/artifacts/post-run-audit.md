# Post-run audit

Status: PASS

- One registered development invocation completed in `69.9s` and wrote the result atomically.
- Registration SHA-256: `61A04E1080DD14DDBF48D73CDACF1532CDEAEBD451C5B7840D7909A1747E6D9D`.
- Result SHA-256: `CF6F304E1217E7CC446A9B5363C38F52D95261AB513E08F56CF6DF50DDE71302`.
- Embedded registration SHA equals the preregistration file SHA.
- All five registered implementation/test/runner hashes still match.
- Raw result contains exactly 64 seed rows in registered order.
- Independent recomputation of four arm means, three lesion means, primary paired mean, and
  factorial interaction mean matches the result exactly.
- Development/burned intersection: 0. Development/reserved-confirmation intersection: 0.
- Duplicate seeds, nonfinite output, label-state bypass, and arm mismatch: all 0.
- A second runner call stopped before evaluation with
  `FileExistsError: registered development result already exists; rerun forbidden`.
- Confirmation seeds were not evaluated or opened.
