# Confirmation post-run audit

Status: PASS

- Single confirmation invocation completed in `80.6s`.
- Registration SHA-256: `8C503A1644BE3DA23FF106ABEFFF46E583F63BA9323D2F7F3A312FDAF22E5C56`.
- Result SHA-256: `E348C6D18CF6D5C11BC287BD2899FDAC52DCA69DB6B3A2FA559D5AF88F8FD6F8`.
- Exactly 64 result rows occur in the pre-development reserved order.
- Independent raw-row recomputation matches all four arm means, three lesion means, primary
  paired mean, and interaction mean exactly.
- Embedded registration SHA and all six registered hashes match.
- Integrity counters are all zero.
- A second runner call failed before evaluation with `confirmation result already exists`.
- No alternate analysis, threshold, task, or seed was opened.
