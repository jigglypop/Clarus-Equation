# Gate A kernel validation

Status: COMPLETE

The focused Rust suite passed 6/6 tests. The final fixture producer reported
39/39 passing analytic checks, including full-atlas signed orientation,
pointwise direct-versus-3D curvature agreement, all six metric components,
nonlinear flat pullback, nonflat curved control, oracle agreement, and
serial/Rayon determinism. The final compact evidence is linked from
`artifacts/GATE_A.md`.

Validation policy is two-tiered:

- `Gate A-KERNEL` is mandatory and fast: Rust tests, fixtures, and NumPy oracle.
- `Gate A-LOCK` is optional until Gate B is ready: release binary provenance,
  create-only manifests, and mutation campaigns.

No Gate B true/null dataset and no PFC/neural outcome was executed. Therefore
this validation supports only the numerical kernel, not the biological theory.
