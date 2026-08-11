# Validation

Status: COMPLETE

The 256 registered validation seeds were executed once. The canonical
artifact was saved with SHA-256
`b5850cafa1b76313f4717e924a2c4e5906095bf8cf7be3b9ceed75bf13cf53e7`
and committed at `5f20748fb16f1d1a09694f0a361954e028132d31`.

Result: `R1_NOT_CONFIRMED`. One check failed:
`v5_parent_improvement_ci`. Every other statistical, leakage, integrity,
stability, and hash-lock clause passed.

The locked test was not simulated. An explicit unlock challenge raised
`PermissionError`, and no V8 test artifact exists. Detailed values and the CE
verification harness outcomes are in `30-build-validation.md`.
