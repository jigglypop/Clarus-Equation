# Validation

Status: COMPLETE

## Environment

`.codex/hooks/python.cmd doctor` selected system Python 3.11.9 with bytecode
disabled and reported the repository Python path correctly.

## Independent residual-EFT calculation

The math lane ran:

`.codex/hooks/python.cmd python _workspace/ce/nonselected-quantum-dark-sector-20260825/artifacts/verify_residual_eft.py`

It obtained `oscillator_w=1.249e-17`, `rho=220.5`, and `vacuum_w=-1` for the
registered finite check. This checks the stated conditional formulas only.

## Focused regression

Command:

`.codex/hooks/python.cmd pytest tests/test_dimensionless.py tests/test_cosmology_closure_gate.py tests/test_density_bridge_variational_audit.py -q`

Result: `41 passed in 0.87s`.

The dimensionless, closure-boundary, and probability-to-density
non-identifiability regressions are green. The paper writer also checked UTF-8,
local links, the M5 forbidden-parent guard, and `git diff --check` on the stable
snapshot. No full test suite was run because the change is confined to theory
documents and the focused checks passed.

These machine results do not prove the physical-map axiom or observed dark
abundances.
