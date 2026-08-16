# Density-bridge variational audit implementation

Status: COMPLETE

Implementation date: 2026-08-15

## Scope enforced from the gate

The implementation is limited to claims approved in `20-audit.md`. It does not
promote the scalar construction to a baryon model or a cosmological prediction.
The potential choice is reported as `MODEL_AXIOM`, the input `D` as
`EXTERNAL_INPUT`, the critical-density bridge as `INCOMPLETE`, and the physical
prediction status as `NONE`.

## Files

- `examples/physics/density_bridge_variational_audit.py` is a standard-library
  audit executable and importable calculation module.
- `tests/test_density_bridge_variational_audit.py` contains thirteen focused
  regression and fail-closed tests.
- This file and `31-validation.md` are the only run-stage documents added by
  the implementation lane.

No existing cosmology constants, canonical documents, or user changes were
modified.

## Implemented claims

1. The dimensionless potential
   `v_D(x)=x log(x)-x+D(x-x^2/2)+C`, its derivative, and its Hessian.
2. A bracketed non-unit fixed-point solver in `y=-log(q)`, which cannot silently
   converge to the unit branch.
3. The local stable-small-branch and unstable-unit-branch checks for `D>1`.
4. The exact additive-offset construction that preserves the root and Hessian
   while changing the stress-energy fraction.
5. The weighted-event covariance identity and the equal-conditional-mean
   necessary-and-sufficient condition.
6. The conditional matter-composition statement `f_b^(m)=q`, kept distinct
   from the critical-density relation `Omega_b=q*Omega_m`.
7. The conserved-dust law `d log(Omega_b)/d log(a)=3*w_total` and the tracker
   transfer ratio `Q_b/(H*rho_b)=-3*w_total`.
8. The static canonical-scalar stress `w=-1`, explicitly distinguished from
   dust.
9. A natural-unit dimension ledger: `x`, `D`, the logarithm argument, and
   `v_D` have mass dimension zero; both action-density terms have dimension
   four.

The declared field domain is open at zero and closed at one. The implementation
therefore uses a separate field-domain validator rather than the strict
probability validator, and the unit endpoint is covered directly by tests.

## CLI policy

The normal CLI exit code tests only the approved mathematical statements. The
additional `--require-physical-bridge` mode is deliberately fail-closed and
returns process exit code 2 while the bridge remains `INCOMPLETE`.

## Commands and raw terminal results

```text
> uv run --extra dev python -m pytest tests/test_density_bridge_variational_audit.py -q
.............                                                            [100%]
13 passed in 1.01s
```

```text
> uv run --extra dev ruff check examples/physics/density_bridge_variational_audit.py tests/test_density_bridge_variational_audit.py
All checks passed!
```

```text
> python -m compileall -q examples/physics/density_bridge_variational_audit.py tests/test_density_bridge_variational_audit.py
<no stdout; process exit code 0>
```

Both CLI modes and their raw JSON/status output are recorded in
`31-validation.md`.
