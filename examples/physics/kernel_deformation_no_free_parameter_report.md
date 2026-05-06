# Kernel Deformation No-Free-Parameter Gate

## Model comparison

| model | c | kappa | fitted params | x | Omega_b pull | n_s pull | A_s GER pull | chi2 | AIC | guardrail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CE minimal kernel | 1.00000000 | 0.00000000 | 0 | 0.04864672 | -1.63 | +0.03 | +0.13 | 2.684 | 2.684 | allowed: kernel fixed before observables |
| c tuned to Omega_b | 0.99627164 | 0.00000000 | 1 | 0.04930000 | -0.00 | +0.03 | +2.91 | 8.493 | 10.493 | fit parameter: cannot be promoted without derivation |
| kappa tuned to Omega_b | 1.00000000 | -0.01246222 | 1 | 0.04930000 | -0.00 | +0.03 | +2.91 | 8.493 | 10.493 | fit parameter: interaction term must be independently derived |

## Verdict

Kernel deformation is not the next safe self-recursive lever.  One-parameter variants can tune Omega_b, but the gain is a fit parameter unless c or kappa is fixed by an independent theorem.

This keeps the core recursion conservative: use kernel deformation only after a no-free-parameter derivation, not as another observable readout correction.
