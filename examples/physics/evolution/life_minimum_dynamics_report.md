# Life minimum dynamics toy gate

This is a toy dynamical gate, not an empirical origin-of-life closure.
It asks whether autocatalysis, boundary retention, and copying are jointly necessary for persistence plus heritable template distinction.

## criteria

- mass_threshold: 0.25
- heredity_threshold: 0.5
- full_min_pass_rate: 0.8
- ablation_max_pass_rate: 0.35

## summary

- passed: `True`

| condition | pass rate | viable rate | heritable rate | mean min mass | mean heredity |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 0.609584 | 1.000000 |
| no_autocatalysis | 0.000 | 0.000 | 1.000 | 0.055588 | 1.000000 |
| no_boundary | 0.000 | 0.000 | 0.510 | 0.007359 | 0.554507 |
| no_copying | 0.020 | 1.000 | 0.020 | 0.475292 | 0.343680 |

## verdict

- Removing autocatalysis destroys repeated growth under dilution.
- Removing boundary retention destroys persistence in the open reactor.
- Removing copying leaves mass but loses heritable template distinction.
- Therefore the minimum life term is kept as a triad, not as any single component.

## equation update

$$
\boxed{
X_{n+1}
=
\Pi_{\mathcal C}
\left[
X_n
+A_{\mathrm{auto}}(X_n)
+B_{\mathrm{boundary}}(X_n)
+C_{\mathrm{copy}}(X_n)
-L_{\mathrm{leak}}(X_n)
\right]
}
$$
