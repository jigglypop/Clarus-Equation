# Dimensionless audit: quantum-neighbor bootstrap

Status: COMPLETE

## Registered dimensions

Use time as the only independent dimension needed by the Markov generator and
set \(\hbar=1\) only inside the displayed master-equation convention.

| Object | Dimension | Audit |
|---|---:|---|
| \(\rho,n_i,\sigma_i^\pm,q_i,s_i\) | \(1\) | dimensionless state, projector, and probability objects |
| \(\kappa_{ij},\gamma_i\) | \(T^{-1}\) | transition and decay rates |
| \(L_{i\leftarrow j},R_i\) | \(T^{-1/2}\) | each dissipator has dimension \(T^{-1}\) |
| \(H\) | \(T^{-1}\) | because \(\hbar=1\); otherwise \(H/\hbar\) enters the generator |
| \(\tau\) | \(T\) | declared generation or observation window |
| \(A_{ji}=\kappa_{ij}\tau\) | \(1\) | admissible mean-offspring entry only under the branching-limit hypotheses |
| \(D=\rho(A)\) or a common row sum | \(1\) | admissible Poisson fixed-point parameter |

## Exponential and fixed-point gate

The multitype Poisson extinction map

\[
q_j=\exp\!\left[\sum_i A_{ji}(q_i-1)\right]
\]

passes the dimensionless gate because both \(A_{ji}\) and \(q_i-1\) are
dimensionless.  In the uniform sector, \(q=\exp[D(q-1)]\) and
\(-D e^{-D}\), the argument of \(W_0\), are also dimensionless.

The expressions \(\exp(\kappa_{ij})\), \(\exp(\gamma_i)\), or an identification
\(D=\kappa_{ij}\) fail unless a reference time has first been supplied.  The
candidate microscopic network therefore does not determine the dimensionless
CE readout \(D_{\rm eff}=d+\delta\) merely by specifying physical jump rates.

## Result

Dimension status: **PASS for the registered formulas** \(\kappa\tau\),
\(\gamma\tau\), \(A\), \(D\), and the extinction maps.  This is dimensional
consistency only; it does not derive the Poisson approximation, the CE readout,
or the residual dark-sector map.
