# Focused validation record

Status: COMPLETE

## Command

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\measurement-wall-dimensionality-20260825\artifacts\verify_measurement_wall.py'
```

## Result

All registered checks met tolerance $10^{-12}$. The maximum nonzero reported algebraic residual was $1.110\times10^{-16}$ for Kraus completeness of $\Phi_\eta$.

For $\gamma_0=2$ and $\Delta t=0.75$:

$$
\eta=0.776869839851570,
\qquad
(\rho_t)_{01}=0.111565080074215.
$$

For the constant binary distribution $(0.8,0.2)$:

$$
\overline C_I=0.293213034199730,
\qquad
C_{\rm wall}=0.227788362921137.
$$

These numbers verify the stated finite model only. They are not empirical fits, energy values, or cosmological predictions.
