# Dimensionless audit

Status: PASS

Natural units $\hbar=c=1$ are used.

| object | mass dimension |
|---|---:|
| $x^\mu$, $T$ | $-1$ |
| $\nabla_\mu$ | $+1$ |
| $X=-\frac12(\nabla T)^2$, $X_*$, $\delta$, $\kappa$, $a$ | $0$ |
| $\Gamma$, $H$, $M_{\rm Pl}$ | $1$ |
| $\rho_\infty$, $P$, $P_X$, $J=P_X\dot T$, $\Pi_{\rm fold}$ | $4$ |
| $P_T$, $\dot J$, $HJ$ | $5$ |

Consequently:

$$
[\Gamma T]=0,\qquad
[1-e^{-\Gamma T}]=0,
$$

$$
\left[\int d^4x\sqrt{-g}\,P\right]=0,
$$

and

$$
\left[
\int dt\,a^3\rho_\infty\Gamma e^{-\Gamma T}
\right]=4=[\Pi_{\rm fold}].
$$

For numerics, the normalized variables are

$$
\tau=H_0T,\qquad
\gamma=\Gamma/H_0,\qquad
A=\rho_\infty/(3M_{\rm Pl}^2H_0^2),\qquad
E=H/H_0,\qquad u=\kappa\delta.
$$

Every argument of an exponential and every ODE state is dimensionless. The
dimensionless operational depth is $\theta=\Gamma T=\gamma\tau$.
