# Dimensionless audit: one-way zero-dimensional boundary

Status: COMPLETE  
Revision: 1 — `Z -> M` one-way channel and directed branching core.

Natural units $c=\hbar=1$ are used. Energy-power notation $E^p$ is mass
dimension $p$, and $T^{-1}=E$.

## 1. 0D boundary and preparation channel

For $\mathcal H_Z\cong\mathbb C$,

$$
\mathcal E(z)=z\rho_M,
\qquad\operatorname{Tr}\rho_M=1.
$$

| Quantity | Dimension | Consistency |
|---|---:|---|
| scalar input $z$ | $E^0$ | channel coordinate, not a spacetime coordinate |
| density operator $\rho_M$ | $E^0$ | unit trace |
| probabilities $p_a=\operatorname{Tr}\mathcal E_a(1)$ | $E^0$ | $\sum_a p_a=1$ |
| channel arrow $Z\to M$ | not a quantity | no dimensional assignment is needed |

Complete positivity and trace preservation are algebraic constraints, not
energy or length dimensions. A static 0D point does not acquire a time unit
from these constraints.

## 2. Cascaded GKSL channel

Write

$$
\dot\rho=-i[H_A+H_B+H_{\rm cas},\rho]+\mathcal D[a+b]\rho,
$$

$$
H_{\rm cas}=\frac{b^\dagger a-a^\dagger b}{2i}.
$$

| Quantity | Dimension | Consistency |
|---|---:|---|
| $\rho$ | $E^0$ | density operator |
| $a,b$ | $E^{1/2}=T^{-1/2}$ | rate factors are absorbed into coupling operators |
| $H_A,H_B,H_{\rm cas}$ | $E^1=T^{-1}$ | $b^\dagger a$ has energy dimension |
| $\mathcal D[a+b]$ | $E^1=T^{-1}$ | same dimension as $\dot\rho$ |

If dimensionless node operators $c_A,c_B$ are used instead, write
$a=\sqrt{\gamma_A}c_A$ and $b=\sqrt{\gamma_B}e^{i\varphi}c_B$. Omitting these
rate factors would be a dimensional error.

## 3. Directed neighbour jumps and branching

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\sigma_i^+n_j,
\qquad R_i=\sqrt{\gamma_i}\sigma_i^-.
$$

| Quantity | Dimension | Consistency |
|---|---:|---|
| $\sigma_i,n_i$ | $E^0$ | dimensionless operators |
| $\kappa_{ij},\gamma_i$ | $E^1=T^{-1}$ | transition rates |
| $L,R$ | $E^{1/2}$ | dissipator has dimension $E$ |
| CTMC rates $b_i,d_i$ | $E^1=T^{-1}$ | population generator rates |
| mean offspring $D$ | $E^0$ | expected count per parent |
| $q$, $1-q$ | $E^0$ | probabilities |

The fixed-point core

$$
q=\exp[-D(1-q)]
$$

is dimensionless. If it is derived from a physical transition rate, a declared
window or lifetime must appear, for example $D=\sum_i\kappa_i\tau_i$, so every
$\kappa_i\tau_i$ is dimensionless. A bare rate may not enter the exponential.

## 4. Residual-history map and four-dimensional EFT

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\nu_{\rm ns}(d\gamma),
$$

$$
S_{\rm res}=\int d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2
-\frac12m^2\phi^2-V_\Lambda\right].
$$

| Quantity | Dimension | Consistency |
|---|---:|---|
| subprobability measure $\nu_{\rm ns}$ | $E^0$ | retains total nonselected weight |
| kernel $\widehat K$ | $E^0$ | declared dimensionless |
| $M_*$, $m$, $\phi$ | $E^1$ | canonical four-dimensional scalar convention |
| $(\nabla\phi)^2,m^2\phi^2,V_\Lambda$ | $E^4$ | energy density/Lagrangian density |
| $d^4x$ | $E^{-4}$ | action is dimensionless |
| $T^{\mu\nu}$ | $E^4$ | stress-energy tensor |
| $\nabla_\mu T^{\mu\nu}$, bulk $J_Z^\nu$ | $E^5$ | continuous injection current density |

A distributional boundary current needs the matching surface and delta-function
dimension stated separately. Dimensional consistency alone does not derive the
kernel, conversion scale, junction law or dark-sector identity.

## 5. Code validation

The dependency-free checker
`artifacts/verify_zero_dimensional_overlap.py` passed all seventeen revised
checks. In particular it reported:

- preparation-state trace and instrument probability sum equal to $1$;
- for the declared $\mathbb C$-input instrument, the two output matrices (the
  complete Choi matrices in this one-dimensional domain) have nonnegative
  minimum eigenvalues, each map is trace-nonincreasing, and the summed-map
  trace-preservation residual is $0$;
- cascaded GKSL expansion residual $0$;
- cascade-Hamiltonian Hermiticity residual $0$;
- upstream feedback residual $0$ and downstream drive norm $0.5$;
- extinction fixed-point residual $0$ at
  $q=0.048646719644028225$;
- $Dq=0.15458752312007412<1$; and
- matching jump, generator, branching and scalar-map dimensions.

The repository-wide legacy dimensionless test still imports the user-split,
currently absent path `reality_stone/python/reality_stone/clarus/dimensionless.py`.
That unavailable apparatus is not used to claim either PASS or FAIL for this
run.

**Dimension status:** PASS inside the declared natural-unit preparation,
cascaded GKSL, directed-branching and four-dimensional residual-EFT models.
