# P-E1 route comparison calculations

Scratch arithmetic for `12-routes.md`. Not a certificate and not a
status upgrade. Fractions are exact when written as fractions; decimals
are display. Scripts: `_route_eval.py`, `_route_eval2.py`.

## 0. Source identities used

Predivision gain and one-step extinction:

$$
g=1+r(1-m)-\lambda(1-b),\qquad
\widetilde m=[m g]_+.
$$

$g\le 0$ iff $m\ge(1+r-\lambda+\lambda b)/r$ (for $r>0$). At the nominal
point this is $9m\ge 6+5b$. The line hits $m=1$ at $b=3/5$, and the area
inside the unit square is $1/10$.

Dividing fixed-point elimination with the $b$-nullcline
$b=\rho m/(\delta+\rho m)$:

$$
\rho r\, m^2+\bigl(r(\delta-\rho)+\rho\bigr)m+\delta(1+\lambda-r)=0.
$$

At the source this is $-\frac9{10}m^2+\frac14 m+\frac1{10}=0$, roots
$-2/9,1/2$.

Dividing-branch Jacobian at a point with $g=2$:

$$
J_{mb}
=
\begin{pmatrix}
1-rm/2 & \lambda m/2\\
\rho(1-b) & 1-\delta-\rho m
\end{pmatrix}.
$$

Jury–Schur: $1-\mathrm{tr}+\det$, $1+\mathrm{tr}+\det$, $1-\det$. Source
witness values $13/80$, $121/80$, $93/80$.

The $q$-map is left uncoupled on every route, so
$q_\ast\in\{1/4,1/2,3/4\}$ and the outer multiplier stays $7/8$.

Source $R_0=[2/5,3/5]\times[4/9,6/11]$ has
$\widetilde m\in[127/150,128/121]$ at the nominal point. A $41\times 41$
grid recovered $[0.8467,1.0579]$.

## 1. R1 / P-H1

$$
r(q)=\frac92\bigl(1+\kappa(2q-1)\bigr).
$$

At $q=1/2$, $r$ is nominal, so the extinction area is exactly $1/10$.

| $\kappa$ | $r(1/4)$ | low-$q$ $(m,b,\widetilde m)$ | divides | LAS | $r(3/4)$ | high-$q$ $(m,b,\widetilde m)$ | divides | LAS |
|---:|---:|---|---|---|---:|---|---|---|
| $1/8$ | $135/32$ | $(0.452,0.475,0.903)$ | yes | yes | $153/32$ | $(0.539,0.519,1.079)$ | yes | yes |
| $1/4$ | $63/16$ | $(7/18,7/16,7/9)$ | yes | yes | $81/16$ | $(0.572,0.534,1.144)$ | yes | yes |
| $1/3$ | $15/4$ | $(1/3,2/5,2/3)$ | no | yes | $21/4$ | $(0.591,0.542,1.183)$ | yes | yes |
| $1/2$ | $27/8$ | no real root | — | — | $45/8$ | $(0.625,0.555,1.249)$ | yes | yes |
| $1$ | $9/4$ | no real root | — | — | $27/4$ | $(0.697,0.582,1.394)$ | yes | no |

Low-$q$ loses $m\ge 3/8$ at $\kappa\approx 0.274$. The mass quadratic
discriminant changes sign at $\kappa\approx 0.478$. High-$q$ Jury
$1+\mathrm{tr}+\det$ becomes negative at $\kappa=1$.

Exact low-$q$ point at $\kappa=1/4$: $r=63/16$,
$b=\rho m/(\delta+\rho m)$ gives $(m,b)=(7/18,7/16)$. Jury
$(0.08142,2.19462,0.86198)$, spectral radius $\approx 0.904$.

$R_0$ predivision at $\kappa=1/4$: low-$q$ $[0.712,0.917]$ (already
below $\theta_D$), high-$q$ $[0.982,1.198]$.

$T=32$ from $R_0$ centre at $\kappa=1/4$:

- $q_0=1/4\to(0.391,0.440,1/4)$, 32 divisions, outside source $R_0$
- $q_0=3/4\to(0.572,0.534,3/4)$, 32 divisions, inside source $R_0$

At $\kappa=1/2$ the low-$q$ orbit from the centre has only 9 divisions
in 32 steps. At $\kappa=1$ it has 0.

The source box slice $\{4.49,4.51\}\times\{2.49,2.51\}\times\{0.199,0.201\}$
at $\kappa=1/4$ still has a dividing LAS root on both sides. This is a
probe, not a 7-dimensional re-proof.

## 2. R2 / P-H2

$$
\rho(q)=\frac15\bigl(1+\kappa(2q-1)\bigr).
$$

One-step $\widetilde m$ does not see $q$. Extinction area at $q=1/2$ is
$1/10$. $R_0$ predivision interval stays $[0.847,1.058]$.

| $\kappa$ | $\rho(1/4)$ | low-$q$ $(m,b,\widetilde m)$ | divides | LAS | $\rho(3/4)$ | high-$q$ | divides | LAS |
|---:|---:|---|---|---|---:|---|---|---|
| $1/4$ | $7/40$ | $(0.474,0.453,0.948)$ | yes | yes | $9/40$ | $(0.522,0.540,1.045)$ | yes | yes |
| $1/2$ | $3/20$ | $(0.444,0.400,0.889)$ | yes | yes | $1/4$ | $(0.542,0.575,1.084)$ | yes | yes |
| $1$ | $1/10$ | $(0.373,0.272,0.746)$ | no | yes | $3/10$ | $(0.574,0.632,1.147)$ | yes | yes |

$T=32$ on $R_0\times\{1/4,3/4\}$: both labels stay alive and both divide
32 times at $\kappa=1/4$. Occupancy of source $R_0$ therefore does not
depend on $\mathrm{sign}(q-1/2)$ on that registered rectangle.

An $11\times 11$ grid on $[0.20,0.80]\times[0.15,0.75]$:

| $\kappa$ | alive mismatches | $R_0$ mismatches | division-count mismatches |
|---:|---:|---:|---:|
| $1/4$ | 0 | 0 | 12 / 121 |
| $1/2$ | 0 | 0 | 20 / 121 |
| $1$ | 0 | 0 | 39 / 121 |

The 12 splits at $\kappa=1/4$ sit off $R_0$ (example: $(0.62,0.20)$
gives 29 vs 31 divisions). Using those points to save bullet 3 is
target-aware.

## 3. R3 / leak

Healthier-high-$q$ sign:

$$
\lambda(q)=\frac52\bigl(1-\kappa(2q-1)\bigr).
$$

Area at $q=1/2$ remains $1/10$. Area at the outer labels (not the P-E1
slice): $0.130$ vs $0.072$ at $\kappa=1/4$, $0.161$ vs $0.045$ at
$\kappa=1/2$, $0.224$ vs $0.006$ at $\kappa=1$.

| $\kappa$ | $\lambda(1/4)$ | low-$q$ | divides | LAS | $\lambda(3/4)$ | high-$q$ | divides | LAS |
|---:|---:|---|---|---|---:|---|---|---|
| $1/4$ | $45/16$ | $(0.448,0.473,0.896)$ | yes | yes | $35/16$ | $(0.545,0.522,1.090)$ | yes | yes |
| $1/2$ | $25/8$ | $(0.386,0.436,0.772)$ | yes | yes | $15/8$ | $(0.586,0.540,1.172)$ | yes | yes |
| $1$ | $15/4$ | no real root | — | — | $5/4$ | $(0.658,0.568,1.316)$ | yes | yes |

$R_0$ min $\widetilde m-\theta_D$ at low $q$: $-0.0075$ already at
$\kappa=1/4$, $-0.112$ at $\kappa=1/2$.

$T=32$ from $R_0$ centre:

- $\kappa=1/4$: both labels remain in $R_0$ (32 divisions)
- $\kappa=1/2$: low $q$ leaves $R_0$, high $q$ stays (32 vs 32)
- $\kappa=1$: low $q$ almost stops dividing (2 divisions)

The contract-parallel sign $\lambda(1+\kappa(2q-1))$ swaps the two
columns and makes high $q$ the first to lose its dividing root.

## 4. R4 / two-daughter survival

$$
p(q)=1-\kappa(1-q),\qquad
\sigma(q)=\frac{1+p(q)}{2}.
$$

The cube map multiplies the source predivision mass by $\sigma(q)$.
At $\kappa=0$, $\sigma=1$. At $q=1/2$, $\sigma=1-\kappa/4>0$, so the
zero-mass wedge is unchanged and the area is $1/10$.

If a second axiom-level reading counts complete daughters, the mean
per division is $1+p(q)$:

| $\kappa$ | $p(1/4)$ | $\sigma(1/4)$ | $1+p_-$ | $p(3/4)$ | $\sigma(3/4)$ | $1+p_+$ |
|---:|---:|---:|---:|---:|---:|---:|
| $1/4$ | $13/16$ | $29/32$ | $1.8125$ | $15/16$ | $31/32$ | $1.9375$ |
| $1/2$ | $5/8$ | $13/16$ | $1.625$ | $7/8$ | $15/16$ | $1.875$ |
| $1$ | $1/4$ | $5/8$ | $1.25$ | $3/4$ | $7/8$ | $1.75$ |

$T=32$ attractors at $\kappa=1/4$: $(0.435,0.466,1/4)$ and
$(0.480,0.490,3/4)$, both 32 divisions, both inside source $R_0$.
Modified Jacobian at the low-$q$ numerical point has Jury all positive
($\approx 0.11,1.97,0.96$). This is a numerical check, not a Jury
certificate.

At $\kappa=1/2$, low-$q$ $R_0$ occupancy fails and
$\widetilde m^\ast\approx 0.695<3/4$. At $\kappa=1$, low $q$ records 0
divisions from $R_0$.

## 5. R5 / threshold

$$
\theta_D(q)=\frac34-\frac\kappa4(2q-1).
$$

Source witnesses still have $\widetilde m=1\ge\theta_D(q)$ for every
$\kappa\in[0,1]$. $J_{mb}$ on the dividing branch does not depend on
$\theta_D$, so the source LAS points persist as points. $q$ never
enters $\widetilde m$ or $b'$. $T=32$ occupancy of $R_0$ is independent
of sign for $\kappa\le 1/2$.

## 6. Shared negatives

None of the five maps advances unless an external caller invokes
`step`. Coupling a label into mass or boundary does not produce the
autonomy conjunct $A$. The numbers above are route evaluations, not
a status change for P-E1.
