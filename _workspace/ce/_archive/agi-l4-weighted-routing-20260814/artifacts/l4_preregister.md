# L4 preregister (written before any $T=32$ occupancy score)

This file names initials, fluxes, the identity weight, and the binary
routers used for $L4$-$E1$, $L4$-$E2$, $L4$-$E3$, and $L4$-$H1$.
It contains no occupancy bits, no $T=32$ images, and no scored
trajectory.

Inherited geometry (cite only; predecessor
`_workspace/ce/agi-l3-ne2-open-set-20260814/artifacts/o_e2_preregister.md`):

$$
R_0
=
\left[\frac25,\frac35\right]
\times
\left[\frac49,\frac6{11}\right],
\qquad
U_0=\operatorname{int}(B_{\mathrm{c}})
=
\left(\frac{13}{30},\frac{17}{30}\right)
\times
\left(\frac{137}{297},\frac{157}{297}\right).
$$

Horizon $T=32$. Occupancy of a copy is
$\mathbf 1[(m_{32},b_{32})\in R_0]$. The output is the pair
$o=(o_{\mathrm{L}},o_{\mathrm{R}})\in\{0,1\}^2$.
Drive law is (L4.1) only. $q$-maps stay uncoupled.

## Registered initials

Both copies start at the same point of $U_0\times\{3/4\}$:

$$
\bigl(m^{\mathrm{L}},b^{\mathrm{L}},q^{\mathrm{L}}\bigr)
=
\bigl(m^{\mathrm{R}},b^{\mathrm{R}},q^{\mathrm{R}}\bigr)
=
(m,b,3/4),
\qquad
(m,b)\in U_0.
$$

No other $q_0$ is scored. The closed box $\overline{U_0}=B_{\mathrm{c}}$
is used only as an outer enclosure of $U_0$, not as a new scoring set.

## Registered fluxes

$$
e^{(1)}=(1,0),\qquad e^{(2)}=(0,1).
$$

## Registered weights and routers

Identity weight (already row-stochastic):

$$
W=I
=
\begin{pmatrix}1&0\\ 0&1\end{pmatrix}.
$$

Complete binary router $A_{\mathbf 1}$ (all-ones, then row-normalized):

$$
A_{\mathbf 1}
=
\begin{pmatrix}1/2&1/2\\ 1/2&1/2\end{pmatrix}.
$$

Symmetric-support binaries (both diagonals equal, both off-diagonals
equal), after the same row-normalization:

$$
A_{I}=I,
\qquad
A_{J}
=
\begin{pmatrix}0&1\\ 1&0\end{pmatrix},
\qquad
A_{\mathbf 1}.
$$

One asymmetric binary (rows not a symmetric pair), already
row-stochastic:

$$
A_{\mathrm{L}}
=
\begin{pmatrix}1&0\\ 1&0\end{pmatrix}.
$$

Routed drive is $u=AE$ or $u=WE$. Each copy then uses (L4.1) with its
own $u\in[0,1]$.

## What is not registered

- No occupancy number.
- No point chosen after seeing a trajectory.
- No second compact defined from $T=32$ images.
- No $q_0=1/4$ pair, no $C$. elegans reading, no autonomy $A$.
- LAS boxes $Q_{\pm}$ are not used to name these initials.
