# L7 preregister (written before any occupancy score)

This file names the three-epoch wash tasks, the registered start, the
identity weight, the body indexing, the named-$I$ law, the frozen-$\sigma$
feedforward, and the $\sigma$-overwrite used for $L7$-$E1$, $L7$-$E2$,
$L7$-$E3$, and $L7$-$H1$.
It contains no occupancy bits, no $T=32$ images, and no scored
trajectory.

Inherited geometry (cite only; predecessor
`_workspace/ce/agi-l6-activity-closure-20260815/artifacts/l6_preregister.md`,
L5
`_workspace/ce/agi-l5-role-split-20260814/artifacts/l5_preregister.md`,
and L3 open-set
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

Horizon $T=32$ per epoch. Occupancy of a copy is
$\mathbf 1[(m_{32},b_{32})\in R_0]$. Drive law is (L4.1) only.
$q$-maps stay uncoupled between copies.

## Body indexing

One body is the predecessor pair $(Z^{\mathrm{S}},Z^{\mathrm{A}})$.
Rows of $W$ are the predecessor $(L,R)$ order: left $\leftarrow E_1$,
right $\leftarrow E_2$. This run sets

$$
Z^{\mathrm{S}}=Z^{\mathrm{L}},\qquad Z^{\mathrm{A}}=Z^{\mathrm{R}}.
$$

Sensor therefore sees the first row of $W$. Action sees the second row.
No swapped indexing is scored. $I$ is a named bit, not a third cube.

## Registered start

Both copies start each epoch at the same point of $U_0\times\{3/4\}$:

$$
\bigl(m^{\mathrm{S}},b^{\mathrm{S}},q^{\mathrm{S}}\bigr)
=
\bigl(m^{\mathrm{A}},b^{\mathrm{A}},q^{\mathrm{A}}\bigr)
=
(m,b,3/4),
\qquad
(m,b)\in U_0.
$$

No other $q_0$ is scored. The closed box $\overline{U_0}=B_{\mathrm{c}}$
is an outer enclosure of $U_0$, not a new scoring set. Between epochs
the wash resets both copies to this start.

## Registered fluxes and weight

$$
e^{(1)}=(1,0),\qquad e^{(2)}=(0,1),
\qquad
W=I
=
\begin{pmatrix}1&0\\ 0&1\end{pmatrix}.
$$

The $W=I$ drive of a copy is $u_I(e)$, the corresponding coordinate of
$Ie$. Explicitly $u_I^{\mathrm{S}}(e^{(1)})=1$,
$u_I^{\mathrm{S}}(e^{(2)})=0$, $u_I^{\mathrm{A}}(e^{(1)})=0$,
$u_I^{\mathrm{A}}(e^{(2)})=1$.

## Registered three-epoch tasks

$$
\phi^{(1)}=(e^{(1)},e^{(2)},e^{(2)}),
\qquad
\phi^{(2)}=(e^{(1)},e^{(1)},e^{(2)}).
\tag{L7.1}
$$

An epoch is $T=32$ steps of one flux. Both tasks share epoch-$\alpha$
flux $e^{(1)}$ and epoch-$\gamma$ flux $e^{(2)}$. They differ only in
epoch $\beta$.

## Named-$I$ law (loop)

After epoch $\alpha$ the sensor copy writes

$$
\sigma=o^{\mathrm{S}}(\alpha)\in\{0,1\}
$$

as in L5. Epoch $\beta$ still uses the L5 gate

$$
u^{\mathrm{A}}=\sigma\,u_I(e^{\beta}).
$$

After epoch $\beta$ the action occupancy writes the internal bit

$$
I=o^{\mathrm{A}}(\beta)\in\{0,1\}.
\tag{L7.2}
$$

Epoch $\gamma$ uses the loop gate

$$
u^{\mathrm{A}}=I\,u_I(e^{\gamma}).
\tag{L7.3}
$$

The sensor copy always receives $u_I$ of the current flux. No other
channel from $I$ into $(m,b,q)$ is authorized.

The readout of a task is the action occupancy $o^{\mathrm{A}}$ after
epoch $\gamma$.

## Controls

- Frozen-$\sigma$ feedforward: same wash and same $W=I$ drives; $I$ is
  ignored. Epoch $\gamma$ uses $u^{\mathrm{A}}=\sigma\,u_I(e^{\gamma})$.
- $\sigma$-overwrite ($L7$-$H1$): after epoch $\beta$ the same named
  slot is overwritten $\sigma\leftarrow o^{\mathrm{A}}(\beta)$. Epoch
  $\gamma$ then uses the L5 gate $u^{\mathrm{A}}=\sigma\,u_I(e^{\gamma})$.
  No third cube is introduced.

## What is not registered

- No occupancy number.
- No point chosen after seeing a trajectory.
- No second compact defined from $T=32$ images.
- No $q_0=1/4$ pair, no mouse / CCF / zebrafish reading, no autonomy
  $A$, no AGI.
- LAS boxes $Q_{\pm}$ are not used to name these initials.
- No swapped body $(Z^{\mathrm{A}},Z^{\mathrm{S}})$.
- No third cube.
