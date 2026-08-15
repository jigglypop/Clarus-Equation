# L4 two-channel weighted routing on the boxed L3 host

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l3-ne2-open-set-20260814`

Mode: full for a new operator, with inherited L0--L3 facts cited only.
Do not re-prove N-E1 boxes, N-E3 area $1/10$, or N-E2 occupancy on
$U_0=\operatorname{int}(B_c)$.

All registered quantities are dimensionless. `AGI GO` is forbidden.
Autonomy $A$, C. elegans identity, BrainRuntime, L5--L8, and metric
modules stay out of scope.

[Axiom: model choice L4-A0] "Living lineage" in this run means the
predecessor's finite boxed L3 construction: $F_{1/4}$ has two dividing
LAS points and a preregistered open occupancy split. It does not mean
experimenter-free autonomy. $C_{\mathrm{strict}}$ remains open.

## 1. Question

The predecessor closed a one-channel chemostat. A single flux $E\equiv 1$
is a sufficient statistic of that niche. L4 asks whether two flux
channels force a routing operator, and whether that operator must be
weighted rather than binary.

Question: on two copies of the boxed $F_{1/4}$ host, do there exist a
nonnegative weight $W$ and two registered flux pairs such that the
$T=32$ occupancy pair differs, while every binary matrix with the same
zero pattern, or the all-ones matrix, fails to separate those pairs?

## 2. Inherited facts (cite, do not redo)

[Cited theorem L4-C1] $Z_\pm$ are nonlinearly LAS for $F_{1/4}$.
Predecessor chain N-E1.

[Cited 산출 L4-C2] Extinction area at $q=1/2$ is $1/10$. N-E3.

[Cited theorem L4-C3] On $U_0=\operatorname{int}(B_c)$, $T=32$ occupancy
splits by $q_0\in\{1/4,3/4\}$. N-E2 GO (occupancy).

[Definition L4-D1] One body is a pair of cube states
$(Z^{\mathrm{L}},Z^{\mathrm{R}})$ with the same $F_{1/4}$ update law
except for a routed drive defined below. Both labels start in
$\{1/4,3/4\}$ as registered.

## 3. Two-channel drive

[Definition L4-D2] Flux is $E=(E_1,E_2)\in[0,1]^2$. A weight is a
nonnegative matrix

$$
W=\begin{pmatrix}w_{\mathrm{L}1}&w_{\mathrm{L}2}\\ w_{\mathrm{R}1}&w_{\mathrm{R}2}\end{pmatrix},
\qquad
w_{\mathrm{L}1}+w_{\mathrm{L}2}=1,\quad
w_{\mathrm{R}1}+w_{\mathrm{R}2}=1.
$$

The routed drives are $u=WE$, so $u_{\mathrm{L}},u_{\mathrm{R}}\in[0,1]$
when $E\in[0,1]^2$.

[Definition L4-D3] A binary router is a matrix $A\in\{0,1\}^{2\times 2}$
with each row not identically zero, followed by row-normalization to
sum one. The all-ones matrix is the complete binary router $A_{\mathbf 1}$.

[Axiom: model choice L4-A1] The drive enters only as a multiplicative
gate on pre-division growth of that copy:

$$
\widetilde m
=
\bigl[m\bigl(1+u\,r(q)(1-m)-\lambda(1-b)\bigr)\bigr]_+.
\tag{L4.1}
$$

When $u=1$ this recovers the predecessor $F_{1/4}$ growth bracket.
When $u=0$ the growth term vanishes. No other channel from $E$ into
$(m,b,q)$ is authorized. $q$-maps stay uncoupled between L and R.

[Definition L4-D4] Occupancy of a copy is
$\mathbf 1[(m_{32},b_{32})\in R_0]$ with the predecessor $R_0$.
The output pair is
$o=(o_{\mathrm{L}},o_{\mathrm{R}})\in\{0,1\}^2$.
Registered initials for each copy lie in $U_0\times\{1/4\}$ or
$U_0\times\{3/4\}$ as named in the scoring file before evaluation.

[Definition L4-D5] Registered fluxes are

$$
e^{(1)}=(1,0),\qquad e^{(2)}=(0,1).
\tag{L4.2}
$$

The identity weight is $W=I$. The complete binary router is
$A_{\mathbf 1}$ (each row $(1/2,1/2)$ after normalization).

## 4. Claims

[Open theorem L4-E1] For $W=I$ and both copies started at the same
registered $q_0=3/4$ point of $U_0$, the occupancy pair after $T=32$
differs between $e^{(1)}$ and $e^{(2)}$. Hybrid branch cuts must be
controlled as in the predecessor (no global continuity of $F^{32}$).

[Open theorem L4-E2] For $A_{\mathbf 1}$ and the same initials, the
occupancy pair is identical on $e^{(1)}$ and $e^{(2)}$.

[Open theorem L4-E3] Therefore the routed family with $W=I$ is not
equal, as an operator from $\{e^{(1)},e^{(2)}\}$ to occupancy pairs, to
the complete binary router. This is a finite two-channel construction.
It is not C. elegans, not $\mathcal L(W)\neq\mathcal L(A)$ for arbitrary
graphs, and not a brain.

[Hypothesis L4-H1] Every binary router with a symmetric support
(both off-diagonals equal, both diagonals equal) fails L4-E1's
separation. Routes must compare $I$, $A_{\mathbf 1}$, and at least one
asymmetric binary router, with a killing test.

Autonomy $A$ stays open: the kernel still advances only when `step` is
called.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No C. elegans identity. Inherited
P-A0--P-A2 and L4-A0--L4-A1 remain axioms.

### G-DIMENSIONLESS

$E_i,u,w,m,b,q$ and every occupancy bit are dimensionless.

### G-MATH

L4-E1 and L4-E2 are theorems on preregistered initials and fluxes, or
killed by a counterexample. A single numeric pair without an enclosure
or an exact branch track is not GO.

### G-CODE

After audit only: extend `universe_life_kernel.py` or add a sibling
module with no repository imports except stdlib (and NumPy if required).
Do not import V15--V18b or `runtime`. Do not write BrainRuntime wiring.

## 6. Decision rules

- `L4 WEIGHTED SEPARATION GO` if L4-E1 and L4-E2 both hold.
- `L4 STOP` if $W=I$ fails to separate or $A_{\mathbf 1}$ also
  separates (then the binary/weighted contrast dies).
- Neither verdict is C. elegans, autonomy, L5, or AGI.

`10-sources.md` is skipped (no new observations). `12-routes.md` runs
because L4-H1 is open.
