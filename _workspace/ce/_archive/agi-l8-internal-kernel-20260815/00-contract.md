# L8 internal kernel $K$: $\hat H = K(H)$ on the L7 host

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l7-region-loop-20260815`

Mode: full for a new operator. Cite L0--L7. Do not re-prove
$N$-$E1$, $N$-$E3$, $O$-$E1$, $L4$--$L7$ $E1$--$E3$, or $L6$-$E1$--$E3$.

All registered quantities are dimensionless. `AGI GO` is forbidden
and is not a decision rule of this run. Autonomy $A$, BrainRuntime,
a third cube, and metric modules stay out of scope.

[Axiom: model choice L8-A0] "Living lineage" means the boxed L3 host
plus `L4 WEIGHTED SEPARATION GO` through `L7 REGION LOOP GO`. It does
not mean experimenter-free autonomy. $C_{\mathrm{strict}}$ remains
open.

[Axiom: model choice L8-A1] An operator is added only when the previous
state is not a sufficient statistic of the registered niche. L7's
readout $o^{\mathrm{A}}$ is a sufficient statistic of the three-epoch
occupancy pair. L8 asks whether that bit is a sufficient statistic of
the **next host tuple**. $K$ must emit the same *kind* of variables as
the host (`artifacts/program-plan.md`: $\hat U=K(Z)$). $K$ is not a
third cube. $K$ is not BrainRuntime.

## 1. Question

The L7 host advances a tuple of the same slots each step: tick, flux,
two cubes, and the named bits. The L7 readout is one bit. L6 already
showed that two cubes with the same $\sigma$ can have different next
$(m,b)$.

Question: on a preregistered finite set $S$ of host tuples, does a
map $K$ whose codomain is that same tuple space equal the true
one-step host law, while the L7 bit $o^{\mathrm{A}}$ fails to
determine the next tuple?

## 2. Inherited facts (cite, do not redo)

[Cited theorem L8-C1] $Z_\pm$ LAS. $N$-$E1$.

[Cited 산출 L8-C2] Extinction area $1/10$. $N$-$E3$.

[Cited theorem L8-C3] $U_0$ occupancy split. $O$-$E1$.

[Cited theorem L8-C4] $L4$-$E1$--$E3$, $L5$-$E1$--$E3$,
$L6$-$E1$--$E3$, $L7$-$E1$--$E3$.

[Cited theorem L8-C5] $L7$-$H1$: $\sigma$-overwrite equals named $I$
as a $\gamma$-gate. A third cube is not required.

[Cited 기각 L8-C6] "A third cube is required" is not an active claim.

## 3. Host tuple and $K$

[Definition L8-D1] The host tuple of this run is

$$
H
=
\bigl(t,E,Z^{\mathrm{S}},Z^{\mathrm{A}},\sigma,I\bigr),
\tag{L8.1}
$$

with $t\in\mathbb{N}$, $E\in[0,1]^2$, $Z^{\mathrm{S}},Z^{\mathrm{A}}\in[0,1]^3$,
and $\sigma,I\in\{0,1\}$. These are the same *kind* of slots the L7
construction already advances. They are not a new cosmology.

[Definition L8-D2] The true one-step $\Phi$ is the predecessor law:
each cube follows $F_{1/4}$ with drive (L4.1) and $W=I$ gates from
$(\sigma,I,E)$ as in L5--L7; $t\mapsto t+1$; $E$ is held on a
registered one-step (no wash inside $S$). Bits are held on this
one-step. No other channel is authorized.

[Definition L8-D3] An internal kernel is a map $K$ with
$\operatorname{codomain}(K)$ equal to the space of $H$. The registered
$K$ is $\Phi$ itself, typed as $H\to H$. This is a construction, not
an AGI.

[Definition L8-D4] The registered set $S$ has two points, written
before evaluation. Both use $E=e^{(2)}$, $\sigma=1$, $I=1$, $t=0$,
$Z^{\mathrm{S}}=Z^{\mathrm{A}}$, and the L6 activity pair

$$
P_{\star}
=
\Bigl(\tfrac12,\tfrac{49}{99},\tfrac34\Bigr),
\qquad
P_{\circ}
=
\Bigl(\tfrac{7}{15},\tfrac{49}{99},\tfrac34\Bigr).
\tag{L8.2}
$$

So $H_{\star}=(0,e^{(2)},P_{\star},P_{\star},1,1)$ and
$H_{\circ}=(0,e^{(2)},P_{\circ},P_{\circ},1,1)$.
With $W=I$ and $I=1$, the action drive is $u^{\mathrm{A}}=1$
and the sensor drive is $u^{\mathrm{S}}=0$.

[Definition L8-D5] The L7 bit of $H$ is $o^{\mathrm{A}}=\mathbf 1[(m^{\mathrm{A}},b^{\mathrm{A}})\in R_0]$.
On $S$ both current bits are $1$ (both points lie in $U_0\subset R_0$).

## 4. Claims

[Open theorem L8-E1] $K(H_{\star})=\Phi(H_{\star})$ and
$K(H_{\circ})=\Phi(H_{\circ})$ as exact Fraction tuples. In
particular $K$ emits tick, flux, two cubes, and bits.

[Open theorem L8-E2] $o^{\mathrm{A}}(H_{\star})=o^{\mathrm{A}}(H_{\circ})$,
but $\Phi(H_{\star})\neq\Phi(H_{\circ})$ in the action $(m',b')$
(cite $L6$-$E1$; do not re-prove the one-step fractions). Therefore
$o^{\mathrm{A}}$ is not a sufficient statistic of $\Phi(H)$ on $S$.

[Open theorem L8-E3] $K$ and $o^{\mathrm{A}}$ are unequal as maps
from $S$ into the next-host space. Finite pair construction. Not
AGI, not autonomy, not BrainRuntime.

[Hypothesis L8-H1] Any map $K_{\mathrm{bit}}$ whose codomain is
$\{0,1\}$ fails $K_{\mathrm{bit}}(H)=\Phi(H)$ on $S$, because the
right-hand side is not a bit. If that holds, "a bit-valued internal
kernel is enough" is false. Routes must compare $H$-valued $K$,
bit-valued $K$, and a third-cube $K$, with a killing test.

Autonomy $A$ stays open: $\Phi$ and $K$ still advance only when
`step` is called.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No `AGI GO`. No mouse identity.
Inherited axioms remain axioms.

### G-DIMENSIONLESS

Every slot of $H$ is dimensionless.

### G-MATH

L8-E1 is type-plus-equality on $S$, or killed by a slot mismatch.
L8-E2 cites $L6$-$E1$ and $U_0\subset R_0$. A float pair is not GO.

### G-CODE

After audit only: a small typed helper on the existing kernel.
Do not import V15--V18b or `runtime`. Do not write BrainRuntime
wiring. Do not add a third cube. Do not grow the kernel by a new
subsystem class if a pair of functions suffices.

## 6. Decision rules

- `L8 INTERNAL KERNEL GO` if L8-E1 and L8-E2 both hold.
- `L8 STOP` if $K$ fails to match $\Phi$ on $S$, or $o^{\mathrm{A}}$
  already determines $\Phi(H)$ on $S$.
- Neither verdict is AGI, autonomy, or BrainRuntime.
- `AGI GO` remains forbidden even if this rule fires.
- L8-H1, if true, does not kill E1--E2. It only kills
  "a bit-valued $K$ is enough."

`10-sources.md` is skipped (no new observations). `12-routes.md` runs
because L8-H1 is open.
