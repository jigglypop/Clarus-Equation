# L7 region loop: decision / action / internal on the L6 host

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l6-activity-closure-20260815`

Mode: full for a new operator. Cite L0--L6. Do not re-prove
$N$-$E1$, $N$-$E3$, $O$-$E1$, $L4$-$E1$--$E3$, $L5$-$E1$--$E3$, or
$L6$-$E1$--$E3$.

All registered quantities are dimensionless. `AGI GO` is forbidden.
Autonomy $A$, mouse / zebrafish / Drosophila / C. elegans identity,
BrainRuntime, L8, and metric modules stay out of scope.

[Axiom: model choice L7-A0] "Living lineage" means the boxed L3 host
plus `L4 WEIGHTED SEPARATION GO`, `L5 ROLE SPLIT GO`, and
`L6 ACTIVITY CLOSURE GO`. It does not mean experimenter-free
autonomy. $C_{\mathrm{strict}}$ remains open.

[Axiom: model choice L7-A1] An operator is added only when the previous
state is not a sufficient statistic of the registered niche. After L5,
frozen $\sigma$ is a sufficient statistic of the two-epoch wash for
$o^{\mathrm{A}}$. L7 therefore uses a three-epoch wash whose epoch-3
flux is shared, so frozen $\sigma$ is not a sufficient statistic of
epoch 3. It does not copy mouse CCF names into the kernel.

## 1. Question

L5 writes $\sigma$ once from epoch $\alpha$ and reads it as a
feedforward gate in epoch $\beta$. L6 showed that activity $P$
predicts $P$ on a pair; occupancy still follows $\sigma$ and $q_0$.

A three-epoch wash presents $\phi=(\alpha,\beta,\gamma)$ and resets
both copies to the same registered point of $U_0\times\{3/4\}$ between
epochs. Frozen $\sigma$ from $\alpha$ does not record what the action
copy did in $\beta$.

Question: does an internal bit $I$ updated from epoch-$\beta$ action
occupancy, then used as the epoch-$\gamma$ action gate, separate two
registered three-epoch tasks that share $\alpha$ and $\gamma$, while
the frozen-$\sigma$ feedforward on the same wash fails to separate
them?

## 2. Inherited facts (cite, do not redo)

[Cited theorem L7-C1] $Z_\pm$ LAS. $N$-$E1$.

[Cited 산출 L7-C2] Extinction area $1/10$. $N$-$E3$.

[Cited theorem L7-C3] $U_0$ occupancy split. $O$-$E1$.

[Cited theorem L7-C4] $W=I$ pairs. $L4$-$E1$--$E3$.

[Cited theorem L7-C5] Wash+$\sigma$ two-epoch split. $L5$-$E1$--$E3$.

[Cited theorem L7-C6] One-step $P\mapsto P'$ on the L6 pair.
$L6$-$E1$--$E3$.

[Cited 산출 L7-C7] $L6$-$H1$: $T=32$ occupancy on that pair is both
$1$ by $O$-$E1$.

[Cited 기각 L7-C8] "Activity is required for occupancy on $U_0$" is
not an active claim.

## 3. Three-epoch wash and the loop

[Definition L7-D1] The body is the predecessor pair
$(Z^{\mathrm{S}},Z^{\mathrm{A}})$ with $W=I$ and drive (L4.1).
Roles stay $(S,A)=(L,R)$. Each epoch starts at the L5 registered
point of $U_0\times\{3/4\}$.

[Definition L7-D2] Registered fluxes are $e^{(1)}=(1,0)$,
$e^{(2)}=(0,1)$. A three-epoch task is
$(\alpha,\beta,\gamma)\in\{e^{(1)},e^{(2)}\}^3$. The registered pair
is

$$
\phi^{(1)}=(e^{(1)},e^{(2)},e^{(2)}),
\qquad
\phi^{(2)}=(e^{(1)},e^{(1)},e^{(2)}).
\tag{L7.1}
$$

Both share epoch $\alpha=e^{(1)}$ and epoch $\gamma=e^{(2)}$. They
differ only in epoch $\beta$.

[Definition L7-D3] After epoch $\alpha$ the sensor occupancy writes
$\sigma=o^{\mathrm{S}}(\alpha)$ as in L5. After epoch $\beta$ the
action occupancy writes the internal bit

$$
I
=
o^{\mathrm{A}}(\beta)
\in\{0,1\}.
\tag{L7.2}
$$

Epoch $\beta$ still uses the L5 gate $u^{\mathrm{A}}=\sigma\,u_I(e^{\beta})$.
Epoch $\gamma$ uses the loop gate

$$
u^{\mathrm{A}}
=
I\,u_I(e^{\gamma}).
\tag{L7.3}
$$

The sensor copy always receives $u_I$ of the current flux. No other
channel from $I$ into $(m,b,q)$ is authorized. $I$ is a named bit, not
a third cube.

[Definition L7-D4] The feedforward control freezes $\sigma$ and
ignores $I$. Epoch $\gamma$ uses $u^{\mathrm{A}}=\sigma\,u_I(e^{\gamma})$.

[Definition L7-D5] The readout of a task is the action occupancy
$o^{\mathrm{A}}$ after epoch $\gamma$.

## 4. Claims

[Open theorem L7-E1] On the loop law (L7.2)--(L7.3), the readout
differs on $\phi^{(1)}$ and $\phi^{(2)}$. Hybrid cuts follow the
predecessors: $u=1$ occupancy cites $O$-$E1$; $u=0$ is one-step
extinction on $B_c$.

[Open theorem L7-E2] On the feedforward control, the readout is
identical on $\phi^{(1)}$ and $\phi^{(2)}$.

[Open theorem L7-E3] Therefore the loop family is not equal, as an
operator from $\{\phi^{(1)},\phi^{(2)}\}$ to action occupancy, to
frozen-$\sigma$ feedforward. This is a finite three-epoch
construction. It is not a mouse region loop, not CCF identity, and
not a brain.

[Hypothesis L7-H1] Overwriting $\sigma\leftarrow o^{\mathrm{A}}(\beta)$
on the same two cubes, with no named $I$, also separates
$\phi^{(1)}$ from $\phi^{(2)}$. If that holds, "a third cube is
required" is false. The operator is the update $I\leftarrow o^{\mathrm{A}}$,
not a new body. Routes must compare the named-$I$ loop, frozen
feedforward, and $\sigma$-overwrite, with a killing test.

Autonomy $A$ stays open: the kernel still advances only when `step`
is called. The wash is an external reset.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No mouse identity.
Inherited axioms remain axioms.

### G-DIMENSIONLESS

$E_i,u,\sigma,I,m,b,q$ and every occupancy bit are dimensionless.

### G-MATH

L7-E1 and L7-E2 are theorems on the registered tasks, or killed by a
counterexample. A single numeric pair without the predecessor
enclosures or the $u=0$ extinction arithmetic is not GO.

### G-CODE

After audit only: extend `universe_life_kernel.py` or add a sibling
with no repository imports except stdlib. Keep the extension small.
Do not import V15--V18b, `delayed_linear_credit`, or `runtime`.
Do not write BrainRuntime wiring.

## 6. Decision rules

- `L7 REGION LOOP GO` if L7-E1 and L7-E2 both hold.
- `L7 STOP` if the loop fails to separate the pair, or feedforward
  also separates it.
- Neither verdict is mouse, autonomy, L8, or AGI.
- L7-H1, if true, does not kill E1--E2. It only kills the stronger
  sentence "a third cube region is required."

`10-sources.md` is skipped (no new observations). `12-routes.md` runs
because L7-H1 is open.
