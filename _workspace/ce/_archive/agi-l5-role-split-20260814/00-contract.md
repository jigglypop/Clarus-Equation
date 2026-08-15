# L5 role split on the two-channel L4 host

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l4-weighted-routing-20260814`

Mode: full for a new operator. Cite L0--L4. Do not re-prove
$N$-$E1$, $N$-$E3$, $N$-$E2$/$O$-$E1$, or $L4$-$E1$--$E3$.

All registered quantities are dimensionless. `AGI GO` is forbidden.
Autonomy $A$, C. elegans / Drosophila identity, BrainRuntime, L6--L8,
and metric modules stay out of scope.

[Axiom: model choice L5-A0] "Living lineage" means the boxed L3 host
plus the predecessor's finite two-channel construction
`L4 WEIGHTED SEPARATION GO`. It does not mean experimenter-free
autonomy. $C_{\mathrm{strict}}$ remains open.

[Axiom: model choice L5-A1] An operator is added only when the previous
state is not a sufficient statistic of the registered niche
(`docs/6_뇌/09_생명에서지능까지/01_개요와공통식.md`). L5 asks whether
a wash niche forces a role bit. It does not copy fly cell-type names
into the kernel.

## 1. Question

L4's two copies share one update law. Their occupancy pair after one
window of length $T=32$ already separates $e^{(1)}$ from $e^{(2)}$ when
$W=I$. That pair is a sufficient statistic of a **one-window** niche.

The wash niche presents two epochs and resets both copies to the same
registered point of $U_0\times\{3/4\}$ between epochs. After the wash,
the L4 state is again that point plus the current flux. It no longer
records which flux arrived in epoch $\alpha$.

Question: does a registered memory bit $\sigma$ written from epoch-
$\alpha$ occupancy, read as a gate on the action copy in epoch $\beta$,
separate two wash tasks that share the same epoch-$\beta$ flux, while
the no-store L4 control on the same wash fails to separate them?

## 2. Inherited facts (cite, do not redo)

[Cited theorem L5-C1] $Z_\pm$ are nonlinearly LAS. $N$-$E1$.

[Cited 산출 L5-C2] Extinction area at $q=1/2$ is $1/10$. $N$-$E3$.

[Cited theorem L5-C3] On $U_0$, $T=32$ occupancy splits by
$q_0\in\{1/4,3/4\}$. $O$-$E1$.

[Cited theorem L5-C4] $W=I$ sends $e^{(1)},e^{(2)}$ to occupancy pairs
$(1,0)$ and $(0,1)$. $A_{\mathbf 1}$ does not separate them.
$L4$-$E1$--$E3$.

[Cited 기각 L5-C5] $L4$-$H1$ and "must be weighted" are not active
claims.

## 3. Wash niche and roles

[Definition L5-D1] One body is a pair $(Z^{\mathrm{S}},Z^{\mathrm{A}})$
of cube states with the predecessor $F_{1/4}$ law and drive (L4.1).
$W=I$ is the only registered router. Both copies start each epoch at
the same registered point of $U_0\times\{3/4\}$.

[Definition L5-D2] Registered fluxes are the predecessor pair

$$
e^{(1)}=(1,0),\qquad e^{(2)}=(0,1).
$$

An epoch is $T=32$ steps of one flux. A wash task is an ordered pair
$(\alpha,\beta)\in\{e^{(1)},e^{(2)}\}^2$. Between epochs both copies
are reset to the registered start. The registered pair is

$$
\tau^{(1)}=(e^{(1)},e^{(2)}),\qquad \tau^{(2)}=(e^{(2)},e^{(2)}).
\tag{L5.1}
$$

Both tasks share epoch-$\beta$ flux $e^{(2)}$. They differ only in
epoch $\alpha$.

[Definition L5-D3] Occupancy of a copy is the predecessor bit
$\mathbf 1[(m_{32},b_{32})\in R_0]$. After epoch $\alpha$ the sensor
copy writes

$$
\sigma
=
o^{\mathrm{S}}(\alpha)
\in\{0,1\}.
\tag{L5.2}
$$

The action copy in epoch $\beta$ receives drive

$$
u^{\mathrm{A}}
=
\sigma\,u_I(e^{\beta}),
\tag{L5.3}
$$

where $u_I$ is the $W=I$ drive of that copy. The sensor copy in epoch
$\beta$ still receives $u_I(e^{\beta})$. No other channel from
$\sigma$ into $(m,b,q)$ is authorized.

[Definition L5-D4] The no-store control uses the same wash and the
same $W=I$ drives, but ignores $\sigma$. Epoch $\beta$ is ordinary L4.

[Definition L5-D5] The readout of a task is the action occupancy
$o^{\mathrm{A}}$ after epoch $\beta$.

## 4. Claims

[Open theorem L5-E1] On the role-split law (L5.2)--(L5.3), the
readout differs on $\tau^{(1)}$ and $\tau^{(2)}$. Hybrid cuts follow
the predecessor: $u=1$ occupancy cites $O$-$E1$; $u=0$ is one-step
extinction on $B_c$.

[Open theorem L5-E2] On the no-store control, the readout is identical
on $\tau^{(1)}$ and $\tau^{(2)}$.

[Open theorem L5-E3] Therefore the role-split family is not equal, as
an operator from $\{\tau^{(1)},\tau^{(2)}\}$ to action occupancy, to
the no-store L4 control. This is a finite wash construction. It is not
Drosophila, not cell-type identity, and not a brain.

[Hypothesis L5-H1] A no-wash continuation of the L4 pair (no reset,
no named $\sigma$) also separates $\tau^{(1)}$ from $\tau^{(2)}$,
because leftover mass already records epoch $\alpha$. If that holds,
"a new bit is the only sufficient statistic" is false. The wash niche
is what forces a named bit. Routes must compare wash+$\sigma$,
no-store wash, and no-wash continuation, with a killing test.

Autonomy $A$ stays open: the kernel still advances only when `step`
is called. The wash is an external reset.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No C. elegans or Drosophila
identity. Inherited axioms remain axioms.

### G-DIMENSIONLESS

$E_i,u,\sigma,m,b,q$ and every occupancy bit are dimensionless.

### G-MATH

L5-E1 and L5-E2 are theorems on the registered tasks, or killed by a
counterexample. A single numeric pair without the predecessor
enclosures or the $u=0$ extinction arithmetic is not GO.

### G-CODE

After audit only: extend `universe_life_kernel.py` or add a sibling
with no repository imports except stdlib. Do not import V15--V18b,
`delayed_linear_credit`, or `runtime`. Do not write BrainRuntime
wiring.

## 6. Decision rules

- `L5 ROLE SPLIT GO` if L5-E1 and L5-E2 both hold.
- `L5 STOP` if the role-split fails to separate the wash pair, or the
  no-store wash also separates it.
- Neither verdict is Drosophila, autonomy, L6, or AGI.
- L5-H1, if true, does not kill E1--E2. It only kills the stronger
  sentence "a named bit is required even without a wash."

`10-sources.md` is skipped (no new observations). `12-routes.md` runs
because L5-H1 is open.
