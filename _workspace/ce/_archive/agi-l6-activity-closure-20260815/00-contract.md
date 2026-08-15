# L6 activity closure: $P$ predicts $P$ on the wash+$\sigma$ host

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l5-role-split-20260814`

Mode: full for a new operator. Cite L0--L5. Do not re-prove
$N$-$E1$, $N$-$E3$, $O$-$E1$, $L4$-$E1$--$E3$, or $L5$-$E1$--$E3$.

All registered quantities are dimensionless. `AGI GO` is forbidden.
Autonomy $A$, zebrafish / Drosophila / C. elegans identity,
BrainRuntime, L7--L8, and metric modules stay out of scope.

[Axiom: model choice L6-A0] "Living lineage" means the boxed L3 host
plus `L4 WEIGHTED SEPARATION GO` and `L5 ROLE SPLIT GO`. It does not
mean experimenter-free autonomy. $C_{\mathrm{strict}}$ remains open.

[Axiom: model choice L6-A1] An operator is added only when the previous
state is not a sufficient statistic of the registered niche. On a
single washed start, $(\sigma,E)$ already determines the whole
trajectory $P_t$. L6 therefore varies the activity state inside a
preregistered pair at fixed $\sigma$. It does not copy zebrafish
calcium names into the kernel.

## 1. Question

L5's named bit $\sigma$ is a sufficient statistic of the wash niche
for the action occupancy $o^{\mathrm{A}}$. After a wash to one
registered point, $(\sigma,E)$ is also a sufficient statistic of the
cube trajectory.

The program name of L6 is $P$ predicts $P$: the activity state, not
the bit, predicts the next activity state
(`docs/6_뇌/09_생명에서지능까지/01_개요와공통식.md`).

Question: do there exist two registered points of $U_0\times\{3/4\}$
with the same $\sigma$ and the same drive such that the one-step
activity $P\mapsto P'$ differs, while every map that sees only
$\sigma$ assigns both points the same next $P'$?

## 2. Inherited facts (cite, do not redo)

[Cited theorem L6-C1] $Z_\pm$ are nonlinearly LAS. $N$-$E1$.

[Cited 산출 L6-C2] Extinction area at $q=1/2$ is $1/10$. $N$-$E3$.

[Cited theorem L6-C3] On $U_0$, $T=32$ occupancy splits by
$q_0\in\{1/4,3/4\}$. $O$-$E1$.

[Cited theorem L6-C4] $W=I$ occupancy pairs and $A_{\mathbf 1}$
non-separation. $L4$-$E1$--$E3$.

[Cited theorem L6-C5] Wash+$\sigma$ separates $\tau^{(1)},\tau^{(2)}$
in $o^{\mathrm{A}}$; no-store wash does not. $L5$-$E1$--$E3$.

[Cited 미완성 L6-C6] $L5$-$H1$ no-wash continuation is not a theorem.

[Cited 기각 L6-C7] "Named bit is the only sufficient statistic" is
not an active claim.

## 3. Activity pair

[Definition L6-D1] Activity of one copy is the cube state
$P=(m,b,q)\in[0,1]^3$. The one-step map is the predecessor $F_{1/4}$
with drive (L4.1). No new channel from $E$ or $\sigma$ into $(m,b,q)$
is authorized beyond L5.

[Definition L6-D2] Registered points, written before evaluation, are
the L5 center and one other interior point of $U_0$:

$$
P_{\star}
=
\Bigl(\tfrac12,\tfrac{49}{99},\tfrac34\Bigr),
\qquad
P_{\circ}
=
\Bigl(\tfrac{7}{15},\tfrac{49}{99},\tfrac34\Bigr).
\tag{L6.1}
$$

Both lie in $U_0\times\{3/4\}$. Both carry the same registered bit
$\sigma=1$ and the same drive $u=1$.

[Definition L6-D3] The activity readout is the one-step pair
$(m',b')$. Label $q'$ is not an observable of this run.

[Definition L6-D4] A bit predictor is any map
$\{0,1\}\to[0,1]^2$ that ignores $P$ and returns one pair
$(m',b')$ from $\sigma$ alone.

## 4. Claims

[Open theorem L6-E1] $F_{1/4}(P_{\star})\neq F_{1/4}(P_{\circ})$ in
$(m',b')$ at $u=1$. Exact Fraction arithmetic. No $T=32$ hull and no
global continuity.

[Open theorem L6-E2] Every bit predictor assigns the same
$(m',b')$ to $P_{\star}$ and $P_{\circ}$. Therefore no bit predictor
equals the true one-step map on this pair.

[Open theorem L6-E3] The one-step activity map on the registered pair
is not a function of $\sigma$. This is a finite pair construction.
It is not zebrafish activity closure, not calcium identity, and not
a brain.

[Hypothesis L6-H1] $T=32$ occupancy on this pair is the same
(both $1$, by $O$-$E1$ at $q=3/4$, $u=1$). If that holds, activity
closure is $P\to P$ and is not an occupancy split. Routes must
compare one-step $(m',b')$, occupancy, and at least one recurrent
drive $u_t=\sigma m_t$, with a killing test.

Autonomy $A$ stays open: the kernel still advances only when `step`
is called.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No zebrafish identity.
Inherited axioms remain axioms.

### G-DIMENSIONLESS

$E_i,u,\sigma,m,b,q$ and every occupancy bit are dimensionless.

### G-MATH

L6-E1 is an exact inequality on the registered pair, or killed by
equality. A float pair is not GO.

### G-CODE

After audit only: extend `universe_life_kernel.py` or add a sibling
with no repository imports except stdlib. Do not import V15--V18b,
`delayed_linear_credit`, or `runtime`. Do not write BrainRuntime
wiring.

## 6. Decision rules

- `L6 ACTIVITY CLOSURE GO` if L6-E1 and L6-E2 both hold.
- `L6 STOP` if the registered pair has the same one-step $(m',b')$,
  or a bit predictor matches both true next states.
- Neither verdict is zebrafish, autonomy, L7, or AGI.
- L6-H1, if true, does not kill E1--E2. It only kills the stronger
  sentence "activity is required for occupancy on $U_0$."

`10-sources.md` is skipped (no new observations). `12-routes.md` runs
because L6-H1 is open.
