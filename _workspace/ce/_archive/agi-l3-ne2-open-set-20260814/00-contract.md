# N-E2 open-set occupancy split at T=32

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-l3-nonlinear-las-t32-20260814`

Mode: light. Do not re-prove N-E1 contracting boxes, N-E3 area $1/10$,
the L2 citation, or the deleted $\forall\kappa\in(0,1]$ parent. Cite
those paths only.

All registered quantities are dimensionless. `AGI GO` is forbidden.
L4--L8, BrainRuntime, metric modules, and autonomy $A$ stay out of scope.

## 1. Question

The predecessor proved local nonlinear stability of

$$
Z_-=\Bigl(\frac7{18},\frac7{16},\frac14\Bigr)
$$

and of the exact high-$q$ partner $Z_+$ at $\kappa=1/4$. It also proved
that the $q=1/2$ extinction area is $1/10$. A $5\times5$ grid on the
preregistered rectangle

$$
R_0=\Bigl[\frac25,\frac35\Bigr]\times\Bigl[\frac49,\frac6{11}\Bigr]
$$

split $25/25$ in occupancy after $T=32$, but a finite grid is not an
open set. Interval hulls on $R_0$ and on the a-priori center box $B_c$
did not close, because $F_{1/4}^{32}$ is not globally continuous.

Question: does there exist a nonempty open $U\subset R_0$, named before
any new trajectory is scored, such that for every $(m,b)\in U$ the
$T=32$ occupancy indicator

$$
\mathbf 1\bigl[(m_{32},b_{32})\in R_0\bigr]
$$

or the exact division count on $\{0,\ldots,31\}$ differs between the
paired initials $(m,b,1/4)$ and $(m,b,3/4)$?

## 2. Inherited facts (cite, do not redo)

[Cited theorem O-C1] $Z_\pm$ are nonlinearly LAS for $F_{1/4}$ on
explicit contracting boxes $Q_\pm$. Predecessor N-E1.

[Cited 산출 O-C2] At $q=1/2$, $r=r_0$ and the extinction area is
$1/10$. Predecessor N-E3.

[Cited construction O-C3] The $5\times5$ grid $G\subset R_0$ splits
$25/25$ in occupancy. This is a witness, not a theorem. Predecessor
N-E2 leftover.

[Definition O-D1] $F_{1/4}$ and the nominal rationals are those of the
predecessor. $\kappa=1/4$, $T=32$, occupancy is membership of
$(m_{32},b_{32})$ in $R_0$.

[Definition O-D2] The only allowed scoring sets are the already named

$$
R_0,\qquad
B_c=\Bigl[\frac{13}{30},\frac{17}{30}\Bigr]\times\Bigl[\frac{137}{297},\frac{157}{297}\Bigr]\subset R_0.
$$

A new open $U$ may be used only if it is a geometric subset of $R_0$ or
$B_c$ written in `artifacts/o_e2_preregister.md` before evaluation
(for example an open ball or open box around the center
$(1/2,49/99)$). Target-aware enlargement after seeing images is
forbidden.

## 3. Claims

[Open theorem O-E1] There exists a nonempty open $U\subset R_0$ as in
O-D2 such that occupancy differs for every paired initial in $U$.
Hybrid branch cuts must be controlled: a proof that assumes global
continuity of $F^{32}$ is not authorized.

[Open theorem O-E2] The same $U$ may instead split on exact division
count. Occupancy and count are separate claims; proving one is enough
for `O-E2 GO` in the decision rules below only if the contract
question's "or" is used. Record which observable splits.

[Hypothesis O-H1] $B_c$ itself, or a concentric open sub-box of
one-third the linear scale of $B_c$, is such a $U$ for occupancy.
This is to be proved by interval arithmetic with explicit branch
tracking, or killed by a counterexample point in that box whose pair
does not split.

Autonomy $A$ stays open by inheritance. Closing O-E1 does not close
L3 coupling's remaining biological readings and does not imply
evolution.

## 4. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No L4--L8. Inherited P-A0--P-A2
remain axioms. N-E1 and N-E3 are not re-proved.

### G-DIMENSIONLESS

$m,b,q,\kappa,T$ and every indicator, count, and area are dimensionless.

### G-MATH

O-E1 is a theorem on a preregistered open $U$, or a complete
counterexample in every registered candidate box, or remains 미완성
with the exact obstruction (for example an unresolvable branch cut).
A finer grid alone is not GO.

### G-CODE

After audit only: tests or a narrow helper beside
`universe_life_kernel.py`. No new $\kappa$ channel. No V15--V18b or
`runtime` imports.

## 5. Decision rules

- `N-E2 GO` if O-E1 or the count form O-E2 holds on a preregistered
  nonempty open $U$.
- `N-E2 STOP` if every registered candidate open set contains a pair
  that does not split and the set was claimed as a universal split.
- `N-E2 STILL OPEN` if enclosures do not decide and no counterexample
  kills the named $U$.
- `L3 COUPLING STILL OPEN` unless the predecessor's N-E1, N-E3 and
  this run's N-E2 GO all hold. Even then the result is a finite boxed
  construction, not autonomy, not AGI.

`10-sources.md` and `12-routes.md` are skipped (light follow-up; no new
observations; no new structural map).
