# L3 boxed growth map: nonlinear LAS and T=32 sign split

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-universe-sim-life-climb-20260814`

Mode: light. Do not re-prove the predecessor's cited L2 classification,
host identity, or the deleted $\forall\kappa\in(0,1]$ parent. Cite those
paths only.

All registered quantities are dimensionless. `AGI GO` is forbidden.
L4--L8, BrainRuntime, metric modules, and autonomy $A$ stay out of scope.

## 1. Question

The predecessor left a boxed growth map

$$
r(q)=r_0\bigl(1+\kappa(2q-1)\bigr),\qquad r_0=\frac92,
\tag{N.1}
$$

on the open interval $I_r=(0,86/315)$, with registered point
$\kappa=1/4\in I_r$. Linear Jury numbers at
$(m,b,q)=(7/18,7/16,1/4)$ are positive. Nonlinear local asymptotic
stability and the $T=32$ sign split were not proved.

Question: at the registered point $\kappa=1/4$ and the predecessor's
nominal rationals, does $F_\kappa$ have two still-dividing locally
asymptotically stable points on opposite sides of $q=1/2$, and does the
indicator of remaining in a registered recursive set after $T=32$ (or
the exact division-count on that horizon) depend on
$\operatorname{sign}(q_0-1/2)$, while the $q=1/2$ extinction area stays
at least $1/20$?

## 2. Inherited facts (cite, do not redo)

[Cited theorem N-C1] $F_0$ has no $q\to(m,b)$ channel. Six cube fixed
points; LAS dividing points only at $q\in\{1/4,3/4\}$ for $F_0$.
Source: predecessor P-C1, P-C2 and
`docs/6_뇌/09_생명에서지능까지/08_원시생명_존재정리.md`.

[Cited construction N-C2] The module
`reality_stone/python/reality_stone/clarus/universe_life_kernel.py`
hosts $F_0$ at $\kappa=0$ and admits $\kappa\in I_r\cup\{0\}$.
Predecessor P-C3 / G-HOST.

[Cited hypothesis N-C3] Universal sufficiency of (N.1) on $(0,1]$ is
deleted. The leftover box is $I_r=(0,86/315)$. Predecessor P-H1 after
revise.

## 3. Registered objects

[Definition N-D1] $F_\kappa$ is the predecessor hybrid map with only $r$
replaced by (N.1). Other nominals stay
$\lambda=5/2$, $\rho=1/5$, $\delta=1/10$, $s=1/2$, $\mu=3/32$,
$\eta=1$, $\theta_D=3/4$, $K=1$.

[Definition N-D2] The low-$q$ algebraic point at $\kappa=1/4$ is

$$
Z_-=\Bigl(\frac7{18},\frac7{16},\frac14\Bigr).
\tag{N.2}
$$

The high-$q$ point $Z_+$ is the unique positive dividing fixed point of
$F_{1/4}$ in $q=3/4$ if the math lane exhibits it exactly; otherwise the
lane must compute it from the same quadratic as the predecessor and
record the exact coordinates.

[Definition N-D3] The source recursive rectangle $R_0$ is the one used
in the predecessor route artifact. This run may replace it by a
$\kappa=1/4$-adapted compact set $R_{1/4}$ only if that set is defined
before scoring and does not depend on looking at $T=32$ outcomes.
Target-aware enlargement after seeing trajectories is forbidden.

[Definition N-D4] Horizon $T=32$. The sign split uses initial
$q_0\in\{1/4,3/4\}$ and a registered $(m,b)$ grid that is fixed in
`artifacts/` before evaluation.

## 4. Claims

[Open theorem N-E1] $Z_-$ and $Z_+$ are locally asymptotically stable
for the nonlinear map $F_{1/4}$ on a positive-volume neighborhood in
the cube. Linear Jury is not enough. A Lyapunov function, an invariant
contracting box, or a complete local counterexample is required.

[Open theorem N-E2] There exists a preregistered compact $R$ (either
$R_0$ or a $R_{1/4}$ named in N-D3) such that the $T=32$ occupancy
indicator, or the exact division count on that horizon, differs between
the paired initials $(m,b,1/4)$ and $(m,b,3/4)$ on a nonempty open set
of $(m,b)\in R$. A single pair of trajectories is not a theorem.

[Open theorem N-E3] At $q=1/2$ and $\kappa=1/4$, the immediate
extinction set in the $(m,b)$ plane still has area at least $1/20$.
The predecessor computed area $1/10$ at $q=1/2$ because $r(1/2)=r_0$.
This lane must confirm the same identity rather than assume it.

[Hypothesis N-H1] N-E1 holds at $Z_-$ with the exact Jury values
already recorded. This is a hypothesis to prove or kill.

P-E2 (autonomy) is not in scope and stays open by inheritance.

## 5. Gates

### G-STATUS

No 유도됨 / 제1원리 / 닫힘. No AGI. No L4--L8. Inherited axioms P-A0--P-A2
remain axioms.

### G-DIMENSIONLESS

All of $m,b,q,\kappa,r_0$ and every area, count, and probability are
dimensionless.

### G-MATH

N-E1 is proved or given a local counterexample. N-E3 is an exact area
identity or a counterexample. N-E2 is either a theorem on a
preregistered $R$ or `STOP` for that $R$ (no split).

### G-CODE

Implementation is authorized only after audit, and only as tests or a
narrow helper beside the existing kernel. Do not add new $\kappa$
channels. Do not import V15--V18b or `runtime`.

## 6. Decision rules

- `N-E1 GO` if a nonlinear LAS proof exists for both $Z_\pm$.
- `N-E1 STOP` if a local counterexample exists at either point.
- `N-E2 GO` if a preregistered $R$ splits at $T=32$.
- `N-E2 STOP` if every preregistered $R$ fails to split.
- `L3 COUPLING STILL OPEN` unless N-E1, N-E2, and N-E3 all hold.
  Even then this is a finite boxed construction, not autonomy, not
  evolution, not AGI.

`10-sources.md` and `12-routes.md` are skipped (light follow-up; no new
observations; no new structural candidates).
