# AGI universe-simulator life-climb program contract

Status: COMPLETE

PREDECESSOR: none

This run does not continue `_workspace/ce/agi-v18b-learned-delayed-credit-20260814`.
That run is a synthetic delayed-credit fixture. This run freezes a different
spine: a universe kernel that hosts primordial self-maintenance and climbs by
adding operators only when the previous internal state is no longer a
sufficient statistic of the niche.

Cited closed mathematics lives in
`docs/6_뇌/09_생명에서지능까지/08_원시생명_존재정리.md` and
`docs/6_뇌/09_생명에서지능까지/09_원시생명_증명가능명제_총정리.md`.
Those documents are sources, not a CE research predecessor run. This contract
does not re-prove their twelve exact obligations.

All registered quantities are dimensionless. `AGI GO` is forbidden.

## 1. Question and authorized scope

[Definition P-D0] A finite universe kernel $U$ is a single discrete step that
updates exogenous flux, every registered subsystem, and an optional residual
field. No agent API, token head, or external reward argument is part of $U$.

Question: can the existing primordial hybrid map be hosted as a subsystem of
such a $U$, and can one coupling gain make the transmitted label change
growth or boundary so that descendant count depends on that label, while the
already proved extinction wedge stays positive?

This run freezes the program ladder L0--L8 as a plan and authorizes only
L0--L3 construction. It does not implement routing, activity closure, region
loops, BrainRuntime wiring, metric flow, delayed-credit learners, SCC towers,
language, tools, or an internal world model $K$.

[Axiom: model choice P-A0] The universe is the unique step. Life is a
subsystem of that step. AGI, if the word is ever used later, would be a
subsystem whose internal kernel $K$ emits the same kind of variables as $U$.
P-A0 is a design axiom, not a theorem and not a cosmology derivation.

[Axiom: model choice P-A1] Intelligence is not a parameter count. An operator
is added only when the previous state is not a sufficient statistic of the
niche. This restates the question already posed in
`docs/6_뇌/09_생명에서지능까지/01_개요와공통식.md`.

[Axiom: model choice P-A2] Occupancy ratios
$(0.0487, 0.262, 0.689)$ are not copied into $U$, into any subsystem, or into
a sparsity mask. The map $q_{\mathrm{ext}}\mapsto\Omega_b$ remains the
cosmology-branch axiom of `docs/3_상수/3_부트스트랩.md` and is outside this
run.

## 2. Program ladder (plan, not a claim of closure)

[Definition P-D1] The program uses eight layers. Only L0--L3 are in scope.

| Layer | Name | Operator added | This-run status |
|---|---|---|---|
| L0 | universe kernel | one `step` | construction authorized |
| L1 | open chemistry | flux and leak | hosted by L0 |
| L2 | protocell | auto / boundary / copy | cite existing exact map |
| L3 | autonomous lineage | label $\to$ phenotype $\to$ descendant count | open construction |
| L4 | weighted routing | $\mathcal L(W)$ | out of scope |
| L5 | role split | cell type / action / memory | out of scope |
| L6 | activity closure | $P$ predicts $P$ | out of scope |
| L7 | region loop | decision / action / internal | out of scope |
| L8 | internal kernel $K$ | $\hat U = K(Z)$ | out of scope |

[Definition P-D2] Symbols that collide across documents are split.

- $q_{\mathrm{label}}\in[0,1]$ is the transmitted label of the primordial map.
- $q_{\mathrm{homeo}}$ is the homeostasis variable of the common climb
  equation. It does not appear in this run.
- $q_{\mathrm{ext}}$ is the Poisson extinction probability. It does not
  appear in this run.

$q_{\mathrm{label}}$ is not a gene. That identification is already refuted as
interpretation in `09_원시생명_증명가능명제_총정리.md`.

## 3. L2 source map (cited, not re-proved)

[Definition P-D3] The source state is $Z_t=(m_t,b_t,q_t)\in[0,1]^3$ with
$q_t:=q_{\mathrm{label}}$. The source map $F_0$ is the hybrid map of
`08_원시생명_존재정리.md`:

$$
\widetilde m_t
=
\left[
m_t\left\{1+r\left(1-\frac{m_t}{K}\right)-\lambda(1-b_t)\right\}
\right]_+,
\tag{P.1}
$$

$$
d_t=\mathbf 1[\widetilde m_t\ge\theta_D],
\qquad
m_{t+1}=\frac{\widetilde m_t}{2^{d_t}},
\tag{P.2}
$$

$$
b_{t+1}=(1-\delta)b_t+\rho m_t(1-b_t),
\tag{P.3}
$$

$$
q_{t+1}
=\frac12+\eta\left[
q_t+s q_t(1-q_t)(2q_t-1)+\mu(1-2q_t)-\frac12
\right].
\tag{P.4}
$$

Nominal parameters are the source document's rationals
$r=9/2$, $\lambda=5/2$, $\rho=1/5$, $\delta=1/10$, $s=1/2$, $\mu=3/32$,
$\eta=1$, $\theta_D=3/4$, $K=1$.

[Claim P-C1: citation] Under those parameters the source document's
classification stands: six cube fixed points; exactly three positive
every-step dividing states; local asymptotic stability only at
$q\in\{1/4,3/4\}$; a positive-volume recursive basin and a positive-area
extinction wedge. This run treats P-C1 as a cited theorem, not a new proof.
The math lane checks that the citation is faithful and that $q$ does not
enter (P.1)--(P.3).

[Claim P-C2: no evolution] Because $q$ is absent from (P.1)--(P.3), $F_0$
does not implement genotype--phenotype coupling or natural selection. This
is already recorded. The math lane must not upgrade $F_0$ to an evolution
theorem.

## 4. L0 kernel

[Definition P-D4] A registered L0 state is the frozen tuple

$$
U_t=(t,E_t,\mathcal Z_t,\phi_t),
\tag{P.5}
$$

where $t\in\mathbb N$ is the tick, $E_t\in[0,1]$ is a single chemostat flux,
$\mathcal Z_t$ is a finite list of subsystem states, and
$\phi_t\in[0,1]$ is a residual scalar. For this run $\phi_t$ may stay $0$.
Opening a nonzero residual law is out of scope.

[Definition P-D5] The chemostat law is

$$
E_{t+1}=\bigl[(1-\nu)E_t+\nu E_\star\bigr]_0^1,
\tag{P.6}
$$

with registered constants $\nu=0$ and $E_\star=1$ unless a later claim
opens depletion. Default $E_t\equiv 1$ recovers the source chemostat.

[Definition P-D6] Every subsystem exposes `step(E) -> Z`. The kernel applies
the same $E_t$ to every subsystem and does not pass a reward, label, or
marker. Hosting $F_0$ means the unique subsystem is $Z_t$ and
$Z_{t+1}=F_0(Z_t)$ while $E_t=1$.

[Claim P-C3: host] There exists a finite implementation of (P.5)--(P.6)
whose one-subsystem trajectory with $E_t\equiv 1$ is byte-identical to
iterating $F_0$ on the same initial $Z_0$ for every registered test point.
This is a construction claim, not a biology claim.

## 5. L3 coupling (open)

[Definition P-D7] A coupling gain $\kappa\in[0,1]$ yields a map $F_\kappa$
on the same cube that satisfies $F_0=F_{\kappa=0}$ pointwise.

[Open theorem P-E1] There exists a Lipschitz (equivalently, a rational
piecewise) $F_\kappa$, a nonempty open interval $I\subset(0,1]$, and a
nonempty open parameter box around the nominal point, such that for every
$\kappa\in I$ in that product box:

1. $q$ enters the pre-division mass or the boundary update;
2. the two source LAS dividing points persist, or move to two still-dividing
   LAS points whose $q$-coordinates remain on opposite sides of $1/2$;
3. mean descendant count, or the indicator of remaining in the recursive
   basin after a registered horizon $T=32$, depends on the initial
   $q_0$ sign relative to $1/2$;
4. the extinction set still has area at least $1/20$ in the $(m,b)$ plane
   at $q=1/2$.

The parent reading that took the $\kappa$-box to be all of $(0,1]$ is
deleted. $I=(0,1]$ is not authorized.

[Hypothesis P-H1] Write $r_0=9/2$. The growth modulation

$$
r(q)=r_0\bigl(1+\kappa(2q-1)\bigr)
\tag{P.7}
$$

is a hypothesis on the open interval $I_r=(0,86/315)$, which contains the
registered point $\kappa=1/4$. It is not a sufficient construction for
P-E1 on $(0,1]$. The points $\kappa=1/2$ and $\kappa=1$ are killing tests
for this parent, not members of $I_r$. Nonlinear LAS and P-E1.3 remain
open even on $I_r$. This is not a theorem of life.

[Hypothesis P-H2] Write $\rho_0=1/5$. The boundary-source modulation

$$
\rho(q)=\rho_0\bigl(1+\kappa(2q-1)\bigr)
\tag{P.8}
$$

is an alternative hypothesis on $\kappa\in(0,86/87)$. The endpoint
$\kappa=1$ is a killing test. On the source rectangle $R_0$, bullet 3 is
weak: $T=32$ occupancy need not split. Routes may still compare (P.7),
(P.8), and leak $\lambda(q)$ with degrees of freedom and a killing test.
Two-daughter survival is not an authorized P-E1 candidate in this run
(it needs an extra axiom P-A3, which is not adopted). Threshold-only
maps remain excluded by bullet 1.

[Open theorem P-E2] Autonomy $A$ in the source conjunction
$C_{\mathrm{strict}}=G\land D\land R\land H\land V\land A\land M$ is not
implied by P-E1. A coupled map that still advances only when an external
caller invokes `step` has not produced experimenter-free lineage.

## 6. Claims the lanes must judge

| ID | Statement | Authorized status if closed |
|---|---|---|
| P-A0--P-A2 | design axioms | axiom |
| P-C1 | faithful citation of the L2 classification | cited theorem or P0 mis-citation |
| P-C2 | $F_0$ has no $q\to(m,b)$ channel | cited theorem |
| P-C3 | $U$ can host $F_0$ with identical trajectories | construction / 산출 |
| P-E1 | some $F_\kappa$ meets the four bullets on an open $I\subset(0,1]$ | open; not $\forall\kappa\in(0,1]$ |
| P-E2 | coupling $\neq$ autonomy | open; do not collapse |
| P-H1 | $r(q)$ on $I_r=(0,86/315)$ | hypothesis ($\kappa$-boxed); killed as universal sufficient |
| P-H2 | $\rho(q)$ on $(0,86/87)$ | hypothesis; $R_0$ bullet 3 weak |

No claim in this table is AGI, cosmology, historical first life, or
universal necessity of the three life terms.

## 7. Gates

### G-STATUS

P-A0--P-A2 remain axioms. P-C1 and P-C2 are not rewritten as new first-party
proofs. P-E2 stays open. The deleted P-H1 parent
($\forall\kappa\in(0,1]$ sufficient for P-E1) must not re-enter. No
sentence uses 유도됨, 제1원리, or 닫힘 for P-A0 or L4--L8.

### G-DIMENSIONLESS

$m,b,q,E,r,\lambda,\rho,\delta,s,\mu,\eta,\theta_D,K,\kappa,\nu$ and every
probability, area, and count ratio are dimensionless. Any new logarithm or
exponential takes a positive dimensionless argument.

### G-HOST

If implementation is authorized after audit, a kernel module with no
repository imports except the standard library and NumPy, plus tests that
compare `kernel.step` against a local copy of $F_0$ on a registered grid,
must show identical states to absolute tolerance $10^{-15}$ for at least
the eight corners of $[0,1]^3$ and the six source fixed points, for $T=8$
ticks.

### G-COUPLE

A candidate $F_\kappa$ may be coded only if the math and route lanes leave
no P0 against that candidate's algebra and a killing test is written. The
coded candidate must fail the $q$-dependence bullet at $\kappa=0$ and pass
it at one in-box registered point. For P-H1 that point is $\kappa=1/4$;
$\kappa\in\{1/2,1\}$ remain killing tests and must not be claimed inside
$I_r$. Extinction area below $1/20$ is `STOP` for that candidate, not a
license to drop the wedge.

## 8. Decision rules

- `LIFE-CLIMB PLAN FROZEN` requires G-STATUS and G-DIMENSIONLESS.
- `L0 HOST CONSTRUCTION` requires G-HOST.
- `L3 COUPLING CONSTRUCTION` requires G-COUPLE and a surviving route with
  no open P0 on P-E1 for that route.
- `L3 COUPLING STOP` if every registered candidate kills P-E1 or the
  extinction wedge.
- Neither positive verdict authorizes L4--L8, BrainRuntime promotion,
  metric-flow reuse, delayed credit, biological identity, cosmology
  occupancy, $C_{\mathrm{strict}}$, or AGI.

V15--V18b modules remain a later L8 parts drawer. This run must not import
them.

## 9. Implementation bound

After `check gate` only, authorized new code is a self-contained kernel
module under `reality_stone/python/reality_stone/clarus/` and matching
tests. The module may contain $F_0$ and at most one $F_\kappa$ named in
the surviving route. It must not call `runtime`, `unified_metric`,
`covariant_metric_flow`, `delayed_linear_credit`, or `nested_scc_tower`.

Canonical `docs/7_AGI/` files are not edited in this run. The plan lives
here. A later `ce-doc-write` pass may cite the final report.

## 10. Sources and skipped observations

This contract cites existing CE documents only. It introduces no new
observational central values, covariance, or PDG/Planck numbers.
`10-sources.md` is therefore skipped.
