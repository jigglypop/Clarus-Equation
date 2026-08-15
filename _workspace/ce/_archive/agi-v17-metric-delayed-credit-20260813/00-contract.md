# AGI V17 metric-only delayed-cue contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v16-covariant-metric-flow-20260813`

## 1. Question and scope

This run asks whether an agent whose only persistent semantic state is the
original-space SPD metric $g_t$ can preserve the orientation of an earlier cue
until a later decision, while retaining the full $GL(d)$ linear-chart covariance established
by V16. It also asks whether copying that same restriction through arbitrarily
many recursive SCC agents can recover information that every component loses.

The primary result sought is a theorem or a counterexample, not a performance
claim. A secondary implementation may test the smallest geometric escape from
the no-go. `AGI GO` is forbidden. The strongest permitted positive conclusion
is a narrow delayed signed-cue memory primitive.

External literature, biological measurements and cosmological observations are
outside this run. The tangent vectors, costs, rewards and coordinates below are
dimensionless synthetic quantities. The metric $g_t$ is not the spacetime
metric $g_{\mu\nu}$ in `docs/axium.md`.

## 2. Definitions

[Definition V17-D1] Let $V=\mathbb R^d$, $d\ge1$, and
$\operatorname{SPD}(V^*)$ be the positive-definite symmetric covariant
two-tensors. A strict metric-only controller has one persistent semantic state
$g_t\in\operatorname{SPD}(V^*)$. It has no persistent vector, covector,
orientation bit, eligibility trace, replay buffer, recurrent hidden state,
retained random seed, agent identifier correlated with the task, or
role-specific weight. This restriction does not imply a general capacity bound:
without regularity, precision or noise assumptions, exact real entries of one
SPD tensor can pathologically encode arbitrarily long sign-even histories.

[Definition V17-D2] A one-observation metric update is a deterministic map

$$
U:\operatorname{SPD}(V^*)\times(V\setminus\{0\})\times\mathbb R_{>0}
\longrightarrow\operatorname{SPD}(V^*). \tag{V17.1}
$$

It is fully $GL(d)$-covariant when, for every $J\in GL(d)$,

$$
U(J^{-T}gJ^{-1},Jx,c)=J^{-T}U(g,x,c)J^{-1}. \tag{V17.2}
$$

The scalar $c$ and the ratio $p/c$, $p=x^Tgx$, are dimensionless. A stochastic
update is represented by deterministic maps $U_\omega$ conditional on a random
seed $\omega$ independent of the cue sign. For almost every fixed $\omega$, the
same map $U_\omega$ must satisfy V17.2 pointwise for every $J$; covariance only
after averaging over $\omega$ is insufficient. All conclusions then apply for
the same fixed seed and after averaging over that seed.

[Definition V17-D3] A terminal policy is metric-only when its action law is a
function only of the current metric, the common terminal observation and fresh
randomness $R_T$ satisfying
$R_T\perp\!\!\!\perp\sigma(S,H_{<T},\Omega,G_T,O_T)$. Here $\Omega$ is the
entire prior seed family and $H_{<T}$ is the preceding history. No earlier
observation or prior seed is supplied again at decision time.

[Definition V17-D4] In the balanced delayed signed-cue task, an oriented unit
reference $u\in V$ and a sign $s\in\{-1,+1\}$ are drawn, with $s$ uniform.
The tuple $(U,\Omega,g_0)$ of public reference, entire update seed family and
initial metric is jointly independent of $s$. The same $u$, including its orientation, is
public to both sign branches and is supplied again in their identical terminal
observation/action embeddings. The terminal observation $O_T$ is a
deterministic function only of this public $u$ and the fixed action ordering;
it contains no additional random key or context. The cue is $x_0=su$ and its positive cost is
identical for both signs. All later predecision observations and costs are
paired and sign-independent. At the terminal time the common action set is
$a\in\{-1,+1\}$ and the unique correct action is $a=s$. The $0$--$1$ loss and
regret are $1[a\ne s]$. The reward is revealed only after the action. Public
$u$ fixes how action labels are embedded but carries no information about $s$.
Paired branches use the same realization $(u,\Omega,g_0)$ at every step.

[Definition V17-D5] A recursive metric-only SCC system contains any finite or
countable collection of components satisfying D1--D3. The tuple containing the
public reference $U$, entire joint seed family $\Omega=(\omega_i)_i$, initial
topology and all initial states is jointly independent of $s$; paired branches
use the same realization of this entire tuple. Each component consumes the signed cue only
through its pointwise-covariant update before any persistent storage or
communication. Thereafter, communication and aggregation are deterministic or
conditionally deterministic, permutation-equivariant, and built only from the
components' metric states, the common public reference $u$, and paired
sign-independent messages. No branch receives a cue-correlated private anchor,
raw signed cue, or sign-correlated random seed through a side channel.

## 3. Claims to prove or kill

### N1. Full-GL sign-blindness

[Open theorem V17-N1] Every update satisfying V17.2 obeys

$$
U(g,-x,c)=U(g,x,c). \tag{V17.3}
$$

The proof candidate must explicitly use the allowed chart $J=-I$ and must cover
randomized updates under the independence condition in D2.

### N2. Delayed signed-cue impossibility

[Open theorem V17-N2] Under D1--D4, paired $+u$ and $-u$ histories reach the
same decision state. Every metric-only terminal policy therefore has balanced
success probability exactly $1/2$ and expected regret $1/2$. The statement is
about this registered aliasing task, not every delayed-credit problem.

### N3. Recursive-agent no rescue

[Open theorem V17-N3] Under D5, increasing the number or nesting depth of
metric-only SCC components cannot raise balanced success above $1/2$. A proof
must be by a finite-depth induction followed, if a countable limit is claimed,
by an explicitly stated compatibility/measurability argument. A finite-prefix
result may not be called an infinite-agent theorem.

### N4. Minimum information separation

[Open theorem V17-N4] For every fixed public oriented reference $u$, any exact
solver of D4 must expose two distinguishable terminal states to the common
terminal policy. Equivalently for uniform $S$, exact solution requires
$H(S\mid G_T,U)=0$ and hence $I(S;G_T\mid U)=1$ bit. The marginal
$I(S;G_T)$ may be zero when $U$ is isotropically randomized. This is a
conditional information lower bound only. It neither bounds the general
capacity of exact-real SPD entries nor proves that a continuous implementation
needs exactly one extra scalar degree of freedom.

### H1. Homogeneous one-SPD escape

[Hypothesis V17-H1] Introduce one declared homogeneous coordinate and store a
single $G_t\in\operatorname{SPD}(d+1)$. This is one matrix field but is not the
strict original-space $g_t$ of D1; it adds $d+1$ ambient real state coordinates
and a declared homogeneous-splitting axiom. In block form its new entries are a
covector and a scalar packaged in the same SPD factor, not evidence that no
additional memory content was introduced. With

$$
z_s=(su,1),\qquad y_a=(au,-1),\qquad G_0=I_{d+1}, \tag{V17.4}
$$

one V16 update at $\eta=1$, prediction $p=2$ and observed cost $c=4$ should
give

$$
G_1=I_{d+1}+\frac12 z_sz_s^T, \tag{V17.5}
$$

and the minimum quadratic terminal cost should select $a=s$. Under a linear
spatial chart $J$, the lift must use $A=\operatorname{diag}(J,1)$ and
$G\mapsto A^{-T}GA^{-1}$ without reprojection.

### H2. Other escape routes

[Hypothesis V17-H2] The route lane must compare at least the following
structurally distinct escapes: an explicit eligibility covector, a Randers or
other directional geometric term, a homogeneous SPD lift, and a signed update
of the original $g$. Each route must count added semantic degrees of freedom,
state its single new axiom, and give a killing test. A signed original-$g$
update that uses a fixed anchor or orientation is not strict D1 and must expose
that extra structure.

## 4. Exact and randomized fixtures

The exact fixture uses $d=3$, arbitrary Euclidean-unit $u$, both signs, $G_0=I$
and equations V17.4--V17.5. The reference costs are

$$
y_s^TG_1y_s=2,\qquad y_{-s}^TG_1y_{-s}=4. \tag{V17.6}
$$

The strict V16 state starts at $g_0=I_3$, observes $x=su$ and common cost $4$,
and is required to expose that its two sign-paired states are identical. A
terminal policy receiving only either state and a common terminal observation
must use the same action distribution in both branches.

Development seeds are 1,719,000--1,719,063. Confirmation seeds are
1,720,000--1,720,255 and are opened once after the production implementation,
evaluator, exact formulas, thresholds and any selected representation are
SHA-256 sealed. Each seed draws a nonzero standard-normal $u$ and normalizes it,
then evaluates both signs. It also draws an unused chart $J$ from two signed-QR
orthogonal matrices and singular values log-uniform on $[0.25,4]$. The exact QR
sign convention is positive when the diagonal is zero. One $J$ is used per
pair, with no resampling except for a zero cue draw.

No rate search is allowed: $\eta=1$, $c=4$ and the embeddings in V17.4 are
fixed analytically. Development may debug implementation and sealing only.

## 5. Preregistered gates

### G-MATH

V17-N1, N2 and N4 must close with no P0. N3 may close either as a finite-depth
theorem plus a separately justified countable extension, or be explicitly
narrowed to finite depth. H1's exact costs must be derived independently from
the production code.

### G-DIMENSIONLESS

$p/c$, $\log(p/c)$, losses, regrets, cue coordinates and the homogeneous
coordinate must all be dimensionless in the synthetic protocol. Passing this
gate is dimensional consistency, not physical evidence.

### G-STRICT-NO-GO

For all 256 paired confirmation seeds, strict original-space V16 must have:

- exact serialized state equality between $+u$ and $-u$ branches;
- action-distribution equality by construction;
- paired balanced accuracy $0.5$ and regret $0.5$;
- the same result for registered finite ensemble sizes
  $N\in\{1,2,4,8,16,64\}$ with sign-independent permutation-equivariant
  aggregation.

This gate passing confirms the registered no-go; it is not a positive
capability score.

### G-LIFT

For all 256 paired confirmation seeds, the homogeneous one-SPD candidate must
have:

- finite-run rate $1.0$;
- exact action accuracy $1.0$ and regret $0$;
- minimum wrong-minus-correct predicted cost margin at least $1.999999999$;
- original-versus-transported action agreement $1.0$;
- maximum relative quadratic-cost defect at most $10^{-10}$;
- one persistent factor field, no optimizer state, and exactly
  $(d+1)(d+2)/2=10$ ambient real state coordinates for $d=3$, four more than
  the original metric.

### G-NUMERIC

Focused tests must include $u$ near coordinate axes, both signs, singular-value
endpoints for $J$, snapshot exactness, nonfinite/zero rejection, state-field
introspection and a killing test showing that dropping the homogeneous
coordinate returns the $1/2$ no-go. Public outputs in the declared binary64
domain must be finite.

## 6. Decision rules

- `V17 METRIC-ONLY NO-GO CLOSED`: G-MATH, G-DIMENSIONLESS,
  G-STRICT-NO-GO and G-NUMERIC pass.
- `V17 HOMOGENEOUS LIFT NARROW GO`: H1 is implemented and G-LIFT plus
  G-NUMERIC pass.
- `V17 STOP`: a required theorem, integrity control or candidate gate fails.
- Neither positive label means delayed credit assignment in general. The lift
  only solves one delayed signed-cue memory task with an immediate analytic
  write.
- `AGI GO`, biological fidelity, cosmological identity, consciousness, tool
  use, semantic OOD and infinite-SCC intelligence growth are forbidden
  conclusions.

## 7. Contamination and reproducibility controls

The confirmation evaluator must be independent of the production route helper
for reference costs and decisions. Before confirmation, one SHA-256 manifest
must cover this contract, production module, public export, evaluator,
development result and all gate thresholds. The evaluator must bind to the
canonical repository root and imported production module, validate every path
against traversal, and exclusively create an opening receipt before touching
seed 1,720,000. A result file and receipt make any second opening fail closed.

Per-seed paired summaries must be stored in the first confirmation result so
that paired episode statistics can be calculated without reopening the block.
Any post-seal change to a bound artifact invalidates this run and requires a
new run and seed block.
