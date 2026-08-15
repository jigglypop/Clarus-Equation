# V17 delayed signed-cue route audit

Status: COMPLETE

## 1. Comparison rule

The route target is deliberately narrower than delayed credit assignment in
general: preserve the orientation of one cue until the registered binary
decision. The comparison is lexicographic:

1. expose every cue-correlated state or fixed geometric structure;
2. state the actual covariance group, including any subgroup restriction;
3. solve or fail the exact paired fixture before considering implementation
   convenience;
4. minimize declared added state coordinates or geometric structure without
   disguising a side channel.

Each route below uses at most one new axiom. “Target-aware” means that the
representation or readout was analytically chosen with knowledge of the
registered signed-cue target. It does not mean that a confirmation target or
seed was inspected. No confirmation seed was opened in this lane.

Let $m=d(d+1)/2$. For the registered $d=3$ fixture, $m=6$. These coordinate
counts are a representation ledger, not Shannon-capacity bounds: absent
regularity, finite precision or noise assumptions, exact real entries can
pathologically encode arbitrarily long sign-even histories. The registered
separation is instead conditional and symmetry-based.

## 2. Route matrix

| Rank | Route | Single additional axiom | Declared state/semantic ledger at $d=3$ | Linear-chart covariance group | Target-aware? | Exact fixture result | Primary killing test |
|---:|---|---|---:|---|---|---|---|
| 1 | R-H: homogeneous SPD lift | One dimensionless homogeneous splitting with anchor coordinate $h=1$ is declared and fixed by spatial chart lifts | $G\in\operatorname{SPD}(4)$: 10 ambient real coordinates, **+4** over $g$; the new block is a covector plus scalar | Embedded spatial $GL(3)$ through $A=\operatorname{diag}(J,1)$; not unrestricted $GL(4)$ | Yes; $z_s$ and $y_a$ are analytic task embeddings | Costs $(2,4)$, correct action $a=s$, exact margin 2 | Set $h=0$ or delete the last row/column: the cue outer product becomes sign-even and the decision returns to a tie |
| 2 | R-E: explicit eligibility covector | One persistent cue-correlated covector $e\in V^*$ may be stored | $(g,e)$: $6+3=9$, **+3** | Full $GL(3)$ when $e\mapsto J^{-T}e$ | Yes; terminal readout maximizes $e(au)$ | Scores $(+1,-1)$ in correct/wrong order; margin 2 | Erase $e$, or incorrectly transport it as a vector under a shear; the former restores the tie and the latter breaks chart invariance |
| 3 | R-F: Randers/Finsler direction term | One persistent bounded odd one-form $\beta$ may augment the quadratic geometry | $(g,\beta)$: $6+3=9$, **+3** | Full $GL(3)$ when $\beta\mapsto J^{-T}\beta$ | Yes; the odd term is written from the cue | $F(au)=1-\kappa as$; correct action for every $0<\kappa<1$ | Put $\beta=0$ to recover evenness; set $\lVert\beta\rVert_{g^{-1}}\ge1$ to kill the declared Randers admissibility |
| 4 | R-A: signed original-$g$ with anchor | One nonzero oriented covector anchor $\alpha$ is globally available and transported with charts | Dynamic $g$: 6; disclosed anchor: **+3 components** (normalized form has two continuous parameters plus polarity) | Full $GL(3)$ only for augmented $(g,\alpha)$; a coordinate-fixed anchor leaves only $\operatorname{Stab}(\alpha)$ | Yes; both write and readout use the anchor and one-step baseline | The two $g_s$ differ and an anchor-aware readout recovers $s$ | Hold $\alpha$ fixed under $J=-I$, or demand a plain quadratic comparison of $+u$ and $-u$; each exposes the claimed escape as invalid in that form |
| 5 | R-S: strict original-space $g$ only | None | $g\in\operatorname{SPD}(3)$: 6 total, **+0** | Full $GL(3)$ | No; inherited V16 update | Paired states identical; every common-state policy has the same action law on both branches | Serialize the $+u$ and $-u$ states and compare exactly; any difference identifies covariance failure or a side channel |
| 6 | R-N: recursive strict metric-only SCC copies | Arbitrarily many sign-independent permutation-equivariant copies may be composed | $6N$ raw components for finite $N$, but **+0 cue-odd information** | Diagonal $GL(3)$ with component permutation equivariance | No | Every finite paired tuple and aggregate remains identical | Hash paired tuples for $N\in\{1,2,4,8,16,64\}$ under the same joint seed realization; any advantage requires a sign-correlated joint seed family, role, message or anchor forbidden by D5 |

R-H and R-E form the useful Pareto boundary. R-E adds one fewer real component
and retains the full original-space covariance group; R-H preserves the
registered “one factor field” implementation shape and has a parameter-free
exact construction. Rank 1 is therefore the implementation priority for this
contract, not a claim that it is information-theoretically minimal. Packaging
the extra covector block and scalar inside one SPD factor does not erase their
memory content.

## 3. R-S and R-N: more even tensors do not create orientation

The strict route is not a failed optimizer; it is on the wrong side of a
symmetry. The allowed chart $J=-I$ fixes every covariant two-tensor $g$ while
mapping $x$ to $-x$. Full covariance therefore forces a one-observation update
to be sign-even. For a stochastic update, the final contract imposes this covariance
pointwise on almost every fixed-seed map $U_\omega$, so the paired states agree
for the same seed rather than merely after averaging. The paired construction
also fixes the same $g_0$ and coupled pre-cue seed independently of $s$. In the
exact V16 fixture,

$$
g_1=I+3uu^T
$$

for both signs. An arbitrary terminal policy sees the same metric, public $u$
and common observation, not merely equal quadratic action scores. Its fresh
terminal randomness is jointly independent of $S$, the preceding history,
prior seed family, terminal metric and observation, so it cannot uncouple the
paired equality.

Replication does not change that input equality. At each finite SCC depth,
every component and sign-independent message has an identical paired value.
The public oriented reference $u$ is also identical on the two branches, so
supplying it to the next deterministic or conditionally deterministic
aggregation map preserves equality. The raw state count $Nm$ is therefore a
misleading capacity measure: none of those coordinates is cue-odd on the
registered histories. This induction couples the entire joint seed family
$\Omega=(\omega_i)_i$ to the same realization on both branches and requires
that family, the topology and all initial states to be jointly independent of
$s$; componentwise marginal independence would be insufficient.

For a countable route, equality at all finite prefixes must be paired with a
declared product sigma-algebra, compatible finite-dimensional laws and a
measurable terminal policy. Without that extension, the route supports only a
finite-depth comparison. The route lane does not turn a finite-prefix result
into an infinite-agent claim.

For each fixed public $U=u$, an exact solver must expose two terminal states to
the common policy. In the registered uniform task this is the conditional
requirement

$$
H(S\mid G_T,U)=0,\qquad I(S;G_T\mid U)=1\ \text{bit}.
$$

It is not generally correct to replace this by $I(S;G_T)=1$: isotropic random
$U$ can make the marginal mutual information zero. Nor does the one-bit
conditional lower bound limit the general capacity of exact-real metric
entries.

## 4. R-H: one lifted matrix, one disclosed homogeneous splitting

The homogeneous route stores

$$
G=\begin{pmatrix}Q&b\\b^T&\gamma\end{pmatrix}
\in\operatorname{SPD}(d+1).
$$

The extra $d$-component cross block $b$ is a covector and is exactly where cue
orientation can live; the last scalar supplies the anchor self-coupling. Thus
calling $G$ or its Cholesky representation “one factor field” does not make it
strict original-space metric-only: it adds $d+1$ ambient real state
coordinates and actual memory content under a declared homogeneous splitting.

For the frozen embeddings, the independent calculation gives

$$
y_a^TG_1y_a=2+\frac12(sa-1)^2.
$$

The correct and wrong costs are exactly 2 and 4. Under
$A=\operatorname{diag}(J,1)$, transporting the initial state as
$A^{-T}G_0A^{-1}$ makes the update and both costs invariant without
reprojection. Resetting the transported initial metric to an identity matrix
would be a chart-dependent implementation error.

Killing tests are:

1. delete or zero the homogeneous coordinate and verify the sign-paired state
   equality returns;
2. use a nonorthogonal chart at the singular-value endpoints and require both
   update congruence and action agreement;
3. introspect persistent state and require one SPD factor field of size
   $d+1$, with no separately cached cue vector or optimizer state;
4. perturb the splitting convention or transport by a general $GL(d+1)$ matrix
   mixing the last coordinate, and reject any claim that the task semantics are
   unchanged.

## 5. R-E: the clean minimal geometric comparator

The eligibility route stores $e=g(su)\in V^*$ in addition to $g$. Its
invariant terminal score is

$$
e(au)=as,
$$

so maximizing it recovers the cue with margin 2. This route makes the needed
odd information explicit, uses only $d$ additional components, and retains
full $GL(d)$ covariance through $e\mapsto J^{-T}e$.

Its disadvantage is architectural rather than mathematical: it has two
persistent semantic fields and is therefore not the homogeneous one-SPD
candidate registered for secondary implementation. A scalar eligibility trace
would not be a covariant substitute; extracting a cue sign from a lone nonzero
vector needs an oriented reference, since $GL(d)$ contains $-I$.

Killing tests are field introspection, exact snapshot round-trip of $e$, an
unused scaled shear, and deletion of $e$ immediately before the decision. The
last test must return the strict paired tie.

## 6. R-F: directional geometry is an explicit one-form, not a property of $g$

The Randers route uses

$$
F(v)=\sqrt{v^Tgv}+\beta(v),\qquad
\lVert\beta\rVert_{g^{-1}}<1.
$$

With $\beta=-\kappa g(su)$, $0<\kappa<1$, minimizing $F(au)$ selects
$a=s$. This is a geometrically meaningful directional action cost and may be
useful when later tasks already require asymmetric travel costs.

For this fixture it carries exactly the same $d$ cue-odd components as the
eligibility route, plus an admissibility condition and nonlinear readout. It is
therefore ranked below R-E. It must not be described as extracting direction
from a Riemannian metric: setting $\beta=0$ removes all odd dependence.

Killing tests are an exact $\beta=0$ ablation, boundary values just below and at
$\lVert\beta\rVert_{g^{-1}}=1$, and a general $GL(d)$ linear-chart transform of both the
quadratic term and one-form pairing.

## 7. R-A: a one-matrix appearance with disclosed external structure

A signed rank-one write can be manufactured inside the original metric by
combining the cue covector with an oriented anchor. In normalized fixture
coordinates,

$$
g_s=I+\kappa(w+su)(w+su)^T.
$$

The cross term is odd in $s$, and the anchor-aware statistic

$$
w^T\left[g_s-I-\kappa(ww^T+uu^T)\right]u
=s\kappa\left(1+(w^Tu)^2\right)
$$

recovers the sign. This does not contradict the strict route: $w$ is the
additional orientation structure that the $J=-I$ proof forbids. If $w$ is
hard-coded by coordinates and not transported, the covariance group shrinks to
its stabilizer. If it is transported, the real state/structure is
$(g,\alpha)$, even if only $g$ changes over time.

There are two further costs. First, a plain quadratic policy still cannot rank
$+u$ against $-u$, because every symmetric quadratic form is even. Second, the
displayed readout knows the pre-write baseline; repeated writes need a new
invertibility proof or additional memory. This makes R-A a useful adversarial
audit route, not the preferred implementation route.

## 8. Exact fixture and development calculation

Full derivations, including covariance formulas and the anchor counterexample,
are in `artifacts/route-comparison-calculations.md`. The independent executable
`artifacts/verify_v17_route_fixtures.py` used only development seeds
1,719,000--1,719,063 and reported:

| Check | Result |
|---|---:|
| strict sign-paired serialized arrays | exact equality on all 64 pairs |
| homogeneous action correctness | 128/128 |
| minimum homogeneous wrong-minus-correct margin | $1.9999999999999996$ |
| maximum homogeneous relative chart-cost defect | $2.34\times10^{-15}$ |
| eligibility action correctness | 128/128 |
| Randers action correctness | 128/128 |
| explicit-anchor readout correctness | 128/128 |
| anchor route with plain quadratic readout | tied on all pairs |

These are development checks of exact constructions, not a scored model search
and not confirmation evidence. There is no learning rate or selected
hyperparameter: the homogeneous candidate retains the contract values
$\eta=1$ and $c=4$.

## 9. Dimensionless and target-awareness ledger

All protocol coordinates are declared dimensionless. Therefore $p/c$ and its
logarithm, the homogeneous coordinate, $e(v)$, $\beta(v)$, $F(v)$,
$\kappa$, losses and regrets are dimensionless. This checks formula
compatibility only; it is not evidence that any route models a biological brain
or spacetime.

R-H, R-E, R-F and R-A are target-aware analytic designs because their writes
and readouts use the registered cue/action relation. No route was selected by
development accuracy. R-S is inherited from the V16 general metric learner,
and R-N only replicates it. A future claim about arbitrary cues, delayed reward
assignment, semantic OOD or recursive intelligence growth needs a new contract
and cannot inherit the exact-fixture result.

## 10. Route recommendation boundary

**Implementation priority: R-H, homogeneous SPD lift.**

The reason is narrow: it is the preregistered, exact, one-factor-field escape
and has a fixed margin with no rate search. R-E is the strongest comparator and
is smaller in added real components; it should be preferred instead if full
original-space $GL(d)$ covariance and minimal storage matter more than the
single-factor-field representation. R-F is useful only when a directional
action geometry is independently wanted. R-A should remain a symmetry-breaking
audit fixture. R-S and R-N are non-escape controls.

This ranking is a route recommendation only. It does not assign theorem,
promotion, gate, or closure status to V17-H1 or V17-N1--N4.
