# AGI V17 metric-only delayed cue: independent mathematical verification

Status: COMPLETE

This lane verifies V17-N1--N4, the exact part of V17-H1, and the
dimensionless core directly from final revision 2 of `00-contract.md`.  It does
not infer a general delayed-credit theorem from the registered delayed
signed-cue task, and it does not infer an infinite-agent theorem from a list of
finite ensemble results.

## 1. Verdict and issue register

| Item | Status | Exact boundary |
|---|---|---|
| V17-N1 full-GL sign-blindness | **THEOREM** | Full $GL(d)$ linear-chart covariance, including $J=-I$; conditionally pathwise for allowed randomized updates |
| V17-N2 delayed signed-cue impossibility | **THEOREM** | D1--D4 and a balanced sign; pathwise equality under the common-$\Omega$ coupling, then equality in law |
| V17-N3 finite recursive no rescue | **THEOREM** | Every finite event/nesting depth and every finite number of D5 components |
| V17-N3 countable extension | **THEOREM WITH EXPLICIT REGULARITY** | A bona fide countable product system or projectively compatible finite laws, with measurable maps and terminal kernel |
| V17-N4 minimum information separation | **THEOREM** | Exact solution of D4 requires one full bit of information about the balanced sign; this is not a scalar-degree count |
| V17-H1 exact homogeneous escape | **THEOREM FOR THE REGISTERED FIXTURE** | Exact costs $2$ and $4$, margin $2$, with the public oriented reference $u$ in the identical terminal action encoding |
| General delayed credit assignment | **NOT ESTABLISHED** | D4 is a delayed-memory/aliasing task, not a general theorem about learning causal credit from delayed reward |

Severity register:

- **P0: none.**  No contradiction was found in N1, N2, N4, or the exact H1
  algebra.
- **P1: none.**  The countable claim closes under the compatibility and
  measurability conditions stated in Section 5; no unconditional finite-prefix
  limit is claimed.
- **P2-1:** permutation equivariance alone does not construct an infinite
  ensemble or an infinite-depth limit.  A compatible measurable countable
  system is required.
- **P2-2:** H1 uses the contract's public oriented reference $u$.  It remembers
  the sign relative to that reference; it does not remember an unknown
  terminal query direction.  Because $u$ is independent of $S$ and identical
  in both paired branches, it is not a sign side channel.
- **P2-3:** the lift adds a declared homogeneous splitting.  Calling
  $G\in\operatorname{SPD}(d+1)$ “one factor field” does not make it a strict
  D1 state or erase its $d+1$ extra ambient real state coordinates and their
  added memory content.
- **P2-4:** N4 is a two-class information lower bound for D4, not a general
  capacity bound.  Exact real tensor entries have pathological unbounded
  coding capacity unless continuity, precision, or noise assumptions are
  added.

## 2. N1: full GL linear-chart covariance forces sign-blindness

Fix $g\in\operatorname{SPD}(V^*)$, $x\ne0$, and $c>0$.  The matrix
$J=-I$ belongs to $GL(d)$ for every $d\ge1$, and

$$
J^{-1}=J^{-T}=-I,
\qquad
J^{-T}gJ^{-1}=(-I)g(-I)=g,
\qquad
Jx=-x.
$$

Substituting this allowed chart into V17.2 gives

$$
U(g,-x,c)
=U(J^{-T}gJ^{-1},Jx,c)
=J^{-T}U(g,x,c)J^{-1}
=U(g,x,c). \tag{1}
$$

This proves V17.3.  The conclusion uses **full** $GL(d)$ covariance.  A
restricted chart group that excludes $-I$ would need a new proof and may admit
orientation structure; that is outside D2.

For an allowed randomized update, let $W$ be its seed, independent of the cue
sign, and write the conditional deterministic map as $U_w$.  D2 requires the
claim to be read conditionally: each $U_w$ obeys V17.2.  Equation (1) then
holds for every admissible $w$.  Coupling the two sign branches with the same
seed therefore gives exact state equality almost surely.  If the branches use
independent copies of $W$, their realized matrices need not be bitwise equal,
but their transition kernels are equal:

$$
\Pr\{U_W(g,-x,c)\in B\}
=\Pr\{U_W(g,x,c)\in B\} \tag{2}
$$

for every Borel set $B$.  Independence of $W$ from the cue sign is essential;
a sign-correlated seed would be the forbidden side channel.

Revision 1 deliberately requires pointwise covariance for almost every fixed
seed.  Covariance of only the **averaged transition law** would be too weak.
An explicit excluded construction already exists in $d=1$.  Let
$\epsilon\in\{-1,+1\}$ be uniform, choose $\alpha>0$, and define the positive
random output

$$
K_\epsilon(g,x,c)=g\exp\bigl(\alpha\epsilon\,\operatorname{sgn}x\bigr).
\tag{3}
$$

For a scalar chart $J\ne0$, the transformed inputs are $g/J^2$ and $Jx$.
When $J>0$, (3) transforms pointwise by the required factor $J^{-2}$.  When
$J<0$, its exponent changes sign, but its **distribution** is unchanged
because $\epsilon$ is symmetric.  Hence the transition law is full-$GL(1)$
covariant and is likewise identical in law for $x$ and $-x$.  For a fixed
$\epsilon$, however, $K_\epsilon(g,-x,c)\ne K_\epsilon(g,x,c)$, and the chart
$J=-1$ violates pointwise V17.2.  Thus law covariance alone would not give
pathwise paired-state equality.  This kernel is excluded by D2.  Equations
(1)--(2), and all later same-seed conclusions, use the stronger registered
condition.

## 3. N2: the paired histories alias exactly

Condition first on one realization of the public oriented reference $U=u$.
By D4 the tuple $(U,\Omega,g_0)$ is jointly independent of $S$, and the paired
experiments use its same realization at every step.  At the cue step the inputs
are $u$ and $-u$, with the same positive cost.  The pointwise fixed-seed form
of N1 gives

$$
g_1^{(+)}=U(g_0,u,c)=U(g_0,-u,c)=g_1^{(-)}. \tag{4}
$$

All subsequent predecision observations and costs are paired and
sign-independent by D4.  Conditioned on the common $\Omega$, if
$g_t^{(+)}=g_t^{(-)}$, the two deterministic conditional updates receive
identical inputs, so $g_{t+1}^{(+)}=g_{t+1}^{(-)}$.  Joint independence of
the whole family matters: merely requiring each seed marginal to be
independent of $S$ would still allow joint correlations that encode the sign.
Induction yields the same terminal metric in the paired histories.  Averaging
the common-$\Omega$ coupling yields equality of terminal-state laws.

The terminal observation supplies the same oriented $u$ to both branches and,
by D4, is a deterministic function only of $u$ and the fixed action ordering.
Thus it contains no random pad or extra context: conditioning on $O_T$ adds no
cue information beyond public $U$.  Public $u$ fixes the meaning of the labels
$a=\pm1$, but it carries no cue sign because $U$ was drawn independently of
$S$, so $I(S;U)=0$.  D3 additionally requires
the fresh terminal randomizer $R_T$ to be jointly independent of
$\sigma(S,H_{<T},\Omega,G_T,O_T)$; it cannot recycle a prior seed or acquire a
cue correlation through the history.  Let $q_u$ be the conditional probability
that the metric-only terminal policy selects $+1$.  Its metric-state law,
public $u$, and admissible fresh-randomness law are identical in both sign
branches, so the same $q_u$ applies to both.  Conditional on every public $u$,

$$
\Pr(A=S\mid U=u)
=\frac12\Pr(A=+1\mid S=+1)
 +\frac12\Pr(A=-1\mid S=-1)
=\frac12q_u+\frac12(1-q_u)
=\frac12. \tag{5}
$$

Averaging over $U$ leaves the value $1/2$.  The registered loss and regret are
$1[A\ne S]$, hence both have expectation $1/2$.  This is not merely a
statement about a deterministic tie breaker: it holds for every action
distribution permitted by D3.

## 4. N3 at finite component count and finite depth

Consider a finite recursive system and expose its computation as events in
causal order.  The induction invariant is:

> Under the paired common-seed coupling, every metric state and every allowed
> message available after event depth $k$ is identical in the two sign
> branches.

Condition on one common realization of the entire tuple

$$
(U,\Omega,\text{initial topology},\text{all initial states}).
$$

D5 makes this common coupling legitimate because this whole tuple is jointly
independent of $S$ and is shared by the paired branches; separate marginal
independence of its entries would not suffice.  At depth zero these conditioned
objects and all paired sign-independent messages are identical.  D5 then
imposes an essential ordering rule: every component that
receives $su$ must consume it through its pointwise-covariant metric update
**before** persistent storage or communication.  N1 makes the resulting metric
identical for $+u$ and $-u$ at each fixed component seed; a component that does
not receive the cue is trivially paired.  After that update D5 permits
communication only from the paired metric states, the common public $u$, and
paired sign-independent messages.  The raw signed cue $su$ is not an allowed
message.

For the induction step, deterministic or conditionally deterministic
communication receives equal metric states, the same public $u$, and equal
allowed messages, so it emits equal messages.  A permutation-equivariant
aggregator receives the same indexed multiset in the two branches and emits
the same result.  The next metric update therefore receives identical allowed
inputs, and its outputs remain equal.

The raw-cue exclusion is a real premise, not a presentation detail.  If a
component could communicate both public $u$ and unprocessed $x=su$, then it
could compute $u^Tx=s$ and transmit the answer.  Such a system would violate
D5 and would not be a counterexample to N3.

Thus every component state is paired at every finite depth.  Applying the N2
calculation to any metric-only terminal policy proves accuracy $1/2$ and regret
$1/2$.  This proof covers every finite component count and finite SCC unrolling
depth.  Increasing either finite number cannot restore information absent from
all inputs.

Permutation equivariance is useful for excluding an agent-index side channel,
but equality of paired inputs is the decisive invariant.  Replicating a
sign-blind state any finite number of times creates redundancy, not cue
information.

## 5. N3 countable extension: compatibility and measurability

A countable theorem requires a defined probability space; a table of finite
$N$ scores is not such a construction.  Let

$$
\mathcal X
=\prod_{i\in\mathbb N}\operatorname{SPD}(V_i^*)
$$

carry its product Borel sigma-algebra.  Each factor is an open subset of a
finite-dimensional Euclidean space and is therefore standard Borel.  Impose
the following explicit regularity conditions:

1. initial-state, seed, communication, aggregation, and update maps define a
   measurable countable system on $\mathcal X$ (or on the corresponding
   countable trajectory product);
2. the whole tuple consisting of public $U$, initial topology, all initial
   states, and the entire seed family $\Omega$ is jointly independent of $S$,
   and the two sign branches use its same realization as required by D5;
3. the terminal policy is a measurable Markov kernel of the countable metric
   state, common terminal observation, and D3-fresh randomness jointly
   independent of the preceding sigma-algebra.

For any finite event depth, the Section 4 induction applies coordinatewise to
all $i\in\mathbb N$: every coordinate of the two countable state vectors is
equal under the coupling.  Coordinatewise equality is equality in the product
space, so every measurable terminal kernel has the same action law in both
branches.  Equation (5) again gives success $1/2$.

The same conclusion can be constructed from finite ensemble laws, but only
with a compatibility condition.  Enumerate the components, let
$\mu_{n,s}$ be the joint law of public $U$ and the first $n$ states for sign
$s$, and require

$$
(\pi_n)_*\mu_{n+1,s}=\mu_{n,s} \tag{6}
$$

for the prefix projection $\pi_n$.  Finite N3 gives
$\mu_{n,+}=\mu_{n,-}$ for every $n$.  Because the finite laws are
projectively compatible on standard Borel spaces, they define a countable
product law.  Equivalently, the two extensions agree on every cylinder set;
cylinder sets form a generating pi-system, so the pi-lambda uniqueness theorem
gives

$$
\mu_{\infty,+}=\mu_{\infty,-}. \tag{7}
$$

A measurable terminal kernel preserves this equality of laws.  This is the
claimed countable-agent theorem.

There is a separate issue for **infinite nesting/event depth**.  If the state
after depth $k$ has a product-topology limit, or the whole countable trajectory
is passed to one declared measurable terminal functional, equality at every
finite $k$ forces equal limits or equal trajectory outputs.  If no limit or
terminal functional is defined, “the infinite-depth action” is undefined and
cannot be assigned a success probability.  Permutation equivariance alone
does not supply (6), convergence, or measurability.  Accordingly, this lane
does not relabel a merely finite-prefix family as an infinite theorem.

## 6. N4: exact solution requires one cue-correlated bit

Let $U$ denote the random public oriented reference and $G_T$ the persistent
terminal metric state exposed to the common policy.  D4 makes $O_T$ a
deterministic function of $U$ and the fixed action order, while D3 makes the
fresh terminal randomizer jointly independent of the full preceding
sigma-algebra.  The contract gives

$$
I(S;U)=0,
$$

so public $U$ fixes the action embedding but does not itself leak the sign.
Suppose an exact solver exists.  Then its terminal action satisfies $A=S$
almost surely as a measurable function or kernel of $(G_T,O_T)$, equivalently
of $(G_T,U)$ in this registered task.  Independent
fresh randomness cannot repair an overlap of the two input laws.  For almost
every fixed $u$, the kernel must select $+1$ with probability one under
$G_T\mid(S=+1,U=u)$ and with probability zero under
$G_T\mid(S=-1,U=u)$.  The two conditional metric-state laws must therefore be
mutually singular.  Equivalently, an exact solver required pointwise over the
D4 fixture exposes two distinguishable terminal-state classes for every fixed
public $u$; under an almost-sure definition, the same statement holds for
almost every $u$.  In either reading $S$ is recoverable from $(G_T,U)$ on the
declared domain, up to the corresponding null sets.

Consequently

$$
H(S\mid G_T,U)=0,
\qquad
I(S;G_T\mid U)
=H(S\mid U)-H(S\mid G_T,U)
=1\ \text{bit}. \tag{8}
$$

This conditional form is essential.  The marginal quantity $I(S;G_T)$ need not
equal one: if $U$ has a centrally symmetric distribution, the hypothetical
encoding $G_T=F(SU)$ can have the same marginal law for both signs.  For
example, in a fixed chart $F(v)=2I+\operatorname{diag}(v)$ is SPD for every
Euclidean-unit $v$ and is injective.  Thus $S$ is recovered from $(G_T,U)$
even though it need not be recovered from $G_T$ alone.  N4 is therefore a
lower bound on
persistent information **relative to the common public reference**, not a
claim that the persistent state has a sign-distinguishable marginal law after
averaging over references.

Equation (8) is an information lower bound.  It does not say that a chosen
continuous architecture requires exactly one real scalar, one tensor entry,
or one trainable degree of freedom.  Nor does it say that one noisy bit is
sufficient: exact decoding requires the full cue bit.

Nor is (8) an upper bound on what one exact-real SPD tensor can store in other
tasks.  As a bare state-space fact, a single positive real can injectively
encode every finite binary string, and even a countable binary sequence can be
embedded in a Cantor-type subset, for example through digits
$1+\sum_{k\ge1}2b_k3^{-k}$.  Such a value can occupy one positive diagonal
entry of an SPD matrix.  This is a discontinuous or infinitely precise coding
boundary, not a robust controller construction and not an evasion of N1 for
the sign-paired cue.  Without declared regularity, precision, or noise, N4
therefore cannot be promoted into a general memory-capacity theorem.

## 7. H1: exact homogeneous one-SPD calculation

Let $\lVert u\rVert_2=1$, $s,a\in\{-1,+1\}$, and

$$
z_s=(su,1),
\qquad
y_a=(au,-1),
\qquad
G_0=I_{d+1}.
$$

The write prediction is

$$
p=z_s^TG_0z_s=\lVert u\rVert_2^2+1=2. \tag{9}
$$

For the V16 update at $\eta=1$ and $c=4$,

$$
r=\log(p/c)=\log(1/2),
\qquad
e^{-r}=2.
$$

Substitution into the rank-one update gives, without reference to production
code,

$$
G_1
=G_0+\frac{e^{-r}-1}{p}(G_0z_s)(G_0z_s)^T
=I_{d+1}+\frac12z_sz_s^T. \tag{10}
$$

Now $y_a^Ty_a=2$ and

$$
z_s^Ty_a=sa-1.
$$

Therefore

$$
y_a^TG_1y_a
=2+\frac12(sa-1)^2
=
\begin{cases}
2,&a=s,\\
4,&a=-s.
\end{cases} \tag{11}
$$

The unique minimum is $a=s$, and the wrong-minus-correct margin is exactly
$2$.  The sign is held in the spatial--homogeneous cross block:

$$
G_1=
\begin{pmatrix}
I_d+\tfrac12uu^T & \tfrac12su\\
\tfrac12su^T & \tfrac32
\end{pmatrix}. \tag{12}
$$

This explains both the success and the escape from N1.  The paired augmented
cues $(u,1)$ and $(-u,1)$ are not global negatives; applying $-I_{d+1}$ would
also negate the declared homogeneous coordinate.  The homogeneous-splitting
axiom has introduced the orientation-sensitive cross block that strict D1
forbids.

An SPD $(d+1)\times(d+1)$ matrix has

$$
\frac{(d+1)(d+2)}2
$$

independent entries, compared with $d(d+1)/2$ for the strict original-space
metric.  The difference is exactly $d+1$; for $d=3$ the totals are $10$ and
$6$, so H1 adds four ambient real state coordinates.  In block form these are
the $d$ entries of a spatial covector and one scalar packaged in the same SPD
factor.  The packaging does not remove their added memory content.

For a spatial chart $J\in GL(d)$, define

$$
A=\operatorname{diag}(J,1),
\qquad
\widetilde z=A z,
\qquad
\widetilde y=A y,
\qquad
\widetilde G=A^{-T}GA^{-1}. \tag{13}
$$

Then

$$
\widetilde z^T\widetilde G\widetilde z=z^TGz,
\qquad
\widetilde y^T\widetilde G\widetilde y=y^TGy. \tag{14}
$$

V16 covariance in the augmented space transports (10) to
$\widetilde G_1=A^{-T}G_1A^{-1}$, so (11) and the chosen action are unchanged.
This requires transporting the initial state as
$\widetilde G_0=A^{-T}I A^{-1}$.  Resetting $G_0$ to the numerical identity in
every nonorthogonal chart, renormalizing $Ju$, or reprojecting after transport
would not be this covariance theorem.

The registered lift covers spatial linear charts of the declared form (13).
It does not establish covariance under an undeclared transformation that mixes
or flips the declared homogeneous coordinate.

## 8. Dimensionless-core audit

The protocol explicitly declares all synthetic coordinates and scalar costs
dimensionless.  With dimension vector $\mathbf 0=(0,0,0,0)$, the audit is:

| Core quantity or argument | Dimension vector | Dimensionless | Reason |
|---|---:|---:|---|
| $x,u,z_s,y_a$ coordinates | $\mathbf 0$ | yes | Synthetic coordinate convention; the last coordinate is the pure number $1$ |
| $g,G,J,A$ components | $\mathbf 0$ | yes | They act on dimensionless coordinates in this registered protocol |
| $p=x^Tgx$ and $z^TGz$ | $\mathbf 0$ | yes | Quadratic forms of dimensionless operands |
| $c$ | $\mathbf 0$ | yes | Declared positive synthetic cost |
| $p/c$ | $\mathbf 0$ | yes | Ratio of like-dimension quantities |
| $\log(p/c)$ | $\mathbf 0$ | yes | Logarithm receives a positive dimensionless argument |
| $\eta$ and $e^{-\eta\log(p/c)}$ | $\mathbf 0$ | yes | $\eta=1$ is dimensionless |
| $1[A\ne S]$, accuracy, regret | $\mathbf 0$ | yes | Probabilities and indicator averages |
| $I(S;G_T\mid U)$ | $\mathbf 0$ | yes | A conditional logarithmic information count, reported in bits |

Thus G-DIMENSIONLESS passes mathematically.  This is only dimensional
consistency.  If $u$ were a dimensionful physical vector, appending the number
$1$ would require a declared reference scale and linear-chart convention;
the present calculation supplies neither physical evidence nor a spacetime
metric interpretation.

## 9. What the no-go does and does not establish

The proved necessary condition is narrow and precise:

> When a terminal decision depends on a past balanced sign, that sign is not
> present in the common terminal observation, and every permitted persistent
> update aliases the two signed histories, exact solution requires some
> cue-correlated persistent information outside the aliased strict metric.

This is a **delayed-memory/partial-observability necessity theorem** for D4.
It is not a theorem that every delayed-reward or delayed-credit problem needs
an eligibility trace.  For example, no extra cue memory is forced when both
histories have the same optimal action, when the relevant cue is supplied
again at decision time, or when the current Markov observation already
contains a sufficient statistic.

General temporal credit assignment asks how a later reward is attributed to
earlier actions or parameters and how the corresponding update is learned.
D4 reveals reward only after the action, but N1--N4 neither learn a credit
kernel nor identify which earlier action caused a reward.  H1 performs one
analytic write and one quadratic read for a registered signed cue.  It is a
narrow memory primitive, not a solution to general delayed credit assignment,
recursive intelligence growth, biological fidelity, or AGI.

Status: COMPLETE
