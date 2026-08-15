# V18b independent mathematical verification

Status: COMPLETE

This lane verifies V18b-E1--E5 from the final repaired contract.  It treats
the public cue marker, exact latch/no-op transitions, binary reward semantics,
and model classes as registered axioms.  It does not inspect implementation
results or confirmation seeds.  In particular, numerical agreement is not
used as a proof.

## 1. Verdict and formal status

| Item | Status | Exact boundary |
|---|---|---|
| V18b-E1 reward decoding | **THEOREM** | Deterministic known binary action, binary label, and exact correctness reward |
| V18b-E2 delayed coordinate learning | **THEOREM** | D1, D4, A1; transfers to A2/A3 only through their exact registered latch/readout |
| V18b-E3 strict even-metric no-go | **THEOREM** | Realized paired accuracy requires the complete D3 whole-trajectory coupling |
| V18b-E4 homogeneous carrier | **THEOREM** | Registered rank-one write in the declared homogeneous splitting |
| V18b-E5 lesion predictions | **THEOREM** | Lesions are applied on every training episode exactly as D6 specifies |
| V18b-H1 production/confirmation pass | **PREDICTION, NOT YET SCORED** | Requires implementation and once-opened confirmation evidence |

Severity register:

- **P0: none.**  The missing whole-trajectory quantifier that invalidated V18
  is explicitly supplied by V18b-D3.
- **P1: none.**  A2's trace lesion is type-correct in the final contract: it
  resets the SPD factor to canonical identity, so its decoded cross block is
  zero without leaving the SPD state space.  A2 distractors are exact no-ops.
- **P2-1:** without D3's common checkpoint and identical complete nuisance
  realization, E3 yields equality in law and expected accuracy $1/2$, not an
  exact realized score for an independently sampled pair.
- **P2-2:** finite-$N$ equality does not prove a countable or infinite-depth
  SCC theorem.  The V17 compatibility, measurability, and terminal-functional
  conditions remain necessary.
- **P2-3:** delay robustness here follows from a task-supplied exact no-op
  latch.  The test does not learn salience or a decay-resistant memory rule.
- **P2-4:** held-out dense vectors are new linear compositions inside the
  registered teacher family, not semantic or arbitrary-distribution OOD.
- **P2-5:** one homogeneous factor has nine added independent real
  coordinates.  Serialization as one object is not one semantic degree of
  freedom.

Thus the mathematical slice `G-MATH` is eligible to pass: E1--E5 close with
no open P0/P1.  H1 and all numerical/integrity gates remain empirical.

## 2. E1: the action and correctness bit recover the label

Let $a,y\in\{-1,+1\}$ and $R=1[a=y]$.  If $a=y$, then $R=1$ and

$$
a(2R-1)=a=y. \tag{V18b.M1}
$$

If $a\ne y$, binary-valuedness forces $a=-y$; then $R=0$ and

$$
a(2R-1)=-a=y. \tag{V18b.M2}
$$

These two exhaustive cases prove E1.  Determinism and the two-point action
and label sets are premises.  With reward noise, censoring, more than two
labels, or an unknown action, the same identity need not hold.

## 3. E2: exact coordinate accumulation and dense composition

On a visit to coordinate $j$, D1 gives

$$
x=s e_j,
\qquad
y=s\theta_j,
\qquad
s^2=1. \tag{V18b.M3}
$$

E1 gives $\widetilde y=y$ independently of the action, including a tie
action.  Under A1 the current trace is exactly $e=x$ and every intervening
distractor is a no-op.  Therefore the only permitted classifier transition,
which occurs after reward, is

$$
\Delta w
=\eta\widetilde y e
=\eta(s\theta_j)(s e_j)
=\eta\theta_j e_j. \tag{V18b.M4}
$$

The cue sign, action, order, and delay have disappeared algebraically.  D4
also makes $w$ identical at episode start, after the cue, after each
distractor, and immediately before reward; hence (V18b.M4) is a delayed
post-reward update rather than an earlier supervised write.

Each epoch visits every coordinate exactly once.  Starting from $w=0$, after
$k$ complete epochs induction on $k$ gives

$$
w^{(k)}=k\eta\theta. \tag{V18b.M5}
$$

With four epochs and $\eta=1/4$, $w^{(4)}=\theta$.  For every accepted dense
query $q=r/\sqrt d$, D2 first verifies the exact integer margin
$m=\theta^Tr\ne0$.  Consequently

$$
w^Tq=\frac{\theta^Tr}{\sqrt d}=\frac{m}{\sqrt d}\ne0, \tag{V18b.M6}
$$

so the classifier returns $\operatorname{sign}(\theta^Tq)$ exactly.  The
negative member has the opposite nonzero margin and is also exact.

For A2, expanding its registered write gives

$$
G=
\begin{pmatrix}
I_d+\tfrac12xx^T & \tfrac12x\\
\tfrac12x^T & \tfrac32
\end{pmatrix}. \tag{V18b.M7}
$$

Thus $2G_{1:d,d+1}=x$.  The factor is the only real-valued episodic memory,
and its distractor transitions are exact no-ops, so the proof of
(V18b.M4)--(V18b.M6) transfers unchanged.  For A3, the registered hard latch
has $h=x$ and the same no-op/reward transition, so the same proof transfers
with $h$ in place of $e$.  These equivalences are state/readout theorems; they
do not establish FLOP or wall-time equivalence.

## 4. E3: strict sign-even states and the whole-trajectory quantifier

First fix one permissible seed realization.  Full-$GL(d)$ covariance itself
already forces evenness when $J=-I$:

$$
U(g,-x)
=U(J^{-T}gJ^{-1},Jx)
=J^{-T}U(g,x)J^{-1}
=U(g,x). \tag{V18b.M8}
$$

D5 also registers this fixed-seed pointwise sign-even property explicitly.
Consider an accepted D2 pair $q,-q$.  D3 starts both branches from the same
post-training checkpoint and couples the *entire* post-checkpoint nuisance
tuple: every distractor and message, topology, update/ensemble seed, and
terminal-policy seed occurs in the same order.  At the marked cue,
(V18b.M8) makes every strict component state equal.  Suppose all component
states and allowed messages are equal immediately before a later event.  The
two branches then receive the same event input and the same fixed seed, so
their next states and messages are equal.  Induction over the finite event
sequence gives equal terminal component states.

For every registered finite ensemble size, equal indexed states imply equal
sorted multisets.  The deterministic permutation-invariant aggregate and
fixed action function therefore return the same realized action $a$ in both
branches.  Because D2 excludes zero margin,

$$
\operatorname{sign}(\theta^T(-q))
=-\operatorname{sign}(\theta^Tq). \tag{V18b.M9}
$$

Exactly one of two opposite labels equals their common action.  Hence every
pair has one correct branch and one incorrect branch, and its realized
accuracy is exactly $1/2$.  Averaging pairs, seeds, or any registered finite
$N\in\{1,2,4,8,16,64\}$ preserves that exact value.

The complete coupling is logically necessary for the realized claim.  If
the two branches instead receive independent distractors, the allowed policy
could, for example, return the sign of a sign-independent bit extracted from
its first distractor.  Its action laws are identical in the two branches, but
the realized actions can differ, producing pair accuracy $0$, $1/2$, or $1$.
Likewise, coupling only terminal policy randomness does not couple stochastic
updates earlier in the trajectory.  Without D3 the symmetry proves only

$$
\mathbb E\!\left[
\frac{1[A_q=y_q]+1[A_{-q}=y_{-q}]}2
\right]=\frac12, \tag{V18b.M10}
$$

provided the two nuisance laws are sign-independent and identical.  V18b-D3
supplies the stronger premise and therefore avoids the V18 counterexample.

For countably many components, finite scores alone say nothing.  E3 extends
only if the full initial-state/seed/topology tuple is jointly coupled, the
finite laws are projectively compatible on standard Borel state spaces, all
updates/messages and the terminal kernel are measurable, and the raw signed
cue is not communicated before the even update.  Equality then holds on all
cylinder sets and hence on the generated product sigma-algebra.  An
infinite-event-depth action additionally needs a declared limit or measurable
trajectory functional; otherwise it is undefined.  This is the conditional
V17 boundary, not a result inferred from the finite ensemble table.

## 5. E4: the homogeneous cross block is precisely the added carrier

Equation (V18b.M7) shows directly that its spatial--homogeneous cross block is
$b=x/2$ and the registered readout is $e=2b=x$.  Deleting the homogeneous row
and column leaves

$$
G_{\mathrm{spatial}}=I_d+\frac12xx^T. \tag{V18b.M11}
$$

Since $(-x)(-x)^T=xx^T$, this reduced state is even and cannot distinguish a
paired cue under E3.  The cross block is therefore the sign-sensitive carrier
for this registered write; this is not a universal minimality theorem over
all possible architectures.

An SPD matrix of size $k$ has $k(k+1)/2$ independent real coordinates.  At
$d=8$,

$$
\dim\operatorname{SPD}(9)=\frac{9\cdot10}{2}=45,
\qquad
\dim\operatorname{SPD}(8)=\frac{8\cdot9}{2}=36. \tag{V18b.M12}
$$

The difference is nine: eight cross-block coordinates forming a covector in
the declared splitting and one homogeneous scalar.  A dense serializer may
store 81 entries, but symmetry leaves 45 independent reals.  Neither number
is reduced by calling the state one factor.

## 6. E5: exact lesion predictions

For the A1 trace lesion, D6 replaces the current trace by $e=0$ immediately
before each reward update.  For A2 it resets the factor to canonical identity
while retaining the active tag; the identity has zero spatial--homogeneous
cross block, so its current decoded $e$ is also zero.  D6 forbids a cached cue.
Thus every lesioned update obeys

$$
\Delta w=\eta\widetilde y\,0=0. \tag{V18b.M13}
$$

Starting at zero, the trace-lesioned and no-trace classifiers remain $w=0$.
Their deterministic tie action is $+1$ for both members of each accepted
$q,-q$ pair, whose labels are opposite by (V18b.M9).  Their realized paired
accuracy is exactly $1/2$.

For immediate reward inversion, $R'=1-R$ and E1 imply

$$
\widetilde y'
=a(2R'-1)
=a(1-2R)
=-a(2R-1)
=-y. \tag{V18b.M14}
$$

The intact current trace remains $x$, so every visit to coordinate $j$ adds

$$
\Delta w'=\eta(-y)x=-\eta\theta_j e_j. \tag{V18b.M15}
$$

The induction from E2 now gives $w=-\theta$ after four epochs for both A1
and A2.  Every accepted query has nonzero margin, and

$$
\operatorname{sign}(w^Tq)
=-\operatorname{sign}(\theta^Tq), \tag{V18b.M16}
$$

so reward-inversion accuracy is exactly zero.  These results show dependence
on the registered trace and reward alignment.  They are not a general causal
discovery or credit-assignment theorem.

## 7. Dimensionless audit

The contract declares all synthetic quantities dimensionless.  With dimension
vector $\mathbf0=(0,0,0,0)$:

| Core argument | Dimension vector | Dimensionless | Normalization/reference |
|---|---:|---:|---|
| $\theta,r,x,q,e,h,w,z$ | $\mathbf0$ | yes | Synthetic coordinates |
| $d,K$ and delay ticks | $\mathbf0$ | yes | Positive integer counts |
| $\sqrt d$ in $q=r/\sqrt d$ | $\mathbf0$ | yes | Positive dimensionless normalizer, $d=8$ |
| $m=\theta^Tr$ and $w^Te$ | $\mathbf0$ | yes | Inner products of dimensionless operands |
| $\eta=1/4$, $R$, loss, regret, accuracy | $\mathbf0$ | yes | Pure numbers/probabilities |
| $G$, its factor entries, and quadratic/readout blocks | $\mathbf0$ | yes | Built from dimensionless $z$ and $I$ |
| $\lVert w-\theta\rVert_\infty$ | $\mathbf0$ | yes | Norm of a dimensionless vector |

No logarithm or exponential occurs in E1--E5.  The only normalization divides
by the positive dimensionless number $\sqrt8$.  Therefore the mathematical
core passes G-DIMENSIONLESS.  This establishes dimensional consistency only;
it supplies no physical, biological, or cosmological interpretation.

## 8. Closure boundary

The proved result is narrow: a task-supplied exact latch retains a marked cue;
the learner's own deterministic binary action and later correctness bit decode
the hidden binary label; four coordinate passes analytically reconstruct a
linear Rademacher teacher.  A fully coupled sign-even strict metric produces
the same action on $q,-q$ and therefore fails one member of every pair.

Nothing here proves learned event selection, noisy or partial-reward credit,
policy-gradient learning, compute superiority, arbitrary tasks, semantic OOD,
robust finite-precision memory, recursive scaling, biological fidelity,
cosmological identity, or AGI.

Status: COMPLETE
