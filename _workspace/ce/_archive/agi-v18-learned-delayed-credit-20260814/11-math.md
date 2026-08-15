# AGI V18 reward-decoded delayed linear credit: mathematics

Status: COMPLETE

## 1. Audit outcome

E1, E2 and E4 are theorems under the registered deterministic task axioms.
E3's information statement and expected-accuracy conclusion survive, but its
claim of realized paired accuracy exactly $1/2$ is false under D2 as written.
D2 couples the terminal-policy draw but not every post-checkpoint distractor,
message and update seed. This is an open P0 because G-MATH requires E3.

## 2. E1: reward decoding

[Theorem V18-E1] For $a,y\in\{-1,+1\}$ and $R=1[a=y]$,

$$
R=\frac{1+ay}{2}. \tag{V18.M1}
$$

Therefore

$$
a(2R-1)=a(ay)=a^2y=y. \tag{V18.M2}
$$

The identity needs a known binary action and deterministic correct/incorrect
reward. It is not a theorem for noisy, partial or general scalar rewards.

## 3. E2: exact coordinate recovery

[Theorem V18-E2] On a visit to coordinate $j$,
$x=se_j$ and $y=s\theta_j$. By E1 the learner uses the true $y$ only after
receiving reward, so its update is

$$
\Delta w=\eta yx
=\eta(s\theta_j)(se_j)
=\eta\theta_j e_j. \tag{V18.M3}
$$

This increment is independent of cue sign, terminal action and delay length.
Each coordinate is visited four times. With $\eta=1/4$ and $w_0=0$,

$$
w_j=4\eta\theta_j=\theta_j,
\qquad w=\theta. \tag{V18.M4}
$$

Hence every evaluation query with $\theta^Tq\ne0$ is classified exactly by
$\operatorname{sign}(w^Tq)$. This is compositional generalization from the
coordinate basis for a linear teacher, not general semantic OOD.

The proof applies to A1 and A3 because their registered latches return $x$.
It applies to A2 in exact arithmetic by the cross-block calculation in E4.
The public marker has already selected which event to latch; no theorem here
learns event relevance.

## 4. E3: surviving information no-go and failed pathwise quantifier

[Theorem: information form] A pointwise full-$GL(d)$-covariant original-space
metric update is even in its vector input by the $J=-I$ argument closed in
V17. Thus the episodic metric state after $+e_j$ and $-e_j$ is the same. Since
$s$ is fair and independent of the fair teacher bit,

$$
Y=S\Theta_j
$$

is fair and independent of $\Theta_j$ conditional on the even state and public
coordinate. Consequently

$$
I(\Theta_j;Y\mid j,G)=0. \tag{V18.M5}
$$

The same conclusion holds jointly for any finite collection of independently
signed visits and for registered side-channel-free finite ensembles.

[Counterexample: E3 as written is false] D2 starts $q$ and $-q$ from a common
checkpoint and couples terminal-policy randomness, but it does not require the
two branches to receive the same entire post-checkpoint distractor/message
stream or the same randomized update seeds. Let an allowed sign-independent
nuisance bit $B$ be produced by that stream and let the terminal action be
$a=B$. Draw independent fair $B_+$ and $B_-$ in the two branches. The labels
are opposite. If $B_+=B_-$ exactly one branch is correct, but if
$B_+=-B_-$ then both branches are correct or both are wrong. The realized pair
accuracy can therefore be $0$, $1/2$ or $1$, contradicting the asserted exact
$1/2$.

The same construction can be implemented through distinct permitted
sign-independent metric updates: evenness in the query sign does not make two
different nuisance trajectories equal.

[Corrected theorem] If both pair members start from the same checkpoint and
use byte-identical post-checkpoint observations, distractors, messages and the
same realization of every update, ensemble and policy seed, then equality of
the even states propagates pathwise. The two actions are identical while the
labels are opposite, so realized paired accuracy is exactly $1/2$. Without
this whole-trajectory coupling, only expected accuracy $1/2$ follows under
independent symmetric nuisance.

The current contract does not state the corrected premise. Because the
contract exhausted its two math-verifier revisions, E3 cannot be silently
narrowed and G-MATH cannot pass this run.

## 5. E4: homogeneous carrier

[Theorem V18-E4] With $z=(x,1)$,

$$
G=I_{d+1}+\frac12zz^T
=
\begin{pmatrix}
I_d+\frac12xx^T & x/2\\
x^T/2 & 3/2
\end{pmatrix}. \tag{V18.M6}
$$

The cross block is $b=x/2$, hence $e=2b=x$. Removing the last row and column
leaves $I_d+xx^T/2$, which is invariant under $x\mapsto-x$ and returns to the
strict sign-blindness no-go.

The independent coordinate counts are

$$
\frac{(d+1)(d+2)}2-\frac{d(d+1)}2=d+1. \tag{V18.M7}
$$

For $d=8$ this is $45-36=9$: an eight-component covector and one scalar are
packaged in the augmented factor. A single factor field is not the strict
original-space metric-only state.

## 6. Dimensionless audit

The cue, trace, classifier and teacher coordinates are declared dimensionless.
Therefore $w^Te$, $yx$, $\eta yx$, reward, loss and accuracy are
dimensionless. Delay is an integer tick count. Dense queries are normalized by
$\sqrt d$, a pure number. No dimensional logarithm or exponential occurs in
E1--E4. Dimensional consistency does not establish biological or physical
meaning.

## 7. Findings

- P0-1: E3's exact realized paired-accuracy quantifier lacks whole-trajectory
  coupling of distractors, messages and every randomized update/ensemble seed.
  A complete counterexample is given above.
- P1-1: A3 matches $2d$ state coordinates but the contract registers no FLOP,
  operation-graph or wall-clock accounting. Calling it exactly compute-matched
  is unsupported; only state-coordinate-matched is justified.
- P1-2: any successor seal must include every imported production dependency,
  not only the new wrapper module.
- P2-1: exact coordinate recovery relies on a public cue marker and exact
  latch, so it is delayed memory plus reward decoding rather than learned event
  selection.
- P2-2: the countable SCC conclusion remains conditional on the V17 joint
  coupling and measurability hypotheses; finite ensemble scores do not prove
  an infinite system.

Current mathematical gate: `BLOCKED` for this contract. The corrected E1, E2,
E3 and E4 package is suitable for a successor contract with a fresh seed block.
