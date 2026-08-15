# AGI V18b reward-decoded delayed-credit route audit

Status: COMPLETE

## 1. Target and comparison rule

The target is the finite deterministic primitive fixed in the contract:
retain one publicly marked cue across unmarked distractors, combine the
retained cue with the learner's own binary action and later binary reward,
and update a linear classifier that transfers from signed coordinate cues to
nonzero-margin dense compositions. The target is not learned salience,
general causal discovery, noisy credit assignment, or AGI.

The contract fixes the parts shared by every registered positive route:

- the public marker identifies the relevant event exactly;
- the deterministic decoder is $\widetilde y=a(2R-1)$;
- the hypothesis class is a dimension-eight linear classifier;
- four coordinate visits, $\eta=1/4$, deterministic tie action $+1$, and
  exact hold across distractors are supplied rather than learned;
- evaluation $q,-q$ branches use the same whole post-checkpoint nuisance
  realization under V18b-D3.

Routes are ranked by exact compatibility with the frozen gates, smallest
disclosed state, fewest unregistered choices, and strongest killing tests.
Mutable classifier coordinates are counted both as live state and learned
parameters. Discrete active tags are listed separately. Fixed constants,
loop indices, and deterministic seed namespaces are not counted as real
parameters. These are implementation ledgers, not exact-real information
capacity bounds.

No development or confirmation seed was read or executed in this lane. The
analysis is algebraic and uses only the V18b contract and the abandoned V18
route audit.

## 2. Status of the V18 repair

The predecessor's run-level P0 is repaired for the registered strict-control
comparison. V18b-D3 now couples the checkpoint, delay, every distractor,
message, topology, update/ensemble seed, and terminal-policy seed in the same
order for both $q$ and $-q$. For a fixed-seed pointwise sign-even transition,
equal paired states therefore propagate pathwise, and the deterministic
aggregate and action are equal. Since the labels are opposite, exactly one
member of each pair is correct.

This conclusion depends on the complete D3 coupling. It must not be exported
to independently sampled branch trajectories, for which only equality in law
and expected accuracy $1/2$ is supported. It also does not turn finite
ensemble results into a countable-SCC theorem.

The predecessor's unsupported phrase "exactly compute-matched" is also
repaired. V18b-A3 calls the hard latch only state-coordinate-matched and makes
no FLOP, operation-graph, memory-traffic, or wall-time claim. The sealing
ledger now names the imported in-repository production dependencies explicitly.

## 3. Route matrix

For $d=8$, an explicit vector has 8 real coordinates,
$\operatorname{SPD}(8)$ has 36 independent coordinates, and
$\operatorname{SPD}(9)$ has 45. The homogeneous implementation may serialize
81 dense factor entries, but only 45 are independent.

| Rank | Route | Maximum live real state | Learned real parameters | Marker and added-axiom disclosure | Target-aware? | Frozen-gate prospect | Primary killing test |
|---:|---|---:|---:|---|---|---|---|
| 1 | R-E: explicit eligibility | $w:8+e:8=16$, plus one active tag | 8 | Contract A1; public marker writes $e\leftarrow x$ exactly. No new axiom | Yes: latch, decoder, $\eta$, and linear target class are supplied | Exact candidate: four visits give $w=\theta$ | Reset $e$ before reward or change it on one distractor; the first must leave $w=0$ and paired accuracy $1/2$, while the second must violate the exact delta/final-defect certificate |
| 2 | R-H: homogeneous factor | $w:8+G:45=53$, plus one active tag; 89 dense serialized real entries if $w$ is included | 8 | Contract A2 and sealed V17 write; public marker selects the factor write. No new axiom | Yes: the homogeneous split and cross-block readout are task-designed | Exact representation candidate when the factor alone gives $2G_{1:8,9}=x$ | Delete the ninth row/column or zero its cross block; paired signed credit must collapse to $1/2$, and field inspection must find no cached cue |
| 3 | R-L: hard-latch equivalence control | $w:8+h:8=16$, plus one active tag | 8; learned gate parameters 0 | Contract A3; public marker is a fixed write/hold gate. No new axiom | Yes: it is the same supplied latch as R-E under a state relabeling | Exact and state-coordinate-matched to R-E; not a learned recurrence | Make one unmarked distractor write, or corrupt one held coordinate by one ULP; exact $h=x$ and final $w=\theta$ certificates must fail |
| 4 | R-A: learned marker attention | Lower bound $w:8+n:8+Z:1=17$ before optimizer state | Lower bound 9: $w:8+\beta:1$ | One extra candidate axiom: a learned marker-scoring/attention rule. Its optimizer, initialization, and hard/soft estimator are not registered | Yes: the scorer is allowed to observe the public relevance marker | Finite soft attention cannot meet the exact classifier gate; an exact hard mask collapses to R-E/R-L | For every finite $\beta$, use aligned or cancelling distractors and require exact $e=x$ and $w=\theta$ rather than accuracy alone |
| 5 | R-P: stochastic policy gradient | $w:8+e:8=16$; 17 with one scalar baseline | 8 without a learned baseline; 9 with one | One extra candidate axiom: a specified stochastic policy-gradient update. Temperature, baseline, entropy, clipping, and rollout budget remain unregistered | Yes: it still receives the exact public-marker trace and a binary linear policy | Ordinary reward-only REINFORCE cannot guarantee the probability-one exact gate in four visits | Enumerate four wrong actions for one initially zero coordinate; the positive-probability path leaves that coordinate unlearned |
| 6 | R-S: strict original-space metric | $36N$ for $N\in\{1,2,4,8,16,64\}$; no separate odd cue state | No separate classifier is permitted by D5; the mutable metric state is fully disclosed | Contract D5; marker may trigger only a fixed-seed pointwise sign-even map, and all messages are sign-independent. No new axiom | It is a deliberately adversarial control, not a solving route | No positive route prospect under its sign-aliasing premise; D3 makes the paired $1/2$ score pathwise | Require byte-identical metric and sorted aggregate serializations for $q,-q$ at every $N$ under the complete shared nuisance tuple; any difference exposes a side channel or coupling error |

Only R-E is recommended for implementation. R-H and R-L remain required
registered comparisons, while R-S is the required no-go control. This
recommendation does not remove any learner, lesion, or control required by the
contract.

## 4. R-E: explicit eligibility is the smallest transparent exact route

At the marked event, R-E stores

$$
e=x=s e_j.
$$

The classifier remains unchanged until reward. The registered decoder then
gives $\widetilde y=y=s\theta_j$, hence

$$
\Delta w
=\eta\widetilde y e
=\frac14(s\theta_j)(s e_j)
=\frac14\theta_j e_j.
$$

Four visits yield $w_j=\theta_j$ exactly. The values $0$, $1/4$, and $1$
are exactly representable in binary64 along this coordinate path, so an
implementation cannot excuse a wrong explicit-route certificate as
roundoff. Delay 0 and delay 128 have the same algebra because every unmarked
distractor is an exact state no-op.

The route is maximally auditable: $w$, $e$, and the active tag expose all
cue-correlated state. Its primary state-machine risks are aliasing $w$ and
$e$, mutating $w$ before reward, failing to clear the trace atomically,
accepting nonfinite cues, or retaining a second hidden cue. Required killing
fixtures are pre-reward byte snapshots, clearing immediately before reward,
one-distractor corruption, and mid-episode snapshot continuation.

R-E learns the coordinate classifier from delayed reward. It does not learn
the eligibility rule, because the public marker and exact latch already solve
event selection.

## 5. R-H: geometric packing is exact but not storage-minimal

The homogeneous write has block form

$$
G
=I_9+\frac12
\begin{pmatrix}x\\1\end{pmatrix}
\begin{pmatrix}x\\1\end{pmatrix}^{T}
=
\begin{pmatrix}
I_8+xx^T/2 & x/2\\
x^T/2 & 3/2
\end{pmatrix}.
$$

Its cue-odd cross block is $b=x/2$, so $e=2b=x$. The route therefore shares
R-E's reward-timed classifier algebra, subject to the registered $10^{-12}$
factor reconstruction tolerance. It is a useful test that the V17 geometric
carrier can participate in a learned post-reward update, but it is not a
smaller memory: 45 episodic coordinates replace 8, and the total live ledger
is 53 rather than 16 real coordinates.

The factor must be the only cue-correlated episodic real state. Caching $x$,
$b$, or decoded $e$ would turn the comparison into hidden R-E. Removing the
homogeneous coordinate leaves $I_8+xx^T/2$, which is sign-even and must lose
polarity. Factor validation, canonical identity reset, dense-versus-
independent coordinate accounting, and snapshot reconstruction are the main
numerical checks.

V18b-D6 fixes the A2 trace lesion unambiguously: reset the factor to canonical
identity while retaining the active tag. This remains inside
$\operatorname{SPD}(9)$, gives decoded current eligibility $e=0$, and prevents
an implementation from substituting either an invalid zero factor or a
different cross-block-only transition. A2 distractors are also registered as
exact factor no-ops.

## 6. R-L: hard recurrence is an equivalence control, not learned memory

The registered transition is

$$
h_t=
\begin{cases}
x_t,&m_t=1,\\
h_{t-1},&m_t=0.
\end{cases}
$$

The bijection $h\leftrightarrow e$ makes R-L semantically identical to R-E
for this task. It has the same 16 live real coordinates and the same eight
learned classifier parameters. The gate itself has zero learned parameters.
Passing delay 128 demonstrates exact holding by a supplied latch; it does not
demonstrate learned recurrence, temporal abstraction, or a compute advantage.

The state-coordinate-matched name in V18b is the strongest supported one.
A standard learned GRU would introduce hundreds of gate parameters and a new
optimization contract. Killing tests must require byte-identical hold across
all distractors, both cue signs on every coordinate, snapshot continuation,
and detection of deliberate one-ULP hidden-state corruption.

## 7. R-A: learned attention is outside the exact V18b route

A minimal soft marker attention illustrates the obstruction:

$$
n=\sum_t \exp(\beta m_t)x_t,
\qquad
Z=\sum_t \exp(\beta m_t),
\qquad
e_\beta=\frac nZ.
$$

With one marked cue and $K$ unmarked distractors $r_k$,

$$
e_\beta
=\frac{\exp(\beta)x+\sum_{k=1}^{K}r_k}
       {\exp(\beta)+K}.
$$

For every finite $\beta$, the cue coefficient is below one and distractor
coefficients are nonzero. Exact four-epoch recovery therefore does not follow,
even if aggregate accuracy happens to be high. A hard marker mask makes
$e=x$, but then the task-supplied marker directly selects the event and the
route reduces to R-E/R-L. Learning that hard selection would require a new
estimator, optimizer, initialization, training budget, and fresh seed block.

The killing test is algebraic: choose aligned distractors, retain finite
$\beta$, and inspect the exact eligibility and final classifier defect.
Stable log-sum-exp may fix overflow but cannot remove finite-temperature
leakage. No attention temperature or optimizer may be selected from V18b
development or confirmation results.

## 8. R-P: ordinary policy gradient lacks the exact finite guarantee

For a binary log-linear policy,

$$
\pi_w(a\mid e)
=\frac{\exp(a w^Te)}{2\cosh(w^Te)},
$$

a standard reward-only REINFORCE step has the form

$$
w^+
=w+\alpha R\left(a-\tanh(w^Te)\right)e.
$$

When $w_j=0$, a wrong action gives $R=0$ and no update. Four wrong actions on
the four visits to coordinate $j$ have probability $2^{-4}=1/16$ under
ordinary independent fair initial decisions. On that event $w_j$ remains
zero, so the probability-one finite-seed requirement and exact
$\lVert w-\theta\rVert_\infty$ certificate fail.

Replacing the estimator by the exact decoded label $a(2R-1)$ imports R-E's
analytic reward decoder; it does not show that ordinary policy gradient found
the update. Policy temperature, learning rate, baseline, entropy coefficient,
clipping, rollout count, and RNG coupling are unregistered look-elsewhere
choices. R-P therefore remains a successor hypothesis with an exhaustive
action-sequence killing test, not an implementation route for V18b.

## 9. R-S: strict metric remains a sign-aliasing control

For every fixed permissible update seed, full-$GL(8)$ pointwise covariance
includes the chart $J=-I$ and makes the original-space metric update even in
the cue. A finite collection of such components and deterministic
permutation-invariant sorted-multiset aggregation cannot create a cue-odd
carrier from sign-independent inputs.

V18b-D3 supplies the premise missing in V18: both $q,-q$ branches start from
the same bytes and consume the same entire nuisance trajectory. Equal states
therefore remain equal, their actions are identical, and their opposite
labels make the registered pair score $1/2$. The implementation killing test
is stronger than aggregate accuracy: compare component and aggregate
serializations after every event for every registered $N$. An accuracy-only
pass could conceal uncoupled streams or a cue-dependent side channel.

The control's $36N$ coordinates do not refute this symmetry obstruction.
Conversely, the obstruction is not a general capacity bound on exact-real SPD
state, does not cover a subgroup excluding inversion, and does not prove an
unconstructed infinite-agent system.

## 10. Lesions, cross-predictions, and look-elsewhere ledger

The registered lesions provide route-independent cross-predictions. Clearing
the current semantic trace before every training reward gives
$\Delta w=0$, so $w=0$ and exact paired evaluation accuracy $1/2$. Immediate
reward inversion gives

$$
a\left(2(1-R)-1\right)=-y,
$$

so four epochs give $w=-\theta$ and zero accuracy on every accepted nonzero-
margin query. These tests establish dependence on the declared trace and
reward alignment only within this construction.

R-E, R-H, and R-L add no tunable choice to the frozen contract. R-A adds at
least attention parameterization, estimator, optimizer, initialization, and
temperature/schedule choices. R-P adds at least policy temperature, update
step, baseline, entropy/clipping, and rollout choices. None of those choices
may be tuned against V18b seed blocks and then described as preregistered.

The dense-query test rules out memorizing only the 16 signed training vectors
as the declared implementation, but it remains linear composition under a
known basis and teacher. It is not semantic OOD. A coordinate-index table is
a useful adversarial fixture: rotating the training basis kills it and exposes
basis-specific preprocessing, but the table is outside A1--A3 and is not a
candidate route.

## 11. Severity register and recommendation

- **P0: 0 open in this route audit.** The previous whole-trajectory coupling
  counterexample is explicitly excluded by V18b-D3. No confirmation seed was
  accessed.
- **P1: 0 open.** V18b-D6 fixes canonical identity reset with the active tag
  retained, and A2's exact distractor no-op is explicit.
- **P2-1:** R-E and R-L are state-relabeling equivalents under the supplied
  marker; neither learns salience.
- **P2-2:** R-H packages an eight-component covector and scalar in one factor
  and is not one semantic degree of freedom or strict metric-only state.
- **P2-3:** finite strict ensembles do not establish a countable or infinite-
  depth SCC result.
- **P2-4:** exact dense-query transfer is linear basis composition, not
  general semantic OOD.

**Single implementation recommendation: R-E, explicit eligibility.** It is
the smallest exact registered route, exposes all cue-correlated state, and has
the most direct timing and lesion certificates. R-H should be retained as the
required geometric representation comparison, R-L as the state-coordinate
equivalence control, and R-S as the sign-aliasing control. R-A and R-P require
new contracts and fresh seed blocks.

This route ranking assigns no theorem, closure, gate, or promotion status.
Those decisions belong to the mathematical verifier and formal status audit.
