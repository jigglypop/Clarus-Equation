# AGI V18 reward-decoded delayed-credit route audit

Status: COMPLETE

## 1. Route question and comparison rule

The route target is the finite registered primitive, not delayed causal
discovery in general: preserve one publicly marked cue, act after distractors,
decode its hidden binary label from the agent's action and terminal reward,
and update a linear classifier that transfers from coordinate cues to dense
queries.

Routes are compared in this order:

1. disclose the task marker and every cue-correlated episodic state;
2. separate active state coordinates from learned parameters;
3. require the exact four-epoch classifier identity before using aggregate
   accuracy as evidence;
4. expose target-aware structure and numerical failure modes;
5. prefer the smallest transparent route among exact routes.

For the registered dimension $d=8$, an explicit vector has 8 real
coordinates, an original-space SPD state has 36 independent coordinates and a
homogeneous $\operatorname{SPD}(9)$ state has 45. State counts below include
the cross-episode classifier and the largest episodic state simultaneously
live. Learned-parameter counts include the mutable classifier $w$, even
though V18 learns it online rather than with a separate optimizer. Fixed
constants, loop counters and deterministic seed namespaces are not counted as
real parameters. These counts are implementation ledgers, not real-number
capacity bounds.

No development or confirmation seed was accessed in this lane. The
comparisons below are analytic and were fixed without route tuning.

### Run-level blocking defect found after route comparison

The final D2 revision couples the post-training checkpoint and terminal policy
draw for each $q,-q$ pair, but it does not explicitly couple the entire
post-checkpoint nuisance trajectory: all 128 distractors, state-transition
randomness, policy randomness and any update RNG must use the same realization
in both branches. A common checkpoint and common terminal draw alone do not
force equal realized terminal states when an intermediate stochastic map or
different nuisance stream is allowed.

Therefore the current contract does not entail the registered exact finite
strict-control accuracy $1/2$. It entails the symmetry/expected result only
under the remaining independence assumptions. This is a run-level P0 for the
auditor, not a theorem status assigned by this route lane. A successor contract
must jointly couple the whole post-checkpoint trajectory before the exact
realized paired gate can be used. The route ranking below remains useful for
that repaired successor, but it cannot rescue the present run.

## 2. Shared reward decoder and task-marker axiom

Every registered positive route uses the public marker bit $m_t$. The task
supplies exactly one $m_t=1$ cue, and all distractors have $m_t=0$. An exact
latch writes only on $m_t=1$. This is a task-selection axiom:

$$
\boxed{\text{V18 does not learn which past event is relevant.}}
$$

For deterministic binary action and reward, the terminal decoder is

$$
\widetilde y=a(2R-1)=y.
$$

The immediate reward-inversion lesion is equally exact:

$$
\widetilde y'
=a\left(2(1-R)-1\right)
=-\widetilde y
=-y.
$$

Thus A1, A2 and A3 differ only in how they carry the marked vector to the
terminal update. None learns the reward decoder, the marker semantics, the
linear hypothesis class, the tie rule, the number of epochs or the learning
rate. All three are target-aware analytic designs.

The task-law independence is also structural. Teacher coordinates and cue
signs are independent fair Rademachers, and teacher, cue, order, distractor
and query namespaces are jointly independent. The exact positive derivation
is pointwise and does not need that randomness, but the strict-control
information argument does: after an even state loses $s$, the decoded
$y=s\theta_j$ is fair and independent of $\theta_j$. A stochastic route
must additionally keep policy RNG independent of every task namespace; a
cue-correlated retained seed would be a forbidden side channel.

## 3. Route matrix

| Rank | Route | Active real state at $d=8$ | Learned real parameters | Task-marker axiom | Target-aware? | Current exact-gate status | Primary killing test |
|---:|---|---:|---:|---|---|---|---|
| 1 | R-E: explicit eligibility vector | $w:8+e:8=16$ | 8 | Public marker performs an exact write $e\leftarrow x$; unmarked inputs are ignored | Yes: latch, decoder and linear update are registered from the target | Analytically exact: four visits give $w=\theta$ | Clear $e$ before reward, or mutate it on one distractor; the first must return paired accuracy $1/2$, and the second must fail the exact classifier certificate |
| 2 | R-H: homogeneous factor memory | $w:8+G:45=53$ | 8 | Public marker writes the V17 homogeneous factor; all distractors leave it fixed | Yes: the homogeneous split and cross-block readout were designed for this cue | Analytically exact when the production factor alone reconstructs $e=2G_{1:8,9}=x$ | Delete the ninth row/column or zero the cross block; cue polarity must disappear and paired accuracy must return to $1/2$ |
| 3 | R-G: state-coordinate-matched fixed gated recurrence | $w:8+h:8=16$ | 8; learned gate parameters 0 | Public marker is a hard gate: write $h=x$, otherwise hold | Yes: it is the same exact latch as R-E with a recurrent name | Analytically exact and state-coordinate-matched, but no FLOP match is registered; semantically it is isomorphic to R-E | Require byte-identical $h$ across all 128 distractors and exact equality $h=x$ at reward; any learned or leaky recurrence is a different route |
| 4 | R-A: learned marker attention | Conceptual streaming state $w:8+n:8+Z:1=17$ | $w:8+\beta:1=9$ for the minimal scalar scorer | The public marker is fed to the learned score; without it this is a new task | Strongly yes: the score is allowed to look directly at the relevance label | Finite soft attention cannot meet the exact $w=\theta$ gate; a hard correct mask collapses to R-E/R-G | Use a finite $\beta$ and adversarial aligned distractors; require exact final defect and not only accuracy |
| 5 | R-P: stochastic policy gradient | $w:8+e:8=16$; an optional learned baseline makes 17 | 8 without a baseline, 9 with one | Still needs the same exact public-marker trace | Yes: binary linear policy and trace are supplied | Standard reward-only REINFORCE has nonzero probability of leaving a coordinate unlearned after four visits, so finite-seed rate 1 is not guaranteed | At $w_j=0$, force or enumerate four wrong actions for coordinate $j$; zero reward leaves $w_j=0$, violating $\lVert w-\theta\rVert_\infty\le10^{-12}$ |
| 6 | R-C: coordinate-index ledger | $w:8+j:1+s:1=10$ real registers; episodic information is 3 index bits plus 1 sign bit | 8 output cells; no gradient parameter is needed | Marker plus the axiom $x\in\{\pm e_j\}$ exposes $(j,s)$ directly | Maximally target-aware and basis-dependent | Can solve the coordinate table exactly, but is outside A1--A3 and does not use the registered $\eta=1/4$ update | Rotate the training basis or present a non-coordinate training cue; the index/sign representation is no longer defined |

R-E is the only implementation recommendation. R-H remains a required
registered comparison because it tests whether the V17 one-factor carrier can
participate in reward-timed learning. R-G remains the strong equivalence
control. The recommendation does not remove any learner or lesion that the
contract requires the evaluator to implement.

## 4. R-E: explicit eligibility is the transparent exact route

The active episodic state is exactly the marked cue,

$$
e=x=s e_j.
$$

After reward, the decoded update is

$$
\Delta w
=\eta\widetilde y e
=\eta(s\theta_j)(s e_j)
=\eta\theta_j e_j.
$$

The cue sign cancels, but it cancels only after the signed cue has survived
until reward. Four visits with $\eta=1/4$ add exactly $\theta_j e_j$ to
each coordinate. Binary64 represents $0$, $1/4$ and $1$ exactly, so the
registered path has no accumulation-roundoff excuse for a nonzero final
defect.

The task marker is indispensable. The route learns a classifier from delayed
reward, but it does not learn an eligibility rule. The most informative
lesions are therefore temporal and causal:

1. snapshot $w$ immediately before reward and require no earlier mutation;
2. clear $e$ immediately before reward and require $w$ not to change;
3. feed $R'=1-R$ immediately after each action and require decoded label
   $-y$, final classifier $w=-\theta$ and dense-query accuracy zero;
4. replay all 128 unmarked distractors and require exact trace invariance;
5. continue from a serialized mid-episode snapshot and require the same
   post-reward classifier.

Primary numerical risks are accidental aliasing between $w$ and $e$,
failure to clear the trace, accepting nonfinite cues, and updating on the
action rather than after reward. These are state-machine risks rather than
conditioning risks.

## 5. R-H: homogeneous geometry carries the same vector at larger cost

The registered write gives

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

Therefore the spatial/homogeneous block is $b=x/2$, and the exact semantic
readout is $e=2b=x$. Its 45 independent factor coordinates replace the 8
coordinates of an explicit episodic vector. Together with $w$, the live
ledger is 53 reals, 37 more than R-E. Relative to the original
$\operatorname{SPD}(8)$ metric, the lift adds nine coordinates: an
eight-component odd cross block plus one scalar.

The route is useful as a representation bridge, not as a storage-minimal
learner. A passing implementation must persist only the production factor
state and must not cache $x$, $b$ or $e$ as a hidden second trace.
Deleting the homogeneous row and column leaves

$$
I_8+\frac12xx^T,
$$

which is even in $x$. That ablation must kill signed credit.

Numerical risk is higher than R-E because a triangular factor may reconstruct
the cross block with roundoff, snapshot code may silently serialize a dense
matrix or hidden cue, and factor validation must reject non-SPD and nonfinite
states. The $10^{-12}$ certificate is still generous relative to the
registered binary values, but it may not be replaced by accuracy alone.

## 6. R-G: the registered recurrence is a state-coordinate equivalence control

The fixed update is

$$
h_t=
\begin{cases}
x_t,&m_t=1,\\
h_{t-1},&m_t=0.
\end{cases}
$$

With the same $w$, terminal decoder and reward update, the map
$h\leftrightarrow e$ is an exact state relabeling. It has the same 16 live
reals and the same 8 learned classifier parameters as R-E. Calling the state
recurrent does not add learned gating, temporal abstraction or causal
selection.

Although A3 calls this control “exactly compute-matched,” the contract fixes
only the $2d$ persistent-coordinate ledger and the semantic updates. It does
not preregister primitive operations, memory traffic, serialization cost,
factor reconstruction, implementation language or a FLOP-counting convention.
The supported description is therefore **state-coordinate-matched**, not
exactly compute-matched. Any runtime or FLOP equivalence claim needs a separate
cost model and benchmark.

A standard learned GRU would not be compute-matched. With hidden size 8 and
input size 9 including the marker, three input maps, recurrent maps and biases
already contribute

$$
3\left(8\cdot9+8\cdot8+8\right)=432
$$

gate parameters before the 8 classifier weights. Such a model requires a new
budget and tuning contract. The V18 control intentionally has zero learned
gate parameters.

Numerical killing tests require exact hold behavior, not approximate retention:
use both cue signs on every coordinate, 128 dense distractors, snapshot
continuation and deliberate one-ULP hidden-state corruption. The deliberate
corruption must be detected by the final exact classifier or state
certificate.

## 7. R-A: soft attention is not an exact learned-selector result

A minimal streaming marker attention can be written as

$$
n=\sum_t \exp(\beta m_t)x_t,\qquad
Z=\sum_t \exp(\beta m_t),\qquad
e_\beta=\frac nZ.
$$

For one cue and $K$ distractors,

$$
e_\beta
=\frac{\exp(\beta)x+\sum_{k=1}^{K}r_k}
       {\exp(\beta)+K}.
$$

For every finite $\beta$, the cue coefficient is strictly below one and the
distractor coefficient is nonzero. Consequently the registered four updates
do not yield the exact identity $w=\theta$, even before considering
adversarial distractor alignment. Increasing $\beta$ after seeing
development scores creates a temperature search and risks exponential
overflow. Stable log-sum-exp removes overflow but not leakage or the exact
coefficient defect.

A hard mask $1[m_t=1]$ eliminates both defects, but then event selection is
task-supplied and the route is R-E/R-G. Training a hard gate from terminal
reward would introduce a discrete estimator, initialization, exploration
budget and optimizer choices absent from V18. Removing the marker entirely
would be the scientifically meaningful learned-selection task, but it changes
the data-generating assumptions and needs a new contract with histories that
rule out sparsity, position and norm shortcuts.

Thus R-A is a next-route hypothesis, not an implementation recommendation for
the exact V18 gate.

## 8. R-P: policy gradient cannot inherit the exact label-decoder theorem

For the standard binary log-linear policy

$$
\pi_w(a\mid e)
=\frac{\exp(a w^Te)}{2\cosh(w^Te)},
\qquad
\nabla_w\log\pi_w(a\mid e)
=\left(a-\tanh(w^Te)\right)e,
$$

a reward-only REINFORCE step is

$$
w^+
=w+\alpha R\left(a-\tanh(w^Te)\right)e.
$$

Suppose coordinate $j$ starts with $w_j=0$. If the sampled action is wrong,
$R=0$ and that visit makes no update. Conditional on remaining at zero, four
wrong actions have probability $2^{-4}=1/16$ under the ordinary independent
policy draws, whose namespace must be independent of the fair cue signs and
teacher. On that event $w_j=0$ after all four visits, so the exact
classifier-defect gate fails. This is a killing event with positive
probability, not merely a concern about mean sample efficiency.

Using $a(2R-1)$ inside a custom estimator imports the exact hidden-label
decoder from R-E. It may be a useful hybrid, but it is no longer evidence that
ordinary stochastic policy gradient discovered the same update. Temperature,
learning rate, baseline, entropy weight, clipping, rollout count and RNG
coupling are all look-elsewhere dimensions. None is registered for V18.

Further killing tests are exhaustive action-sequence enumeration for one
coordinate, separate action/task RNG namespaces, saturated logits, and
comparison of expected progress with the probability-one final-defect gate.
Mean accuracy cannot substitute for the registered finite-seed rate of one.

## 9. R-C: a basis-table shortcut defines the target-awareness ceiling

Because training cues are exactly signed coordinate vectors, an episodic
finite-state record $(j,s)$ is sufficient. After reward it can assign

$$
\theta_j=\widetilde y s
$$

in one visit. This route is numerically simple and uses only four episodic
bits, but it was constructed from the coordinate support and bypasses the
registered incremental rule. It is neither coordinate-free nor a general
credit mechanism.

R-C is retained as an audit adversary: an implementation that appears to use
a dense eligibility vector may actually extract an argmax index and sign.
Field introspection, a dense rotated training-cue killing fixture and
serialization of the episodic state expose that shortcut. It must not be
promoted under A1--A3.

## 10. Controls and state-aliasing boundary

The strict original-space metric control has 36 coordinates per component and
$36N$ for registered finite ensembles. Its large state count does not supply
an odd cue carrier: the contract's full-$GL(8)$ pointwise covariance
candidate implies the same episodic state for $+e_j$ and $-e_j$.
Because $s$ is fair and independent of $\theta_j$, the decoded label
$y=s\theta_j$ is conditionally fair after that aliasing and conveys no
teacher-bit information. Permutation-invariant replication preserves the
paired aliasing. The killing test is exact serialization equality for every
sign pair and every registered $N$. To turn this into paired realized
accuracy exactly $1/2$, both branches must start from the same checkpoint
and couple the **entire** post-checkpoint trajectory: distractors, nuisance
inputs, state-transition draws, policy draws and update RNG. Current D2 names
only the checkpoint, transactional evaluation and terminal policy randomness;
that is insufficient if any intermediate source differs. Without the full
coupling the theorem supports expected accuracy $1/2$, not necessarily the
registered exact finite realized score.

The no-trace and trace-lesion controls expose the other boundary. Once the
episodic vector is zero at reward,

$$
\Delta w=\eta\widetilde y\,0=0.
$$

The reward still contains a decodable label, but there is no retained feature
to which it can assign credit. On exact $q,-q$ query pairs, the resulting
common action has accuracy $1/2$. Immediate reward inversion tests the
complementary failure while preserving timing and the trace. It gives
$\widetilde y'=-y$, so every coordinate visit adds
$-\eta\theta_j e_j$, four epochs give $w=-\theta$, and every registered
nonzero-margin dense query is classified oppositely for accuracy zero. No
reward buffering, future inspection or cross-episode permutation is involved.

These controls establish necessity within the registered construction. They
do not prove that eligibility vectors, homogeneous factors or hard latches are
necessary for every delayed-reward problem.

## 11. Look-elsewhere ledger

R-E, R-H and R-G introduce no route search inside V18. Dimension, epoch count,
learning rate, delay set, marker rule, homogeneous coefficient, decoder and
tie action are fixed by the contract. Their comparison may report state cost
and certificates, but must not select a winner from confirmation accuracy.

The following choices are intentionally unregistered and cannot be tuned
against the V18 development block and then scored on its confirmation block:

- attention temperature, scorer features, hard/soft estimator and optimizer;
- GRU width, depth, gate parameterization and truncation horizon;
- policy temperature, baseline, entropy regularization, clipping, rollout
  count and reward transformation;
- table encodings or basis-specific preprocessing;
- any post hoc lesion other than the registered immediate transformation
  $R'=1-R$.

Any implementation claim for learned event selection or stochastic
policy-gradient learning needs a new run, a fresh confirmation block and a
compute/sample budget fixed before development results are inspected.

## 12. Single recommendation and claim boundary

**Implementation recommendation: R-E, explicit eligibility vector.**

It is exact, smallest among the registered real-vector routes, numerically
transparent and makes the delayed credit carrier directly auditable. R-H
should still be implemented as the registered geometric representation
comparison, and R-G as the registered state-coordinate equivalence control.
This recommendation is conditional on a successor contract repairing the
whole-trajectory paired coupling; it does not authorize confirmation for the
blocked current run. R-A and R-P should not be added to a scored implementation
because their necessary search spaces are unregistered and their standard
forms fail the exact gate. R-C should remain a killing fixture only.

This recommendation assigns no theorem, gate or promotion status. Even a
fully passing V18 would establish only a reward-decoded delayed linear-credit
primitive under a public relevance marker. It would not establish learned
event selection, noisy-reward credit assignment, arbitrary recurrent
learning, biological fidelity, cosmological identity or AGI.
