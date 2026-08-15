# AGI V18 reward-decoded delayed linear credit contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v17-metric-delayed-credit-20260813`

## 1. Question and authorized scope

V17 proved that a fully $GL(d)$-covariant original-space SPD metric update is
even in the cue and therefore cannot preserve cue polarity. It implemented a
homogeneous one-factor escape, but the successful write was analytic and the
reward was not used to learn a cross-episode rule.

V18 asks the next narrower question: can a finite agent retain a cue across a
long distractor delay, receive only binary reward after acting, decode the
supervision carried by that reward, and update a persistent classifier that
generalizes from signed coordinate cues to unseen dense compositions?

The run compares an explicit eligibility vector, the V17 homogeneous SPD
memory, an exactly compute-matched gated recurrent vector, a strict even-metric
control, a no-trace control and reward/trace lesions. It does not claim to
learn which event is relevant: a public cue marker supplies that selection.
The strongest authorized positive verdict is a narrow reward-decoded delayed
linear-credit primitive. `AGI GO` is forbidden.

All coordinates, costs, rewards, learning rates and time steps are synthetic
and dimensionless. No biological or cosmological identity is in scope.

## 2. Registered task

[Definition V18-D1] Fix $d=8$. For each scored seed, draw the coordinates of a
hidden teacher independently and uniformly from the Rademacher law,

$$
\theta\in\{-1,+1\}^{d}. \tag{V18.1}
$$

A training episode chooses a coordinate $j$ and draws a cue sign $s$ as an
independent fair Rademacher, using namespaces jointly independent of the
teacher, episode order, distractors and evaluation queries. It shows the marked cue

$$
x=s e_j,\qquad y=\operatorname{sign}(\theta^T x)=s\theta_j. \tag{V18.2}
$$

The label $y$ is never supplied directly. After the marked cue, the agent
receives $K$ unmarked dense Rademacher distractors, then a common zero terminal
observation. It chooses $a\in\{-1,+1\}$ and only then receives

$$
R=1[a=y]. \tag{V18.3}
$$

The agent knows its own action. The training schedule contains four epochs;
each epoch visits every coordinate exactly once in a seed-fixed random order,
with independently drawn cue signs. The registered learning rate is
$\eta=1/4$. Training delays are sampled from $\{4,8,16\}$.

[Definition V18-D2] Evaluation contains 256 dense queries in 128 exact
$q,-q$ pairs. Each $q\in\{-1,+1\}^d/\sqrt d$ is drawn independently and
redrawn only when $\theta^Tq=0$. Neither $q$ nor $-q$ appears in the coordinate
training support. The evaluation label is
$\operatorname{sign}(\theta^Tq)$, and the evaluation delay is fixed at
$K=128$. Evaluation reward is scored but does not update the learner. Both
members of every pair start from the same post-training checkpoint, evaluation
is transactional, and any terminal policy randomness uses the same coupled
draw in both branches. Thus a sign-even action law produces the same realized
action for $q$ and $-q$, not merely equal marginal laws.

[Definition V18-D3] A delayed-credit update is registered only when its
persistent classifier changes after reward and not before reward, and the
post-reward update depends on an episodic state created before the distractor
interval. An implementation that is handed $y$, $\theta$, the cue again at
terminal time, or a cue-correlated retained random seed is disallowed.

## 3. Registered learners and controls

[Axiom: model choice V18-A1] The explicit eligibility learner has persistent
cross-episode classifier $w\in\mathbb R^d$ and within-episode trace
$e\in\mathbb R^d$. Initially $w=0$. A marked cue writes $e=x$; unmarked
distractors do not change it. The terminal action is

$$
a=\operatorname{tie}_{+1}\!\left(\operatorname{sign}(w^Te)\right). \tag{V18.4}
$$

After observing reward it computes

$$
\widetilde y=a(2R-1),\qquad
w^+=w+\eta\widetilde y e. \tag{V18.5}
$$

The trace is cleared after the update. The public cue marker and exact latch
are task-supplied structure, not learned event selection.

[Axiom: model choice V18-A2] The homogeneous learner shares the same $w$ and
reward update, but stores the marked cue in one
$G\in\operatorname{SPD}(d+1)$ using the registered V17 write

$$
z=(x,1),\qquad G=I_{d+1}+\frac12zz^T. \tag{V18.6}
$$

It recovers the eligibility vector from the spatial/homogeneous cross block
$b=G_{1:d,d+1}$ by $e=2b$. It must use the production factor state and may not
store a second hidden copy of $x$.

[Axiom: model choice V18-A3] The compute-matched gated recurrent control has
the same $2d$ persistent real coordinates as A1: classifier $w$ and hidden
vector $h$. A marked cue writes $h=x$, unmarked distractors leave $h$ fixed,
and equation V18.5 uses $h$ in place of $e$. It is a strong equivalence
control, not a baseline V18 is required to beat.

[Definition V18-D4] The strict metric control may persist an original-space
SPD state updated by a pointwise full-$GL(d)$-covariant cue map and any
sign-independent messages, but no vector/covector, homogeneous coordinate,
raw cue replay or sign-correlated seed. Registered finite ensemble sizes are
$N\in\{1,2,4,8,16,64\}$ with permutation-invariant aggregation.

[Definition V18-D5] The no-trace control has the same classifier update budget
but its episodic vector is zero at reward. The trace lesion sets the positive
learner's trace to zero before reward. The reward-inversion lesion immediately
feeds $R'=1-R$ to the otherwise identical update after each action; it does not
buffer, permute or inspect future rewards. Hence its decoded label is $-y$.

## 4. Claims to prove, narrow or kill

### E1. Binary reward decodes the hidden label

[Open theorem V18-E1] For $a,y\in\{-1,+1\}$ and $R=1[a=y]$,

$$
a(2R-1)=y. \tag{V18.7}
$$

The theorem is specific to deterministic binary reward and known binary
action. It does not extend automatically to noisy, scalar or partial reward.

### E2. Exact delayed recovery and compositional generalization

[Open theorem V18-E2] Under D1 and A1, every visit to coordinate $j$ adds
$\eta\theta_j e_j$ to $w$, independently of cue sign and delay. After four
complete epochs with $\eta=1/4$,

$$
w=\theta. \tag{V18.8}
$$

Consequently every nonzero-margin dense query in D2 is classified exactly.
The same theorem candidate applies to A2 and A3 if their recovered episodic
vector equals $x$.

### E3. Even metric state cannot learn the signed teacher

[Open theorem V18-E3] Under D4, the paired cues $+e_j$ and $-e_j$ reach the
same episodic metric state, while their labels are opposite for a fixed
$\theta_j$. Conditional on the even state and public coordinate identity,
the reward-decoded label is uniform and contains no information about
$\theta_j$. With the D2 common-checkpoint/common-randomness coupling, strict
metric-only learning and every registered finite ensemble have realized paired
evaluation accuracy exactly $1/2$; without that coupling the theorem gives
only expected accuracy $1/2$. A countable extension is
authorized only under the measurability and joint-coupling conditions already
closed in V17; finite scores are not evidence of an undefined infinite SCC.

### E4. Homogeneous cross block is the added credit carrier

[Open theorem V18-E4] Equation V18.6 has cross block $b=x/2$, so the exact
readout $e=2b$ recovers the cue. Deleting the homogeneous row/column leaves
$I+xx^T/2$, which is even in $x$ and restores E3. For $d=8$, the homogeneous
factor has $45$ ambient real coordinates versus $36$ for the original metric;
the added nine package an eight-component covector plus one scalar.

### H1. Narrow reward-trained capability

[Hypothesis V18-H1] The production eligibility and homogeneous learners will
pass the preregistered development and sealed confirmation gates below,
including reward/trace lesions and $K=128$ delay/composition shift. Passing is
evidence for this finite deterministic task only.

### H2. Structurally distinct next routes

[Hypothesis V18-H2] The route lane must compare at least: explicit eligibility,
homogeneous factor memory, gated recurrence, learned cue-selection/attention,
and a stochastic policy-gradient route. Each route must disclose state and
parameter counts, task-provided markers, target awareness, numerical risks and
a killing test. At most one route may be recommended for implementation.

## 5. Seed blocks and fixed protocol

Development seeds are 1,819,000--1,819,063. Confirmation seeds are
1,820,000--1,820,255 and may be opened once only after the contract,
production module, public export, evaluator, thresholds and development result
are SHA-256 sealed.

There is no hyperparameter search: $d=8$, four epochs, $\eta=1/4$, training
delays $\{4,8,16\}$, evaluation delay 128, 128 paired dense query draws and
the tie action $+1$ are fixed analytically. Development may repair code and
sealing only. Confirmation stores every seed summary.

The evaluator must generate teacher, episode order, cue signs, distractors and
queries from separate deterministic seed namespaces. Candidate and controls
receive byte-identical task streams and action/reward timing. Reference labels,
reward decoding and aggregate scores must be implemented independently of
production helpers.

## 6. Preregistered gates

### G-MATH

E1--E4 close with no P0. Any stochastic or noisy extension not proved is
marked incomplete.

### G-DIMENSIONLESS

$w^Te$, $\eta$, reward, loss, accuracy, delay ticks and normalized dense
queries are dimensionless. Any logarithm or normalization introduced by an
implementation must have a positive dimensionless argument or reference.

### G-LEARN

Across all 256 confirmation seeds:

- eligibility and homogeneous finite-seed rate equal $1$;
- pretraining paired dense-query accuracy equals $0.5$;
- post-training accuracy equals $1$ and regret equals $0$;
- classifier state is byte-identical before reward and changes only after
  reward on at least one training episode;
- the exact final classifier defect
  $\lVert w-\theta\rVert_\infty$ is at most $10^{-12}$.

### G-DELAY-COMPOSE

- eligibility, homogeneous and gated recurrent accuracy at delay 128 equal
  $1$ on dense queries unseen during coordinate training;
- their delay-0 versus delay-128 accuracy difference is zero;
- strict metric and no-trace paired accuracy equal $0.5$;
- all registered strict ensemble sizes remain at $0.5$.

### G-CAUSAL-LESION

- trace lesion accuracy equals $0.5$;
- no-trace accuracy equals $0.5$;
- reward-inversion accuracy equals $0$;
- intact minus trace/no-trace accuracy equals $0.5$, and intact minus
  reward-inversion accuracy equals $1$.

The lesion gate shows dependence on registered trace/reward alignment, not a
general causal-discovery theorem.

### G-NUMERIC

Focused tests cover both cue signs and every coordinate, zero/nonfinite/type
rejection, reward timing, exact trace clearing, homogeneous factor-only state,
cross-block decoding, homogeneous deletion, snapshot continuation, query-pair
balance, seed-role separation, manifest traversal/duplicate-key rejection,
exclusive receipt/result creation and missing-episode fail-closed behavior.
All declared binary64 outputs are finite.

## 7. Decision rules

- `V18 METRIC-ONLY DELAYED-CREDIT NO-GO CLOSED`: E3 and the strict/no-trace
  portions of G-DELAY-COMPOSE pass with G-MATH and G-NUMERIC.
- `V18 REWARD-DECODED ELIGIBILITY NARROW GO`: G-MATH,
  G-DIMENSIONLESS, G-LEARN, G-DELAY-COMPOSE, G-CAUSAL-LESION and G-NUMERIC
  all pass.
- `V18 STOP`: any required theorem, integrity control or scored gate fails.

Neither positive label establishes learned event selection, noisy reward
credit, policy-gradient learning, arbitrary tasks, semantic OOD, infinite-SCC
intelligence growth, biological fidelity, cosmological identity or AGI.

## 8. Sealing and reproducibility

The confirmation manifest must contain exactly this contract, the production
module, public export, evaluator and development result. The evaluator binds
to the canonical repository root and imported production path, rejects path
traversal and duplicate JSON keys, verifies development provenance, creates an
exclusive opening receipt before confirmation seed access, consumes an opaque
single-use access capability, rehashes sealed inputs before exclusive result
write and blocks every second opening.

Per-seed results must retain teacher, final classifiers, paired query counts,
all learner/control accuracies, lesions, timing assertions, ensemble summaries
and finite/certificate fields needed for an independent rescore without seed
reopening. Any post-seal change requires a new run and seed block.
