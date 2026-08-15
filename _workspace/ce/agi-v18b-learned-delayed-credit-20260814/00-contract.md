# AGI V18b reward-decoded delayed linear credit contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v18-learned-delayed-credit-20260814`

## 1. Question, correction, and authorized scope

V17 proved a sign-aliasing no-go for an original-space SPD metric and built a
homogeneous signed-cue memory. V18 then proposed a delayed reward-learning
test, but its exact paired-score theorem coupled only terminal policy
randomness. A complete counterexample used different post-checkpoint
distractors or update seeds in the two branches. V18 is therefore abandoned;
none of its confirmation seeds were opened.

V18b asks the repaired finite question: can a marked cue be retained across a
long distractor delay, combined with the learner's own binary action and a
later binary reward to recover supervision, and accumulated into a persistent
linear classifier that generalizes from coordinate cues to unseen dense
compositions?

The public cue marker, exact episodic latch, teacher dimension, and binary
reward semantics are task axioms. The run does not learn event selection. It
tests deterministic synthetic credit assignment, not noisy reward,
policy-gradient learning, arbitrary tasks, semantic OOD, biological fidelity,
cosmology, infinite-SCC intelligence growth, or AGI. `AGI GO` is forbidden.

All registered quantities are dimensionless.

## 2. Registered task and randomness

[Definition V18b-D1] Fix $d=8$. For scored seed $n$, draw the coordinates of
the hidden teacher independently and uniformly,

$$
\theta\in\{-1,+1\}^{d}. \tag{V18b.1}
$$

Training has four epochs. Each epoch visits every coordinate once in a
seed-fixed random order. At a visit to $j$, an independent fair Rademacher cue
sign $s$ gives the marked unit cue

$$
x=s e_j,\qquad y=\operatorname{sign}(\theta^Tx)=s\theta_j. \tag{V18b.2}
$$

The label is hidden. After the cue, the agent receives $K$ unmarked unit
Rademacher distractors, where $K\in\{4,8,16\}$ is drawn independently. It then
receives a zero terminal observation, chooses deterministic
$a\in\{-1,+1\}$, and only afterward receives

$$
R=1[a=y]. \tag{V18b.3}
$$

Teacher bits, epoch orders, cue signs, delays, distractors, evaluation queries,
lesions, update randomness, ensemble randomness/topology, and policy
randomness have separate deterministic seed namespaces whose joint base tuple
is independent of $\theta$ except where the task equations explicitly use
$\theta$. Any redraw has a finite registered cap and fails closed when
exhausted.

[Definition V18b-D2] Evaluation uses 128 integer Rademacher vectors
$r\in\{-1,+1\}^d$. The integer margin $m=\theta^Tr$ is computed before any
normalization; $m=0$ is redrawn up to 1,024 attempts and then fails closed.
Each accepted $q=r/\sqrt d$ is evaluated together with $-q$, giving 256
queries. Coordinate cues are the only training support; these dense vectors
are held-out compositions. Evaluation reward is scored but never updates the
learner. The registered long delay is $K=128$, with a delay-zero control.

[Definition V18b-D3: whole-trajectory paired coupling] Every $q,-q$ member
starts transactionally from the byte-identical post-training checkpoint.
Their complete post-checkpoint nuisance tuple is the same realization and in
the same order: delay, all distractors, messages, topology, update/ensemble
seeds, and terminal-policy seed. Evaluation cannot commit state changes. A
strict control uses a fixed-seed pointwise sign-even update and the same fixed
action function in both branches. Therefore an equal paired state implies the
same realized action. Without this full coupling the general stochastic claim
is only equality in law and expected accuracy $1/2$; no exact realized theorem
is authorized.

[Definition V18b-D4: temporal credit] A scored learner must keep its classifier
$w$ byte-identical at episode start, after the marked cue, after every
distractor, and immediately before reward, for all 32 training episodes. Only
the post-reward transition may change $w$. That transition must read the
current episodic state created at the marked cue. Supplying $y$, $\theta$, a
terminal cue replay, a cached cue outside the declared state, or a
cue-correlated private seed is forbidden.

## 3. Learners and controls

[Axiom: model choice V18b-A1] The explicit eligibility learner has a
cross-episode classifier $w\in\mathbb R^d$, an episodic trace
$e\in\mathbb R^d$, and one discrete active/cleared tag. Initially $w=e=0$.
A marked cue writes $e=x$; unmarked distractors are exact no-ops. The terminal
action is

$$
a=\operatorname{tie}_{+1}(\operatorname{sign}(w^Te)). \tag{V18b.4}
$$

After receiving reward, it computes

$$
\widetilde y=a(2R-1),\qquad
w^+=w+\eta\widetilde y e,\qquad \eta=\frac14, \tag{V18b.5}
$$

then atomically clears $e$ and the active tag.

[Axiom: model choice V18b-A2] The homogeneous learner shares the same $w$ and
reward rule. Its only episodic real-valued memory is the canonical factor of
one $G\in\operatorname{SPD}(d+1)$. A marked cue uses the registered V17
analytic write

$$
z=(x,1),\qquad G=I_{d+1}+\frac12zz^T, \tag{V18b.6}
$$

and reads $e=2G_{1:d,d+1}$. It stores no second copy of $x$. After reward, the
factor is atomically reset to the canonical identity and the active tag is
cleared. Unmarked distractors are exact factor no-ops. At $d=8$, the factor has 45 independent real coordinates (81 dense
serialized entries), nine more than an original SPD(8) factor; $w$ adds eight
declared classifier coordinates. The V18b production module implements this
closed formula self-contained and imports no V16/V17 repository module.

[Axiom: model choice V18b-A3] The hard-latch equivalence control has $w$, a
hidden vector $h\in\mathbb R^d$, and an active tag. It performs the same
registered writes and reward transition as A1 with $h$ in place of $e$. It is
state-coordinate-matched to A1 at $2d$ real coordinates, not FLOP-, wall-time-,
or operation-graph-matched. V18b makes no compute-superiority claim.

[Definition V18b-D5] The strict metric control may persist only an
original-space SPD state updated by a fixed-seed pointwise full-$GL(d)$
sign-even cue map and sign-independent messages. It has no vector/covector,
homogeneous coordinate, raw cue replay, cue-dependent seed, or hidden role
state. Registered finite ensemble sizes are
$N\in\{1,2,4,8,16,64\}$; aggregation is a deterministic sorted-multiset
function invariant to member permutation. The registered implementation is
deterministic after its fixed seed and obeys D3.

[Definition V18b-D6] The no-trace control has the same classifier budget but
uses $e=0$ at reward. The trace lesion independently sets A1's trace to zero
and resets A2's factor to canonical identity immediately before reward; both
operations preserve the active tag and therefore make the decoded current
eligibility exactly zero without leaving either state space. The reward-inversion lesion is applied
independently to A1 and A2 and feeds $R'=1-R$ immediately after each action;
the update must read the lesioned current state and may not use a cached cue.
The intact and lesioned learners otherwise receive byte-identical streams.

## 4. Claims to prove, narrow, or kill

### E1. Binary reward decodes the label

[Open theorem V18b-E1] For deterministic $a,y\in\{-1,+1\}$ and
$R=1[a=y]$,

$$
a(2R-1)=y. \tag{V18b.7}
$$

This does not automatically extend to noisy, scalar, censored, or partial
reward.

### E2. Exact delayed learning and composition

[Open theorem V18b-E2] Under D1 and A1, each visit to coordinate $j$ adds

$$
\Delta w=\eta\,y x=\eta\theta_j e_j, \tag{V18b.8}
$$

independently of cue sign, action, and delay. Four visits with $\eta=1/4$
give $w=\theta$. Thus every D2 nonzero-margin dense query is classified
exactly. The claim transfers to A2 and A3 only after proving their episodic
readout equals $x$ and their classifier timing obeys D4.

### E3. Strict even metric no-go, with the correct quantifier

[Open theorem V18b-E3] For each fixed permissible update seed, the strict
metric state reached from $q$ equals that reached from $-q$. Under D3 the
complete paired trajectory and deterministic aggregate/action are equal.
Since the two labels are opposite, exactly one branch is correct; realized
paired accuracy is exactly $1/2$ seed-by-seed and ensemble-size-by-ensemble-
size. Without D3, only equal conditional action laws and expected accuracy
$1/2$ follow. The statement is a symmetry no-go for this registered aliasing
task, not a capacity theorem for exact-real SPD coordinates. A countable SCC
extension requires the V17 joint-coupling, projective-compatibility, and
measurability hypotheses; finite experiments do not establish it.

### E4. The homogeneous cross block is the carrier

[Open theorem V18b-E4] Equation V18b.6 has cross block $b=x/2$, hence
$e=2b=x$. Deleting the homogeneous row and column leaves
$I+xx^T/2$, which is even in $x$ and falls under E3. The added nine independent
real coordinates package an eight-component covector and one scalar; one
factor is a serialization choice, not proof of one semantic degree of freedom.

### E5. Registered lesions have exact predictions

[Open theorem V18b-E5] A1/A2 trace deletion and the no-trace control give
$\Delta w=0$, so paired dense accuracy is $1/2$. Immediate reward inversion
gives $\widetilde y'=-y$, hence after four epochs $w=-\theta$ and accuracy is
zero on every nonzero-margin query. These are consequences of the registered
deterministic task, not a general causal-credit theorem.

[Hypothesis V18b-H1] The production learners and independent evaluator pass
the preregistered development and once-opened confirmation gates. A pass is a
narrow finite synthetic reward-decoded delayed-credit result only.

## 5. Frozen seed blocks and protocol

Development seeds are 1,821,000--1,821,063. Confirmation seeds are
1,822,000--1,822,255. Confirmation may be opened once only after the contract,
self-contained production code, public export, evaluator, thresholds, and
development result are SHA-256 sealed. During confirmation, the evaluator
loads production directly from its sealed file without importing the
`reality_stone.clarus` package. Immediately before every seed access and again
before result write, it fails closed unless every loaded module whose resolved
`__file__` lies under the repository root belongs to the sealed execution set.
Standard-library and installed third-party dependencies are outside that
repository-closure claim; their Python and NumPy versions are recorded.

There is no hyperparameter search: $d=8$, four epochs, $\eta=1/4$, training
delays $\{4,8,16\}$, evaluation delays 0 and 128, 128 accepted paired base
queries, redraw cap 1,024, ensemble sizes, and tie action $+1$ are fixed.
Development may repair code, numerical defects, and sealing only. Confirmation
stores every seed summary needed for independent rescoring.

Candidate and controls receive byte-identical task streams and event timing.
Reference teacher generation, integer-margin filtering, label computation,
reward decoding, timing assertions, lesions, and scores are implemented in the
evaluator without calling production helper formulas.

## 6. Preregistered gates

### G-MATH

E1--E5 close with no open P0/P1. Any stochastic/noisy extension remains
explicitly incomplete.

### G-DIMENSIONLESS

$w^Te$, $\eta$, reward, regret, accuracies, normalized Rademacher vectors, and
delay ticks are dimensionless. Any introduced logarithm or exponential has a
positive dimensionless argument; any normalization names a positive reference.

### G-LEARN

For every one of 256 confirmation seeds, separately for A1 and A2:

- all declared outputs are finite;
- pretraining paired accuracy is exactly $0.5$;
- post-training accuracy is exactly $1$ and regret is zero;
- for all 32 episodes,
  `w_start == w_after_cue == w_after_every_distractor == w_pre_reward` byte for
  byte, followed by the independently predicted post-reward delta;
- final $\lVert w-\theta\rVert_\infty\le10^{-12}$;
- A2 has one factor-valued episodic memory, no hidden cue field, recovers
  $e=x$ within $10^{-12}$, and resets to canonical identity after every reward.

### G-DELAY-COMPOSE

- A1, A2, and A3 accuracy at delays 0 and 128 is exactly 1, with zero
  delay-induced difference;
- strict metric, no-trace, and every registered strict ensemble have realized
  paired accuracy exactly $0.5$ for every seed and every registered $N$;
- paired strict state and aggregate serializations are byte-identical for
  $q,-q$ under the complete D3 nuisance tuple.

### G-CAUSAL-LESION

Independently for A1 and A2 on every seed:

- trace lesion accuracy is exactly $0.5$;
- immediate reward-inversion accuracy is exactly 0;
- intact-minus-trace accuracy is $0.5$ and intact-minus-inversion is 1.

The no-trace control is also exactly $0.5$.

### G-NUMERIC

Focused tests cover both cue signs and all coordinates, reward order,
classifier timing on every distractor, atomic clearing/reset, absence of a
hidden cue field, cross-block recovery, homogeneous deletion, deterministic
ties, snapshot continuation, integer-margin redraw/cap, whole-trajectory
paired coupling, namespace separation, all ensemble sizes, zero/nonfinite/type
rejection, public export identity, exact required manifest paths, duplicate
key/traversal rejection, canonical-root/isolated-production binding, loaded
repository-module closure enforcement, exclusive receipt and
result creation, active single-use confirmation capability, closing rehash,
and fail-closed missing episode/branch coverage. All required tests and static
checks pass.

## 7. Decision rules

- `V18b STRICT-METRIC DELAYED-CREDIT NO-GO CLOSED` requires E3,
  G-MATH, G-DELAY-COMPOSE strict slices, and G-NUMERIC.
- `V18b REWARD-DECODED ELIGIBILITY NARROW GO` requires G-MATH,
  G-DIMENSIONLESS, G-LEARN, G-DELAY-COMPOSE, G-CAUSAL-LESION, and G-NUMERIC.
- `V18b STOP` applies if any required theorem, integrity condition, or scored
  gate fails.

Neither positive verdict authorizes learned salience, general delayed credit,
noisy reward learning, compute superiority, semantic OOD, recursive-agent
scaling, biological/cosmological identity, or AGI.

## 8. Sealing and reproducibility

The confirmation manifest contains exactly these five paths:

1. this contract;
2. `reality_stone/python/reality_stone/clarus/delayed_linear_credit.py`;
3. `reality_stone/python/reality_stone/clarus/__init__.py`;
4. this run's `artifacts/run_v18b_benchmark.py`;
5. this run's `artifacts/development-results.json`.

The evaluator binds to the canonical repository and isolated sealed production
path, never imports the package initializer during scoring, and enforces the
loaded repository-module closure described above. It rejects
absolute/traversing/duplicate manifest entries; verifies the
exact path set, hashes, development provenance, and fixed protocol; creates an
exclusive receipt before confirmation seed access; issues and consumes an
unforgeable in-process single-use capability; rehashes all sealed files before
exclusive result write; and refuses every second opening. Receipt/result files
are procedural evidence, not cryptographic secrecy or an external signature.

Per-seed output retains the teacher, final classifiers, accepted query integer
vectors and margins or sufficient exact counts, all paired branch counts,
learner/control/lesion/timing fields, ensemble-size results, finite checks, and
certificate fields needed for independent rescoring without reopening seeds.
Any post-seal code or contract change requires a new run and fresh seed block.
