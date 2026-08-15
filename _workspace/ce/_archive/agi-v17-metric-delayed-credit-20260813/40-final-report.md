# AGI V17 metric-only delayed cue: final report

Status: COMPLETE

## 1. Question and exact verdict

The run asked whether the original-space Riemannian metric $g_t$ alone can
retain the orientation of an earlier signed cue until a later binary decision,
and whether arbitrarily many recursive SCC copies of that same restricted
state can restore information each component loses.

The strict answer is no, and the smallest implemented one-factor escape in
this contract succeeds only by adding disclosed geometric memory:

$$
\boxed{\text{V17 METRIC-ONLY NO-GO CLOSED}}
$$

$$
\boxed{\text{V17 HOMOGENEOUS LIFT NARROW GO}}.
$$

Both verdicts passed every preregistered gate. They are not contradictory: the
strict theorem concerns $g\in\operatorname{SPD}(3)$ with full $GL(3)$
covariance, while the successful candidate stores
$G\in\operatorname{SPD}(4)$ under a declared homogeneous splitting. The AGI
verdict is `AGI STOP`; `AGI GO` is forbidden and not authorized.

## 2. Mathematical closure

[Theorem] Full $GL(d)$ covariance contains $J=-I$. This transformation sends
$x$ to $-x$ but fixes a covariant two-tensor, forcing every admissible
metric-only update to satisfy

$$
U(g,-x,c)=U(g,x,c).
$$

The result holds pathwise for almost every allowed fixed random seed and hence
also in law. Covariance only after averaging would be insufficient and is
excluded by the contract.

[No-go theorem] Coupling the two sign branches with the same public reference,
initial state and whole independent seed family gives identical post-cue and
terminal metric states. Every permitted terminal policy therefore has balanced
accuracy exactly $1/2$ and expected regret $1/2$ on the registered task.

[No-go theorem] Replicating the same restriction does not create cue-odd
information. Equality propagates through every finite SCC event depth and
finite component count. A countable extension holds only for a defined
measurable compatible product/trajectory system and measurable terminal
kernel. No unconditional theorem or runtime experiment for infinite event
depth is claimed.

[Theorem] Exact solution needs one bit of conditional separation,
$I(S;G_T\mid U)=1$, for the balanced two-class task. This neither says that
$I(S;G_T)=1$ nor gives a coordinate lower bound or a general capacity bound for
exact-real tensors.

## 3. What the homogeneous escape adds

[Axiom: model choice] The lift declares a dimensionless anchor coordinate and
uses

$$
z_s=(su,1),\qquad y_a=(au,-1),\qquad
G_1=I_4+\frac12z_sz_s^T.
$$

[Derived] The independent readout formula

$$
y_a^TG_1y_a=2+\frac12(sa-1)^2
$$

gives costs 2 and 4 and selects $a=s$ with exact margin 2. Under spatial charts
it is covariant only through $A=\operatorname{diag}(J,1)$ with the initial
metric transported and no reprojection.

[Definition] One canonical factor is persisted, but its ten independent real
coordinates are four more than the original metric's six. The extra block is
a three-component covector plus a scalar anchor coupling, so the lift contains
real orientation memory. It is not a proof that only the original $g$ changes.
An explicit eligibility covector would use nine total coordinates and retain
full original-space $GL(3)$ covariance, but would require two semantic fields;
therefore the homogeneous route is an implementation-shape choice, not a
minimality theorem.

Deleting the homogeneous coordinate removes the cue-odd block and restores
the strict sign-paired tie. This killing result identifies precisely where the
escape lives.

## 4. Sealed confirmation result

[Prediction protocol] Development seeds 1,719,000--1,719,063 checked the fixed
analytic design with no rate or representation search. Five artifacts were
then SHA-256 sealed, an exclusive receipt was created before confirmation seed
access, and seeds 1,720,000--1,720,255 were run once. The manifest, receipt and
result hashes are respectively:

- `898ab27369cd5580fc0b7f67f44fe048bfbf6361d55d574ce652b9ef571a63d1`;
- `2022f4778ae47e49627913442de61064234586f707762dea12678291f0a81ed1`;
- `35324a7fe1f4570a5d66c3cca6ed65298191c1c651eec73cdec24dbca677e01f`.

[Numerical result] The stored confirmation summaries give:

| Registered result | Value |
|---|---:|
| strict serialized state equality | $1.0$ |
| strict action-law equality | $1.0$ |
| strict balanced accuracy / regret | $0.5/0.5$ |
| finite ensemble result for $N=1,2,4,8,16,64$ | same $0.5/0.5$ no-go |
| lift action accuracy, 512 signed branches | $1.0$ |
| lift mean regret | $0$ |
| minimum lift margin | $1.999999999999996$ |
| transported action agreement | $1.0$ |
| maximum relative quadratic-cost defect | $4.4408920985006072\times10^{-15}$ |
| persistent factor fields / optimizer fields | $1/0$ |
| augmented state coordinates | $10$ ($+4$) |

An independent read-only rescore confirmed 256 unique consecutive seeds and
all stored per-seed strict, lift and finite-ensemble entries without importing
the evaluator or reopening the block. Its recomputed summaries matched the
result, all 17 gate booleans were true and open P0/P1 findings were 0/0. Every
bound artifact still matches the manifest.

[Incomplete: integrity boundary] The receipt is a local procedural control, not an
externally signed commitment. The JSON has no internal execution timestamp,
and raw lift factors/certificate objects were not stored per seed, so those
parts were checked against the sealed source and aggregate certificate rather
than independently reconstructed from per-seed matrices.

## 5. Verification state

[Numerical result] The focused V17 plus inherited metric/dimensionless suite
passed 100 tests, and the 16-file SCC/metric-related slice passed 337 tests.
Ruff passed the V17 production, evaluator, test and dimensionless changes;
`__init__.py` passed with its eight pre-existing F401 warnings ignored.
Compileall, the seal diff check and the CE contract/lanes/gate hooks passed.

[Incomplete] A repository-wide test run reached 73% under a 600-second limit,
showed inherited failures/errors, and timed out before producing a valid final
total. The result is therefore focused and related-slice green, not
repository-wide green.

## 6. Evidence boundary

This run closes one symmetry question: a fully $GL(d)$-covariant
original-space Riemannian metric update cannot preserve cue polarity, and no
side-channel-free recursive replication of that same even state can recover it.
It also demonstrates one exact synthetic escape by adding a homogeneous
covector/scalar block.

It does not establish:

- general delayed credit assignment or learning which past event caused a
  delayed reward;
- memory for unknown terminal query directions, multiple cues or arbitrary
  horizons;
- noisy or finite-precision long-term robustness;
- an unconditional countable-agent theorem or an infinite event-depth SCC;
- raw perception, semantic OOD, planning, autonomous tool use or
  consciousness;
- biological fidelity, a brain--cosmos identity or a spacetime metric; or
- AGI.

The registered reward is revealed after the action, but the successful route
performs a target-aware analytic cue write before that delay. It is delayed
memory, not learned causal attribution. More recursive agents do not change
this boundary.

## 7. Next falsifiable breakthrough

[Prediction] The next useful run should test an explicit odd memory state,
rather than increase the number of strict metric-only SCC copies. A clean V18
contract would compare the homogeneous lift with an explicit eligibility
covector and a compute-matched recurrent baseline on variable terminal query
directions, multiple intervening distractors and delayed rewards. The write
must be learned from reward rather than analytically supplied, and the
confirmation block should test unseen cue--query relations.

The killing criterion should be state aliasing: construct two histories with
the same candidate state and current observation but different correct credit
updates. If such histories exist, that state is not Markov-sufficient for the
task. Only after a finite system survives that test should a later run define a
measurable SCC limit and ask a separate countable or infinite-depth question.
