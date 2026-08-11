# AGI implementation gap audit

Status: COMPLETE

## What is already implemented

- Layer A-E runtime mechanics: activation, inhibition, refractory/adaptation/STP, sparse activity, lifecycle, global modes, sleep pressure, hippocampal tensor memory, and warm snapshots.
- A runnable `RuntimeAgent` tick with discrete/continuous action selection, environment prediction, critic computation, working memory, cerebellar predictor, and causal one-step-delayed STDP gating.
- Sleep cycle plumbing with replay, NREM/REM updates, guards, and rollback.
- STDP mechanism, four-neuromodulator state/effect functions, consciousness proxy monitor, CE relax/decoder primitives, Rust fallback paths, and measurement/gate utilities.

These are mechanisms, not evidence of general intelligence.

## Missing

### Intelligence-producing chain

- A learned, uncertainty-aware world/belief model integrated into `RuntimeAgent`.
- Multi-step planner or receding-horizon action search.
- A real environment/tool action contract; current text environment renders a small fixed set of templates.
- Standard typed self-critic and residual-field contract shared by runtime, agent, STDP, sleep, and logging.
- General task credit assignment that produces held-out learning gain.
- Stateful LLM sidecar/bridge carrying persistent CE state between turns.

### Persistence and memory semantics

- Cold checkpoint containing structure, weights, and long-term memory.
- Append-only live journal and crash recovery.
- Explicit ADD/UPDATE/DELETE/NOOP memory operations, provenance, temporal update, abstention, and deletion audit.

### Canonical architecture and substrate

- Explicit spike-time SNN substrate and natural target-ratio emergence test.
- Perturbative bidirectional channel mixing and canonical curvature/CFC coupling.
- Sparse-native/fused runtime rather than post-hoc masking.

### ACBSM-specific missing mechanisms

- Posterior-SNR/horizon trust that attenuates corrections using covariance.
- Robust Huber or Student-t innovation/noise fitting.
- Multi-horizon objective.
- Episode-hierarchical loading/noise shrinkage.
- Regime/change-point state, explicit environment belief, episodic prior, hypothesis beam, edge adaptation, and planner.
- Full augmented-Jacobian or equivalent weighted-norm stability check.

## Scaffold-only or partially integrated

- Neuromodulators: state equations and effect mapping exist, but they do not close a per-tick feedback loop over canonical runtime parameters.
- Metacognition: a toy contraction step exists; it does not yet alter action, relaxation, or sleep and evaluate its own effect.
- Working memory is primarily a bounded deque; hippocampal recall is tensor similarity retrieval, not audited semantic/episodic long memory.
- `RuntimeTextAgent` is a runnable shell, not learned language reasoning or tool use.
- ACBSM is a separate development predictor, not the agent's world model.
- CloudCell/graph/manifold modules largely measure datasets or formal gates; they are not distributed cognition, planning, or live orchestration.
- Consciousness monitor and PCI regression are proxies/utilities; external PCI validation is absent.

## Implemented but unverified

- Working-memory and cerebellar predictor task advantage.
- Sleep consolidation advantage on continual-learning/forgetting/drift benchmarks.
- Hippocampal retrieval's causal contribution to task success.
- Automatic WAKE/NREM/REM occupancy convergence to the target ratio.
- Hallucination V1's robust value on TruthfulQA/HaluEval/FactScore-style evaluation.
- Standalone decoder/PQ codebook quality and large-scale performance.
- Length-OOD result beyond its one tested axis: semantic, compositional, ICL, and reasoning-depth transfer remain open.

## Implemented and failed

- STDP wiring: next-step A/B was `NO-EFFECT`; held-out guard `FAIL`. Keep disabled by default.
- Natural self-organization to the target activity ratio: forced scheduling can match it, but autonomous convergence is not demonstrated.
- Sparse causal bridge confirmation: V8 failed the parent-anchored confirmatory clause; post-failure routes did not recover it.
- Graph/manifold/shared-gain exploratory routes reported failure or no supported joint gain.

## Evidence caveats

- ACBSM score 67.13 is an inherited eight-training-episode direction screen, not the unopened fresh development seeds.
- ACBSM tests check core behavior, not predictive superiority. One H5 path assertion is tautological, and hidden-state poison is not injected into the predictor. Future-state poison is meaningful.
- V7/V8 pytest coverage emphasizes registration/hash/API/poison checks; it does not independently rerun every full validation gate.

