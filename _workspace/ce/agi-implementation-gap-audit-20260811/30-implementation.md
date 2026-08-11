# Dependency-ordered implementation plan

Status: COMPLETE

## Checkpoint 0 — preserve the successful path

Freeze the current Layer A-E runtime, `RuntimeAgent`, sleep pipeline, warm snapshot, and all locked V1-V8/ACBSM artifacts as regression baselines. Do not open fresh ACBSM seeds yet.

Exit gate: current focused tests pass; all new components have feature-off reduction to the checkpoint.

## Checkpoint 1 — close one model's contracts

Target model: `ClosedLoopBrainRuntime`.

1. Add typed `CriticScores` and one `critic_scores(state, goal, output, cfg)` backend contract.
2. Add a causal local temporal residual/belief state with uncertainty and a minimal world-model interface.
3. Feed critic prediction error into DA/NE/5HT/ACh and apply the resulting modulation to actual runtime/action/sleep parameters each tick.
4. Add explicit memory operations and provenance.
5. Add cold checkpoint and append-only journal.
6. Add a stateful LLM sidecar interface and a non-template environment/tool adapter.

Keep planner, metacognition, diffusion, graph/cloud, and CFC off in this checkpoint.

Exit gate: zero future leakage; common score scale; off-limit equivalence; deterministic save/restore continuity; complete causal logs; full augmented-state stability diagnostic.

## Checkpoint 2 — prove learning before adding architecture

Use a small task suite with partial observability, delayed reward, memory dependence, and a distribution shift. Compare matched variants:

- checkpoint baseline;
- + belief/world model;
- + neuromodulator feedback;
- + memory operations;
- + corrected credit assignment.

STDP is not accepted merely because weights change. It must improve held-out return or prediction while passing stability and no-leak guards. If it again has no effect, replace its gate with an explicit eligibility-times-prediction-error rule rather than adding more modules.

Exit gate: preregistered task gain on held-out seeds with no stability regression.

## Checkpoint 3 — add planning

Implement short-horizon model-predictive planning over the learned belief state, uncertainty-penalized and receding-horizon. Verify that actions causally alter the environment and that planning beats reactive action selection.

Exit gate: planner gain, calibrated uncertainty, bounded compute, and no future-state access.

## Checkpoint 4 — memory and sleep efficacy

Test episodic/semantic retrieval ablations, restart persistence, continual learning, forgetting, drift, and sleep rollback. Promote memory and sleep only if their task contribution is positive against matched baselines.

## Checkpoint 5 — language and tool loop

Connect the stateful LLM sidecar and real tool adapter. Feed execution results and self-critique into the next persistent state. Then implement curvature-triggered retry and evaluate hallucination/error benchmarks.

## Separate scientific track — explicit SNN

In parallel only after Checkpoint 1 contracts are stable, build the tiny spike-time SNN and test natural activity-ratio emergence. Its result governs CE substrate claims, not the functional-agent release.

## Deferred until the core loop wins

- GaugeLatticeV2/CFC and fused sparse scale-up;
- distributed CloudCell orchestration;
- metacognitive depth and external PCI bridge;
- multimodal grounding and score-diffusion generation;
- reopening sparse causal bridge families or fresh ACBSM validation.

