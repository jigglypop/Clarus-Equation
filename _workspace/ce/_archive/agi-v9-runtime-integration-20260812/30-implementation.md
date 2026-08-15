# V9 runtime integration implementation

Status: COMPLETE

## Outcome

The finite nested-SCC controller is now an explicit opt-in action path inside
`RuntimeAgent`. When enabled, the selected action is obtained only from the policy read with
the controller-issued state token. The default path remains disabled and unchanged.

This is an executable research-agent integration, not an AGI, biology, convergence, or task-
utility claim.

## Data path

```text
finite runtime observation
  -> cosine_action_evidence(observation, fixed action embeddings)
  -> CausalEvent(controller.tick + 1, evidence)
  -> AdaptiveTowerController.observe
  -> TowerStateToken
  -> AdaptiveTowerController.read_policy(token, action_mask)
  -> RuntimeAgentStep.action_index = policy.selected_action
```

The encoder produces one dimensionless coordinate per action. Zero norms map to zero and all
coordinates are clamped to the Cauchy--Schwarz range $[-1,1]$. The controller's existing
normalizer, state seal, exact tick, token, and policy-simplex checks remain active.

## Files and API

- `agent.py`
  - added `RuntimeAgentConfig.nested_scc_enabled`, default `False`;
  - added `cosine_action_evidence`;
  - added optional constructor injection `nested_scc_controller`;
  - added `action_mask` to `step` only for the V9 branch;
  - added token, policy, and evidence to `RuntimeAgentStep`;
  - rejects simultaneous belief-control and V9 control;
  - replaced the old “executable AGI core” docstring with a research-agent boundary.
- `clarus/__init__.py`
  - exports the V9 generator/controller/token/policy types and evidence encoder.
- `nested_scc_tower.py`
  - removed dormant `depth_error_tolerance` and `hysteresis_ticks`;
  - renamed `generated_parameter_count` to `serialized_operator_scalar_count` and documented
    that it is not capacity, trainable parameters, or MACs.
- `tests/test_agent.py`, `tests/test_nested_scc_tower.py`
  - added output-mediation, same-current-input/different-history, mask, invalid-input,
    composition, legacy-path, package-export, and cleanup tests.
- `nested_scc_runtime_agent_demo.py`
  - added a deterministic three-tick non-evidence demo.

## Fail-closed boundaries

- Invalid mask, nonfinite supplied observation, or generator seal is checked before
  `BrainRuntime.step`.
- Shell width must equal the actual action-embedding count.
- Supplying a V9 controller while the flag is disabled cannot affect the legacy path.
- A mask supplied to a non-V9 agent is rejected instead of silently ignored.
- The V9 branch cannot fall through to legacy similarity argmax or belief-control output.
- Controller snapshots remain same-process only; no new persistence claim was added.

## Cleanup scope

Only the two proven-dormant configuration fields, the misleading metadata name, and the AGI-
claiming docstring were cleaned. The worktree already contained 17 modified, 140 deleted, and
27 untracked paths from other work at the cleanup audit point. None was reverted, deleted, or
garbage-collected by this implementation.

## Locked source hashes

| File | SHA-256 |
|---|---|
| `agent.py` | `34983C5A3AFA8A1DED9DA302CB44D4C6333AF78653626DEB80DEB579E70A64AC` |
| `clarus/__init__.py` | `6B2E16C1859B0107A9B77EADCD1E90D9E0BE00F1F740D3316F2EB04D8F8F08FA` |
| `nested_scc_tower.py` | `3C101AD966FE9AEE8D1F41E9319AB55D88D027B94FEF4584203A416F081652E7` |
| `adaptive_scc_tower_controller.py` | `9204DDDBF893A0C15DC34DE503E1E9C853A14FAAD164FA5AFB0F32BB1822E028` |
| `test_agent.py` | `0851595453E60C536D55648DFAE454186653062FBC34950A784D0FEFBB8F923C` |
| `test_nested_scc_tower.py` | `EFF7DF726B78CCE777106835DC45546FCB424B547E40A226DC11E2FD7D8EC126` |
| `test_adaptive_scc_tower_controller.py` | `7D7D3FBC04E441E7F33711C5B1D0026E9FB2EADE24142D01FB62B23BCFF877ED` |
| `nested_scc_runtime_agent_demo.py` | `1AB2F48503B6EEAFE3489BC1EFCC962241FB7F9F4392BA6C268E3DECB452CEDA` |

## Unchanged evidence status

V8 locked tests, ACBSM fresh blocks, V9 development seeds, and confirmation data were not
opened. Development remains `0/256 BLOCKED`.
