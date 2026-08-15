# V9 runtime integration research contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v9-nested-infinite-scc-20260811

Mode: light

## Question

Can the already-audited finite V9 nested-SCC controller be connected to the existing
`RuntimeAgent` so that observations causally update persistent tower state and the selected
action is read only from the resulting tower token, while leaving the default runtime path
unchanged?

## Authorized scope

- Add an opt-in V9 action path to `RuntimeAgent`.
- Convert a finite runtime observation and fixed action embeddings into a dimensionless,
  shell-width action-evidence vector by cosine similarity.
- Require the V9 shell width to equal the action count and reject simultaneous belief-control
  and V9 control in this first integration.
- Expose the issued tower token and policy in `RuntimeAgentStep` for causal inspection.
- Remove the dormant `depth_error_tolerance` and `hysteresis_ticks` settings.
- Rename `generated_parameter_count` to a non-capacity metadata name.
- Add focused tests, a deterministic non-evidence demo, and documentation.

## Forbidden scope

- Do not open V8 locked tests, ACBSM fresh blocks, or V9 development/confirmation seeds.
- Do not claim AGI, biological identity, predictive superiority, infinite physical execution,
  adaptive truncation, or matched-compute equivalence.
- Do not change the default `RuntimeAgent` action path when V9 is disabled.
- Do not delete or revert unrelated dirty-worktree content.

## Symbols and invariants

- $o_t\in\mathbb{R}^d$: current finite observation.
- $a_i\in\mathbb{R}^d$: fixed embedding for action $i$.
- $e_{t,i}$: dimensionless cosine action evidence, defined as zero if either norm is zero.
- $C$: sealed `AdaptiveTowerController` with shell width equal to the action count.
- $\tau_t=C.\operatorname{observe}(t,e_t)$: issued immutable state token.
- $\pi_t=C.\operatorname{read\_policy}(\tau_t,m_t)$: selected policy under a boolean mask.

Required invariants:

1. V9 action selection is exactly `pi_t.selected_action`; no external posterior or legacy
   action value may replace it.
2. `e_t` is finite and lies in $[-1,1]^{A}$ before the controller normalizer.
3. Invalid observation, mask, controller seal, or causal tick fails before `RuntimeAgent`
   working-memory/goal/critic state is committed.
4. With V9 disabled, the legacy action path is bitwise behavior-compatible under the same
   deterministic fixture.
5. All new numeric inputs are dimensionless or explicitly normalized.

## Tolerances

- Probability simplex checks: absolute tolerance $10^{-12}$.
- Cosine evidence comparison: absolute tolerance $10^{-7}$ for Torch-to-float conversion.
- Default legacy equality: exact selected action and exact absence of V9 outputs.

## Claim status on entry

- Nested-SCC mathematics: inherited conditional theorems from the predecessor run.
- Finite controller integrity: inherited isolated-unit result.
- Runtime causal integration: unimplemented and untested on entry.
- V9 efficacy, AGI, and brain-wide identity: untested and outside this run.
