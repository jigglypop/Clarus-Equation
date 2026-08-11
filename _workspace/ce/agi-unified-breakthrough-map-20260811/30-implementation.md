# Implementation direction

Status: COMPLETE

No implementation was performed in this run.

The eventual development-only module should be isolated from V1–V8:

- `reality_stone/python/reality_stone/clarus/integrated_latent_state_bridge.py`
- `examples/agi/integrated_latent_state_bridge_development.py`
- `tests/test_integrated_latent_state_bridge.py`

Reuse frozen sparse/dense mechanisms, training-only scales, `PrefixReader`,
normalized metrics, parent hashes, and existing stability instrumentation.
Use the current single-mode residual filter only as a baseline/ablation.

Suggested types are `ResidualStatePrior`, `FilteredResidualState`,
`IntegratedDevelopmentContext`, and `IntegratedPredictions`. Preserve the
public prediction boundary
`predict_from_prefix(prefix_states, context, configuration)`.

The first active configuration contains only frozen sparse transition,
rank-two fast/slow residual state, prefix observer, internal correction, and
posterior-SNR trust. Regime, episodic memory, graph adaptation, beam rollout,
and planning remain disabled switches in the same architecture.
