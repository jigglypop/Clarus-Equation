from __future__ import annotations

from examples.pre_eq.toy_gate_ablation import (
    ToyGateConfig,
    delayed_disambiguation_trial,
    run_ablation,
)


def test_delayed_disambiguation_needs_residual_injection() -> None:
    control = delayed_disambiguation_trial(ToyGateConfig(alpha_phi=0.0))
    treatment = delayed_disambiguation_trial(ToyGateConfig(alpha_phi=3.0))

    assert control["selected"] == [0, 0]
    assert treatment["selected"] == [0, 1]
    assert treatment["accuracy"] > control["accuracy"]


def test_shuffled_residual_control_blocks_structured_gain() -> None:
    report = run_ablation()

    assert report["treatment_alpha_phi_3"]["accuracy"] == 1.0
    assert report["shuffled_residual_control"]["accuracy"] < 1.0

