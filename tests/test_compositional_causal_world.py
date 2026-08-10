from __future__ import annotations

from pathlib import Path

import numpy as np

from reality_stone.clarus.compositional_causal_world import (
    ForceCoefficients,
    LocalBasisModel,
    _episode,
    fit_local_coefficients,
    run_compositional_gate,
)
from reality_stone.clarus.nonlinear_object_world import ObjectWorldConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "compositional_causal_ood_v1.json"


def test_local_basis_recovers_noiseless_coefficients() -> None:
    truth = (0.12, 0.24, 0.07, -0.05)
    episode = _episode(77, truth, objects=4, noise=0.0)
    fitted = fit_local_coefficients([episode], 20)
    assert np.allclose(fitted.array()[:4], truth, atol=1e-10)
    assert np.isclose(fitted.action_gain, 1.0, atol=1e-10)


def test_local_basis_action_effect_has_correct_direction() -> None:
    coefficients = ForceCoefficients(0.1, 0.2, 0.06, 0.04, 1.0)
    model = LocalBasisModel(ObjectWorldConfig(), coefficients)
    episode = _episode(78, coefficients.array()[:4], objects=2, noise=0.0)
    state = episode.states[0]
    positive = model.step(state, np.array([0.2, 0.0]))
    negative = model.step(state, np.array([-0.2, 0.0]))
    assert np.all(positive[:, 2] > negative[:, 2])


def test_validation_gate_respects_zero_download_policy() -> None:
    report = run_compositional_gate(CONFIG, split="validation")
    assert report["resource_usage"]["external_download_bytes"] == 0
    assert report["resource_usage"]["trajectory_files_written"] == 0
    assert report["resource_passed"]
