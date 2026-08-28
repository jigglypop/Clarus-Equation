from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple,
    simplicity_residual,
)
from examples.physics.urbantke_shape_matching_rg import (
    centered_shape_fluctuation_scaling,
    common_metric_block,
    conformal_metric_residual,
    normalized_urbantke_metric,
    optimal_internal_alignment,
    repeated_coherent_mismatch_residual,
    shape_matching_rg_verdict,
    shape_matching_weight,
    urbantke_metric_density,
)


def _rotation(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if float(np.linalg.det(rotation)) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _mismatch() -> np.ndarray:
    return geometric_self_dual_triple(
        np.array(
            [
                [0.74341184, 0.80016662, 0.30490388, -0.47971556],
                [0.02980649, 1.23067583, -0.07551285, 0.27316411],
                [-0.02660693, 0.26689902, 1.57540904, -0.27026490],
                [0.08125544, -0.18532303, 0.05090736, 0.52512219],
            ]
        )
    )


def test_urbantke_metric_recovers_identity() -> None:
    triple = geometric_self_dual_triple(np.eye(4))
    assert urbantke_metric_density(triple) == pytest.approx(np.eye(4))
    assert normalized_urbantke_metric(triple) == pytest.approx(np.eye(4))


def test_scale_and_internal_rotation_preserve_metric() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    candidate = 2.3 * _rotation(3) @ reference
    audit = optimal_internal_alignment(reference, candidate)
    assert audit.metric_residual < 1.0e-12
    assert audit.orbit_residual < 1.0e-12
    assert audit.relative_scale == pytest.approx(2.3)
    assert audit.block_residual < 1.0e-12


def test_common_orbit_blocking_is_associative() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    triples = tuple(
        scale * _rotation(seed) @ reference
        for seed, scale in zip((1, 2, 3, 4), (0.7, 1.2, 2.0, 0.4))
    )
    direct = common_metric_block(triples)
    left = common_metric_block(triples[:2])
    right = common_metric_block(triples[2:])
    regrouped = common_metric_block((left.blocked_triple, right.blocked_triple))
    assert direct.blocked_simplicity_residual < 1.0e-12
    assert regrouped.blocked_triple == pytest.approx(direct.blocked_triple, abs=1.0e-11)


def test_nonconformal_cells_fail_common_orbit() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    candidate = _mismatch()
    audit = optimal_internal_alignment(reference, candidate)
    assert simplicity_residual(candidate) < 1.0e-12
    assert audit.metric_residual > 0.5
    assert audit.orbit_residual > 0.5
    assert audit.block_residual > 0.09
    with pytest.raises(ValueError, match="common conformal-metric orbit"):
        common_metric_block((reference, candidate))


def test_soft_weight_penalizes_mismatch() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    aligned = 1.4 * _rotation(8) @ reference
    assert shape_matching_weight(
        reference,
        aligned,
        metric_width=0.05,
        orbit_width=0.05,
        block_width=0.05,
    ) == pytest.approx(1.0)
    assert shape_matching_weight(
        reference,
        _mismatch(),
        metric_width=0.05,
        orbit_width=0.05,
        block_width=0.05,
    ) < 0.1


def test_coherent_mismatch_survives_repetition() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    once = repeated_coherent_mismatch_residual(reference, _mismatch(), repeats=1)
    many = repeated_coherent_mismatch_residual(reference, _mismatch(), repeats=100)
    assert once > 0.0
    assert many == pytest.approx(once, rel=1.0e-12)


def test_centered_mismatch_decreases() -> None:
    scaling = centered_shape_fluctuation_scaling()
    assert scaling.mean_residuals[-1] < scaling.mean_residuals[0]
    assert -0.8 < scaling.fitted_power < -0.2


def test_metric_residual_is_scale_rotation_invariant() -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    candidate = 3.2 * _rotation(11) @ reference
    assert conformal_metric_residual(reference, candidate) < 1.0e-12


def test_verdict_separates_exact_and_statistical_closure() -> None:
    verdict = shape_matching_rg_verdict()
    assert verdict.exact_common_orbit_closed
    assert verdict.exact_common_orbit_associative
    assert not verdict.coherent_mismatch_suppressed_by_blocking
    assert verdict.centered_mismatch_decreases
    assert verdict.centered_mismatch_fitted_power < 0.0
    assert "Lorentzian" in verdict.remaining_obligation


@pytest.mark.parametrize("width", (0.0, -1.0, math.inf))
def test_invalid_widths_are_rejected(width: float) -> None:
    reference = geometric_self_dual_triple(np.eye(4))
    with pytest.raises(ValueError, match="finite and positive"):
        shape_matching_weight(
            reference,
            reference,
            metric_width=width,
            orbit_width=0.1,
            block_width=0.1,
        )
