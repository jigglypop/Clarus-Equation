from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from examples.physics.rendering.ce_rendering_registry import (
    AEM_INV_MZ,
    alpha_em_inv,
    calibrated_alpha_s,
    core,
    exterior_channels,
    pmns_s2,
    rendering_amplitude,
    rendering_amplitude_closed,
    score,
    transition_factor,
)


def _haar(n: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def test_rendering_amplitude_is_subspace_independent_and_closed_form() -> None:
    alpha_s, _ = calibrated_alpha_s()
    rng = np.random.default_rng(11)
    for m in (1, 2, 3):
        for _ in range(50):
            basis = _haar(3, rng)[:, :m]
            assert rendering_amplitude(alpha_s, basis) == pytest.approx(
                rendering_amplitude_closed(alpha_s, m), rel=1e-12)
    assert rendering_amplitude_closed(alpha_s, 2) ** 2 == pytest.approx(core(alpha_s)["s2"], rel=1e-14)


def test_generation_weight_gives_the_unique_transition_sign_pattern() -> None:
    c = core(calibrated_alpha_s()[0])
    x = 1 + c["d"] / (2 * math.pi)
    assert transition_factor(c, 1, 2) == pytest.approx(1 / x)
    assert transition_factor(c, 2, 3) == pytest.approx(x)
    assert transition_factor(c, 1, 3) == pytest.approx(1.0)
    solutions = {
        (n1 - n3, n2 - n3)
        for n1, n2, n3 in itertools.product(range(-2, 3), repeat=3)
        if (n1 - n2, n2 - n3, n1 - n3) == (-1, 1, 0)
    }
    assert solutions == {(0, 1)}


def test_exterior_channel_counts_fix_the_pmns_coefficients() -> None:
    assert [exterior_channels(m) for m in (1, 2, 3)] == [1, 3, 7]
    assert exterior_channels(3) + 1 == 2 ** 3
    c = core(calibrated_alpha_s()[0])
    assert pmns_s2(c, 1) == pytest.approx(c["d"] / 8)
    assert pmns_s2(c, 2) == pytest.approx((1 - 3 * c["d"] / 8) / 3)
    assert pmns_s2(c, 3) == pytest.approx((1 + 7 * c["d"] / 8) / 2)


def test_one_channel_loop_restores_the_sum_rule_alpha_em() -> None:
    c = core(calibrated_alpha_s()[0])
    assert abs(alpha_em_inv(c, channel_loop=False) - AEM_INV_MZ) > 1.0
    assert abs(alpha_em_inv(c, channel_loop=True) - AEM_INV_MZ) < 0.05


@pytest.mark.parametrize(
    ("variant", "pmns", "expected"),
    [("I", "SK", 1.587), ("II", "SK", 1.278), ("I", "noSK", 1.346), ("II", "noSK", 0.964)],
)
def test_joint_rmse_is_frozen(variant: str, pmns: str, expected: float) -> None:
    result = score(variant, pmns)
    assert result["N"] == 37
    assert result["bits"] == pytest.approx(17.0)
    assert result["rmse_all"] == pytest.approx(expected, abs=1.5e-3)
    assert result["k_continuous"] == (2 if variant == "I" else 1)
