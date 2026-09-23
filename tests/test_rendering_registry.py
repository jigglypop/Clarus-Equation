from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from examples.physics.rendering.ce_rendering_registry import (
    AEM_INV_MZ,
    alpha_em_inv,
    bao_chi2,
    bao_chi2_if_expansion_weakened,
    calibrated_alpha_s,
    circulant_eigenvector_drift,
    ckm_triangle,
    core,
    delta_pmns_tm1,
    distinction_channels,
    exterior_channels,
    pmns_matrix,
    pmns_s2,
    readout_amplitude,
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


def test_distinction_channels_fix_the_pmns_coefficients() -> None:
    assert [exterior_channels(m) for m in (1, 2, 3)] == [1, 3, 7]
    assert [distinction_channels(m) for m in (1, 2, 3)] == [1, 2, 4]
    c = core(calibrated_alpha_s()[0])
    s13 = pmns_s2(c, 1)
    assert s13 == pytest.approx(c["d"] / 8)
    assert pmns_s2(c, 2) == pytest.approx((1 - 2 * c["d"] / 8) / 3)
    assert pmns_s2(c, 2) == pytest.approx((1 - 3 * s13) / (3 * (1 - s13)), abs=5e-4)  # TM1 to first order
    assert pmns_s2(c, 3) == pytest.approx((1 - 4 * c["d"] / 8) / 2)
    assert pmns_s2(c, 3) < 0.5  # "나" = last rendered -> lower octant


def test_tm1_condition_fixes_delta_up_to_circulation() -> None:
    c = core(calibrated_alpha_s()[0])
    minus, plus = delta_pmns_tm1(c, -1), delta_pmns_tm1(c, +1)
    assert minus + plus == pytest.approx(360.0, abs=1e-9)
    U = pmns_matrix(pmns_s2(c, 2), pmns_s2(c, 3), pmns_s2(c, 1), math.radians(minus))
    assert abs(U[1][0]) == pytest.approx(abs(U[2][0]), abs=1e-12)
    assert 255.0 < minus < 262.0


def test_grade_partition_triangle_derives_vub_and_mirror_orientation() -> None:
    c = core(calibrated_alpha_s()[0])
    vub, delta, jarlskog = ckm_triangle(c["a"])
    assert 0.00360 < vub < 0.00375
    assert 1.15 < delta < 1.20
    assert jarlskog > 0
    assert ckm_triangle(c["a"], -1)[2] == pytest.approx(-jarlskog, rel=1e-6)
    assert delta_pmns_tm1(c) > 180.0  # M1: lepton circulation opposite to quarks


def test_hubble_readout_weakening_is_positive_and_not_in_the_expansion() -> None:
    alpha_s = calibrated_alpha_s()[0]
    c = core(alpha_s)
    amplitude = readout_amplitude(alpha_s)
    assert 0.05 < amplitude < 0.12  # weaker distinction -> larger local readout
    assert bao_chi2_if_expansion_weakened(c, amplitude) > bao_chi2(c["Om"]) + 10.0


def test_cosmic_cyclic_phase_does_not_move_the_mixing() -> None:
    for theta in (0.3, 0.7, 2.0, 4.0):
        assert circulant_eigenvector_drift(theta) < 1e-12


def test_one_channel_loop_restores_the_sum_rule_alpha_em() -> None:
    c = core(calibrated_alpha_s()[0])
    assert abs(alpha_em_inv(c, channel_loop=False) - AEM_INV_MZ) > 1.0
    assert abs(alpha_em_inv(c, channel_loop=True) - AEM_INV_MZ) < 0.05


@pytest.mark.parametrize(
    ("variant", "pmns", "expected"),
    [("I", "SK", 0.831), ("II", "SK", 0.773), ("I", "noSK", 1.526), ("II", "noSK", 1.495)],
)
def test_joint_rmse_is_frozen(variant: str, pmns: str, expected: float) -> None:
    result = score(variant, pmns)
    assert result["N"] == 39
    assert result["bits"] == pytest.approx(18.0)
    assert result["rmse_all"] == pytest.approx(expected, abs=1.5e-3)
    assert result["k_continuous"] == (3 if variant == "I" else 2)
