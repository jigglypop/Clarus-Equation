from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from test_support.paths import PREREGISTRATION_ROOT, REPO_ROOT

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_cycle as CY
from examples.physics.rendering import ce_rendering_spiral as SP
from examples.physics.rendering import ce_rendering_complex_scale as CX
from examples.physics.rendering.ce_rendering_registry import (
    AEM_INV_MZ,
    alpha_em_inv,
    bao_chi2,
    bao_chi2_if_expansion_changed,
    cycle_average_occupation,
    hubble_kms,
    hubble_readout,
    calibrated_alpha_s,
    circulant_eigenvector_drift,
    colour_winding,
    ladder_clock_rescaling_ratio,
    ouroboros_generator,
    ouroboros_unitary,
    precessing_axis_mean,
    time_average_occupation,
    ckm_triangle,
    core,
    delta_pmns_tm1,
    distinction_channels,
    exterior_channels,
    pmns_matrix,
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


def test_ouroboros_cycle_renders_every_axis_evenly() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = rng.normal(size=3)
        assert cycle_average_occupation(n, 3) == pytest.approx(np.full(3, 1 / 3), abs=1e-14)
        assert cycle_average_occupation(n, 2) == pytest.approx(np.full(2, 1 / 2), abs=1e-14)


def test_direct_readouts_see_the_tilted_axis_and_rings_do_not() -> None:
    c = core(calibrated_alpha_s()[0])
    assert hubble_readout(c, False) == pytest.approx(hubble_kms(c))
    assert hubble_readout(c, True) == pytest.approx(hubble_kms(c) / math.cos(math.pi / 8))
    assert 72.0 < hubble_readout(c, True) < 73.5
    assert bao_chi2_if_expansion_changed(c, 0.08) > bao_chi2(c["Om"]) + 10.0


def test_uniform_clock_rescaling_cancels_inside_a_calibrated_ladder() -> None:
    for k in (0.9, math.cos(math.pi / 8), 1.1):
        assert ladder_clock_rescaling_ratio(k) == pytest.approx(1.0, abs=1e-12)


def test_colour_winding_gives_the_quark_time_channel_only() -> None:
    assert colour_winding("quark", 2, 3) == 1
    assert colour_winding("lepton", 2, 3) == 0
    assert colour_winding("quark", 1, 2) == 0


def test_ouroboros_generator_conserves_probability_and_energy() -> None:
    H = ouroboros_generator()
    assert np.allclose(H, H.conj().T)
    U = ouroboros_unitary(2 * math.pi / 3)
    assert np.abs(U.conj().T @ U - np.eye(3)).max() < 1e-12
    S = np.roll(np.eye(3), 1, axis=0)
    assert min(np.abs(U - S).max(), np.abs(U - S.T).max()) < 1e-12
    rng = np.random.default_rng(9)
    psi = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi /= np.linalg.norm(psi)
    energies = [np.vdot(ouroboros_unitary(x) @ psi, H @ (ouroboros_unitary(x) @ psi)).real for x in (0.0, 0.7, 2.5)]
    assert max(energies) - min(energies) < 1e-12
    assert time_average_occupation(psi) == pytest.approx(np.full(3, 1 / 3), abs=1e-12)
    assert precessing_axis_mean() == pytest.approx(np.full(3, math.cos(math.pi / 8) / math.sqrt(3)), abs=1e-12)


def test_cosmic_cyclic_phase_does_not_move_the_mixing() -> None:
    for theta in (0.3, 0.7, 2.0, 4.0):
        assert circulant_eigenvector_drift(theta) < 1e-12


def test_one_channel_loop_restores_the_sum_rule_alpha_em() -> None:
    c = core(calibrated_alpha_s()[0])
    assert abs(alpha_em_inv(c, channel_loop=False) - AEM_INV_MZ) > 1.0
    assert abs(alpha_em_inv(c, channel_loop=True) - AEM_INV_MZ) < 0.05


@pytest.mark.parametrize(
    ("variant", "pmns", "expected"),
    [("I", "SK", 0.830), ("II", "SK", 0.746), ("I", "noSK", 1.525), ("II", "noSK", 1.481)],
)
def test_joint_rmse_is_frozen(variant: str, pmns: str, expected: float) -> None:
    result = score(variant, pmns)
    assert result["N"] == 39
    assert result["bits"] == pytest.approx(18.0)
    assert result["rmse_all"] == pytest.approx(expected, abs=1.5e-3)
    assert result["k_continuous"] == (2 if variant == "I" else 1)


def test_rendering_predictions_v1_is_frozen_and_reproduced() -> None:
    path = PREREGISTRATION_ROOT / "rendering_predictions_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == manifest["manifest_sha256"]
    assert manifest["manifest_sha256"] == "e02bf7f66b2b0839d7a8424863eb852e88fd75676ebf77ccb4ac3a5481d68737"
    registry = REPO_ROOT / manifest["model"]["registry_path"]
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == manifest["model"]["registry_sha256"], (
        "registry changed after freeze: create rendering_predictions_v2.json and keep v1")
    a = calibrated_alpha_s()[0]
    c = core(a)
    values = {p["id"]: p["value"] for p in manifest["predictions"]}
    assert values["P01"] == pytest.approx(pmns_s2(c, 3), rel=1e-5)
    assert values["P02"] == pytest.approx(delta_pmns_tm1(c), rel=1e-5)
    assert values["P03"] == pytest.approx(pmns_s2(c, 2), rel=1e-5)
    assert values["P06"] == pytest.approx(ckm_triangle(a)[0], rel=1e-5)
    assert values["P07"] == pytest.approx(hubble_readout(c, True), rel=1e-5)
    assert values["P08"] == pytest.approx(hubble_readout(c, False), rel=1e-5)


def test_horizon_h0_fails_the_acoustic_angle_and_theta_calibration_restores_it() -> None:
    a = calibrated_alpha_s()[0]
    c = core(a)
    assert DV.ce_theta_pull(c) < -8.0
    h = DV.h_from_theta(a)
    assert 0.672 < h < 0.682
    assert abs(DV.ce_theta_pull(c, h)) < 1e-6
    res = DV.score_variant_iii("SK")
    assert res["N"] == 38 and res["k_continuous"] == 3
    assert res["rmse_all"] == pytest.approx(0.841, abs=2e-3)
    assert max(abs(o["pull"]) for o in res["rows"]) < 2.0


def test_tilt_is_not_a_lorentz_boost_of_the_observer() -> None:
    beta = DV.boost_equivalent_beta()
    assert 0.35 < beta < 0.41
    assert beta > 100 * DV.CMB_DIPOLE_BETA


def test_rendering_predictions_v2_keeps_v1_and_is_frozen() -> None:
    v1 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v2.json").read_text(encoding="utf-8"))
    assert v2["supersedes_manifest_id"] == v1["manifest_id"]
    body = {k: v for k, v in v2.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v2["manifest_sha256"]
    for key in ("registry", "derivations"):
        path = REPO_ROOT / v2["model"][f"{key}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v2["model"][f"{key}_sha256"], (
            f"{key} changed after v2 freeze: create v3 and keep v1, v2")
    values = {p["id"]: p["value"] for p in v2["predictions"]}
    a = calibrated_alpha_s()[0]
    assert values["P11"] == pytest.approx(1 / math.cos(math.pi / 8), rel=1e-5)
    assert values["P07"] == pytest.approx(100 * DV.h_from_theta(a) / math.cos(math.pi / 8), rel=1e-4)
    assert "P08" not in values


def test_planck_unit_loop_transfers_to_the_horizon_readout() -> None:
    a = calibrated_alpha_s()[0]
    c = core(a)
    h = PL.h_rings(c)
    assert abs(DV.ce_theta_pull(c, h)) < 2.0
    for rejected in (1 + a / (16 * math.pi), 1 + c["d"] / (2 * math.pi)):
        assert abs(DV.ce_theta_pull(c, hubble_readout(c, False) / 100 * rejected)) > 5.0
    res = PL.score_variant_iv("SK")
    assert res["N"] == 39 and res["k_continuous"] == 2
    assert res["rmse_all"] == pytest.approx(0.834, abs=2e-3)
    assert max(abs(o["pull"]) for o in res["rows"]) < 2.0


def test_tilt_is_a_euclidean_rotation_with_imaginary_rapidity() -> None:
    phi = math.pi / 8
    assert PL.killing_alignment(phi) == pytest.approx(math.cos(phi), abs=1e-14)
    z = PL.imaginary_rapidity_factor(phi)
    assert z.imag == pytest.approx(0.0, abs=1e-15)
    assert z.real == pytest.approx(math.cos(phi), abs=1e-15)


def test_rendering_predictions_v3_keeps_v1_v2_and_is_frozen() -> None:
    v2 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v2.json").read_text(encoding="utf-8"))
    v3 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v3.json").read_text(encoding="utf-8"))
    assert v3["supersedes_manifest_id"] == v2["manifest_id"]
    body = {k: v for k, v in v3.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v3["manifest_sha256"]
    for key in ("registry", "derivations", "planck_readout"):
        path = REPO_ROOT / v3["model"][f"{key}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v3["model"][f"{key}_sha256"], (
            f"{key} changed after v3 freeze: create v4 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v3["predictions"]}
    c = core(calibrated_alpha_s()[0])
    assert values["P08"] == pytest.approx(100 * PL.h_rings(c), rel=1e-4)
    assert values["P07"] == pytest.approx(100 * PL.h_rings(c) / math.cos(math.pi / 8), rel=1e-4)
    assert values["P11"] == pytest.approx(1 / math.cos(math.pi / 8), rel=1e-5)


def test_fixed_bao_ruler_exposes_the_cmb_bao_scale_tension() -> None:
    c = core(calibrated_alpha_s()[0])
    rd, h = BR.ce_rd_and_h(c)
    assert 146.0 < rd < 149.0
    h0, s = BR.h0_from_bao(c)
    assert 2.5 < (h0 - 100 * h) / s < 3.6
    res = BR.score_variant_v("SK")
    assert res["k_continuous"] == 1 and res["N"] == 39
    assert res["bao_chi2"] == pytest.approx(21.26, abs=0.05)
    assert res["rmse_all"] == pytest.approx(0.970, abs=2e-3)


def test_rendering_predictions_v4_keeps_earlier_manifests_and_is_frozen() -> None:
    v3 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v3.json").read_text(encoding="utf-8"))
    v4 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v4.json").read_text(encoding="utf-8"))
    assert v4["supersedes_manifest_id"] == v3["manifest_id"]
    body = {k: v for k, v in v4.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v4["manifest_sha256"]
    for key in ("registry", "derivations", "planck_readout", "bao_ruler"):
        path = REPO_ROOT / v4["model"][f"{key}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v4["model"][f"{key}_sha256"], (
            f"{key} changed after v4 freeze: create v5 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v4["predictions"]}
    rd, h = BR.ce_rd_and_h(core(calibrated_alpha_s()[0]))
    assert values["P14"] == pytest.approx(h * rd, rel=1e-4)


def test_cosmic_cycle_tilt_agrees_with_the_vacuum_channel_angle() -> None:
    c = core(calibrated_alpha_s()[0])
    g = CY.cycle_geometry(c)
    assert 5200 < g["radius_mpc"] < 5450
    assert 13.5 < g["age_gyr"] < 14.1
    tilt = CY.tangent_chord_tilt(c["Om"])
    assert abs(tilt / (math.pi / 8) - 1) < 0.02
    om = CY.omega_m_for_phase()
    cb = dict(c)
    cb["Om"] = om
    assert DV.ce_theta_pull(cb, PL.h_rings(c)) > 10.0  # exact pi/4 phase is rejected by theta*


def test_spiral_tension_gives_the_hubble_ratio_without_pi_over_8() -> None:
    c = core(calibrated_alpha_s()[0])
    routes = SP.three_routes(c)
    assert max(routes.values()) - min(routes.values()) < 0.01
    ol = 1 - c["Om"]
    assert SP.direct_over_rings(ol) == pytest.approx(math.sqrt(1 + ol / 4))
    assert SP.direct_readout(c) ** 2 == pytest.approx(
        (100 * PL.h_rings(c)) ** 2 + (100 * PL.h_rings(c) * math.sqrt(ol) / 2) ** 2)
    assert abs(SP.direct_readout(c) - 73.17) / 0.86 < 1.0


def test_rendering_predictions_v5_is_frozen() -> None:
    v4 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v4.json").read_text(encoding="utf-8"))
    v5 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v5.json").read_text(encoding="utf-8"))
    assert v5["supersedes_manifest_id"] == v4["manifest_id"]
    body = {k: v for k, v in v5.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v5["manifest_sha256"]
    for key in ("registry", "derivations", "planck_readout", "bao_ruler", "cycle", "spiral"):
        path = REPO_ROOT / v5["model"][f"{key}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v5["model"][f"{key}_sha256"], (
            f"{key} changed after v5 freeze: create v6 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v5["predictions"]}
    c = core(calibrated_alpha_s()[0])
    assert values["P15"] == pytest.approx(SP.direct_over_rings(1 - c["Om"]), rel=1e-5)


def test_complex_scale_factor_restores_flrw_and_reads_the_spiral_rate() -> None:
    c = core(calibrated_alpha_s()[0])
    sol, hub, t0, omega, h0 = CX.background(c)
    assert 13.5 < t0 < 14.1
    rate = CX.complex_rate(c, t0)
    assert rate.real / CX.KMS_MPC_PER_GYR == pytest.approx(100 * PL.h_rings(c), rel=1e-6)
    assert abs(rate) / CX.KMS_MPC_PER_GYR == pytest.approx(SP.direct_readout(c), rel=1e-6)
    assert CX.friedmann_residual(c) < 1e-6
    assert CX.ring_ratio_invariance(c, 0.3) < 1e-12
    assert CX.record_phase_gap(c) < 1e-4
