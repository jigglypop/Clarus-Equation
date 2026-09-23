from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from test_support.paths import PREREGISTRATION_ROOT, REPO_ROOT

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_cycle as CY
from examples.physics.rendering import ce_rendering_spiral as SP
from examples.physics.rendering import ce_rendering_complex_scale as CX
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_event_scale as ES
from examples.physics.rendering import ce_rendering_neutrino as NU
from examples.physics.rendering import ce_rendering_gauge as GA
from examples.physics.rendering import ce_rendering_generations as GE
from examples.physics.rendering import ce_rendering_ewsb as EW
from examples.physics.rendering import ce_rendering_inflation as IN
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_w_branch as WB
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT
from examples.physics.rendering import ce_rendering_vacuum_harmonic as VH
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_light_limit as LL
from examples.physics.rendering import ce_rendering_probability_weight as PW
from examples.physics.rendering import ce_rendering_record_update as RU
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_closure as CL
from examples.physics.rendering import ce_rendering_reverse_derivations as RD
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


def test_zero_fit_s8_sides_with_cmb_and_kids_legacy() -> None:
    c = core(calibrated_alpha_s()[0])
    s8, S8 = GR.ce_s8(c)
    assert 0.80 < s8 < 0.82 and 0.81 < S8 < 0.83
    res = GR.score_variant_iv_with_lensing("SK")
    pulls = {o["key"]: o["pull"] for o in res["rows"]}
    assert abs(pulls["S8 KiDS-Legacy"]) < 1.0
    assert pulls["S8 DES Y3 3x2pt"] > 2.0
    assert res["N"] == 41 and res["rmse_all"] == pytest.approx(0.910, abs=3e-3)


def test_frame_rotation_and_spectral_phase_are_different_circles() -> None:
    c = core(calibrated_alpha_s()[0])
    assert GR.theta_equals_three_phi_vacuum_ratio(c) < 0.75


def test_rendering_predictions_v6_is_frozen() -> None:
    v5 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v5.json").read_text(encoding="utf-8"))
    v6 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v6.json").read_text(encoding="utf-8"))
    assert v6["supersedes_manifest_id"] == v5["manifest_id"]
    body = {k: v for k, v in v6.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v6["manifest_sha256"]
    for key in ("registry", "derivations", "planck_readout", "bao_ruler", "cycle", "spiral", "complex_scale", "growth"):
        path = REPO_ROOT / v6["model"][f"{key}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v6["model"][f"{key}_sha256"], (
            f"{key} changed after v6 freeze: create v7 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v6["predictions"]}
    assert values["P16"] == pytest.approx(GR.ce_s8(core(calibrated_alpha_s()[0]))[1], rel=1e-4)


def test_rendering_event_is_at_the_z_pole_only() -> None:
    table = ES.event_scale_table()
    pulls = {k: (v - ES.A_WORLD) / ES.A_WORLD_ERR for k, v in table.items()}
    assert abs(pulls["M_Z"]) < 1.0
    assert all(abs(p) > 2.5 for k, p in pulls.items() if k != "M_Z")


def test_neutrino_masses_near_the_normal_ordering_minimum() -> None:
    c = core(calibrated_alpha_s()[0])
    m1, m2, m3 = NU.neutrino_masses_mev(c)
    assert m1 < 1.0 and 58.0 < m1 + m2 + m3 < 61.0
    dm21, dm31 = NU.splittings_ev2(c)
    assert abs(dm21 - 7.49e-5) / 0.19e-5 < 1.0
    assert abs(dm31 - 2.513e-3) / 0.021e-3 < 1.0
    res = NU.score_full("SK")
    assert res["N"] == 43 and res["rmse_all"] == pytest.approx(0.895, abs=3e-3)


def test_rendering_predictions_v7_is_frozen() -> None:
    v6 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v6.json").read_text(encoding="utf-8"))
    v7 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v7.json").read_text(encoding="utf-8"))
    assert v7["supersedes_manifest_id"] == v6["manifest_id"]
    body = {k: v for k, v in v7.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v7["manifest_sha256"]
    for key, rel in v7["model"]["files"].items():
        assert hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() == v7["model"]["sha256"][key], (
            f"{key} changed after v7 freeze: create v8 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v7["predictions"]}
    c = core(calibrated_alpha_s()[0])
    assert values["P17"] == pytest.approx(sum(NU.neutrino_masses_mev(c)), rel=1e-4)


def test_rendered_stages_give_one_anomaly_free_sm_generation() -> None:
    assert GA.matches_one_generation()
    assert all(v == 0 for v in GA.anomaly_sums().values())
    assert GA.unification_sin2() == Fraction(3, 8)


def test_stage_preserving_symmetry_is_the_sm_gauge_group() -> None:
    g = GA.gauge_structure_checks()
    assert g["[Y,su3]"] < 1e-12 and g["[Y,su2]"] < 1e-12 and g["[su3,su2]"] < 1e-12
    assert g["[Y,stage-mixing]"] > 0.1
    assert abs(g["trace y"]) < 1e-15 and g["even sector invariant"] < 1e-12


def test_rendering_channel_is_cptp_and_non_signalling() -> None:
    r = GA.rendering_channel_checks()
    assert r["isometry error"] < 1e-12
    assert r["choi min eigenvalue"] > -1e-12
    assert r["no-signalling error"] < 1e-12


def test_cycle_generations_explain_small_quark_and_large_lepton_mixing() -> None:
    assert GE.generation_count() == 3
    for seed in range(6):
        ckm = GE.ckm_from_circulants(seed)
        assert np.allclose(np.sort(ckm, axis=1)[:, -1], 1.0, atol=1e-10)   # permutation: no leading-order mixing
        assert np.allclose(np.sort(ckm, axis=1)[:, :-1], 0.0, atol=1e-10)
    assert np.abs(GE.lepton_mixing_leading() - GE.TBM_SQ).max() < 1e-12


def test_tensor_modes_travel_at_light_speed_in_the_abs_a_background() -> None:
    t = GE.tensor_mode_checks()
    assert abs(t["c_T/c"] - 1.0) < 1e-4
    assert t["amplitude*a spread (Xi-1 proxy)"] < 0.02


def test_weak_channel_breaking_leaves_electric_charge_and_quantizes_it() -> None:
    assert all((3 * q).denominator == 1 for q in EW.charges_all_states())
    s = EW.higgs_channel()
    assert EW.weak_t3(s) + GA.hypercharge(s) == 0
    m = EW.gauge_boson_masses()
    assert sorted(m["masses_GeV"])[0] < 1e-6
    assert m["photon_is_Q"] < 1e-12
    assert m["M_W/M_Z"] == pytest.approx(m["c_W"], abs=1e-12)
    assert 79.0 < m["M_W"] < 81.5 and 90.0 < m["M_Z"] < 92.5


def test_inflation_gauge_count_is_the_stage_preserving_subalgebra() -> None:
    assert len(IN.su_basis(5)) == 24
    assert IN.stage_preserving_dimension() == 12
    c = core(calibrated_alpha_s()[0])
    assert IN.inflation_efolds(c) == pytest.approx(c["Ne"], rel=1e-14)


def test_neutrino_ledger_removes_ce_neutrinos_from_early_cold_matter() -> None:
    c = core(calibrated_alpha_s()[0])
    wb, wc, h = NL.early_densities(c)
    assert NL.omega_nu_h2(c) == pytest.approx(sum(NU.neutrino_masses_mev(c)) / 1000 / 93.14, rel=1e-12)
    assert wc == pytest.approx((c["Om"] - c["q"]) * h * h - NL.omega_nu_h2(c), rel=1e-12)
    h0, s = NL.h0_from_bao(c)
    assert (h0 - 100 * h) / s == pytest.approx(2.65, abs=0.02)
    assert NL.score("IV", "fixed")["rmse_all"] == pytest.approx(0.950, abs=2e-3)
    assert NL.score("IV", "free")["rmse_all"] == pytest.approx(0.849, abs=2e-3)   # worse row kept, not hidden


def test_rendering_predictions_v8_is_frozen() -> None:
    v7 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v7.json").read_text(encoding="utf-8"))
    v8 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v8.json").read_text(encoding="utf-8"))
    assert v8["supersedes_manifest_id"] == v7["manifest_id"]
    body = {k: v for k, v in v8.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v8["manifest_sha256"]
    for key, rel in v8["model"]["files"].items():
        assert hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() == v8["model"]["sha256"][key], (
            f"{key} changed after v8 freeze: create v9 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v8["predictions"]}
    c = core(calibrated_alpha_s()[0])
    rd, h = NL.rd_and_h(c)
    assert values["P14"] == pytest.approx(h * rd, rel=1e-4)
    assert values["P13"] == pytest.approx(NL.early_densities(c)[1], rel=1e-4)


def test_repo_dark_energy_branch_w1_is_rejected_under_fixed_h() -> None:
    c = core(calibrated_alpha_s()[0])
    w = WB.w1(c)
    assert w[0] == pytest.approx(-0.768, abs=1e-3) and w[1] == pytest.approx(-0.214, abs=1e-3)
    assert w[0] + w[1] > -1.0                                   # never crosses -1
    lam, br = WB.branch_rows(c, WB.LAMBDA), WB.branch_rows(c, w)
    assert br["bao_chi2_free"] < lam["bao_chi2_free"]           # the shape alone improves ...
    assert br["theta_pull"] > 30 and br["cmb_bao_tension"] < -5  # ... but theta* and the ruler break
    assert WB.joint_rmse(c, w)[0] > 5 * WB.joint_rmse(c, WB.LAMBDA)[0]
    assert WB.theta_fitted_h(c, w)["omega_b_pull"] < -10         # detour via theta-fitted h also fails


def test_vacuum_tilt_w2_half_frequency_passes_the_preregistered_rules() -> None:
    lam, w2 = VT.score(None), VT.score(VT.ADOPTED_NU)
    assert w2["rmse_all"] < lam["rmse_all"] and w2["rmse_all"] == pytest.approx(0.909, abs=2e-3)
    assert max(abs(o["pull"]) for o in w2["rows"]) < 3.0
    assert w2["branch"]["cmb_bao_tension"] < lam["branch"]["cmb_bao_tension"]
    for nu in (1.0, 3.0):                                   # the other two pre-declared frequencies fail on theta*
        th = next(o for o in VT.score(nu)["rows"] if o["key"] == "100 theta*")
        assert th["pull"] > 3.0
    c = core(calibrated_alpha_s()[0])
    assert -1.0 < VT.w_of(c, VT.ADOPTED_NU, 0.0) < -0.98    # never crosses -1


def test_rendering_predictions_v9_is_frozen() -> None:
    v8 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v8.json").read_text(encoding="utf-8"))
    v9 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v9.json").read_text(encoding="utf-8"))
    assert v9["supersedes_manifest_id"] == v8["manifest_id"]
    body = {k: v for k, v in v9.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v9["manifest_sha256"]
    for key, rel in v9["model"]["files"].items():
        assert hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() == v9["model"]["sha256"][key], (
            f"{key} changed after v9 freeze: create v10 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v9["predictions"]}
    c = core(calibrated_alpha_s()[0])
    assert values["P21"] == pytest.approx(VT.w_of(c, VT.ADOPTED_NU, 0.0), rel=1e-3)


def test_w2_is_the_lowest_real_harmonic_of_the_complex_scale_phase() -> None:
    c = core(calibrated_alpha_s()[0])
    assert VH.phase_rate_over_h_lambda(c) == pytest.approx(0.5, abs=1e-12)      # harmonic k -> nu = k/2
    assert c["a"] ** (2 / 3) == pytest.approx(math.sqrt(4 * c["a"] ** (4 / 3)) / 2, rel=1e-14)  # xi^2 = sin(theta_W)/2
    f1, f_vt = VH.harmonic_density(c, 1), VT.density(c, 0.5)
    assert all(f1(a) == pytest.approx(f_vt(a), rel=1e-14) for a in (0.3, 0.6, 0.9))
    plus, minus = VH.score(1, 1.0), VH.score(1, -1.0)
    assert plus["rmse_all"] < minus["rmse_all"]                                  # the sign is the one data-chosen bit


def test_mimetic_clock_action_conserves_energy_and_fixes_the_sign() -> None:
    m = MV.model("b")
    assert MV.continuity_residual(m) < 1e-4                        # Bianchi: vacuum decay feeds dust
    assert MV.model("b", sign=-1.0)["min_dust"] < 0                # c_1 < 0 needs negative-energy dust
    assert m["min_dust"] > -1e-20
    b, a = MV.score("b"), MV.score("a")
    assert b["rmse_all"] == pytest.approx(0.920, abs=3e-3) and b["cmb_bao_tension"] < 2.4
    assert a["rmse_all"] > 1.2                                      # primary reading fails
    assert -1.0 < MV.w_eff(m, 0.0) < -0.99
    assert MV.score("b", "free", rows_from="full")["rmse_all"] > 0.896   # mixed on 43 rows: kept as competing branch


TIME_AXIS_TILT_VALUE = math.pi / 8


def test_light_speed_as_rendering_limit_fixes_the_tilt_and_bounds_all_signals() -> None:
    assert LL.bisector_tilt() == pytest.approx(TIME_AXIS_TILT_VALUE, abs=1e-15)   # O1 = half the light-cone angle
    c = core(calibrated_alpha_s()[0])
    sp = LL.spiral_equality_test(c)
    assert abs(sp["tan_psi_spiral"] / sp["tan_pi8"] - 1) < 5e-3
    assert sp["theta_pull_if_equal"] > 5                                          # not an exact equality
    s = LL.signal_speeds(c)
    assert s["W2 min(1+w)"] >= 0 and s["W3 min dust"] > -1e-20
    assert s["photon-baryon sound at z*"] < s["radiation limit 1/sqrt3"] < 1.0
    assert abs(s["graviton"] - 1.0) < 1e-4
    hz = LL.rendering_horizon(c)
    assert 122.0 < hz["log10_entropy"] < 123.0


def test_probability_weight_gravity_is_the_only_no_signalling_reading() -> None:
    assert PW.signalling("branch") > 0.4          # branch-sourced gravity signals faster than light
    assert PW.signalling("weight") < 1e-12         # probability-weight sourcing does not
    b = PW.bmv()
    assert b["concurrence_quantum_gravity"] > 0.02 and b["concurrence_probability_weight"] < 1e-12
    c = core(calibrated_alpha_s()[0])
    w = PW.cosmic_weights(c)
    assert w["Omega_total"] == pytest.approx(1.0, abs=1e-15)


def test_rendering_predictions_v10_is_frozen() -> None:
    v9 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v9.json").read_text(encoding="utf-8"))
    v10 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v10.json").read_text(encoding="utf-8"))
    assert v10["supersedes_manifest_id"] == v9["manifest_id"]
    body = {k: v for k, v in v10.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v10["manifest_sha256"]
    for key, rel in v10["model"]["files"].items():
        assert hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() == v10["model"]["sha256"][key], (
            f"{key} changed after v10 freeze: create v11 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v10["predictions"]}
    assert values["P23"] == 0.0 and values["P24"] == 0.0


def test_records_update_the_gravity_source_only_inside_the_light_cone() -> None:
    assert RU.page_geilker_correlation("record") == pytest.approx(1.0)
    assert RU.page_geilker_correlation("average") == 0.0
    assert RU.outside_light_cone_signalling() < 1e-12 < 0.4 < RU.instant_remote_update_signalling()
    assert RU.dp_event_scale_length()["orders_below_bound"] > 5     # spontaneous DP collapse at M_Z rejected


def test_cmb_bao_gradient_c3_shapes_fail_and_matter_share_amplitude_closes_the_tension() -> None:
    for prof in ("G1", "G2", "G3"):                                   # pre-registered C3 shapes overshoot
        assert GD.joint_v(prof) > GD.joint_v("none")
    fit = GD.amplitude_fit("G1")
    assert fit["k_lo"] < fit["Om"] < fit["k_hi"] and fit["chi2_k1"] > fit["chi2_k0"] > fit["chi2_min"] + 4
    g = GD.bao_rows("G1m")
    assert abs(g["tension_sigma"]) < 0.5
    assert GD.joint_v("G1m") == pytest.approx(0.834, abs=2e-3)


def test_rendering_predictions_v11_is_frozen() -> None:
    v10 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v10.json").read_text(encoding="utf-8"))
    v11 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v11.json").read_text(encoding="utf-8"))
    assert v11["supersedes_manifest_id"] == v10["manifest_id"]
    body = {k: v for k, v in v11.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v11["manifest_sha256"]
    for key, rel in v11["model"]["files"].items():
        assert hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() == v11["model"]["sha256"][key], (
            f"{key} changed after v11 freeze: create v12 and keep earlier manifests")
    values = {q["id"]: q["value"] for q in v11["predictions"]}
    assert values["P25"] == pytest.approx(1.0126, abs=2e-4)


def test_record_completion_is_thermodynamic_and_horizon_is_not_an_integer_channel_count() -> None:
    v = CL.record_rule_verdicts()
    assert v["RL"] == [] and len(v["R2"]) == 1          # redundancy-2 contradicted by GHZ coherence
    for r in CL.horizon_integer_channels().values():
        assert r["mismatch"] > 0.25


def test_inspirations_reduce_to_probability_weight_null_energy_and_bisector() -> None:
    assert RD.vacuum_frame_invariance() < 1e-12
    g = RD.g1m_derivation_check()
    assert g["max_diff_to_G1m"] < 1e-12
    assert g["present weights"]["V39"] < g["epoch weights"]["V39"]      # pre-registered present-weight reading wins
    w = RD.w2_sign_by_null_energy()
    assert w["minus_is_phantom"] and w["w0_plus"] > -1
    tri = RD.unitarity_triangle_from_bisector()
    assert tri["angles_deg"] == pytest.approx((22.5, 67.5, 90.0)) and tri["sum_deg"] == pytest.approx(180.0)
    r = RD.rpl_from_channel_loop()
    assert r["planck_unit_factor - 1"] == pytest.approx(r["4 * a/(16 pi)"], rel=1e-12)
