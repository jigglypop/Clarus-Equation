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
from examples.physics.rendering import ce_rendering_axiom_proofs as AP
from examples.physics.rendering import ce_rendering_complex_boost as CB
from examples.physics.rendering import ce_rendering_e4_trace as E4
from examples.physics.rendering import ce_rendering_intermittent as IR
from examples.physics.rendering import ce_rendering_boundary as BD
from examples.physics.rendering import ce_rendering_staircase as ST
from examples.physics.rendering import ce_rendering_bool as BO
from examples.physics.rendering import ce_rendering_open_checks as OC
from examples.physics.rendering import ce_rendering_nu_audit as NA
from examples.physics.rendering import ce_rendering_ledger as LG
from examples.physics.rendering import ce_rendering_o1_map as OM
from examples.physics.rendering import ce_rendering_spread as SP2
from examples.physics.rendering import ce_rendering_phase_lock as PLK
from examples.physics.rendering import ce_rendering_causal_lock as CLK
from examples.physics.rendering import ce_rendering_lock_history as LH
from examples.physics.rendering import ce_rendering_thermal_time as TT
from examples.physics.rendering import ce_rendering_branching as BRN
from examples.physics.rendering import ce_rendering_e4_anchor as E4A
from examples.physics.rendering import ce_rendering_biased_coin as BCN
from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_higgs_cosmos as HGC
from examples.physics.rendering import ce_rendering_distinction as DST
from examples.physics.rendering import ce_rendering_ladder as LAD
from examples.physics.rendering import ce_rendering_one_event as OE
from examples.physics.rendering import ce_rendering_pole as POLE
from examples.physics.rendering import ce_rendering_fp_ladder as FPL
from examples.physics.rendering import ce_rendering_higgs_weight as HW
from examples.physics.rendering import ce_rendering_open_predictions as OP
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


def _line_ending_hashes(path) -> set[str]:
    """SHA-256 of a frozen module under LF and CRLF endings: a checkout's autocrlf must not break the freeze."""
    lf = path.read_bytes().replace(b"\r\n", b"\n")
    return {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}



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
    assert manifest["model"]["registry_sha256"] in _line_ending_hashes(registry), (
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
        assert v2["model"][f"{key}_sha256"] in _line_ending_hashes(path), (
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
        assert v3["model"][f"{key}_sha256"] in _line_ending_hashes(path), (
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
        assert v4["model"][f"{key}_sha256"] in _line_ending_hashes(path), (
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
        assert v5["model"][f"{key}_sha256"] in _line_ending_hashes(path), (
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
        assert v6["model"][f"{key}_sha256"] in _line_ending_hashes(path), (
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
        assert v7["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
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
        assert v8["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
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
        assert v9["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
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
        assert v10["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
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
        assert v11["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
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
    assert r["planck_unit_factor - 1"] == pytest.approx(r["g^2/(16 pi^2)"], rel=1e-12)   # canonical one-loop factor


def test_axiom_proofs_qg1_c6_light_limit_and_bisector() -> None:
    s = AP.qg1_signalling_scan(trials=60)
    assert s["branch"] > 0.1 and s["weight"] < 1e-12            # Theorem A: only rho-sourcing is no-signalling
    assert AP.affine_functional_is_trace_form() < 1e-12          # affine => Tr(rho X)
    w = AP.wirtinger_scan(trials=500)
    assert w["min_ratio_random"] > 1.0 == w["ratio_cos"]         # Theorem B: cos is the unique minimiser
    assert abs(AP.imaginary_rapidity_speed(math.pi / 4)) == pytest.approx(1.0)   # Lemma C: limit at pi/4
    assert AP.minimax_bisector() == pytest.approx(math.pi / 8, abs=1e-9)         # Theorem D


def test_complex_boost_reproof_of_light_limit_and_bisector() -> None:
    for phi in (0.2, math.pi / 4, 1.2):                                        # C-0: timelike test is vacuous
        assert CB.tilted_axis_norm(phi) == pytest.approx(-1.0, abs=1e-12)
    assert CB.strip_equivalence_scan(trials=4000)["mismatches"] == 0            # C': E_rec <=> |Im z| < pi/4 <=> |tanh z| < 1
    assert CB.real_boost_invariance(trials=50) < 1e-12                          # C' (iii)
    r = CB.real_branch()                                                        # branch R rejected by the dipole
    assert r["null_angle"] == pytest.approx([math.pi / 4] * 2) and r["beta_over_dipole"] > 300
    assert r["gamma"] != pytest.approx(1 / math.cos(math.pi / 8), abs=1e-3)
    assert CB.haar_bisector() == pytest.approx(math.pi / 8, abs=1e-5)           # D'
    assert CB.lorentz_factor(1j * math.pi / 8) == pytest.approx(math.cos(math.pi / 8))
    t = CB.tilt_readings_today()                                                # K4
    assert max(t["fixed"], t["tangent_chord"], t["spiral"], t["spiral_future_max"]) < math.pi / 4


def test_e4_is_an_event_relation_not_a_weighted_trace_identity() -> None:
    assert E4.unweighted_ratio() == Fraction(3, 8)                              # class I at a = 1
    assert E4.class_i_kill()["e4_physical_alpha_s_max"] == pytest.approx(2 ** -1.5)
    scan = E4.class_ii_scan()
    assert scan["a^|S|"]["ratio"] == pytest.approx(0.375) and scan["a^2|S|"]["ratio"] == pytest.approx(0.375)
    assert all(abs(r["pull"]) > 3 for k, r in scan.items() if k != "E4 4a^4")  # all six families fail
    assert abs(scan["E4 4a^4"]["pull"]) < 1


def test_intermittent_rendering_blurs_and_reemerges_on_any_convex_cycle() -> None:
    for kind in ("circle", "ellipse:0.6", "ellipse:0.9", "random"):
        s = IR.loop_structure(kind)
        assert s["monotone"] and s["total"] == pytest.approx(math.pi, abs=1e-3)
        assert s["cross_pi4"] == 1 and s["cross_3pi4"] == 1
    t = IR.circle_timetable()
    assert t["psi0"] < math.pi / 4 and t["blur_starts_Gyr"] == pytest.approx(27.24, abs=0.05)
    assert IR.typical_tilt(n=100001)["phase_uniform"] == pytest.approx(math.pi / 8, abs=1e-9)
    f = IR.flux_variant_chi2()                                                  # real part is a gate, not a flux
    assert f["flux_rel"] - f["best_lcdm"] > 9 and f["flux_abs"] - f["best_lcdm"] > 9


def test_octant_is_the_boundary_between_me_and_outside() -> None:
    c = core(calibrated_alpha_s()[0])
    assert BD.best_tbm_overlap(BD.contrast_vector(0)) == pytest.approx(1.0)     # me = e is the TBM first column
    assert BD.best_tbm_overlap(BD.contrast_vector(2)) < 0.9                     # me = tau is no TBM column
    s13 = pmns_s2(c, 1)
    juno = BD.S12_DATA["JUNO 2025"]
    assert abs(BD.pull(BD.s12_tm1(s13), juno)) < 1.5 < 3 < BD.pull(BD.s12_tm2(s13), juno)   # boundary drawn (TM1)
    tm1 = BD.delta_from_column1(BD.s12_tm1(s13), pmns_s2(c, 3), s13)
    assert delta_pmns_tm1(c) == pytest.approx(tm1, abs=0.05)                   # T1 is the TM1 sum rule
    p = BD.tm1_phase_for_s2(c)
    assert p["cos_phi"] > 0 and p["s23sq_cos_pos"] < 0.5 < p["s23sq_cos_neg"]   # octant = side of the boundary
    assert BD.tm1_delta_for_octant(c, 0.5) == pytest.approx(270.0, abs=0.05)
    assert 250 < BD.tm1_delta_for_octant(c, 0.5445) < 290


def test_rendering_staircase_fixes_two_steps_and_leaves_the_octant_step_open() -> None:
    c = core(calibrated_alpha_s()[0])
    rules = ST.score_rules(c)
    survivors = {k for k, r in rules.items() if r["K1_pass"]}
    assert survivors == {"doubling 2^(m-1) (S2)", "linear m"}                 # B + data fix (h1, h2) = (1, 2)
    dbl, lin = rules["doubling 2^(m-1) (S2)"], rules["linear m"]
    assert dbl["s23sq"] == pytest.approx(pmns_s2(c, 3)) and lin["s23sq"] == pytest.approx(0.4667, abs=1e-4)
    assert dbl["cos_phi_over_half_sqrt_d"] == pytest.approx(1.0, abs=0.02)
    h3 = ST.data_step_height(c)["SK"]
    assert abs(h3[0] - 3) < h3[2] and abs(h3[0] - 4) < h3[2]                   # current data cannot pick the stair
    sk, nosk = ST.alignment("SK"), ST.alignment("noSK")
    assert sk["nu3_tau_minus_mu"] > 0 and sk["nu2_mu_minus_tau"] > 0           # staircase alignment holds in both columns
    assert nosk["nu3_tau_minus_mu"] < 0 < nosk["nu2_mu_minus_tau"]


def test_true_false_worlds_force_the_doubling_staircase_and_the_not_me_mirror() -> None:
    assert ST.world_heights() == (1, 2, 4)                                     # Bool: worlds where "me" is true
    assert ST.truth_bias_for_height(4) == 0.5 and ST.truth_bias_for_height(3) == 0.375   # linear needs a false bias
    p = ST.not_me_pair(core(calibrated_alpha_s()[0]))
    assert p["me"] + p["not_me"] == pytest.approx(1.0)                         # complement flips the octant
    assert p["me_vs_SK"] == pytest.approx(p["not_me_vs_noSK"], abs=0.05)


def test_bool_dictionary_complement_halves_ckm_right_angle_and_charge_conjugation() -> None:
    h = BO.complement_symmetry()["halves"]
    assert all(v == Fraction(1, 2) for v in h.values())                       # me, odd, majority: one 1/2
    t = BO.ckm_triangle_from_partition()
    assert t["partition"] and t["deg"] == pytest.approx((22.5, 67.5, 90.0))   # right angle forced
    cc = BO.charge_conjugation_on_full_space()
    assert cc["Q_flips"] and cc["Y_flips"] and cc["parity_flips"] and cc["even_is_generation"]
    assert all(d["value"] == d["bool"] for d in BO.dictionary())
    assert BO.coverage()["covered"] == "3/7"                                   # vocabulary is restrictive


def test_open_checks_projection_rank_o1_direction_2026_rescore_and_ledger() -> None:
    r = OC.projection_rank_scan()
    assert r["rank1_today"]["V39"] == pytest.approx(0.834, abs=1e-3)          # length readout: cos
    assert min(v["V39"] for k, v in r.items() if k != "rank1_today") > 0.9
    e = OC.frame_epochs()
    assert e["ruler_diff"] < 1e-6 < 0.05 < e["galaxy_min"]                    # C5 same-frame vs cross-frame
    o = OC.o1_direction()
    assert o["clock_dilation"]["SH0ES"] < -10 and abs(o["projection"]["SH0ES"]) < 0.5
    s = OC.rescore_2026()
    assert s["V39"]["after"] == pytest.approx(0.876, abs=1e-3) and s["43rows"]["after"] == pytest.approx(0.914, abs=1e-3)
    assert s["43rows"]["max_abs_pull"] < 3
    led = OC.ledger_chi2()
    assert led["chi2_without_chain"] == pytest.approx(35.21, abs=0.01) and led["net"] == pytest.approx(2.38, abs=0.01)


def test_ring_rule_is_first_order_weight_response_and_open_items_are_predictions() -> None:
    c = core(calibrated_alpha_s()[0])
    e = OP.ring_first_order_error(c)
    assert e["max_second_order_gap"] < 1e-3
    p = OP.predictions(c)
    assert p["P26_siren_H0"] == pytest.approx(73.356, abs=1e-2) and p["P26_if_rule_wrong"] == pytest.approx(67.772, abs=1e-2)
    assert p["P28_MH_over_MZ"] == pytest.approx(c["F"])


def test_rendering_predictions_v12_is_frozen() -> None:
    v11 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v11.json").read_text(encoding="utf-8"))
    v12 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v12.json").read_text(encoding="utf-8"))
    assert v12["supersedes_manifest_id"] == v11["manifest_id"]
    body = {k: v for k, v in v12.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v12["manifest_sha256"]
    for key, rel in v12["model"]["files"].items():
        assert v12["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v12 freeze: create v13 and keep earlier manifests")
    ids = {q["id"] for q in v12["predictions"]}
    assert {"P26", "P27", "P28"} <= ids and len(ids) == 28


def test_neutrino_mass_formula_audit() -> None:
    c = core(calibrated_alpha_s()[0])
    for key in ("NuFIT 6.0", "JUNO 2025"):
        e = NA.exponent_fit(key)
        assert abs(e["pull_5_8"]) < 1.5 and e["fractions_den_le_12_in_1sigma"] == ["5/8"]   # sharp exponent
    s = NA.scale_look_elsewhere(c)
    assert s["family"] == 2100 and s["actual_in_family"] and s["hits_1sigma"] == 2
    assert s["selection_bits"] == pytest.approx(10.04, abs=0.01)
    d = NA.distinctiveness(c)
    assert 0 < d["sum_minus_min"] < 0.5 and d["sum_meV"] < d["DESI_DR2_95_meV"]         # P17 ~ minimal normal ordering
    j = NA.juno_outlook(c)
    assert -1 < j["pull_now"] < 0 and j["pull_final_0.3pct_if_central_holds"] < -4


def test_hypothesis_ledger_and_val01_baseline_comparison() -> None:
    t = LG.ledger_totals()
    assert t["39"]["before"] == pytest.approx(23.2) and t["39"]["now"] == pytest.approx(20.9)
    assert t["43"]["now"] == pytest.approx(34.6)
    c39 = LG.compare("IV")
    assert c39["CE"]["rmse"] == pytest.approx(0.834, abs=1e-3) and c39["baseline"]["k"] == 19
    assert c39["CE"]["S"] < c39["baseline"]["S"] and LG.break_even_bits(c39) > 60        # overall: CE ahead
    blocks = LG.block_scores("IV")
    assert blocks["Q"]["AIC_base"] < blocks["Q"]["AIC_CE"] and blocks["Q"]["BIC_CE"] < blocks["Q"]["BIC_base"]
    b2 = LG.flexible_baseline_m("IV")
    assert b2["AIC_CE"] < b2["AIC_B2"] and b2["BIC_CE"] < b2["BIC_B2"]                    # macro edge survives B2
    for rf in ("IV", "full"):
        for mp in (False, True):
            m = LG.mdl_blocks(rf, mp)
            assert m["Q"]["L_base"] - m["Q"]["L_CE"] > 75                            # same currency: Q block too
            assert m["M"]["L_base"] - m["M"]["L_CE"] > 75


def test_o1_observable_map_projection_between_frames() -> None:
    inv = OM.ring_invariance()
    assert inv["bao_max_rel_change"] < 1e-12 and inv["theta_ratio"] == pytest.approx(1.0)   # K1
    rd = OM.present_units_rd()
    assert abs(rd["pull"]) < 1 and rd["rd_CE_present"] == pytest.approx(136.3, abs=0.2)      # K3
    cl = OM.classification()
    assert all(abs(r[4]) < 2 for r in cl["chapter_rows"])                                     # K4
    assert all(not r[4] for r in cl["file_display_only"])                                     # display only


def test_spread_within_calibration_classes() -> None:
    a = SP2.conjecture_a()
    assert abs(a["calibration_offset"]) < 0.2                                   # analysis reproduces DESI BAO+BBN
    assert a["CE_with_G1m"]["H0"] == pytest.approx(68.46, abs=0.05) and abs(a["CE_pred_vs_DESI_sigma"]) < 0.5
    assert a["G1m_shift"] > 0.5 and a["CE_no_gradient"]["H0"] == pytest.approx(67.8, abs=0.1)
    b = SP2.conjecture_b()
    assert b["order_pred"] != b["order_obs"]                                    # pre-registered order criterion kills B
    c = SP2.conjecture_c()
    assert c["C_weighted"] == pytest.approx(69.4, abs=0.05)


def test_phase_locking_reproduces_record_window_bisector_and_intermittency() -> None:
    assert PLK.locked_point(1 / math.sqrt(2))["theta_star"] == pytest.approx(math.pi / 8)   # D': half-load lock
    assert PLK.locked_point(1.0)["theta_star"] == pytest.approx(math.pi / 4)                 # C': lock limit
    assert not PLK.locked_point(1.01)["locked"]
    lam = PLK.laminar_lengths(loads=(1.002, 1.032), steps=200000)
    assert all(3.5 < v["scaled"] < 4.8 for v in lam.values())                                # type-I scaling
    d = PLK.slip_depth(steps=200000)
    assert d["edge"] > 0.9 and d["deep_blur"] < 0.05                                         # brief deep blur
    t = PLK.data_test()
    assert t["bao_chi2"] < t["bao_chi2_none"] and t["V39"] < 0.909 and t["V39"] > t["V39_current"]


def test_causal_lock_half_of_half_of_right_angle() -> None:
    h = CLK.two_halvings()
    assert h["theta_is_pi_8"] and h["phi_star_deg"] == pytest.approx(45.0)
    d = CLK.direction()
    assert abs(d["causal_SH0ES"]) < 1 and d["anti_causal_SH0ES"] < -10                # causality fixes the direction
    one_way = CLK.two_phase_lock(0.0, steps=100000)
    assert one_way["lock"] == pytest.approx(math.pi / 8, abs=1e-5) and abs(one_way["past_drift"]) < 1e-9
    mutual = CLK.two_phase_lock(0.2, steps=100000)
    assert mutual["lock"] < 0.3 and abs(mutual["past_drift"]) > 1                    # back-coupling drags records


def test_lock_history_relaxation_variants_fail_and_phase_is_still_turning() -> None:
    ref = LH.score("ref")
    assert ref["V39"] == pytest.approx(0.834, abs=1e-3) and not ref["kill"]
    for v in ("V-a", "V-b", "V-c"):
        assert LH.score(v)["kill"]                                                   # all relaxation histories fail
    assert LH.score("V-a")["theta_today"] < math.pi / 8                             # relaxation too slow to reach pi/8


def test_thermal_time_final_horizon_sets_the_rotation_and_cycle() -> None:
    ds = TT.score("V-dS")
    assert not ds["kill"] and ds["V39"] == pytest.approx(0.8407, abs=1e-3)       # one phase history for O1 + G1m
    eh = TT.score("V-EH")
    assert eh["kill"] and eh["h0_pulls"]["H0 SH0ES"] > 10                       # instantaneous horizon fails
    cy = TT.cycle_timetable()
    assert cy["restart_Gyr"] == pytest.approx(4 * cy["blur_Gyr"]) and cy["half_point_Gyr"] < cy["today_Gyr"]
    assert TT.exact_half_now()["Om_star"] == pytest.approx(0.3163, abs=2e-4)


def test_branching_extinction_derives_the_cosmic_composition() -> None:
    c = core(calibrated_alpha_s()[0])
    it = BRN.iterate_to_extinction(c["D"])
    assert it["limit"] == pytest.approx(c["q"], abs=1e-9)                      # GW: small root = extinction
    mc = BRN.monte_carlo(c["D"], trials=20000)
    assert abs(mc["extinct_fraction"] - c["q"]) < 3 * mc["mc_sigma"]
    laws = BRN.law_specificity()
    assert abs(laws["Poisson"]["pull"]) < 1 and all(abs(r["pull"]) > 10 for k, r in laws.items() if k != "Poisson")
    comp = BRN.composition()
    assert comp["Om_branch"] == pytest.approx(comp["Om_core"]) and comp["borel_P1"] == pytest.approx(0.8568, abs=1e-3)


def test_e4_anchor_and_self_consistent_fixed_point() -> None:
    anc = E4A.anchor()
    assert anc["e4_holds"] and anc["alpha_s"] == 0.125 and anc["sin_thetaW"] == 0.5          # Ind vacuum point
    fp = E4A.fixed_point(*E4A.ADOPTED)
    assert abs(fp["pull_s2"]) < 1 and abs(fp["pull_as_world"]) < 1
    scan = E4A.family_scan()
    assert scan["hits_1sigma"] == ["a/2pi|1+d/2pi"] and scan["n"] == 24                    # unique in the declared family
    nest = E4A.nested_truncations()
    assert nest[0]["pull_s2"] < -20 and abs(nest[-1]["pull_s2"] - fp["pull_s2"]) < 1e-6   # only the infinite nesting works
    rs = E4A.resummation_scan()
    assert rs["V1 exp outer"]["pull_s2"] > 3 and abs(rs["V3 exp inner"]["pull_s2"]) < 2


def test_fixed_point_is_a_biased_coin_on_gauge_axes_only() -> None:
    assert BCN.identity_with_fp() < 1e-12                                          # FP = coin 1/2 (1 +- a)
    assert BCN.coin_fixed_point("G")["pull_s2"] > 3                              # 1/g form rejected
    u = BCN.universality()
    assert abs(u["unbiased"]["pull_sin2beta"]) < 1 and u["biased"]["pull_sin2beta"] < -3   # flavour coin stays fair
    assert u["biased"]["sum"] == pytest.approx(180.0)


def test_boundary_loop_is_linear_and_lepton_ratio_disfavours_fixed_point() -> None:
    rel = BLP.alpha_free_relation()
    assert abs(rel["linear"]["pull"]) < 1 and rel["geometric"]["pull"] > 3 and 1 < rel["compound"]["pull"] < 3
    lep = BLP.alpha_from_lepton_rule()
    assert lep["alpha_s_lepton"] == pytest.approx(0.1179196, abs=2e-7) and abs(lep["E4_vs_lepton"]) < 1
    assert lep["FP_vs_lepton"] < -3                                                   # internal tension of FP


def test_higgs_mass_predicts_cosmic_matter_fraction() -> None:
    idn = HGC.identity()
    assert idn["F_minus_1"] == pytest.approx(idn["OmDM_over_OmL"])                 # alpha_s-free relation
    h = HGC.from_higgs()
    assert h["Om_true_from_Higgs"] == pytest.approx(0.3071, abs=2e-4)
    comp = HGC.comparisons()
    assert all(abs(v["pull"]) < 3 for k, v in comp.items() if isinstance(v, dict))


def test_z_is_the_pure_distinction_coin_of_me_at_the_e4_anchor() -> None:
    idn = DST.anchor_identity()
    assert idn["g_V_e"] == pytest.approx(0.0, abs=1e-15) and idn["g_L_e"] == -idn["g_R_e"] == -0.25
    assert idn["P_L_e"] == 0.5 and idn["width_ratio_me_over_other"] == pytest.approx(0.5)
    assert idn["anchor_owner"] == ["e"]                                                # only "me" owns the anchor
    assert DST.pure_axial_points() == {"e": 0.25, "nu": None, "u": pytest.approx(0.375), "d": pytest.approx(0.75)}
    b = DST.bosons_on_me(0.25)
    assert b["photon"]["g_L"] == b["photon"]["g_R"] and b["W"]["P_L"] == 1.0            # no distinction / projection


def test_distinction_coin_reads_the_carrier_coupling_not_the_decay_record() -> None:
    v = DST.verdict()
    for key in ("chi2", "chi2_with_FP"):
        assert v[key]["best"] == "MS-bar" and not v[key]["MS-bar"]["killed"]
        assert all(v[key][s]["killed"] for s in ("MS-bar (ND)", "effective", "on-shell"))
    direct = DST.sensitivity_direct()["effective (direct avg)"]
    assert 1 < direct["P36"] < 2                                                       # direct average cannot decide yet


def test_present_ladders_cannot_differ_physically_and_cchp_gap_is_analysis() -> None:
    spread = LAD.physical_spread()
    assert spread["theta_now"] == pytest.approx(math.pi / 8, rel=0.02)
    assert spread["z_max"]["kms"] < 0.15 and spread["z_eff"]["kms"] < 0.02         # R1: <= 0.12 km/s/Mpc
    w = LAD.worlds()
    assert w["M (73.36)"]["chi2_O1_O2"] < 1 and w["CCHP world (70.39)"]["chi2_O1_O2"] > 9
    assert not LAD.kill_check()["killed"]                                             # section 43.48 kill not met
    d = LAD.cchp_dissection()
    assert not d["R4_trouble"] and d["gap_matched_vs_R22"] < 1 < d["gap_published_vs_R22"]
    assert abs(d["rows"]["v2.7, all TRGB calibrators (35)"]["pull_M"]) < 1


def test_rendering_predictions_v13_is_frozen() -> None:
    v12 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v12.json").read_text(encoding="utf-8"))
    v13 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v13.json").read_text(encoding="utf-8"))
    assert v13["supersedes_manifest_id"] == v12["manifest_id"]
    body = {k: v for k, v in v13.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v13["manifest_sha256"]
    assert v13["manifest_sha256"] == "c4c45c266889c4e894be1d7036e110e0bd3d29a9c518fa60dbf27ba334721a66"
    for key, rel in v13["model"]["files"].items():
        assert v13["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v13 freeze: create v14 and keep earlier manifests")
    old = {q["id"]: q for q in v12["predictions"]}
    for q in v13["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v13["predictions"]}
    assert {"P29", "P30"} <= ids and len(ids) == 30
    assert "duplicate" in next(q for q in v13["predictions"] if q["id"] == "P20")["status_v13"]


def test_rendering_predictions_v14_is_frozen() -> None:
    v13 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v13.json").read_text(encoding="utf-8"))
    v14 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v14.json").read_text(encoding="utf-8"))
    assert v14["supersedes_manifest_id"] == v13["manifest_id"]
    body = {k: v for k, v in v14.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v14["manifest_sha256"]
    assert v14["manifest_sha256"] == "d8802b2fd3c13c66217dcaf0ef55669d57e8be2e118fbb77ee0e514860c2b5c2"
    for key, rel in v14["model"]["files"].items():
        assert v14["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v14 freeze: create v15 and keep earlier manifests")
    old = {q["id"]: q for q in v13["predictions"]}
    for q in v14["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v14["predictions"]}
    assert {"P31", "P32", "P33"} <= ids and len(ids) == 33 and len(v14["model"]["files"]) == 30


def test_rendering_predictions_v15_is_frozen() -> None:
    v14 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v14.json").read_text(encoding="utf-8"))
    v15 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v15.json").read_text(encoding="utf-8"))
    assert v15["supersedes_manifest_id"] == v14["manifest_id"]
    body = {k: v for k, v in v15.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v15["manifest_sha256"]
    assert v15["manifest_sha256"] == "13c22f3c935a183a6f8aef8a26bebd06536d2355051001f10cb7d30987e57d01"
    for key, rel in v15["model"]["files"].items():
        assert v15["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v15 freeze: create v16 and keep earlier manifests")
    old = {q["id"]: q for q in v14["predictions"]}
    for q in v15["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v15["predictions"]}
    assert "P34" in ids and len(ids) == 34 and len(v15["model"]["files"]) == 32


def test_rendering_predictions_v16_is_frozen() -> None:
    v15 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v15.json").read_text(encoding="utf-8"))
    v16 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v16.json").read_text(encoding="utf-8"))
    assert v16["supersedes_manifest_id"] == v15["manifest_id"]
    body = {k: v for k, v in v16.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v16["manifest_sha256"]
    assert v16["manifest_sha256"] == "339e3f1fff9a9b5086f27d28af25d294dd94bbd8e8cba613a4b3cf836d7702f4"
    for key, rel in v16["model"]["files"].items():
        assert v16["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v16 freeze: create v17 and keep earlier manifests")
    old = {q["id"]: q for q in v15["predictions"]}
    for q in v16["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v16["predictions"]}
    assert {"P35", "P36"} <= ids and len(ids) == 36 and len(v16["model"]["files"]) == 35


def test_rendering_predictions_v17_is_frozen() -> None:
    v16 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v16.json").read_text(encoding="utf-8"))
    v17 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v17.json").read_text(encoding="utf-8"))
    assert v17["supersedes_manifest_id"] == v16["manifest_id"]
    body = {k: v for k, v in v17.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v17["manifest_sha256"]
    assert v17["manifest_sha256"] == "c54eb49474aa01667c6ace5d72abc87584ced5f6cf58167c7f3c9c913647758a"
    for key, rel in v17["model"]["files"].items():
        assert v17["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v17 freeze: create v18 and keep earlier manifests")
    old = {q["id"]: q for q in v16["predictions"]}
    assert {q["id"] for q in v17["predictions"]} == set(old)                   # no prediction added
    for q in v17["predictions"]:
        assert q["value"] == old[q["id"]]["value"]                            # no value changed
    p36 = next(q for q in v17["predictions"] if q["id"] == "P36")
    assert "decoupled" in p36["status_v17"] and len(v17["model"]["files"]) == 36


def test_rendering_predictions_v18_is_frozen() -> None:
    v17 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v17.json").read_text(encoding="utf-8"))
    v18 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v18.json").read_text(encoding="utf-8"))
    assert v18["supersedes_manifest_id"] == v17["manifest_id"]
    body = {k: v for k, v in v18.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v18["manifest_sha256"]
    assert v18["manifest_sha256"] == "2300ba99c67dbbf1d74018349dd222c5edcedc2dd4c07dc8ab034765a254ae42"
    for key, rel in v18["model"]["files"].items():
        assert v18["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v18 freeze: create v19 and keep earlier manifests")
    old = {q["id"]: q for q in v17["predictions"]}
    for q in v18["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v18["predictions"]}
    assert "P37" in ids and len(ids) == 37 and len(v18["model"]["files"]) == 38


def test_rendering_predictions_v19_is_frozen() -> None:
    v18 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v18.json").read_text(encoding="utf-8"))
    v19 = json.loads((PREREGISTRATION_ROOT / "rendering_predictions_v19.json").read_text(encoding="utf-8"))
    assert v19["supersedes_manifest_id"] == v18["manifest_id"]
    body = {k: v for k, v in v19.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == v19["manifest_sha256"]
    assert v19["manifest_sha256"] == "17274b1d032e922a4a4eebeb9ef5f26d4b0c6aa335a02058f9d9417602cd8f2d"
    for key, rel in v19["model"]["files"].items():
        assert v19["model"]["sha256"][key] in _line_ending_hashes((REPO_ROOT / rel)), (
            f"{key} changed after v19 freeze: create v20 and keep earlier manifests")
    old = {q["id"]: q for q in v18["predictions"]}
    for q in v19["predictions"]:
        if q["id"] in old:
            assert q["value"] == old[q["id"]]["value"]                        # no carried value changed
    ids = {q["id"] for q in v19["predictions"]}
    assert "P38" in ids and len(ids) == 38 and len(v19["model"]["files"]) == 41


def test_distinction_equals_indistinction_at_the_z_pole_record() -> None:
    assert POLE.coin_reading() == {"mean_distinction": 0.0, "g_V": 0.0}
    p = POLE.pole_reading()
    assert p["real_at_pole"] == 0.0 and abs(p["phase_deg_at_pole"]) == pytest.approx(90.0)  # pure record
    assert all(abs(abs(v) - 90) == pytest.approx(45, abs=0.5) for v in p["phase_deg_half_width"].values())
    r = POLE.record_alpha_check()
    assert not r["killed"] and -2 < r["Z pole (R_l, Gamma_Z, sigma_had)"]["E4"] < -1   # current -1.55 sigma


def test_fixed_point_e_ladder_cannot_identify_the_missing_correction() -> None:
    t = FPL.target_eps()
    assert t["eps"] == pytest.approx(-4.94e-5, abs=0.05e-5) and t["sigma"] == pytest.approx(1.32e-5, abs=0.05e-5)
    s = FPL.scan()
    assert s["hits"] == ["F2 -lam^1 e^-6", "F2 -lam^2 e^-2", "F3 -lam q^2"]
    assert all(s["rows"][f"F1 n={n}"]["pull_lepton"] > 10 for n in (2, 3, 4, "inf"))   # full e-series rejected
    seen = FPL.seen_before()
    assert all(abs(v["pull_lepton"]) < 0.3 for v in seen.values())                    # four-way tie
    ch = FPL.chance_hits()
    assert 0.2 < ch["F2"]["p_at_least_one"] < 0.35 and ch["F1"]["p_at_least_one"] < 0.05


def test_higgs_weight_is_the_survival_partition_one_step() -> None:
    idn = HW.partition_identity()
    assert all(abs(v) < 1e-12 for v in idn.values())                                  # F = survival partition
    forms = HW.step_forms()
    assert not forms["one step 1+m"]["killed"]
    assert forms["compound e^m"]["killed"] and forms["full recursion 1/(1-m)"]["killed"]
    h = HW.hierarchy_cross_check()
    assert abs(h["implied_vs_core"]) < 1 and 1 < h["implied_vs_higgs"] < 2
    assert h["M_H_required"] == pytest.approx(125.36, abs=0.01) and h["M_H_required_sigma"] < 0.05
    q = HW.q2_universality()
    assert q["q2"]["chi2"] < q["plain"]["chi2"] and not q["adopt"]                    # breaks at v/M_Pl
    assert q["q2"]["pulls"]["v/M_Pl (th+exp)"] > 2


def test_one_coin_one_event_unique_crossing_at_mz() -> None:
    mono = OE.monotonicity()
    assert mono["crossings"] == 1 and mono["f_increasing"] and mono["robust_below_MW"]
    ev = OE.event_scale()
    assert ev["mu_lo"] < 91.1876 < ev["mu_hi"] and abs(ev["pull_MZ"]) < 1
    assert ev["pulls"]["M_W"] > 2 and ev["pulls"]["M_H"] < -3


def test_pantheon_holdout_keeps_all_ce_branches_within_two_sigma() -> None:
    from examples.physics.rendering import ce_rendering_sn_holdout as SN
    out = SN.score()
    assert out["N"] == 40
    assert 0.25 < out["best"]["Om"] < 0.35
    v = SN.verdict(out)
    assert v["L0"] == v["W2"] == v["W3"] == "pass" and v["W2 vs L0"] == "kept"
    assert out["W2"] - out["best"]["chi2"] == pytest.approx(0.470, abs=0.02)


def test_exact_fd_neutrinos_resolve_theta_path_gap_without_flipping_verdicts() -> None:
    from examples.physics.rendering import ce_rendering_theta_nu as TN
    p = TN.pulls()
    assert abs(p["L0"] - p["L0_pathD"]) < 0.5
    assert abs(p["W2"]) < abs(p["L0"])          # W2 still improves theta* over constant vacuum
    assert abs(p["W3b"]) < 1.0 and p["W3a"] < -5.0   # W3 (b) passes, (a) stays rejected
    assert TN.variant_v_with_fd_theta() == pytest.approx(0.833, abs=2e-3)


def test_w3_growth_transfer_removes_its_s8_excess_but_w3_stays_competing() -> None:
    from examples.physics.rendering import ce_rendering_w3_growth as W3
    out = W3.compare()
    w2, first, corr = out["W2"], out["W3_first_order"], out["W3_corrected"]
    assert corr["S8"] < w2["S8"] < first["S8"]          # dilution by homogeneous new dust lowers S8
    assert corr["rows43"] < w2["rows43"]
    assert corr["V39"] > w2["V39"]                        # pre-registered rule needs both lower
    assert out["verdict"] == "W3 stays competing"


def test_data_version_sensitivity_is_dominated_by_the_nufit_sk_choice() -> None:
    from examples.physics.rendering import ce_rendering_data_sensitivity as DS
    tab = {(r["pmns"], r["lens"], r["h0"]): r for r in DS.table()}
    assert not any(r["flag_3sigma"] for k, r in tab.items() if k[0] == "SK")
    assert all(r["flag_3sigma"] and r["W2"] > 1.4 for k, r in tab.items() if k[0] == "noSK")
    # the W2/W3 ranking follows the DES lensing row only
    assert all((r["better"] == "W3") == (r["lens"] in ("both", "DES")) for r in tab.values())
