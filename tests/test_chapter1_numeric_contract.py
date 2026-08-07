from __future__ import annotations

import math
from pathlib import Path

from examples.physics.ce_residual_forward_model import (
    bao_chi2_with_covariance,
    chapter1_canonical_params,
    chi_square_survival,
    chi_square_verdict,
    early_hubble_rate_s_inverse,
    named_bao_dataset,
    sound_horizon_selection,
)


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "docs" / "1_강의"

ALPHA_S = 0.1180
ALPHA_EM = 1.0 / 127.95


def _bisect(function, low: float, high: float, *, steps: int = 200) -> float:
    f_low = function(low)
    f_high = function(high)
    assert f_low * f_high < 0.0
    for _ in range(steps):
        middle = 0.5 * (low + high)
        f_middle = function(middle)
        if f_low * f_middle <= 0.0:
            high = middle
            f_high = f_middle
        else:
            low = middle
            f_low = f_middle
    return 0.5 * (low + high)


def _track_a() -> tuple[float, float, float, float, float, float, float]:
    sin2_a = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta_a = sin2_a * (1.0 - sin2_a)
    depth_a = 3.0 + delta_a

    def fixed_point(x: float) -> float:
        return math.exp(-depth_a * (1.0 - x)) - x

    x_star = _bisect(fixed_point, 0.0, 1.0 / depth_a)

    ratio_dark = ALPHA_S * depth_a * (1.0 + x_star * delta_a)
    omega_cdm = (1.0 - x_star) * ratio_dark / (1.0 + ratio_dark)
    omega_de = (1.0 - x_star) / (1.0 + ratio_dark)
    return sin2_a, delta_a, depth_a, x_star, ratio_dark, omega_cdm, omega_de


def _inflation_observables() -> tuple[float, float, float, float, float]:
    xi = ALPHA_S ** (1.0 / 3.0)
    target_efolds = 57.1999
    scalar_amplitude = 2.10e-9

    def epsilon(phi: float) -> float:
        return 8.0 / (phi * phi * (1.0 + xi * (1.0 + 6.0 * xi) * phi * phi))

    phi_end = _bisect(lambda phi: epsilon(phi) - 1.0, 0.1, 5.0)

    def efolds(phi: float) -> float:
        return (
            (1.0 + 6.0 * xi) * (phi * phi - phi_end * phi_end) / 8.0
            - 0.75
            * math.log((1.0 + xi * phi * phi) / (1.0 + xi * phi_end * phi_end))
        )

    phi_star = _bisect(lambda phi: efolds(phi) - target_efolds, phi_end, 30.0)

    p = phi_star
    conformal = 1.0 + xi * p * p
    kinetic = (1.0 + xi * (1.0 + 6.0 * xi) * p * p) / conformal**2
    d_log_u = 4.0 / (p * conformal)
    d2_log_u = -4.0 * (1.0 + 3.0 * xi * p * p) / (p * p * conformal**2)
    u_second_over_u = d2_log_u + d_log_u**2

    numerator = 1.0 + xi * (1.0 + 6.0 * xi) * p * p
    numerator_prime = 2.0 * xi * (1.0 + 6.0 * xi) * p
    conformal_prime = 2.0 * xi * p
    kinetic_prime = (
        numerator_prime / conformal**2
        - 2.0 * numerator * conformal_prime / conformal**3
    )

    eps = epsilon(p)
    eta = u_second_over_u / kinetic - kinetic_prime * d_log_u / (2.0 * kinetic**2)
    n_s = 1.0 - 6.0 * eps + 2.0 * eta
    tensor_ratio = 16.0 * eps
    lambda_4 = (
        scalar_amplitude
        * 96.0
        * math.pi**2
        * eps
        * conformal**2
        / p**4
    )
    return phi_end, phi_star, n_s, tensor_ratio, lambda_4


def test_track_a_current_numeric_contract() -> None:
    sin2_a, delta_a, depth_a, x_star, ratio_dark, omega_cdm, omega_de = _track_a()

    assert math.isclose(sin2_a, 0.2315097758079336, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(delta_a, 0.17791299951329392, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(depth_a, 3.177912999513294, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(x_star, 0.04863825851598631, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(ratio_dark, 0.3782386966438831, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(omega_cdm, 0.26108817435761356, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(omega_de, 0.6902735671264001, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(x_star + omega_cdm + omega_de, 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert depth_a * x_star < 1.0
    assert abs(math.exp(-depth_a * (1.0 - x_star)) - x_star) < 1e-15
    assert math.isclose(
        -math.log(x_star) / (1.0 - x_star),
        depth_a,
        rel_tol=0.0,
        abs_tol=1e-14,
    )


def test_track_b_has_two_positive_roots_and_sm_hierarchy_output() -> None:
    def equation(a: float) -> float:
        return (
            a
            + ALPHA_EM / (4.0 * a ** (4.0 / 3.0))
            + ALPHA_EM
            - 1.0 / (2.0 * math.pi)
        )

    low = _bisect(equation, 0.03, 0.08)
    sm = _bisect(equation, 0.08, 0.2)
    alpha_w = ALPHA_EM / (4.0 * sm ** (4.0 / 3.0))
    sin2_w = ALPHA_EM / alpha_w

    assert math.isclose(low, 0.0528678687103, rel_tol=0.0, abs_tol=5e-13)
    assert math.isclose(sm, 0.1173186646973, rel_tol=0.0, abs_tol=5e-13)
    assert math.isclose(sin2_w, 0.22972916798, rel_tol=0.0, abs_tol=5e-11)
    assert math.isclose(alpha_w, 0.03402072544, rel_tol=0.0, abs_tol=5e-11)
    assert sm > alpha_w > ALPHA_EM


def test_plus_sign_finite_xi_benchmark_is_recomputed() -> None:
    phi_end, phi_star, n_s, tensor_ratio, lambda_4 = _inflation_observables()

    assert math.isclose(
        ALPHA_S ** (1.0 / 3.0),
        0.4904868132,
        rel_tol=0.0,
        abs_tol=5e-11,
    )
    assert math.isclose(phi_end, 1.3385417693, rel_tol=0.0, abs_tol=5e-10)
    assert math.isclose(phi_star, 11.0974588093, rel_tol=0.0, abs_tol=5e-10)
    assert math.isclose(n_s, 0.96617113848, rel_tol=0.0, abs_tol=5e-11)
    assert math.isclose(tensor_ratio, 0.00434561033, rel_tol=0.0, abs_tol=5e-11)
    assert math.isclose(lambda_4, 1.3434991214e-10, rel_tol=5e-10, abs_tol=0.0)


def test_baryon_to_photon_density_conversion_is_recomputed() -> None:
    _, _, _, omega_b, _, _, _ = _track_a()
    hubble_si = 67.4 * 1000.0 / 3.0856775814913673e22
    gravitational_constant = 6.67430e-11
    proton_mass = 1.67262192595e-27
    boltzmann_constant = 1.380649e-23
    cmb_temperature = 2.7255
    reduced_planck_constant = 1.054571817e-34
    speed_of_light = 299792458.0
    zeta_3 = 1.202056903159594

    critical_density = (
        3.0 * hubble_si**2 / (8.0 * math.pi * gravitational_constant)
    )
    photon_density = (
        2.0
        * zeta_3
        / math.pi**2
        * (
            boltzmann_constant
            * cmb_temperature
            / (reduced_planck_constant * speed_of_light)
        )
        ** 3
    )
    eta_b = omega_b * critical_density / (proton_mass * photon_density)

    assert math.isclose(
        eta_b,
        6.041176330217133e-10,
        rel_tol=0.0,
        abs_tol=1e-22,
    )


def test_declared_desi_dr2_bao_only_partial_gate_is_recomputed() -> None:
    _, _, _, omega_b, ratio_dark, omega_cdm, omega_de = _track_a()
    dataset = named_bao_dataset("desi-dr2-all")
    assert len(dataset.data) == 13
    assert len(dataset.covariance) == 13

    common = {
        "h0": 67.4,
        "w0": -1.0,
        "wa": 0.0,
    }
    h = common["h0"] / 100.0
    omega_rad0_eh = (
        2.469e-5 * (1.0 + 0.22710731766 * 3.044) / h**2
    )
    omega_rem0_eh = 1.0 - omega_b - omega_rad0_eh
    omega_cdm0_eh = omega_rem0_eh * ratio_dark / (1.0 + ratio_dark)
    omega_de0_eh = omega_rem0_eh / (1.0 + ratio_dark)
    assert math.isclose(
        omega_rad0_eh,
        9.192332265998932e-5,
        rel_tol=0.0,
        abs_tol=1e-18,
    )
    assert math.isclose(
        omega_cdm0_eh,
        0.26106294726317864,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        omega_de0_eh,
        0.6902068708981751,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        omega_b + omega_cdm0_eh + omega_de0_eh + omega_rad0_eh,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    external_params = chapter1_canonical_params(
        **common,
        rd_mode="external",
        rd_mpc=147.09,
    )
    early_params = chapter1_canonical_params(
        **common,
        rd_mode="early-universe",
    )
    assert math.isclose(external_params.omega_b0, omega_b, abs_tol=1e-15)
    assert math.isclose(external_params.omega_dm0, omega_cdm, abs_tol=1e-15)
    assert math.isclose(external_params.omega_lambda0, omega_de, abs_tol=1e-15)
    assert math.isclose(early_params.omega_b0, omega_b, abs_tol=1e-15)
    assert math.isclose(early_params.omega_dm0, omega_cdm0_eh, abs_tol=1e-15)
    assert math.isclose(early_params.omega_lambda0, omega_de0_eh, abs_tol=1e-15)
    assert external_params.density_preset == "chapter1"
    assert early_params.density_preset == "chapter1"
    assert math.isclose(
        early_params.omega_k0_background,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    cases = (
        (
            external_params,
            147.09,
            40.2014508578499,
            1.2828316784869182e-4,
        ),
        (
            early_params,
            151.50842877450663,
            41.90607733129171,
            6.784763339878187e-5,
        ),
    )
    for params, expected_rd, expected_chi2, expected_p in cases:
        rd_mpc = sound_horizon_selection(params).rd_mpc
        chi2 = bao_chi2_with_covariance(dataset.data, dataset.covariance, params)
        p_value = chi_square_survival(chi2, len(dataset.data))
        assert math.isclose(rd_mpc, expected_rd, rel_tol=0.0, abs_tol=5e-9)
        assert math.isclose(chi2, expected_chi2, rel_tol=0.0, abs_tol=5e-10)
        assert math.isclose(p_value, expected_p, rel_tol=0.0, abs_tol=5e-14)
        assert p_value < 1e-3
        assert chi_square_verdict(p_value) == "REJECT"
        if params.rd_mode == "early-universe":
            h0_from_early_background = (
                early_hubble_rate_s_inverse(0.0, params)
                * 3.0856775814913673e19
            )
            assert math.isclose(
                h0_from_early_background,
                params.h0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )


def test_ckm_benchmark_jarlskog_is_recomputed_from_declared_inputs() -> None:
    s12, s23, s13 = 0.22724210, 0.04168209, 0.00372494
    delta_q_rad = 1.2
    c12 = math.sqrt(1.0 - s12**2)
    c23 = math.sqrt(1.0 - s23**2)
    c13 = math.sqrt(1.0 - s13**2)
    jarlskog = c12 * c23 * c13**2 * s12 * s23 * s13 * math.sin(delta_q_rad)

    assert math.isclose(
        jarlskog,
        3.199594285262745e-5,
        rel_tol=0.0,
        abs_tol=1e-16,
    )


def test_repeated_display_values_are_consistent_across_chapter() -> None:
    texts = {
        path.name: path.read_text(encoding="utf-8")
        for path in CHAPTER.glob("*.md")
    }
    required = (
        "0.2315097758",
        "0.1779129995",
        "3.1779129995",
        "0.0486382585",
        "0.3782386966",
        "0.2610881744",
        "0.6902735671",
        "0.96617114",
        "0.00434561",
        "1.3434991\\times10^{-10}",
    )
    for filename in ("A_연역적_유도.md", "C_다섯_상수.md"):
        for value in required:
            assert value in texts[filename], (filename, value)

    for filename in texts:
        assert "0.4904868132" in texts[filename], filename

    ledger_values = (
        "0.2315097758079336",
        "0.17791299951329392",
        "3.177912999513294",
        "0.04863825851598631",
        "0.3782386966438831",
        "0.26108817435761356",
        "0.6902735671264001",
        "0.0528678687103",
        "0.1173186646973",
        "0.22972916798",
        "0.03402072544",
        "40.20145086",
        "41.90607733",
        "151.50842877",
        "6.78476334\\times10^{-5}",
        "9.192332265998932\\times10^{-5}",
        "0.26106294726317864",
        "0.6902068708981751",
        "6.041176330\\times10^{-10}",
    )
    for value in ledger_values:
        assert value in texts["D_정합성_원장.md"], value

    for filename in ("B_귀납적_유도.md", "D_정합성_원장.md"):
        for value in ("40.20145086", "41.90607733", "151.50842877"):
            assert value in texts[filename], (filename, value)
    assert "3.199594285\\times10^{-5}" in texts["B_귀납적_유도.md"]
    for value in ("0.22724210", "0.04168209", "0.00372494", "1.2\\,{\\rm rad}"):
        assert value in texts["D_정합성_원장.md"], value

    combined = "\n".join(texts.values())
    for stale in (
        "3.17776",
        "0.0486466333",
        "6.041176336\\times10^{-10}",
        "3.7\\times10^{-18}",
        "41.19455358",
        "8.86018138\\times10^{-5}",
        "41.28551570",
        "151.50522753",
        "0.6901816438037401",
        "41.28589334",
        "8.56244625\\times10^{-5}",
    ):
        assert stale not in combined, stale


def test_retired_eh_hybrid_snapshot_is_absent_from_all_docs() -> None:
    retired_tokens = (
        "151.505",
        "41.19455",
        "8.8602",
        "41.285515",
        "41.285893",
        "EH-hybrid",
        "EH hybrid",
        "1019.907163",
        "4.3915841176",
        "0.01270646192",
        "147.649757605",
        "관측 공동검정 | 실행됨",
        "이제 CE는 대화에서 완성된 이론으로",
    )
    hits: list[tuple[str, str]] = []
    for path in (ROOT / "docs").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for token in retired_tokens:
            if token.casefold() in text.casefold():
                hits.append((str(path.relative_to(ROOT)), token))
    assert not hits, hits
