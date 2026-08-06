"""Recompute the canonical CE benchmark using only Python's standard library."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "CANONICAL_NUMERIC_MANIFEST_2026-08-06.json"


def bisect(fn, lo: float, hi: float, *, tol: float = 1e-14) -> float:
    flo = fn(lo)
    fhi = fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"root is not bracketed: f({lo})={flo}, f({hi})={fhi}")
    for _ in range(240):
        mid = (lo + hi) / 2.0
        fmid = fn(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    raise ArithmeticError("bisection did not converge")


def adaptive_simpson(fn, lo: float, hi: float, *, tol: float = 1e-12) -> float:
    def simpson(a: float, b: float) -> float:
        c = (a + b) / 2.0
        return (b - a) * (fn(a) + 4.0 * fn(c) + fn(b)) / 6.0

    def recurse(a: float, b: float, whole: float, eps: float, depth: int) -> float:
        c = (a + b) / 2.0
        left = simpson(a, c)
        right = simpson(c, b)
        correction = left + right - whole
        if depth <= 0 or abs(correction) <= 15.0 * eps:
            return left + right + correction / 15.0
        return recurse(a, c, left, eps / 2.0, depth - 1) + recurse(
            c, b, right, eps / 2.0, depth - 1
        )

    return recurse(lo, hi, simpson(lo, hi), tol, 24)


def regularized_gamma_q(shape: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(shape, x), without SciPy."""

    if shape <= 0.0 or x < 0.0:
        raise ValueError("regularized_gamma_q requires shape > 0 and x >= 0")
    if x == 0.0:
        return 1.0

    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)
    epsilon = 3.0e-15
    tiny = 1.0e-300

    if x < shape + 1.0:
        term = 1.0 / shape
        series = term
        shifted_shape = shape
        for _ in range(1, 10000):
            shifted_shape += 1.0
            term *= x / shifted_shape
            series += term
            if abs(term) <= abs(series) * epsilon:
                return 1.0 - series * math.exp(log_prefactor)
        raise ArithmeticError("lower incomplete-gamma series did not converge")

    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for index in range(1, 10000):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        update = d * c
        fraction *= update
        if abs(update - 1.0) <= epsilon:
            return math.exp(log_prefactor) * fraction
    raise ArithmeticError("upper incomplete-gamma continued fraction did not converge")


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(
            f"{name}: actual={actual:.17g}, expected={expected:.17g}, tol={tol:g}"
        )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inputs = manifest["inputs"]
    expected = manifest["derived"]
    diagnostics = manifest["diagnostics"]

    alpha_s = float(inputs["alpha_s_mz"])
    s_a2 = 4.0 * alpha_s ** (4.0 / 3.0)
    delta_n = s_a2 * (1.0 - s_a2)
    depth = 3.0 + delta_n
    fixed = bisect(lambda x: math.exp(-(1.0 - x) * depth) - x, 0.0, 1.0 / depth)
    multiplier = depth * fixed
    ratio = alpha_s * depth * (1.0 + fixed * delta_n)
    omega_b = fixed
    omega_dm = (1.0 - fixed) * ratio / (1.0 + ratio)
    omega_de = (1.0 - fixed) / (1.0 + ratio)

    assert_close("s_A2", s_a2, expected["s_A2"], 2e-15)
    assert_close("delta_n", delta_n, expected["delta_n"], 2e-15)
    delta_n_squared = delta_n * delta_n
    assert_close("delta_n_squared", delta_n_squared, expected["delta_n_squared"], 2e-15)
    assert_close("D_n", depth, expected["D_n"], 2e-15)
    assert_close("bootstrap_x", fixed, expected["bootstrap_x"], 2e-14)
    assert_close("bootstrap residual", math.exp(-(1.0 - fixed) * depth) - fixed, 0.0, 2e-14)
    assert_close("inverse D(x)", -math.log(fixed) / (1.0 - fixed), depth, 3e-13)
    assert_close("bootstrap_multiplier", multiplier, expected["bootstrap_multiplier"], 7e-14)
    if not multiplier < 1.0:
        raise AssertionError("the selected low branch is not contractive")
    assert_close("dark_ratio_R", ratio, expected["dark_ratio_R"], 3e-15)
    assert_close("omega_b", omega_b, expected["omega_b"], 2e-14)
    assert_close("omega_dm", omega_dm, expected["omega_dm"], 2e-14)
    assert_close("omega_de", omega_de, expected["omega_de"], 2e-14)
    assert_close("density closure", omega_b + omega_dm + omega_de, 1.0, 2e-15)

    # Present-density conversion to eta_b. H0 and T_CMB are explicit inputs.
    megaparsec_m = 3.0856775814913673e22
    newton_g = 6.67430e-11
    proton_mass_kg = 1.67262192369e-27
    boltzmann = 1.380649e-23
    hbar = 1.054571817e-34
    light_speed = 299792458.0
    zeta_3 = 1.202056903159594
    h0_si = float(inputs["h0_km_s_mpc"]) * 1000.0 / megaparsec_m
    rho_critical = 3.0 * h0_si**2 / (8.0 * math.pi * newton_g)
    photon_density = (
        2.0
        * zeta_3
        / math.pi**2
        * (boltzmann * float(inputs["cmb_temperature_k"]) / (hbar * light_speed)) ** 3
    )
    eta_b_density = omega_b * rho_critical / (proton_mass_kg * photon_density)
    assert_close("eta_b_density", eta_b_density, expected["eta_b_density"], 2e-22)

    # Horizon-readout arithmetic contract.  This verifies the displayed formula,
    # not the physical validity of the readout ansatz or its selector.
    horizon_efolds = (
        1.5 * depth * float(inputs["gauge_state_count"])
    )
    horizon_defect = delta_n * (1.0 - fixed)
    horizon_log_s_low = (
        math.pi**2 * horizon_efolds / 2.0 - math.pi * horizon_defect
    )
    horizon_h0_low = (
        math.sqrt(math.pi / math.exp(horizon_log_s_low))
        / float(inputs["planck_time_s"])
        * megaparsec_m
        / 1000.0
    )
    horizon_h0_high = horizon_h0_low * math.exp(horizon_defect / 2.0)
    assert_close(
        "horizon_efolds", horizon_efolds, expected["horizon_efolds"], 2e-13
    )
    assert_close(
        "horizon_defect", horizon_defect, expected["horizon_defect"], 2e-15
    )
    assert_close(
        "h0_readout_low_km_s_mpc",
        horizon_h0_low,
        expected["h0_readout_low_km_s_mpc"],
        2e-11,
    )
    assert_close(
        "h0_readout_high_km_s_mpc",
        horizon_h0_high,
        expected["h0_readout_high_km_s_mpc"],
        2e-11,
    )

    # The chi-square values are frozen outputs of the documented 13-block
    # covariance run.  Recompute their survival probabilities and enforce the
    # preregistered REJECT threshold; do not present this as a likelihood rerun.
    external_chi2 = float(diagnostics["fixed_background_external_rd_chi2"])
    external_dof = int(diagnostics["fixed_background_external_rd_dof"])
    external_p = regularized_gamma_q(external_dof / 2.0, external_chi2 / 2.0)
    assert_close(
        "fixed_background_external_rd_p",
        external_p,
        diagnostics["fixed_background_external_rd_p"],
        2e-12,
    )
    eh_chi2 = float(diagnostics["fixed_background_eh_rd_chi2"])
    eh_dof = int(diagnostics["fixed_background_eh_rd_dof"])
    eh_p = regularized_gamma_q(eh_dof / 2.0, eh_chi2 / 2.0)
    assert_close(
        "fixed_background_eh_rd_p",
        eh_p,
        diagnostics["fixed_background_eh_rd_p"],
        2e-12,
    )
    if external_p >= 0.0027 or eh_p >= 0.0027:
        raise AssertionError("fixed-background rejection verdict is inconsistent")
    if diagnostics["fixed_background_verdict"] != "REJECT":
        raise AssertionError("fixed-background verdict label is inconsistent")

    # The two distinct scalar bridge scales must follow the current Track-A delta.
    light_mass = float(inputs["proton_mass_mev"]) * delta_n_squared
    light_length = float(inputs["hbar_c_mev_fm"]) / light_mass
    portal_bare_mass = float(inputs["portal_bare_mass_gev"])
    portal_mass = math.sqrt(
        portal_bare_mass**2
        + delta_n_squared * float(inputs["electroweak_vev_gev"]) ** 2
    )
    assert_close("light_bridge_mass_mev", light_mass, expected["light_bridge_mass_mev"], 2e-12)
    assert_close("light_bridge_length_fm", light_length, expected["light_bridge_length_fm"], 2e-13)
    assert_close("portal_bridge_mass_gev", portal_mass, expected["portal_bridge_mass_gev"], 2e-12)

    # The same normalized Z2 portal benchmark must reproduce its invisible width.
    higgs_mass = float(inputs["higgs_mass_gev"])
    portal_lambda = delta_n_squared
    portal_width_gev = (
        portal_lambda**2
        * float(inputs["electroweak_vev_gev"]) ** 2
        / (8.0 * math.pi * higgs_mass)
        * math.sqrt(1.0 - 4.0 * portal_mass**2 / higgs_mass**2)
    )
    portal_width_mev = 1000.0 * portal_width_gev
    portal_br = portal_width_mev / (
        portal_width_mev + float(inputs["higgs_sm_width_mev"])
    )
    portal_br_limit = float(inputs["higgs_invisible_br_limit_95cl"])
    assert_close(
        "portal_invisible_width_mev",
        portal_width_mev,
        expected["portal_invisible_width_mev"],
        2e-12,
    )
    assert_close(
        "portal_invisible_branching_ratio",
        portal_br,
        expected["portal_invisible_branching_ratio"],
        2e-14,
    )
    if portal_br <= portal_br_limit:
        raise AssertionError("canonical portal benchmark no longer violates BR_inv")

    # Correct CP-even neutral-scalar g-2 kernel.  The light limit is 3/2,
    # not 1/2; the finite canonical mass must reproduce the documented ratio.
    muon_mass = float(inputs["muon_mass_mev"])
    scalar_r = light_mass / muon_mass
    scalar_i_mu = adaptive_simpson(
        lambda z: (1.0 - z) ** 2
        * (1.0 + z)
        / ((1.0 - z) ** 2 + z * scalar_r**2),
        0.0,
        1.0,
    )
    scalar_ratio = scalar_i_mu / 1.5
    alpha_zero = 1.0 / float(inputs["alpha_em_zero_inverse"])
    portal_mass_mev = portal_mass * 1000.0
    wilson_b_mu = (
        alpha_zero
        / (2.0 * math.pi * math.e)
        * (muon_mass / portal_mass_mev) ** 2
    )
    finite_scalar_same_coupling = wilson_b_mu * scalar_ratio
    assert_close("scalar_loop_i_mu", scalar_i_mu, expected["scalar_loop_i_mu"], 5e-13)
    assert_close(
        "scalar_loop_ratio_to_light_limit",
        scalar_ratio,
        expected["scalar_loop_ratio_to_light_limit"],
        5e-13,
    )
    assert_close("wilson_b_mu", wilson_b_mu, expected["wilson_b_mu"], 2e-20)
    assert_close(
        "finite_scalar_same_coupling_mu",
        finite_scalar_same_coupling,
        expected["finite_scalar_same_coupling_mu"],
        2e-20,
    )

    # Track B: independent alpha_em(M_Z) input has two positive CE roots.
    alpha_em = 1.0 / float(inputs["alpha_em_mz_inverse_track_b"])
    track_b = lambda a: a + alpha_em / (4.0 * a ** (4.0 / 3.0)) + alpha_em - 1.0 / (2.0 * math.pi)
    root_low = bisect(track_b, 0.03, 0.08)
    root_sm = bisect(track_b, 0.08, 0.15)
    assert_close("track_b_low", root_low, 0.052867868709837505, 2e-14)
    assert_close("track_b_sm", root_sm, 0.11731866469727119, 2e-14)

    # Gamma recurrence: path-count CDF and energy-weighted CDF are unequal.
    shape = 2.5
    threshold = 1.3
    lower_a = adaptive_simpson(
        lambda u: (u ** (shape - 1.0)) * math.exp(-u), 0.0, threshold
    )
    lower_a1 = adaptive_simpson(
        lambda u: (u**shape) * math.exp(-u), 0.0, threshold
    )
    p_count = lower_a / math.gamma(shape)
    p_energy = lower_a1 / math.gamma(shape + 1.0)
    recurrence = threshold**shape * math.exp(-threshold) / math.gamma(shape + 1.0)
    assert_close("gamma recurrence", p_count - p_energy, recurrence, 2e-11)
    if not p_count > p_energy:
        raise AssertionError("count and energy-weighted Gamma CDFs were conflated")

    # A finite discrete B2 model: energy weighting must be normalized explicitly.
    probabilities = (0.2, 0.3, 0.5)
    energies = (1.0, 2.0, 4.0)
    survives = (1.0, 0.0, 1.0)
    weighted = sum(p * e * s for p, e, s in zip(probabilities, energies, survives))
    total = sum(p * e for p, e in zip(probabilities, energies))
    p_energy_discrete = weighted / total
    assert_close("B2 discrete normalization", p_energy_discrete, 2.2 / 2.8, 2e-15)

    # Charge-aware two-sector toy: the second input column has baryon weight x.
    # U conserves total energy in the degenerate sector and B-L, not baryon number.
    unitary = (
        (math.sqrt(1.0 - fixed), math.sqrt(fixed)),
        (-math.sqrt(fixed), math.sqrt(1.0 - fixed)),
    )
    column_norm = sum(unitary[row][1] ** 2 for row in range(2))
    baryon_weight = unitary[0][1] ** 2
    assert_close("B2 unitary column norm", column_norm, 1.0, 2e-15)
    assert_close("B2 unitary baryon weight", baryon_weight, fixed, 2e-15)

    # Exact finite-xi quartic inflation benchmark.
    n_star = float(inputs["inflation_n_star"])
    scalar_amplitude = float(inputs["scalar_amplitude"])
    xi = alpha_s ** (1.0 / 3.0)
    c = xi * (1.0 + 6.0 * xi)

    def kinetic(x: float) -> float:
        return (1.0 + c * x * x) / (1.0 + xi * x * x) ** 2

    def log_u_prime(x: float) -> float:
        return 4.0 / (x * (1.0 + xi * x * x))

    def epsilon(x: float) -> float:
        return 0.5 * log_u_prime(x) ** 2 / kinetic(x)

    x_end = bisect(lambda x: epsilon(x) - 1.0, 0.1, 5.0)

    def efolds(x: float) -> float:
        return (
            (1.0 + 6.0 * xi) * (x * x - x_end * x_end)
            - 6.0 * math.log((1.0 + xi * x * x) / (1.0 + xi * x_end * x_end))
        ) / 8.0

    x_star = bisect(lambda x: efolds(x) - n_star, x_end, 30.0)
    eps_star = epsilon(x_star)
    b = 1.0 + xi * x_star * x_star
    a = 1.0 + c * x_star * x_star
    l_u = log_u_prime(x_star)
    l_u_prime = -l_u * (1.0 / x_star + 2.0 * xi * x_star / b)
    l_k = 2.0 * c * x_star / a - 4.0 * xi * x_star / b
    eta_star = (l_u_prime + l_u * l_u - 0.5 * l_u * l_k) / kinetic(x_star)
    n_s = 1.0 - 6.0 * eps_star + 2.0 * eta_star
    tensor_ratio = 16.0 * eps_star
    u_shape = x_star**4 / (4.0 * b * b)
    lambda_4 = scalar_amplitude * 24.0 * math.pi**2 * eps_star / u_shape

    assert_close("inflation_xi", xi, expected["inflation_xi"], 2e-15)
    assert_close("inflation_x_end", x_end, expected["inflation_x_end"], 2e-13)
    assert_close("inflation_x_star", x_star, expected["inflation_x_star"], 2e-12)
    assert_close("inflation_epsilon_star", eps_star, expected["inflation_epsilon_star"], 2e-15)
    assert_close("inflation_n_s", n_s, expected["inflation_n_s"], 2e-13)
    assert_close("inflation_r", tensor_ratio, expected["inflation_r"], 3e-14)
    assert_close("inflation_lambda_4", lambda_4, expected["inflation_lambda_4"], 2e-22)

    print("NUMERIC CONSISTENCY: PASS")
    print(
        "Track A: "
        f"sA2={s_a2:.10f}, delta={delta_n:.10f}, D={depth:.10f}, "
        f"x={fixed:.10f}, R={ratio:.10f}"
    )
    print(
        "Density: "
        f"(b,dm,de)=({omega_b:.10f},{omega_dm:.10f},{omega_de:.10f}), "
        f"eta_b={eta_b_density:.8e}, m_light={light_mass:.8f} MeV"
    )
    print(
        "H0 readout diagnostic: "
        f"low={horizon_h0_low:.6f}, high={horizon_h0_high:.6f} "
        "km s^-1 Mpc^-1"
    )
    print(
        "Fixed-background covariance snapshot: "
        f"external-rd chi2/dof={external_chi2:.8f}/{external_dof}, "
        f"p={external_p:.8e}; EH-rd chi2/dof={eh_chi2:.8f}/{eh_dof}, "
        f"p={eh_p:.8e}; REJECT"
    )
    print(
        "Portal rejection diagnostic: "
        f"Gamma_inv={portal_width_mev:.6f} MeV, BR_inv={portal_br:.8f} "
        f"> {portal_br_limit:.3f} (95% CL direct limit)"
    )
    print(
        "Scalar diagnostic: "
        f"I_mu={scalar_i_mu:.9f}, I_mu/(3/2)={scalar_ratio:.9f}, "
        f"finite={finite_scalar_same_coupling * 1e11:.5f}e-11"
    )
    print(
        "Inflation: "
        f"xi={xi:.10f}, ns={n_s:.8f}, r={tensor_ratio:.8f}, lambda4={lambda_4:.6e}"
    )


if __name__ == "__main__":
    main()
