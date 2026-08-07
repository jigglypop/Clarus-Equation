"""Executable audit for ``docs/2_경로적분과_응용``.

This module checks algebraic consequences and explicit counterexamples used by
the chapter.  Passing it does *not* validate any stress-energy, RG-matching, or
observational bridge.  Those claims require independent actions and likelihoods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


ALPHA_S_MZ = 0.1180
ALPHA_S_SIGMA = 0.0009
ALPHA_EM_MZ = 1.0 / 127.95
FINITE_XI = 0.4904868132
FINITE_XI_PHI_STAR = 11.0974588093
CODATA_INVERSE_ALPHA = 137.035999177
CODATA_INVERSE_ALPHA_SIGMA = 0.000000021
CODATA_PROTON_ELECTRON_RATIO = 1836.152673426
CODATA_PROTON_ELECTRON_RATIO_SIGMA = 0.000000032


def _bisect(function, low: float, high: float, *, steps: int = 240) -> float:
    f_low = function(low)
    f_high = function(high)
    if f_low * f_high >= 0.0:
        raise ValueError("the interval must bracket a root")
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


def low_poisson_extinction(mean_offspring: float) -> float:
    """Return the minimal Poisson Galton--Watson extinction probability."""

    if not math.isfinite(mean_offspring) or mean_offspring < 0.0:
        raise ValueError("mean_offspring must be finite and nonnegative")
    if mean_offspring <= 1.0:
        return 1.0
    critical_excess = mean_offspring - 1.0
    if critical_excess <= 1e-8:
        # Dividing log(1-p)+mp=0 by p removes its ever-present p=0 root:
        #   (m-1) - p/2 - p^2/3 - ... = 0.
        # The short series retains the nontrivial root even when m is the next
        # representable float above one and a direct residual rounds to zero.
        survival = 2.0 * critical_excess
        for _ in range(8):
            p2 = survival * survival
            p3 = p2 * survival
            p4 = p3 * survival
            residual = (
                critical_excess
                - survival / 2.0
                - p2 / 3.0
                - p3 / 4.0
                - p4 / 5.0
            )
            derivative = (
                -0.5
                - 2.0 * survival / 3.0
                - 3.0 * p2 / 4.0
                - 4.0 * p3 / 5.0
            )
            updated = survival - residual / derivative
            if updated == survival:
                break
            survival = updated
        return 1.0 - survival
    if mean_offspring >= 32.0:
        # In the strongly supercritical regime the minimal root is close to
        # exp(-m).  Iteration from zero converges to that minimal fixed point
        # without an absolute-width bisection floor; for m > ~745, returning
        # zero is the correct IEEE-754 underflow representation.
        extinction = math.exp(-mean_offspring)
        for _ in range(32):
            updated = math.exp(-mean_offspring * (1.0 - extinction))
            if updated == extinction:
                break
            extinction = updated
        return extinction

    # Near criticality q is too close to one for a direct residual to be
    # well-conditioned.  Solve for survival p=1-q using log1p instead.
    def survival_residual(survival: float) -> float:
        return math.log1p(-survival) + mean_offspring * survival

    low = (mean_offspring - 1.0) / mean_offspring**2
    high = 1.0 - math.exp(-mean_offspring)
    survival = _bisect(survival_residual, low, high)
    return 1.0 - survival


def canonical_chain(alpha_s: float) -> tuple[float, float, float, float, float, float]:
    """Evaluate the registered algebraic chain, without promoting its bridges."""

    domain_limit = 4.0 ** (-3.0 / 4.0)
    if not math.isfinite(alpha_s) or not 0.0 < alpha_s < domain_limit:
        raise ValueError("alpha_s is outside the registered smooth domain")
    mixing_candidate = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = mixing_candidate * (1.0 - mixing_candidate)
    depth = 3.0 + delta
    extinction = low_poisson_extinction(depth)
    dark_ratio = alpha_s * depth * (1.0 + extinction * delta)
    omega_cdm = (1.0 - extinction) * dark_ratio / (1.0 + dark_ratio)
    omega_de = (1.0 - extinction) / (1.0 + dark_ratio)
    return mixing_candidate, delta, depth, extinction, omega_cdm, omega_de


def two_channel_path_kernel(
    path_measure: dict[str, float],
    *,
    effective_depth: float,
    fixed_point: float,
) -> dict[str, tuple[float, float]]:
    """Preserve every finite path in survival and suppressed channels.

    This is the finite-space realization of Theorem 15.2.  The first component
    is the surviving mass and the second the suppressed mass.  It is a concrete
    model, not a claim that every quantum path integral is a positive measure.
    """

    if not path_measure or any(
        not math.isfinite(weight) or weight < 0.0 for weight in path_measure.values()
    ):
        raise ValueError("path_measure must have finite nonnegative weights")
    total_mass = sum(path_measure.values())
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("path_measure must have finite positive total mass")
    if (
        not math.isfinite(effective_depth)
        or effective_depth < 0.0
        or not math.isfinite(fixed_point)
        or not 0.0 <= fixed_point <= 1.0
    ):
        raise ValueError("effective_depth and fixed_point are outside their domains")
    survival_probability = math.exp(-effective_depth * (1.0 - fixed_point))
    return {
        path: (
            weight * survival_probability,
            weight * (1.0 - survival_probability),
        )
        for path, weight in path_measure.items()
    }


def regularized_fixed_point_potential(depth: float, survival: float) -> float:
    """The D=0-continuous potential from Theorem 15.8."""

    if (
        not math.isfinite(depth)
        or depth < 0.0
        or not math.isfinite(survival)
        or not 0.0 <= survival <= 1.0
    ):
        raise ValueError("depth and survival are outside their domains")
    if depth == 0.0:
        return 0.5 * survival**2 + 1.0 - survival
    return 0.5 * survival**2 - math.expm1(-depth * (1.0 - survival)) / depth


def regularized_potential_gradient(depth: float, survival: float) -> float:
    """Derivative in survival; its zeros are exactly the fixed points."""

    if (
        not math.isfinite(depth)
        or depth < 0.0
        or not math.isfinite(survival)
        or not 0.0 <= survival <= 1.0
    ):
        raise ValueError("depth and survival are outside their domains")
    return survival - math.exp(-depth * (1.0 - survival))


def critical_raw_depth(kappa: float) -> float:
    """Return the raw-depth bifurcation point from the invariant kappa*D=1."""

    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    return 1.0 / kappa


def track_b_positive_roots(alpha_em: float) -> tuple[float, ...]:
    """Classify all positive roots of the Track-B convex boundary equation."""

    if not math.isfinite(alpha_em) or alpha_em <= 0.0:
        raise ValueError("alpha_em must be finite and positive")

    def residual(alpha_s: float) -> float:
        return (
            alpha_s
            + alpha_em / (4.0 * alpha_s ** (4.0 / 3.0))
            + alpha_em
            - 1.0 / (2.0 * math.pi)
        )

    minimum = (alpha_em / 3.0) ** (3.0 / 7.0)
    minimum_residual = residual(minimum)
    if minimum_residual > 0.0:
        return ()
    if minimum_residual == 0.0:
        return (minimum,)

    left = 0.5 * minimum
    for _ in range(256):
        if residual(left) > 0.0:
            break
        left *= 0.5
    else:
        raise RuntimeError("failed to bracket the lower Track-B root")

    right = 2.0 * minimum
    for _ in range(256):
        if residual(right) > 0.0:
            break
        right *= 2.0
    else:
        raise RuntimeError("failed to bracket the upper Track-B root")

    return (
        _bisect(residual, left, minimum),
        _bisect(residual, minimum, right),
    )


@dataclass(frozen=True)
class Chapter2Audit:
    hodge_bivector_vector_dimensions: tuple[int, ...]
    poisson_q_at_two: float
    poisson_survival_at_two: float
    reducible_extinction_vector: tuple[float, float]
    kappa_half_critical_depth: float
    mixing_probability_domain_limit: float
    track_b_minimum_residual: float
    track_b_positive_roots: tuple[float, ...]
    canonical_x: float
    canonical_x_endpoint_half_span: float
    canonical_omega_cdm: float
    canonical_omega_de: float
    two_channel_total_mass_residual: float
    two_channel_self_consistency_residual: float
    two_channel_paths_preserved: bool
    two_channel_surviving_fraction: float
    conditional_flat_energy_identity_residual: float
    regularized_d0_potential: float
    regularized_fixed_point_gradient: float
    sample_lyapunov_rate: float
    ger_response_reparameterization_residual: float
    ger_complement_log_action_residual: float
    ger_weight_reparameterization_residual: float
    koide_geometry_residual: float
    large_xi_tensor_ratio: float
    finite_xi_tensor_ratio: float
    tensor_ratio_relative_gap: float
    portal_loop_scale: float
    portal_to_quartic_ratio: float
    dimension_six_to_quartic_per_c6: float
    einstein_slope_ratio_per_c6: float
    c6_unit_slope_tolerance_bound: float
    inverse_alpha_candidate: float
    inverse_alpha_pull_sigma: float
    proton_electron_candidate: float
    proton_electron_pull_sigma: float


def build_audit() -> Chapter2Audit:
    q_two = low_poisson_extinction(2.0)
    track_b_minimum = (ALPHA_EM_MZ / 3.0) ** (3.0 / 7.0)
    track_b_minimum_residual = (
        7.0 * track_b_minimum / 4.0
        + ALPHA_EM_MZ
        - 1.0 / (2.0 * math.pi)
    )
    track_b_roots = track_b_positive_roots(ALPHA_EM_MZ)

    # A = diag(2, 1/2) has rho(A)=2 but its second type cannot reach a
    # supercritical strongly connected component, hence q_2=1.
    reducible = (q_two, low_poisson_extinction(0.5))

    _, _, depth, canonical_x, omega_cdm, omega_de = canonical_chain(ALPHA_S_MZ)
    lower_x = canonical_chain(ALPHA_S_MZ - ALPHA_S_SIGMA)[3]
    upper_x = canonical_chain(ALPHA_S_MZ + ALPHA_S_SIGMA)[3]
    x_endpoint_half_span = 0.5 * abs(upper_x - lower_x)

    n_e = 1.5 * depth * 12.0
    large_xi_r = 12.0 / n_e**2
    finite_xi_epsilon = 8.0 / (
        FINITE_XI_PHI_STAR**2
        * (
            1.0
            + FINITE_XI
            * (1.0 + 6.0 * FINITE_XI)
            * FINITE_XI_PHI_STAR**2
        )
    )
    finite_xi_r = 16.0 * finite_xi_epsilon
    relative_gap = abs(finite_xi_r - large_xi_r) / finite_xi_r

    delta = 4.0 * ALPHA_S_MZ ** (4.0 / 3.0)
    delta *= 1.0 - delta
    portal = delta**2
    portal_loop = portal**2 / (16.0 * math.pi**2)
    quartic = 1.3434991214e-10

    phi_star = FINITE_XI_PHI_STAR
    dimension_six_ratio = 4.0 * phi_star**2 / quartic
    einstein_slope_ratio = (
        2.0
        * phi_star**2
        * (3.0 + FINITE_XI * phi_star**2)
        / quartic
    )
    c6_slope_bound = 1.0 / einstein_slope_ratio

    inverse_alpha_candidate = 4.0 * math.pi**3 + math.pi**2 + math.pi
    proton_electron_candidate = 6.0 * math.pi**5

    input_measure = {"gamma_0": 0.2, "gamma_1": 0.3, "gamma_2": 0.5}
    output_measure = two_channel_path_kernel(
        input_measure,
        effective_depth=depth,
        fixed_point=canonical_x,
    )
    output_total = sum(surviving + suppressed for surviving, suppressed in output_measure.values())
    output_surviving = sum(surviving for surviving, _ in output_measure.values())
    input_total = sum(input_measure.values())
    sample_x = 0.2
    sample_gradient = regularized_potential_gradient(depth, sample_x)
    sample_mobility = sample_x**2 * (1.0 - sample_x) ** 2

    susceptibility_tau = canonical_x * (1.0 - canonical_x)
    delta_raw_depth = 0.01
    kappa = 1.7
    response_in_depth = kappa * susceptibility_tau * delta_raw_depth
    # Reparameterize with D_tilde=2D.  The derivative gets a factor 1/2 and
    # the differential a factor 2, so the contracted response is unchanged.
    response_in_tilde_depth = (kappa * susceptibility_tau / 2.0) * (
        2.0 * delta_raw_depth
    )

    sigma = 1.0 - canonical_x
    complement_log_action = -math.log(sigma)
    raw_depth = depth / kappa
    operational_tau = kappa * raw_depth
    internal_action = (
        operational_tau / (operational_tau + 1.0) * complement_log_action
    )
    weighted_factor = math.exp(-internal_action)
    rescaled_raw_depth = 2.0 * raw_depth
    rescaled_kappa = kappa / 2.0
    rescaled_tau = rescaled_kappa * rescaled_raw_depth
    rescaled_weighted_factor = sigma ** (rescaled_tau / (rescaled_tau + 1.0))

    # This is a separate conditional algebra check of Theorem 15.6.  It does
    # not identify the path-kernel output with stress energy.
    assumed_x_e = canonical_x
    rho_total = 7.0
    rho_critical = rho_total
    rho_baryon = assumed_x_e * rho_total
    conditional_energy_residual = rho_baryon / rho_critical - assumed_x_e

    masses = (0.51099895, 105.6583755, 1776.86)
    roots = tuple(math.sqrt(value) for value in masses)
    q_koide = sum(masses) / sum(roots) ** 2
    cos2 = sum(roots) ** 2 / (3.0 * sum(value * value for value in roots))

    return Chapter2Audit(
        hodge_bivector_vector_dimensions=tuple(
            dimension
            for dimension in range(1, 17)
            if dimension * (dimension - 1) // 2 == dimension
        ),
        poisson_q_at_two=q_two,
        poisson_survival_at_two=1.0 - q_two,
        reducible_extinction_vector=reducible,
        kappa_half_critical_depth=critical_raw_depth(0.5),
        mixing_probability_domain_limit=4.0 ** (-3.0 / 4.0),
        track_b_minimum_residual=track_b_minimum_residual,
        track_b_positive_roots=track_b_roots,
        canonical_x=canonical_x,
        canonical_x_endpoint_half_span=x_endpoint_half_span,
        canonical_omega_cdm=omega_cdm,
        canonical_omega_de=omega_de,
        two_channel_total_mass_residual=output_total - input_total,
        two_channel_self_consistency_residual=(
            output_surviving / input_total - canonical_x
        ),
        two_channel_paths_preserved=set(output_measure) == set(input_measure),
        two_channel_surviving_fraction=output_surviving / input_total,
        conditional_flat_energy_identity_residual=conditional_energy_residual,
        regularized_d0_potential=regularized_fixed_point_potential(0.0, 1.0),
        regularized_fixed_point_gradient=regularized_potential_gradient(depth, canonical_x),
        sample_lyapunov_rate=-sample_mobility * sample_gradient**2,
        ger_response_reparameterization_residual=(
            response_in_depth - response_in_tilde_depth
        ),
        ger_complement_log_action_residual=(
            weighted_factor - sigma ** (operational_tau / (operational_tau + 1.0))
        ),
        ger_weight_reparameterization_residual=(
            weighted_factor - rescaled_weighted_factor
        ),
        koide_geometry_residual=q_koide - 1.0 / (3.0 * cos2),
        large_xi_tensor_ratio=large_xi_r,
        finite_xi_tensor_ratio=finite_xi_r,
        tensor_ratio_relative_gap=relative_gap,
        portal_loop_scale=portal_loop,
        portal_to_quartic_ratio=portal_loop / quartic,
        dimension_six_to_quartic_per_c6=dimension_six_ratio,
        einstein_slope_ratio_per_c6=einstein_slope_ratio,
        c6_unit_slope_tolerance_bound=c6_slope_bound,
        inverse_alpha_candidate=inverse_alpha_candidate,
        inverse_alpha_pull_sigma=(inverse_alpha_candidate - CODATA_INVERSE_ALPHA)
        / CODATA_INVERSE_ALPHA_SIGMA,
        proton_electron_candidate=proton_electron_candidate,
        proton_electron_pull_sigma=(
            proton_electron_candidate - CODATA_PROTON_ELECTRON_RATIO
        )
        / CODATA_PROTON_ELECTRON_RATIO_SIGMA,
    )


def validate(audit: Chapter2Audit) -> None:
    assert math.isclose(audit.poisson_q_at_two, 0.20318786998, abs_tol=5e-12)
    assert audit.poisson_survival_at_two > audit.poisson_q_at_two
    assert audit.reducible_extinction_vector[1] == 1.0
    assert audit.kappa_half_critical_depth == 2.0
    assert audit.track_b_minimum_residual < 0.0
    assert len(audit.track_b_positive_roots) == 2
    assert math.isclose(
        audit.track_b_positive_roots[0], 0.0528678687, abs_tol=5e-11
    )
    assert math.isclose(
        audit.track_b_positive_roots[1], 0.1173186647, abs_tol=5e-11
    )
    assert audit.track_b_positive_roots[1] < audit.mixing_probability_domain_limit
    assert math.isclose(audit.canonical_x, 0.04863825851598631, abs_tol=1e-14)
    assert audit.canonical_x_endpoint_half_span > 1e-5
    assert abs(audit.two_channel_total_mass_residual) < 1e-15
    assert abs(audit.two_channel_self_consistency_residual) < 1e-15
    assert audit.two_channel_paths_preserved
    assert math.isclose(
        audit.two_channel_surviving_fraction, audit.canonical_x, abs_tol=1e-15
    )
    assert abs(audit.conditional_flat_energy_identity_residual) < 1e-15
    assert math.isfinite(audit.regularized_d0_potential)
    assert abs(audit.regularized_fixed_point_gradient) < 1e-15
    assert audit.sample_lyapunov_rate < 0.0
    assert audit.hodge_bivector_vector_dimensions == (3,)
    assert abs(audit.ger_response_reparameterization_residual) < 1e-15
    assert abs(audit.ger_complement_log_action_residual) < 1e-15
    assert abs(audit.ger_weight_reparameterization_residual) < 1e-15
    assert abs(audit.koide_geometry_residual) < 1e-15
    assert audit.tensor_ratio_relative_gap > 0.1
    assert audit.portal_to_quartic_ratio > 10_000.0
    assert audit.dimension_six_to_quartic_per_c6 > 1e12
    assert audit.einstein_slope_ratio_per_c6 > 1e14
    assert 8e-15 < audit.c6_unit_slope_tolerance_bound < 9e-15
    assert abs(audit.inverse_alpha_pull_sigma) > 10_000.0
    assert abs(audit.proton_electron_pull_sigma) > 1_000_000.0


def main() -> None:
    audit = build_audit()
    validate(audit)
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
    print("scope: algebra_and_counterexamples_only")
    print("physical_bridges_validated: false")


if __name__ == "__main__":
    main()
