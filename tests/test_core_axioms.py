from __future__ import annotations

from functools import partial
import math

from reality_stone.clarus.core_axioms import (
    ElectroweakCoherence,
    additive_effective_depth,
    bootstrap_fixed_points,
    bootstrap_map,
    bootstrap_orbit,
    bivector_vector_closure_dimensions,
    bootstrap_residual,
    bootstrap_stability_multiplier,
    complement_feedback,
    composition_residual,
    effective_depth_from_fixed_point,
    electroweak_effective_depth,
    exponential_survival,
    hodge_target_dimension,
    low_bootstrap_fixed_point,
    low_branch_depth_sensitivity,
    low_branch_rate_sensitivity,
    mixture_affinity_residual,
    power_survival,
    powered_feedback,
    poisson_probability,
    stretched_exponential_survival,
    thinned_trigger_mean,
    weak_mixing_from_fixed_point,
    zero_trigger_survival,
)


DEPTH_GRID = (0.0, 0.2, 0.7, 1.4, 3.0)
MIXTURE_GRID = (0.0, 0.1, 0.5, 0.9, 1.0)
WEIGHT_GRID = (0.0, 0.2, 0.5, 0.8, 1.0)


def test_exponential_is_selected_by_exact_composition() -> None:
    assert composition_residual(exponential_survival, DEPTH_GRID) < 1e-14
    assert (
        composition_residual(
            partial(stretched_exponential_survival, power=2.0),
            DEPTH_GRID,
        )
        > 1e-3
    )
    assert composition_residual(power_survival, DEPTH_GRID) > 1e-3


def test_poisson_zero_trigger_model_fixes_optical_depth_normalization() -> None:
    depth = 1.7
    survival = 0.2
    expected = depth * (1.0 - survival)

    assert math.isclose(zero_trigger_survival(expected), math.exp(-expected))
    assert math.isclose(thinned_trigger_mean(survival, depth), expected)
    assert math.isclose(bootstrap_map(survival, depth), math.exp(-expected))
    assert math.isclose(
        sum(poisson_probability(count, expected) for count in range(32)),
        1.0,
        abs_tol=1e-14,
    )


def test_complement_is_selected_by_mixture_affinity() -> None:
    assert (
        mixture_affinity_residual(
            complement_feedback,
            MIXTURE_GRID,
            WEIGHT_GRID,
        )
        < 1e-15
    )
    assert (
        mixture_affinity_residual(
            partial(powered_feedback, power=2.0),
            MIXTURE_GRID,
            WEIGHT_GRID,
        )
        > 1e-3
    )


def test_electroweak_coherence_is_normalized_off_diagonal_intensity() -> None:
    mixing = ElectroweakCoherence(g=0.65, g_prime=0.36)
    matrix = mixing.normalized_mass_matrix

    assert math.isclose(matrix[0][0] + matrix[1][1], 1.0)
    assert math.isclose(
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(matrix[0][1] ** 2, mixing.intensity)
    assert math.isclose(
        mixing.intensity,
        mixing.sin2_theta * mixing.cos2_theta,
    )

    swapped = ElectroweakCoherence(g=0.36, g_prime=0.65)
    assert math.isclose(mixing.intensity, swapped.intensity)


def test_effective_depth_is_direct_sum_additive() -> None:
    assert additive_effective_depth(3, ()) == 3.0
    assert additive_effective_depth(3, (0.1, 0.2)) == 3.3
    assert additive_effective_depth(3, (0.2, 0.1)) == 3.3

    mixing = ElectroweakCoherence(g=0.65, g_prime=0.36)
    assert math.isclose(
        electroweak_effective_depth(3, g=0.65, g_prime=0.36),
        3.0 + mixing.intensity,
    )


def test_hodge_degree_and_minimal_recursive_type_closure_select_d3() -> None:
    assert hodge_target_dimension(2, 1) == 3
    assert hodge_target_dimension(2, 2) == 4
    assert bivector_vector_closure_dimensions(max_dimension=12) == (3,)


def test_bootstrap_has_stable_low_and_unstable_identity_branches() -> None:
    depth = 3.17776
    low = low_bootstrap_fixed_point(depth)

    assert math.isclose(low, 0.0486466333, rel_tol=0.0, abs_tol=1e-10)
    assert abs(bootstrap_residual(low, depth)) < 1e-13
    assert abs(bootstrap_residual(1.0, depth)) < 1e-15
    assert bootstrap_stability_multiplier(low, depth) < 1.0
    assert bootstrap_stability_multiplier(1.0, depth) > 1.0


def test_bootstrap_root_count_changes_only_at_unit_effective_depth() -> None:
    assert bootstrap_fixed_points(0.75) == (1.0,)
    assert bootstrap_fixed_points(1.0) == (1.0,)

    roots = bootstrap_fixed_points(3.17776)
    assert len(roots) == 2
    assert roots[0] < 1.0 / 3.17776
    assert roots[1] == 1.0
    assert all(abs(bootstrap_residual(root, 3.17776)) < 1e-13 for root in roots)


def test_zero_trigger_dynamics_selects_low_branch_without_observation() -> None:
    depth = 3.17776
    low = low_bootstrap_fixed_point(depth)

    for initial in (0.0, 0.01, 0.2, 0.5, 0.9, 0.999):
        orbit = bootstrap_orbit(initial, depth, iterations=100)
        assert math.isclose(orbit[-1], low, abs_tol=1e-12)
        if initial < low:
            assert all(left <= right for left, right in zip(orbit, orbit[1:]))
        else:
            assert all(left >= right for left, right in zip(orbit, orbit[1:]))

    assert bootstrap_orbit(1.0, depth, iterations=8) == (1.0,) * 9


def test_low_branch_sensitivity_matches_finite_difference() -> None:
    depth = 3.17776
    low = low_bootstrap_fixed_point(depth)
    step = 1e-5
    numerical = (
        low_bootstrap_fixed_point(depth + step)
        - low_bootstrap_fixed_point(depth - step)
    ) / (2.0 * step)

    assert math.isclose(
        low_branch_depth_sensitivity(low, depth),
        numerical,
        rel_tol=1e-8,
        abs_tol=1e-10,
    )


def test_low_branch_rate_sensitivity_and_robustness_sweep() -> None:
    depth = 3.17776
    rate = 1.0
    low = low_bootstrap_fixed_point(depth, rate=rate)
    step = 1e-5
    numerical = (
        low_bootstrap_fixed_point(depth, rate=rate + step)
        - low_bootstrap_fixed_point(depth, rate=rate - step)
    ) / (2.0 * step)

    assert math.isclose(
        low_branch_rate_sensitivity(low, depth, rate=rate),
        numerical,
        rel_tol=1e-8,
        abs_tol=1e-10,
    )
    assert math.isclose(
        low_bootstrap_fixed_point(depth, rate=0.95),
        0.0582492,
        abs_tol=1e-7,
    )
    assert math.isclose(
        low_bootstrap_fixed_point(depth, rate=1.05),
        0.0407321,
        abs_tol=1e-7,
    )


def test_conditional_chain_is_invertible_without_observable_identification() -> None:
    sin2_theta = 0.23122
    g = math.sqrt(1.0 - sin2_theta)
    g_prime = math.sqrt(sin2_theta)
    depth = electroweak_effective_depth(3, g=g, g_prime=g_prime)
    low = low_bootstrap_fixed_point(depth)

    assert math.isclose(effective_depth_from_fixed_point(low), depth, abs_tol=1e-12)
    assert math.isclose(
        weak_mixing_from_fixed_point(low),
        sin2_theta,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
