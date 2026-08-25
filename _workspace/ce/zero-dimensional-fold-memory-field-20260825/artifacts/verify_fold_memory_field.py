"""Deterministic checks for the zero-dimensional fold memory-field model.

Standard-library only.  The script checks algebraic identities and declared
effective-model properties; it does not validate a quantum-gravity ontology.
"""

from __future__ import annotations

import json
import math


TOL = 1.0e-12
FD_TOL = 1.0e-8


def add_dim(*vectors: tuple[int, int]) -> tuple[int, int]:
    return tuple(sum(values) for values in zip(*vectors, strict=True))


def rhs(
    psi: float,
    *,
    amplitude: float,
    tau: float,
    beta: float,
    psi_s: float,
    lambda_0: float,
) -> float:
    return (
        -psi / tau
        + amplitude * lambda_0
        + amplitude * beta * psi / (1.0 + psi / psi_s)
    )


def jacobian(
    psi: float,
    *,
    amplitude: float,
    tau: float,
    beta: float,
    psi_s: float,
) -> float:
    return -1.0 / tau + amplitude * beta / (1.0 + psi / psi_s) ** 2


def positive_fixed_point(
    *,
    reproduction: float,
    eta: float,
    psi_s: float,
) -> float:
    discriminant = (reproduction + eta - 1.0) ** 2 + 4.0 * eta
    x_plus = (
        reproduction + eta - 1.0 + math.sqrt(discriminant)
    ) / 2.0
    return psi_s * x_plus


def poisson_extinction(reproduction: float) -> tuple[float, int]:
    if reproduction <= 1.0:
        return 1.0, 0
    q = 0.0
    for iteration in range(1, 100_001):
        updated = math.exp(reproduction * (q - 1.0))
        if abs(updated - q) <= 1.0e-15:
            return updated, iteration
        q = updated
    raise RuntimeError("Poisson extinction iteration did not converge")


def simpson_integral(function, lower: float, upper: float, panels: int) -> float:
    if panels % 2:
        raise ValueError("Simpson panels must be even")
    step = (upper - lower) / panels
    total = function(lower) + function(upper)
    total += 4.0 * sum(
        function(lower + index * step)
        for index in range(1, panels, 2)
    )
    total += 2.0 * sum(
        function(lower + index * step)
        for index in range(2, panels, 2)
    )
    return total * step / 3.0


def causal_kernel_witness(
    time: float,
    radius: float,
    *,
    ell: float,
    speed: float,
    tau: float,
) -> float:
    delay = ell / speed
    if time < delay or radius > speed * time:
        return 0.0
    return (
        math.exp(-(time - delay) / tau)
        * 3.0
        / (4.0 * math.pi * (speed * time) ** 3)
    )


def saturating_readout(value: float) -> float:
    return value / (1.0 + value)


def persistent_carrier_rhs(
    state: list[float],
    *,
    amplitude: float,
    weights: list[list[float]],
    tau: float,
    baseline: list[float],
) -> list[float]:
    return [
        (
            -(state[index] - baseline[index])
            + amplitude
            * sum(
                weight * saturating_readout(state[source])
                for source, weight in enumerate(weights[index])
            )
        )
        / tau
        for index in range(len(state))
    ]


def nearest_neighbor_ring(size: int) -> list[list[float]]:
    weights = [[0.0 for _ in range(size)] for _ in range(size)]
    for index in range(size):
        weights[index][(index - 1) % size] = 0.5
        weights[index][(index + 1) % size] = 0.5
    return weights


def main() -> None:
    dimless = (0, 0)
    dim_a = dimless
    dim_beta = (0, -1)
    dim_tau = (0, 1)
    dim_kernel = (-3, 0)
    dim_volume4 = (3, 1)
    dim_psi = (-3, 0)
    dim_intensity = (-3, -1)
    dim_carrier_kernel = (0, -1)
    dim_time = (0, 1)

    reproduction_dimension = add_dim(
        dim_a, dim_beta, dim_kernel, dim_volume4
    )
    intensity_feedback_dimension = add_dim(dim_beta, dim_psi)
    persistent_carrier_integral_dimension = add_dim(
        dim_a, dim_carrier_kernel, dim_time
    )

    amplitude = 1.0
    tau = 2.5
    psi_s = 1.0
    lambda_0 = 0.0

    cases: list[dict[str, float | bool | None]] = []
    for reproduction in (0.8, 1.0, 1.2):
        beta = reproduction / (amplitude * tau)
        psi_star = (
            psi_s * (reproduction - 1.0)
            if reproduction > 1.0
            else 0.0
        )
        analytic_j = jacobian(
            psi_star,
            amplitude=amplitude,
            tau=tau,
            beta=beta,
            psi_s=psi_s,
        )
        step = 1.0e-6
        finite_j = (
            rhs(
                psi_star + step,
                amplitude=amplitude,
                tau=tau,
                beta=beta,
                psi_s=psi_s,
                lambda_0=lambda_0,
            )
            - rhs(
                psi_star - step,
                amplitude=amplitude,
                tau=tau,
                beta=beta,
                psi_s=psi_s,
                lambda_0=lambda_0,
            )
        ) / (2.0 * step)
        cases.append(
            {
                "reproduction": reproduction,
                "psi_star": psi_star,
                "rhs_residual": abs(
                    rhs(
                        psi_star,
                        amplitude=amplitude,
                        tau=tau,
                        beta=beta,
                        psi_s=psi_s,
                        lambda_0=lambda_0,
                    )
                ),
                "analytic_jacobian": analytic_j,
                "finite_difference_jacobian": finite_j,
                "jacobian_error": abs(analytic_j - finite_j),
                "positive_fixed_point_stable": (
                    analytic_j < 0.0 if reproduction > 1.0 else None
                ),
            }
        )

    reproduction = 1.2
    beta = reproduction / (amplitude * tau)
    lambda_0 = 0.1
    eta = amplitude * lambda_0 * tau / psi_s
    positive = positive_fixed_point(
        reproduction=reproduction,
        eta=eta,
        psi_s=psi_s,
    )
    positive_residual = abs(
        rhs(
            positive,
            amplitude=amplitude,
            tau=tau,
            beta=beta,
            psi_s=psi_s,
            lambda_0=lambda_0,
        )
    )
    positive_jacobian = jacobian(
        positive,
        amplitude=amplitude,
        tau=tau,
        beta=beta,
        psi_s=psi_s,
    )

    carrier_size = 8
    carrier_weights = nearest_neighbor_ring(carrier_size)
    carrier_row_sums = [sum(row) for row in carrier_weights]
    carrier_row_sum = carrier_row_sums[0]
    carrier_amplitude = 1.2
    carrier_bootstrap_gain = carrier_amplitude * carrier_row_sum
    carrier_positive_state = carrier_bootstrap_gain - 1.0
    carrier_vector = [carrier_positive_state] * carrier_size
    carrier_residual = max(
        abs(value)
        for value in persistent_carrier_rhs(
            carrier_vector,
            amplitude=carrier_amplitude,
            weights=carrier_weights,
            tau=tau,
            baseline=[0.0] * carrier_size,
        )
    )
    carrier_weight_eigenvalues = [
        math.cos(2.0 * math.pi * mode / carrier_size)
        for mode in range(carrier_size)
    ]
    carrier_jacobian_eigenvalues = [
        (
            -1.0
            + carrier_amplitude
            * eigenvalue
            / carrier_bootstrap_gain**2
        )
        / tau
        for eigenvalue in carrier_weight_eigenvalues
    ]
    carrier_effective_gain = (
        carrier_amplitude
        * carrier_row_sum
        / carrier_bootstrap_gain**2
    )
    carrier_zero_rhs = persistent_carrier_rhs(
        [0.0] * carrier_size,
        amplitude=carrier_amplitude,
        weights=carrier_weights,
        tau=tau,
        baseline=[0.0] * carrier_size,
    )
    carrier_subcritical_amplitude = 0.8
    carrier_zero_perron_jacobian = (
        -1.0 + carrier_subcritical_amplitude * carrier_row_sum
    ) / tau

    q, q_iterations = poisson_extinction(reproduction)
    q_residual = abs(q - math.exp(reproduction * (q - 1.0)))

    memory_total = 1.0
    memory_saturation = 1.0
    no_event_hazard = (
        beta
        * tau
        * memory_saturation
        * math.log1p(memory_total / memory_saturation)
    )
    no_event_probability = math.exp(-no_event_hazard)

    temporal_integral = simpson_integral(
        lambda time: math.exp(-time / tau),
        0.0,
        50.0 * tau,
        100_000,
    )
    temporal_normalization_residual = abs(temporal_integral - tau)

    ell = 0.2
    propagation_speed = 0.8
    causal_delay = ell / propagation_speed
    causal_time = causal_delay + 0.7
    causal_radius = 0.5 * propagation_speed * causal_time
    causal_inside_value = causal_kernel_witness(
        causal_time,
        causal_radius,
        ell=ell,
        speed=propagation_speed,
        tau=tau,
    )
    causal_outside_value = causal_kernel_witness(
        causal_time,
        1.01 * propagation_speed * causal_time,
        ell=ell,
        speed=propagation_speed,
        tau=tau,
    )
    causal_pre_delay_value = causal_kernel_witness(
        0.99 * causal_delay,
        0.0,
        ell=ell,
        speed=propagation_speed,
        tau=tau,
    )
    causal_spatial_integral = (
        causal_inside_value
        * (4.0 * math.pi * (propagation_speed * causal_time) ** 3)
        / 3.0
    )
    causal_expected_spatial_integral = math.exp(
        -(causal_time - causal_delay) / tau
    )
    causal_spacetime_integral = simpson_integral(
        lambda time: math.exp(-(time - causal_delay) / tau),
        causal_delay,
        causal_delay + 50.0 * tau,
        100_000,
    )

    gaussian_l2 = 1.0 / (8.0 * math.pi ** 1.5 * ell**3)
    gaussian_gradient_l2 = (
        3.0 / (16.0 * math.pi ** 1.5 * ell**5)
    )
    uv_ratio_l2 = (
        1.0 / (8.0 * math.pi ** 1.5 * (ell / 2.0) ** 3)
    ) / gaussian_l2
    uv_ratio_gradient = (
        3.0 / (16.0 * math.pi ** 1.5 * (ell / 2.0) ** 5)
    ) / gaussian_gradient_l2

    derrick_t = 3.0
    derrick_u = -derrick_t / 3.0
    derrick_first = -derrick_t - 3.0 * derrick_u
    derrick_second = 2.0 * derrick_t + 12.0 * derrick_u

    checks = {
        "reproduction_number_is_dimensionless": (
            reproduction_dimension == dimless
        ),
        "persistent_carrier_integral_is_dimensionless": (
            persistent_carrier_integral_dimension == dimless
        ),
        "feedback_matches_intensity_dimension": (
            intensity_feedback_dimension == dim_intensity
        ),
        "persistent_carrier_ring_has_constant_row_sum": all(
            math.isclose(
                row_sum,
                carrier_row_sum,
                rel_tol=0.0,
                abs_tol=TOL,
            )
            for row_sum in carrier_row_sums
        ),
        "persistent_carrier_has_no_self_loops": all(
            carrier_weights[index][index] == 0.0
            for index in range(carrier_size)
        ),
        "persistent_carrier_positive_branch_residual_small": (
            carrier_residual <= TOL
        ),
        "persistent_carrier_positive_branch_all_modes_stable": (
            max(carrier_jacobian_eigenvalues) < 0.0
        ),
        "persistent_carrier_delayed_stability_bound_holds": (
            carrier_effective_gain < 1.0
        ),
        "persistent_carrier_zero_needs_a_seed": all(
            value == 0.0 for value in carrier_zero_rhs
        ),
        "persistent_carrier_subcritical_zero_is_stable": (
            carrier_zero_perron_jacobian < 0.0
        ),
        "all_fixed_point_residuals_small": all(
            float(case["rhs_residual"]) <= TOL for case in cases
        ),
        "all_finite_difference_jacobians_match": all(
            float(case["jacobian_error"]) <= FD_TOL for case in cases
        ),
        "positive_baseline_root_residual_small": positive_residual <= TOL,
        "positive_baseline_root_is_stable": positive_jacobian < 0.0,
        "poisson_extinction_fixed_point_matches": q_residual <= TOL,
        "finite_seed_has_nonzero_immediate_extinction_route": (
            0.0 < no_event_probability < 1.0
        ),
        "kernel_time_normalization_matches": (
            temporal_normalization_residual <= 1.0e-10
        ),
        "explicit_kernel_is_positive_inside_cone": causal_inside_value > 0.0,
        "explicit_kernel_vanishes_outside_cone": causal_outside_value == 0.0,
        "explicit_kernel_vanishes_before_delay": causal_pre_delay_value == 0.0,
        "explicit_kernel_spatial_normalization_matches": math.isclose(
            causal_spatial_integral,
            causal_expected_spatial_integral,
            rel_tol=0.0,
            abs_tol=TOL,
        ),
        "explicit_kernel_spacetime_normalization_matches": math.isclose(
            causal_spacetime_integral,
            tau,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ),
        "explicit_kernel_speed_is_subluminal": (
            0.0 < propagation_speed <= 1.0
        ),
        "gaussian_l2_has_ell_minus_3_scaling": math.isclose(
            uv_ratio_l2, 8.0, rel_tol=0.0, abs_tol=TOL
        ),
        "gaussian_gradient_has_ell_minus_5_scaling": math.isclose(
            uv_ratio_gradient, 32.0, rel_tol=0.0, abs_tol=TOL
        ),
        "derrick_stationarity_condition_matches": abs(derrick_first) <= TOL,
        "derrick_second_variation_is_negative": derrick_second < 0.0,
    }

    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "dimensions_LT": {
            "reproduction_number": reproduction_dimension,
            "feedback_intensity": intensity_feedback_dimension,
            "persistent_carrier_integral": (
                persistent_carrier_integral_dimension
            ),
        },
        "persistent_carrier_exact_network": {
            "size": carrier_size,
            "row_sums": carrier_row_sums,
            "amplitude": carrier_amplitude,
            "bootstrap_gain": carrier_bootstrap_gain,
            "positive_uniform_state": carrier_positive_state,
            "fixed_point_residual": carrier_residual,
            "weight_eigenvalues": carrier_weight_eigenvalues,
            "jacobian_eigenvalues": carrier_jacobian_eigenvalues,
            "largest_jacobian_eigenvalue": max(
                carrier_jacobian_eigenvalues
            ),
            "delayed_stability_effective_gain": carrier_effective_gain,
            "zero_state_rhs": carrier_zero_rhs,
            "subcritical_zero_perron_jacobian": (
                carrier_zero_perron_jacobian
            ),
        },
        "fixed_point_cases": cases,
        "positive_baseline_case": {
            "reproduction": reproduction,
            "eta": eta,
            "psi_star": positive,
            "rhs_residual": positive_residual,
            "jacobian": positive_jacobian,
        },
        "poisson_supercritical_example": {
            "reproduction": reproduction,
            "extinction_probability": q,
            "survival_probability": 1.0 - q,
            "fixed_point_residual": q_residual,
            "iterations": q_iterations,
        },
        "finite_seed_no_event_certificate": {
            "hazard": no_event_hazard,
            "probability": no_event_probability,
        },
        "kernel_normalization": {
            "numeric_integral": temporal_integral,
            "declared_tau": tau,
            "residual": temporal_normalization_residual,
        },
        "explicit_causal_kernel_witness": {
            "ell": ell,
            "speed_in_c_units": propagation_speed,
            "delay": causal_delay,
            "inside_value": causal_inside_value,
            "outside_value": causal_outside_value,
            "pre_delay_value": causal_pre_delay_value,
            "spatial_integral": causal_spatial_integral,
            "expected_spatial_integral": causal_expected_spatial_integral,
            "spacetime_integral": causal_spacetime_integral,
            "declared_tau": tau,
        },
        "uv_scaling": {
            "ell": ell,
            "gaussian_l2": gaussian_l2,
            "gaussian_gradient_l2": gaussian_gradient_l2,
            "halving_ell_l2_ratio": uv_ratio_l2,
            "halving_ell_gradient_ratio": uv_ratio_gradient,
        },
        "derrick_certificate": {
            "T": derrick_t,
            "U": derrick_u,
            "first_variation": derrick_first,
            "second_variation": derrick_second,
        },
        "tolerances": {
            "identity": TOL,
            "finite_difference_jacobian": FD_TOL,
            "kernel_normalization": 1.0e-10,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
