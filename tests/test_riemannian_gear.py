from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.riemannian_gear import (
    RiemannianGearNetwork,
    compose_candidate_energy,
    fisher_metric,
    generalized_phase_locking,
    gibbs_posterior,
    natural_gradient_step,
    nonselected_residual_field,
    pair_lock_time,
    principal_angle,
    run_demo,
    warped_gaussian_curvature,
)


def _chain() -> RiemannianGearNetwork:
    incidence = np.array([[2, 3, 0], [0, 3, -1]])
    target = np.array([0.2, -0.1, 0.4])
    return RiemannianGearNetwork(
        incidence,
        stiffness=np.array([1.2, 0.8]),
        offset=incidence @ target,
        mass=np.diag([1.0, 1.5, 0.7]),
    )


def test_principal_angle_interval_and_pair_lock_time() -> None:
    wrapped = principal_angle(np.array([-3.0 * np.pi, -np.pi, np.pi, 3.0 * np.pi]))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    time = pair_lock_time(1.1, coupling_rate=3.2, epsilon=0.01)
    exact = 2.0 * math.atan(math.tan(1.1 / 2.0) * math.exp(-3.2 * time))
    assert math.isclose(exact, 0.01, rel_tol=1e-12)


def test_gradient_matches_finite_difference_and_energy_decreases() -> None:
    network = _chain()
    theta = np.array([0.7, 0.2, -0.3])
    epsilon = 1e-7
    numeric = np.array(
        [
            (network.energy(theta + epsilon * np.eye(3)[index])
             - network.energy(theta - epsilon * np.eye(3)[index]))
            / (2.0 * epsilon)
            for index in range(3)
        ]
    )
    assert np.allclose(network.gradient(theta), numeric, atol=1e-8)
    trajectory = network.simulate(theta, duration=3.0, step=0.002)
    assert np.max(np.diff(trajectory.energy)) < 1e-10
    assert trajectory.energy[-1] < trajectory.energy[0] * 1e-6


def test_tangent_drive_preserves_lock_and_gear_ratio() -> None:
    network = _chain()
    locked = np.linalg.pinv(network.incidence) @ network.offset
    tangent = network.tangent_projection(np.array([1.0, 0.0, 0.0]))

    def drive(_time: float, _state: np.ndarray) -> np.ndarray:
        return tangent

    trajectory = network.simulate(locked, duration=1.0, step=0.002, drive=drive, wrap_state=False)
    assert np.linalg.norm(network.incidence @ tangent) < 1e-12
    assert np.max(np.abs(trajectory.residual)) < 1e-12
    assert np.allclose(trajectory.theta[-1] - trajectory.theta[0], tangent, atol=1e-11)


def test_state_metric_is_positive_and_cosine_bounds_hold() -> None:
    network = _chain()
    theta = np.array([0.21, -0.09, 0.38])
    metric = network.state_metric(theta, alpha=np.array([0.5, 0.3]))
    assert float(np.min(np.linalg.eigvalsh(metric))) > 0.0
    lower, cosine, upper = network.cosine_quadratic_bounds(theta, None, radius=0.5)
    assert lower <= cosine <= upper


def test_frustration_consistency_dimension_and_winding_validation() -> None:
    network = _chain()
    assert network.locking_dimension == 1
    assert network.is_lift_consistent()
    assert network.frustration().normal_equation_error < 1e-12
    with pytest.raises(ValueError, match="integers"):
        network.frustration(np.array([0.5, 0.0]))


def test_contraction_iss_and_spectral_truncation_bounds() -> None:
    network = _chain()
    spectrum = network.spectrum()
    eta = 1.0 / float(spectrum.positive_values[-1])
    q = network.contraction_factor(eta)
    assert 0.0 <= q < 1.0
    assert network.iss_bound(1.0, 0.01, eta, 10) >= 0.0
    state = np.array([0.7, -0.2, 0.4])
    full = network.spectral_evolution(state, time=0.5)
    reduced = network.spectral_evolution(state, time=0.5, keep_positive=1)
    error = np.linalg.norm(full - reduced)
    assert error <= network.spectral_truncation_bound(state, time=0.5, keep_positive=1)
    with pytest.raises(ValueError, match="contraction"):
        network.contraction_factor(2.0 / float(spectrum.positive_values[-1]))


def test_phase_locking_statistic_detects_integer_relation() -> None:
    theta = np.linspace(0.0, 20.0, 1000)
    phases = np.column_stack((theta, -2.0 * theta / 3.0 + 0.4))
    magnitude, offset = generalized_phase_locking(phases, np.array([2.0, 3.0]))
    assert magnitude > 1.0 - 1e-12
    assert math.isclose(offset, 1.2, abs_tol=1e-12)


def test_gibbs_residual_fisher_and_natural_gradient() -> None:
    energy = compose_candidate_energy(
        prediction=np.array([0.1, 0.5, 0.8]),
        frustration=np.array([0.0, 0.2, 0.5]),
        complexity=np.array([0.4, 0.2, 0.0]),
        intervention=np.array([0.0, 0.3, 0.4]),
        weights=(1.0, 0.1, 2.0),
    )
    posterior = gibbs_posterior(("gear", "kuramoto", "independent"), energy, beta=4.0)
    assert posterior.manifest_index == 0
    assert math.isclose(float(np.sum(posterior.probability)), 1.0)
    kernels = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    residual = nonselected_residual_field(posterior, kernels)
    max_norm = max(float(np.linalg.norm(row)) for row in kernels)
    assert np.linalg.norm(residual) <= max_norm * posterior.nonselected_mass + 1e-12
    gradients = np.array([[0.0, 0.0], [1.0, -0.2], [-0.4, 1.1]])
    metric = fisher_metric(gradients, posterior.probability, beta=4.0)
    assert float(np.min(np.linalg.eigvalsh(metric))) >= -1e-12
    parameter = np.array([0.4, -0.3])
    objective_gradient = np.array([0.2, -0.1])
    updated = natural_gradient_step(parameter, objective_gradient, metric, step=0.01)
    displacement = updated - parameter
    assert float(objective_gradient @ displacement) <= 1e-12


def test_langevin_is_seeded_finite_and_explores_nonzero_energy() -> None:
    network = _chain()
    theta = np.linalg.pinv(network.incidence) @ network.offset
    first = network.simulate_langevin(theta, 0.2, 0.001, temperature=0.03, seed=9)
    second = network.simulate_langevin(theta, 0.2, 0.001, temperature=0.03, seed=9)
    assert np.array_equal(first.theta, second.theta)
    assert np.all(np.isfinite(first.theta))
    assert float(np.mean(first.energy)) > 0.0


def test_warped_curvature_and_end_to_end_demo() -> None:
    coordinate = np.array([0.0, np.pi / 2.0, np.pi])
    curvature = warped_gaussian_curvature(coordinate, epsilon=0.2)
    assert np.allclose(curvature, np.array([1.0 / 6.0, 0.0, -0.25]), atol=1e-12)
    report, trajectory = run_demo()
    assert report.passed
    assert report.selected_candidate == "gear-2:3"
    assert trajectory.energy[-1] < trajectory.energy[0]
