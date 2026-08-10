"""Executable Riemannian gear-network model.

The implementation follows ``docs/6_뇌/11_리만_톱니_결합_진화가설.md``.
It deliberately separates exact mathematical certificates from empirical
claims about biological neural systems.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
Drive = Callable[[float, Array], Array]
Clutch = Callable[[float], Array]


def principal_angle(value: Array) -> Array:
    """Map angles to the half-open principal interval [-pi, pi)."""
    array = np.asarray(value, dtype=float)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _positive_definite(matrix: Array, name: str) -> Array:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(result, result.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(result))) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return result


@dataclass(frozen=True)
class FrustrationCertificate:
    winding: tuple[int, ...]
    value: float
    theta_star: Array
    weighted_cycle_residual: Array
    normal_equation_error: float


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: Array
    eigenvectors: Array
    zero_count: int

    @property
    def positive_values(self) -> Array:
        return self.eigenvalues[self.zero_count :]

    @property
    def positive_vectors(self) -> Array:
        return self.eigenvectors[:, self.zero_count :]


@dataclass(frozen=True)
class Trajectory:
    time: Array
    theta: Array
    residual: Array
    energy: Array


@dataclass(frozen=True)
class CandidatePosterior:
    names: tuple[str, ...]
    energy: Array
    prior: Array
    probability: Array
    manifest_index: int
    nonselected_mass: float


@dataclass(frozen=True)
class DemoReport:
    seed: int
    initial_energy: float
    final_energy: float
    initial_max_residual: float
    final_max_residual: float
    tangent_drive_error: float
    positive_spectrum: tuple[float, ...]
    frustration_consistent: float
    frustration_inconsistent: float
    posterior_names: tuple[str, ...]
    posterior_probability: tuple[float, ...]
    selected_candidate: str
    fisher_eigenvalues: tuple[float, ...]
    stochastic_mean_energy: float
    passed: bool


class RiemannianGearNetwork:
    """A finite toroidal gear network with cosine edge potentials."""

    def __init__(
        self,
        incidence: Array,
        stiffness: Array,
        offset: Array,
        mass: Array | None = None,
    ) -> None:
        incidence_array = np.asarray(incidence, dtype=float)
        if incidence_array.ndim != 2 or not np.all(np.isfinite(incidence_array)):
            raise ValueError("incidence must be a finite E x N matrix")
        if not np.allclose(incidence_array, np.rint(incidence_array), atol=1e-12):
            raise ValueError("gear incidence entries must be integers")
        edge_count, node_count = incidence_array.shape
        stiffness_array = np.asarray(stiffness, dtype=float)
        if stiffness_array.ndim == 1:
            stiffness_array = np.diag(stiffness_array)
        if stiffness_array.shape != (edge_count, edge_count):
            raise ValueError("stiffness must have shape E or E x E")
        if not np.allclose(stiffness_array, np.diag(np.diag(stiffness_array))):
            raise ValueError("the documented model requires diagonal stiffness")
        if np.any(np.diag(stiffness_array) <= 0.0):
            raise ValueError("every edge stiffness must be positive")
        offset_array = np.asarray(offset, dtype=float)
        if offset_array.shape != (edge_count,):
            raise ValueError("offset must have shape E")
        mass_array = np.eye(node_count) if mass is None else mass

        self.incidence = incidence_array
        self.stiffness = stiffness_array
        self.offset = principal_angle(offset_array)
        self.mass = _positive_definite(np.asarray(mass_array, dtype=float), "mass")
        if self.mass.shape != (node_count, node_count):
            raise ValueError("mass must have shape N x N")
        self.mass_inverse = np.linalg.inv(self.mass)
        eigenvalues, eigenvectors = np.linalg.eigh(self.mass)
        self.mass_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        self.mass_inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T

    @property
    def edge_count(self) -> int:
        return self.incidence.shape[0]

    @property
    def node_count(self) -> int:
        return self.incidence.shape[1]

    def _clutch_vector(self, clutch: Array | None) -> Array:
        result = np.ones(self.edge_count) if clutch is None else np.asarray(clutch, dtype=float)
        if result.shape != (self.edge_count,):
            raise ValueError("clutch must have shape E")
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("clutch entries must lie in [0, 1]")
        return result

    def lifted_residual(self, theta: Array, winding: Array | None = None) -> Array:
        state = np.asarray(theta, dtype=float)
        if state.shape != (self.node_count,):
            raise ValueError("theta must have shape N")
        lift = np.zeros(self.edge_count) if winding is None else np.asarray(winding, dtype=float)
        if lift.shape != (self.edge_count,):
            raise ValueError("winding must have shape E")
        return self.incidence @ state - self.offset - 2.0 * np.pi * lift

    def residual(self, theta: Array) -> Array:
        state = np.asarray(theta, dtype=float)
        if state.shape != (self.node_count,):
            raise ValueError("theta must have shape N")
        return principal_angle(self.incidence @ state - self.offset)

    @property
    def locking_dimension(self) -> int:
        """Dimension N-rank(B) of every consistent lifted locking chart."""
        return self.node_count - int(np.linalg.matrix_rank(self.incidence))

    def is_lift_consistent(self, winding: Array | None = None, tolerance: float = 1e-10) -> bool:
        """Test whether B theta = offset + 2 pi winding has a solution."""
        return self.frustration(winding).value <= tolerance**2 / 2.0

    def energy(self, theta: Array, clutch: Array | None = None) -> float:
        engagement = self._clutch_vector(clutch)
        edge_energy = np.diag(self.stiffness) * (1.0 - np.cos(self.residual(theta)))
        return float(engagement @ edge_energy)

    def gradient(self, theta: Array, clutch: Array | None = None) -> Array:
        engagement = self._clutch_vector(clutch)
        edge_force = engagement * np.diag(self.stiffness) * np.sin(self.residual(theta))
        return self.incidence.T @ edge_force

    def gear_laplacian(self, clutch: Array | None = None) -> Array:
        engagement = self._clutch_vector(clutch)
        effective = np.diag(engagement * np.diag(self.stiffness))
        return self.incidence.T @ effective @ self.incidence

    def state_metric(
        self,
        theta: Array,
        alpha: Array,
        q: Callable[[Array], Array] | None = None,
    ) -> Array:
        coefficient = np.asarray(alpha, dtype=float)
        if coefficient.shape != (self.edge_count,) or np.any(coefficient < 0.0):
            raise ValueError("alpha must be a nonnegative vector with shape E")
        residual = self.residual(theta)
        q_value = 1.0 - np.cos(residual) if q is None else np.asarray(q(residual), dtype=float)
        if q_value.shape != (self.edge_count,) or np.any(q_value < 0.0):
            raise ValueError("q(residual) must be nonnegative with shape E")
        metric = self.mass + self.incidence.T @ np.diag(coefficient * q_value) @ self.incidence
        return _positive_definite(metric, "state metric")

    def velocity(
        self,
        time: float,
        theta: Array,
        drive: Drive | None = None,
        clutch: Array | None = None,
        metric_alpha: Array | None = None,
    ) -> Array:
        external = np.zeros(self.node_count) if drive is None else np.asarray(drive(time, theta))
        if external.shape != (self.node_count,):
            raise ValueError("drive must return shape N")
        mobility = self.mass_inverse
        if metric_alpha is not None:
            mobility = np.linalg.inv(self.state_metric(theta, metric_alpha))
        return external - mobility @ self.gradient(theta, clutch)

    def simulate(
        self,
        theta_0: Array,
        duration: float,
        step: float,
        drive: Drive | None = None,
        clutch: Clutch | None = None,
        metric_alpha: Array | None = None,
        wrap_state: bool = True,
    ) -> Trajectory:
        if duration <= 0.0 or step <= 0.0:
            raise ValueError("duration and step must be positive")
        count = int(math.ceil(duration / step))
        times = np.linspace(0.0, duration, count + 1)
        states = np.empty((count + 1, self.node_count))
        residuals = np.empty((count + 1, self.edge_count))
        energies = np.empty(count + 1)
        states[0] = np.asarray(theta_0, dtype=float)

        def current_clutch(value: float) -> Array:
            return self._clutch_vector(None if clutch is None else clutch(value))

        residuals[0] = self.residual(states[0])
        energies[0] = self.energy(states[0], current_clutch(0.0))
        for index in range(count):
            time = times[index]
            dt = times[index + 1] - time
            state = states[index]

            def field(at: float, value: Array) -> Array:
                return self.velocity(
                    at,
                    value,
                    drive=drive,
                    clutch=current_clutch(at),
                    metric_alpha=metric_alpha,
                )

            k1 = field(time, state)
            k2 = field(time + dt / 2.0, state + dt * k1 / 2.0)
            k3 = field(time + dt / 2.0, state + dt * k2 / 2.0)
            k4 = field(time + dt, state + dt * k3)
            next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            states[index + 1] = principal_angle(next_state) if wrap_state else next_state
            residuals[index + 1] = self.residual(states[index + 1])
            energies[index + 1] = self.energy(states[index + 1], current_clutch(times[index + 1]))
        return Trajectory(times, states, residuals, energies)

    def simulate_langevin(
        self,
        theta_0: Array,
        duration: float,
        step: float,
        temperature: float,
        seed: int,
        burn_in: float = 0.0,
    ) -> Trajectory:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if duration <= 0.0 or step <= 0.0 or not 0.0 <= burn_in < duration:
            raise ValueError("require duration, step > 0 and 0 <= burn_in < duration")
        count = int(math.ceil(duration / step))
        times = np.linspace(0.0, duration, count + 1)
        states = np.empty((count + 1, self.node_count))
        residuals = np.empty((count + 1, self.edge_count))
        energies = np.empty(count + 1)
        states[0] = principal_angle(np.asarray(theta_0, dtype=float))
        residuals[0] = self.residual(states[0])
        energies[0] = self.energy(states[0])
        rng = np.random.default_rng(seed)
        noise_factor = math.sqrt(2.0 * temperature) * self.mass_inverse_sqrt
        for index in range(count):
            dt = times[index + 1] - times[index]
            drift = -self.mass_inverse @ self.gradient(states[index])
            noise = math.sqrt(dt) * noise_factor @ rng.normal(size=self.node_count)
            states[index + 1] = principal_angle(states[index] + dt * drift + noise)
            residuals[index + 1] = self.residual(states[index + 1])
            energies[index + 1] = self.energy(states[index + 1])
        start = int(math.ceil(burn_in / duration * count))
        return Trajectory(times[start:], states[start:], residuals[start:], energies[start:])

    def frustration(self, winding: Array | None = None) -> FrustrationCertificate:
        raw_lift = np.zeros(self.edge_count) if winding is None else np.asarray(winding, dtype=float)
        if not np.allclose(raw_lift, np.rint(raw_lift), atol=1e-12):
            raise ValueError("winding entries must be integers")
        lift = np.rint(raw_lift).astype(int)
        if lift.shape != (self.edge_count,):
            raise ValueError("winding must have shape E")
        target = self.offset + 2.0 * np.pi * lift
        sqrt_k = np.diag(np.sqrt(np.diag(self.stiffness)))
        operator = sqrt_k @ self.incidence
        weighted_target = sqrt_k @ target
        theta_star = np.linalg.pinv(operator) @ weighted_target
        cycle_residual = weighted_target - operator @ theta_star
        value = 0.5 * float(cycle_residual @ cycle_residual)
        error = float(np.linalg.norm(operator.T @ cycle_residual))
        return FrustrationCertificate(tuple(int(v) for v in lift), value, theta_star, cycle_residual, error)

    def best_frustration(self, windings: Sequence[Array]) -> FrustrationCertificate:
        certificates = [self.frustration(winding) for winding in windings]
        if not certificates:
            raise ValueError("at least one preregistered winding is required")
        return min(certificates, key=lambda item: item.value)

    def spectrum(self, tolerance: float = 1e-10) -> Spectrum:
        whitened = self.mass_inverse_sqrt @ self.gear_laplacian() @ self.mass_inverse_sqrt
        values, vectors = np.linalg.eigh(whitened)
        zero_count = int(np.sum(values <= tolerance))
        return Spectrum(values, vectors, zero_count)

    def contraction_factor(self, eta: float, tolerance: float = 1e-10) -> float:
        """Return q on the normal subspace, enforcing 0 < eta < 2/lambda_max."""
        positive = self.spectrum(tolerance).positive_values
        if positive.size == 0:
            raise ValueError("network has no normal restoring modes")
        if not 0.0 < eta < 2.0 / float(positive[-1]):
            raise ValueError("eta violates the documented contraction condition")
        return float(np.max(np.abs(1.0 - eta * positive)))

    def iss_bound(self, initial_norm: float, disturbance_bound: float, eta: float, steps: int) -> float:
        if initial_norm < 0.0 or disturbance_bound < 0.0 or steps < 0:
            raise ValueError("norms and steps must be nonnegative")
        q = self.contraction_factor(eta)
        return q**steps * initial_norm + (1.0 - q**steps) * disturbance_bound / (1.0 - q)

    def tangent_projection(self, vector: Array, tolerance: float = 1e-10) -> Array:
        _, singular, right = np.linalg.svd(self.incidence, full_matrices=True)
        rank = int(np.sum(singular > tolerance))
        kernel = right[rank:].T
        return kernel @ (kernel.T @ np.asarray(vector, dtype=float))

    def discrete_normal_step(self, state: Array, eta: float, disturbance: Array | None = None) -> Array:
        whitened = self.mass_inverse_sqrt @ self.gear_laplacian() @ self.mass_inverse_sqrt
        forcing = np.zeros(self.node_count) if disturbance is None else np.asarray(disturbance, dtype=float)
        return (np.eye(self.node_count) - eta * whitened) @ np.asarray(state, dtype=float) + forcing

    def spectral_evolution(self, state: Array, time: float, keep_positive: int | None = None) -> Array:
        spectrum = self.spectrum()
        coefficients = spectrum.eigenvectors.T @ np.asarray(state, dtype=float)
        factors = np.exp(-spectrum.eigenvalues * time)
        if keep_positive is not None:
            if not 0 <= keep_positive <= len(spectrum.positive_values):
                raise ValueError("invalid number of positive modes")
            factors[spectrum.zero_count + keep_positive :] = 0.0
        return spectrum.eigenvectors @ (factors * coefficients)

    def spectral_truncation_bound(self, state: Array, time: float, keep_positive: int) -> float:
        spectrum = self.spectrum()
        positive = spectrum.positive_values
        if not 0 <= keep_positive < len(positive):
            raise ValueError("keep_positive must leave at least one positive mode out")
        return math.exp(-float(positive[keep_positive]) * time) * float(
            np.linalg.norm(np.asarray(state, dtype=float))
        )

    def cosine_quadratic_bounds(
        self,
        theta: Array,
        winding: Array | None,
        radius: float,
    ) -> tuple[float, float, float]:
        """Return c_r Q, V, Q inside a fixed lifted principal chart."""
        if not 0.0 < radius < np.pi:
            raise ValueError("radius must lie in (0, pi)")
        delta = self.lifted_residual(theta, winding)
        if float(np.max(np.abs(delta))) > radius:
            raise ValueError("state lies outside the requested lifted chart")
        quadratic = 0.5 * float(delta @ self.stiffness @ delta)
        cosine = float(np.diag(self.stiffness) @ (1.0 - np.cos(delta)))
        coefficient = 2.0 * (1.0 - math.cos(radius)) / radius**2
        return coefficient * quadratic, cosine, quadratic


def generalized_phase_locking(phases: Array, weights: Array) -> tuple[float, float]:
    """Return generalized phase-locking magnitude and circular offset."""
    observations = np.asarray(phases, dtype=float)
    coefficient = np.asarray(weights, dtype=float)
    if observations.ndim != 2 or coefficient.shape != (observations.shape[1],):
        raise ValueError("phases must be T x N and weights must have shape N")
    circular_mean = np.mean(np.exp(1j * (observations @ coefficient)))
    return float(abs(circular_mean)), float(np.angle(circular_mean))


def gibbs_posterior(
    names: Sequence[str],
    energy: Array,
    prior: Array | None = None,
    beta: float = 1.0,
) -> CandidatePosterior:
    values = np.asarray(energy, dtype=float)
    if values.ndim != 1 or values.shape != (len(names),) or len(names) == 0:
        raise ValueError("names and energy must describe the same nonempty candidate set")
    baseline = np.ones(len(names), dtype=float) if prior is None else np.asarray(prior, dtype=float)
    if baseline.shape != values.shape or np.any(baseline < 0.0) or not np.any(baseline > 0.0):
        raise ValueError("prior must be nonnegative with nonempty support")
    log_prior = np.full_like(baseline, -np.inf)
    support = baseline > 0.0
    log_prior[support] = np.log(baseline[support])
    log_weight = -beta * values + log_prior
    log_weight -= np.max(log_weight)
    probability = np.exp(log_weight)
    probability /= np.sum(probability)
    manifest = int(np.argmax(probability))
    return CandidatePosterior(
        tuple(names), values, baseline / np.sum(baseline), probability, manifest, 1.0 - probability[manifest]
    )


def nonselected_residual_field(posterior: CandidatePosterior, kernels: Array) -> Array:
    values = np.asarray(kernels, dtype=float)
    if values.shape[0] != len(posterior.names):
        raise ValueError("kernels must have one leading entry per candidate")
    weights = posterior.probability.copy()
    weights[posterior.manifest_index] = 0.0
    return np.tensordot(weights, values, axes=(0, 0))


def fisher_metric(energy_gradients: Array, probability: Array, beta: float) -> Array:
    gradients = np.asarray(energy_gradients, dtype=float)
    weights = np.asarray(probability, dtype=float)
    if gradients.ndim != 2 or weights.shape != (gradients.shape[0],):
        raise ValueError("energy_gradients must be candidates x parameters")
    if np.any(weights < 0.0) or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1e-10):
        raise ValueError("probability must sum to one")
    centered = gradients - weights @ gradients
    return beta**2 * (centered.T * weights) @ centered


def natural_gradient_step(parameter: Array, objective_gradient: Array, metric: Array, step: float) -> Array:
    if step <= 0.0:
        raise ValueError("step must be positive")
    return np.asarray(parameter, dtype=float) - step * np.linalg.pinv(metric) @ np.asarray(
        objective_gradient, dtype=float
    )


def compose_candidate_energy(
    prediction: Array,
    frustration: Array,
    complexity: Array,
    intervention: Array,
    weights: tuple[float, float, float],
) -> Array:
    """Implement E_pred + lambda_F F + lambda_C C + lambda_I E_intervention."""
    terms = [np.asarray(item, dtype=float) for item in (prediction, frustration, complexity, intervention)]
    if any(item.shape != terms[0].shape for item in terms[1:]) or terms[0].ndim != 1:
        raise ValueError("all candidate-energy terms must be vectors of equal shape")
    if any(value < 0.0 for value in weights):
        raise ValueError("candidate-energy weights must be nonnegative")
    return terms[0] + weights[0] * terms[1] + weights[1] * terms[2] + weights[2] * terms[3]


def pair_lock_time(delta_0: float, coupling_rate: float, epsilon: float) -> float:
    """Exact epsilon-locking time for 0 < epsilon < |delta_0| < pi."""
    if not 0.0 < epsilon < abs(delta_0) < np.pi or coupling_rate <= 0.0:
        raise ValueError("require 0 < epsilon < |delta_0| < pi and coupling_rate > 0")
    return math.log(abs(math.tan(delta_0 / 2.0) / math.tan(epsilon / 2.0))) / coupling_rate


def warped_gaussian_curvature(theta_1: Array, epsilon: float) -> Array:
    """Curvature for ds^2=dtheta_1^2+(1+epsilon cos(theta_1))^2 dtheta_2^2."""
    if abs(epsilon) >= 1.0:
        raise ValueError("warped metric requires |epsilon| < 1")
    coordinate = np.asarray(theta_1, dtype=float)
    return epsilon * np.cos(coordinate) / (1.0 + epsilon * np.cos(coordinate))


def _demo_network() -> tuple[RiemannianGearNetwork, Array]:
    incidence = np.array(
        [[2, 3, 0, 0], [0, 3, 4, 0], [0, 0, 4, 5], [2, 0, 0, 5]],
        dtype=int,
    )
    target = np.array([0.10, -0.20, 0.15, 0.08])
    offset = principal_angle(incidence @ target)
    network = RiemannianGearNetwork(
        incidence,
        stiffness=np.array([1.0, 1.4, 0.9, 1.2]),
        offset=offset,
        mass=np.diag([1.0, 1.2, 0.8, 1.1]),
    )
    return network, target


def run_demo(seed: int = 20260810) -> tuple[DemoReport, Trajectory]:
    network, target = _demo_network()
    rng = np.random.default_rng(seed)
    initial = target + rng.normal(scale=0.35, size=network.node_count)
    tangent = network.tangent_projection(np.array([0.6, -0.2, 0.4, -0.1]))

    def drive(_time: float, _state: Array) -> Array:
        return 0.18 * tangent

    trajectory = network.simulate(initial, duration=8.0, step=0.002, drive=drive)
    tangent_error = float(np.linalg.norm(network.incidence @ tangent))

    consistent = network.frustration()
    inconsistent_network = RiemannianGearNetwork(
        network.incidence,
        np.diag(network.stiffness),
        network.offset + np.array([0.0, 0.0, 0.0, 0.35]),
        network.mass,
    )
    inconsistent = inconsistent_network.frustration()

    candidate_energy = np.array(
        [trajectory.energy[-1], trajectory.energy[-1] + 0.7, trajectory.energy[-1] + 1.5]
    )
    posterior = gibbs_posterior(("gear-2:3", "kuramoto-1:1", "independent"), candidate_energy, beta=5.0)
    gradients = np.array([[0.2, -0.1], [1.0, 0.4], [-0.5, 1.2]])
    fisher = fisher_metric(gradients, posterior.probability, beta=5.0)

    stochastic = network.simulate_langevin(
        target,
        duration=4.0,
        step=0.002,
        temperature=0.04,
        seed=seed + 1,
        burn_in=2.0,
    )
    positive = network.spectrum().positive_values
    final_residual = float(np.max(np.abs(trajectory.residual[-1])))
    report = DemoReport(
        seed=seed,
        initial_energy=float(trajectory.energy[0]),
        final_energy=float(trajectory.energy[-1]),
        initial_max_residual=float(np.max(np.abs(trajectory.residual[0]))),
        final_max_residual=final_residual,
        tangent_drive_error=tangent_error,
        positive_spectrum=tuple(float(value) for value in positive),
        frustration_consistent=consistent.value,
        frustration_inconsistent=inconsistent.value,
        posterior_names=posterior.names,
        posterior_probability=tuple(float(value) for value in posterior.probability),
        selected_candidate=posterior.names[posterior.manifest_index],
        fisher_eigenvalues=tuple(float(value) for value in np.linalg.eigvalsh(fisher)),
        stochastic_mean_energy=float(np.mean(stochastic.energy)),
        passed=bool(
            trajectory.energy[-1] < trajectory.energy[0] * 1e-4
            and final_residual < 1e-4
            and tangent_error < 1e-10
            and consistent.value < 1e-20
            and inconsistent.value > 1e-4
            and posterior.manifest_index == 0
            and float(np.min(np.linalg.eigvalsh(fisher))) >= -1e-10
            and np.all(np.isfinite(stochastic.theta))
        ),
    )
    return report, trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args(argv)
    report, trajectory = run_demo(args.seed)
    payload = asdict(report)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.trajectory_output is not None:
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        columns = np.column_stack(
            (trajectory.time, trajectory.energy, trajectory.theta, trajectory.residual)
        )
        header = ",".join(
            ["time", "energy"]
            + [f"theta_{index}" for index in range(trajectory.theta.shape[1])]
            + [f"residual_{index}" for index in range(trajectory.residual.shape[1])]
        )
        np.savetxt(args.trajectory_output, columns, delimiter=",", header=header, comments="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
