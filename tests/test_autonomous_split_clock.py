"""자율 분할의 시계 꼬리·에너지·출력 보존을 독립 식과 대조한다."""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq

PATH = Path(__file__).resolve().parents[1]/"verify/Q-0020/autonomous_split_clock.py"
SPEC = importlib.util.spec_from_file_location("autonomous_split_clock", PATH)
clock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clock)


def vacuum_generator_moments(metric):
    k, cutoff = len(metric)//2, 4
    annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q = (annihilation+annihilation.T)/math.sqrt(2)
    p = (annihilation-annihilation.T)/(1j*math.sqrt(2))
    coordinates = []
    for mode in range(k):
        for local in (q, p):
            operator = np.ones((1, 1))
            for index in range(k):
                operator = np.kron(operator, local if index == mode else np.eye(cutoff))
            coordinates.append(operator)
    vacuum = np.eye(cutoff**k)[:, 0]
    applied = sum(metric[a, b]*(coordinates[a] @ (coordinates[b] @ vacuum))/2
                  for a in range(2*k) for b in range(2*k))
    return float(np.vdot(vacuum, applied).real), float(np.vdot(applied, applied).real)


def test_gaussian_tails_from_independent_density_integrals():
    sigma, distance, momentum, mass, length, time = .8, 2.0, 1.3, 1.2, .7, 3.5
    data = clock.clock_bound(time, sigma, distance, momentum, mass, length)
    density = lambda x, mean, width: math.exp(-(x-mean)**2/(2*width**2))/(math.sqrt(2*math.pi)*width)
    initial = quad(lambda x: density(x, -distance, sigma), 0, np.inf)[0]
    negative = quad(lambda p: density(p, momentum, 1/(2*sigma)), -np.inf, 0)[0]
    width = math.sqrt(sigma**2+time**2/(4*mass**2*sigma**2))
    unfinished = quad(lambda x: density(x, -distance+momentum*time/mass, width), -np.inf, length)[0]
    assert data["initial_position_tail"] == pytest.approx(initial, abs=1e-12)
    assert data["negative_momentum_probability"] == pytest.approx(negative, abs=1e-12)
    assert data["unfinished_probability"] == pytest.approx(unfinished, abs=1e-12)
    assert abs(initial-negative) > 1e-3


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-4])
def test_completion_time_and_all_later_times(tolerance):
    time = clock.completion_time(tolerance)
    independent = brentq(lambda value: clock.clock_bound(value)["trace_distance_bound"]-tolerance, 0, 1e5)
    assert time == pytest.approx(independent, rel=1e-10)
    values = [clock.clock_bound(value)["trace_distance_bound"] for value in np.geomspace(time, 1e6, 30)]
    assert max(values) <= tolerance*(1+1e-12)
    assert np.max(np.diff(values)) < 0


@pytest.mark.parametrize("k", [2, 3])
def test_output_preserves_split_with_reference_and_free_parent_phase(k):
    symplectic, form = clock.transported_form(k)
    original = clock.generator.source.source_dilation(k)
    omega = np.kron(np.eye(k), clock.generator.source.J)
    # 임의 부모 입력을 보존하는 두 열과 진공 보조 열의 공분산을 별도로 대조한다.
    assert np.linalg.norm(symplectic[:, :2]-original[:, :2]) < 1e-11
    assert np.linalg.norm(symplectic[:, 2:] @ symplectic[:, 2:].T
                          -original[:, 2:] @ original[:, 2:].T) < 1e-11
    time = .73
    output_evolution = expm(time*omega @ form)
    parent_evolution = expm(time*clock.generator.source.J)
    assert np.linalg.norm(output_evolution @ symplectic[:, :2]
                          -symplectic[:, :2] @ parent_evolution) < 1e-11
    ancilla = .5*symplectic[:, 2:] @ symplectic[:, 2:].T
    assert np.linalg.norm(output_evolution @ ancilla @ output_evolution.T-ancilla) < 1e-11
    # 부모·참조 상관 블록은 부모 두 열로 전달되므로 참조 자유도는 건드리지 않는다.
    cross = np.array([[.3, -.2], [.1, .4]])
    assert np.linalg.norm(output_evolution @ symplectic[:, :2] @ cross
                          -symplectic[:, :2] @ parent_evolution @ cross) < 1e-11
    _, wound = clock.transported_form(k, winding=1)
    assert np.linalg.norm(form-wound) < 1e-11


@pytest.mark.parametrize("k,winding", [(2, 0), (2, 1), (3, 0), (3, 1)])
def test_vacuum_energy_moments_against_fock_operators(k, winding):
    metric = clock.generator.witness(k, winding)["generator"]
    mean, second = vacuum_generator_moments(metric)
    energy = clock.initial_vacuum_energy(k, distance=1, momentum=1, winding=winding)
    assert energy["generator_vacuum_mean"] == pytest.approx(mean, rel=1e-12)
    assert energy["generator_vacuum_second_moment"] == pytest.approx(second, rel=1e-12)
    assert energy["gauge_square_energy"] > 0
    assert energy["total_energy"] > energy["clock_kinetic_energy"]+k/2


def test_initial_product_energy_by_direct_derivative_quadrature():
    k, sigma, distance, momentum, mass = 2, 1.0, 1.0, 1.0, 1.0
    metric = clock.generator.witness(k)["generator"]
    omega = np.kron(np.eye(k), clock.generator.source.J)
    mean, second = vacuum_generator_moments(metric)

    def density_energy(x):
        fraction, slope = clock.profile(x)
        density = math.exp(-(x+distance)**2/2)/math.sqrt(2*math.pi)
        derivative_norm = momentum**2+(x+distance)**2/4+2*momentum*slope*mean+slope*slope*second
        inverse = expm(-float(fraction)*omega @ metric)
        potential = np.trace(inverse.T @ inverse)/4
        return float(density*(derivative_norm/(2*mass)+potential))

    independent = sum(quad(density_energy, a, b, epsabs=1e-11)[0] for a, b in [(-15, 0), (0, 1), (1, 15)])
    result = clock.initial_vacuum_energy(k, sigma, distance, momentum, mass)
    assert result["total_energy"] == pytest.approx(independent, rel=1e-11)


def test_finite_grid_full_hamiltonian_with_entangled_reference():
    result = clock.finite_grid_witness()
    assert result["minimum_total_eigenvalue"] > 0
    assert result["noncommutation_norm"] > .1
    assert result["state_factorization_residual"] < 1e-11
    assert result["energy_conservation_residual"] < 1e-11
    assert result["norm_residual"] < 1e-11
    assert result["product_input_trace_distance"] <= result["discrete_trace_distance_bound"]+1e-11
    assert not result["continuum_scattering_or_infinite_time_proved_by_grid"]


@pytest.mark.parametrize("k", [2, 3])
def test_omitting_square_gives_negative_quadratic_growth(k):
    states = [clock.missing_square_witness(k, n) for n in (0, 20, 40)]
    energies = [state["truncated_form_energy"] for state in states]
    assert all(state["full_square_energy"] > 0 for state in states)
    assert energies[-1] < -100
    assert energies[2]-2*energies[1]+energies[0] < -1
    # 완전제곱의 에너지는 n에 선형이고, 누락된 양의 항은 n에 이차다.
    good = [state["full_square_energy"] for state in states]
    assert good[2]-2*good[1]+good[0] == pytest.approx(0, abs=1e-10)


def test_unreachable_tolerance_and_invalid_clock_are_rejected():
    with pytest.raises(ValueError):
        clock.completion_time(1e-10)
    with pytest.raises(ValueError):
        clock.clock_bound(-1)
    with pytest.raises(ValueError):
        clock.clock_bound(1, momentum=0)
    with pytest.raises(ValueError):
        clock.profile(0, length=0)


def test_saved_artifact_hashes_and_scope():
    import json
    result = json.loads(PATH.with_suffix(".json").read_text(encoding="utf-8"))
    for name, expected in result["source_sha256"].items():
        assert hashlib.sha256((PATH.parent/name).read_bytes()).hexdigest() == expected
    assert result["scope"]["finite_product_preparation_error_bounded_with_reference"]
    assert not result["scope"]["clock_and_ancilla_preparation_derived"]
    assert not result["scope"]["same_bare_input_output_hamiltonian"]
    assert not result["scope"]["full_action_is_quadratic"]
    assert not result["scope"]["common_metric_selected"]

