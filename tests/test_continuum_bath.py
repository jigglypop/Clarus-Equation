"""연속 스펙트럼을 독립 유한 환경과 Hamiltonian 모멘트로 교차 검산한다."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("continuum_bath_under_test", HERE / "continuum_bath.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from split_quantum_source import recursive_source_maps
finally:
    sys.path[:] = original_path


def _independent_star(epsilon, kappa, nodes):
    x, weights = np.polynomial.laguerre.laggauss(nodes)
    h = np.diag(np.r_[epsilon, x])
    h[0, 1:] = h[1:, 0] = np.sqrt(kappa*x*weights)
    return np.linalg.eigh(h)


@pytest.mark.parametrize("epsilon,kappa", [(2., .5), (2., 1.5)])
def test_continuum_moments_match_operator_powers(epsilon, kappa):
    # <0|h|0>=epsilon, <0|h²|0>=epsilon²+integral J.
    for power, expected in enumerate((1., epsilon, epsilon**2+kappa)):
        value, error = module.spectral_moment(epsilon, kappa, power)
        assert value == pytest.approx(expected, abs=2e-11)
        assert error < 1e-9


def test_density_matches_independent_principal_value_integral():
    for kappa in (.5, 1.5):
        for energy in (.1, 1., 2., 6.):
            j = kappa*energy*math.exp(-energy)
            def regular(x):
                if x == energy:
                    return -kappa*math.exp(-energy)*(1-energy)
                return (kappa*x*math.exp(-x)-j)/(energy-x)
            sigma = quad(regular, 0., 64., epsabs=1e-12)[0]
            sigma += j*math.log(energy/(64-energy))
            expected = j/((energy-2.-sigma)**2+(math.pi*j)**2)
            assert module.spectral_density(energy, 2., kappa) == pytest.approx(expected, rel=1e-11)
    assert module.spectral_density(1000., 2., .5) == 0.


@pytest.mark.parametrize("kappa", [.5, 1.5])
def test_finite_star_converges_at_fixed_times(kappa):
    times = (.5, 1., 2., 5.)
    reference = {t: module.boundary_amplitude(t, 2., kappa)[0] for t in times}
    errors = []
    for count in (64, 128):
        energies, vectors = _independent_star(2., kappa, count)
        assert energies[0] > 0
        errors.append(max(abs(np.sum(vectors[0]**2*np.exp(-1j*energies*t))-reference[t]) for t in times))
    assert errors[1] < 5e-9
    assert errors[1] < errors[0]/1000


def test_actual_split_covariance_matches_finite_unitary_transport():
    energies, vectors = _independent_star(2., .5, 128)
    unitary = (vectors*np.exp(-1j*energies)) @ vectors.T
    real_map = np.zeros((2*len(energies), 2*len(energies)))
    real_map[0::2, 0::2] = real_map[1::2, 1::2] = unitary.real
    real_map[0::2, 1::2] = -unitary.imag
    real_map[1::2, 0::2] = unitary.imag
    # k=3 source contrast has q=3/2, p=1/6; all other modes are vacuum.
    initial = .5*np.eye(len(real_map))
    initial[0, 0], initial[1, 1] = 1.5, 1/6
    final = real_map @ initial @ real_map.T
    amplitude, _ = module.boundary_amplitude(1., 2., .5)
    expected_q = .5+abs(amplitude)**2/3+(2*amplitude**2/3).real
    assert final[0, 0] == pytest.approx(expected_q, abs=2e-12)
    assert (final[0, 0]+final[1, 1]-1)/2 == pytest.approx(abs(amplitude)**2/3, abs=2e-12)


def test_pair_threshold_is_positive_and_cubic():
    energy = 1e-4
    for kappa in (.5, 1.5):
        value, _ = module.pair_density(energy, 2., kappa)
        coefficient = (kappa/(2-kappa)**2)**2/6
        assert value > 0
        assert value/energy**3 == pytest.approx(coefficient, rel=.02)
        assert module.pair_density(4., 2., kappa)[0] > 0


def test_full_source_bound_keeps_higher_sectors_and_vacuum_variance():
    for kappa in (.5, 1.5):
        result = module.source_budget(3, 2., kappa, 8.)
        assert result["mean_number"] == pytest.approx(1/3)
        assert result["two_particle_probability"] == pytest.approx(math.sqrt(3)/16)
        assert result["initial_energy_over_Estar"] == pytest.approx(10/3)
        assert result["nonlinear_full_source_limsup_number_upper_bound"] == pytest.approx(1/3-math.sqrt(3)/8)
        assert result["linear_asymptotic_q_variance"] == .5


@pytest.mark.parametrize("epsilon,kappa", [(2., 0.), (2., 2.), (2., 3.), (float("nan"), 1.), (2., float("inf"))])
def test_parameters_exclude_threshold_and_unstable_domain(epsilon, kappa):
    with pytest.raises(ValueError):
        module.spectral_density(1., epsilon, kappa)


def test_output_does_not_promote_partial_scattering_to_CE():
    result = module.run()
    for flag in ("finite_numerics_prove_infinite_time_limit",
                 "nonlinear_full_source_complete_emission_proved",
                 "zero_occupation_means_zero_coordinate_variance",
                 "source_preparation_and_energy_recycling_derived",
                 "microscopic_coupling_derived_from_CE", "common_metric_selection_proved"):
        assert result[flag] is False
    assert len(result["cases"]) == 2
    assert len(result["refinement"]["rows"]) == 8
    assert not result["refinement"]["interleaved_splitting_and_emission_solved"]
    assert not result["refinement"]["normalized_control"]["normalization_physically_derived"]


def test_actual_refinement_owner_counts():
    rows = module.refinement_owner_counts(4)
    for row in rows:
        depth = row["depth"]
        assert row["old_edge_owners"] == 3**depth
        assert row["max_owners"] == 4*3**(depth-1)
        assert sum(k*count for k, count in row["owner_histogram"].items()) == 10*5**depth
    assert rows[-1]["owner_histogram"] == {4: 625, 12: 125, 36: 25, 81: 10, 108: 5}


def test_actual_gluing_and_all_pairs_share_kernel_but_not_energy():
    from itertools import combinations
    refine, gluing = module._refinement_functions()
    points = {i: np.eye(5)[i] for i in range(5)}
    cells = refine([tuple(range(5))], points)
    first_owner = gluing(cells)
    owners = {}
    for cell_index, cell in enumerate(cells):
        for edge_index, edge in enumerate(combinations(cell, 2)):
            owners.setdefault(tuple(sorted(edge)), []).append(10*cell_index+edge_index)
    rows = []
    for indices in owners.values():
        for first, second in combinations(indices, 2):
            row = np.zeros(50)
            row[first], row[second] = 1., -1.
            rows.append(row)
    all_pairs = np.asarray(rows)
    projectors = []
    for matrix in (first_owner, all_pairs):
        _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
        active = vectors[singular > 1e-10]
        assert len(active) == 35
        projectors.append(active.T @ active)
    np.testing.assert_allclose(*projectors, atol=2e-14)
    assert np.linalg.norm(first_owner.T @ first_owner-all_pairs.T @ all_pairs) > 1


def test_fixed_pair_strength_needs_counterterm_for_all_depths():
    assert not module.collective_stability(3, 2., .5, 0.)["negative_mode_present"]
    assert module.collective_stability(9, 2., .5, 0.)["negative_mode_present"]
    for owners in (3, 9, 81, 6561):
        branch = module.collective_stability(owners, 2., .5, 1.)
        assert branch["schur_coefficient"] == 2
        assert branch["nonnegative_for_every_owner_count"]
        assert not branch["schur_coefficient_is_full_spectral_gap"]
    result = module._negative_mode(2., 4.5)
    eigenvalues, _ = _independent_star(2., 4.5, 128)
    assert result["energy"] < 0
    assert eigenvalues[0] == pytest.approx(result["energy"], abs=1e-10)


@pytest.mark.parametrize("epsilon,strength", [(1., 3.), (2., 27.), (2.5, 1000.)])
def test_quasimode_identity_and_survival_against_independent_matrix(epsilon, strength):
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    energies, vectors = _independent_star(epsilon+strength, strength, 64)
    psi = np.r_[1., np.sqrt(nodes*weights/strength)]/math.sqrt(1+1/strength)
    for time in (.25, 1., 5.):
        bound = module.quasimode_bound(strength, epsilon, time)
        actual_residual = (vectors*(energies-bound["quasimode_energy"])) @ (vectors.T @ psi)
        assert np.linalg.norm(actual_residual)**2 == pytest.approx(
            bound["quasimode_residual_squared"], abs=2e-11)
        amplitude = np.sum(vectors[0]**2*np.exp(-1j*energies*time))
        assert abs(amplitude)**2 >= bound["survival_probability_lower_bound"]-1e-11


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_recursive_source_energy_uses_repeated_three_way_split(depth):
    q_map, p_map = recursive_source_maps(3, depth)
    n = 3**depth
    covariance_q, covariance_p = .5*q_map @ q_map.T, .5*p_map @ p_map.T
    number = (covariance_q+covariance_p-np.eye(n))/2
    projector = np.eye(n)-np.ones((n, n))/n
    expected_energy = np.trace((2*np.eye(n)+.5*n*projector) @ number)
    actual = module.recursive_preparation_energy(depth, 2., .5)
    assert actual["total_energy_over_Estar"] == pytest.approx(expected_energy, abs=2e-9)
    assert actual["contrast_number"] == pytest.approx(np.trace(projector @ number), abs=2e-12)
    assert actual["branching_per_step"] == 3
    # A single n-way source differs from D repeated three-way preparations.
    if depth > 1:
        assert actual["total_number"] > (3*n*n-2*n+3)/(8*n)


def test_release_clock_lower_bound_grows_with_refinement():
    shallow = module.quasimode_bound(1e5, 2., 1.)
    deep = module.quasimode_bound(4e5, 2., 1.)
    assert deep["time_to_target_lower_bound"] > 1.95*shallow["time_to_target_lower_bound"]
    assert deep["survival_probability_lower_bound"] > .999


@pytest.mark.parametrize("time", [0., 1., 2.])
def test_collision_switching_work_matches_independent_bath_energy(time):
    result = module.collision_response(2., 1.5, time)
    x, weights = np.polynomial.laguerre.laggauss(128)
    h = np.diag(np.r_[3.5, x])
    h[0, 1:] = h[1:, 0] = np.sqrt(1.5*x*weights)
    energies, vectors = np.linalg.eigh(h)
    column = vectors @ (vectors[0]*np.exp(-1j*energies*time))
    bath = float(np.sum(x*abs(column[1:])**2))
    interaction = 2*(column[0].conjugate()*(h[0, 1:] @ column[1:])).real
    off = -1.5*abs(column[0])**2-interaction
    assert result["bath_energy_per_initial_number"] == pytest.approx(bath, abs=2e-10)
    assert result["switch_off_work_per_initial_number"] == pytest.approx(off, abs=2e-10)
    assert result["net_switch_work_per_initial_number"] == pytest.approx(
        bath+2*(abs(column[0])**2-1), abs=2e-10)


def test_interleaved_cohorts_match_actual_full_source_and_channel():
    from scipy.linalg import block_diag
    local = module.source_dilation(3)
    result = module.interleaved_source(4)
    response = result["response"]
    amplitude = complex(*response["amplitude"])
    rotation = lambda z: np.array([[z.real, -z.imag], [z.imag, z.real]])
    common_rotation = rotation(np.exp(-2j))
    contrast_rotation = rotation(amplitude)
    dense, basis = np.eye(2)/2, np.ones((1, 1))
    for depth in range(1, 5):
        parents, owners = 3**(depth-1), 3**depth
        parent_map = np.kron(np.eye(parents), local[:, :2])
        added_noise = np.kron(np.eye(parents), .5*local[:, 2:] @ local[:, 2:].T)
        dense = parent_map @ dense @ parent_map.T+added_noise
        common = np.ones((owners, owners))/owners
        contrast = np.eye(owners)-common
        channel = np.kron(common, common_rotation)+np.kron(contrast, contrast_rotation)
        dense = channel @ dense @ channel.T
        dense += (1-abs(amplitude)**2)*np.kron(contrast, np.eye(2))/2
        inherited = np.kron(np.eye(parents), np.ones((3, 1))/math.sqrt(3))
        basis = np.hstack((inherited @ basis,
                           np.kron(np.eye(parents), module.mode_basis(3)[:, 1:])))
        reduced = module.interleaved_source(depth)
        expected = block_diag(reduced["final_common_covariance"], *[
            cohort["covariance"] for cohort in reduced["final_contrast_cohorts"]
            for _ in range(cohort["multiplicity"])])
        actual = np.kron(basis.T, np.eye(2)) @ dense @ np.kron(basis, np.eye(2))
        assert np.max(abs(actual-expected)) < 2e-12
        assert np.trace(dense)-owners == pytest.approx(
            reduced["rows"][-1]["system_energy_over_Estar"], abs=2e-11)


def test_zero_wait_recovers_actual_recursive_preparation():
    result = module.interleaved_source(4, time=0.)
    for row in result["rows"]:
        qmap, pmap = recursive_source_maps(3, row["depth"])
        expected = (np.trace(.5*qmap @ qmap.T)+np.trace(.5*pmap @ pmap.T)-row["owners"])
        assert row["system_energy_over_Estar"] == pytest.approx(expected, abs=2e-11)
        assert row["total_external_work_over_Estar"] == pytest.approx(expected, abs=2e-11)
        assert row["emitted_energy_over_Estar"] == 0.


def test_interleaved_energy_balance_and_density_limit_keep_resource_cost():
    result = module.interleaved_source(24)
    for row in result["rows"]:
        assert row["balance_relative_residual"] < 2e-13
        assert row["contrast_number_identity_residual"] < 2e-13
        assert row["contrast_number_before"] >= (2/9)*row["owners"]-1e-12
        assert row["total_external_work_over_Estar"] >= row["external_work_lower_bound_over_Estar"]
    limit = result["limiting"]
    assert limit["contraction_upper_bound"] < 1
    assert limit["common_vanishes_per_owner"]
    assert limit["lyapunov_residual"] < 2e-14
    assert result["rows"][-1]["system_energy_per_owner"] == pytest.approx(
        limit["system_energy_per_owner_if_common_vanishes"], abs=2e-10)
    assert result["minimum_external_work_per_owner"] > .3
    assert not result["bath_energy_recycled"]
    assert not result["pump_clock_and_reservoir_autonomous"]


def test_saved_continuum_artifact_matches_current_source_dependencies():
    import hashlib
    import json
    saved = json.loads((HERE / "continuum_bath.json").read_text(encoding="utf-8"))
    assert "refinement" in saved and "interleaved_work" in saved["refinement"]
    for relative, expected in saved["source_sha256"].items():
        assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected
