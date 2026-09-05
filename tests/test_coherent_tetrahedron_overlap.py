"""Independent representation, Haar, measure, and shape-localization checks."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
spec = importlib.util.spec_from_file_location("coherent_tetrahedron_under_test", HERE/"coherent_tetrahedron_overlap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("n", [1, 2, 4, 16])
def test_coupling_is_orthogonal_and_has_exact_singlet_coefficients(n):
    blocks = module.coupling(n)
    transform = np.concatenate([block.reshape(2*k+1, -1) for k, block in enumerate(blocks)])
    np.testing.assert_allclose(transform @ transform.T, np.eye((n+1)**2), atol=2e-13)
    expected = np.zeros((n+1, n+1))
    for a in range(n+1):
        expected[a, n-a] = (-1)**(n-a)/math.sqrt(n+1)
    np.testing.assert_allclose(blocks[0][0], expected, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tensor_basis_is_orthonormal_singlet_and_pair_casimir_has_correct_eigenvalues(n):
    basis = module.invariant_basis(n).reshape(n+1, -1)
    np.testing.assert_allclose(basis @ basis.T, np.eye(n+1), atol=1e-13)
    first, second, volume, closure = module.shape_operators(n)
    assert closure < 1e-12
    j = n/2
    expected = [k*(k+1)/2-j*(j+1) for k in range(n+1)]
    np.testing.assert_allclose(first, np.diag(expected), atol=1e-12)
    np.testing.assert_allclose(second, second.conj().T, atol=1e-12)
    np.testing.assert_allclose(volume, volume.conj().T, atol=1e-12)
    np.testing.assert_allclose(first @ second-second @ first, 1j*volume, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 4, 16])
def test_group_integral_matches_coupled_projection_for_nonclosed_normals(n):
    rng = np.random.default_rng(451)
    normals = rng.normal(size=(8, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    left, right = normals[:4], normals[4:]
    a, norm = module.intertwiner(n, left, normalize=False)
    b, _ = module.intertwiner(n, right, normalize=False)
    assert norm > 0
    actual = module.haar_projected_overlap(n, left, right)
    assert actual == pytest.approx(np.vdot(a, b), abs=2e-13)


def test_regular_normals_close_and_rotations_preserve_the_quantum_ray():
    normals = module.shape_normals(*module.REGULAR)
    np.testing.assert_allclose(normals.sum(axis=0), 0., atol=1e-14)
    gram = normals @ normals.T
    np.testing.assert_allclose(gram, np.eye(4)*4/3-np.ones((4, 4))/3, atol=1e-14)
    q, _ = np.linalg.qr(np.random.default_rng(718).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    a, norm_a = module.intertwiner(4, normals)
    b, norm_b = module.intertwiner(4, normals @ q.T)
    assert abs(np.vdot(a, b))**2 == pytest.approx(1., abs=1e-12)
    assert norm_a == pytest.approx(norm_b, abs=1e-13)


@pytest.mark.parametrize("n", [2, 4, 16])
def test_shape_precision_matches_independent_fidelity_differences(n):
    precision, geometry = module.shape_precision(n)
    assert min(np.linalg.eigvalsh(precision)) > 0
    assert min(np.linalg.eigvalsh(geometry)) > -1e-10
    center = module.shape_state(n, module.REGULAR)
    direction = np.array([.6, -.8])
    h = 2e-4
    def cost(delta):
        displaced = module.shape_state(n, module.REGULAR+delta*direction)
        return -math.log(abs(np.vdot(center, displaced))**2)
    measured = (cost(h)+cost(-h))/h**2
    assert measured == pytest.approx(direction @ precision @ direction, rel=2e-6, abs=2e-6)
    half, _ = module.shape_precision(n, 5e-5)
    np.testing.assert_allclose(half, precision, atol=1e-8)
    np.testing.assert_allclose(module.leading_shape_precision(n), np.diag([2*n, n/3]), atol=1e-12)


def test_finite_spin_samples_narrow_and_approach_the_leading_saddle_comparison():
    spins, precisions, errors, fidelities = [], [], [], []
    for n in (2, 4, 8, 16):
        precision, _ = module.shape_precision(n)
        leading = module.leading_shape_precision(n)
        spins.append(n/2)
        precisions.append(np.linalg.eigvalsh(precision))
        errors.append(np.linalg.norm(precision-leading)/np.linalg.norm(leading))
        a = module.shape_state(n, module.REGULAR)
        b = module.shape_state(n, module.REGULAR+np.array([.12, .16]))
        fidelities.append(abs(np.vdot(a, b))**2)
    assert np.all(np.diff(precisions, axis=0) > 0)
    assert np.all(np.diff(errors) < 0)
    assert np.all(np.diff(fidelities) < 0)
    assert errors[-1] < .05


@pytest.mark.parametrize("n", [1, 2, 4, 16])
def test_positive_single_spin_quadrature_resolves_identity(n):
    _, weights, frame = module.single_spin_frame(n)
    assert min(weights) > 0
    assert sum(weights) == pytest.approx(n+1, abs=1e-12)
    np.testing.assert_allclose(frame, np.eye(n+1), atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
def test_projected_frame_measure_composes_amplitudes_but_not_squared_moduli(n):
    row = module.frame_composition(n)
    assert row["invariant_frame_residual"] < 1e-12
    assert row["normalized_measure_total"] == pytest.approx(n+1, abs=1e-12)
    assert row["amplitude_composition_residual"] < 1e-12
    assert row["omitting_projection_norm_residual"] > 1.
    assert row["direct_fidelity"]-row["squared_moduli_composition"] > .3
    assert row["measurement_channel_trace"] == pytest.approx(1., abs=1e-12)
    assert 1/(n+1)-1e-12 <= row["measurement_channel_purity"] < .9
    assert row["instrument_uses_this_finite_frame"]
    if n == 2:
        assert row["omitting_norm_after_trace_rescaling_residual"] > .1


@pytest.mark.parametrize("n", [1, 2, 4])
def test_shape_observables_have_nonzero_uncertainty_floor(n):
    row = module.operator_case(n)
    assert row["commutator_triple_product_residual"] < 1e-12
    assert row["robertson_lower_bound"] > 0
    assert row["robertson_determinant"] >= row["robertson_lower_bound"]-1e-12


@pytest.mark.parametrize("bad", [0, True, 1.5, 17])
def test_invalid_spin_is_rejected(bad):
    with pytest.raises(ValueError):
        module.spin_coherent(bad, [0., 0., 1.])


def test_invalid_or_zero_projection_states_are_not_silently_normalized():
    with pytest.raises(ValueError):
        module.intertwiner(2, np.tile([0., 0., 1.], (4, 1)))
    with pytest.raises(ValueError):
        module.spin_coherent(2, [0., 0., .5])
    with pytest.raises(ValueError):
        module.shape_normals(module.REGULAR[0], 0.)
    value, norm = module.intertwiner(2, np.tile([0., 0., 1.], (4, 1)), normalize=False)
    assert norm == 0.
    np.testing.assert_array_equal(value, 0.)


def test_artifact_hash_and_scope_are_current():
    report = json.loads((HERE/"coherent_tetrahedron_overlap.json").read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    scope = report["scope"]
    assert scope["full_complex_overlap_retained"]
    assert scope["normalized_projection_measure_retained"]
    for name in ("closed_shape_slice_used_as_full_frame_measure",
                 "asymptotic_width_law_rigorously_proven_here",
                 "physical_length_resolution_derived_from_ce",
                 "regge_action_phase_or_refinement_dynamics_derived",
                 "microscopic_split_merge_energy_budget_derived",
                 "stationary_common_metric_sector_selected", "continuum_einstein_limit_derived"):
        assert scope[name] is False
