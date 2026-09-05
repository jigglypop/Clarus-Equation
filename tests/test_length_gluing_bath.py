"""실제 길이 접착, 직접 유니터리, 유한 분해능의 에너지 비용을 검산한다."""

import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("length_gluing_bath_under_test", HERE / "length_gluing_bath.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original_path


@pytest.mark.parametrize("depth", [1,2])
def test_face_local_constraints_have_the_existing_global_edge_kernel(depth):
    data, values, _, dark = module.spectrum(depth)
    c = data["constraint"]
    old = module.reference().gluing_rows(data["cells"])
    old_kernel = np.eye(c.shape[1])-old.T @ np.linalg.solve(old @ old.T, old)
    np.testing.assert_allclose(c @ old_kernel, 0, atol=1e-12)
    n = 5**depth
    assert dark == (5*n+35)//4
    assert len(values)-dark == 35*(n-1)//4
    assert len(c) == 15*(n-1)
    np.testing.assert_allclose(old_kernel, data["kernel_basis"] @ data["kernel_basis"].T, atol=1e-12)
    result = module.graph_case(depth)
    assert result["internal_displacement_rank"] == n-1
    assert result["displacement_in_kernel_residual"] < 1e-12


def test_first_refinement_spectrum_matches_independent_complete_owner_graphs():
    _, values, _, dark = module.spectrum(1)
    # 원래 변 열 개는 세 소유자, 새 변 다섯 개는 네 소유자를 갖는다.
    expected = np.r_[np.full(20,3.), np.full(15,4.)]
    np.testing.assert_allclose(values[dark:], expected, atol=1e-12)


def _face_gram(lengths):
    edges = dict(zip(combinations(range(4),2), lengths))
    return np.array([
        edges[(0,i)]**2 if i==j else (edges[(0,i)]**2+edges[(0,j)]**2-edges[(i,j)]**2)/2
        for i,j in ((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))
    ])


def test_face_length_matching_is_equivalent_to_induced_metric_matching():
    data = module.length_gluing(1)
    _, _, face = data["faces"][0]
    lengths = np.array([
        np.linalg.norm(data["points"][left]-data["points"][right])
        for left,right in combinations(face,2)
    ])
    jacobian = module.face_gram_jacobian(lengths)
    step = 1e-5
    numerical = np.column_stack([
        (_face_gram(lengths+step*direction)-_face_gram(lengths-step*direction))/(2*step)
        for direction in np.eye(6)
    ])
    np.testing.assert_allclose(jacobian,numerical,atol=1e-10)
    assert abs(np.linalg.det(jacobian)) == pytest.approx(8*np.prod(lengths),rel=1e-12)


def test_constant_volume_does_not_determine_shared_face_shape():
    result = module.volume_preserving_shape_case()
    assert result["cell_volume_ratio"] == pytest.approx(1.,abs=1e-14)
    assert result["initial_length_mismatch_norm"] > 0.01
    assert result["evolved_squared_phase_space_mismatch"] < 1e-9
    assert not result["shared_face_lengths_initially_match"]


def test_wick_position_measure_has_a_pure_canonical_completion():
    covariance = module.wick_preparation()
    omega = np.kron(np.eye(50),np.array([[0.,1.],[-1.,0.]]))
    np.testing.assert_allclose(covariance @ omega @ covariance,omega/4,atol=1e-11)
    assert np.linalg.eigvalsh(covariance).min()>0
    data = module.length_gluing(1)
    initial = np.trace(data["constraint"] @ covariance[0::2,0::2] @ data["constraint"].T)
    assert initial == pytest.approx(162.06716829026382,rel=1e-7)


def test_full_finite_environment_reproduces_continuum_covariance_at_short_time():
    data, _, _, _, x, y, _ = module.channel(1,0.5)
    c, laplacian = data["constraint"], data["laplacian"]
    # 짧은 시간에는 유한 적분 환경과 연속 스펙트럼을 직접 비교할 수 있다.
    nodes, weights = np.polynomial.laguerre.laggauss(8)
    count = c.shape[1]
    coupling = math.sqrt(.5)*np.kron(c,np.sqrt(nodes*weights)[:,None])
    h = np.diag(np.r_[np.zeros(count),np.tile(nodes,len(c))])
    h[:count,:count] = 2*np.eye(count)+.5*laplacian
    h[count:,:count], h[:count,count:] = coupling,coupling.T
    energy,vectors = np.linalg.eigh(h)
    assert energy[0]>0
    u = (vectors*np.exp(-.5j*energy)) @ vectors.T
    # 복소 진폭의 실수 표현을 테스트 안에서 독립 구성한다.
    a = u[:count,:count]
    direct_x = np.empty_like(x)
    direct_x[0::2,0::2] = direct_x[1::2,1::2] = a.real
    direct_x[0::2,1::2],direct_x[1::2,0::2] = -a.imag,a.imag
    np.testing.assert_allclose(direct_x,x,atol=2e-7)
    initial = module.wick_preparation()
    finite_output = direct_x @ initial @ direct_x.T+(np.eye(2*count)-direct_x @ direct_x.T)/2
    np.testing.assert_allclose(finite_output,x @ initial @ x.T+y,atol=2e-7)


def test_long_time_emission_retains_vacuum_length_mismatch():
    result = module.evolve_wick(time=200.)
    assert result["remaining_bright_number"] < 1e-7
    assert result["final_total_mismatch_variance"] == pytest.approx(60.,abs=1e-7)
    assert result["vacuum_covariance_residual"] < 1e-7
    assert not result["quadrature_error_is_rigorous_bound"]


@pytest.mark.parametrize("error", [1.,.1,.01])
def test_finite_resolution_preparation_is_physical_and_beats_equal_allocation(error):
    result = module.resolution_preparation(1,error)
    lam = np.array(result["positive_eigenvalues"])
    q = np.array(result["position_variances"])
    p = np.array(result["momentum_variances"])
    np.testing.assert_allclose(q*p,.25,atol=1e-14)
    assert result["achieved_mean_squared_error"] <= error*(1+1e-12)
    achieved_energy = np.sum((2+.5*lam)*(q+p-1))/2
    assert achieved_energy == pytest.approx(result["product_vacuum_bath_minimum_preparation_energy"])
    assert achieved_energy+1e-10 >= result["universal_total_energy_lower_bound"]
    equal_q = 60*error/np.sum(lam)
    equal_energy = np.sum((2+.5*lam)*(equal_q+1/(4*equal_q)-1))/2
    assert achieved_energy <= equal_energy+1e-10


def test_resolution_energy_diverges_when_exact_equality_is_approached():
    rows = [module.resolution_preparation(1,error) for error in (.01,.001,.0001)]
    energies = [row["universal_total_energy_lower_bound"] for row in rows]
    assert energies[1]>9*energies[0]
    assert energies[2]>9*energies[1]


def test_energy_bound_holds_with_correlated_mixed_gaussian_modes():
    data,values,vectors,dark = module.spectrum(1)
    rng = np.random.default_rng(20260906)
    count = len(values)
    basis,_ = np.linalg.qr(rng.normal(size=(count,count)))
    scales = np.exp(rng.uniform(-2,2,size=count))
    q = (basis*scales) @ basis.T
    shear = rng.normal(size=(count,count))
    shear = (shear+shear.T)/2
    p = np.linalg.inv(q)/4+shear @ q @ shear+.3*np.eye(count)
    bright_q = np.diag(vectors.T @ q @ vectors)[dark:]
    bright_p = np.diag(vectors.T @ p @ vectors)[dark:]
    error = float(values[dark:] @ bright_q)/len(data["constraint"])
    result = module.resolution_preparation(1,error)
    bright_energy = float(np.sum(bright_q+bright_p-1))
    assert bright_energy+1e-10 >= result["universal_total_energy_lower_bound"]


@pytest.mark.parametrize("error", [0.,-1.,float("nan"),float("inf")])
def test_resolution_rejects_invalid_or_exact_zero_error(error):
    with pytest.raises(ValueError):
        module.resolution_preparation(1,error)


def test_artifact_hashes_and_physical_scope_match_current_sources():
    report = json.loads((HERE/"length_gluing_bath.json").read_text(encoding="utf-8"))
    for relative,expected in report["source_sha256"].items():
        assert hashlib.sha256((HERE/relative).read_bytes()).hexdigest()==expected
    status = report["conditional_results"]
    assert status["face_local_length_coupling_constructed"]
    assert status["finite_resolution_squeezed_preparation_constructed"]
    assert not status["exact_quantum_length_equality_obtained"]
    assert not status["canonical_length_kinetic_term_derived_from_regge"]
    assert not status["common_metric_tensor_selected"]
    assert not status["continuum_einstein_equations_derived"]


def test_shared_global_geometry_modes_evolve_freely_without_bath_noise():
    time = 0.7
    data, _, _, dark, x, y, _ = module.channel(1,time)
    kernel = np.kron(data["kernel_basis"],np.eye(2))
    angle = 2*time
    rotation = np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
    np.testing.assert_allclose(x @ kernel,kernel @ np.kron(np.eye(dark),rotation),atol=1e-12)
    np.testing.assert_allclose(y @ kernel,0,atol=1e-12)
