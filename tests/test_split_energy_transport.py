"""이차 에너지 전달을 직접 에너지 평가와 저장된 분할 공분산으로 대조한다."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("split_energy_transport_under_test", HERE / "split_energy_transport.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_energy_is_preserved_for_general_input_form_and_displacements(k):
    rng = np.random.default_rng(930 + k)
    base = rng.normal(size=(2*k, 2*k))
    initial_energy = base.T @ base + np.eye(2*k)
    s = module.source.source_dilation(k)
    final_energy = module.energy_transport(s, initial_energy)
    assert np.linalg.eigvalsh(final_energy)[0] > 0
    for state in rng.normal(size=(5, 2*k)):
        output = s @ state
        assert output @ final_energy @ output == pytest.approx(state @ initial_energy @ state, rel=1e-12)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_first_split_independence_and_second_split_witness(k):
    energy = module.local_energy(k)
    assert energy[0, 2] == pytest.approx((3-k)/(2*k*k), abs=1e-14)
    assert energy[1, 3] == pytest.approx((k-3)/(k-1), abs=1e-14)
    assert (module.independent_child_residual(energy) < 1e-12) == (k == 3)
    if k == 3:
        np.testing.assert_allclose(energy, np.diag([1/3, 3]*3), atol=1e-13)
        second = module.local_energy(3, parent=(1/3, 3))
        assert second[0, 2] == pytest.approx(-2/27)
        assert second[1, 3] == pytest.approx(2.)


@pytest.mark.parametrize("k,depth", [(2, 2), (3, 2), (3, 3), (4, 2)])
def test_tree_energy_reproduces_original_sampler_and_fixed_energy_bound(k, depth):
    energy, transform = module.frontier_energy(k, depth)
    q = module.source.frontier_position_covariance(k, depth)
    covariance = .5*transform @ transform.T
    np.testing.assert_allclose(np.linalg.solve(energy, np.eye(len(energy)))/2, covariance, atol=2e-12)
    np.testing.assert_allclose(covariance[::2, ::2], q, atol=2e-12)
    bound = module.source.frontier_resource_spectrum(k, depth)[3]
    assert (np.trace(covariance)-k**depth)/2 == pytest.approx(float(bound))
    assert module.independent_child_residual(energy) > .1


def test_hamiltonian_change_and_state_change_are_distinct_controls():
    matched = module.local_energy(3, parent=(1/3, 3), ancilla=(1/3, 3))
    np.testing.assert_allclose(matched, np.diag([1/9, 9]*3), atol=1e-13)
    vacuum, ground = module.ancilla_controls()
    assert vacuum["contrast_q_noise"] == pytest.approx(1.5)
    assert vacuum["ancilla_excess_energy"] == pytest.approx(2/3)
    assert ground["contrast_q_noise"] == pytest.approx(4.5)
    assert ground["ancilla_excess_energy"] == pytest.approx(0., abs=1e-14)


def test_noncanonical_or_nonpositive_energy_inputs_rejected():
    for transform, energy in ((2*np.eye(2), np.eye(2)), (np.eye(2), np.diag([1., -1.])),
                               (np.eye(2), np.array([[1., .1], [0., 1.]])),
                               (np.eye(2), np.full((2, 2), np.nan))):
        with pytest.raises(ValueError):
            module.energy_transport(transform, energy)


def test_artifact_matches_current_sources_and_retains_physical_limits():
    result = json.loads((HERE / "split_energy_transport.json").read_text(encoding="utf-8"))
    for name, digest in result["source_hashes"].items():
        path = module.source.SPLIT_SOURCE if name == module.source.SPLIT_SOURCE.name else HERE / name
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("transport_is_autonomous_generator", "hamiltonian_switch_work_derived",
                 "bath_coupling_derived", "physical_spatial_locality_tested",
                 "branching_three_implies_spatial_dimension_three", "common_metric_selection_proved",
                 "fixed_free_energy_preparation_bound_removed"):
        assert result[name] is False
