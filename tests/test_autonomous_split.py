"""기존 분할 파동함수, 정확한 부문 진화, 배터리와 역병합을 독립 검산한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("autonomous_split_under_test", HERE / "autonomous_split.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_coordinate_differential_energy_matches_transported_fock_frame(k):
    for occupation in range(5):
        for offset in (0., .43):
            result = module.coordinate_eigen_check(k, occupation, np.linspace(-1., .8, k)+offset)
            assert result["residual"] < 1e-13


@pytest.mark.parametrize("k", [2, 3, 4])
def test_parent_reference_entanglement_and_battery_transfer_return_exactly(k):
    data = module.model(k)
    size = len(data["free"])
    p, c = data["parent_indices"], data["child_indices"]
    rng = np.random.default_rng(k+710)
    amplitudes = rng.normal(size=(len(p), 2)) + 1j*rng.normal(size=(len(p), 2))
    amplitudes /= np.linalg.norm(amplitudes)
    state = np.zeros((size, 2), dtype=complex)
    state[p] = amplitudes
    tau = math.pi/(2*data["coupling"])
    final = module.propagator(data, tau) @ state
    target = np.zeros_like(state)
    target[c] = -1j*np.exp(-1j*np.diag(data["free"])[p]*tau)[:, None]*amplitudes
    np.testing.assert_allclose(final, target, atol=2e-13)
    returned = module.propagator(data, 2*tau) @ state
    target[:] = 0
    target[p] = -np.exp(-2j*np.diag(data["free"])[p]*tau)[:, None]*amplitudes
    np.testing.assert_allclose(returned, target, atol=3e-13)
    np.testing.assert_allclose(data["free"] @ data["w"], data["w"] @ data["free"], atol=1e-14)
    assert np.sum(data["battery"][:, None]*np.abs(final)**2) < 1e-25
    assert np.sum(data["battery"][:, None]*np.abs(returned)**2) == pytest.approx(data["battery_gap"])


@pytest.mark.parametrize("gap,expected", [(0., .2), (.5, .5), (1., 1.), (2., .2)])
def test_detuning_restricts_transfer_and_interaction_accounts_for_energy(gap, expected):
    result = module.evolution_check(3, battery_gap=gap)
    assert result["maximum_probability"] == pytest.approx(expected)
    assert result["rows"][2]["child_probability"] == pytest.approx(expected)
    assert max(v for row in result["rows"] for v in row["residuals"].values()) < 1e-12
    if gap != 1:
        assert result["free_energy_commutator_norm"] > .1
        assert abs(result["rows"][2]["interaction_energy"]) > .1


def test_ancilla_excitation_is_dark_and_only_split_image_merges():
    data = module.model(3)
    dark_index = data["child_start"]+2*data["occupations"].index((0, 1, 0))
    image_index = data["child_indices"][0]
    state = np.zeros(len(data["free"]), dtype=complex)
    state[[dark_index, image_index]] = 1/math.sqrt(2)
    final = module.propagator(data, math.pi/(2*data["coupling"])) @ state
    assert np.sum(np.abs(final[:data["child_start"]])**2) == pytest.approx(.5)
    assert abs(final[dark_index])**2 == pytest.approx(.5)
    assert np.linalg.norm(data["w"].T[:, dark_index]) == 0


def test_occupation_window_is_exactly_invariant_not_a_gaussian_fock_cutoff():
    small, large = module.model(3, maximum=2), module.model(3, maximum=3)
    indices = list(range(2*(small["maximum"]+1)))
    indices += [large["child_start"]+2*large["occupations"].index(n)+b
                for n in small["occupations"] for b in (0, 1)]
    outside = sorted(set(range(len(large["free"])))-set(indices))
    np.testing.assert_allclose(large["hamiltonian"][np.ix_(indices, indices)], small["hamiltonian"])
    assert np.linalg.norm(large["hamiltonian"][np.ix_(indices, outside)]) == 0


@pytest.mark.parametrize("k,depth", [(2, 4), (3, 3), (4, 2)])
def test_gap_ledger_counts_new_modes_and_does_not_replace_fixed_energy(k, depth):
    from fractions import Fraction
    expected = sum(Fraction(k-1, 2)*k**d for d in range(depth))
    assert module.gap_ledger(k, depth) == expected
    assert module.source.frontier_resource_spectrum(k, depth)[3] != expected


@pytest.mark.parametrize("options", [{"coupling": 0}, {"coupling": float("nan")},
                                    {"battery_gap": -1}, {"maximum": True}])
def test_invalid_parameters_rejected(options):
    with pytest.raises(ValueError):
        module.model(3, **options)


def test_artifact_sources_and_physical_boundaries():
    result = json.loads((HERE/"autonomous_split.json").read_text(encoding="utf-8"))
    for name, digest in result["source_hashes"].items():
        path = module.source.SPLIT_SOURCE if name == module.source.SPLIT_SOURCE.name else HERE/name
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["time_independent_hamiltonian_constructed"]
    assert result["reversible_cycle_returns_parent_and_battery"]
    for field in ("bare_fock_gaussian_truncation_used", "cycle_leaves_net_children_or_records",
                  "split_output_is_stationary", "merges_arbitrary_child_states",
                  "global_tree_hamiltonian_constructed", "fixed_free_energy_bound_replaced",
                  "CE_local_action_derived", "bath_coupling_derived", "common_metric_selection_proved"):
        assert result[field] is False
