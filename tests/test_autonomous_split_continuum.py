"""연속 방출 분할의 스펙트럼, 실제 부문 진화와 에너지 보존을 독립 검사한다."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "autonomous_split_continuum_under_test", HERE / "autonomous_split_continuum.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_full_child_sector_and_reference_follow_the_same_autonomous_hamiltonian(k):
    delta, gap, strength, modes = (k-1)/2, (k-1)/2+1, .25, 8
    old = module.split.model(k, maximum=1, battery_gap=gap)
    parent_e = old["oscillator"][old["parent_indices"]]
    child_e = old["oscillator"][old["child_start"]::2]
    nparent, nchild = len(parent_e), len(child_e)
    v = np.zeros((nchild, nparent))
    for n in range(nparent):
        v[old["occupations"].index((n,)+(0,)*(k-1)), n] = 1
    star = module.finite_star(1., strength, modes)
    nodes, couplings = np.diag(star)[1:], star[0, 1:]
    oscillator = np.r_[parent_e, np.repeat(child_e, modes)]
    battery = np.r_[np.full(nparent, gap), np.zeros(nchild*modes)]
    environment = np.r_[np.zeros(nparent), np.tile(nodes, nchild)]
    interaction = np.zeros((len(oscillator),)*2)
    interaction[nparent:, :nparent] = np.kron(v, couplings[:, None])
    interaction[:nparent, nparent:] = interaction[nparent:, :nparent].T
    h = np.diag(oscillator+battery+environment)+interaction
    energies, vectors = np.linalg.eigh(h)
    clock_e, clock_v = np.linalg.eigh(star)
    rng = np.random.default_rng(400+k)
    amplitudes = rng.normal(size=(nparent, 2))+1j*rng.normal(size=(nparent, 2))
    amplitudes /= np.linalg.norm(amplitudes)
    initial = np.zeros((len(h), 2), dtype=complex)
    initial[:nparent] = amplitudes
    initial_probs = np.sum(abs(initial)**2, axis=1)
    for tau in (.37, 3.12):
        final = (vectors*np.exp(-1j*energies*tau)) @ vectors.T @ initial
        clock = clock_v @ (clock_v[0]*np.exp(-1j*clock_e*tau))
        evolved_data = np.exp(-1j*(parent_e+delta)*tau)[:, None]*amplitudes
        expected = np.zeros_like(final)
        expected[:nparent] = clock[0]*evolved_data
        expected[nparent:] = np.einsum("jr,x->jxr", v @ evolved_data, clock[1:]).reshape(-1, 2)
        np.testing.assert_allclose(final, expected, atol=4e-13)
        probabilities = np.sum(abs(final)**2, axis=1)
        child_probability = probabilities[nparent:].sum()
        assert oscillator @ (probabilities-initial_probs) == pytest.approx(delta*child_probability)
        assert battery @ (probabilities-initial_probs) == pytest.approx(-gap*child_probability)
        finite = module.finite_response(1., strength, tau, modes)
        assert environment @ probabilities == pytest.approx(finite["outgoing_energy"], abs=2e-12)
        int_energy = np.vdot(final, interaction @ final).real
        assert int_energy == pytest.approx(finite["interaction_energy"], abs=2e-12)
        assert abs((oscillator+battery) @ (probabilities-initial_probs)
                   +environment @ probabilities+int_energy) < 3e-12
        # 환경을 버린 자식 상태는 참조 얽힘을 포함한 V 상태의 확률 가중 사영이다.
        child = final[nparent:].reshape(nchild, modes, 2)
        reduced = np.einsum("jxr,kxs->jrks", child, child.conjugate()).reshape(2*nchild, 2*nchild)
        target = (v @ evolved_data).reshape(-1)
        np.testing.assert_allclose(reduced, child_probability*np.outer(target, target.conjugate()),
                                   atol=3e-13)
    # V의 상에 속하지 않는 보조 들뜸은 방출 결합과 연결되지 않는다.
    dark = nparent+modes*old["occupations"].index((0, 1)+(0,)*(k-2))
    assert np.linalg.norm(interaction[:, dark]) == 0


@pytest.mark.parametrize("tau", [.25, 1., 3.])
@pytest.mark.parametrize("endpoint", [.3, 1.])
def test_continuum_fourier_response_matches_independent_finite_environment(tau, endpoint):
    actual = module.response(battery_gap=1+endpoint, tau=tau)
    finite = module.finite_response(endpoint, .25, tau, 96)
    assert complex(*actual["amplitude"]) == pytest.approx(complex(*finite["amplitude"]), abs=2e-9)
    assert actual["outgoing_energy"] == pytest.approx(finite["outgoing_energy"], abs=2e-9)
    assert actual["interaction_energy"] == pytest.approx(finite["interaction_energy"], abs=2e-9)
    assert actual["energy_balance_residual"] < 2e-13
    assert actual["quadrature_error_estimate"] < 1e-8


def test_long_time_continuum_sample_has_retained_children_and_paid_emission():
    initial = module.response(tau=0.)
    late = module.response(tau=100.)
    assert initial["child_probability"] == pytest.approx(0)
    assert initial["outgoing_energy"] == pytest.approx(0)
    assert late["parent_probability"] < 1e-7
    assert late["outgoing_energy"] == pytest.approx(1., abs=1e-6)
    assert abs(late["interaction_energy"]) < 1e-6
    assert late["battery_energy_change"] == pytest.approx(-2., abs=1e-6)
    # 이 유한 시간 표본은 문서의 스펙트럼 증명을 대신하지 않는다.


@pytest.mark.parametrize("strength", [.25, .5, 1.])
def test_gap_only_battery_has_a_bound_atom_not_full_asymptotic_transfer(strength):
    result = module.gap_only_bound_state(strength)
    values, vectors = np.linalg.eigh(module.finite_star(0., strength, 128))
    assert np.count_nonzero(values < 0) == 1
    assert result["bound_energy"] == pytest.approx(values[0], abs=2e-8)
    assert result["endpoint_atom_mass"] == pytest.approx(vectors[0, 0]**2, abs=1e-6)
    assert result["secular_residual"] < 1e-12
    assert .1 < result["asymptotic_parent_probability"] < 1


@pytest.mark.parametrize("gap", [0., 1., 1.25])
def test_strict_decay_branch_does_not_silently_accept_threshold_or_bound_state(gap):
    with pytest.raises(ValueError):
        module.response(3, gap, .25)


def test_artifact_provenance_and_physical_scope():
    data = json.loads((HERE / "autonomous_split_continuum.json").read_text(encoding="utf-8"))
    for name, digest in data["source_hashes"].items():
        path = module.split.source.SPLIT_SOURCE if name == module.split.source.SPLIT_SOURCE.name else HERE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert data["max_checked_residual"] < 1e-8
    assert data["conditional_asymptotic_child_probability_one"]
    for key in ("stationary_global_state_proved", "physical_pointer_label_record_derived",
                "autonomous_initial_preparation_derived", "battery_recharged_with_children_retained",
                "global_tree_hamiltonian_constructed", "CE_local_action_derived",
                "common_metric_selection_proved", "finite_environment_proves_infinite_time_limit"):
        assert not data[key]
