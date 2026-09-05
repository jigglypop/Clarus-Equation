"""환경의 에너지 이동·기억을 독립 검산한다."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
old_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("bath_recycling_under_test", HERE/"bath_recycling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = old_path


@pytest.mark.parametrize("count,alpha", [(1,.2), (2,.7), (4,.1), (8,2.)])
def test_energy_map_inverse_jacobian_and_conservation(count, alpha):
    initial = np.r_[np.linspace(.2,1.2,count),1.5]
    def transform(values):
        mapped = module.positive_energy_map(values[:-1],values[-1],alpha)
        return np.r_[mapped["bath"],mapped["battery"]]
    result = module.positive_energy_map(initial[:-1],initial[-1],alpha)
    final = transform(initial)
    inverse = module.positive_energy_map(final[:-1],final[-1],1/alpha)
    assert np.r_[inverse["bath"],inverse["battery"]] == pytest.approx(initial, abs=2e-14)
    assert np.sum(final) == pytest.approx(np.sum(initial), abs=2e-14)
    step = 1e-5
    independent = np.column_stack([
        (transform(initial+step*axis)-transform(initial-step*axis))/(2*step)
        for axis in np.eye(count+1)])
    assert np.linalg.det(independent) == pytest.approx(math.exp(result["log_jacobian"]), rel=2e-8)
    if alpha < 1:
        assert np.all(final[:-1]<initial[:-1])
        assert final[-1]>initial[-1]


def test_vacuum_sector_is_identity_and_composition_uses_alpha_product():
    assert module.positive_energy_map([],1.5,.1) == {
        "bath":[],"battery":1.5,"log_jacobian":0.}
    values=np.array([.3,.8,1.1])
    first=module.positive_energy_map(values,1.5,.4)
    second=module.positive_energy_map(first["bath"],first["battery"],.3)
    direct=module.positive_energy_map(values,1.5,.12)
    assert second["bath"] == pytest.approx(direct["bath"], abs=2e-14)
    assert second["battery"] == pytest.approx(direct["battery"], abs=2e-14)


@pytest.mark.parametrize("alpha",[0.,-1.,float("nan"),float("inf")])
def test_invalid_energy_map_parameters_are_rejected(alpha):
    with pytest.raises(ValueError):
        module.positive_energy_map([1.],1.5,alpha)


def test_memory_and_echo_keep_cross_covariance_and_switching_energy():
    result=module.memory_control()
    assert result["system_bath_cross_covariance_norm"]>.1
    assert abs(result["same_environment"][0]["system_number"]-
               result["fresh_second_system_number"])>.01
    assert max(row["balance_residual"] for row in result["same_environment"])<2e-12
    assert result["echo_system_number"] == pytest.approx(1/3, abs=2e-13)
    assert result["echo_remaining_bath_number"]<2e-25
    assert abs(result["echo_phase_free_energy_change"])<2e-12
    assert abs(result["echo_net_switch_work_over_Estar"])<2e-12


def test_all_photon_recovery_matches_independent_count_sampling():
    # 두 입력 압축 모드의 총 photon 수는 2m, P(m)=3/(4^(m+1)).
    packet=module.emitted_packet(128)
    rng=np.random.default_rng(902173)
    trials=250000
    input_number=2*(rng.geometric(.75,size=trials)-1)
    emitted_number=rng.binomial(input_number,packet["emitted_probability"])
    energies=rng.choice(packet["energies"],size=int(np.sum(emitted_number)),
                        p=packet["probabilities"])
    totals=np.bincount(np.repeat(np.arange(trials),emitted_number),weights=energies,minlength=trials)
    batteries=rng.uniform(1.,2.,size=trials)
    alpha=.1
    gains=(1-alpha)*batteries*totals/(batteries+alpha*totals)
    independent_mean=float(np.mean(gains))
    standard_error=float(np.std(gains,ddof=1)/math.sqrt(trials))
    result=module.recovery_energy(alpha)
    assert abs(result["battery_energy_gain_over_Estar"]-independent_mean)<6*standard_error
    assert not result["photon_number_truncation_used"]
    assert result["bath_vacuum_probability_preserved"]<.9


def test_energy_integral_converges_without_claiming_reusable_work():
    result=module.run()
    assert result["max_64_128_energy_difference"]<2e-10
    assert result["laplace_64_128_difference"]<2e-8
    assert result["bath_energy_difference_from_continuum_integral"]<2e-10
    gains=[row["battery_energy_gain_over_Estar"] for row in result["recovery"]]
    assert gains[0]<gains[1]<gains[2]
    assert module.recovery_energy(1.)["battery_energy_gain_over_Estar"]==0.
    for row in result["recovery"]:
        assert row["remaining_bath_energy_over_Estar"]>0
        assert row["final_battery_mean_over_Estar"]+row["remaining_bath_energy_over_Estar"] == pytest.approx(
            1.5+row["initial_bath_energy_over_Estar"],abs=2e-14)
    assert not result["conditional_results"]["battery_low_entropy_or_reusable_work_proved"]
    assert not result["conditional_results"]["local_causal_CE_action_derived"]
    assert not result["physical_candidate_adopted"]


def test_artifact_source_hashes_match_current_files():
    import hashlib
    import json
    saved=json.loads((HERE/"bath_recycling.json").read_text(encoding="utf-8"))
    for name,expected in saved["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected


@pytest.mark.parametrize("survival", [0., .25, .5, 1.])
def test_next_split_vacuum_bound_matches_actual_gaussian_source(survival):
    source = sys.modules["split_quantum_source"]
    dilation = source.source_dilation(3)
    undo_mixing = np.kron(source.mode_basis(3).T, np.eye(2))
    modes = undo_mixing @ (.5*dilation @ dilation.T) @ undo_mixing.T
    environment = .5*np.eye(4)+(1-survival)*(modes[2:,2:]-.5*np.eye(4))
    independent_vacuum = 1/math.sqrt(np.linalg.det(environment+.5*np.eye(4)))
    result = module.next_split_bounds(.1, survival=survival)
    assert result["joint_vacuum_probability"] == pytest.approx(independent_vacuum, abs=2e-15)
    assert result["joint_resource_trace_distance_lower_bound"] == pytest.approx(
        independent_vacuum-.75, abs=2e-15)
    assert not result["zero_bound_proves_preparation"]


@pytest.mark.parametrize("alpha,first_missing", [(.5,1), (.1,5), (.01,50), (np.nextafter(.5,0),2)])
def test_finite_battery_support_excludes_unbounded_squeezed_tail(alpha, first_missing):
    result = module.next_split_bounds(float(alpha))
    assert result["battery_only_first_missing_pair"] == first_missing
    assert result["battery_only_tail_exact"] == {"base":4,"exponent":-first_missing}
    assert result["battery_only_trace_distance_lower_bound_float"] == pytest.approx(4.**(-first_missing))
    assert result["finite_alpha_battery_only_exact_preparation_excluded"]


def test_energy_projection_bound_with_coherence_and_entangled_reference():
    # 초기 자원은 기존 계와 얽히고 에너지 사이 결맞음도 갖는다.
    rng = np.random.default_rng(40152)
    resource_energies = np.array([1,2,5,6,9])
    target_energies = np.array([0,2,4,6,8])
    width = len(target_energies)
    amplitudes = rng.normal(size=(3,5))+1j*rng.normal(size=(3,5))
    amplitudes[:,:2] *= math.sqrt(.8/np.sum(abs(amplitudes[:,:2])**2))
    amplitudes[:,2:] *= math.sqrt(.2/np.sum(abs(amplitudes[:,2:])**2))
    resource = amplitudes.T @ amplitudes.conj()
    total_energy = (resource_energies[:,None]+target_energies).ravel()
    unitary = np.zeros((25,25), dtype=complex)
    for energy in np.unique(total_energy):
        indices = np.flatnonzero(total_energy == energy)
        raw = rng.normal(size=(len(indices),len(indices)))+1j*rng.normal(size=(len(indices),len(indices)))
        block, _ = np.linalg.qr(raw)
        unitary[np.ix_(indices,indices)] = block
    high_target = np.diag(np.tile(target_energies>=4,5))
    pulled_back = (unitary.conj().T @ high_target @ unitary)[::width,::width]
    high_input = np.diag(resource_energies>=4)
    assert np.max(np.linalg.eigvalsh(pulled_back-high_input)) < 2e-14
    assert float(np.trace(pulled_back @ resource).real) <= .2+2e-14
    assert np.linalg.norm(resource-np.diag(np.diag(resource))) > .1


def test_moment_bound_is_analytic_and_pooling_gate_is_only_necessary():
    from fractions import Fraction
    result = module.survival_moment_bound()
    assert result["survival_probability_lower_bound"] == 1/16
    assert result["joint_resource_trace_distance_lower_bound"] == float(Fraction(1,1364))
    assert not result["uses_numerical_spectral_integral"]
    _, h, _, _ = module.star_state(32)
    independent_variance = float((h @ h)[0,0]-h[0,0]**2)
    assert independent_variance == pytest.approx(1.5, abs=1e-12)
    assert module.next_split_bounds(.1)["joint_resource_trace_distance_lower_bound"] > float(Fraction(1,1364))
    pooled = module.next_split_bounds(.1, resource_copies=2)
    assert pooled["joint_resource_trace_distance_lower_bound"] == 0.
    assert not pooled["zero_bound_proves_preparation"]


def test_pooled_photon_sampling_recovers_independent_energy_and_vacuum_moments():
    result = module.pooled_tail_probe(2, trials=50000)
    assert abs(result["mean_resource_energy_over_Estar"]-result["expected_mean_resource_energy_over_Estar"]) < 6*result["mean_standard_error"]
    probability = result["joint_vacuum_probability"]
    vacuum_error = math.sqrt(probability*(1-probability)/result["trials"])
    assert abs(result["joint_vacuum_frequency"]-probability) < 6*vacuum_error
    assert not result["photon_number_truncation_used"]
    assert not result["coherent_preparation_unitary_constructed"]
    assert result["numerical_necessary_condition_screen_only"]


@pytest.mark.parametrize("copies", [1,2,4,8])
def test_pooling_cannot_exactly_replace_or_multiply_the_same_number_of_sources(copies):
    replacement = module.pooled_replacement_bound(copies, copies)
    branching = module.pooled_replacement_bound(copies, 3*copies)
    assert replacement["trace_distance_lower_bound"] > 0.
    assert branching["trace_distance_lower_bound"] > replacement["trace_distance_lower_bound"]
    assert 0 < replacement["exact_count_ratio_necessary_upper_bound"] < 1
    assert not replacement["fixed_error_asymptotic_capacity_proved"]
    exhausted_source = module.pooled_replacement_bound(copies, copies, survival=0.)
    assert exhausted_source["trace_distance_lower_bound"] == 0.
    assert not exhausted_source["zero_bound_proves_preparation"]


@pytest.mark.parametrize("nodes", [32,128])
def test_retired_source_transfer_preserves_an_entangled_reference(nodes):
    _, unitary = module.retired_transfer_unitary(nodes)
    modes = len(unitary)
    symplectic = np.empty((2*modes,2*modes))
    symplectic[0::2,0::2] = symplectic[1::2,1::2] = unitary.real
    symplectic[0::2,1::2] = -unitary.imag
    symplectic[1::2,0::2] = unitary.imag
    full = np.eye(2*(modes+1))
    full[2:,2:] = symplectic
    initial = np.eye(len(full))/2
    variance, correlation = math.cosh(.8)/2, math.sinh(.8)/2
    initial[:4,:4] = np.array([[variance,0,correlation,0],[0,variance,0,-correlation],
                              [correlation,0,variance,0],[0,-correlation,0,variance]])
    final = full @ initial @ full.T
    retained = [0,1,len(full)-2,len(full)-1]
    retired = list(range(2,len(full)-2))
    assert final[np.ix_(retained,retained)] == pytest.approx(initial[:4,:4], abs=2e-12)
    assert final[np.ix_(retired,retired)] == pytest.approx(np.eye(len(retired))/2, abs=2e-12)
    assert np.linalg.norm(final[np.ix_(retained,retired)]) < 2e-12
    result = module.retired_source_transfer(nodes)
    assert result["one_particle_transfer_residual"] < 2e-12
    assert result["new_mode_number"] == pytest.approx(1/3, abs=2e-12)
    assert abs(result["free_energy_change_over_Estar"]) < 2e-12
    assert abs(result["ideal_net_switch_work_over_Estar"]) < 2e-12
    assert result["existing_system_participates"]
    assert result["additional_source_copies_created"] == 0
    assert not result["autonomous_clock_controller_cycle_closed"]
