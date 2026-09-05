"""양의 분할 생성자, 유리수 부호 증인과 준비·전환 에너지를 검사한다."""

from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "split_quadratic_generator_under_test", HERE/"split_quadratic_generator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("k", [2, 3])
def test_rational_sign_certificate_isolates_every_simple_unit_circle_pair(k):
    certificate = module.exact_spectral_certificate(k)
    prior = Fraction(-2)
    assert len(certificate["root_intervals"]) == k
    for interval in certificate["root_intervals"]:
        left, right = Fraction(interval["left"]), Fraction(interval["right"])
        assert prior <= left < right < 2
        assert left > -2
        prior = right
        assert interval["signs"][0]*interval["signs"][1] == -1
        for bounds, sign in zip(interval["polynomial_value_bounds"], interval["signs"]):
            lo, hi = map(Fraction, bounds)
            assert lo <= hi
            assert lo > 0 if sign == 1 else hi < 0


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_specified_full_dilation_has_an_expanding_eigenvalue(k):
    result = module.original_dilation_obstruction(k)
    assert result["actual_q_spectral_radius"] >= result["spectral_radius_lower_bound"]-1e-12
    assert result["spectral_radius_lower_bound"] > 1
    if k == 3:
        eigenvalues = np.linalg.eigvals(module.source.source_dilation(k))
        np.testing.assert_allclose(sorted(abs(eigenvalues)),
                                   [1/math.sqrt(3)]*3+[math.sqrt(3)]*3, atol=2e-14)


@pytest.mark.parametrize("k", [2, 3])
def test_positive_generator_changes_full_dilation_but_keeps_input_isometry(k):
    row = module.witness(k)
    omega = np.kron(np.eye(k), module.source.J)
    evolved = expm(omega @ row["generator"])
    original = module.source.source_dilation(k)
    passive = np.linalg.solve(original, evolved)
    assert min(row["generator_eigenvalues"]) > .07
    assert np.linalg.norm(evolved-original) > .1
    np.testing.assert_allclose(passive[:2], np.eye(2*k)[:2], atol=2e-13)
    np.testing.assert_allclose(passive[:, :2], np.eye(2*k)[:, :2], atol=2e-13)
    np.testing.assert_allclose(passive @ passive.T, np.eye(2*k), atol=3e-13)
    np.testing.assert_allclose(passive @ omega @ passive.T, omega, atol=3e-13)


def _bell_characteristic(first, second):
    # (|00>+|11>)/sqrt(2)의 Weyl 특성함수. 진공·한 입자 행렬원소를 직접 쓴다.
    def displacement(v):
        alpha = (1j*v[0]-v[1])/math.sqrt(2)
        return math.exp(-abs(alpha)**2/2)*np.array(
            [[1, -alpha.conjugate()], [alpha, 1-abs(alpha)**2]], dtype=complex)
    return np.sum(displacement(first)*displacement(second))/2


@pytest.mark.parametrize("k", [2, 3])
def test_non_gaussian_parent_reference_characteristic_function_is_preserved(k):
    row = module.witness(k)
    omega = np.kron(np.eye(k), module.source.J)
    evolved = expm(omega @ row["generator"])
    original = module.source.source_dilation(k)
    rng = np.random.default_rng(20260905+k)
    for _ in range(12):
        output, reference = rng.normal(size=2*k), rng.normal(size=2)
        actual_input, expected_input = evolved.T @ output, original.T @ output
        actual = _bell_characteristic(actual_input[:2], reference)*math.exp(
            -float(actual_input[2:] @ actual_input[2:])/4)
        expected = _bell_characteristic(expected_input[:2], reference)*math.exp(
            -float(expected_input[2:] @ expected_input[2:])/4)
        assert actual == pytest.approx(expected, abs=2e-13)


@pytest.mark.parametrize("k,occupation", [(2, 0), (2, 2), (3, 0), (3, 2)])
def test_same_generator_energy_and_external_switch_work_close_for_excited_input(k, occupation):
    row = module.witness(k)
    metric = row["generator"]
    omega = np.kron(np.eye(k), module.source.J)
    covariance = np.eye(2*k)/2
    covariance[:2, :2] += occupation*np.eye(2)
    mean = np.zeros(2*k)
    mean[:2] = [.3, -.7]
    second_moment = covariance+np.outer(mean, mean)
    initial = float(np.trace(metric @ second_moment)/2)
    for tau in (0., .23, 1., 1.7):
        transform = expm(omega @ metric*tau)
        final = transform @ second_moment @ transform.T
        assert np.trace(metric @ final)/2 == pytest.approx(initial, abs=1e-11)
    transform = expm(omega @ metric)
    final = transform @ second_moment @ transform.T
    bare_initial, bare_final = np.trace(second_moment)/2, np.trace(final)/2
    on, off = initial-bare_initial, bare_final-initial
    original = module.source.source_dilation(k)
    expected = original @ second_moment @ original.T
    assert on+off == pytest.approx((np.trace(expected)-np.trace(second_moment))/2, abs=1e-11)


@pytest.mark.parametrize("k,work", [(2, 11/16), (3, 1.)])
def test_winding_changes_internal_energy_but_not_the_split_or_net_switch_work(k, work):
    low, high = module.witness(k), module.witness(k, winding=1)
    omega = np.kron(np.eye(k), module.source.J)
    np.testing.assert_allclose(expm(omega @ low["generator"]),
                               expm(omega @ high["generator"]), atol=4e-13)
    assert min(np.linalg.eigvalsh(high["generator"]-low["generator"])) > 0
    assert high["prepared_energy_above_ground"] > low["prepared_energy_above_ground"]
    assert high["external_switching"]["switch_on_work"] > low["external_switching"]["switch_on_work"]
    for row in (low, high):
        assert row["external_switching"]["net_work"] == pytest.approx(work)


@pytest.mark.parametrize("k,angles,winding", [(2, [0.], 0), (2, [1., 2.], 0),
                                             (3, [math.nan, 1.], 0), (2, [2.], -1)])
def test_invalid_or_non_elliptic_inputs_are_rejected(k, angles, winding):
    with pytest.raises(ValueError):
        module.positive_generator(k, angles, winding)


def test_artifact_provenance_and_scope():
    data = json.loads((HERE/"split_quadratic_generator.json").read_text(encoding="utf-8"))
    for name, digest in data["source_hashes"].items():
        path = module.source.SPLIT_SOURCE if name == module.source.SPLIT_SOURCE.name else HERE/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert data["max_residual"] < 1e-10
    for key in ("generator_energy_uniquely_fixed_by_split", "child_output_permanently_retained",
                "autonomous_switching_or_battery_preparation_derived", "all_branching_numbers_proved",
                "joined_to_emission_bath_in_one_local_action", "CE_local_action_derived",
                "common_metric_selection_proved"):
        assert not data[key]

