"""기존 분할의 위치 라벨 법칙을 공급된 양자 변환으로 실현하고 검산한다.

정준 좌표는 무차원 [q,p]=i, 진공 공분산은 I/2이다.
한 부모를 k개의 현재 모드로 대체한다. 역사 속 모든 노드를 동시에
독립 정준 모드로 해석하지 않는다. 압축 상태 준비와 환경은 추가 입력이다.
Q-0016의 기각된 물리 카드를 부활시키거나 Q-0017과 결합하지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from interface_bath import bound_pair

ROOT = Path(__file__).resolve().parents[2]
SPLIT_SOURCE = ROOT / "verify/Q-0016/F-01/predict_split_kernel.py"
J = np.array([[0., 1.], [-1., 0.]])


def child_count(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("children must be an integer >= 2")
    return k


@lru_cache(maxsize=1)
def classical_split():
    """원본의 작은 행렬 함수만 불러오며 몬테카를로 실행부는 호출하지 않는다."""
    spec = importlib.util.spec_from_file_location("q0016_split_source", SPLIT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def mode_basis(k):
    """첫 열은 공통 방향, 나머지는 정규직교 자식 차이 방향이다."""
    child_count(k)
    basis = np.zeros((k, k))
    basis[:, 0] = 1 / math.sqrt(k)
    for j in range(1, k):
        basis[:j, j] = 1 / math.sqrt(j * (j + 1))
        basis[j, j] = -j / math.sqrt(j * (j + 1))
    return basis


def source_dilation(k):
    """모든 입력이 진공일 때 부모와 보조 모드를 압축한 뒤 섞는 정준 변환."""
    child_count(k)
    nu = k / (k - 1)
    gains = np.full(k, math.sqrt(2 * nu))
    gains[0] = math.sqrt(k)
    squeeze = np.diag(np.column_stack((gains, 1 / gains)).ravel())
    return np.kron(mode_basis(k), np.eye(2)) @ squeeze


def cp_matrix(x, noise):
    """공분산 규약 V_vac=I/2에서 가우시안 채널의 완전양성 행렬."""
    omega = np.kron(np.eye(x.shape[0] // 2), J)
    return noise + .5j * (omega - x @ J @ x.T)


def source_check(k):
    child_count(k)
    nu = k / (k - 1)
    parent = [-1] + [0] * k
    source = classical_split()
    classical_covariance = source.split_C(parent)[1:, 1:]
    labels = source.split_labels(parent, np.eye(k + 1))
    innovations = labels[1:] - labels[0]
    dilation = source_dilation(k)
    x = dilation[:, :2]
    noise = .5 * dilation[:, 2:] @ dilation[:, 2:].T
    covariance = .5 * dilation @ dilation.T
    omega = np.kron(np.eye(k), J)
    naive_x = np.tile(np.eye(2), (k, 1))
    naive_noise = np.kron(classical_covariance, np.eye(2))
    naive_min = float(np.linalg.eigvalsh(cp_matrix(naive_x, naive_noise))[0])
    x_expected = np.tile(np.diag([1., 1 / k]), (k, 1))
    n0 = (nu + 1 / (4 * nu) - 1) / 2
    output_number = (np.trace(covariance) - k) / 2
    return {
        "children": k,
        "contrast_q_variance": nu,
        "contrast_p_variance": 1 / (4 * nu),
        "contrast_initial_number_per_mode": n0,
        "contrast_initial_number_total": (k - 1) * n0,
        "vacuum_input_output_number": float(output_number),
        "common_mode_number_after_squeeze": (k + 1 / k - 2) / 4,
        "naive_both_quadrature_cp_min_eigenvalue": naive_min,
        "position_only_cp_min_eigenvalue": float(np.linalg.eigvalsh(cp_matrix(x, noise))[0]),
        "residuals": {
            "stored_sampler_covariance": float(np.linalg.norm(innovations @ innovations.T - classical_covariance)),
            "position_noise_matches_stored_source": float(np.linalg.norm(noise[::2, ::2] - classical_covariance)),
            "parent_map": float(np.linalg.norm(x - x_expected)),
            "symplectic": float(np.linalg.norm(dilation @ omega @ dilation.T - omega)),
            "naive_cp_negative_eigenvalue": abs(naive_min + (k - 1) / 2),
            "vacuum_input_number_formula": abs(float(output_number) - (3 * k*k - 2*k + 3) / (8*k)),
        },
    }


def bath_number_check(k, coupling_squared, sites=64):
    """유한 사슬에서 전체 공분산 전파와 단일입자 진폭의 점유식을 교차 검산한다.

    이 함수의 유한 시간·유한 사슬 결과로 무한시간 방출을 추론하지 않는다.
    시간은 hbar/g 단위이며 모든 위치의 에너지는 10g이다.
    """
    child_count(k)
    if isinstance(sites, bool) or not isinstance(sites, int) or sites < 2:
        raise ValueError("sites must be an integer >= 2")
    coupling_squared = float(coupling_squared)
    if not math.isfinite(coupling_squared) or coupling_squared <= 0:
        raise ValueError("coupling squared must be finite and positive")
    h = 10 * np.eye(sites) + np.diag(np.ones(sites - 1), 1) + np.diag(np.ones(sites - 1), -1)
    h[0, 1] = h[1, 0] = math.sqrt(coupling_squared)
    energies, vectors = np.linalg.eigh(h)
    nu = k / (k - 1)
    initial = .5 * np.eye(2 * sites)
    initial[0, 0], initial[sites, sites] = nu, 1 / (4 * nu)
    n0 = (nu + 1 / (4 * nu) - 1) / 2
    rows = []
    for tau in (0., math.pi / 20, .3, 1., 5., 20.):
        u = (vectors * np.exp(-1j * energies * tau)) @ vectors.T
        transform = np.block([[u.real, -u.imag], [u.imag, u.real]])
        final = transform @ initial @ transform.T
        boundary_number = (final[0, 0] + final[sites, sites] - 1) / 2
        anomalous = (nu - 1 / (4 * nu)) / 2
        predicted_q_variance = .5 + abs(u[0, 0])**2 * n0 + (u[0, 0]**2 * anomalous).real
        rows.append({
            "time_g_over_hbar": tau,
            "q_variance_from_covariance": float(final[0, 0]),
            "q_variance_from_amplitude": float(predicted_q_variance),
            "q_variance_identity_residual": float(abs(final[0, 0] - predicted_q_variance)),
            "boundary_number_from_covariance": float(boundary_number),
            "boundary_number_from_amplitude": float(abs(u[0, 0])**2 * n0),
            "number_identity_residual": float(abs(boundary_number - abs(u[0, 0])**2 * n0)),
            "total_number_residual": float(abs((np.trace(final) - sites) / 2 - n0)),
        })
    return rows


def asymptotic_number(k, coupling_squared):
    """E-028 반무한 사슬 정리와 진공 환경의 점유 항등식으로 계산한다."""
    child_count(k)
    spectrum = bound_pair(coupling_squared)
    nu = k / (k - 1)
    n0 = (nu + 1 / (4 * nu) - 1) / 2
    return (k - 1) * n0 * spectrum["long_time_mean_survival"]


def _frontier_size(k, depth):
    child_count(k)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    # 작은 직접 행렬 증인만 생성한다. 큰 깊이는 아래 정확한 스펙트럼 식으로 계산한다.
    if depth > 8 or k**depth > 256:
        raise ValueError("direct frontier witness is limited to 256 modes and depth 8")
    return k**depth


def frontier_position_covariance(k, depth, root_variance=.5):
    """저장된 분할 함수를 전체 트리에 적용한 뒤 현재 잎만 읽는다.

    root_variance=1은 원본 고전 뿌리 분산, .5는 E-030 진공 입력이다.
    """
    leaves = _frontier_size(k, depth)
    root_variance = float(root_variance)
    if not math.isfinite(root_variance) or root_variance <= 0:
        raise ValueError("root variance must be finite and positive")
    nodes = (k * leaves - 1) // (k - 1)
    parent = [-1] + [(i - 1) // k for i in range(1, nodes)]
    noise_map = np.eye(nodes)
    noise_map[0, 0] = math.sqrt(root_variance)
    labels = classical_split().split_labels(parent, noise_map)[-leaves:]
    return labels @ labels.T


def recursive_source_maps(k, depth, root_variance=.5):
    """E-030의 국소 정준 변환을 각 부모에 실제로 합성한다."""
    _frontier_size(k, depth)
    root_variance = float(root_variance)
    if not math.isfinite(root_variance) or root_variance <= 0:
        raise ValueError("root variance must be finite and positive")
    q_map = np.array([[math.sqrt(2 * root_variance)]])
    p_map = np.array([[1 / math.sqrt(2 * root_variance)]])
    if depth == 0:
        return q_map, p_map
    local = source_dilation(k)
    local_q, local_p = local[::2, ::2], local[1::2, 1::2]
    for _ in range(depth):
        parent_count = len(q_map)
        old_q = np.kron(np.eye(parent_count), local_q[:, :1]) @ q_map
        old_p = np.kron(np.eye(parent_count), local_p[:, :1]) @ p_map
        q_map = np.hstack((old_q, np.kron(np.eye(parent_count), local_q[:, 1:])))
        p_map = np.hstack((old_p, np.kron(np.eye(parent_count), local_p[:, 1:])))
    return q_map, p_map


def frontier_resource_spectrum(k, depth, root_variance=None):
    """계층별 고유값과 에너지 하한을 유리수로 계산한다. 행렬을 만들지 않는다."""
    from fractions import Fraction

    child_count(k)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 1024:
        raise ValueError("exact depth must be an integer from 0 to 1024")
    v0 = Fraction(1, 2) if root_variance is None else Fraction(root_variance)
    if v0 <= 0:
        raise ValueError("root variance must be positive")
    n = k**depth
    nu = Fraction(k, k - 1)
    eigenvalues = [(n * v0, 1)]
    eigenvalues.extend(
        (nu * k**(depth - level), (k - 1) * k**(level - 1))
        for level in range(1, depth + 1)
    )
    trace_q = n * (v0 + depth)
    trace_inverse = 1 / (n * v0) + Fraction(k - 1, k + 1) * (n - Fraction(1, n))
    minimum_number = (trace_q + trace_inverse / 4 - n) / 2
    return eigenvalues, trace_q, trace_inverse, minimum_number


def minimum_number_at_position_covariance(q_covariance):
    """임의 상태의 불확정성으로 얻는 고정 위치 공분산·같은 주파수의 하한."""
    q = np.asarray(q_covariance, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[0] != q.shape[1]:
        raise ValueError("position covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(q)) or not np.allclose(q, q.T, atol=1e-13, rtol=0):
        raise ValueError("position covariance must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(q)
    if eigenvalues[0] <= 0:
        raise ValueError("position covariance must be positive definite")
    return float((eigenvalues.sum() + .25 * np.sum(1 / eigenvalues) - len(q)) / 2)


def resource_budget_check():
    cases = []
    for k, max_depth in ((2, 6), (3, 4), (4, 3)):
        for depth in range(1, max_depth + 1):
            q = frontier_position_covariance(k, depth)
            q_map, p_map = recursive_source_maps(k, depth)
            actual_q, actual_p = .5 * q_map @ q_map.T, .5 * p_map @ p_map.T
            spectrum, trace_q, trace_inverse, lower = frontier_resource_spectrum(k, depth)
            expected_eigenvalues = np.sort(np.concatenate([
                np.full(multiplicity, float(value)) for value, multiplicity in spectrum
            ]))
            residuals = {
                "stored_source_vs_quantum_q": float(np.linalg.norm(q - actual_q)),
                "hierarchical_spectrum": float(np.max(np.abs(np.linalg.eigvalsh(q) - expected_eigenvalues))),
                "canonical_pair": float(np.linalg.norm(q_map @ p_map.T - np.eye(len(q)))),
                "energy_lower_bound": abs(minimum_number_at_position_covariance(q) - float(lower)),
                "attained_energy": abs(float((np.trace(actual_q) + np.trace(actual_p) - len(q)) / 2) - float(lower)),
                "minimum_momentum_covariance": float(np.linalg.norm(actual_p - np.linalg.solve(q, np.eye(len(q))) / 4)),
            }
            if max(residuals.values()) > 1e-9:
                raise RuntimeError("recursive resource bound failed")
            cases.append({
                "branching": k, "depth": depth, "frontier_modes": len(q),
                "minimum_number_exact": str(lower), "minimum_number": float(lower),
                "trace_q_exact": str(trace_q), "trace_inverse_q_exact": str(trace_inverse),
                "residuals": residuals,
            })
    return {
        "scope": "고정된 현재 위치 공분산과 같은 진동수에서의 자유 에너지 하한 및 반복 분할 자원 제약",
        "root_q_variance": .5,
        "lower_bound_requires_gaussian_state": False,
        "bound_attained_by_supplied_pure_gaussian": True,
        "universal_preparation_work_minimum_proved": False,
        "normal_ordered_energy_unit": "hbar*omega",
        "all_initial_system_modes_vacuum_for_battery_bound": True,
        "battery_bound_requires_nonnegative_battery_hamiltonian": True,
        "battery_bound_requires_conservation_of_system_plus_battery_free_energy": True,
        "infinite_depth_possible_with_fixed_gap_and_finite_mean_battery": False,
        "exact_squeezing_from_sharp_finite_total_energy_support_possible": False,
        "finite_mean_energy_alone_excludes_finite_depth_exact_preparation": False,
        "physical_gap_or_battery_derived_from_CE": False,
        "merge_or_bath_between_splits_included": False,
        "cases": cases,
    }


def run():
    rows = []
    for k in (2, 3, 4, 9, 12):
        row = source_check(k)
        if max(row["residuals"].values()) > 1e-12 or row["position_only_cp_min_eigenvalue"] < -1e-12:
            raise RuntimeError("source dilation check failed")
        bath_rows = bath_number_check(k, k)
        if max(max(item["number_identity_residual"], item["total_number_residual"], item["q_variance_identity_residual"]) for item in bath_rows) > 1e-11:
            raise RuntimeError("full covariance bath propagation failed")
        row.update({
            "raw_coupling_squared": k,
            "raw_contrast_number_long_time_mean": asymptotic_number(k, k),
            "normalized_coupling_squared": 1,
            "normalized_contrast_number_long_time_mean": asymptotic_number(k, 1),
            "finite_bath_number_checks": bath_rows,
        })
        rows.append(row)
    sources = {
        "source_sha256": Path(__file__),
        "classical_split_source_sha256": SPLIT_SOURCE,
        "bath_source_sha256": Path(__file__).with_name("interface_bath.py"),
    }
    return {
        "scope": "기존 위치 라벨 분할을 재현하는 공급된 양자 변환과 진공 환경의 조건부 점유 계산",
        "python": sys.version.split()[0], "numpy": np.__version__,
        **{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "quadrature_convention": "[q,p]=i; vacuum covariance I/2",
        "output_number_unit": "dimensionless",
        "free_excitation_energy": "hbar*omega*output_number; equal-frequency energy above k-mode vacuum",
        "energy_is_universal_minimum_work": False,
        "infinite_bath_limit_precedes_time_average": True,
        "bath_initial_state": "각 차이 모드와 독립인 반무한 진공 사슬",
        "common_onsite_over_bath_hopping": 10,
        "all_historical_labels_are_simultaneous_canonical_modes": False,
        "q0016_physical_card_revived": False,
        "q0017_merge_rule_combined": False,
        "autonomous_source_action_derived_from_CE": False,
        "microscopic_coupling_derived_from_CE": False,
        "common_metric_selection_proved": False,
        "resource_budget": resource_budget_check(),
        "cases": rows,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
