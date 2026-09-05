"""기존 정준 분할에서 이차 에너지 형식을 옮기는 조건부 검산.

R_out=S R_in, H/(hbar*omega)=R^T G R/2의 무차원 규약이다.
모든 입력 상태에서 에너지 형식을 보존하는 유일한 출력 행렬은
G_out=S^{-T} G_in S^{-1}이다. 이 항등식은 S를 발생시키는 자율
Hamiltonian이나 Hamiltonian 교체에 필요한 일/제어기를 주지 않는다.

독립 단위 진동자 입력이면 자식별 독립 출력 에너지는 첫 단계 k=3에서만
가능하다. 여기서 독립은 서로 다른 자식의 q,p 교차항 부재를 뜻하며
공간적 국소성 또는 공간 차원을 뜻하지 않는다. 같은 단위 잡음의 두 번째
3진 분할에는 교차항이 생긴다. 보조 모드의 에너지를 부모에 맞춘 대조는
그 바닥상태를 쓰면 잡음이 세 배, 기존 진공을 쓰면 준비 들뜸이 2/3이다.

전체 현재 잎의 전달 형식 Gq=(2Q)^-1, Gp=2Q도 기존 분할 공분산과
직접 대조한다. 서로 다른 단계의 Hamiltonian을 같은 에너지라 부르거나
자유 진동자에 대한 E-030 준비 하한을 지우는 해석은 하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import split_quantum_source as source


def energy_transport(transform, input_energy):
    """주어진 정준 변환과 양의 입력 형식에서 출력 이차 형식을 계산한다."""
    s, g = np.asarray(transform, dtype=float), np.asarray(input_energy, dtype=float)
    if (s.ndim != 2 or s.shape[0] == 0 or s.shape[0] != s.shape[1]
            or s.shape[0] % 2 or g.shape != s.shape):
        raise ValueError("equal nonempty even square matrices required")
    if not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError("finite matrices required")
    if not np.allclose(g, g.T, atol=1e-13, rtol=0) or np.linalg.eigvalsh(g)[0] <= 0:
        raise ValueError("positive symmetric input energy required")
    omega = np.kron(np.eye(len(s) // 2), source.J)
    if not np.allclose(s @ omega @ s.T, omega, atol=1e-11, rtol=0):
        raise ValueError("canonical transform required")
    inverse = np.linalg.solve(s, np.eye(len(s)))
    out = inverse.T @ g @ inverse
    return (out + out.T) / 2


def local_energy(k, parent=(1., 1.), ancilla=(1., 1.)):
    """부모와 보조 모드의 q,p 계수에 기존 분할 행렬을 적용한다."""
    source.child_count(k)
    parent, ancilla = np.asarray(parent, dtype=float), np.asarray(ancilla, dtype=float)
    if parent.shape != (2,) or ancilla.shape != (2,):
        raise ValueError("two q,p coefficients per mode required")
    coefficients = np.tile(ancilla, k)
    coefficients[:2] = parent
    return energy_transport(source.source_dilation(k), np.diag(coefficients))


def independent_child_residual(energy):
    """자식별 2x2 대각 블록 밖의 가장 큰 계수."""
    mask = np.arange(len(energy))[:, None] // 2 != np.arange(len(energy))[None, :] // 2
    return float(np.max(np.abs(energy[mask]), initial=0.))


def frontier_energy(k, depth):
    q_map, p_map = source.recursive_source_maps(k, depth)
    n = len(q_map)
    transform = np.zeros((2 * n, 2 * n))
    transform[::2, ::2], transform[1::2, 1::2] = q_map, p_map
    energy = energy_transport(transform, np.eye(2 * n))
    return energy, transform


def ancilla_controls():
    k = 3
    transform = source.source_dilation(k)
    parent = np.array([1/3, 3.])
    controls = []
    for name, variance in (("original_vacuum", np.array([.5, .5])),
                           ("new_energy_ground_state", np.array([1.5, 1/6]))):
        covariance = np.diag(np.tile(variance, k - 1))
        noise = transform[:, 2:] @ covariance @ transform[:, 2:].T
        contrast = source.mode_basis(k)[:, 1]
        controls.append({
            "ancilla_state": name,
            "input_q_variance": float(variance[0]),
            "contrast_q_noise": float(contrast @ noise[::2, ::2] @ contrast),
            "ancilla_excess_energy": float((k - 1) * (parent @ variance / 2 - .5)),
            "unit_noise_preserved": name == "original_vacuum",
        })
    return controls


def run():
    single = []
    for k in (2, 3, 4, 5):
        s = source.source_dilation(k)
        energy = local_energy(k)
        p0 = np.ones((k, k)) / k
        perp = np.eye(k) - p0
        q = (k - 1) / (2 * k) * perp + p0 / k
        p = 2 * k / (k - 1) * perp + k * p0
        residuals = {
            "energy_identity": float(np.linalg.norm(s.T @ energy @ s - np.eye(2 * k))),
            "q_projector_formula": float(np.linalg.norm(energy[::2, ::2] - q)),
            "p_projector_formula": float(np.linalg.norm(energy[1::2, 1::2] - p)),
        }
        single.append({"branching": k, "q_off_diagonal": float(energy[0, 2]),
                       "p_off_diagonal": float(energy[1, 3]),
                       "child_independence_residual": independent_child_residual(energy),
                       "residuals": residuals})
    frontier = []
    for k, depth in ((2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 2)):
        energy, s = frontier_energy(k, depth)
        n = k**depth
        q = source.frontier_position_covariance(k, depth)
        covariance = .5 * s @ s.T
        residuals = {
            "energy_identity": float(np.linalg.norm(s.T @ energy @ s - np.eye(2 * n))),
            "q_precision": float(np.linalg.norm(energy[::2, ::2] - np.linalg.solve(2*q, np.eye(n)))),
            "p_coefficient": float(np.linalg.norm(energy[1::2, 1::2] - 2*q)),
            "ground_covariance": float(np.linalg.norm(energy @ covariance - .5*np.eye(2*n))),
        }
        frontier.append({"branching": k, "depth": depth, "frontier_modes": n,
                         "child_independence_residual": independent_child_residual(energy),
                         "transported_energy_above_its_ground": float(np.trace(energy @ covariance)/2 - n/2),
                         "fixed_free_excitation_energy": float((np.trace(covariance) - n)/2),
                         "residuals": residuals})
    if max(value for row in single + frontier for value in row["residuals"].values()) > 1e-10:
        raise RuntimeError("energy transport identity failed")
    second = local_energy(3, parent=(1/3, 3.))
    equal = local_energy(3, parent=(1/3, 3.), ancilla=(1/3, 3.))
    paths = (Path(__file__), Path(source.__file__), source.SPLIT_SOURCE,
             Path(__file__).with_name("interface_bath.py"))
    return {
        "scope": "고정된 정준 분할과 입력 에너지의 전달 형식 및 반복 분할의 준비 조건",
        "energy_unit": "hbar*omega", "quadratures": "dimensionless; [q,p]=i",
        "numpy": np.__version__,
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "first_split": single, "recursive_frontier": frontier,
        "second_ternary_split": {"q_off_diagonal": float(second[0, 2]),
                                 "p_off_diagonal": float(second[1, 3]),
                                 "energy": second.tolist()},
        "matched_ancilla_energy": {"child_independence_residual": independent_child_residual(equal),
                                   "states": ancilla_controls()},
        "input_energy_is_supplied": True,
        "transport_is_autonomous_generator": False,
        "hamiltonian_switch_work_derived": False,
        "bath_coupling_derived": False,
        "physical_spatial_locality_tested": False,
        "branching_three_implies_spatial_dimension_three": False,
        "common_metric_selection_proved": False,
        "fixed_free_energy_preparation_bound_removed": False,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(result["first_split"]) + len(result["recursive_frontier"]),
                      "second_split": result["second_ternary_split"],
                      "controls": result["matched_ancilla_energy"]}, ensure_ascii=False))
