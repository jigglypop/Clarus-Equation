"""국소 반발 상호작용이 있는 환경의 두 입자 결합상태와 준비 비용을 검산한다.

에너지는 bath hopping g로 나눈다. H/g=dGamma(h)+u*n0*(n0-1)/2,
u=U/g>=0이며 h의 onsite는 10, 첫 hopping은 sqrt(q), 나머지는 1이다.
유한 사슬의 고유값과 무한 반직선의 조건부 하한을 서로 구분한다.
초기 분할 상태와 상호작용은 외부 입력이며 CE 미시 작용에서 유도되지 않았다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from interface_bath import bound_pair
from split_quantum_source import mode_basis, source_dilation


def _parameters(coupling_squared, interaction):
    q, u = float(coupling_squared), float(interaction)
    if not math.isfinite(q) or q <= 0:
        raise ValueError("coupling squared must be finite and positive")
    if not math.isfinite(u) or u < 0:
        raise ValueError("interaction must be finite and nonnegative")
    if q > 2 and bound_pair(q)["relative_energies"][1] >= 10:
        raise ValueError("the infinite one-particle Hamiltonian must be positive")
    return q, u


def one_particle_matrix(coupling_squared, sites):
    q, _ = _parameters(coupling_squared, 0)
    if isinstance(sites, bool) or not isinstance(sites, int) or not 2 <= sites <= 256:
        raise ValueError("sites must be an integer between 2 and 256")
    matrix = 10 * np.eye(sites)
    matrix += np.diag(np.ones(sites - 1), 1) + np.diag(np.ones(sites - 1), -1)
    matrix[0, 1] = matrix[1, 0] = math.sqrt(q)
    return matrix


def finite_pair(coupling_squared, interaction, sites=64):
    """유한 사슬의 최상위 두 입자 고유값과 |2_0> 중첩을 계산한다.

    자유 두 입자 에너지의 합성 측도에 rank-one resolvent를 적용한다.
    유한 사슬의 최상위 상태 존재만으로 무한계 결합상태를 주장하지 않는다.
    """
    q, u = _parameters(coupling_squared, interaction)
    energies, vectors = np.linalg.eigh(one_particle_matrix(q, sites))
    single_weights = vectors[0] ** 2
    sums = (energies[:, None] + energies[None, :]).ravel()
    weights = np.outer(single_weights, single_weights).ravel()
    top = float(2 * energies[-1])
    atom = float(single_weights[-1] ** 2)
    if u == 0:
        return {"sites": sites, "energy_over_g": top, "local_pair_weight": atom,
                "secular_residual": 0.0, "finite_unperturbed_top_over_g": top}
    lower, upper = u * atom, u
    if lower == 0 or top + upper == top:
        raise ValueError("interaction is below floating-point resolution")
    offsets = top - sums
    for _ in range(85):
        delta = (lower + upper) / 2
        if u * np.sum(weights / (offsets + delta)) > 1:
            lower = delta
        else:
            upper = delta
    delta = (lower + upper) / 2
    denominators = offsets + delta
    scaled_inverse = u / denominators
    secular = float(abs(np.sum(weights * scaled_inverse) - 1))
    overlap = float(1 / np.sum(weights * scaled_inverse**2))
    if not math.isfinite(overlap) or secular > 1e-11:
        raise RuntimeError("finite pair resolvent check failed")
    return {"sites": sites, "energy_over_g": top + delta,
            "local_pair_weight": overlap, "secular_residual": secular,
            "finite_unperturbed_top_over_g": top}


def source_budget(children, interaction):
    """실제 정준 분할이 만드는 한 차이 모드의 상태와 초기 에너지를 계산한다."""
    _, u = _parameters(1, interaction)
    dilation = source_dilation(children)
    basis = np.kron(mode_basis(children), np.eye(2))
    covariance = basis.T @ (.5 * dilation @ dilation.T) @ basis
    q_variance, p_variance = covariance[2, 2], covariance[3, 3]
    squeeze = .25 * math.log(q_variance / p_variance)
    number = float((q_variance + p_variance - 1) / 2)
    p2 = .5 * math.tanh(squeeze)**2 / math.cosh(squeeze)
    factorial_second = 3 * number**2 + number
    return {"children": children, "mean_number": number,
            "two_particle_probability": p2,
            "factorial_second_moment": factorial_second,
            "initial_energy_over_g": 10 * number + .5 * u * factorial_second}


def infinite_retention_bound(children, coupling_squared, interaction):
    """무한 반직선 증명의 충분조건을 대입한 값이다. 평균 점유의 예측값이 아니다.

    q>2이면 기존 상단 결합상태에서 w>=Z^2를 얻는다. q<=2에서는
    u>4가 새 결합상태의 충분조건이고 w>=1-4/u이다. u=4가 정확한
    문턱이라는 주장은 하지 않는다. 두 경우 liminf 평균 n0 >= 2*p2*w^2.
    """
    q, u = _parameters(coupling_squared, interaction)
    source = source_budget(children, u)
    if q > 2:
        bound = bound_pair(q)
        weight_bound = bound["boundary_weight_per_state"] ** 2
        essential_top = 22 + bound["relative_energies"][1]
        reason = "existing isolated upper pair survives every finite nonnegative u"
    elif u > 4:
        weight_bound = 1 - 4 / u
        essential_top = 24.0
        reason = "u>4 is sufficient, not asserted to be the exact threshold"
    else:
        return {"positive_bound_established": False,
                "time_mean_liminf_number_lower_bound": None,
                "reason": "this argument gives no positive bound; complete emission is not proved"}
    return {"positive_bound_established": True,
            "local_pair_weight_lower_bound": weight_bound,
            "infinite_essential_top_over_g": essential_top,
            "time_mean_liminf_number_lower_bound": 2 * source["two_particle_probability"] * weight_bound**2,
            "reason": reason}


def run():
    cases = []
    for children, q, interactions in ((3, 3, (0., .1, 1., 8.)),
                                       (4, 4, (0., .1, 1., 8.)),
                                       (3, 1, (8.,))):
        for u in interactions:
            finite = finite_pair(q, u)
            bound = infinite_retention_bound(children, q, u)
            if finite["energy_over_g"] <= bound["infinite_essential_top_over_g"]:
                raise RuntimeError("finite check did not resolve the predicted isolated pair")
            if finite["local_pair_weight"] < bound["local_pair_weight_lower_bound"] - 1e-10:
                raise RuntimeError("finite overlap contradicts the conditional bound")
            cases.append({"children": children, "coupling_squared": q, "interaction_over_g": u,
                          "source": source_budget(children, u), "finite_chain": finite,
                          "infinite_conditional_bound": bound})
    here = Path(__file__).resolve().parent
    paths = {"source_sha256": Path(__file__), "linear_bath_sha256": here / "interface_bath.py",
             "split_source_sha256": here / "split_quantum_source.py"}
    return {"scope": "supplied repulsive endpoint interaction and squeezed source",
            "python": sys.version.split()[0], "numpy": np.__version__,
            **{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
            "energy_unit": "g", "interaction_parameter": "u=U/g",
            "finite_chain_proves_infinite_time_limit": False,
            "full_time_mean_existence_asserted": False,
            "bound_is_predicted_retained_number": False,
            "initial_energy_fixed_while_varying_u": False,
            "all_nonlinear_environments_excluded": False,
            "autonomous_source_action_derived_from_CE": False,
            "common_metric_selection_proved": False, "cases": cases}


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
