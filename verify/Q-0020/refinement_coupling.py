"""상위 결합을 보존하는 분할에서도 새 차이 방향의 결합은 미정임을 검산한다.

Dimensionless supplied oscillator model: R=C.T@C. C is the coupling
coefficient matrix in units of bath hopping g.
Positive additive weights and copied parent coordinates are extra premises,
not geometric weights or a bath Hamiltonian derived from CE.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from interface_bath import bound_pair, finite_chain_check


def fixture():
    """두 부모 중 하나를 둘로 나누는 등거리 사상과 새 차이 방향."""
    old = np.array([1., 1.]) / math.sqrt(2)
    new = np.array([1 / math.sqrt(2), .5, .5])
    transfer = np.array([[1., 0.], [0., 1 / math.sqrt(2)], [0., 1 / math.sqrt(2)]])
    innovation = np.array([0., 1., -1.]) / math.sqrt(2)
    return old, new, transfer, innovation


def refined_coupling(beta):
    """새 자식 차이 방향에만 결합을 더하는 공통 모드 보존 계열."""
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    _, new, _, innovation = fixture()
    projection = np.eye(3) - np.outer(new, new)
    coefficient = beta / (math.sqrt(1 + beta) + 1)
    return projection + coefficient * np.outer(innovation, innovation)


def symbolic_identities():
    """유리수와 근호로 정의한 반례의 행렬 항등식을 정확히 확인한다."""
    import sympy as sp

    beta = sp.Symbol("beta", nonnegative=True)
    s = sp.sqrt(2) / 2
    old = sp.Matrix([s, s])
    new = sp.Matrix([s, sp.Rational(1, 2), sp.Rational(1, 2)])
    transfer = sp.Matrix([[1, 0], [0, s], [0, s]])
    innovation = sp.Matrix([0, s, -s])
    parent = sp.eye(2) - old * old.T
    projection = sp.eye(3) - new * new.T
    extra = innovation * innovation.T
    gram = projection + beta * extra
    coupling = projection + (sp.sqrt(1 + beta) - 1) * extra

    def zero(matrix):
        return matrix.applyfunc(sp.simplify).is_zero_matrix is True

    checks = {
        "isometry": zero(transfer.T * transfer - sp.eye(2)),
        "common_mode_transport": zero(transfer * old - new),
        "new_direction_invisible_to_parent": zero(transfer.T * innovation),
        "common_mode_kernel": zero(gram * new),
        "same_parent_pullback_for_all_beta": zero(transfer.T * gram * transfer - parent),
        "coupling_factorization": zero(coupling.T * coupling - gram),
        "new_direction_eigenvalue": zero(gram * innovation - (1 + beta) * innovation),
    }
    if not all(checks.values()):
        raise RuntimeError("exact refinement identity failed")
    return checks


def run():
    old, new, transfer, innovation = fixture()
    parent = np.eye(2) - np.outer(old, old)
    rows = []
    for beta in (0., 2.):
        coupling = refined_coupling(beta)
        gram = coupling.T @ coupling
        q = 1 + beta
        energy, weights = finite_chain_check(q)
        spectral = bound_pair(q)
        if len(energy) != spectral["bound_states"]:
            raise RuntimeError("finite-chain spectral check failed")
        if len(energy):
            np.testing.assert_allclose(energy, spectral["relative_energies"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(weights, spectral["boundary_weight_per_state"], atol=1e-12, rtol=0)
        residuals = {
            "parent_pullback_residual": float(np.linalg.norm(transfer.T @ gram @ transfer - parent)),
            "common_mode_residual": float(np.linalg.norm(coupling @ new)),
            "new_direction_eigen_residual": float(np.linalg.norm(gram @ innovation - q * innovation)),
        }
        if max(residuals.values()) > 1e-12:
            raise RuntimeError("refinement constraint failed")
        rows.append({
            "beta": beta, "new_mode_coupling_squared": q,
            "coupling_gram": gram.tolist(), **residuals,
            "new_mode_spectrum": spectral,
            "finite_chain_bound_states": len(energy),
        })
    return {
        "scope": "counterexample to coupling uniqueness from common-mode preservation and parent pullback",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bath_source_sha256": hashlib.sha256(Path(__file__).with_name("interface_bath.py").read_bytes()).hexdigest(),
        "parent_weights": [.5, .5], "child_weights": [.5, .25, .25],
        "initial_new_mode": innovation.tolist(),
        "exact_symbolic_checks": symbolic_identities(), "cases": rows,
        "bath_initial_state": "vacuum; identical semi-infinite chains in both cases",
        "common_onsite_over_bath_hopping": 10,
        "infinite_bath_limit_precedes_time_average": True,
        "conserved_geometric_weights_derived_from_CE": False,
        "microscopic_coupling_derived_from_CE": False,
        "common_metric_selection_proved": False,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
