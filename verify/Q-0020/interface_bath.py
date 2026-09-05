"""공유 변의 차이 결합이 공통 모드를 보존해도 방출을 보장하지 않음을 검산한다.

Supplied oscillator interface model, not a coupling derived from CE.
One identical vacuum chain per owner pair; one-excitation sector.
All couplings and energies below are divided by the bath hopping g > 0.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np


def pair_incidence(owners):
    if isinstance(owners, bool) or not isinstance(owners, int) or owners < 2:
        raise ValueError("owners must be an integer >= 2")
    matrix = np.zeros((math.comb(owners, 2), owners))
    for row, (first, second) in enumerate(combinations(range(owners), 2)):
        matrix[row, first], matrix[row, second] = 1., -1.
    return matrix


def owner_histogram(depth):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2):
        raise ValueError("this diagnostic covers depths 1 and 2")
    cells, next_vertex = [tuple(range(5))], 5
    for _ in range(depth):
        refined = []
        for cell in cells:
            refined.extend((next_vertex,) + tuple(v for v in cell if v != omitted) for omitted in cell)
            next_vertex += 1
        cells = refined
    counts = Counter(tuple(sorted(edge)) for cell in cells for edge in combinations(cell, 2))
    return dict(sorted(Counter(counts.values()).items()))


def bound_pair(coupling_squared):
    """Exact bound pair for an endpoint link v with v²>2; common onsite omitted."""
    q = float(coupling_squared)
    if not math.isfinite(q) or q <= 0:
        raise ValueError("coupling squared must be finite and positive")
    if q <= 2:
        return {"bound_states": 0, "long_time_mean_survival": 0.0}
    weight = (q - 2) / (2 * (q - 1))
    energy = q / math.sqrt(q - 1)
    return {
        "bound_states": 2, "relative_energies": [-energy, energy],
        "boundary_weight_per_state": weight,
        "long_time_mean_survival": 2 * weight**2,
        "asymptotic_survival_oscillates": True,
    }


def finite_chain_check(coupling_squared, sites=96):
    """유한 사슬 검산. 문턱 근처의 무한계 결합상태 개수를 보증하지 않는다."""
    if isinstance(sites, bool) or not isinstance(sites, int) or sites < 2:
        raise ValueError("sites must be an integer >= 2")
    coupling_squared = float(coupling_squared)
    if not math.isfinite(coupling_squared) or coupling_squared <= 0:
        raise ValueError("coupling squared must be finite and positive")
    matrix = np.diag(np.ones(sites - 1), 1) + np.diag(np.ones(sites - 1), -1)
    matrix[0, 1] = matrix[1, 0] = math.sqrt(coupling_squared)
    energy, vectors = np.linalg.eigh(matrix)
    outside = np.abs(energy) > 2 + 1e-10
    return energy[outside], vectors[0, outside]**2


def run():
    checks = []
    for owners in (3, 4, 9, 12):
        incidence = pair_incidence(owners)
        laplacian = incidence.T @ incidence
        expected = owners*np.eye(owners) - np.ones((owners, owners))
        bound = bound_pair(owners)
        energy, weights = finite_chain_check(owners)
        row = {
            "owners": owners, "pair_channels": len(incidence), "mismatch_modes": owners - 1,
            "laplacian_residual": float(np.linalg.norm(laplacian - expected)),
            "common_mode_coupling_residual": float(np.linalg.norm(incidence @ np.ones(owners))),
            "raw_effective_coupling_squared": owners, "raw_bound_pair": bound,
            "finite_chain_bound_energy_residual": float(np.max(np.abs(energy - bound["relative_energies"]))),
            "finite_chain_boundary_weight_residual": float(np.max(np.abs(weights - bound["boundary_weight_per_state"]))),
            "normalized_pair_strength": 1 / math.sqrt(owners),
            "normalized_effective_coupling_squared": 1,
            "normalized_bound_states": bound_pair(1)["bound_states"],
        }
        if max(row[key] for key in (
            "laplacian_residual", "common_mode_coupling_residual",
            "finite_chain_bound_energy_residual", "finite_chain_boundary_weight_residual",
        )) > 1e-10:
            raise RuntimeError("interface spectral check failed")
        checks.append(row)
    return {
        "scope": "post hoc diagnostic of supplied pair-difference oscillator/bath coupling",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "owner_histograms": {str(depth): owner_histogram(depth) for depth in (1, 2)},
        "common_onsite_over_bath_hopping": 10,
        "finite_chain_check_sites": 96, "owner_checks": checks,
        "threshold": "v^2 > 2 creates two normalizable bound states; v^2=2 is not a bound state",
        "infinite_bath_limit_precedes_time_average": True,
        "literal_length_matching_from_vacuum_proved": False,
        "microscopic_CE_coupling_derived": False,
        "normalization_physically_selected": False,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
