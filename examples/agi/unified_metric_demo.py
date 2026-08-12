"""Deterministic finite one-metric demonstration for the V15 research track."""

from __future__ import annotations

import json

import numpy as np

from reality_stone.clarus.unified_metric import UnifiedMetricCore


def main() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [2.0, 0.0],
        ]
    )
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    core = UnifiedMetricCore(points, adjacency)
    identity = core.identity_state()
    source = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    source[1] = 4.0 * np.eye(2)
    deformed = core.make_state(source)
    certificate = core.certificate(deformed)
    result = {
        "status": "finite-metric-graph-research-primitive",
        "identity_goal": core.minimum_cost_targets(identity, 0, [1, 2]).minimizers,
        "deformed_goal": core.minimum_cost_targets(deformed, 0, [1, 2]).minimizers,
        "deformed_path": core.shortest_path(deformed, 0, 3).nodes,
        "condition_number": certificate.condition_number,
        "persistent_state": certificate.persistent_state,
        "world_scope": certificate.world_scope,
        "full_geodesic_verified": certificate.full_geodesic_verified,
        "continuum_limit_verified": certificate.continuum_limit_verified,
        "agi_evidence": certificate.agi_evidence,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
