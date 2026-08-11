"""Demonstrate the isolated finite SCC atlas foundation.

The output is an artificial graph certificate, not a biological brain atlas.
"""

from __future__ import annotations

import json

from reality_stone.clarus.scc_atlas import (
    certify_dag_block_gain,
    construct_arch1,
    decoder_f_contraction_error_bound,
    forward_time_unroll,
    threshold_scc_filtration,
)


def main() -> None:
    modules = (("sensory-0", "sensory-1"), ("integrator",), ("motor-0", "motor-1"))
    architecture = construct_arch1(modules, ((0, 1), (1, 2)))
    filtration = threshold_scc_filtration(
        architecture.nodes,
        tuple((source, target, 0.8) for source, target in architecture.edges),
        (0.9, 0.7),
        edge_semantics="artificial dimensionless effective gain",
        layer="demo-union",
        score_name="declared edge score",
    )
    unroll = forward_time_unroll(
        ("state-a", "state-b"),
        (("state-a", "state-b", 1), ("state-b", "state-a", 1)),
        horizon=4,
    )
    gain = certify_dag_block_gain(
        (
            (0.20, 0.00, 0.00),
            (0.40, 0.25, 0.00),
            (0.00, 0.35, 0.30),
        ),
        normalization_scales=(1.0, 1.0, 1.0),
        schedule="simultaneous",
    )
    rollout = decoder_f_contraction_error_bound(
        initial_decoder_error=0.10,
        decoder_defect=0.01,
        f_contraction=gain.contraction_factor,
        steps=8,
    )
    payload = {
        "scope": "artificial finite SCC foundation; no biological identification",
        "architecture": {
            "components": architecture.validation.decomposition.components,
            "condensation_edges": architecture.validation.decomposition.condensation_edges,
            "topological_order": architecture.validation.decomposition.topological_order,
            "valid": architecture.validation.valid,
        },
        "threshold_component_counts": [
            len(level.decomposition.components) for level in filtration.levels
        ],
        "unroll": {
            "event_vertex_count": len(unroll.event_nodes),
            "component_count": len(unroll.decomposition.components),
            "all_components_singleton": all(
                len(component) == 1 for component in unroll.decomposition.components
            ),
        },
        "gain_certificate": {
            "orientation": gain.gain_orientation,
            "schedule": gain.schedule,
            "spectral_radius": gain.spectral_radius,
            "weighted_contraction": gain.contraction_factor,
            "condition_number": gain.condition_number,
            "certified": gain.certified,
        },
        "decoded_rollout_bound": {
            "premise": rollout.premise,
            "steps": rollout.steps,
            "finite_horizon": rollout.finite_horizon_bound,
            "asymptotic": rollout.asymptotic_bound,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
