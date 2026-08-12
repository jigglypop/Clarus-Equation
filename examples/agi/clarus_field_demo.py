"""Run the bounded Clarus-field primitive on a deterministic ring graph."""

from __future__ import annotations

import json

import numpy as np

from reality_stone.clarus.clarus_field import (
    ClarusField,
    ClarusFieldConfig,
    bounded_hrr_bind,
    prediction_error_gate_scores,
)


def main() -> None:
    nodes = 8
    adjacency = np.zeros((nodes, nodes), dtype=np.float64)
    for index in range(nodes):
        adjacency[index, (index - 1) % nodes] = 1.0
        adjacency[index, (index + 1) % nodes] = 1.0
    runtime = ClarusField(
        adjacency,
        ClarusFieldConfig(
            width=4,
            field_decay=0.3,
            diffusion_strength=0.8,
            gate_threshold=0.6,
            structure_threshold=0.2,
        ),
    )
    state = runtime.zero_state()
    occupancy = []
    for tick in range(32):
        prediction = np.zeros((nodes, 4))
        observation = np.zeros((nodes, 4))
        observation[tick % nodes, tick % 4] = 1.0
        gates = prediction_error_gate_scores(
            observation,
            prediction,
            reference_scale=1.0,
        )
        candidate = np.zeros((nodes, 4))
        candidate[tick % nodes, tick % 4] = 1.0
        result = runtime.step(state, runtime.make_drive(gates, candidate))
        state = result.state
        occupancy.append(result.occupancy.as_tuple())

    memory = np.asarray(state.memory)
    payload = {
        "status": "research-primitive",
        "tick": state.tick,
        "memory_max_norm": float(np.max(np.linalg.norm(memory, axis=1))),
        "field_norm": float(np.linalg.norm(state.field)),
        "mean_occupancy_active_structural_frozen": np.mean(occupancy, axis=0).tolist(),
        "bounded_hrr_example": bounded_hrr_bind(memory[0], memory[1]).tolist(),
        "certificate": {
            "cf1_bounded_positive_field": runtime.certificate.cf1_bounded_positive_field,
            "cf2_exact_closed_gate": runtime.certificate.cf2_exact_closed_gate,
            "cf3_scope": runtime.certificate.cf3_scope,
            "p_star_self_convergence": runtime.certificate.p_star_self_convergence,
            "v14_route_l_inherited": runtime.certificate.v14_route_l_inherited,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
