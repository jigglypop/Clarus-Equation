"""Deterministic, non-evidence demo of the isolated nested-SCC unit mechanism."""

from __future__ import annotations

from dataclasses import asdict
import json

from reality_stone.clarus.adaptive_scc_tower_controller import (
    AdaptiveTowerController,
    CausalEvent,
    CutUp,
)
from reality_stone.clarus.nested_scc_tower import NestedTowerGenerator, TowerSpec


def main() -> None:
    generator = NestedTowerGenerator(
        TowerSpec(
            shell_width=3,
            maximum_depth=3,
            observation_scales=(2.0, 2.0, 2.0),
        )
    )
    prefix_audits = [
        asdict(generator.audit_prefix(depth)) for depth in range(generator.spec.maximum_depth + 1)
    ]
    zero_compatibility = generator.compatibility_certificate(1, domain="zero_state_zero_input")
    generic_compatibility = generator.compatibility_certificate(1, domain="append_zero_unit_cube")
    causal_cone = generator.backward_causal_cone(((0, 1),), horizon=3)

    controller = AdaptiveTowerController(generator)
    observations = (
        (1.0, -0.5, 0.25),
        (-0.2, 0.7, 0.1),
        (0.4, 0.1, -0.8),
    )
    token = None
    for tick, observation in enumerate(observations):
        token = controller.observe(CausalEvent(tick, observation))
    assert token is not None
    forecast = controller.read_forecast(token)

    cut = controller.with_intervention(CutUp(0))
    cut_token = cut.observe(CausalEvent(controller.tick + 1, (0.0, 0.0, 0.0)))
    output = {
        "status": "deterministic unit fixture; not development evidence",
        "parameter_manifest": asdict(generator.manifest),
        "prefix_audits": prefix_audits,
        "contraction": asdict(generator.certify_prefix(3)),
        "zero_fixture_compatibility": asdict(zero_compatibility),
        "generic_append_zero_compatibility": asdict(generic_compatibility),
        "causal_cone": asdict(causal_cone),
        "intact_forecast": forecast,
        "cut_up_forecast": cut.read_forecast(cut_token),
        "cut_trace": asdict(cut.last_trace) if cut.last_trace is not None else None,
        "claim_boundary": (
            "finite unit behavior only; no truncation, performance, biological, or AGI claim"
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
