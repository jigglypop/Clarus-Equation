"""Deterministic non-evidence demo of the opt-in V9 RuntimeAgent path."""

from __future__ import annotations

import torch

from reality_stone.clarus.agent import RuntimeAgent, RuntimeAgentConfig
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


def main() -> None:
    torch.manual_seed(19)
    weight = torch.zeros(3, 3)
    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=3,
            active_ratio=1.0,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
        ),
        backend="torch",
        device="cpu",
    )
    agent = RuntimeAgent(
        runtime,
        action_embeddings=torch.eye(3),
        config=RuntimeAgentConfig(action_count=3, nested_scc_enabled=True),
    )
    observations = (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    for observation in observations:
        result = agent.step(observation=observation, force_mode=RuntimeMode.WAKE)
        print(
            {
                "tick": result.nested_scc_token.tick,
                "depth": result.nested_scc_token.active_depth,
                "evidence": result.nested_scc_evidence,
                "probabilities": result.nested_scc_policy.probabilities,
                "action": result.action_index,
            }
        )


if __name__ == "__main__":
    main()
