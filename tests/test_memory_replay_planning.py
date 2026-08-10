from pathlib import Path
import numpy as np
from reality_stone.clarus.memory_replay_planning import PriorityReplayMemory, run_memory_replay_gate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "memory_replay_planning_v1.json"

def test_priority_memory_respects_capacity() -> None:
    memory = PriorityReplayMemory(2, np.ones(2), 0.2, 0.5)
    for value in (np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])):
        memory.observe(value)
    assert len(memory.items) == 2

def test_gate_uses_only_bounded_prototypes() -> None:
    report = run_memory_replay_gate(CONFIG)
    assert report["prototype_bytes"] <= 1024
    assert report["resource_usage"]["external_download_bytes"] == 0
