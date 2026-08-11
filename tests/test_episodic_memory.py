import torch

from reality_stone.clarus.episodic_memory import AuditedEpisodicMemory
from reality_stone.clarus.episodic_memory_benchmark import (
    EpisodicMemoryBenchConfig,
    evaluate_episodic_memory,
)


def test_update_delete_and_abstention_are_audited() -> None:
    memory = AuditedEpisodicMemory(4, capacity=2)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert memory.upsert(key, 1, "old", priority=2.0, timestamp=0) == "ADD"
    assert memory.upsert(key, 2, "new", priority=1.0, timestamp=1) == "UPDATE"
    recalled = memory.recall(key)
    assert (recalled.value, recalled.evidence_id) == (2, "new")
    assert memory.recall(torch.tensor([0.0, 1.0, 0.0, 0.0])).abstained
    assert memory.delete("new")
    assert memory.recall(key).abstained
    assert [row["operation"] for row in memory.audit_log] == ["ADD", "UPDATE", "DELETE"]


def test_small_episodic_benchmark_is_bounded() -> None:
    result = evaluate_episodic_memory(EpisodicMemoryBenchConfig(seeds=4))
    assert result["schema"] == "clarus.episodic-memory.validation.v1"
    assert result["means"]["candidate"]["capacity_ok"] == 1.0
