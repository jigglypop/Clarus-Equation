import torch

from reality_stone.clarus.agent import RuntimeAgent, RuntimeAgentConfig
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from reality_stone.clarus.runtime_temporal_memory import (
    RuntimeTemporalAgent,
    TemporalAgentQuery,
    TemporalMemoryController,
)
from reality_stone.clarus.temporal_memory import (
    TemporalAuditedMemory,
    TemporalMemoryEvent,
    TemporalOperation,
)


def make_runtime(dim: int = 8) -> BrainRuntime:
    weight = torch.zeros(dim, dim)
    return BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=dim,
            active_ratio=0.25,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
        ),
        backend="torch",
        device="cpu",
    )


def make_wrapper(*, enabled: bool) -> tuple[RuntimeTemporalAgent, TemporalAuditedMemory]:
    runtime = make_runtime()
    base = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=4))
    memory = TemporalAuditedMemory(capacity=16)
    wrapper = RuntimeTemporalAgent(
        base,
        controller=TemporalMemoryController(memory, enabled=enabled),
        answer_action_index=0,
        abstain_action_index=1,
    )
    return wrapper, memory


def test_disabled_default_preserves_base_action_and_does_not_recall() -> None:
    wrapper, memory = make_wrapper(enabled=False)
    output = wrapper.step(
        query=TemporalAgentQuery("q", "fact", "p", relation="lives_in"),
        observation=torch.ones(8),
        force_mode=RuntimeMode.WAKE,
    )
    assert output.action_index == output.base_step.action_index
    assert output.decision.route == "disabled"
    assert memory.recall_count == 0


def test_context_precedence_does_not_touch_long_term_memory() -> None:
    wrapper, memory = make_wrapper(enabled=True)
    memory.ingest(TemporalMemoryEvent("p", "lives_in", "stored", 1, 1, "e1"))
    output = wrapper.step(
        query=TemporalAgentQuery(
            "q",
            "context",
            "p",
            relation="lives_in",
            context_value="current-context",
            context_evidence_id="ctx",
        ),
        observation=torch.ones(8),
        force_mode=RuntimeMode.WAKE,
    )
    assert output.action_index == 0
    assert output.value == "current-context"
    assert output.evidence_id == "ctx"
    assert memory.recall_count == 0


def test_memory_answer_then_tombstone_abstention() -> None:
    wrapper, memory = make_wrapper(enabled=True)
    memory.ingest(TemporalMemoryEvent("p", "lives_in", "city", 1, 1, "e1"))
    answer = wrapper.step(
        query=TemporalAgentQuery("q1", "fact", "p", relation="lives_in"),
        observation=torch.ones(8),
        force_mode=RuntimeMode.WAKE,
    )
    assert answer.action_index == 0
    assert answer.value == "city"
    assert answer.evidence_id == "e1"

    memory.ingest(
        TemporalMemoryEvent(
            "p",
            "lives_in",
            None,
            2,
            1,
            "e2",
            operation=TemporalOperation.DELETE,
        )
    )
    deleted = wrapper.step(
        query=TemporalAgentQuery("q2", "fact", "p", relation="lives_in"),
        observation=torch.ones(8),
        force_mode=RuntimeMode.WAKE,
    )
    assert deleted.action_index == 1
    assert deleted.value is None
    assert deleted.decision.abstained
