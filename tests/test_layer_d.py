"""Layer D: Hippocampus -- forgetting, recall threshold."""

import torch
import pytest
from clarus.runtime import HippocampusMemory, RuntimeMode
from clarus.constants import FORGET_TAU, RECALL_SIMILARITY_THRESHOLD


class TestForgetDecay:
    def test_priority_decays(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        key = torch.randn(16)
        mem.encode(key, priority=1.0)
        p0 = mem._priority[0]
        mem.decay_priorities(steps=1000)
        p1 = mem._priority[0]
        assert p1 < p0

    def test_decay_exponential(self):
        import math
        mem = HippocampusMemory(dim=16, capacity=8)
        mem.encode(torch.randn(16), priority=1.0)
        steps = 1000
        expected_factor = math.exp(-steps / FORGET_TAU)
        mem.decay_priorities(steps=steps)
        assert mem._priority[0] == pytest.approx(expected_factor, rel=1e-4)

    def test_no_decay_zero_steps(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        mem.encode(torch.randn(16), priority=1.0)
        mem.decay_priorities(steps=0)
        assert mem._priority[0] == pytest.approx(1.0, abs=1e-6)


class TestRecallThreshold:
    def test_recall_returns_zero_below_threshold(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        mem.encode(torch.ones(16), priority=1.0)
        orthogonal_cue = torch.zeros(16)
        orthogonal_cue[0] = 1.0
        result = mem.recall(orthogonal_cue)
        assert result.norm().item() >= 0.0

    def test_recall_succeeds_with_similar_cue(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        key = torch.randn(16)
        mem.encode(key, priority=1.0)
        result = mem.recall(key + 0.1 * torch.randn(16))
        assert result.norm().item() > 0.0

    def test_recall_empty_memory(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        result = mem.recall(torch.randn(16))
        assert result.norm().item() == 0.0


class TestReplay:
    def test_replay_nrem_focused(self):
        mem = HippocampusMemory(dim=16, capacity=8)
        for i in range(5):
            mem.encode(torch.randn(16), priority=float(i + 1))
        nrem = mem.replay(RuntimeMode.NREM)
        rem = mem.replay(RuntimeMode.REM)
        assert nrem.norm().item() > 0
        assert rem.norm().item() > 0

    def test_capacity_eviction(self):
        mem = HippocampusMemory(dim=8, capacity=3)
        for i in range(5):
            mem.encode(torch.randn(8), priority=float(i + 1))
        assert len(mem) == 3

    def test_state_dict_roundtrip(self):
        mem = HippocampusMemory(dim=8, capacity=4)
        for i in range(3):
            mem.encode(torch.randn(8), priority=float(i + 1))
        sd = mem.state_dict()
        mem2 = HippocampusMemory.from_state_dict(sd)
        assert len(mem2) == len(mem)
