"""Working memory + cerebellum tests (F.20)."""

import torch
import pytest
from clarus.agent import WorkingMemory, CerebellumPredictor
from clarus.constants import WM_CAPACITY


class TestWorkingMemory:
    def test_fifo_capacity(self):
        wm = WorkingMemory(capacity=WM_CAPACITY)
        for i in range(20):
            wm.append(f"action_{i}", f"obs_{i}")
        assert len(wm) == WM_CAPACITY

    def test_fifo_order(self):
        wm = WorkingMemory(capacity=3)
        wm.append("a0", "o0")
        wm.append("a1", "o1")
        wm.append("a2", "o2")
        wm.append("a3", "o3")
        contents = wm.contents()
        assert contents[0] == ("a1", "o1")
        assert contents[-1] == ("a3", "o3")

    def test_empty(self):
        wm = WorkingMemory()
        assert len(wm) == 0
        assert wm.contents() == []


class TestCerebellum:
    def test_prediction_converges(self):
        cb = CerebellumPredictor(dim=4, alpha=0.1)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for _ in range(200):
            cb.update(target)
        pred = cb.predict()
        assert torch.allclose(pred, target, atol=0.1)

    def test_correction_direction(self):
        cb = CerebellumPredictor(dim=4)
        obs = torch.ones(4)
        correction = cb.update(obs)
        assert correction.shape == (4,)

    def test_initial_prediction_zero(self):
        cb = CerebellumPredictor(dim=8)
        pred = cb.predict()
        assert pred.norm().item() == 0.0
