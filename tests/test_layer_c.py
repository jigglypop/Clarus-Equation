"""Layer C: Global mode, circadian, NREM length."""

import math
import torch
import pytest
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from clarus.constants import CIRCADIAN_PERIOD, NREM_LENGTH_DECAY


def make_runtime(dim=32):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(dim, dim, generator=g)
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    cfg = BrainRuntimeConfig(dim=dim, dale_law=False, axon_delay=False, noise_sigma=0.0)
    return BrainRuntime(w, config=cfg, backend="torch")


class TestCircadian:
    def test_circadian_phase_advances(self):
        rt = make_runtime()
        assert rt.circadian_phase == 0.0
        rt.step()
        assert rt.circadian_phase > 0.0

    def test_circadian_periodic(self):
        rt = make_runtime()
        values = []
        for _ in range(100):
            rt.step()
            values.append(rt._circadian_value)
        assert all(isinstance(v, float) for v in values)
        assert all(abs(v) < 2.0 for v in values)


class TestNREMLength:
    def test_nrem_length_decreases(self):
        rt = make_runtime()
        lengths = []
        for i in range(5):
            rt.nrem_cycle_count = i
            lengths.append(rt.nrem_target_length())
        for i in range(1, len(lengths)):
            assert lengths[i] < lengths[i - 1]

    def test_nrem_length_decay_rate(self):
        rt = make_runtime()
        rt.nrem_cycle_count = 0
        l0 = rt.nrem_target_length()
        rt.nrem_cycle_count = 1
        l1 = rt.nrem_target_length()
        ratio = l1 / l0
        assert ratio == pytest.approx(NREM_LENGTH_DECAY, abs=0.01)


class TestModeTransition:
    def test_wake_to_nrem_on_pressure(self):
        rt = make_runtime()
        rt.sleep_pressure = 1.5
        rt.mode = RuntimeMode.WAKE
        new_mode = rt._auto_mode(0.0)
        assert new_mode == RuntimeMode.NREM

    def test_nrem_to_rem(self):
        rt = make_runtime()
        rt.mode = RuntimeMode.NREM
        rt.sleep_pressure = 0.3
        new_mode = rt._auto_mode(0.0)
        assert new_mode == RuntimeMode.REM

    def test_rem_to_wake(self):
        rt = make_runtime()
        rt.mode = RuntimeMode.REM
        rt.sleep_pressure = 0.1
        new_mode = rt._auto_mode(0.5)
        assert new_mode == RuntimeMode.WAKE

    def test_energy_budget_by_mode(self):
        cfg = BrainRuntimeConfig(dim=100, dale_law=False, axon_delay=False)
        bw = cfg.energy_budget(RuntimeMode.WAKE)
        bn = cfg.energy_budget(RuntimeMode.NREM)
        br = cfg.energy_budget(RuntimeMode.REM)
        assert bn < bw
        assert br < bw
        assert bn <= br
