"""Consciousness / metacognition tests (F.17)."""

import pytest
from clarus.agent import ConsciousnessMonitor
from clarus.constants import BOOTSTRAP_CONTRACTION, ACTIVE_RATIO


class TestConsciousnessDepth:
    def test_depth_range(self):
        mon = ConsciousnessMonitor()
        for _ in range(50):
            mon.record_deviation(0.1)
        depth = mon.consciousness_depth()
        assert 0.0 <= depth <= 1.0

    def test_perfect_alignment_high_depth(self):
        mon = ConsciousnessMonitor()
        for _ in range(50):
            mon.record_deviation(ACTIVE_RATIO)
        depth = mon.consciousness_depth()
        assert depth > 0.9

    def test_large_deviation_low_depth(self):
        mon = ConsciousnessMonitor()
        for _ in range(50):
            mon.record_deviation(0.9)
        depth = mon.consciousness_depth()
        assert depth < 0.5

    def test_d_tau_zero_when_empty(self):
        mon = ConsciousnessMonitor()
        assert mon.d_tau() == 0.0


class TestMetacognition:
    def test_recursive_contraction(self):
        mon = ConsciousnessMonitor()
        steps = mon.metacognition_step(1.0)
        assert len(steps) == 3
        for i in range(1, len(steps)):
            assert steps[i] < steps[i - 1]

    def test_contraction_rate(self):
        mon = ConsciousnessMonitor()
        steps = mon.metacognition_step(1.0)
        ratio = steps[1] / steps[0]
        assert ratio == pytest.approx(BOOTSTRAP_CONTRACTION, abs=0.01)

    def test_three_iterations_sufficient(self):
        mon = ConsciousnessMonitor()
        steps = mon.metacognition_step(1.0)
        assert steps[-1] < 0.04


class TestC3SelfConsistency:
    def test_fixed_point_equation(self):
        import math
        from clarus.constants import AD
        d_eff = 3.0 + AD * (1 - AD)
        eps2 = math.exp(-(1 - ACTIVE_RATIO) * d_eff)
        assert eps2 == pytest.approx(ACTIVE_RATIO, abs=0.02)
