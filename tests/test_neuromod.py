"""Neuromodulation tests (F.19)."""

import pytest
from clarus.neuromod import (
    NeuromodulatorState, step_neuromodulators, apply_modulation, ModulationEffect,
)
from clarus.constants import NEURO_BASELINE_DA


class TestNeuromodulatorDynamics:
    def test_baseline_equilibrium(self):
        state = NeuromodulatorState()
        for _ in range(5000):
            state = step_neuromodulators(state)
        assert state.da == pytest.approx(NEURO_BASELINE_DA, abs=0.05)

    def test_da_responds_to_prediction_error(self):
        state = NeuromodulatorState()
        state = step_neuromodulators(state, c_pred=1.0)
        assert state.da > NEURO_BASELINE_DA

    def test_ne_responds_to_novelty(self):
        state = NeuromodulatorState()
        state = step_neuromodulators(state, c_nov=1.0)
        assert state.ne > NEURO_BASELINE_DA

    def test_5ht_responds_to_discount(self):
        state = NeuromodulatorState()
        state = step_neuromodulators(state, discount=1.0)
        assert state.sht < NEURO_BASELINE_DA + 0.1

    def test_ach_responds_to_salience(self):
        state = NeuromodulatorState()
        state = step_neuromodulators(state, salience=1.0)
        assert state.ach > NEURO_BASELINE_DA

    def test_bounded(self):
        state = NeuromodulatorState()
        for _ in range(100):
            state = step_neuromodulators(state, c_pred=10.0, salience=10.0)
        assert 0.0 <= state.da <= 2.0
        assert 0.0 <= state.ne <= 2.0
        assert 0.0 <= state.sht <= 2.0
        assert 0.0 <= state.ach <= 2.0

    def test_time_constants_differ(self):
        state = NeuromodulatorState(da=1.5, ne=1.5, sht=1.5, ach=1.5)
        state = step_neuromodulators(state)
        diffs = [abs(state.da - 1.5), abs(state.ne - 1.5),
                 abs(state.sht - 1.5), abs(state.ach - 1.5)]
        assert len(set(round(d, 6) for d in diffs)) > 1


class TestModulationEffect:
    def test_apply_modulation(self):
        state = NeuromodulatorState()
        effect = apply_modulation(state)
        assert isinstance(effect, ModulationEffect)
        assert effect.n_iter_boost > 0
        assert effect.encode_threshold_scale > 0
        assert effect.temperature_scale > 0

    def test_high_ne_increases_iter(self):
        low_ne = NeuromodulatorState(ne=0.1)
        high_ne = NeuromodulatorState(ne=1.5)
        e_low = apply_modulation(low_ne)
        e_high = apply_modulation(high_ne)
        assert e_high.n_iter_boost > e_low.n_iter_boost

    def test_high_ach_lowers_threshold(self):
        low_ach = NeuromodulatorState(ach=0.1)
        high_ach = NeuromodulatorState(ach=1.5)
        e_low = apply_modulation(low_ach)
        e_high = apply_modulation(high_ach)
        assert e_high.encode_threshold_scale < e_low.encode_threshold_scale
