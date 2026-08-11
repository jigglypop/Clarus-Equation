from dataclasses import asdict

import pytest

from reality_stone.clarus.dimensionless import (
    DIMENSIONLESS,
    ENERGY,
    TIME,
    Quantity,
    audit_dimensionless,
    nondimensionalize,
    require_dimensionless,
)
from reality_stone.clarus.dual_scc_basal_ganglia import DualSCCConfig
from reality_stone.clarus.dual_scc_controller import DualSCCControllerConfig


def test_all_dual_scc_core_and_controller_coefficients_are_dimensionless() -> None:
    quantities = [
        Quantity(name, float(value), DIMENSIONLESS)
        for config in (DualSCCConfig(), DualSCCControllerConfig())
        for name, value in asdict(config).items()
        if not isinstance(value, int)
    ]
    result = audit_dimensionless(quantities, context="dual-SCC tanh/policy core")
    assert result.passed


def test_physical_time_and_reward_require_named_reference_scales() -> None:
    elapsed = Quantity("Delta_t", 0.020, TIME)
    time_scale = Quantity("tau_fast", 0.050, TIME)
    reward = Quantity("reward", 2.0, ENERGY)
    reward_scale = Quantity("reward_0", 4.0, ENERGY)
    with pytest.raises(ValueError, match="must be dimensionless"):
        require_dimensionless(elapsed, context="dual-SCC input")
    with pytest.raises(ValueError, match="must be dimensionless"):
        require_dimensionless(reward, context="dual-SCC input")
    assert nondimensionalize(elapsed, [time_scale]).value == pytest.approx(0.4)
    assert nondimensionalize(reward, [reward_scale]).value == pytest.approx(0.5)
