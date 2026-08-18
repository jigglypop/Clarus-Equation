from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT
    / "reality_stone"
    / "python"
    / "reality_stone"
    / "clarus"
    / "cosmology_registry.py"
)


def _load_registry():
    """Load the standard-library math core without importing the ML facade."""

    module_name = "ce_bootstrap_math_registry"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_low_extinction_root_has_a_machine_precision_certificate() -> None:
    registry = _load_registry()
    result = registry.solve_low_extinction_root(
        3.17776,
        model_id="TEST_LEGACY_DELTA_5DP",
        precision="binary64 test input",
    )

    assert math.isclose(result.q_ext, 0.04864663333721408, rel_tol=0.0, abs_tol=1e-16)
    assert result.absolute_residual <= 8.0 * math.ulp(result.q_ext)
    assert 0.0 < result.q_ext < 1.0 / result.d_eff
    assert result.contraction < 1.0
    assert result.survival == 1.0 - result.q_ext


def test_bootstrap_residual_derivative_matches_central_difference() -> None:
    registry = _load_registry()
    d_eff = 3.17776
    q_ext = 0.04864663333721408
    step = 1e-6
    numerical = (
        registry.bootstrap_residual(q_ext + step, d_eff)
        - registry.bootstrap_residual(q_ext - step, d_eff)
    ) / (2.0 * step)
    analytic = 1.0 - d_eff * math.exp(-d_eff * (1.0 - q_ext))

    assert math.isclose(analytic, numerical, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("d_eff", [1.0, 0.0, -1.0, math.inf, math.nan])
def test_low_extinction_root_rejects_invalid_effective_depth(d_eff: float) -> None:
    registry = _load_registry()

    with pytest.raises(ValueError, match="finite D > 1"):
        registry.solve_low_extinction_root(
            d_eff,
            model_id="INVALID_DEPTH",
            precision="invalid test input",
        )
