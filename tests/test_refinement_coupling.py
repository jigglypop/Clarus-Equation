"""부모에게 보이지 않는 결합 자유도와 서로 다른 방출 결과를 검산한다."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("refinement_coupling_under_test", HERE / "refinement_coupling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parent_observations_cannot_distinguish_fine_couplings():
    old, new, transfer, innovation = module.fixture()
    parent = np.eye(2) - np.outer(old, old)
    for beta in (0., .3, 2., 10.):
        coupling = module.refined_coupling(beta)
        gram = coupling.T @ coupling
        np.testing.assert_allclose(transfer.T @ gram @ transfer, parent, atol=1e-13, rtol=0)
        np.testing.assert_allclose(coupling @ new, 0, atol=1e-13)
        assert innovation @ gram @ innovation == pytest.approx(1 + beta)


def test_exact_symbolic_counterexample_and_distinct_emission_results():
    result = module.run()
    assert all(result["exact_symbolic_checks"].values())
    free, trapped = result["cases"]
    assert free["new_mode_spectrum"]["bound_states"] == 0
    assert trapped["new_mode_spectrum"]["bound_states"] == 2
    assert trapped["new_mode_spectrum"]["long_time_mean_survival"] == pytest.approx(1 / 8)
    assert not result["microscopic_coupling_derived_from_CE"]


def test_copying_parent_does_not_populate_new_antisymmetric_mode():
    _, _, transfer, innovation = module.fixture()
    copied = transfer @ np.array([2., -3.])
    assert innovation @ copied == pytest.approx(0, abs=1e-14)
    assert innovation @ innovation == pytest.approx(1)


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_invalid_new_mode_strength_is_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        module.refined_coupling(beta)
