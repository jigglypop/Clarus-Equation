"""상위 길이의 평균 사상과 게이지 방향 적분의 독립 저차원 검산."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location("coarse_map_under_test",HERE/"coarse_length_map.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_delta_slice_changes_physical_covariance_but_integration_does_not():
    precision = np.array([[2.,1.],[1.,2.]])
    first = np.array([[1.],[0.]])
    second = np.array([[1.],[1.]])
    gauge = np.array([[0.],[1.]])
    assert module.covariance_on_slice(first,precision)[0,0] == pytest.approx(1/2)
    assert module.covariance_on_slice(second,precision)[0,0] == pytest.approx(1/6)
    assert module.marginalized_covariance(first,gauge,precision)[0,0] == pytest.approx(2/3)
    assert module.marginalized_covariance(second,gauge,precision)[0,0] == pytest.approx(2/3)


def test_average_readout_recovers_shared_edge_value():
    cells=[(0,1,2,3,4),(0,1,2,3,5)]
    edge=(0,1)
    readout=module.edge_readout(cells,[edge])
    fine=np.zeros(20)
    fine[0]=3.
    fine[10]=3.
    assert (readout @ fine)[0] == 3.
    fine[0]=2.
    fine[10]=4.
    assert (readout @ fine)[0] == 3.


def test_missing_coarse_edge_is_rejected():
    with pytest.raises(ValueError):
        module.edge_readout([(0,1,2,3,4)],[(0,5)])
