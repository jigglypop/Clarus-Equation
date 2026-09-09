"""Independent feasibility checks and affine-mixture restriction."""
from fractions import Fraction as F
import numpy as np
import pytest
from scipy.optimize import linprog
from verify import dimension_pmns_screen as model


@pytest.mark.parametrize("name", ["IC19_NO", "IC19_IO", "IC24_SK_NO", "IC24_SK_IO"])
def test_report_interval_feasibility_against_independent_linear_program(name):
    report = model.build_report()
    # Independently spelled out original repository equations, in 12,23,13 order.
    a = np.array([1/3, 1/2, 0])
    b = np.array([-1/8, 7/16, 1/8])
    for level in ("one_sigma", "three_sigma"):
        intervals = np.array(report["source"]["variants"][name][level])
        lp = linprog([0], A_ub=np.r_[b, -b][:, None],
                     b_ub=np.r_[intervals[:, 1]-a, a-intervals[:, 0]],
                     bounds=[(0, .25)], method="highs")
        result = report["variants"][name][level+"_box"]
        assert lp.success == result["nonempty"]
        assert result["nonempty"] == (level == "three_sigma")
        if result["nonempty"]:
            pred = np.array(model.predict(result["witness_delta"]), float)
            assert np.all(pred >= intervals[:, 0]) and np.all(pred <= intervals[:, 1])
    assert report["joint_rmse"] is None and not report["scientific_success"]


def test_probability_average_cannot_escape_affine_sum_rules():
    weights = (F(1, 7), F(2, 7), F(4, 7))
    deltas = (F(0), F(1, 10), F(1, 4))
    vectors = [model.predict(d) for d in deltas]
    mean = tuple(sum(w*v[i] for w, v in zip(weights, vectors)) for i in range(3))
    assert mean == model.predict(sum(w*d for w, d in zip(weights, deltas)))
    assert mean[0]+mean[2] == F(1, 3)
    assert mean[1]-F(7, 2)*mean[2] == F(1, 2)
