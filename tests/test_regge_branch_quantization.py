"""실제 가지 역함수와 독립 적분·파동 미분으로 양자 전달을 대조한다."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"verify"/"Q-0020"))
import regge_branch_quantization as model


@pytest.mark.parametrize("h,kind,phase",[(.9,"length",0.),(1.,"squared",20.),(1.1,"length",5.)])
def test_full_kernel_and_norm_survive_both_branches(h,kind,phase):
    result = model.coarea_check(h,kind,phase,order=192)
    assert result["kernel_error"] < 1e-7
    assert result["norm_error"] < 1e-7
    assert result["normalization_error"] < 1e-7
    assert result["drop_right_error"] > 1e-4
    assert result["omit_jacobian_error"] > 1e-3


@pytest.mark.parametrize("kind",["length","squared"])
def test_zero_phase_branch_mass_is_geometric(kind):
    result = model.coarea_check(1.,kind,0.,order=192)
    fraction = 2/model.full.limit(1.)
    expected = [fraction,1-fraction] if kind=="length" else [fraction**2,1-fraction**2]
    np.testing.assert_allclose(result["branch_masses"],expected,atol=1e-8,rtol=0)


def test_distinct_self_adjoint_junctions_change_spectrum():
    result = model.boundary_check()
    left,right = result["extensions"]
    assert left["zero_multiplicity"]==2 and right["zero_multiplicity"]==1
    assert min(left["positive_frequencies"]) > right["positive_frequencies"][0]
    assert max(x["boundary_form_error"] for x in result["extensions"]) < 1e-14
    assert max(x["spectral_boundary_error"] for x in result["extensions"]) < 1e-14
    assert result["local_only_defect"] == [0.,-2.]


def test_actual_fold_density_has_integrable_inverse_square_root():
    result = model.critical_density_check()
    errors = [abs(row["ratio"]-1) for row in result["rows"]]
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 3e-5


def test_original_uniform_state_is_not_in_flat_momentum_derivative_domain():
    result = model.domain_check()
    rows = result["rows"]
    assert rows[-1]["derivative_integral"] > 60*rows[0]["derivative_integral"]
    assert abs(rows[-1]["ratio"]-1)<2e-5


def test_inherited_constraint_keeps_parent_eigenstates():
    result = model.inherited_constraint_check()
    assert max(row["error"] for row in result["rows"])<1e-7
    assert max(row["omit_connection_error"] for row in result["rows"])>.1
    assert result["outer_trace_error"]<1e-14


@pytest.mark.parametrize("edge",[1.2,2.,2.08])
def test_curvature_used_in_domain_integral_matches_independent_difference(edge):
    step=1e-5
    derivative=(model.projection.symmetric_g_e(edge+step,1.)-
                model.projection.symmetric_g_e(edge-step,1.))/(2*step)
    assert math.isclose(model.g_second(edge),derivative,rel_tol=2e-7,abs_tol=1e-7)


def test_branch_map_rejects_degenerate_or_invalid_charts():
    for h,beta in [(.5,5.),(1.,0.),(1.,-1.)]:
        with pytest.raises(ValueError):
            model.branch_data(h,beta)
    data=model.branch_data()
    with pytest.raises(ValueError):
        model.inverse_momentum(data["pc"],0,data)

