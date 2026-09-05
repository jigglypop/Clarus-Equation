"""실제 그램 기하와 해가 알려진 위상 모형으로 보존 전달을 검증한다."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"verify"/"Q-0020"))
import regge_conditional_dynamics as model


@pytest.mark.parametrize("h",[.5,1.,1.1])
def test_joint_gram_domain_and_full_canonical_update(h):
    result=model.geometry_check(h)
    assert result["action_error"]<1e-10
    assert result["canonical_error"]<1e-10
    assert result["support_error"]<1e-10
    assert result["outside_rejections"]==3
    assert result["minimum_gram"]>0


def test_product_measure_has_exact_interaction_budget_and_spectrum():
    joint=np.full((2,2),.25)
    action=np.array([[1.,-1.],[-1.,1.]])
    assert model.interaction_budget(joint,action)["coefficient"]==pytest.approx(1.)
    result=model.transfer_check(joint,action,.2)
    assert result["norm_squared"]==pytest.approx(math.cos(.2)**2,abs=1e-14)
    assert result["optimal_remainder"]==pytest.approx(math.sin(.2)**2,abs=1e-14)
    assert result["budget_error"]<1e-14


@pytest.mark.parametrize("h,projection_rank,pullback_rank",[(.9,28,28),(1.,27,26),(1.1,28,28)])
def test_lagrangian_relation_survives_singular_projections(h,projection_rank,pullback_rank):
    result=model.canonical_relation_check(h)
    assert result["relation_rank"]==28
    assert result["input_projection_rank"]==projection_rank
    assert result["output_projection_rank"]==projection_rank
    assert result["pullback_rank"]==pullback_rank
    assert result["lagrangian_error"]<1e-8


def test_discrete_phase_revival_does_not_imply_continuous_twist_exception():
    joint=np.full((2,2),.25)
    action=np.array([[1.,-1.],[-1.,1.]])
    result=model.transfer_check(joint,action,math.pi/2)
    assert result["norm_squared"]==pytest.approx(1.,abs=1e-14)


@pytest.mark.parametrize("kind",["length","squared"])
def test_actual_measure_budget_converges_and_boundary_phases_cancel(kind):
    coarse=model.mesh_check(64,kind=kind)
    fine=model.mesh_check(128,kind=kind)
    assert abs(fine["coefficient"]/coarse["coefficient"]-1)<.02
    assert fine["volume_error"]<1e-9
    assert fine["minimum_gram"]>0
    assert fine["normal_equation_error"]<1e-10
    assert fine["boundary_phase_coefficient_error"]<1e-12
    assert fine["separable_coefficient"]<1e-20
    assert fine["separable_norm_squared"]==pytest.approx(1.,abs=1e-13)
    rows=fine["transfer"]
    assert rows[0]["norm_squared"]==pytest.approx(1.,abs=1e-13)
    assert rows[0]["second_singular"]<.8
    assert all(0<r["norm_squared"]<1 for r in rows[1:])
    assert max(r["budget_error"] for r in rows)<1e-12
    assert max(r["spectral_budget_error"] for r in rows)<1e-12
    assert abs(rows[1]["weak_phase_ratio"]/fine["coefficient"]-1)<1e-4
    # 약한 위상 계수의 오차는 위상 제곱에 비례한다.
    errors=[abs(r["weak_phase_ratio"]-fine["coefficient"]) for r in rows[1:4]]
    assert 3.9<errors[1]/errors[0]<4.1
    assert 3.9<errors[2]/errors[1]<4.1
    assert fine["constant_energy_cross_error"]<1e-12
    assert fine["boundary_phase_energy_error"]<1e-12
    for first,last in zip(coarse["energy"],fine["energy"]):
        for key in ("minimum","maximum"):
            assert abs(last[key]/first[key]-1)<.02
        assert last["minimum"]<0<last["maximum"]
        assert last["hermitian_error"]<1e-12
        assert last["diagonal_phase_error"]<1e-12
        assert last["spectral_error"]<1e-12
        assert max(s["budget_error"] for s in last["states"])<1e-12
        assert max(s["norm_error"] for s in last["states"])<1e-12
    assert fine["energy"][0]["zero_phase_identity_error"]<1e-12
    assert fine["energy"][0]["positive_input"]["cross"]>0


def test_exact_separable_boundary_phase_preserves_singular_spectrum():
    mesh=model.joint_mesh(24)
    action=mesh["action"]
    row=np.linspace(-.8,.2,24)
    column=np.linspace(.5,-.3,24)
    first=model.transfer_check(mesh["joint"],action,1.)
    second=model.transfer_check(mesh["joint"],action+row[:,None]+column[None,:],1.)
    assert abs(first["norm_squared"]-second["norm_squared"])<1e-13


def test_domain_rejects_invalid_lengths_and_mesh_inputs():
    with pytest.raises(ValueError):
        model.y_limit(model.full.limit(1.)*1.001)
    with pytest.raises(ValueError):
        model.e_limit(float(model.y_limit(0.))*1.001)
    with pytest.raises(ValueError):
        model.joint_mesh(4)


@pytest.mark.parametrize("beta",[0.,1.,math.pi/2,5.])
def test_two_point_energy_cross_term_is_exact_and_phase_independent(beta):
    joint=np.full((2,2),.25)
    action=np.array([[1.,-1.],[-1.,1.]])
    levels=np.array([1.,2.])
    result=model.energy_check(joint,action,beta,levels)
    assert result["minimum"]==pytest.approx(-.5,abs=1e-14)
    assert result["maximum"]==pytest.approx(.5,abs=1e-14)
    assert max(s["budget_error"] for s in result["states"])<1e-14
    assert result["spectral_error"]<1e-14
    first,last=result["states"]
    assert first["total"]==pytest.approx(1.)
    assert last["total"]==pytest.approx(2.)
    assert first["kept"]+first["remainder"]==pytest.approx(1.5)
    assert last["kept"]+last["remainder"]==pytest.approx(1.5)


def test_probe_energy_requires_positive_finite_levels():
    joint=np.full((2,2),.25)
    for levels in ([1.],[-1.,2.],[float("nan"),2.]):
        with pytest.raises(ValueError):
            model.energy_cross_matrix(joint,np.zeros((2,2)),1.,levels)
