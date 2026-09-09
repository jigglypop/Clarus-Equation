import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad

HERE=Path(__file__).resolve().parents[1]/'verify'
saved=sys.path[:]
try:
    sys.path.insert(0,str(HERE))
    from dimension_neutrino_background import NeutrinoBackground
    from dimension_neutrino_stress import population
finally:
    sys.path[:]=saved


def test_momentum_quadrature_against_adaptive_stress_integral():
    model=NeutrinoBackground(momentum_order=96)
    for n in (-4.,-2.,0.):
        rho,p,source,masses=model.neutrinos(n,.02)
        direct=population(masses,model.betas*masses,model.t0*np.exp(-n))
        assert rho==pytest.approx(direct['rho']/model.density_unit,rel=2e-8)
        assert p==pytest.approx(direct['pressure']/model.density_unit,rel=2e-8)
        assert source==pytest.approx(direct['scalar_source_d_rho_dq']/model.density_unit,rel=2e-8)


def test_zero_coupling_recovers_independent_gr_distance():
    model=NeutrinoBackground(betas=(0,0,0)).calibrate()
    def inverse_h(z):
        n=-np.log1p(z)
        rho=model.neutrinos(n,0)[0]
        return np.sqrt(3/(model.dust0*np.exp(-3*n)+model.photon0*np.exp(-4*n)+rho+model.lam))
    actual=model.diagnostics()
    expected=[quad(inverse_h,0,z,epsabs=1e-12)[0] for z in (.38,.698,1.48)]
    assert actual['DM_Href_over_c_at_z_038_0698_148']==pytest.approx(expected,abs=3e-9)
    assert actual['q_today']==0
    assert actual['maximum_raychaudhuri_constraint_error']<1e-8


def test_coupled_energy_constraint_and_refinement():
    model=NeutrinoBackground(betas=(0,1,1)).calibrate()
    fine=NeutrinoBackground(betas=(0,1,1),cells=40,momentum_order=96,rtol=2e-11,atol=2e-13).calibrate()
    a,b=model.diagnostics(),fine.diagnostics()
    assert a['maximum_raychaudhuri_constraint_error']<1e-8
    assert b['maximum_raychaudhuri_constraint_error']<1e-9
    assert a['H0_over_Href']==pytest.approx(1,abs=1e-10)
    assert a['DM_Href_over_c_at_z_038_0698_148']==pytest.approx(b['DM_Href_over_c_at_z_038_0698_148'],abs=1e-9)
    assert abs(a['q_today'])>1e-4
