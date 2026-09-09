import numpy as np
import pytest
from scipy.special import zeta
from verify.dimension_neutrino_stress import moments,population


def test_massless_and_nonrelativistic_limits_with_absolute_normalization():
    zero=moments(0,1)
    assert zero['rho']==pytest.approx(7*np.pi**2/120,rel=1e-11)
    assert zero['w']==pytest.approx(1/3)
    assert zero['trace']==0 and zero['drho_dm']==0
    assert zero['number_density']==pytest.approx(3*zeta(3)/(2*np.pi**2),rel=1e-11)
    nr=moments(1e4,1)
    assert nr['rho']/(1e4*nr['number_density'])==pytest.approx(1,rel=1e-7)
    assert nr['w']<1e-7


@pytest.mark.parametrize('ratio',[.001,1,100])
def test_mass_derivative_trace_and_expanding_continuity(ratio):
    m,t=ratio*.01,.01
    r=moments(m,t)
    h=1e-4
    # Along N=ln(a): T=T0 exp(-N), m=m0 exp(beta N).
    beta=.23
    rp=moments(m*np.exp(beta*h),t*np.exp(-h))['rho']
    rm=moments(m*np.exp(-beta*h),t*np.exp(h))['rho']
    predicted=-3*(r['rho']+r['pressure'])+beta*r['trace']
    assert (rp-rm)/(2*h)==pytest.approx(predicted,rel=5e-8)
    assert m*r['drho_dm']==pytest.approx(r['trace'],rel=1e-12)
    assert r['rho']-3*r['pressure']==pytest.approx(r['trace'],abs=1e-22,rel=1e-8)


def test_scalar_source_uses_mass_derivatives_without_massless_log_singularity():
    masses=np.array([0,.01,.05]); derivatives=np.array([0,.002,-.003])
    t=.003;h=1e-4
    result=population(masses,derivatives,t)
    def rho(q):
        return population(masses+q*derivatives,derivatives,t)['rho']
    assert (rho(h)-rho(-h))/(2*h)==pytest.approx(result['scalar_source_d_rho_dq'],rel=1e-8)
