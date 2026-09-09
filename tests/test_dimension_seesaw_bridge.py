import numpy as np
import pytest
from verify.dimension_seesaw_bridge import match, rank_two_masses


def test_matching_derivative_full_block_and_rank_two_null_vector():
    rng=np.random.default_rng(13)
    d=.03*(rng.normal(size=(3,2))+1j*rng.normal(size=(3,2)))
    dc=.01*(rng.normal(size=(3,2))+1j*rng.normal(size=(3,2)))
    n=np.diag([10.,17.]).astype(complex)
    nc=np.array([[.4,.1j],[.1j,-.2]])
    result=match(d,n,dc,nc)
    h=1e-4
    def mass(q):
        dq=d+q*dc
        return -dq@np.linalg.solve(n+q*nc,dq.T)
    assert (mass(h)-mass(-h))/(2*h) == pytest.approx(result['slope'],abs=1e-13)
    # Independent full (3+2)-state Takagi singular values approach the Schur result.
    errors=[]
    for scale in (1.,.5):
        block=np.block([[np.zeros((3,3)),scale*d],[scale*d.T,n]])
        light=np.sort(np.linalg.svd(block,compute_uv=False))[:3]
        effective=np.sort(np.linalg.svd(scale**2*result['mass'],compute_uv=False))
        errors.append(np.max(abs(light[1:]/effective[1:]-1)))
    # Leading matching error is quadratic in the heavy-light mixing; check
    # convergence instead of confusing a finite approximation error with roundoff.
    assert errors[0] < 2*result['heavy_mixing_norm']**2
    assert errors[1]/errors[0] == pytest.approx(.25,rel=.01)
    for q in (-.2,0,.3):
        assert np.linalg.svd(mass(q),compute_uv=False)[-1] < 1e-18


@pytest.mark.parametrize('ordering,atm',[('NO',.002513),('IO',-.002484)])
def test_readout_preserves_signed_splittings_and_lightest_zero(ordering,atm):
    m=rank_two_masses(.0000749,atm,ordering)
    assert min(m)==0
    assert m[1]**2-m[0]**2 == pytest.approx(.0000749)
    ell=0 if ordering=='NO' else 1
    assert m[2]**2-m[ell]**2 == pytest.approx(atm)


def test_universal_rescaling_matches_single_overall_light_mass_rescaling():
    d=np.array([[.1,0],[0,.2],[.03,.04]])
    n=np.diag([10.,20.])
    alpha=.07
    result=match(d,n,alpha*d,alpha*n)
    assert result['slope'] == pytest.approx(alpha*result['mass'],abs=1e-17)
