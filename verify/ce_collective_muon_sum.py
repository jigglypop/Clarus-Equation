"""Chapter 13 phase portal: include every canonical collective mode.

Unit endpoint coefficient is a disclosed input. This is a bound on the
pseudoscalar one-loop subset, not the full muon prediction.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from common_spectrum_muon import M_MU


def run():
    q, N, f_over_Mp, Mp = 3, 4, .05, 2.435e18
    f = f_over_Mp*Mp
    D = np.eye(N, N+1)-q*np.eye(N, N+1, k=1)
    ell = q**np.arange(N, -1, -1, dtype=float)
    slow = ell/np.linalg.norm(ell)
    # Orthogonal canonical modes: y_jk = c_j*m_mu*O_jk/f.
    values, modes = np.linalg.eigh(D.T@D)
    completeness_error = float(np.max(abs(modes@modes.T-np.eye(N+1))))
    assert completeness_error < 2e-15
    total_bound = M_MU**2/(16*np.pi**2*f**2)
    rows = []
    for endpoint in range(N+1):
        squared = (M_MU/f*modes[endpoint])**2
        assert abs(squared.sum()/(M_MU/f)**2-1) < 2e-15
        rows.append(dict(endpoint=endpoint, supplied_portal_coefficient=1,
                         slow_fraction=float(slow[endpoint]**2),
                         other_modes_fraction=float(1-slow[endpoint]**2),
                         total_abs_delta_a_upper_bound=total_bound))
    # Check the kernel bound without assigning a cosmological mass scale.
    integrals = []
    for ratio in [0., .1, 1., 10., 100.]:
        integral = quad(lambda x: x**3/(x*x+(1-x)*ratio**2), 0, 1,
                        epsabs=1e-13, epsrel=1e-12)[0]
        assert 0 < integral <= .5+1e-14
        integrals.append(dict(mediator_over_muon_mass=ratio, integral=integral))
    return dict(source='chapter 13 equation 13.7, all canonical modes',
                identity='sum_k y_jk^2 = c_j^2 m_mu^2/f^2',
                sign='nonpositive for this pseudoscalar one-loop subset',
                q=q,N=N,f_over_Mp=f_over_Mp,Mp_GeV=Mp,
                completeness_error=completeness_error,endpoints=rows,kernel_checks=integrals,
                fitted_parameters=[],joint_rmse=None,
                limitation='Conditional single endpoint portal; electroweak completion, anomaly terms and other diagrams not included.')


if __name__ == '__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
