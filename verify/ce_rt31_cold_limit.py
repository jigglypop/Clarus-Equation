"""RT31 particle stress -> OBS32 cold equal-occupation limit.

Normalized radial Gaussian momentum distribution is a diagnostic family, not
the generated RT31 state. Set coherence u=0 as an explicit averaged-state
assumption. Check mass force and expansion dilution with fixed comoving n.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_genlaguerre, gamma


def run(order=96):
    nodes,weights=roots_genlaguerre(order,.5)
    weights=weights/gamma(1.5)
    def spectrum(theta):
        angles=(theta+2*np.pi*np.arange(3))/3
        return 1+.7*np.cos(angles),-.7/3*np.sin(angles)
    def stress(a,theta,width):
        x,A=spectrum(theta)
        # k^2=2 width^2 t, measure t^(1/2)e^-t dt/Gamma(3/2).
        p2=2*width**2*nodes/a**2
        energies=np.sqrt(x[:,None]+p2[None,:])
        rho=float(np.sum((energies@weights))/a**3)
        pressure=float(np.sum((p2[None,:]/(3*energies))@weights)/a**3)
        source=float(np.sum((A[:,None]/(2*energies))@weights)/a**3)
        return rho,pressure,source
    rows=[]
    for theta in [.1,.5,1.2]:
        x,A=spectrum(theta);cold_energy=np.sqrt(x).sum();cold_force=np.sum(A/(2*np.sqrt(x)))
        for width in [.1,.03,.01]:
            rho,p,J=stress(1,theta,width)
            h=1e-4
            # Five-point differences, independent of analytic mode force.
            derivative=lambda fn:(fn(-2*h)-8*fn(-h)+8*fn(h)-fn(2*h))/(12*h)
            force=derivative(lambda d:stress(1,theta+d,width)[0])
            expansion=derivative(lambda d:stress(np.exp(d),theta,width)[0])
            assert abs(force-J)<1e-10
            assert abs(expansion+3*(rho+p))/rho<1e-10
            rows.append(dict(theta=theta,comoving_momentum_width=width,
                             energy_relative_excess=rho/cold_energy-1,
                             pressure_over_energy=p/rho,
                             cold_force_relative_difference=abs(J/cold_force-1),
                             mass_force_absolute_check=abs(force-J),
                             expansion_continuity_relative_check=abs(expansion+3*(rho+p))/rho))
    for theta in [.1,.5,1.2]:
        selected=[r for r in rows if r['theta']==theta]
        assert all(selected[i+1]['energy_relative_excess']<selected[i]['energy_relative_excess'] for i in range(2))
    return dict(quadrature_order=order,rows=rows,
                occupation='equal fixed comoving total particle numbers per species',
                coherence='u=0 explicitly assumed; not inferred from RT31 rotating state',
                fitted_parameters=[],observational_rmse=None,
                conclusion='Cold mass energy and mass force follow from the same mode stress; the state preparation remains an input.')


if __name__=='__main__':
    result=run();refined=run(192)
    difference=max(abs(a['energy_relative_excess']-b['energy_relative_excess']) for a,b in zip(result['rows'],refined['rows']))
    assert difference<1e-12
    result['quadrature_refinement_max_difference']=difference
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
