"""Joint cooling/abundance check of the exact weak-pulse distributions.

Assume dephased conserved occupations and fixed theta=0 after the prescribed
pulse. Scale factor B is expansion since production, not OBS32's ai.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from ce_symmetric_pulse_generation import run_case
from ce_symmetric_small_f_stability import ce, StableLightSpectrum


def evaluate(r,amplitude,order=96,cutoff=10.,tails=24.):
    original,(k,weights,n)=run_case(r=r,amplitude=amplitude,order=order,cutoff=cutoff,tails=tails,return_modes=True)
    sp=StableLightSpectrum(ce.Parameters(r,0.))
    x,x1,x2=sp.eigs(0.)
    numbers=2*(n@weights)
    Utop=sp.top
    U2=sp.evaluate(0.)[2]*Utop
    def stress(B):
        momentum2=k[None,:]**2/B**2
        energy=np.sqrt(x[:,None]+momentum2)
        rho=float(np.sum(2*n*weights*energy)/B**3)
        pressure=float(np.sum(2*n*weights*momentum2/(3*energy))/B**3)
        rest=float(numbers@np.sqrt(x)/B**3)
        curvature=float(np.sum(2*n*weights*(x2[:,None]/(2*energy)-x1[:,None]**2/(4*energy**3)))/B**3)
        speed2=float(np.sum(2*n*weights*momentum2/energy**2)/numbers.sum())
        return dict(B=B,rho_over_Utop=rho/Utop,pressure_over_rho=pressure/rho,
                    kinetic_fraction=1-rest/rho,number_weighted_rms_speed=np.sqrt(speed2),
                    rest_over_Utop=rest/Utop,particle_curvature_over_abs_vacuum=curvature/abs(U2))
    initial=stress(1.)
    coldB=brentq(lambda B:stress(B)['kinetic_fraction']-.01,1,100,xtol=1e-11)
    cold=stress(coldB)
    # For all B>=1, drop the negative term in E'' and bound 1/E<=1/sqrt(x).
    # Heavy curvature is negative, so omit it for a rigorous particle upper bound.
    curvature_upper=float(np.sum(numbers*np.maximum(x2,0)/(2*np.sqrt(x))))
    assert curvature_upper/abs(U2)<1
    a=2.;h=1e-4
    def energy(logB):return stress(np.exp(logB))['rho_over_Utop']
    loga=np.log(a)
    derivative=(energy(loga-2*h)-8*energy(loga-h)+8*energy(loga+h)-energy(loga+2*h))/(12*h)
    state=stress(a)
    wanted=-3*state['rho_over_Utop']*(1+state['pressure_over_rho'])
    continuity=abs(derivative/wanted-1)
    assert continuity<1e-10
    return dict(r=r,amplitude=amplitude,initial=initial,one_percent_kinetic_state=cold,
                fixed_OBS32_dark_to_potential_ratio=.84*3/7,
                cold_density_over_fixed_OBS32_dark_density=cold['rho_over_Utop']/(.84*3/7),
                all_expansion_particle_curvature_upper_over_abs_vacuum=curvature_upper/abs(U2),
                expansion_energy_continuity_relative_error=continuity,
                number_fraction=original['species_fractions'])


def run():
    cases=[evaluate(r,a) for r in [.15,.35] for a in [.1,.05,.025]]
    checks=[]
    for r in [.15,.35]:
        coarse=next(c for c in cases if c['r']==r and c['amplitude']==.1)
        fine=evaluate(r,.1,order=192,cutoff=12.,tails=28.)
        err=abs(fine['cold_density_over_fixed_OBS32_dark_density']/coarse['cold_density_over_fixed_OBS32_dark_density']-1)
        assert err<1e-7
        checks.append(dict(r=r,refined_cold_density_relative_difference=err))
    return dict(cases=cases,refinement=checks,fitted_parameters=[],
                hypotheses=['Single prescribed weak pulse; fixed post-pulse theta=0.',
                            'Discard coherence explicitly; conserve species occupations while redshifting momenta.',
                            'Use 1 percent kinetic fraction as a diagnostic coldness criterion, not an observational fit.'],
                self_consistent_history=False,full_joint_rmse=None,
                conclusion='Correct species ratios alone do not supply the required cold abundance or enough stabilizing matter curvature.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
