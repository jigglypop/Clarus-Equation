"""Closed RT31 small-kick research, with Friedmann-driven expansion.

Uses the existing RT31 preparation and OBS32 dimensionless mass ratio. These
instantaneous n_j are not asymptotic particles from the external-pulse test.
"""
import json
from pathlib import Path
import numpy as np
from ce_rt31_flrw_reproduction import run as evolve


def preparation_numbers(H, velocity, r, cutoff):
    x=np.array([1+2*r,1-r,1-r])
    derivative=np.array([0.,-r/np.sqrt(3),r/np.sqrt(3)])
    z=cutoff/np.sqrt(x)
    integral=(np.arctan(z)+z*(z*z-1)/(1+z*z)**2)/8
    return (2*H*x+derivative*velocity)**2/(64*np.pi**2*x**1.5)*integral


def run():
    r=.35; speed=.05/.7
    cases=[evolve(cutoff=24,order=768,end_time=100,f=1e6,MR=1e6,eps=r,theta0=0.,velocity0=v)
           for v in [speed,-speed]]
    initial_error=max(float(np.max(abs(preparation_numbers(c['initial']['H'],c['supplied_inputs']['theta_dot0'],r,24)
                               /c['initial']['state_moments']['comoving_particle_numbers']-1))) for c in cases)
    assert initial_error<1e-10
    plus,minus=cases[0]['final'],cases[1]['final']
    reflection_error=float(np.max(abs(np.array(plus['state_moments']['comoving_particle_numbers'])
                             /np.array(minus['state_moments']['comoving_particle_numbers'])[[0,2,1]]-1)))
    assert reflection_error<1e-7
    assert abs(plus['theta']+minus['theta'])<1e-10
    refined=evolve(cutoff=48,order=1536,end_time=100,f=1e6,MR=1e6,eps=r,theta0=0.,velocity0=speed)
    higher=evolve(cutoff=48,order=3072,end_time=100,f=1e6,MR=1e6,eps=r,theta0=0.,velocity0=speed)
    quadrature_error=max(abs(higher['final'][k]/refined['final'][k]-1)
                         for k in ['a','H','rho_ex','theta_velocity'])
    assert quadrature_error<1e-6
    kinetic_limit_factor=1+3*(speed/np.sqrt(6))*100
    kinetic_theta=np.sqrt(2/3)*np.log(kinetic_limit_factor)
    assert abs(plus['theta']-kinetic_theta)<1e-10
    # Infinite-cutoff initial number fraction in the kinetic-dominated limit.
    scale_conditions=[]
    for rr in [.15,.35]:
        xh,xl=1+2*rr,1-rr
        ch=-rr/(9*np.sqrt(xh));cl=rr*(2-5*rr)/(36*xl**1.5)
        max_heavy=cl/(cl-ch)
        root_sum=np.sqrt(xh)+2*np.sqrt(xl)
        def fraction(eta):
            h=eta/np.sqrt(6)
            return 4*h*h*np.sqrt(xh)/(4*h*h*root_sum+2*rr*rr/(3*xl**1.5))
        eta_limit=np.sqrt(max_heavy*rr*rr/(xl**1.5*(np.sqrt(xh)-max_heavy*root_sum)))
        assert abs(fraction(eta_limit)-max_heavy)<1e-12
        scale_conditions.append(dict(r=rr, initial_heavy_fraction_at_f_over_Mp_1=fraction(1),
                                     necessary_max_f_over_Mp_for_heavy_fraction=eta_limit,
                                     sufficient_for_full_stability=False))
    return dict(cases=cases, refined_positive_case=refined, higher_order_positive_case=higher,
                fixed_cutoff_quadrature_refinement_relative=quadrature_error,
                initial_number_analytic_relative_error=initial_error,
                reflection_number_relative_error=reflection_error,
                kinetic_limit_theta=kinetic_theta,
                cutoff_change_final_rho_relative=float(refined['final']['rho_ex']/plus['rho_ex']-1),
                cutoff_change_final_number_fractions=(np.array(refined['final']['state_moments']['number_fractions'])
                                                    -plus['state_moments']['number_fractions']).tolist(),
                scale_conditions=scale_conditions, fitted_parameters=[], external_pulse=False,
                preparation='Supplied RT31 Gaussian covariance with H-dependent terms retained.',
                full_joint_rmse=None,
                conclusion='Expansion and preparation spoil the prescribed-pulse population selection at f/Mp=1; not a theorem for all initial states.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','refined_positive_case']},indent=2))
