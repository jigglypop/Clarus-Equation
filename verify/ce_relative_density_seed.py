"""Compensated light-species number noise coupled to the CE phase.

Cold collisionless fluid hypothesis; initial independent Poisson counts at
ai=.01. No primordial spectrum, pressure, or squeezed-state noise is inferred.
"""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from ce_symmetric_small_f_stability import ce, StableLightSpectrum


def run_case(r,mass,order=128,feedback=True):
    Mp=2.435e18
    p=ce.Parameters(r,0.,f=1/30,s_over_Mp2=(mass*1e-9/Mp)**2)
    sp=StableLightSpectrum(p)
    _,_,U2,_,_,m2,Z,_=sp.evaluate(0.)
    prior=json.loads(Path('verify/ce_phase_vacuum_seed.json').read_text())
    Hi=next(v['initial_H_GeV'] for v in prior['cases'] if v['r']==r)
    Hi2=(p.Rm/p.ai**3+1)/3
    Utop=Hi**2*Mp**2/Hi2
    rho_ci=p.fc*p.Rm*Utop/p.ai**3
    number_density=rho_ci/(mass*1e-9*np.sqrt(1-r))
    kappa=r/(2*np.sqrt(3)*(1-r))
    nodes,w=leggauss(order);ys=4*(nodes+1);weights=4*w
    def coefficients(N,y):
        a=np.exp(N);H2=(p.Rm/a**3+1)/3
        hn=-p.Rm/a**3/(2*H2)
        gradient=y*y*(p.ai/a)**2*Hi2/H2
        mass_term=(U2+p.fc*p.Rm/a**3*m2)/(Z*H2)
        source=kappa*p.fc*p.Rm/a**3/(Z*H2)
        return hn,gradient,mass_term,source
    def vector_rhs(N,state):
        theta,v,D,wD=state.reshape(4,order)
        hn,g,mass_term,source=coefficients(N,ys)
        return np.array([v,-(3+hn)*v-(g+mass_term)*theta-source*D,
                         wD,-(2+hn)*wD-kappa*g*theta if feedback else np.zeros(order)]).ravel()
    initial=np.array([np.zeros(order),np.zeros(order),np.ones(order),np.zeros(order)]).ravel()
    sol=solve_ivp(vector_rhs,(np.log(p.ai),0.),initial,method='DOP853',rtol=2e-11,atol=2e-13,max_step=.02)
    assert sol.success
    theta,v,D,wD=sol.y[:,-1].reshape(4,order)
    measure=weights*ys*ys*np.exp(-ys*ys)
    prefactor=Hi**3/(2*np.pi**2*number_density)
    initial_variance=Hi**3/(8*np.pi**1.5*number_density)
    assert abs(prefactor*measure.sum()/initial_variance-1)<1e-12
    phase_rms=np.sqrt(prefactor*(measure@theta**2))
    density_rms=np.sqrt(prefactor*(measure@D**2))
    checks=[]
    if order==256 and feedback:
        for target in [0.,1.,2.]:
            i=int(np.argmin(abs(ys-target)))
            def rhs(N,state):
                t,vv,d,dd=state
                hn,g,mt,S=coefficients(N,ys[i])
                return [vv,-(3+hn)*vv-(g+mt)*t-S*d,dd,-(2+hn)*dd-kappa*g*t]
            alt=solve_ivp(rhs,(np.log(p.ai),0.),[0.,0.,1.,0.],method='Radau',rtol=1e-10,
                          atol=1e-12,max_step=.02)
            assert alt.success
            diff=float(np.max(abs(alt.y[:,-1]/sol.y[:,-1].reshape(4,order)[:,i]-1)))
            assert diff<1e-7
            checks.append(dict(initial_p_over_H=float(ys[i]),relative_mode_difference=diff))
    return dict(r=r,order=order,relative_flow_feedback=feedback,kappa=kappa,
                initial_number_density_GeV3=number_density,
                particles_in_effective_initial_Hubble_window=float(1/initial_variance),
                initial_relative_density_rms=float(np.sqrt(initial_variance)),
                final_relative_density_rms=float(density_rms),final_phase_rms=float(phase_rms),
                independent_mode_checks=checks,
                initial_relative_noise_rms_needed_for_final_phase_0p1=float(.1/phase_rms*np.sqrt(initial_variance)))


def run():
    rows=[]
    for r,mass in [(.15,.027615),(.35,.014414)]:
        coarse=run_case(r,mass,128);fine=run_case(r,mass,256)
        difference=abs(fine['final_phase_rms']/coarse['final_phase_rms']-1)
        assert difference<1e-6
        frozen=run_case(r,mass,128,False)
        fine['quadrature_refinement_relative']=difference
        fine['frozen_relative_density_final_phase_rms']=frozen['final_phase_rms']
        fine['feedback_over_frozen_rms']=fine['final_phase_rms']/frozen['final_phase_rms']
        rows.append(fine)
    return dict(cases=rows,fitted_parameters=[],
                initial_state='Independent cold Poisson species counts, equal mean light populations, compensated relative mode.',
                window='Fixed comoving Gaussian region of initial Hubble radius at ai=.01.',
                initial_covariance_from_creation_derived=False,full_joint_rmse=None,
                limitations=['Cold collisionless fluid; quantum pair correlations and velocity dispersion omitted.',
                             'No replacement of stochastic RMS by coherent background phase.',
                             'Compensated relative mode only, not full adiabatic density growth or CMB.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
