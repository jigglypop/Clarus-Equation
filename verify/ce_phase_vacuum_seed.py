"""Phase vacuum fluctuations about the exact cold symmetric background.

Specified state: instantaneous canonical oscillator vacuum at ai=.01, Gaussian
smoothing over its initial Hubble radius. Linear gradients retained. This is
not a prediction of primordial preparation or a coherent homogeneous seed.
"""
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp, quad
from ce_symmetric_small_f_stability import ce, StableLightSpectrum


def run_case(r,mass,order=128):
    Mp=2.435e18
    p=ce.Parameters(r,0.,f=1/30,s_over_Mp2=(mass*1e-9/Mp)**2)
    sp=StableLightSpectrum(p)
    _,_,U2,_,_,m2,Z,_=sp.evaluate(0.)
    with mp.workdps(60):
        rr=mp.mpf(str(r));ss=(mp.mpf(str(mass))*mp.mpf('1e-9'))**2
        def raw(t):
            xs=[ss*(1+2*rr*mp.cos((t+2*mp.pi*j)/3)) for j in range(3)]
            return sum(x*x*(mp.log(x)-mp.mpf('1.5')) for x in xs)/(32*mp.pi**2)
        Utop=float(raw(0)-raw(mp.pi))
    Hi2=(p.Rm/p.ai**3+1)/3
    Hi=np.sqrt(Utop*Hi2)/Mp
    eta2=(U2+p.fc*p.Rm/p.ai**3*m2)/(Z*Hi2)
    assert eta2>0
    nodes,weights=leggauss(order)
    y=4*(nodes+1);weights=4*weights # p_initial/H_initial in [0,8]
    nu=np.sqrt(y*y+eta2)
    def rhs(N,state):
        q,v=state.reshape(2,order)
        a=np.exp(N);H2=(p.Rm/a**3+1)/3
        hn=-p.Rm/a**3/(2*H2)
        gradient=y*y*(p.ai/a)**2*Hi2/H2
        mass_term=(U2+p.fc*p.Rm/a**3*m2)/(Z*H2)
        return np.r_[v,-(3+hn)*v-(gradient+mass_term)*q]
    # q here is divided by its vacuum initial amplitude. Symmetrized q-p
    # covariance is zero and dq/dN=-i omega_i/H_i in the specified vacuum.
    initial=np.r_[np.ones(order),-1j*nu]
    sol=solve_ivp(rhs,(np.log(p.ai),0.),initial,method='DOP853',rtol=2e-11,atol=2e-13,max_step=.02)
    assert sol.success
    q,v=sol.y[:,-1].reshape(2,order)
    mode_checks=[]
    if order==256:
        for target in [0.,1.,2.]:
            index=int(np.argmin(abs(y-target)))
            yy=y[index];frequency=nu[index]
            def real_rhs(N,state):
                qq=state[0]+1j*state[1];vv=state[2]+1j*state[3]
                a=np.exp(N);H2=(p.Rm/a**3+1)/3
                hn=-p.Rm/a**3/(2*H2)
                coefficient=yy*yy*(p.ai/a)**2*Hi2/H2+(U2+p.fc*p.Rm/a**3*m2)/(Z*H2)
                acceleration=-(3+hn)*vv-coefficient*qq
                return [vv.real,vv.imag,acceleration.real,acceleration.imag]
            alt=solve_ivp(real_rhs,(np.log(p.ai),0.),[1.,0.,0.,-frequency],method='Radau',
                          rtol=1e-10,atol=1e-12,max_step=.02)
            assert alt.success
            altq=alt.y[0,-1]+1j*alt.y[1,-1]
            relative=abs(altq/q[index]-1)
            assert relative<1e-7
            mode_checks.append(dict(initial_p_over_H=float(yy),relative_complex_mode_difference=float(relative)))
    measure=weights*y*y*np.exp(-y*y)/nu
    I0=float(measure.sum());If=float(measure@abs(q)**2)
    prefactor=Hi**2/(4*np.pi**2*Z*Mp**2)
    sigma_initial=np.sqrt(prefactor*I0)
    sigma_final=np.sqrt(prefactor*If)
    check=quad(lambda yy:yy*yy*np.exp(-yy*yy)/np.sqrt(yy*yy+eta2),0,np.inf,
               epsabs=1e-13,epsrel=1e-12)[0]
    assert abs(I0/check-1)<1e-11
    # A linear spectator phase has no first-order metric/matter source at the
    # exact symmetry point: theta_dot=U'=m'=0. Species entropy sources omitted.
    Hfinal2=(p.Rm+1)/3
    kinetic=Z*Hfinal2*prefactor*float(measure@abs(v)**2)/2
    gradient=Z*prefactor*float(measure@(abs(q)**2*y*y*p.ai**2*Hi2))/2
    potential=(U2+p.fc*p.Rm*m2)*prefactor*If/2
    return dict(r=r,order=order,initial_H_GeV=float(Hi),initial_mass_over_H=float(np.sqrt(eta2)),
                initial_phase_rms=float(sigma_initial),final_phase_rms=float(sigma_final),
                rms_amplification=float(sigma_final/sigma_initial),
                fixed_seed_1e_minus8_over_initial_rms=float(1e-8/sigma_initial),
                initial_variance_quadrature_relative_error=float(abs(I0/check-1)),
                independent_Radau_mode_checks=mode_checks,
                smoothed_quadratic_energy_in_Utop_units=dict(kinetic=float(kinetic),gradient=float(gradient),potential=float(potential)),
                sum_absolute_smoothed_energy_over_background=float((abs(kinetic)+abs(gradient)+abs(potential))/(p.Rm+1)))


def run():
    cases=[]
    for r,mass in [(.15,.027615),(.35,.014414)]:
        coarse=run_case(r,mass,128);fine=run_case(r,mass,256)
        error=abs(fine['final_phase_rms']/coarse['final_phase_rms']-1)
        assert error<1e-6
        fine['quadrature_refinement_final_rms_relative']=error
        cases.append(fine)
    return dict(initial_state='Instantaneous phase oscillator vacuum at ai=.01, Gaussian W=exp[-p_i^2/(2H_i^2)].',
                smoothing='Fixed comoving region equal to initial Hubble radius; no UV vacuum stress claim.',
                cases=cases,fitted_parameters=[],primordial_state_derived=False,
                coherent_homogeneous_initial_phase_predicted=False,full_joint_rmse=None,
                conclusion='This specified late vacuum cannot supply the previously assumed 1e-8 coherent phase seed; investigate earlier state preparation or sourced fluctuations.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
