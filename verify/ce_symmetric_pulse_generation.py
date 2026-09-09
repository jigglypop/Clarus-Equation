"""Exact mode evolution in a specified odd phase pulse at a=1.

Research diagnostic: same CE masses; no observational fit, no self-consistent
pulse source or expansion. Incoming instantaneous vacuum at +/- finite tails.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss


def run_case(r=.35, amplitude=.05, tau=.7, order=96, cutoff=10., tails=24.,return_modes=False):
    nodes, w = leggauss(order)
    k = (nodes+1)*cutoff/2
    weights = w*cutoff/2*k*k/(2*np.pi**2)
    def pulse(t):
        z = t/tau
        sech = 1/np.cosh(z)
        return amplitude*sech*np.tanh(z), amplitude/tau*sech*(1-2*np.tanh(z)**2)
    def eigs(theta):
        angles = (theta+2*np.pi*np.arange(3))/3
        return 1+2*r*np.cos(angles), -2*r/3*np.sin(angles)
    shape = (3, order)
    size = 3*order
    def rhs(t,y):
        phase, alpha, beta = y[:-1].reshape(3, 3, order)
        theta, speed = pulse(t)
        x, A = eigs(theta)
        omega = np.sqrt(k[None,:]**2+x[:,None])
        rate = A[:,None]*speed/(4*omega**2)
        factor = np.exp(2j*phase.real)
        coherence = np.real(alpha*np.conj(beta)/factor)
        source = np.sum(weights[None,:]*A[:,None]/omega*(abs(beta)**2+coherence))
        return np.r_[np.array([omega, rate*factor*beta, rate/factor*alpha]).ravel(),speed*source]
    initial = np.r_[np.zeros(size),np.ones(size),np.zeros(size),0.].astype(complex)
    sol = solve_ivp(rhs,(-tails*tau,tails*tau),initial, method='DOP853', rtol=2e-10, atol=2e-13)
    assert sol.success
    phase, alpha, beta = sol.y[:-1,-1].reshape(3,3,order)
    n = abs(beta)**2
    numbers = 2*(n@weights)
    purity = float(np.max(abs(abs(alpha)**2-abs(beta)**2-1)))
    assert purity < 1e-9
    x,A = eigs(0.)
    omega = np.sqrt(k[None,:]**2+x[:,None])
    final_x,_=eigs(pulse(tails*tau)[0])
    final_energy=float(np.sum(weights[None,:]*2*np.sqrt(k[None,:]**2+final_x[:,None])*n))
    work=float(sol.y[-1,-1].real)
    work_relative_error=abs(work/final_energy-1)
    assert work_relative_error<1e-6
    light_born = A[:,None]**2*amplitude**2*np.pi**2*tau**4/np.cosh(np.pi*omega*tau)**2
    b=2*omega[0]*tau
    transform_square = np.pi*tau*b*(2-b*b)/(6*np.sinh(np.pi*b/2))
    heavy_born = (r*amplitude**2*transform_square/(18*omega[0]))**2
    light_number = 2*(light_born[1:]@weights)
    heavy_number = float(2*(heavy_born@weights))
    fraction = numbers/numbers.sum()
    result=dict(r=r, amplitude=amplitude, tau=tau, order=order, cutoff=cutoff, tails=tails,
                species_numbers=numbers.tolist(), species_fractions=fraction.tolist(),
                long_wavelength_pair_count_Fano=(2*((n*(1+n))@weights)/(n@weights)).tolist(),
                relative_light_asymmetry=float(abs(numbers[1]-numbers[2])/np.mean(numbers[1:])),
                heavy_number_over_amplitude_fourth=float(numbers[0]/amplitude**4),
                light_numbers_over_amplitude_squared=(numbers[1:]/amplitude**2).tolist(),
                heavy_Born_relative_error=float(numbers[0]/heavy_number-1),
                light_Born_max_relative_error=float(np.max(abs(numbers[1:]/light_number-1))),
                Bogoliubov_invariant_error=purity,
                final_pair_energy=final_energy, integrated_phase_work=work,
                work_energy_relative_error=work_relative_error,
                cold_mass_curvature_at_zero=float((fraction@(np.array([-r/(9*np.sqrt(1+2*r)),
                    r*(2-5*r)/(36*(1-r)**1.5),r*(2-5*r)/(36*(1-r)**1.5)])))/(fraction@np.sqrt(x))))
    return (result,(k,weights,n)) if return_modes else result


def run():
    cases = [run_case(r=r,amplitude=amp) for r in [.15,.35] for amp in [.1,.05,.025]]
    checks = []
    for r in [.15,.35]:
        rows = [c for c in cases if c['r']==r]
        assert all(c['cold_mass_curvature_at_zero']>0 for c in rows)
        assert max(c['relative_light_asymmetry'] for c in rows)<1e-6
        assert abs(rows[-1]['heavy_Born_relative_error']) < abs(rows[0]['heavy_Born_relative_error'])
        assert rows[-1]['light_Born_max_relative_error'] < rows[0]['light_Born_max_relative_error']
        assert rows[-1]['species_fractions'][0]<rows[0]['species_fractions'][0]
        # Refine cutoff, grid and tail duration for the smaller signal.
        refined=run_case(r=r, amplitude=.025, order=192, cutoff=12., tails=28.)
        rel = float(np.max(abs(np.array(refined['species_numbers'])/rows[-1]['species_numbers']-1)))
        assert rel<1e-5
        checks.append(dict(r=r, refined_max_number_relative_difference=rel))
    return dict(pulse='theta(t)=amplitude*sech(t/tau)*tanh(t/tau)', background='a=1 prescribed external pulse',
                cases=cases, refinement=checks, fitted_parameters=[],
                pulse_generation_self_consistent=False, cosmological_abundance_predicted=False,
                observational_rmse=None)


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
