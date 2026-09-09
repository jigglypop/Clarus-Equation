"""ZIP chapter 31: reproduce absorption, pair energy and kinetic response."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad


def run():
    s,eps,theta,tau=1.,.35,.7,.7
    angles=(theta+2*np.pi*np.arange(3))/3
    masses2=s+2*eps*np.cos(angles)
    A=-2*eps/3*np.sin(angles)
    def pulse(nu):
        v=np.pi*nu*tau/2
        return 2*np.pi*nu*tau*tau*np.exp(-v)/(-np.expm1(-2*v))
    absorption=pairs=kinetic=0.
    for x,force in zip(masses2,A):
        threshold=2*np.sqrt(x)
        def density(nu):return force*force/(16*np.pi)*np.sqrt(max(0,1-4*x/nu**2))
        absorption+=quad(lambda nu:nu*density(nu)*pulse(nu)**2/np.pi,
                         threshold,np.inf,epsabs=1e-14,epsrel=1e-11)[0]
        def pair(k):
            omega=np.sqrt(k*k+x)
            return k*k/(2*np.pi**2)*force*force*pulse(2*omega)**2/(2*omega)
        pairs+=quad(pair,0,np.inf,epsabs=1e-14,epsrel=1e-11)[0]
        kinetic+=2/np.pi*quad(lambda nu:density(nu)/nu**3,threshold,np.inf,
                              epsabs=1e-14,epsrel=1e-11)[0]
    expected=float(np.sum(A*A/masses2)/(96*np.pi**2))
    assert abs(absorption/pairs-1)<1e-10
    assert abs(kinetic/expected-1)<1e-10
    assert abs(absorption/0.00041436184863791366-1)<1e-10
    return dict(inputs=dict(s=s,epsilon=eps,theta=theta,tau=tau),
                absorption_energy_per_pulse_amplitude_squared=absorption,
                pair_energy_per_pulse_amplitude_squared=pairs,
                kinetic_from_dispersion=kinetic,kinetic_from_static_spectrum=expected,
                relative_pair_identity_error=abs(absorption/pairs-1),
                relative_kinetic_identity_error=abs(kinetic/expected-1),
                fitted_parameters=[],observational_rmse=None,
                scope='Chapter31 small external pulse and vacuum linear response; not cosmological abundance.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
