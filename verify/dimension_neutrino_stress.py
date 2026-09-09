"""Decoupled homogeneous Fermi-Dirac gas with adiabatically varying masses.

Natural units. f(p)=1/(exp(p/T)+1), NOT a massive thermal-equilibrium f(E).
T scales as a^-1. No collisions, particle production, coherent transitions,
chemical potential, spectral distortions, or cosmological perturbations.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit


def moments(mass, temperature, *, degeneracy=2.):
    mass, temperature, degeneracy = map(float, (mass,temperature,degeneracy))
    if not np.isfinite([mass,temperature,degeneracy]).all() or mass < 0 or temperature <= 0 or degeneracy <= 0:
        raise ValueError("finite mass>=0, temperature>0, degeneracy>0 required")
    x=mass/temperature
    if not np.isfinite(x):
        raise ValueError("mass/temperature exceeds numerical range")
    def integrand(y):
        e=np.hypot(y,x)
        if e==0:
            return np.zeros(5)
        f=expit(-y)
        return f*np.array([y*y*e,y**4/(3*e),y*y*x*(x/e),y*y*(x/e),y*y])
    value,error=quad_vec(integrand,0,np.inf,epsabs=1e-11,epsrel=2e-11)
    pref=degeneracy/(2*np.pi**2)
    rho,pressure,trace=pref*temperature**4*value[:3]
    drho_dm=pref*temperature**3*value[3]
    number=pref*temperature**3*value[4]
    return {"rho":float(rho),"pressure":float(pressure),"trace":float(trace),
            "drho_dm":float(drho_dm),"number_density":float(number),
            "w":float(pressure/rho),"dimensionless_quadrature_error":float(error)}


def population(masses, mass_derivatives, temperature):
    """d rho/dq at fixed comoving distribution; derivatives are dm_i/dq.

    Using dm rather than dlog(m) handles an exactly massless state.
    """
    masses,derivatives=np.asarray(masses,float),np.asarray(mass_derivatives,float)
    if masses.ndim!=1 or masses.size==0 or derivatives.shape!=masses.shape or not np.isfinite(derivatives).all():
        raise ValueError("matching nonempty finite mass and derivative vectors required")
    rows=[moments(m,temperature) for m in masses]
    return {"species":rows,"rho":sum(r['rho'] for r in rows),
            "pressure":sum(r['pressure'] for r in rows),
            "scalar_source_d_rho_dq":float(sum(d*r['drho_dm'] for d,r in zip(derivatives,rows)))}


def report():
    source=json.loads(Path(__file__).with_name('dimension_seesaw_bridge.json').read_text())
    result={}
    for name,row in source['rows'].items():
        result[name]={}
        for z in (0,99,999):
            # Illustrative supplied temperature; no present-day precision claim.
            value=population(row['masses_eV'],[0,0,0],1.68e-4*(1+z))
            result[name][str(z)]={"w":value['pressure']/value['rho'],
                "rho_eV4":value['rho'],"pressure_eV4":value['pressure']}
    return {"input":"dimension_seesaw_bridge.json","T0_eV":1.68e-4,
            "mass_evolution":"constant_masses_for_this_diagnostic_only",
            "rows":result,"status":"background_moments_only_no_friedmann_fit",
            "joint_rmse":None,"scientific_success":False}


if __name__=='__main__':
    result=report()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,row in result['rows'].items():
        print(name,{z:r['w'] for z,r in row.items()})
