"""Conditional boson/fermion pairing: vacuum cancellation and visible response.

Two complex scalars plus one vectorlike Dirac fermion per spectral eigenvalue.
This is NOT a supersymmetric completion, nor a full g-2 or cosmology model.
"""
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from common_spectrum_muon import ALPHA, M_MU, Q2, exact_em, spectrum
from common_spectrum_vacuum_budget import shape, shape_integral


def dirac_em(mass, e, theta, spectral=False):
    x = spectrum(e, theta)
    r = (M_MU/mass)**2
    def outer(t):
        ez = math.exp(-t)
        z = -math.expm1(-t)
        y = r*z*z/ez
        def inner(u):
            if spectral:
                return ez*ez/r*float(np.sum((y/x)*u*u*(3-u*u)/
                                           (3*(4+(y/x)*(1-u*u)))))
            v = u*(1-u)
            return ez*ez/r*2*v*float(np.log1p(y*v/x).sum())
        return quad(inner,0,1,epsabs=1e-15,epsrel=2e-10,limit=150)[0]
    value=quad(outer,0,80,points=[1,5,15,30],epsabs=1e-15,
               epsrel=2e-10,limit=150)[0]
    return (ALPHA/math.pi)**2*Q2*r*value


def run():
    checks=[]
    for mass,e,theta in [(1000.,.025,.5), (1.,.2,.3), (.05,.4,2.4)]:
        f=dirac_em(mass,e,theta)
        g=dirac_em(mass,e,theta,spectral=True)
        s=exact_em(mass,e,theta)
        leading=(ALPHA/math.pi)**2*Q2*(M_MU/mass)**2*float(sum(1/spectrum(e,theta)))/45
        checks.append(dict(mass_GeV=mass,e=e,theta=theta,scalar_em=s,dirac_em=f,
                           paired_em=2*s+f,dirac_integral_relative_error=abs(f-g)/f,
                           dirac_heavy_upper_bound=leading,
                           paired_to_scalar_ratio=(2*s+f)/s))
    assert all(c['dirac_integral_relative_error']<1e-8 and
               c['dirac_em']<=c['dirac_heavy_upper_bound'] for c in checks)

    with mp.workdps(120):
        mass=mp.mpf('1000'); e=mp.mpf('.025'); theta=mp.mpf('.5'); d=16
        u=shape(e,theta)
        # Fixed epsilon; only the scalars' common diagonal mass squared shifts.
        def soft_potential(t):
            return 2*d*mass**4*((1+t)**2*shape(e/(1+t),theta)-u)
        def soft_integral(t):
            return 2*d*mass**4*((1+t)**2*shape_integral(e/(1+t),theta)-shape_integral(e,theta))
        dark=mp.mpf('.027615e-9')**4*shape(mp.mpf('.15'),theta)
        slope=mp.diff(soft_potential,mp.mpf(0))
        estimate=dark/abs(slope)
        root=mp.findroot(lambda q: mp.log(-soft_potential(mp.exp(q))/dark),
                         mp.log(estimate),tol=mp.mpf('1e-100'))
        bound=mp.exp(root)
        sensitivity=[]
        for t in [mp.mpf(0),mp.mpf('1e-8'),mp.mpf('1e-16'),bound]:
            v=soft_potential(t); vi=soft_integral(t)
            err=abs(v-vi)/max(abs(v),dark)
            assert err<mp.mpf('1e-50')
            sensitivity.append(dict(scalar_mass_squared_fractional_shift=float(t),
                residual_potential_GeV4=float(v),absolute_residual_over_dark=float(abs(v)/dark),
                independent_integral_error_scaled=float(err)))
        b=(1-2*e)*(1+e)**2; k=2*e**3*(1+mp.cos(theta))
        original_gauge=-mp.mpf(str(Q2))/(12*mp.pi)*mp.log1p(k/b)
        # SU(5)-normalized hypercharge: (3/5)*T_Y = T_2 = T_3 = 2.
        normalized_indices=[mp.mpf(3*10)/(5*3),mp.mpf(2),mp.mpf(2)]
        shifts=[-6*t/(12*mp.pi)*mp.log1p(k/b) for t in normalized_indices]
        gauge=dict(original_delta_alpha_star_inverse=float(original_gauge),
                   paired_delta_alpha_star_inverse=float(6*original_gauge),
                   ratio=6,normalized_gauge_shifts=[float(v) for v in shifts],
                   shifts_of_pairwise_inverse_coupling_differences=[float(shifts[i]-shifts[j]) for i,j in [(0,1),(1,2),(0,2)]],
                   scope="one-loop phase threshold at exact degeneracy; universal shift cannot change coupling differences")

    reference=json.loads(Path(__file__).with_name('common_spectrum_muon.json').read_text())
    data=reference['frozen_summary']; gap=data['residual']; sd=data['combined_sigma']
    s=checks[0]['scalar_em']; p=checks[0]['paired_em']
    return dict(role="conditional_pairing_screen",field_content="two complex scalar triplets and one vectorlike Dirac triplet",
        charged_representation_dimension=16,neutral_sector="unchanged, unpaired benchmark",
        common_phase=.5,mass_GeV=1000.,epsilon_over_s=.025,
        one_loop_flat_vacuum_weight=2+2-4,exact_degeneracy_vacuum_cancelled=True,
        soft_sensitivity=sensitivity,scalar_mass_squared_shift_at_dark_scale=float(bound),
        matching=gauge,em_checks=checks,
        frozen_muon_comparison=dict(n=1,sm_rmse=abs(gap)/sd,
            scalar_subset_rmse=abs(gap-s)/sd,paired_subset_rmse=abs(gap-p)/sd,
            paired_minus_sm=(abs(gap-p)-abs(gap))/sd,paired_minus_scalar=(abs(gap-p)-abs(gap-s))/sd,
            independent_holdout=False,source=reference['sources']),
        missing=["symmetry enforcing degeneracy and interactions", "curvature and kinetic matching",
                 "full muon diagrams", "common cosmological state and perturbations", "joint observational predictions"],
        supersymmetry_established=False,joint_rmse=None,scientific_success=False)


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
