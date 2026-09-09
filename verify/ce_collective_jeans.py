"""Conditional spatial scale of chapter 13's occupied lowest gear.

Free harmonic, nonrelativistic, single-component Jeans approximation. Match
loop energy to a supplied present dark-energy density; this is not an abundance
or H0 prediction. Length targets are diagnostic requirements, not measured data.
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'_workspace/ce_collective_original'))
from collective_phase_check import Loop, difference


def run():
    q, N, f, stiffness = 3, 4, .05, 128.
    H0, om, oc = 67.4, .315, .120/.674**2
    ode = 1-om
    phi = np.pi-.6
    W = Loop().values(phi)[0]
    eigenvalue = float(np.linalg.eigvalsh(difference(N,q)@difference(N,q).T)[0])
    mass_over_H_squared = stiffness*eigenvalue*3*ode/(f*f*W)
    H_per_Mpc = H0/299792.458
    kJ = (6*oc*mass_over_H_squared)**.25*H_per_Mpc
    length = 2*np.pi/kJ
    # Independent physical-unit reconstruction of k_J^4 = 2 rho_c m^2/Mp^2.
    Mp, hbar, Mpc_km = 2.435e18, 6.582119569e-25, 3.0856775814913673e19
    H_GeV = H0/Mpc_km*hbar
    rho_c = 3*Mp**2*H_GeV**2*oc
    mass_GeV = np.sqrt(mass_over_H_squared)*H_GeV
    kJ_GeV = (2*rho_c*mass_GeV**2/Mp**2)**.25
    check = kJ_GeV/H_GeV*H_per_Mpc
    assert abs(check/kJ-1)<1e-14
    targets=[]
    for L in [1., .1, .01]:
        required = stiffness*(length/L)**4
        targets.append(dict(target_physical_wavelength_Mpc=L,
                            minimum_stiffness_for_Jeans_length_below_target=required,
                            interpretation='necessary Jeans diagnostic in this approximation; not a fitted or adopted parameter'))
    return dict(q=q,N=N,local_f_over_Mp=f,stiffness=stiffness,phase_for_scale_matching=phi,
                supplied_H0=H0,supplied_omega_c=oc,supplied_omega_DE=ode,
                mass_over_H0=float(np.sqrt(mass_over_H_squared)),mass_eV=mass_GeV*1e9,
                jeans_k_per_Mpc=kJ,jeans_physical_wavelength_Mpc=length,
                physical_unit_relative_check=abs(check/kJ-1),requirements=targets,
                fitted_parameters=[],observational_rmse=None,
                source='https://arxiv.org/abs/astro-ph/0003365',
                limitations=['Single cold harmonic mode; ignores self interaction and coupled perturbations.',
                             'Present loop-energy matching is supplied, not the original synthetic run clock.',
                             'No transfer function, baryonic forcing, halo or likelihood calculation.',
                             'Increasing stiffness changes an input and is not a derived prediction.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
