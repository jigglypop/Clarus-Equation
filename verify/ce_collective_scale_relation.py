"""A tied family, not independent fits of slow/fast/muon coefficients.

Keep the reference leading slow kinetic scale, loop normalization and R_H.
Vary disclosed integer chain lengths; derive local f and all other scales.
No chain length is selected by an observational score.
"""
import json
from pathlib import Path
import numpy as np
from ce_collective_jeans import run as reference_run
from common_spectrum_muon import M_MU


def run():
    ref = reference_run()
    q, Mp = 3, 2.435e18
    S4 = sum(float(q)**(2*k) for k in range(5))
    F = .05*np.sqrt(S4)
    lam4 = q*q+1-2*q*np.cos(np.pi/5)
    rows=[]
    for N in [4, 12, 20, 28, 36]:
        S = sum(float(q)**(2*k) for k in range(N+1))
        f = F/np.sqrt(S)
        lam = q*q+1-2*q*np.cos(np.pi/(N+1))
        D = np.eye(N,N+1)-q*np.eye(N,N+1,k=1)
        err = abs(np.linalg.eigvalsh(D@D.T)[0]-lam)
        assert err<1e-13
        mass_ratio = .05/f*np.sqrt(lam/lam4)
        length = ref['jeans_physical_wavelength_Mpc']/np.sqrt(mass_ratio)
        bound = M_MU**2/(16*np.pi**2*(f*Mp)**2)
        rows.append(dict(N=N,phase_fields=N+1,local_f_GeV=f*Mp,
                         leading_F_over_Mp=f*np.sqrt(S),
                         lowest_gear_mass_eV=ref['mass_eV']*mass_ratio,
                         jeans_wavelength_Mpc=length,
                         unit_endpoint_muon_abs_upper_bound=bound,
                         smallest_gear_eigenvalue=lam,eigenvalue_error=err,
                         invariant=lam*length**4*bound))
    error=max(abs(row['invariant']/rows[0]['invariant']-1) for row in rows)
    assert error<1e-13
    return dict(fixed_R_H=128,fixed_leading_F_over_Mp=F,reference=ref,
                rows=rows,invariant_relative_error=error,
                fitted_parameters=[],selected_N=None,observational_rmse=None,
                limitations=['N and common scale are not derived by this relation.',
                             'Fixed F preserves leading slow dynamics, not all finite-lock corrections.',
                             'Jeans approximation and unit endpoint pseudoscalar subset only.',
                             'Mode abundance, reheating, portals and UV protection remain uncomputed.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['rows'],indent=2))
