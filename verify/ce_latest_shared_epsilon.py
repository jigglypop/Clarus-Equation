"""Reproduce the literal common theta,epsilon assumption in ZIP CE formula.

Sector diagonal masses differ; epsilon/s is NOT held equal between sectors.
No observational fit, no full common cosmological state claimed.
"""
import json
from pathlib import Path
import hashlib
import mpmath as mp


def run():
    source=next((Path(__file__).resolve().parents[1]/'_workspace/ce_obs32_reproduction/CE-OBS32/source_context').glob('*.txt'))
    mp.mp.dps=160
    angle=mp.mpf('.5');sH=mp.mpf(1000)**2
    rows=[]
    def U(s,e):
        def value(t):
            roots=[s+2*e*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
            return sum(x*x*(mp.log(x)-mp.mpf('1.5')) for x in roots)/(32*mp.pi**2)
        return value(angle)-value(mp.pi)
    for mass,r in [('0.027615','.15'),('0.014414','.35')]:
        sD=(mp.mpf(mass)*mp.mpf('1e-9'))**2
        eps=mp.mpf(r)*sD
        dark=U(sD,eps);heavy=U(sH,eps)
        leading=eps**3*(1+mp.cos(angle))/(16*mp.pi**2*sH)
        discrepancy=abs(heavy/leading-1)
        assert discrepancy<mp.mpf('1e-50')
        # Exact determinant identities keep the tiny shared-epsilon signal.
        det_pi=sH**3-3*sH*eps**2-2*eps**3
        q=2*eps**3*(1+mp.cos(angle))/det_pi
        Uss=mp.log1p(q)/(16*mp.pi**2)
        Usss=-q*(3*sH**2-3*eps**2)/(det_pi*(1+q)*16*mp.pi**2)
        direct_Uss=mp.diff(lambda ss:U(ss,eps),sH,2)
        assert abs(direct_Uss/Uss-1)<mp.mpf('1e-50')
        gauge=[-T*Uss/3 for T in [mp.mpf(10)/3,mp.mpf(2),mp.mpf(2)]]
        relative_mu=2*(1/mp.mpf('137.035999084'))**2*mp.mpf('.1056583755')**2*(mp.mpf(16)/3)*Usss/45
        # For positive epsilon and s>2 epsilon, 0<=q<=qmax. Both log1p(q)
        # and q/(1+q) are monotone, so these are analytic all-phase bounds.
        qmax=4*eps**3/det_pi
        max_Uss=mp.log1p(qmax)/(16*mp.pi**2)
        max_Usss=qmax*(3*sH**2-3*eps**2)/(det_pi*(1+qmax)*16*mp.pi**2)
        em_prefactor=2*(1/mp.mpf('137.035999084'))**2*mp.mpf('.1056583755')**2*(mp.mpf(16)/3)/45
        max_mu=em_prefactor*max_Usss
        sigma=mp.sqrt(mp.mpf(145)**2+mp.mpf(620)**2)*mp.mpf('1e-12')
        # Triangle inequality for the whitened seven-row residual norm.
        max_score_change=max_mu/(sigma*mp.sqrt(7))
        for j in range(65):
            qq=2*eps**3*(1+mp.cos(2*mp.pi*j/64))/det_pi
            assert 0<=qq<=qmax
            assert mp.log1p(qq)/(16*mp.pi**2)<=max_Uss
        assert abs(relative_mu)<=max_mu
        rows.append(dict(supplied_dark_mass_eV=float(mass),supplied_dark_r=float(r),
                         supplied_charged_mass_GeV=1000,shared_epsilon_GeV2=float(eps),
                         resulting_charged_r=float(eps/sH),theta=float(angle),
                         dark_U_GeV4=float(dark),charged_U_per_component_GeV4=float(heavy),
                         charged_16_components_over_dark_U=float(16*heavy/dark),
                         relative_inverse_gauge_coupling_Y_2_3=[float(v) for v in gauge],
                         relative_muon_EM_leading=float(relative_mu),
                         all_phase_max_abs_inverse_gauge_Y_2_3=[float(T*max_Uss/3) for T in [mp.mpf(10)/3,mp.mpf(2),mp.mpf(2)]],
                         all_phase_max_abs_relative_muon_EM_leading=float(max_mu),
                         seven_row_rmse_max_change_from_muon_phase_only=float(max_score_change),
                         determinant_vs_potential_second_derivative_relative_error=float(abs(direct_Uss/Uss-1)),
                         high_mass_expansion_relative_difference=float(discrepancy)))
    return dict(source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                assumption='Same dimensionful epsilon and theta; distinct s_D and s_H.',
                precision_digits=160,rows=rows,fitted_parameters=[],joint_rmse=None,
                correction='Previous independently supplied charged r benchmarks do not implement this common-epsilon premise.',
                limitations=['Dark and charged mass scales and common splitting remain supplied inputs.',
                             'Relative loop potential only; absolute vacuum, full charged action and real-time state not solved.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
