"""Reconstruct mean population ratios from supplied CE-QP27 preparation.

Inflationary H, duration, phase and masses are inputs. No abundance fit.
"""
import json
from pathlib import Path
import mpmath as mp


def run():
    rows=[]
    with mp.workdps(80):
        h=mp.mpf('1e7') # GeV, CE-QP27 supplied benchmark.
        theta=mp.mpf('.5')
        for frac,mass_ev in [('.15','.027615'),('.35','.014414')]:
            r=mp.mpf(frac); mass=mp.mpf(mass_ev)*mp.mpf('1e-9'); s=mass**2
            def spectrum(t):
                return [s*(1+2*r*mp.cos((t+2*mp.pi*j)/3)) for j in range(3)]
            x=spectrum(theta)
            for duration in [60,10**11]:
                variance=[3*h**4/(8*mp.pi**2*a)*(-mp.expm1(-2*a*duration/(3*h*h))) for a in x]
                # Common H_R and formation coefficient cancel from number fractions.
                number=[v/a**mp.mpf('.25') for v,a in zip(variance,x)]
                number=[a/sum(number) for a in number]
                def energy(t):
                    return sum(n*mp.sqrt(a) for n,a in zip(number,spectrum(t)))
                q=mp.diff(energy,theta)/energy(theta)
                def equal_energy(t):
                    return sum(mp.sqrt(a) for a in spectrum(t))
                q_equal=mp.diff(equal_energy,theta)/equal_energy(theta)
                short=[a**(-mp.mpf('.25')) for a in x]
                short=[a/sum(short) for a in short]
                short_error=max(abs(a-b) for a,b in zip(number,short))
                assert short_error<mp.mpf('1e-20')
                rows.append(dict(r=float(r),N_pre=duration,number_fractions=[float(a) for a in number],
                    prepared_q_at_preparation_phase=float(q),equal_number_q=float(q_equal),
                    prepared_q_at_zero_phase=float(mp.diff(energy,mp.mpf(0))/energy(0)),
                    short_duration_fraction_error=float(short_error)))
    return dict(source='CE-QP27 equations (3) and formation law N_j proportional to v_j/sqrt(M_j)',
        H_I_GeV=1e7,preparation_theta=.5,cases=rows,absolute_abundance_predicted=False,
        fitted_parameters=0,joint_rmse=None,
        conclusion='Supplied stochastic preparation does not yield equal conserved number densities; its population ratios need a new consistent background evolution.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
