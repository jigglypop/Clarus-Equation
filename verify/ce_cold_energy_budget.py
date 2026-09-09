"""Integrated energy constraint on the existing six cold CE backgrounds.

No new state, force, or fitted parameter. Independent quadrature of the
continuity equation explains the direction of H0, then rechecks the BAO score.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from ce_symmetric_small_f_stability import ce, StableBackground, StableLightSpectrum


def gaps(sp, theta):
    """Return U/Utop - 1 and m/m(0) - 1 without tiny-theta cancellation."""
    if abs(theta) < .01:
        return tuple(sum(c*theta**(2*j) for j,c in enumerate(coeff) if j)
                     for coeff in [sp.uc, sp.mc])
    values = sp.evaluate(theta)
    return values[0]-1., values[3]-1.


def budget(bg):
    p, sp = bg.p, bg.sp
    ug0, mg0 = gaps(sp, p.theta_initial)
    # Initial dtheta/dN is zero in all these inherited cases.
    initial = p.ai**3*ug0 + p.fc*p.Rm*mg0
    def integrand(N, term):
        q = bg.quantities(N)
        ug, _ = gaps(sp, q['theta'])
        return 3*np.exp(3*N)*(-ug if term == 'potential' else q['kinetic'])
    terms = {}
    for term in ['potential', 'kinetic']:
        value, error = quad(lambda N: integrand(N,term), bg.Ni, 0.,
                            points=bg.sol.t[1:-1], limit=max(1000, len(bg.sol.t)+50),
                            epsabs=1e-27, epsrel=2e-9)
        terms[term] = float(value)
        terms[term+'_quadrature_error'] = float(error)
    end = bg.quantities(0.)
    ug, mg = gaps(sp,end['theta'])
    direct = end['kinetic']+ug+p.fc*p.Rm*mg
    integral = initial-terms['potential']-terms['kinetic']
    scale = max(abs(direct),terms['potential']+terms['kinetic'],1e-100)
    closure = abs(integral-direct)/scale
    assert closure < 2e-7
    assert terms['potential'] >= 0 and terms['kinetic'] >= 0
    assert direct <= initial+2e-7*scale
    null_density = p.Rm+1
    ratio = np.sqrt(1+direct/null_density)
    # Finite seed energy is retained in the bound; not silently set to zero.
    upper_excess = np.expm1(.5*np.log1p(initial/null_density))
    return dict(initial_comoving_excess=initial, **terms,
                final_energy_excess_direct=direct,
                final_energy_excess_integrated=integral,
                relative_integrated_identity_error=closure,
                H0_ratio_to_same_r_null=float(ratio),
                H0_upper_bound_minus_one=float(upper_excess),
                final_kinetic=end['kinetic'], final_potential_gap=ug,
                final_matter_gap=p.fc*p.Rm*mg)


def run():
    prior_path=Path(__file__).with_name('ce_symmetric_bao_ruler.json')
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    z,y,kind,cov,*_=ce.load_data()
    chol=np.linalg.cholesky(cov)
    null=ce.Background(ce.Parameters(.35,0.))
    top0=null.sp.top*(.014414e-9)**4
    keys={'DM_over_rs':'DM_H0_over_c','DH_over_rs':'DH_H0_over_c','DV_over_rs':'DV_H0_over_c'}
    rows=[]
    for old in prior['rows']:
        p=ce.Parameters(old['r'],old['theta_initial'],f=1/30,
                        s_over_Mp2=(old['sqrt_s_eV']*1e-9/2.435e18)**2)
        sp=StableLightSpectrum(p)
        bg=StableBackground(p,sp)
        result=budget(bg)
        top=sp.top*(old['sqrt_s_eV']*1e-9)**4
        physical_ratio=result['H0_ratio_to_same_r_null']*np.sqrt(top/top0)
        assert abs(physical_ratio-old['physical_H0_ratio_to_null']) < 1e-10
        A=prior['baseline']['A']/physical_ratio
        distances={zz:bg.distances(zz) for zz in set(z)}
        pred=A*np.array([distances[zz][keys[kk]] for zz,kk in zip(z,kind)])
        rr=np.linalg.solve(chol,pred-y)
        score=float(np.sqrt((rr@rr+prior['muon_standardized_residual']**2)/14))
        assert abs(score-old['partial_rmse_14']) < 1e-9
        row=dict(r=p.r,theta_initial=p.theta_initial,**result,
                 partial_rmse_14=score,delta_rmse_14=score-prior['baseline']['partial_rmse_14'])
        if p.theta_initial == 1e-8:
            independent=budget(StableBackground(p,sp,method='Radau'))
            difference=abs(independent['final_energy_excess_integrated']-
                           result['final_energy_excess_integrated'])
            assert difference < 1e-8*max(abs(result['final_energy_excess_integrated']),1e-100)
            row['Radau_integrated_energy_difference']=difference
        rows.append(row)
    return dict(rows=rows,baseline_rmse_14=prior['baseline']['partial_rmse_14'],
                score_source_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                fitted_parameters=[],
                scope='Conserved cold populations, positive scalar kinetic energy, same potential top and gravity. No full CMB/H0 prediction.',
                full_joint_rmse=None)


if __name__ == '__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
