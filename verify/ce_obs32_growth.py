"""Conditional subhorizon, long-range linear growth on the CE-OBS32 background.

Adiabatic initial perturbations are supplied; no primordial amplitude or RSD fit.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1
from ce_obs32_profiled_matter import ce


def growth(bg, method='DOP853'):
    ai = bg.p.ai
    # Same finite-time adiabatic growing-mode preparation for all candidates.
    x = -(7/3)*ai**3
    initial = ai*hyp2f1(1/3,1,11/6,x)
    initial_prime = initial+ai*3*x*(2/11)*hyp2f1(4/3,2,17/6,x)
    def rhs(n,y):
        b = bg.quantities(n)
        ob, oc = b['rho_b']/(3*b['H2']), b['rho_c']/(3*b['H2'])
        q = b['m1']/b['m']
        fifth = 2*q*q/b['Z']
        db, vb, dc, vc = y
        return [vb, -(2+b['hN'])*vb+1.5*(ob*db+oc*dc),
                vc, -(2+b['hN']+q*b['v'])*vc+1.5*(ob*db+oc*(1+fifth)*dc)]
    sol = solve_ivp(rhs,(bg.Ni,0),[initial,initial_prime,initial,initial_prime],
                    method=method,rtol=2e-10,atol=1e-12,dense_output=True,max_step=.05)
    assert sol.success
    b = bg.quantities(0)
    db,vb,dc,vc = sol.y[:,-1]
    wc = b['rho_c']/(b['rho_b']+b['rho_c']); wb = 1-wc
    wcp = wc*wb*b['m1']/b['m']*b['v']
    total = wb*db+wc*dc
    rate = (wb*vb+wc*vc+wcp*(dc-db))/total
    return dict(D_today=float(total),f_today=float(rate),
                dark_to_baryon_contrast=float(dc/db),
                fifth_force_over_gravity_today=float(2*(b['m1']/b['m'])**2/b['Z']))


def run():
    reference=json.loads(Path(__file__).with_name('ce_obs32_common_mass.json').read_text())
    baseline=growth(ce.Background(ce.Parameters(.35,0)))
    analytic=float(hyp2f1(1/3,1,11/6,-7/3))
    assert abs(baseline['D_today']/analytic-1)<1e-8
    rows=[]
    for item in reference['cases']:
        bg=ce.Background(ce.Parameters(item['r'],item['theta_initial'],s_over_Mp2=item['s_over_Mp2']))
        result=growth(bg); other=growth(bg,'Radau')
        err=max(abs(result[k]-other[k]) for k in result)
        assert err<1e-8
        rows.append(dict(r=item['r'],initial_theta=item['theta_initial'],**result,
            D_change_percent=100*(result['D_today']/baseline['D_today']-1),
            solver_absolute_difference=err))
    return dict(scope='leading subhorizon long-range approximation, no scalar horizon-scale dynamics',
        equations='q=dln(m)/dtheta; dark friction 2+H_N/H+q*theta_N; Gcc/G=1+2q^2/Z',
        assumptions=['uncoupled pressureless baryons','nonrelativistic dark matter represented as one effective fluid with mass m(theta)',
          'k/a much greater than H and effective scalar mass magnitude','same supplied adiabatic preparation at a=.01',
          'scalar density perturbation corrections suppressed in this limit'],
        baseline=baseline,baseline_growth_integral_check=abs(baseline['D_today']/analytic-1),
        cases=rows,fitted_parameters=0,sigma8_predicted=False,growth_observational_rmse=None,joint_rmse=None,
        source_context='https://arxiv.org/abs/1305.7457',
        conclusion='Growth transfer calculated conditionally; no observational improvement established.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
