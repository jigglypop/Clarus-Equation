"""Resolve the three conserved spectral populations in the short-wave limit."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1
from ce_obs32_profiled_matter import ce
from ce_obs32_growth import growth


def mass_ratio(bg,n):
    b=bg.quantities(n)
    curvature=b['U2']+b['rho_c']*b['m2']/b['m']
    force=b['U1']+b['rho_c']*b['m1']/b['m']
    return (curvature-.5*b['Z1']/b['Z']*force)/(b['Z']*b['H2'])


def resolved(bg, method='DOP853', k_h_Mpc=None):
    ai=bg.p.ai; x=-(7/3)*ai**3
    d=ai*hyp2f1(1/3,1,11/6,x)
    v=d+ai*3*x*(2/11)*hyp2f1(4/3,2,17/6,x)
    def coefficients(n):
        b=bg.quantities(n)
        masses,mass_derivatives,_=bg.sp.eigs(b['theta'])
        weights=np.sqrt(masses); weights/=weights.sum()
        q=np.r_[0.,mass_derivatives/(2*masses)]
        densities=np.r_[b['rho_b'], b['rho_c']*weights]
        assert np.isclose(weights@q[1:],b['m1']/b['m'],rtol=1e-10,atol=1e-16)
        return b,q,densities
    def rhs(n,y):
        b,q,rho=coefficients(n)
        contrasts=y[:4]; velocities=y[4:]
        response=1.
        if k_h_Mpc is not None:
            kh=(2997.92458*k_h_Mpc*bg.H0/(np.exp(n)*b['H']))**2
            denominator=kh+mass_ratio(bg,n)
            if denominator<=0: raise ValueError('Outside nonsingular quasistatic domain')
            response=kh/denominator
        attraction=(1+2*np.outer(q,q)/b['Z']*response)@(rho/(3*b['H2'])*contrasts)
        return np.r_[velocities,-(2+b['hN']+q*b['v'])*velocities+1.5*attraction]
    sol=solve_ivp(rhs,(bg.Ni,0),np.r_[np.full(4,d),np.full(4,v)],method=method,
                  rtol=2e-10,atol=1e-12,max_step=.05,dense_output=True)
    assert sol.success
    b,q,rho=coefficients(0)
    weights=rho/rho.sum(); wprime=weights*(q-weights@q)*b['v']
    contrasts=sol.y[:4,-1]; velocities=sol.y[4:,-1]
    total=weights@contrasts
    return dict(D_today=float(total),f_today=float((weights@velocities+wprime@contrasts)/total),
                dark_contrasts_over_baryon=(contrasts[1:]/contrasts[0]).tolist(),
                species_q=q[1:].tolist())


def run():
    reference=json.loads(Path(__file__).with_name('ce_obs32_common_mass.json').read_text())
    null=resolved(ce.Background(ce.Parameters(.35,0)))
    assert abs(null['D_today']/hyp2f1(1/3,1,11/6,-7/3)-1)<1e-8
    assert max(abs(np.array(null['dark_contrasts_over_baryon'])-1))<1e-12
    rows=[]
    for case in reference['cases']:
        bg=ce.Background(ce.Parameters(case['r'],case['theta_initial'],s_over_Mp2=case['s_over_Mp2']))
        result=resolved(bg); check=resolved(bg,'Radau'); effective=growth(bg)
        error=max(abs(result[k]-check[k]) for k in ['D_today','f_today'])
        assert error<1e-8
        wavelength_checks=[]
        for k in [.01,.1]:
            finite=resolved(bg,k_h_Mpc=k)
            domain=[]
            for n in np.linspace(bg.Ni,0,101):
                b=bg.quantities(n)
                kh=(2997.92458*k*bg.H0/(np.exp(n)*b['H']))**2
                domain.append((1/kh,abs(mass_ratio(bg,n))/kh))
            wavelength_checks.append(dict(k_h_Mpc=k,
                max_aH_over_k_squared=max(x[0] for x in domain),
                max_abs_scalar_mass_squared_over_physical_k_squared=max(x[1] for x in domain),
                finite_mass_D_relative_change=finite['D_today']/result['D_today']-1))
        rows.append(dict(r=case['r'],initial_theta=case['theta_initial'],**result,
            wavelength_checks=wavelength_checks,
            D_change_from_baseline_percent=100*(result['D_today']/null['D_today']-1),
            D_change_from_effective_fluid_percent=100*(result['D_today']/effective['D_today']-1),
            solver_absolute_difference=error))
    return dict(scope='three nonrelativistic conserved equal-number populations plus baryons; short-wave long-range limit',
        assumptions=['same supplied adiabatic preparation','no collisions or conversion between eigenstates',
                     'no primordial normalization or survey likelihood'],baseline=null,cases=rows,
        fitted_parameters=0,growth_observational_rmse=None,joint_rmse=None)


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
