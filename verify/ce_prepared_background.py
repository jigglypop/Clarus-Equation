"""Finite-time background response to QP27 mean population ratios.

Matched initial density and zero velocity at a=.01, not an inflation-to-now solution.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from ce_obs32_profiled_matter import ce
from common_spectrum_muon import ALPHA,Q2,exact_em
from common_spectrum_pairing import dirac_em


class PreparedSpectrum(ce.Spectrum):
    def __init__(self,p,weights):
        super().__init__(p)
        self.number_weights=np.asarray(weights,dtype=float)
        self.number_weights/=self.number_weights.sum()
        x=self.eigs(p.theta_initial)[0]
        # Same initial physical matter density as the equal-number control.
        old_mass=np.sqrt(x).sum()/self.mnorm
        self.weighted_norm=(self.number_weights@np.sqrt(x))/old_mass

    def evaluate(self,theta):
        values=list(super().evaluate(theta))
        x,x1,x2=self.eigs(theta); mass=np.sqrt(x); w=self.number_weights/self.weighted_norm
        values[3]=float(w@mass)
        values[4]=float(w@(x1/(2*mass)))
        values[5]=float(w@(x2/(2*mass)-x1*x1/(4*mass**3)))
        return tuple(values)


class PreparedBackground(ce.Background):
    def __init__(self,p,weights,method='DOP853'):
        self.p=p; self.sp=PreparedSpectrum(p,weights); self.Ni=np.log(p.ai)
        self.sol=solve_ivp(self.rhs,(self.Ni,0),[p.theta_initial,0.],method=method,
            rtol=2e-10,atol=2e-12,dense_output=True,max_step=.05)
        assert self.sol.success
        self.H0=self.quantities(0.)['H']


def run():
    def read(name):return json.loads(Path(__file__).with_name(name).read_text())
    preparation=read('ce_prepared_populations.json')
    scales=read('ce_obs32_common_mass.json')
    mu=read('common_spectrum_muon.json')['frozen_summary']
    _,_,_,_,zs,obs,cov=ce.load_data(); chol=np.linalg.cholesky(cov)
    baseline_partial=read('common_phase_partial_score.json')['baseline_partial_rmse']
    rows=[]
    for item in preparation['cases']:
        if item['N_pre']!=60:continue # Both supplied durations give the same ratios here.
        scale=next(x for x in scales['cases'] if x['r']==item['r'] and x['theta_initial']==.5)
        p=ce.Parameters(item['r'],.5,s_over_Mp2=scale['s_over_Mp2'])
        cases=[]
        for label,weights in [('equal',[1/3]*3),('prepared',item['number_fractions'])]:
            bg=PreparedBackground(p,weights); check=PreparedBackground(p,weights,'Radau')
            pred=np.array([bg.distances(z)['F_AP'] for z in zs])
            other=np.array([check.distances(z)['F_AP'] for z in zs])
            assert np.max(abs(pred-other))<1e-8
            white=np.linalg.solve(chol,pred-obs); ap=float(np.linalg.norm(white)/np.sqrt(6))
            end=bg.quantities(0);theta=end['theta']
            hgev=bg.H0*np.sqrt(scale['potential_scale_GeV4'])/2.435e18
            hyear=hgev/6.582119569e-25*31557600
            rc=.025
            slope=6*Q2/(12*np.pi)*2*rc**3*np.sin(theta)/(1-3*rc*rc+2*rc**3*np.cos(theta))
            drift=-ALPHA*slope*end['v']*hyear
            cr=abs(drift-1e-18)/1.1e-18
            correction=2*exact_em(1000,rc,theta)+dirac_em(1000,rc,theta)
            mr=abs(mu['residual']-correction)/mu['combined_sigma']
            partial=float(np.sqrt((6*ap*ap+cr*cr+mr*mr)/8))
            cases.append(dict(label=label,AP_rmse=ap,partial_rmse=partial,
                theta_today=theta,theta_N_today=end['v'],clock_drift_per_year=drift,
                initial_rho_c=bg.quantities(bg.Ni)['rho_c'],
                continuity=bg.source_checks()['relative_total_continuity'],
                solver_max_AP_difference=float(np.max(abs(pred-other)))))
        assert np.isclose(cases[0]['initial_rho_c'],cases[1]['initial_rho_c'],rtol=1e-13)
        rows.append(dict(r=item['r'],cases=cases,prepared_minus_control_partial=cases[1]['partial_rmse']-cases[0]['partial_rmse']))
    return dict(scope='finite-time matched initial-state comparison, not full primordial prediction',
        initial_theta=.5,initial_theta_N=0,ai=.01,baseline_partial_rmse=baseline_partial,
        cases=rows,fitted_parameters=0,joint_rmse=None,
        caveat='Overall abundance, preparation background and late initial state remain supplied; partial score inherits charged-sector assumptions.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
