"""Use original RT31 f/M_R=1/30 in the conditional cold symmetric state.

Fixed diagnostic seeds, no data optimization. The early generation and its
seed distribution are not derived by this late-time calculation.
"""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from ce_obs32_occupation_background import ce, OccupationSpectrum, OccupationBackground
from common_spectrum_muon import exact_em


class StableLightSpectrum(OccupationSpectrum):
    """Even Taylor series prevents a roundoff force from seeding instability."""
    def __init__(self,p):
        super().__init__(p,[0,1,1])
        with mp.workdps(60):
            r=mp.mpf(str(p.r))
            def x(t): return [1+2*r*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
            def raw(t): return sum(v*v*(mp.log(v)-mp.mpf('1.5')) for v in x(t))/(32*mp.pi**2)
            top=raw(0)-raw(mp.pi)
            def mass(t): return (mp.sqrt(x(t)[1])+mp.sqrt(x(t)[2]))/(2*mp.sqrt(1-r))
            self.uc=[1.]+[float(mp.diff(raw,0,n)/top/mp.factorial(n)) for n in [2,4,6]]
            self.mc=[1.]+[float(mp.diff(mass,0,n)/mp.factorial(n)) for n in [2,4,6]]
    def evaluate(self,theta):
        values=list(super().evaluate(theta))
        if abs(theta)<.01:
            for offset,coeff in [(0,self.uc),(3,self.mc)]:
                values[offset]=sum(c*theta**(2*j) for j,c in enumerate(coeff))
                values[offset+1]=sum(2*j*c*theta**(2*j-1) for j,c in enumerate(coeff) if j)
                values[offset+2]=sum(2*j*(2*j-1)*c*theta**(2*j-2) for j,c in enumerate(coeff) if j)
        return tuple(values)


class StableBackground(ce.Background):
    def __init__(self,p,sp,method='DOP853'):
        self.p=p;self.sp=sp;self.Ni=np.log(p.ai)
        self.sol=solve_ivp(self.rhs,(self.Ni,0),[p.theta_initial,0.],method=method,
                           rtol=1e-10,atol=p.theta_initial*1e-8,dense_output=True,max_step=.02)
        assert self.sol.success
        self.H0=self.quantities(0.)['H']


def run():
    *_, zs, observed, cov = ce.load_data()
    chol=np.linalg.cholesky(cov)
    base=ce.Background(ce.Parameters(.35,0.))
    def prediction(bg): return np.array([bg.distances(z)['F_AP'] for z in zs])
    residual=np.linalg.solve(chol,prediction(base)-observed)
    chi0=float(residual@residual)
    sigma=np.hypot(145,620)*1e-12;gap=385e-12
    muon=exact_em(1000.,.35*(.014414e-9)**2/1e6,0.)
    baseline=float(np.sqrt((chi0+(gap/sigma)**2)/7))
    matched=float(np.sqrt((chi0+((gap-muon)/sigma)**2)/7))
    rows=[]
    for r,mass in [(.15,.027615),(.35,.014414)]:
        physical_s_ratio=(mass*1e-9/2.435e18)**2
        p=ce.Parameters(r,0.,f=1/30,s_over_Mp2=physical_s_ratio)
        sp=StableLightSpectrum(p)
        _,_,U2,_,_,m2,Z,_=sp.evaluate(0.)
        def linear(N,y):
            a=np.exp(N);H2=(p.Rm/a**3+1)/3;hn=-p.Rm/a**3/(2*H2)
            return [y[1],-(3+hn)*y[1]-(U2+p.fc*p.Rm/a**3*m2)/(Z*H2)*y[0]]
        lin=solve_ivp(linear,(np.log(p.ai),0),[1.,0.],method='DOP853',rtol=1e-11,atol=1e-13)
        assert lin.success
        cases=[]
        for seed in [1e-12,1e-10,1e-8]:
            pp=ce.Parameters(r,seed,f=1/30,s_over_Mp2=physical_s_ratio)
            bg=StableBackground(pp,sp)
            pred=prediction(bg);rr=np.linalg.solve(chol,pred-observed)
            score=float(np.sqrt((rr@rr+((gap-muon)/sigma)**2)/7))
            end=bg.quantities(0.)
            cases.append(dict(seed=seed,final_theta=end['theta'],final_w_phi=end['w_phi'],
                              final_omega_phi=end['omega_phi'], AP_rmse=float(np.sqrt(rr@rr/6)),
                              partial_rmse=score,delta_from_SM_baseline=score-baseline,
                              delta_from_same_charged_baseline=score-matched,
                              linear_prediction=float(seed*lin.y[0,-1]),prediction=pred.tolist()))
        # Check both tiny-seed resolution and the largest nonlinear change.
        solver_checks=[]
        for index in [0,2]:
            seed=cases[index]['seed']
            pp=ce.Parameters(r,seed,f=1/30,s_over_Mp2=physical_s_ratio)
            bg=StableBackground(pp,sp,method='Radau')
            pred=prediction(bg)
            solver_diff=float(np.max(abs(pred-cases[index]['prediction'])))
            rr=np.linalg.solve(chol,pred-observed)
            check_score=float(np.sqrt((rr@rr+((gap-muon)/sigma)**2)/7))
            continuity=bg.source_checks()['relative_total_continuity']
            assert solver_diff<1e-8
            assert abs(check_score-cases[index]['partial_rmse'])<1e-8
            assert continuity<1e-11
            solver_checks.append(dict(seed=seed, AP_max_difference=solver_diff,
                                      partial_rmse_difference=check_score-cases[index]['partial_rmse'],
                                      relative_continuity=continuity))
        rows.append(dict(r=r, f_over_Mp=1/30, linear_transfer=float(lin.y[0,-1]),
                         cases=cases,independent_solver_checks=solver_checks))
    return dict(baseline_partial_rmse=baseline,same_charged_baseline=matched,rows=rows,
                fitted_parameters=[],seed_distribution_predicted=False,
                full_joint_rmse=None,scope='Conditional late cold sector; no initial creation, full muon, CMB or four-force likelihood.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
